"""Lightweight causal temporal residual policy.

The model is trained offline from ordered replay decisions.  At runtime it
keeps no hidden information: features use the current public state plus the
ordered semantic decisions already stored by :class:`CausalHistory`.

The linear utility model is a small recurrent state-space approximation:
short/medium/long exponentially decayed action traces are interacted with each
currently legal action.  It only reranks one-choice MAIN prompts for the
matchup encoded in ``temporal_policy.json`` and leaves all safety guards in
``main.py`` authoritative.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter


# CABT numeric API constants, duplicated to keep this module dependency-free.
_MAIN_CONTEXT = 0
_MAIN_SELECT = 0
_OPT_PLAY = 7
_OPT_ATTACH = 8
_OPT_EVOLVE = 9
_OPT_ABILITY = 10
_OPT_RETREAT = 12
_OPT_ATTACK = 13
_OPT_END = 14

_AREA_DECK = 1
_AREA_HAND = 2
_AREA_DISCARD = 3
_AREA_ACTIVE = 4
_AREA_BENCH = 5
_AREA_PRIZE = 6
_AREA_STADIUM = 7
_AREA_LOOKING = 12

_DRAG_LINE = {119, 120, 121}
_GRASS_SIGNATURE = {89, 90, 92, 93, 96, 150, 709, 710, 917, 918, 920}


def _int(value, default=0):
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _cards_from_area(observation, area, index, player_index=None):
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    me = _int(current.get("yourIndex"), 0)
    if player_index is None:
        player_index = me
    if area == _AREA_DECK:
        cards = select.get("deck") or []
    elif area == _AREA_STADIUM:
        cards = current.get("stadium") or []
    elif area == _AREA_LOOKING:
        cards = current.get("looking") or []
    else:
        players = current.get("players") or []
        if not (0 <= player_index < len(players)):
            return None
        key = {
            _AREA_HAND: "hand",
            _AREA_DISCARD: "discard",
            _AREA_ACTIVE: "active",
            _AREA_BENCH: "bench",
            _AREA_PRIZE: "prize",
        }.get(area)
        cards = players[player_index].get(key, []) if key else []
        cards = cards or []
    if 0 <= _int(index, -1) < len(cards):
        card = cards[_int(index, -1)]
        return card if isinstance(card, dict) else None
    return None


def _source_card(observation, option):
    typ = _int(option.get("type"), -1)
    current = observation.get("current") or {}
    me = _int(current.get("yourIndex"), 0)
    if typ == _OPT_PLAY:
        return _cards_from_area(observation, _AREA_HAND, option.get("index"), me)
    if typ in (_OPT_ATTACH, _OPT_EVOLVE, _OPT_ABILITY):
        return _cards_from_area(
            observation,
            _int(option.get("area"), -1),
            option.get("index"),
            _int(option.get("playerIndex"), me),
        )
    if typ in (_OPT_RETREAT, _OPT_ATTACK):
        return _cards_from_area(observation, _AREA_ACTIVE, 0, me)
    return None


def _target_card(observation, option):
    if option.get("inPlayArea") is None:
        return None
    current = observation.get("current") or {}
    return _cards_from_area(
        observation,
        _int(option.get("inPlayArea"), -1),
        option.get("inPlayIndex"),
        _int(current.get("yourIndex"), 0),
    )


def semantic_option(observation, index):
    """Return a CausalHistory-compatible semantic record for one raw option."""
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        return None
    option = options[index]
    source = _source_card(observation, option)
    target = _target_card(observation, option)
    effect = select.get("effect") or {}
    current = observation.get("current") or {}
    return {
        "turn": _int(current.get("turn"), 0),
        "turn_action_count": _int(current.get("turnActionCount"), 0),
        "context": _int(select.get("context"), -1),
        "effect_card_id": _int(effect.get("id"), 0),
        "option_index": index,
        "option_type": _int(option.get("type"), -1),
        "card_id": _int((source or {}).get("id", option.get("cardId")), 0),
        "serial": _int((source or {}).get("serial", option.get("serial")), -1),
        "target_card_id": _int((target or {}).get("id"), 0),
        "target_serial": _int((target or {}).get("serial"), -1),
        "attack_id": _int(option.get("attackId"), 0),
        "number": option.get("number"),
    }


def action_token(semantic):
    if not semantic:
        return "invalid"
    typ = _int(semantic.get("option_type"), -1)
    cid = _int(semantic.get("card_id"), 0)
    target = _int(semantic.get("target_card_id"), 0)
    if typ == _OPT_PLAY:
        return f"p{cid}"
    if typ == _OPT_ATTACH:
        return f"h{cid}>{target}"
    if typ == _OPT_EVOLVE:
        return f"e{cid}>{target}"
    if typ == _OPT_ABILITY:
        return f"b{cid}"
    if typ == _OPT_RETREAT:
        return f"r{cid}"
    if typ == _OPT_ATTACK:
        return f"x{_int(semantic.get('attack_id'), 0)}"
    if typ == _OPT_END:
        return "z"
    return f"t{typ}:c{cid}"


def any_action_token(semantic):
    """Tokenize every own selection, including effect-resolution prompts."""
    if not semantic:
        return "invalid"
    if _int(semantic.get("context"), -1) == _MAIN_CONTEXT:
        return action_token(semantic)
    return "q{}:e{}:t{}:c{}".format(
        _int(semantic.get("context"), -1),
        _int(semantic.get("effect_card_id"), 0),
        _int(semantic.get("option_type"), -1),
        _int(semantic.get("card_id"), 0),
    )


def _energy_ids(pokemon):
    if not isinstance(pokemon, dict):
        return set()
    return {_int(e.get("id"), 0) for e in (pokemon.get("energyCards") or []) if isinstance(e, dict)}


def _bucket(value, cuts):
    value = _int(value, 0)
    for i, cut in enumerate(cuts):
        if value <= cut:
            return str(i)
    return str(len(cuts))


def _phase(turn):
    turn = _int(turn, 0)
    if turn <= 4:
        return "open"
    if turn <= 8:
        return "build"
    if turn <= 12:
        return "pivot"
    return "late"


def _candidate_static(observation, semantic, history_decisions):
    current = observation.get("current") or {}
    players = current.get("players") or []
    me = _int(current.get("yourIndex"), 0)
    mine = players[me] if 0 <= me < len(players) else {}
    typ = _int(semantic.get("option_type"), -1)
    target_serial = _int(semantic.get("target_serial"), -1)
    target = None
    for p in (mine.get("active") or []) + (mine.get("bench") or []):
        if isinstance(p, dict) and _int(p.get("serial"), -2) == target_serial:
            target = p
            break
    out = []
    if typ == _OPT_ATTACH and target is not None:
        energies = _energy_ids(target)
        eid = _int(semantic.get("card_id"), 0)
        out.append("attach_new=" + str(int(eid not in energies)))
        out.append("attach_target=" + str(_int(target.get("id"), 0)))
        if _int(target.get("id"), 0) in _DRAG_LINE and eid in (2, 5):
            other = 5 if eid == 2 else 2
            out.append("attach_completes_fp=" + str(int(other in energies and eid not in energies)))
        if _int(target.get("id"), 0) == 112 and eid == 7:
            out.append("attach_arms_munk=" + str(int(7 not in energies)))
    if typ == _OPT_EVOLVE and target is not None:
        used = any(
            _int(d.get("turn"), -1) == _int(current.get("turn"), -2)
            and _int(d.get("option_type"), -1) == _OPT_ABILITY
            and _int(d.get("serial"), -3) == target_serial
            for d in history_decisions
        )
        out.append("evolve_target_ability_used=" + str(int(used)))
    return out


def candidate_features(observation, semantic, history_decisions):
    """Sparse feature dictionary for candidate utility scoring."""
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    me = _int(current.get("yourIndex"), 0)
    mine = players[me] if 0 <= me < len(players) else {}
    opp = players[1 - me] if len(players) >= 2 else {}
    board = [p for p in (mine.get("active") or []) + (mine.get("bench") or []) if isinstance(p, dict)]
    opp_board = [p for p in (opp.get("active") or []) + (opp.get("bench") or []) if isinstance(p, dict)]
    active = next((p for p in (mine.get("active") or []) if isinstance(p, dict)), None)
    opp_active = next((p for p in (opp.get("active") or []) if isinstance(p, dict)), None)
    counts = Counter(_int(p.get("id"), 0) for p in board)
    ready_drag = sum(1 for p in board if _int(p.get("id"), 0) == 121 and {2, 5}.issubset(_energy_ids(p)))
    options = select.get("option") or []
    option_semantics = [semantic_option(observation, i) for i in range(len(options))]
    available_tokens = {action_token(s) for s in option_semantics if s}
    token = action_token(semantic)
    kind = token[:1]
    turn = _int(current.get("turn"), 0)

    main_history = [d for d in history_decisions if _int(d.get("context"), -1) == _MAIN_CONTEXT]
    previous_main = [action_token(d) for d in main_history]
    previous_any = [any_action_token(d) for d in history_decisions]
    this_turn = [d for d in main_history if _int(d.get("turn"), -1) == turn]
    this_turn_tokens = [action_token(d) for d in this_turn]

    base_context = [
        "phase=" + _phase(turn),
        "parity=" + str(turn & 1),
        "turnpos=" + _bucket((turn + 1) // 2, (1, 2, 3, 5, 7)),
        "ta=" + _bucket(current.get("turnActionCount"), (2, 5, 9, 14, 20)),
        "active=" + str(_int((active or {}).get("id"), 0)),
        "oppactive=" + str(_int((opp_active or {}).get("id"), 0)),
        "line={}:{}:{}".format(counts[119], counts[120], counts[121]),
        "ready=" + str(min(2, ready_drag)),
        "budew=" + str(min(2, counts[235])),
        "munk=" + str(min(2, counts[112])),
        "fez=" + str(min(1, counts[140])),
        "bench=" + _bucket(len(mine.get("bench") or []), (1, 3, 4)),
        "hand=" + _bucket(mine.get("handCount", len(mine.get("hand") or [])), (2, 4, 6, 9)),
        "myprize=" + _bucket(len(mine.get("prize") or []), (1, 2, 3, 4, 5)),
        "opprize=" + _bucket(len(opp.get("prize") or []), (1, 2, 3, 4, 5)),
        "supporter=" + str(int(bool(current.get("supporterPlayed")))),
        "energy=" + str(int(bool(current.get("energyAttached")))),
        "retreated=" + str(int(bool(current.get("retreated")))),
        "has_phantom=" + str(int("x154" in available_tokens)),
        "has_budew=" + str(int("x323" in available_tokens)),
        "has_drak_ability=" + str(int("b120" in available_tokens)),
        "has_drag_evolve=" + str(int(any(x.startswith("e121>") for x in available_tokens))),
        "has_retreat=" + str(int(any(x.startswith("r") for x in available_tokens))),
    ]

    features = {"A=" + token: 1.0, "K=" + kind: 0.35}
    # Action/state interactions are the core utility features.
    for context in base_context:
        features[context + "|A=" + token] = 1.0
        features[context + "|K=" + kind] = 0.25
    for extra in _candidate_static(observation, semantic, history_decisions):
        features[extra + "|A=" + token] = 1.0

    # Exact ordered lags.
    for lag in range(1, 5):
        prev = previous_main[-lag] if len(previous_main) >= lag else "START"
        features[f"lag{lag}={prev}|A={token}"] = 1.0
    for lag in range(1, 3):
        prev = previous_any[-lag] if len(previous_any) >= lag else "START"
        features[f"anylag{lag}={prev}|A={token}"] = 0.55

    # Exponentially decayed recurrent traces at three horizons.
    for label, decay, limit in (("s", 0.42, 6), ("m", 0.72, 12), ("l", 0.90, 24)):
        trace = Counter()
        value = 1.0
        for prev in reversed(previous_main[-limit:]):
            trace[prev] += value
            value *= decay
        for prev, amount in trace.items():
            if amount >= 0.08:
                features[f"trace{label}={prev}|A={token}"] = round(amount, 6)

    # Turn-local plan completion: preserve the set and order of actions already
    # committed this turn without encoding card serials.
    for prev in sorted(set(this_turn_tokens)):
        features[f"seen={prev}|A={token}"] = 0.7
    if this_turn_tokens:
        features[f"turnlast={this_turn_tokens[-1]}|A={token}"] = 0.9
        features[f"turnlen={_bucket(len(this_turn_tokens),(1,3,6,10,15))}|A={token}"] = 0.45
    else:
        features[f"turnlast=START|A={token}"] = 0.9

    return features


def _dot(weights, features):
    return sum(float(weights.get(key, 0.0)) * float(value) for key, value in features.items())


class TemporalResidualPolicy:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = {}
        self.weights = {}
        self.stats = Counter()
        try:
            with open(model_path, encoding="utf-8") as handle:
                self.model = json.load(handle)
            self.weights = {str(k): float(v) for k, v in (self.model.get("weights") or {}).items()}
        except Exception:
            self.model = {}
            self.weights = {}

    @property
    def enabled(self):
        return bool(self.model.get("enabled", False) and self.weights)

    def _visible_opponent_ids(self, observation):
        current = observation.get("current") or {}
        players = current.get("players") or []
        me = _int(current.get("yourIndex"), 0)
        if len(players) < 2:
            return set()
        op = players[1 - me]
        ids = set()
        for card in (op.get("active") or []) + (op.get("bench") or []) + (op.get("discard") or []):
            if isinstance(card, dict):
                ids.add(_int(card.get("id"), 0))
        return ids

    def rerank(self, observation, chosen, history, matchup=None, confidence=0.0):
        """Return a conservative temporal override or the original action."""
        if not self.enabled:
            return chosen
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        if not isinstance(chosen, list) or len(chosen) != 1:
            return chosen
        if _int(select.get("type"), -1) != _MAIN_SELECT or _int(select.get("context"), -1) != _MAIN_CONTEXT:
            return chosen
        if _int(select.get("minCount"), 0) != 1 or _int(select.get("maxCount"), 0) != 1:
            return chosen
        target_matchup = str(self.model.get("matchup", "grass_ogerpon"))
        visible = self._visible_opponent_ids(observation)
        signature = set(self.model.get("opponent_signature") or _GRASS_SIGNATURE)
        if matchup != target_matchup:
            return chosen
        if float(confidence) < float(self.model.get("min_recognition_confidence", 0.12)) and not (visible & signature):
            return chosen
        turn = _int(current.get("turn"), 0)
        if turn > _int(self.model.get("max_turn", 18), 18):
            return chosen

        options = select.get("option") or []
        candidates = []
        decisions = list(getattr(history, "decisions", []) or [])
        for index in range(len(options)):
            semantic = semantic_option(observation, index)
            if not semantic:
                continue
            typ = _int(semantic.get("option_type"), -1)
            if typ not in (_OPT_PLAY, _OPT_ATTACH, _OPT_EVOLVE, _OPT_ABILITY, _OPT_RETREAT, _OPT_ATTACK, _OPT_END):
                continue
            token = action_token(semantic)
            # Duplicate copies of the same semantic action are equivalent; use
            # the lowest legal index so the override is deterministic.
            features = candidate_features(observation, semantic, decisions)
            score = _dot(self.weights, features)
            candidates.append((score, index, token, semantic))
        if len(candidates) < 2:
            return chosen

        base_index = chosen[0]
        base = next((row for row in candidates if row[1] == base_index), None)
        if base is None:
            return chosen
        candidates.sort(key=lambda row: (-row[0], row[1]))
        best = candidates[0]
        # Never learn a voluntary pass as an override.  END remains available
        # to the audited base policy and hard safety guards.
        if best[3].get("option_type") == _OPT_END:
            return chosen
        if best[1] == base_index or best[2] == base[2]:
            # Multiple legal options can encode the same semantic action with
            # different engine-local indices.  Preserve the audited base index
            # rather than manufacturing a meaningless override.
            self.stats["kept_top"] += 1
            return chosen

        token_support = self.model.get("token_support") or {}
        support = token_support.get(best[2], {})
        winner_games = _int(support.get("winner_games"), 0)
        winner_choices = _int(support.get("winner_choices"), 0)
        margin = float(best[0] - base[0])
        required = float(self.model.get("override_margin", 0.70))
        if winner_games < _int(self.model.get("min_winner_games", 1), 1):
            required += float(self.model.get("low_support_margin_add", 0.55))
        if winner_choices < _int(self.model.get("min_winner_choices", 1), 1):
            return chosen

        # Preserve universally dominant tactical actions.  The temporal layer
        # may insert a Drakloak draw before Phantom Dive, but may not replace a
        # legal Phantom Dive with arbitrary setup.
        base_token = base[2]
        if base_token == "x154" and best[2] != "b120":
            return chosen
        # Likewise do not evolve a Drakloak before using its currently legal
        # draw ability; the existing ordering guard remains authoritative.
        if best[2].startswith("e121>") and any(row[2] == "b120" for row in candidates):
            return chosen

        if margin < required:
            self.stats["below_margin"] += 1
            return chosen
        self.stats["overrides"] += 1
        self.stats["override_" + best[2]] += 1
        return [best[1]]

    def get_stats(self):
        return {"temporal_" + str(k): int(v) for k, v in self.stats.items()}


def default_model_path(root):
    return os.path.join(root, "temporal_policy.json")
