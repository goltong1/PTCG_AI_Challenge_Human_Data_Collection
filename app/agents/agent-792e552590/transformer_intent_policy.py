"""Offline-trained transformer intent advisor for the Dragapult policy.

The competition runtime has no network dependency.  A four-layer, 128-dimensional Transformer is trained before packaging
from public replay prefixes.  At
runtime it reads only the current legal observation plus the agent's causal
public history, predicts a high-level action intent, and may replace the
baseline action only when a matchup-specific confidence gate is satisfied.
All legality and deterministic safety guards remain downstream.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

# Keep tiny single-observation matrix products single-threaded.  Multi-agent
# league evaluation otherwise oversubscribes BLAS threads heavily.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import numpy as np
except Exception:  # Kaggle-safe fail-closed fallback
    np = None

# CABT numeric API values.
MAIN_CONTEXT = 0
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

DREEPY = 119
DRAKLOAK = 120
DRAGAPULT = 121
MUNKIDORI = 112
BUDEW = 235
FEZANDIPITI = 140
MEOWTH = 1071
FIRE = 2
PSYCHIC = 5
DARK = 7
POFFIN = 1086
ULTRA_BALL = 1121
POKE_PAD = 1152
STRETCHER = 1097
HAMMER = 1120
JAMMING = 1246
STAMP = 1080
JUDGE = 1213
LILLIE = 1227
BOSS = 1182
CRISPIN = 1198
DAWN = 1231
DRAG_LINE = {DREEPY, DRAKLOAK, DRAGAPULT}
CRUSTLE_IDS = {344, 345}


def _int(value, default=0):
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _bucket(value, cuts):
    value = _int(value, 0)
    for i, cut in enumerate(cuts):
        if value <= cut:
            return str(i)
    return str(len(cuts))


def _cards_from_area(observation, area, index, player_index=None):
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    me = _int(current.get("yourIndex"), 0)
    if player_index is None:
        player_index = me
    if area == AREA_DECK:
        cards = select.get("deck") or []
    elif area == AREA_STADIUM:
        cards = current.get("stadium") or []
    elif area == AREA_LOOKING:
        cards = current.get("looking") or []
    else:
        players = current.get("players") or []
        if not (0 <= player_index < len(players)):
            return None
        key = {
            AREA_HAND: "hand", AREA_DISCARD: "discard", AREA_ACTIVE: "active",
            AREA_BENCH: "bench", AREA_PRIZE: "prize",
        }.get(area)
        cards = players[player_index].get(key, []) if key else []
        cards = cards or []
    index = _int(index, -1)
    if 0 <= index < len(cards):
        card = cards[index]
        return card if isinstance(card, dict) else None
    return None


def _source_card(observation, option):
    typ = _int(option.get("type"), -1)
    current = observation.get("current") or {}
    me = _int(current.get("yourIndex"), 0)
    if typ == OPT_PLAY:
        return _cards_from_area(observation, AREA_HAND, option.get("index"), me)
    if typ in (1, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY):
        return _cards_from_area(
            observation, _int(option.get("area"), -1), option.get("index"),
            _int(option.get("playerIndex"), me),
        )
    if typ in (OPT_RETREAT, OPT_ATTACK):
        return _cards_from_area(observation, AREA_ACTIVE, 0, me)
    return None


def _target_card(observation, option):
    if option.get("inPlayArea") is None:
        return None
    current = observation.get("current") or {}
    return _cards_from_area(
        observation, _int(option.get("inPlayArea"), -1),
        option.get("inPlayIndex"), _int(current.get("yourIndex"), 0),
    )


def semantic_option(observation, index):
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        return None
    option = options[index]
    source = _source_card(observation, option)
    target = _target_card(observation, option)
    current = observation.get("current") or {}
    effect = select.get("effect") or {}
    return {
        "index": index,
        "turn": _int(current.get("turn"), 0),
        "context": _int(select.get("context"), -1),
        "option_type": _int(option.get("type"), -1),
        "card_id": _int((source or {}).get("id", option.get("cardId")), 0),
        "serial": _int((source or {}).get("serial", option.get("serial")), -1),
        "target_card_id": _int((target or {}).get("id"), 0),
        "target_serial": _int((target or {}).get("serial"), -1),
        "target_area": _int(option.get("inPlayArea"), -1),
        "attack_id": _int(option.get("attackId"), 0),
        "effect_card_id": _int(effect.get("id"), 0),
        "number": option.get("number"),
    }


def semantic_key(semantic):
    if not semantic:
        return ("invalid",)
    return (
        _int(semantic.get("option_type"), -1), _int(semantic.get("card_id"), 0),
        _int(semantic.get("target_serial"), -1),
        _int(semantic.get("attack_id"), 0), semantic.get("number"),
    )


def _role(card_id):
    return {
        DREEPY: "dreepy", DRAKLOAK: "drakloak", DRAGAPULT: "dragapult",
        MUNKIDORI: "munkidori", BUDEW: "budew", FEZANDIPITI: "fezandipiti",
        MEOWTH: "meowth",
    }.get(_int(card_id), "other")


def _energy_ids(card):
    if not isinstance(card, dict):
        return set()
    return {_int(x.get("id"), 0) for x in (card.get("energyCards") or []) if isinstance(x, dict)}


def _board_card_by_serial(observation, serial):
    current = observation.get("current") or {}
    players = current.get("players") or []
    me = _int(current.get("yourIndex"), 0)
    if not (0 <= me < len(players)):
        return None, "none"
    mine = players[me]
    for zone, cards in (("active", mine.get("active") or []), ("bench", mine.get("bench") or [])):
        for card in cards:
            if isinstance(card, dict) and _int(card.get("serial"), -2) == _int(serial, -1):
                return card, zone
    return None, "none"


def base_intent(semantic):
    if not semantic:
        return "invalid"
    typ = _int(semantic.get("option_type"), -1)
    card_id = _int(semantic.get("card_id"), 0)
    if typ == OPT_PLAY:
        return {
            DREEPY: "develop_dreepy", MUNKIDORI: "develop_munkidori",
            BUDEW: "develop_budew", FEZANDIPITI: "develop_fezandipiti",
            MEOWTH: "develop_meowth", POFFIN: "search_basic",
            ULTRA_BALL: "search_evolution", POKE_PAD: "search_trainer",
            STRETCHER: "recover", HAMMER: "disrupt_energy",
            JAMMING: "stadium_lock", STAMP: "hand_reset", JUDGE: "hand_reset",
            LILLIE: "draw_reset", BOSS: "gust", CRISPIN: "energy_acceleration",
            DAWN: "search_supporter",
        }.get(card_id, "play_other")
    if typ == OPT_ATTACH:
        energy = {FIRE: "fire", PSYCHIC: "psychic", DARK: "dark"}.get(card_id, "other")
        return "attach_{}_to_{}".format(energy, _role(semantic.get("target_card_id")))
    if typ == OPT_EVOLVE:
        return {DRAKLOAK: "evolve_drakloak", DRAGAPULT: "evolve_dragapult"}.get(card_id, "evolve_other")
    if typ == OPT_ABILITY:
        return {DRAKLOAK: "draw_drakloak", FEZANDIPITI: "draw_fezandipiti", MUNKIDORI: "move_damage"}.get(card_id, "ability_other")
    if typ == OPT_RETREAT:
        return "retreat"
    if typ == OPT_ATTACK:
        return {154: "phantom_dive", 153: "jet_headbutt", 323: "budew_lock", 150: "dreepy_attack", 141: "mind_bend"}.get(_int(semantic.get("attack_id"), 0), "attack_other")
    if typ == OPT_END:
        return "pass"
    return "other"


def intent_key(observation, semantic, prior_decisions=()):
    """Action class detailed enough to retain plan-completion information."""
    root = base_intent(semantic)
    typ = _int((semantic or {}).get("option_type"), -1)
    if typ == OPT_ATTACH:
        target, zone = _board_card_by_serial(observation, semantic.get("target_serial"))
        energies = _energy_ids(target)
        eid = _int(semantic.get("card_id"), 0)
        target_id = _int((target or {}).get("id", semantic.get("target_card_id")), 0)
        detail = "other"
        if eid in energies:
            detail = "duplicate"
        elif target_id in DRAG_LINE and eid in (FIRE, PSYCHIC):
            other = PSYCHIC if eid == FIRE else FIRE
            detail = "complete" if other in energies else ("seed" if not energies else "add")
        elif target_id == MUNKIDORI and eid == DARK:
            detail = "arm" if DARK not in energies else "duplicate"
        return "{}__{}__{}".format(root, zone, detail)
    if typ == OPT_EVOLVE:
        target, _ = _board_card_by_serial(observation, semantic.get("target_serial"))
        serial = _int((target or {}).get("serial"), semantic.get("target_serial"))
        turn = _int((observation.get("current") or {}).get("turn"), 0)
        used = any(
            _int(d.get("turn"), -1) == turn and _int(d.get("option_type"), -1) == OPT_ABILITY
            and _int(d.get("serial"), -3) == serial for d in (prior_decisions or [])
        )
        return root + ("__after_draw" if used else "__before_draw")
    return root


def _phase(turn):
    turn = _int(turn, 0)
    if turn <= 4:
        return "open"
    if turn <= 8:
        return "build"
    if turn <= 12:
        return "pressure"
    return "late"


def _card_tokens(prefix, cards, max_cards=8):
    out = []
    for i, card in enumerate((cards or [])[:max_cards]):
        if not isinstance(card, dict):
            continue
        cid = _int(card.get("id"), 0)
        hp = _int(card.get("hp"), 0)
        mhp = max(1, _int(card.get("maxHp"), hp or 1))
        ratio = int(max(0, min(4, math.floor(5.0 * hp / mhp))))
        out.extend([
            f"{prefix}{i}:id={cid}", f"{prefix}{i}:hp={ratio}",
            f"{prefix}{i}:e={len(card.get('energyCards') or [])}",
        ])
        for eid in sorted(_energy_ids(card)):
            out.append(f"{prefix}{i}:eid={eid}")
        tool = card.get("tool")
        if isinstance(tool, dict):
            out.append(f"{prefix}{i}:tool={_int(tool.get('id'),0)}")
    return out


def build_tokens(observation, history_decisions=(), public_events=(), matchup="unknown", confidence=0.0, legal_intents=(), max_tokens=96):
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    me = _int(current.get("yourIndex"), 0)
    mine = players[me] if 0 <= me < len(players) else {}
    opp = players[1 - me] if len(players) >= 2 else {}
    turn = _int(current.get("turn"), 0)
    tokens = [
        "[CLS]", f"matchup={matchup or 'unknown'}",
        "recognition=" + _bucket(round(float(confidence) * 100), (10, 25, 50, 75, 90)),
        f"phase={_phase(turn)}", f"turn_parity={turn & 1}",
        "turn=" + _bucket(turn, (2, 4, 6, 8, 12, 16, 22)),
        "action_count=" + _bucket(current.get("turnActionCount"), (2, 5, 9, 14, 20)),
        "context=" + str(_int(select.get("context"), -1)),
        "my_prize=" + _bucket(len(mine.get("prize") or []), (1, 2, 3, 4, 5)),
        "opp_prize=" + _bucket(len(opp.get("prize") or []), (1, 2, 3, 4, 5)),
        "my_hand=" + _bucket(mine.get("handCount", len(mine.get("hand") or [])), (2, 4, 6, 9)),
        "opp_hand=" + _bucket(opp.get("handCount", len(opp.get("hand") or [])), (2, 4, 6, 9)),
        "my_deck=" + _bucket(mine.get("deckCount", len(mine.get("deck") or [])), (5, 10, 20, 30, 40)),
        "opp_deck=" + _bucket(opp.get("deckCount", len(opp.get("deck") or [])), (5, 10, 20, 30, 40)),
        "supporter=" + str(int(bool(current.get("supporterPlayed")))),
        "energy_attached=" + str(int(bool(current.get("energyAttached")))),
        "retreated=" + str(int(bool(current.get("retreated")))),
        "first_player=" + str(_int(current.get("firstPlayer"), -1) == me),
    ]
    tokens += _card_tokens("my_active", mine.get("active") or [], 1)
    tokens += _card_tokens("my_bench", mine.get("bench") or [], 8)
    tokens += _card_tokens("opp_active", opp.get("active") or [], 1)
    tokens += _card_tokens("opp_bench", opp.get("bench") or [], 8)

    hand_counts = Counter(_int(c.get("id"), 0) for c in (mine.get("hand") or []) if isinstance(c, dict))
    for cid, count in sorted(hand_counts.items()):
        tokens.append(f"hand_id={cid}:n={min(count,4)}")
    discard_counts = Counter(_int(c.get("id"), 0) for c in (opp.get("discard") or []) if isinstance(c, dict))
    for cid, count in sorted(discard_counts.items())[-12:]:
        tokens.append(f"opp_discard={cid}:n={min(count,4)}")
    stadium = current.get("stadium") or []
    for c in stadium[:1]:
        if isinstance(c, dict):
            tokens.append(f"stadium={_int(c.get('id'),0)}")

    # The most recent causal decisions carry ordered plan state.
    recent_decisions = list(history_decisions or [])[-16:]
    for lag, d in enumerate(reversed(recent_decisions), 1):
        sem = dict(d)
        ik = base_intent(sem)
        tokens.extend([
            f"d{lag}:intent={ik}", f"d{lag}:type={_int(d.get('option_type'),-1)}",
            f"d{lag}:card={_int(d.get('card_id'),0)}",
            f"d{lag}:target={_int(d.get('target_card_id'),0)}",
            f"d{lag}:attack={_int(d.get('attack_id'),0)}",
            f"d{lag}:turn_delta={min(6,max(0,turn-_int(d.get('turn'),turn)))}",
        ])

    # Received public events represent opponent sequencing and revealed cards.
    recent_events = list(public_events or [])[-16:]
    for lag, e in enumerate(reversed(recent_events), 1):
        pi = _int(e.get("player_index", e.get("playerIndex")), -1)
        rel = "me" if pi == me else ("opp" if pi == 1 - me else "other")
        tokens.extend([
            f"e{lag}:who={rel}",
            f"e{lag}:type={_int(e.get('log_type',e.get('type')),-1)}",
            f"e{lag}:card={_int(e.get('card_id',e.get('cardId')),0)}",
            f"e{lag}:before={_int(e.get('card_id_before',e.get('cardIdBefore')),0)}",
            f"e{lag}:after={_int(e.get('card_id_after',e.get('cardIdAfter')),0)}",
            f"e{lag}:attack={_int(e.get('attack_id',e.get('attackId')),0)}",
            f"e{lag}:move={_int(e.get('from_area',e.get('fromArea')),-1)}>{_int(e.get('to_area',e.get('toArea')),-1)}",
        ])

    for intent in sorted(set(legal_intents or [])):
        tokens.append("legal=" + str(intent))
    # Preserve [CLS] and the most recent information if the sequence overflows.
    if len(tokens) > max_tokens:
        tokens = [tokens[0]] + tokens[-(max_tokens - 1):]
    return tokens


def stable_hash(token, vocab_size):
    h = 2166136261
    for b in str(token).encode("utf-8", errors="replace"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return 2 + (h % max(1, int(vocab_size) - 2))


def tokens_to_ids(tokens, vocab_size, max_tokens):
    ids = [1 if t == "[CLS]" else stable_hash(t, vocab_size) for t in tokens[:max_tokens]]
    if not ids:
        ids = [1]
    return ids


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-12)


class TransformerIntentPolicy:
    def __init__(self, model_path):
        self.model_path = model_path
        self.meta = {}
        self.weights = {}
        self.stats = Counter()
        if np is None:
            self.stats["numpy_unavailable"] += 1
            return
        try:
            with np.load(model_path, allow_pickle=False) as z:
                raw = bytes(z["meta_json"].astype(np.uint8).tolist()).decode("utf-8")
                self.meta = json.loads(raw)
                for key in z.files:
                    if key != "meta_json":
                        self.weights[key] = np.asarray(z[key], dtype=np.float32)
            self.stats["model_loaded"] += 1
        except Exception:
            self.meta = {}
            self.weights = {}
            self.stats["load_error"] += 1

    @property
    def enabled(self):
        return bool(self.meta.get("enabled", False) and self.weights and np is not None)

    def reset(self):
        # Keep lifetime counters for league auditing while marking boundaries.
        self.stats["games"] += 1

    def get_stats(self):
        return {"transformer_" + str(k): int(v) for k, v in self.stats.items()}

    def _encode(self, ids):
        W = self.weights
        max_tokens = _int(self.meta.get("max_tokens"), 96)
        ids = ids[:max_tokens]
        x = W["token_embedding"][np.asarray(ids, dtype=np.int64)] + W["position_embedding"][:len(ids)]
        d_model = x.shape[-1]
        heads = _int(self.meta.get("heads"), 4)
        head_dim = d_model // heads
        scale = 1.0 / math.sqrt(max(1, head_dim))
        for layer in range(_int(self.meta.get("layers"), 2)):
            p = "l%d_" % layer
            y = _layer_norm(x, W[p + "ln1_g"], W[p + "ln1_b"])
            qkv = y @ W[p + "qkv_w"] + W[p + "qkv_b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(len(ids), heads, head_dim).transpose(1, 0, 2)
            k = k.reshape(len(ids), heads, head_dim).transpose(1, 0, 2)
            v = v.reshape(len(ids), heads, head_dim).transpose(1, 0, 2)
            att = np.matmul(q, k.transpose(0, 2, 1)) * scale
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att /= np.maximum(att.sum(axis=-1, keepdims=True), 1e-12)
            z = np.matmul(att, v).transpose(1, 0, 2).reshape(len(ids), d_model)
            x = x + z @ W[p + "out_w"] + W[p + "out_b"]
            y = _layer_norm(x, W[p + "ln2_g"], W[p + "ln2_b"])
            y = _gelu(y @ W[p + "ff1_w"] + W[p + "ff1_b"])
            x = x + y @ W[p + "ff2_w"] + W[p + "ff2_b"]
        pooled = _layer_norm(x[:1], W["final_ln_g"], W["final_ln_b"])[0]
        return pooled @ W["head_w"] + W["head_b"]

    def _visible_opponent_ids(self, observation):
        current = observation.get("current") or {}
        players = current.get("players") or []
        me = _int(current.get("yourIndex"), 0)
        if len(players) < 2:
            return set()
        opp = players[1 - me]
        return {_int(c.get("id"), 0) for c in (opp.get("active") or []) + (opp.get("bench") or []) + (opp.get("discard") or []) if isinstance(c, dict)}

    def _candidate_rank(self, observation, semantic, prior_decisions):
        typ = _int(semantic.get("option_type"), -1)
        # Prefer completion/armed candidates inside one predicted intent.
        key = intent_key(observation, semantic, prior_decisions)
        rank = 0
        if "__complete" in key or "__arm" in key or "__after_draw" in key:
            rank -= 20
        if "__duplicate" in key or "__before_draw" in key:
            rank += 20
        if typ == OPT_ATTACK and _int(semantic.get("attack_id"), 0) == 154:
            rank -= 10
        return rank, _int(semantic.get("index"), 0)

    def rerank(self, observation, chosen, history, matchup=None, confidence=0.0):
        self.stats["calls"] += 1
        if not self.enabled:
            self.stats["disabled"] += 1
            return chosen
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        if not isinstance(chosen, list) or len(chosen) != 1:
            self.stats["not_single"] += 1
            return chosen
        if _int(select.get("context"), -1) != MAIN_CONTEXT or _int(select.get("minCount"), 0) != 1 or _int(select.get("maxCount"), 0) != 1:
            self.stats["not_main"] += 1
            return chosen
        matchup = str(matchup or "unknown")
        gates = self.meta.get("gates") or {}
        gate = gates.get(matchup)
        if not isinstance(gate, dict) or not gate.get("enabled", False):
            self.stats["matchup_off_" + matchup] += 1
            return chosen
        visible = self._visible_opponent_ids(observation)
        signature = set(_int(x,0) for x in (gate.get("signature") or []))
        # A family name inferred from generic trainers is not enough to allow
        # an override.  Wait for a distinctive public Pokémon signature.
        if signature and not (visible & signature):
            self.stats["signature_wait_" + matchup] += 1
            return chosen
        if _int(current.get("turn"), 0) > _int(gate.get("max_turn", self.meta.get("max_turn", 18)), 18):
            self.stats["late"] += 1
            return chosen
        if visible & CRUSTLE_IDS:
            self.stats["crustle_defer"] += 1
            return chosen
        options = select.get("option") or []
        decisions = list(getattr(history, "decisions", []) or [])
        candidates = []
        legal_intents = []
        intent_to_rows = {}
        allowed_types = {OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY, OPT_RETREAT, OPT_ATTACK, OPT_END}
        for index in range(len(options)):
            semantic = semantic_option(observation, index)
            if not semantic or _int(semantic.get("option_type"), -1) not in allowed_types:
                continue
            intent = intent_key(observation, semantic, decisions)
            legal_intents.append(intent)
            row = (index, intent, semantic)
            candidates.append(row)
            intent_to_rows.setdefault(intent, []).append(row)
        if len(candidates) < 2:
            self.stats["few_candidates"] += 1
            return chosen
        base = next((r for r in candidates if r[0] == chosen[0]), None)
        if base is None:
            self.stats["base_unparsed"] += 1
            return chosen
        base_int = base[1]
        # League-calibrated residual: the replay-supported corrections were
        # all cases where the audited core voluntarily passed despite a safe
        # development action.  Do not let imitation override attacks,
        # attachments, retreat plans, draw abilities, or evolutions.
        if base_int != "pass":
            self.stats["base_intent_gate"] += 1
            return chosen
        intents = list(self.meta.get("intents") or [])
        intent_index = {name: i for i, name in enumerate(intents)}
        legal_known = [name for name in sorted(set(legal_intents)) if name in intent_index]
        if len(legal_known) < 2:
            self.stats["few_known_intents"] += 1
            return chosen
        tokens = build_tokens(
            observation, decisions, list(getattr(history, "public_events", []) or []),
            matchup, confidence, legal_known, _int(self.meta.get("max_tokens"), 96),
        )
        ids = tokens_to_ids(tokens, _int(self.meta.get("vocab_size"), 4096), _int(self.meta.get("max_tokens"), 96))
        try:
            logits = self._encode(ids)
        except Exception:
            self.stats["inference_error"] += 1
            return chosen
        masked = np.full_like(logits, -1e9)
        for name in legal_known:
            masked[intent_index[name]] = logits[intent_index[name]]
        probs = _softmax(masked)
        order = np.argsort(-probs)
        top_i = int(order[0])
        second_i = int(order[1]) if len(order) > 1 else top_i
        top_intent = intents[top_i]
        top_prob = float(probs[top_i])
        margin = top_prob - float(probs[second_i])
        self.stats["evaluated"] += 1
        self.stats["top_" + top_intent] += 1
        if top_intent == base_int:
            self.stats["agreed"] += 1
            return chosen
        support = (((self.meta.get("support") or {}).get(matchup) or {}).get(top_intent, 0))
        if support < _int(gate.get("min_support", 12), 12):
            self.stats["low_support"] += 1
            return chosen
        if top_prob < float(gate.get("min_probability", 0.80)) or margin < float(gate.get("min_margin", 0.20)):
            self.stats["low_confidence"] += 1
            return chosen
        rows = intent_to_rows.get(top_intent) or []
        if not rows:
            self.stats["no_candidate"] += 1
            return chosen
        best = min(rows, key=lambda r: self._candidate_rank(observation, r[2], decisions))
        semantic = best[2]
        typ = _int(semantic.get("option_type"), -1)
        # Fail closed on strategic commitments most vulnerable to imitation error.
        if typ == OPT_END:
            self.stats["pass_block"] += 1
            return chosen
        if base_int == "phantom_dive" and top_intent != "phantom_dive":
            self.stats["phantom_veto"] += 1
            return chosen
        if top_intent.endswith("__before_draw"):
            self.stats["before_draw_veto"] += 1
            return chosen
        blocked = set(gate.get("blocked_intents") or [])
        if top_intent in blocked:
            self.stats["intent_blocked"] += 1
            return chosen
        self.stats["overrides"] += 1
        self.stats["override_" + matchup] += 1
        self.stats["from_" + base_int] += 1
        self.stats["to_" + top_intent] += 1
        return [int(best[0])]
