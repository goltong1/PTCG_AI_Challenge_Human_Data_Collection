"""Offline semantic card-text reasoner for the Dragapult agent.

The competition runtime has no reliable network/API access, so this module does
not call a hosted language model.  Instead it executes a compact, LLM-distilled
reasoning graph against the *actual* CardData skill/attack text exposed by the
CABT card tables.  It is card-name agnostic: new Stadiums, Tools, Special Energy,
Abilities and attacks are interpreted from their English rules text at runtime.

The reasoner is deliberately conservative.  It only overrides the merged
causal+temporal policy when the text proves a large tactical effect (for example
an attack-cost tax that disables our Tera attacker, a Stadium that prevents
Phantom Dive's bench counters, or a hand-scaling attack that becomes lethal).
The existing causal and deterministic guards remain authoritative afterwards.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

# CABT enum integers.  Kept dependency-free for Kaggle raw-exec loading.
_MAIN_SELECT = 0
_MAIN_CONTEXT = 0
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
_CARD_POKEMON = 0
_CARD_ITEM = 1
_CARD_TOOL = 2
_CARD_SUPPORTER = 3
_CARD_STADIUM = 4
_CARD_ENERGY = 6

# This deck's semantic anchors.  The parser itself does not special-case the
# opposing card names or IDs.
_DREEPY = 119
_DRAKLOAK = 120
_DRAGAPULT = 121
_MUNKIDORI = 112
_BUDEW = 235
_FEZ = 140
_PHANTOM_DIVE = 154
_JET_HEADBUTT = 153
_ITCHY_POLLEN = 323
_FIRE = 2
_PSYCHIC = 5
_DARK = 7

_ENERGY_SYMBOL_TO_ID = {
    "G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6,
    "D": 7, "M": 8, "N": 9, "C": 0,
}


def _int(value, default=0):
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _norm(text):
    text = str(text or "").replace("\xa0", " ")
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _skill_text(card):
    if card is None:
        return ""
    chunks = []
    for skill in list(getattr(card, "skills", []) or []):
        chunks.append(str(getattr(skill, "text", "") or ""))
    return "\n".join(chunks)


def _attack_text(attack):
    if attack is None:
        return ""
    return str(getattr(attack, "text", "") or "")


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
        if not (0 <= _int(player_index, -1) < len(players)):
            return None
        key = {
            _AREA_HAND: "hand", _AREA_DISCARD: "discard", _AREA_ACTIVE: "active",
            _AREA_BENCH: "bench", _AREA_PRIZE: "prize",
        }.get(area)
        cards = players[_int(player_index)].get(key, []) if key else []
        cards = cards or []
    idx = _int(index, -1)
    if 0 <= idx < len(cards) and isinstance(cards[idx], dict):
        return cards[idx]
    return None


def _source_card(observation, option):
    typ = _int(option.get("type"), -1)
    current = observation.get("current") or {}
    me = _int(current.get("yourIndex"), 0)
    if typ == _OPT_PLAY:
        return _cards_from_area(observation, _AREA_HAND, option.get("index"), me)
    if typ in (_OPT_ATTACH, _OPT_EVOLVE, _OPT_ABILITY):
        return _cards_from_area(
            observation, _int(option.get("area"), -1), option.get("index"),
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
        observation, _int(option.get("inPlayArea"), -1), option.get("inPlayIndex"),
        _int(current.get("yourIndex"), 0),
    )


def _semantic(observation, index):
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not isinstance(index, int) or not (0 <= index < len(options)):
        return None
    option = options[index]
    source = _source_card(observation, option)
    target = _target_card(observation, option)
    return {
        "index": index,
        "type": _int(option.get("type"), -1),
        "card_id": _int((source or {}).get("id", option.get("cardId")), 0),
        "serial": _int((source or {}).get("serial", option.get("serial")), -1),
        "target_id": _int((target or {}).get("id"), 0),
        "target_serial": _int((target or {}).get("serial"), -1),
        "attack_id": _int(option.get("attackId"), 0),
    }


def _token(sem):
    if not sem:
        return "invalid"
    typ, cid, tgt = sem["type"], sem["card_id"], sem["target_id"]
    if typ == _OPT_PLAY:
        return "p%d" % cid
    if typ == _OPT_ATTACH:
        return "h%d>%d#%d" % (cid, tgt, _int(sem.get("target_serial"), -1))
    if typ == _OPT_EVOLVE:
        return "e%d>%d#%d" % (cid, tgt, _int(sem.get("target_serial"), -1))
    if typ == _OPT_ABILITY:
        return "b%d" % cid
    if typ == _OPT_RETREAT:
        return "r%d" % cid
    if typ == _OPT_ATTACK:
        return "x%d" % sem["attack_id"]
    if typ == _OPT_END:
        return "z"
    return "t%d:c%d" % (typ, cid)


def _board(player):
    return [p for p in (player.get("active") or []) + (player.get("bench") or []) if isinstance(p, dict)]


def _active(player):
    return next((p for p in (player.get("active") or []) if isinstance(p, dict)), None)


def _energy_cards(pokemon):
    return [e for e in (pokemon.get("energyCards") or []) if isinstance(e, dict)] if isinstance(pokemon, dict) else []


def _visible_cards(player):
    out = []
    for key in ("hand", "discard", "active", "bench"):
        for card in player.get(key) or []:
            if not isinstance(card, dict):
                continue
            out.append(card)
            for pre in card.get("preEvolution") or []:
                if isinstance(pre, dict):
                    out.append(pre)
            for tool in card.get("tools") or []:
                if isinstance(tool, dict):
                    out.append(tool)
            for energy in card.get("energyCards") or []:
                if isinstance(energy, dict):
                    out.append(energy)
    return out


def _history_turn_tokens(history, turn):
    out = []
    for d in list(getattr(history, "decisions", []) or []):
        if _int(d.get("context"), -1) != _MAIN_CONTEXT or _int(d.get("turn"), -1) != turn:
            continue
        typ = _int(d.get("option_type"), -1)
        cid = _int(d.get("card_id"), 0)
        if typ == _OPT_PLAY:
            out.append("p%d" % cid)
        elif typ == _OPT_ATTACH:
            out.append("h%d>%d" % (cid, _int(d.get("target_card_id"), 0)))
        elif typ == _OPT_EVOLVE:
            out.append("e%d>%d" % (cid, _int(d.get("target_card_id"), 0)))
        elif typ == _OPT_ABILITY:
            out.append("b%d" % cid)
        elif typ == _OPT_ATTACK:
            out.append("x%d" % _int(d.get("attack_id"), 0))
        elif typ == _OPT_RETREAT:
            out.append("r%d" % cid)
        elif typ == _OPT_END:
            out.append("z")
    return out


class TextProfile:
    __slots__ = (
        "text", "cost_more", "affects_tera", "affects_basic", "affects_bench",
        "affects_active", "both_sides", "opponent_only", "self_only", "hp_bonus",
        "prevent_bench_counters", "prevent_damage", "prevent_effects", "tool_disable",
        "ability_disable", "item_lock", "search", "heal", "hand_reset",
        "opponent_draw", "own_draw", "energy_discard", "energy_accel",
        "hand_counter_mult", "energy_counter_mult", "damage_counter_flat",
    )

    def __init__(self, text=""):
        t = _norm(text)
        self.text = t
        self.affects_tera = "tera pokemon" in t or "tera pokémon" in t
        self.affects_basic = "basic pokemon" in t or "basic pokémon" in t
        self.affects_bench = "benched pokemon" in t or "benched pokémon" in t or "bench" in t
        self.affects_active = "active pokemon" in t or "active pokémon" in t or "active spot" in t
        self.both_sides = "both yours and your opponent" in t or "each player" in t
        self.opponent_only = "your opponent's" in t or "your opponent’s" in t
        self.self_only = ("your pokemon" in t or "your pokémon" in t) and not self.both_sides
        self.cost_more = 0
        if "attacks used by" in t and " cost " in t and " more" in t:
            m = re.search(r"cost\s+((?:\{[a-z0-9]+\}\s*)+)more", t)
            if m:
                self.cost_more = max(1, len(re.findall(r"\{[^}]+\}", m.group(1))))
            else:
                self.cost_more = 1
        m = re.search(r"gets?\s+\+(\d+)\s*hp", t)
        self.hp_bonus = _int(m.group(1), 0) if m else 0
        self.prevent_bench_counters = "prevent all damage counters" in t and self.affects_bench
        self.prevent_damage = "prevent all damage" in t and not self.prevent_bench_counters
        self.prevent_effects = "prevent all effects of attacks" in t
        self.tool_disable = ("pokemon tools" in t or "pokémon tools" in t) and "have no effect" in t
        self.ability_disable = ("abilities" in t or "ability" in t) and ("have no effect" in t or "can't use" in t or "cannot use" in t)
        self.item_lock = ("can't play any item" in t or "cannot play any item" in t)
        self.search = "search your deck" in t
        self.heal = 0
        m = re.search(r"heal\s+(\d+)\s+damage", t)
        if m:
            self.heal = _int(m.group(1), 0)
        self.hand_reset = "shuffle" in t and "hand" in t and "deck" in t
        self.opponent_draw = None
        m = re.search(r"opponent draws?\s+(\d+)\s+cards?", t)
        if m:
            self.opponent_draw = _int(m.group(1), 0)
        elif "each player" in t:
            m = re.search(r"draws?\s+(\d+)\s+cards?", t)
            if m:
                self.opponent_draw = _int(m.group(1), 0)
        self.own_draw = None
        m = re.search(r"you draw\s+(\d+)\s+cards?", t)
        if m:
            self.own_draw = _int(m.group(1), 0)
        elif "each player" in t:
            m = re.search(r"draws?\s+(\d+)\s+cards?", t)
            if m:
                self.own_draw = _int(m.group(1), 0)
        self.energy_discard = "discard an energy" in t or "discard a special energy" in t
        self.energy_accel = "attach" in t and "energy" in t
        self.hand_counter_mult = 0
        m = re.search(r"(?:place|put)\s+(\d+)\s+damage counters?[^.]*for each card in (?:your|their) hand", t)
        if m:
            self.hand_counter_mult = 10 * _int(m.group(1), 0)
        else:
            m = re.search(r"(\d+)\s+damage[^.]*for each card in (?:your|their) hand", t)
            if m:
                self.hand_counter_mult = _int(m.group(1), 0)
        self.energy_counter_mult = 0
        m = re.search(r"(?:place|put)\s+(\d+)\s+damage counters?[^.]*for each energy", t)
        if m:
            self.energy_counter_mult = 10 * _int(m.group(1), 0)
        else:
            m = re.search(r"(\d+)\s+damage[^.]*for each energy", t)
            if m:
                self.energy_counter_mult = _int(m.group(1), 0)
        self.damage_counter_flat = 0
        m = re.search(r"(?:place|put)\s+(\d+)\s+damage counters?", t)
        if m and not self.hand_counter_mult and not self.energy_counter_mult:
            self.damage_counter_flat = 10 * _int(m.group(1), 0)


class SemanticLLMReasoner:
    def __init__(self, config_path, card_table=None, attack_table=None, deck_ids=None):
        self.card_table = card_table or {}
        self.attack_table = attack_table or {}
        self.deck_ids = list(deck_ids or [])
        self.deck_counts = Counter(_int(x, 0) for x in self.deck_ids)
        self.config = {}
        try:
            with open(config_path, encoding="utf-8") as handle:
                self.config = json.load(handle)
        except Exception:
            self.config = {}
        self.stats = Counter()
        self._card_profiles = {}
        self._attack_profiles = {}
        self._deck_tera_ids = [
            cid for cid, count in self.deck_counts.items()
            if count > 0 and self._is_tera_card_id(cid)
        ]
        self._tera_line_names = self._build_tera_line_names()

    @property
    def enabled(self):
        return bool(self.config.get("enabled", True))

    def reset(self):
        self.stats.clear()

    def card_profile(self, card_id):
        cid = _int(card_id, 0)
        if cid not in self._card_profiles:
            card = self.card_table.get(cid)
            self._card_profiles[cid] = TextProfile(_skill_text(card))
            self.stats["card_text_profiles"] += 1
        return self._card_profiles[cid]

    def attack_profile(self, attack_id):
        aid = _int(attack_id, 0)
        if aid not in self._attack_profiles:
            attack = self.attack_table.get(aid)
            self._attack_profiles[aid] = TextProfile(_attack_text(attack))
            self.stats["attack_text_profiles"] += 1
        return self._attack_profiles[aid]

    def _is_tera_card_id(self, card_id):
        data = self.card_table.get(_int(card_id, 0))
        return bool(getattr(data, "tera", False)) if data is not None else False

    def _build_tera_line_names(self):
        names = set()
        by_name = {}
        for cid, data in self.card_table.items():
            name = str(getattr(data, "name", "") or "")
            if name:
                by_name.setdefault(name, []).append(data)
        frontier = []
        for cid in self._deck_tera_ids:
            data = self.card_table.get(cid)
            if data is not None:
                name = str(getattr(data, "name", "") or "")
                if name:
                    names.add(name); frontier.append(data)
        # Follow evolvesFrom recursively so the logic generalizes to any Tera line.
        seen = set()
        while frontier:
            data = frontier.pop()
            prev = str(getattr(data, "evolvesFrom", "") or "")
            if not prev or prev in seen:
                continue
            seen.add(prev); names.add(prev)
            frontier.extend(by_name.get(prev, []))
        return names

    def _is_tera_line_card(self, card_id):
        data = self.card_table.get(_int(card_id, 0))
        return bool(data is not None and str(getattr(data, "name", "") or "") in self._tera_line_names)

    def _has_live_tera_plan(self, player):
        if not self._deck_tera_ids:
            return False
        for card in _visible_cards(player):
            if self._is_tera_line_card(card.get("id")):
                return True
        return any(self.deck_counts.get(cid, 0) > 0 for cid in self._deck_tera_ids)

    def _is_tera(self, pokemon):
        if not isinstance(pokemon, dict):
            return False
        data = self.card_table.get(_int(pokemon.get("id"), 0))
        return bool(getattr(data, "tera", False)) if data is not None else False

    def _is_basic(self, pokemon):
        if not isinstance(pokemon, dict):
            return False
        data = self.card_table.get(_int(pokemon.get("id"), 0))
        return bool(getattr(data, "basic", False)) if data is not None else False

    def _energy_type(self, energy_card):
        cid = _int((energy_card or {}).get("id"), 0)
        data = self.card_table.get(cid)
        if data is None:
            return cid if 0 <= cid <= 9 else 0
        return _int(getattr(data, "energyType", None), cid if 0 <= cid <= 9 else 0)

    def _attack_ready(self, pokemon, attack, extra_colorless=0):
        if not isinstance(pokemon, dict) or attack is None:
            return False
        supplies = [self._energy_type(e) for e in _energy_cards(pokemon)]
        requirements = list(getattr(attack, "energies", []) or []) + [0] * max(0, _int(extra_colorless, 0))
        used = [False] * len(supplies)
        # Pay typed costs first.
        for req in [r for r in requirements if _int(r, 0) != 0]:
            found = None
            for i, supplied in enumerate(supplies):
                if not used[i] and _int(supplied, 0) == _int(req, 0):
                    found = i
                    break
            if found is None:
                return False
            used[found] = True
        colorless = sum(1 for r in requirements if _int(r, 0) == 0)
        return sum(1 for x in used if not x) >= colorless

    def _stadium_affects(self, profile, pokemon):
        if not isinstance(pokemon, dict):
            return False
        if profile.affects_tera and not self._is_tera(pokemon):
            return False
        if profile.affects_basic and not self._is_basic(pokemon):
            return False
        return profile.affects_tera or profile.affects_basic or profile.affects_bench or profile.affects_active or True

    def _scan_scene_text(self, current, sems, me):
        """Parse every currently visible or legally actionable card once.

        The cached profiles make this inexpensive after first sight, while the
        scan guarantees that a card never has to be pre-registered by name.
        """
        players = current.get("players") or []
        ids = set()
        for player in players:
            for card in _visible_cards(player):
                ids.add(_int(card.get("id"), 0))
        for card in current.get("stadium") or []:
            if isinstance(card, dict):
                ids.add(_int(card.get("id"), 0))
        for sem in sems:
            if sem and sem.get("card_id"):
                ids.add(_int(sem.get("card_id"), 0))
        for cid in ids:
            if cid <= 0:
                continue
            self.card_profile(cid)
            data = self.card_table.get(cid)
            if data is not None:
                for aid in list(getattr(data, "attacks", []) or []):
                    self.attack_profile(aid)
        self.stats["visible_text_scans"] += 1
        self.stats["visible_cards_interpreted"] += len(ids)

    def _tool_value(self, player):
        value = 0.0
        for pokemon in _board(player):
            for tool in pokemon.get("tools") or []:
                if not isinstance(tool, dict):
                    continue
                p = self.card_profile(tool.get("id"))
                value += p.hp_bonus * 0.30
                value += p.heal * 0.15
                if p.prevent_damage or p.prevent_effects:
                    value += 25.0
                if not (p.hp_bonus or p.heal or p.prevent_damage or p.prevent_effects):
                    value += 10.0
        return value

    def _has_bench_counter_plan(self, player):
        for pokemon in _board(player):
            data = self.card_table.get(_int(pokemon.get("id"), 0))
            if data is None:
                continue
            for aid in list(getattr(data, "attacks", []) or []):
                text = _norm(_attack_text(self.attack_table.get(_int(aid, 0))))
                if "damage counters" in text and ("benched" in text or "bench" in text):
                    return True
            for skill in list(getattr(data, "skills", []) or []):
                text = _norm(getattr(skill, "text", ""))
                if "damage counters" in text and ("benched" in text or "bench" in text):
                    return True
        return False

    def _stadium_value(self, card_id, owner, current, me):
        cid = _int(card_id, 0)
        profile = self.card_profile(cid)
        players = current.get("players") or []
        if len(players) < 2:
            return 0.0, []
        mine, opp = players[me], players[1 - me]
        own_board, opp_board = _board(mine), _board(opp)
        reasons = []
        value = 0.0

        if profile.cost_more:
            own_affected = [p for p in own_board if self._stadium_affects(profile, p)]
            opp_affected = [p for p in opp_board if self._stadium_affects(profile, p)]
            value += 24.0 * profile.cost_more * (len(opp_affected) - len(own_affected))
            own_active, opp_active = _active(mine), _active(opp)
            for pokemon, sign, label in ((own_active, -1.0, "own"), (opp_active, 1.0, "opp")):
                if not self._stadium_affects(profile, pokemon):
                    continue
                data = self.card_table.get(_int((pokemon or {}).get("id"), 0))
                if data is None:
                    continue
                for aid in list(getattr(data, "attacks", []) or []):
                    attack = self.attack_table.get(_int(aid, 0))
                    if self._attack_ready(pokemon, attack, 0) and not self._attack_ready(pokemon, attack, profile.cost_more):
                        value += sign * 115.0
                        reasons.append(label + "_ready_attack_taxed")
                        break
            # A text tax can be strategically decisive even before the Tera
            # attacker reaches play.  Penalize it when this deck has a live
            # Tera evolution plan and the opponent is not equally taxed.
            if profile.affects_tera and not own_affected and self._has_live_tera_plan(mine):
                opp_has_tera_plan = any(self._is_tera(p) for p in opp_board)
                if not opp_has_tera_plan:
                    value -= float(self.config.get("future_tera_tax_penalty", 46.0)) * profile.cost_more
                    reasons.append("taxes_core_tera_plan")
            reasons.append("attack_cost_tax")

        if profile.hp_bonus:
            own_n = sum(1 for p in own_board if (not profile.affects_basic or self._is_basic(p)))
            opp_n = sum(1 for p in opp_board if (not profile.affects_basic or self._is_basic(p)))
            value += 0.10 * profile.hp_bonus * (own_n - opp_n)
            reasons.append("hp_modifier")

        if profile.prevent_bench_counters:
            if self._has_bench_counter_plan(mine):
                value -= 95.0
                reasons.append("blocks_own_bench_counter_plan")
            if self._has_bench_counter_plan(opp):
                value += 50.0
                reasons.append("blocks_opponent_bench_counter_plan")

        if profile.tool_disable:
            swing = self._tool_value(opp) - self._tool_value(mine)
            value += swing
            reasons.append("tool_disable")

        if profile.ability_disable:
            own_abilities = sum(len(getattr(self.card_table.get(_int(p.get("id"), 0)), "skills", []) or []) for p in own_board)
            opp_abilities = sum(len(getattr(self.card_table.get(_int(p.get("id"), 0)), "skills", []) or []) for p in opp_board)
            value += 20.0 * (opp_abilities - own_abilities)
            reasons.append("ability_disable")

        # Unknown clauses receive only a small owner prior.  This allows unseen
        # text to be handled once a supported semantic is detected without
        # blindly replacing every unfamiliar Stadium.
        if not reasons:
            value += 5.0 if _int(owner, -1) == me else -6.0
            reasons.append("unknown_owner_prior")
        return value, reasons

    def _current_stadium(self, current):
        stadium = current.get("stadium") or []
        if not stadium or not isinstance(stadium[0], dict):
            return 0, -1
        return _int(stadium[0].get("id"), 0), _int(stadium[0].get("playerIndex"), -1)

    def _attack_extra_cost(self, current, me, pokemon):
        sid, _owner = self._current_stadium(current)
        if not sid:
            return 0
        profile = self.card_profile(sid)
        return profile.cost_more if self._stadium_affects(profile, pokemon) else 0

    def _estimated_attack_damage(self, attack, profile, attacker, player):
        damage = _int(getattr(attack, "damage", 0), 0)
        hand_n = _int(player.get("handCount"), len(player.get("hand") or []))
        if profile.hand_counter_mult:
            damage = max(damage, profile.hand_counter_mult * hand_n)
        if profile.energy_counter_mult:
            total_energy = sum(len(_energy_cards(p)) for p in _board(player))
            damage = max(damage, profile.energy_counter_mult * total_energy)
        if profile.damage_counter_flat:
            damage = max(damage, profile.damage_counter_flat)
        return damage

    def _opponent_lethal(self, current, me):
        players = current.get("players") or []
        if len(players) < 2:
            return None
        mine, opp = players[me], players[1 - me]
        target = _active(mine)
        attacker = _active(opp)
        if target is None or attacker is None:
            return None
        data = self.card_table.get(_int(attacker.get("id"), 0))
        if data is None:
            return None
        extra = self._attack_extra_cost(current, 1 - me, attacker)
        best = None
        for aid in list(getattr(data, "attacks", []) or []):
            attack = self.attack_table.get(_int(aid, 0))
            if attack is None or not self._attack_ready(attacker, attack, extra):
                continue
            profile = self.attack_profile(aid)
            damage = self._estimated_attack_damage(attack, profile, attacker, opp)
            row = {"attack_id": _int(aid), "damage": damage, "profile": profile, "target_hp": _int(target.get("hp"), 0)}
            if best is None or row["damage"] > best["damage"]:
                best = row
        return best

    def _reset_candidates(self, sems):
        rows = []
        for sem in sems:
            if not sem or sem["type"] != _OPT_PLAY:
                continue
            profile = self.card_profile(sem["card_id"])
            if profile.hand_reset and profile.opponent_draw is not None:
                rows.append((profile.opponent_draw, sem["index"], sem["card_id"]))
        return sorted(rows)

    def _stadium_candidates(self, sems):
        out = []
        for sem in sems:
            if not sem or sem["type"] != _OPT_PLAY:
                continue
            data = self.card_table.get(sem["card_id"])
            if data is not None and _int(getattr(data, "cardType", -1), -1) == _CARD_STADIUM:
                out.append(sem)
        return out

    def _high_impact_taxed_attack(self, current, me, pokemon, extra):
        data = self.card_table.get(_int((pokemon or {}).get("id"), 0))
        if data is None:
            return None
        best = None
        for aid in list(getattr(data, "attacks", []) or []):
            attack = self.attack_table.get(_int(aid, 0))
            if attack is None or not self._attack_ready(pokemon, attack, 0) or self._attack_ready(pokemon, attack, extra):
                continue
            text = _norm(_attack_text(attack))
            impact = _int(getattr(attack, "damage", 0), 0)
            if "damage counters" in text and ("benched" in text or "bench" in text):
                impact += 60
            if impact >= _int(self.config.get("taxed_attack_min_impact", 150), 150):
                row = (impact, _int(aid, 0), attack)
                if best is None or row[0] > best[0]:
                    best = row
        return best

    def _evolution_toward_tera(self, sems):
        rows = [s for s in sems if s and s["type"] == _OPT_EVOLVE and self._is_tera_line_card(s["card_id"])]
        if not rows:
            return None
        return max(rows, key=lambda s: (
            bool(self._is_tera_card_id(s["card_id"])),
            _int(getattr(self.card_table.get(s["card_id"]), "stage2", False), 0),
            s["card_id"],
        ))["index"]

    def _remaining_definitely_zero(self, card_ids, mine):
        wanted = {_int(x, 0) for x in card_ids}
        total = sum(self.deck_counts.get(cid, 0) for cid in wanted)
        if total <= 0:
            return True
        visible = sum(1 for c in _visible_cards(mine) if _int(c.get("id"), 0) in wanted)
        return visible >= total

    def _search_is_proven_empty(self, card_id, mine):
        profile = self.card_profile(card_id)
        text = profile.text
        if "doesn't have a rule box" in text or "does not have a rule box" in text:
            eligible = []
            for cid, count in self.deck_counts.items():
                if count <= 0:
                    continue
                data = self.card_table.get(cid)
                if data is None or _int(getattr(data, "cardType", -1), -1) != _CARD_POKEMON:
                    continue
                if not bool(getattr(data, "ex", False)) and not bool(getattr(data, "megaEx", False)):
                    eligible.append(cid)
            return self._remaining_definitely_zero(eligible, mine)
        if "70 hp or less" in text and "basic" in text:
            eligible = []
            for cid, count in self.deck_counts.items():
                data = self.card_table.get(cid)
                if count > 0 and data is not None and bool(getattr(data, "basic", False)) and _int(getattr(data, "hp", 0), 0) <= 70:
                    eligible.append(cid)
            return self._remaining_definitely_zero(eligible, mine)
        return False

    def _choose(self, chosen, sems, index, rule):
        if index is None or not isinstance(index, int):
            return chosen
        base = chosen[0] if isinstance(chosen, list) and len(chosen) == 1 else None
        if base == index:
            self.stats["kept_" + rule] += 1
            return chosen
        if base is not None and 0 <= base < len(sems) and _token(sems[base]) == _token(sems[index]):
            self.stats["kept_semantic_duplicate"] += 1
            return chosen
        self.stats["overrides"] += 1
        self.stats["rule_" + rule] += 1
        return [index]

    def rerank(self, observation, chosen, history, matchup=None, confidence=0.0):
        if not self.enabled or not isinstance(chosen, list) or len(chosen) != 1:
            return chosen
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        if _int(select.get("type"), -1) != _MAIN_SELECT or _int(select.get("context"), -1) != _MAIN_CONTEXT:
            return chosen
        if _int(select.get("minCount"), 0) != 1 or _int(select.get("maxCount"), 0) != 1:
            return chosen
        options = select.get("option") or []
        sems = [_semantic(observation, i) for i in range(len(options))]
        if not sems:
            return chosen
        players = current.get("players") or []
        me = _int(current.get("yourIndex"), 0)
        self._scan_scene_text(current, sems, me)
        if len(players) < 2:
            return chosen
        mine, opp = players[me], players[1 - me]
        own_active = _active(mine)
        turn = _int(current.get("turn"), 0)
        ta = _int(current.get("turnActionCount"), 0)
        turn_tokens = _history_turn_tokens(history, turn)
        base_sem = sems[chosen[0]] if 0 <= chosen[0] < len(sems) else None
        base_type = base_sem["type"] if base_sem else -1
        base_card = base_sem["card_id"] if base_sem else 0
        base_attack = base_sem["attack_id"] if base_sem else 0

        def first(pred):
            return next((s["index"] for s in sems if s and pred(s)), None)

        phantom = first(lambda s: s["type"] == _OPT_ATTACK and s["attack_id"] == _PHANTOM_DIVE)
        any_attack = [s for s in sems if s and s["type"] == _OPT_ATTACK]

        # 1) Read the current Stadium text and immediately replace a proven
        # harmful effect with the best legal Stadium.  This catches Nighttime
        # Mine, Battle Cage, and future cards with equivalent wording.
        sid, owner = self._current_stadium(current)
        stadium_options = self._stadium_candidates(sems)
        if sid and stadium_options:
            current_value, current_reasons = self._stadium_value(sid, owner, current, me)
            ranked = []
            for sem in stadium_options:
                value, reasons = self._stadium_value(sem["card_id"], me, current, me)
                ranked.append((value, sem["index"], sem["card_id"], reasons))
            best_value, best_index, _best_cid, _best_reasons = max(ranked)
            critical = any(r in current_reasons for r in ("own_ready_attack_taxed", "blocks_own_bench_counter_plan"))
            threshold = float(self.config.get("stadium_harm_threshold", -30.0))
            margin = float(self.config.get("stadium_replace_margin", 24.0))
            if (critical or current_value <= threshold) and best_value >= current_value + margin:
                # When the text only threatens a future Tera plan, take a legal
                # evolution toward that plan first; the replacement remains
                # available afterward.  Free abilities are also safe first.
                future_only = "taxes_core_tera_plan" in current_reasons and "own_ready_attack_taxed" not in current_reasons
                if future_only:
                    evo = self._evolution_toward_tera(sems)
                    if evo is not None:
                        self.stats["read_stadium_text"] += 1
                        return self._choose(chosen, sems, evo, "develop_before_future_stadium_replacement")
                if base_type != _OPT_ABILITY:
                    self.stats["read_stadium_text"] += 1
                    return self._choose(chosen, sems, best_index, "semantic_stadium_replacement")

        # 2) If an unseen Stadium adds an attack cost and no replacement is in
        # hand, attach to the Active only when that single attachment restores a
        # proven high-impact attack.  This avoids wasting the turn to enable a
        # low-value attack while still fixing taxes on Phantom-Dive-like plays.
        if sid and own_active is not None and not stadium_options:
            p = self.card_profile(sid)
            extra = p.cost_more if self._stadium_affects(p, own_active) else 0
            taxed_attack = self._high_impact_taxed_attack(current, me, own_active, extra) if extra else None
            if taxed_attack is not None:
                active_serial = _int(own_active.get("serial"), -1)
                attach_rows = [s for s in sems if s and s["type"] == _OPT_ATTACH and s["target_serial"] == active_serial]
                # Verify the selected attachment actually makes the attack legal.
                valid_attach = None
                for sem in attach_rows:
                    source = next((c for c in (mine.get("hand") or []) if isinstance(c, dict) and _int(c.get("serial"), -2) == sem["serial"]), None)
                    if source is None:
                        continue
                    simulated = dict(own_active)
                    simulated["energyCards"] = list(_energy_cards(own_active)) + [source]
                    if self._attack_ready(simulated, taxed_attack[2], extra):
                        valid_attach = sem["index"]
                        break
                if valid_attach is not None and base_type != _OPT_ABILITY:
                    self.stats["read_stadium_text"] += 1
                    return self._choose(chosen, sems, valid_attach, "pay_high_impact_semantic_attack_tax")

        # 3) Read the opposing Active attack text.  If a hand-scaling attack is
        # already ready and a legal reset proves a large/lethal reduction, use
        # the reset before optional development.  This handles Powerful Hand
        # without an Alakazam-specific ID rule.
        threat = self._opponent_lethal(current, me)
        resets = self._reset_candidates(sems)
        if threat and resets and threat["profile"].hand_counter_mult:
            before = threat["damage"]
            draw_n, reset_index, _reset_cid = resets[0]
            after = threat["profile"].hand_counter_mult * draw_n
            lethal_before = before >= threat["target_hp"] > 0
            lethal_after = after >= threat["target_hp"] > 0
            reduction = before - after
            if (lethal_before and not lethal_after) or reduction >= _int(self.config.get("hand_threat_reduction", 120), 120):
                if base_type not in (_OPT_ATTACK, _OPT_ABILITY):
                    self.stats["read_attack_text"] += 1
                    return self._choose(chosen, sems, reset_index, "semantic_hand_threat_reset")

        # 4) Text-proven empty search: do not spend a search item when every
        # eligible card is visibly outside the deck.  Prefer an available
        # attack; otherwise preserve the original policy if no safe alternative
        # can be proven.
        if bool(self.config.get("enable_semantic_empty_search_override", False)) and base_sem and base_type == _OPT_PLAY and self._search_is_proven_empty(base_card, mine):
            if any_attack:
                best = max(any_attack, key=lambda s: (_int(getattr(self.attack_table.get(s["attack_id"]), "damage", 0), 0), s["attack_id"]))
                return self._choose(chosen, sems, best["index"], "semantic_empty_search_to_attack")

        # 5) Action-efficiency guard.  Once Phantom Dive is legal, repeated
        # low-impact search or a very long MAIN sequence yields to the attack.
        if bool(self.config.get("enable_efficiency_commit", False)) and phantom is not None:
            low_impact = False
            if base_type == _OPT_END or (base_type == _OPT_ATTACK and base_attack == _JET_HEADBUTT):
                low_impact = True
            elif base_type == _OPT_PLAY:
                profile = self.card_profile(base_card)
                repeated = turn_tokens.count("p%d" % base_card) > 0
                low_impact = (profile.search and repeated) or (profile.energy_discard and repeated)
            if low_impact and (ta >= _int(self.config.get("max_pre_attack_actions", 14), 14) or (base_type == _OPT_PLAY and turn_tokens.count("p%d" % base_card) > 0)):
                return self._choose(chosen, sems, phantom, "efficient_phantom_commit")

        # 6) Never voluntarily pass a legal attack.  This remains generic and
        # only fires on END, leaving all development sequencing to the merged
        # base policy.
        if bool(self.config.get("enable_attack_over_pass", True)) and any_attack and base_type == _OPT_END:
            best = max(any_attack, key=lambda s: (_int(getattr(self.attack_table.get(s["attack_id"]), "damage", 0), 0), s["attack_id"]))
            return self._choose(chosen, sems, best["index"], "attack_over_voluntary_pass")

        self.stats["kept"] += 1
        return chosen

    def get_stats(self):
        return {"semantic_llm_" + str(k): int(v) for k, v in self.stats.items()}


def default_config_path(root):
    return os.path.join(root, "semantic_llm_config.json")
