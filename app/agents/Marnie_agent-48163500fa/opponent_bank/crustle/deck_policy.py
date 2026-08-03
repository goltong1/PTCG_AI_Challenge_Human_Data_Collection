"""Replay-informed Crustle / Mega Kangaskhan agent for Pokémon TCG AI Battle.

The public entry point intentionally consumes the raw observation dictionary and
has no import-time dependency on ``cg``.  The policy is deterministic and uses
board, hand, threat, prize-race, gust, spread-damage, healing, and deck-out
signals rather than a fixed action ordering.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# CABT enum values
# ---------------------------------------------------------------------------

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_PRE_EVOLUTION = 10
AREA_PLAYER = 11
AREA_LOOKING = 12

CONTEXT_MAIN = 0
CONTEXT_SETUP_ACTIVE = 1
CONTEXT_SETUP_BENCH = 2
CONTEXT_SWITCH = 3
CONTEXT_TO_ACTIVE = 4
CONTEXT_TO_BENCH = 5
CONTEXT_TO_FIELD = 6
CONTEXT_TO_HAND = 7
CONTEXT_DISCARD = 8
CONTEXT_TO_DECK = 9
CONTEXT_TO_DECK_BOTTOM = 10
CONTEXT_TO_PRIZE = 11
CONTEXT_NOT_MOVE = 12
CONTEXT_DAMAGE_COUNTER = 13
CONTEXT_DAMAGE_COUNTER_ANY = 14
CONTEXT_DAMAGE = 15
CONTEXT_REMOVE_DAMAGE_COUNTER = 16
CONTEXT_HEAL = 17
CONTEXT_EVOLVES_FROM = 18
CONTEXT_EVOLVES_TO = 19
CONTEXT_DEVOLVE = 20
CONTEXT_ATTACH_FROM = 21
CONTEXT_ATTACH_TO = 22
CONTEXT_DETACH_FROM = 23
CONTEXT_LOOK = 24
CONTEXT_EFFECT_TARGET = 25
CONTEXT_DISCARD_ENERGY_CARD = 26
CONTEXT_DISCARD_TOOL_CARD = 27
CONTEXT_SWITCH_ENERGY_CARD = 28
CONTEXT_DISCARD_CARD_OR_ATTACHED = 29
CONTEXT_DISCARD_ENERGY = 30
CONTEXT_TO_HAND_ENERGY = 31
CONTEXT_TO_DECK_ENERGY = 32
CONTEXT_SWITCH_ENERGY = 33
CONTEXT_SKILL_ORDER = 34
CONTEXT_ATTACK = 35
CONTEXT_DISABLE_ATTACK = 36
CONTEXT_EVOLVE = 37
CONTEXT_DRAW_COUNT = 38
CONTEXT_DAMAGE_COUNTER_COUNT = 39
CONTEXT_REMOVE_DAMAGE_COUNTER_COUNT = 40
CONTEXT_IS_FIRST = 41
CONTEXT_MULLIGAN = 42
CONTEXT_ACTIVATE = 43
CONTEXT_FIRST_EFFECT = 44
CONTEXT_MORE_DEVOLVE = 45
CONTEXT_COIN_HEAD = 46
CONTEXT_AFFECT_SPECIAL_CONDITION = 47
CONTEXT_RECOVER_SPECIAL_CONDITION = 48

OPTION_NUMBER = 0
OPTION_YES = 1
OPTION_NO = 2
OPTION_CARD = 3
OPTION_TOOL_CARD = 4
OPTION_ENERGY_CARD = 5
OPTION_ENERGY = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_ABILITY = 10
OPTION_DISCARD = 11
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14
OPTION_SKILL = 15
OPTION_SPECIAL_CONDITION = 16

# ---------------------------------------------------------------------------
# Deck IDs
# ---------------------------------------------------------------------------

MEGA_KANGASKHAN_EX = 756
DWEBBLE = 344
CRUSTLE = 345

LILLIES_DETERMINATION = 1227
BOSSS_ORDERS = 1182
TEAM_ROCKETS_PETREL = 1219
HILDA = 1225
ERI = 1186
XEROSICS_MACHINATIONS = 1197
# The competition card database bundled with the supplied agent does not expose
# Pokémon Center Lady MEG 123.  Cook (1212) is the engine-valid healing slot
# used in the supplied Crustle replay list and is kept as the compatibility ID.
HEAL_SUPPORTER = 1212
BIANCAS_DEVOTION = 1190
LISIAS_APPEAL = 1204
JUMBO_ICE_CREAM = 1147
POKEGEAR_30 = 1122
BUDDY_BUDDY_POFFIN = 1086
ULTRA_BALL = 1121
SWITCH = 1123
HAND_TRIMMER = 1087
HEROS_CAPE = 1159
HANDHELD_FAN = 1161
TEAM_ROCKETS_FACTORY = 1257
COMMUNITY_CENTER = 1242
FESTIVAL_GROUNDS = 1245

SPIKY_ENERGY = 14
GROW_GRASS_ENERGY = 18
MIST_ENERGY = 11
BASIC_GRASS_ENERGY = 1

ENERGY_IDS = {SPIKY_ENERGY, GROW_GRASS_ENERGY, MIST_ENERGY, BASIC_GRASS_ENERGY}
POKEMON_IDS = {MEGA_KANGASKHAN_EX, DWEBBLE, CRUSTLE}
SUPPORTER_IDS = {
    LILLIES_DETERMINATION,
    BOSSS_ORDERS,
    TEAM_ROCKETS_PETREL,
    HILDA,
    ERI,
    XEROSICS_MACHINATIONS,
    HEAL_SUPPORTER,
    BIANCAS_DEVOTION,
    LISIAS_APPEAL,
}
STADIUM_IDS = {TEAM_ROCKETS_FACTORY, COMMUNITY_CENTER, FESTIVAL_GROUNDS}
TOOL_IDS = {HEROS_CAPE, HANDHELD_FAN}
ITEM_IDS = {JUMBO_ICE_CREAM, POKEGEAR_30, BUDDY_BUDDY_POFFIN, ULTRA_BALL, SWITCH, HAND_TRIMMER}

# Known attacks in the supplied engine/replays.
ASCENSION = 478
CRUSTLE_ATTACK = 479
RAPID_FIRE_COMBO = 1092

# High-impact common threats.  This is only a bonus layer; unknown opponents are
# still evaluated from HP, energy, evolution depth, damage, and board position.
SPREAD_OR_BENCH_PRESSURE_IDS = {
    121,   # Dragapult ex
    648,   # Marnie's Grimmsnarl ex
    104,   # Froslass
    112,   # Munkidori
    133,   # Dusknoir
    190,   # Archaludon ex
}
SETUP_ENGINE_IDS = {
    65, 66, 119, 120, 173, 174, 235, 741, 742, 743, 1071,
}

# Publicly observed non-ex / effect-bypassing routes that can invalidate the
# Crustle wall.  These IDs are used as bounded tactical bonuses only; generic
# board/energy/HP evaluation remains active for unknown decks.
BYPASS_SETUP_IDS = {
    848,       # Buneary -> Mega Lopunny ex
    741, 742,  # Abra / Kadabra -> Alakazam
    708, 709,  # Chikorita / Bayleef -> Meganium
}
KNOWN_BYPASS_ATTACKER_IDS = {
    849,       # Mega Lopunny ex (Spiky Hopper bypass route)
    743, 245,  # Alakazam variants
    710,       # Meganium
}
BYPASS_READY_ENERGY = {849: 2, 743: 1, 245: 1, 710: 2}
MIRROR_LINE_IDS = {DWEBBLE, CRUSTLE}

# Exact 60-card, engine-valid deck.  See HEAL_SUPPORTER note above.
DECK = (
    [MEGA_KANGASKHAN_EX] * 4
    + [DWEBBLE] * 3
    + [CRUSTLE] * 3
    + [LILLIES_DETERMINATION] * 4
    + [BOSSS_ORDERS] * 4
    + [TEAM_ROCKETS_PETREL] * 4
    + [HILDA] * 2
    + [ERI] * 2
    + [XEROSICS_MACHINATIONS]
    + [HEAL_SUPPORTER]
    + [BIANCAS_DEVOTION]
    + [LISIAS_APPEAL]
    + [JUMBO_ICE_CREAM] * 4
    + [POKEGEAR_30] * 3
    + [BUDDY_BUDDY_POFFIN] * 2
    + [ULTRA_BALL]
    + [SWITCH]
    + [HAND_TRIMMER]
    + [HEROS_CAPE]
    + [HANDHELD_FAN]
    + [TEAM_ROCKETS_FACTORY]
    + [COMMUNITY_CENTER]
    + [FESTIVAL_GROUNDS]
    + [SPIKY_ENERGY] * 4
    + [GROW_GRASS_ENERGY] * 4
    + [MIST_ENERGY] * 4
    + [BASIC_GRASS_ENERGY]
)

# ---------------------------------------------------------------------------
# Generic raw-observation helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_index(values: Any, index: Any) -> Any:
    if not isinstance(index, int):
        return None
    values = _as_list(values)
    return values[index] if 0 <= index < len(values) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any) -> int:
    return _int(_field(card, "id", 0), 0)


def _hp(card: Any) -> int:
    return max(0, _int(_field(card, "hp", 0), 0))


def _max_hp(card: Any) -> int:
    return max(_hp(card), _int(_field(card, "maxHp", _hp(card)), _hp(card)))


def _damage(card: Any) -> int:
    return max(0, _max_hp(card) - _hp(card))


def _energy_cards(card: Any) -> list[Any]:
    return _as_list(_field(card, "energyCards", []))


def _energy_types(card: Any) -> list[Any]:
    return _as_list(_field(card, "energies", []))


def _energy_count(card: Any) -> int:
    # energies contains effective energy units, while energyCards contains cards.
    # Effective units are the better readiness signal.
    units = _energy_types(card)
    return len(units) if units else len(_energy_cards(card))


def _tools(card: Any) -> list[Any]:
    return _as_list(_field(card, "tools", []))


def _pre_evolutions(card: Any) -> list[Any]:
    return _as_list(_field(card, "preEvolution", []))


def _has_attached(card: Any, attached_id: int) -> bool:
    return any(_card_id(x) == attached_id for x in _energy_cards(card) + _tools(card))


def _players(obs: Any) -> list[Any]:
    return _as_list(_field(_field(obs, "current", {}), "players", []))


def _your_index(obs: Any) -> int:
    return _int(_field(_field(obs, "current", {}), "yourIndex", 0), 0)


def _player(obs: Any, player_index: int) -> Any:
    return _safe_index(_players(obs), player_index)


def _me(obs: Any) -> Any:
    return _player(obs, _your_index(obs))


def _opponent(obs: Any) -> Any:
    return _player(obs, 1 - _your_index(obs))


def _active(player: Any) -> Any:
    return _safe_index(_field(player, "active", []), 0)


def _bench(player: Any) -> list[Any]:
    return [x for x in _as_list(_field(player, "bench", [])) if x is not None]


def _board(player: Any) -> list[Any]:
    active = _active(player)
    return ([active] if active is not None else []) + _bench(player)


def _hand(player: Any) -> list[Any]:
    return _as_list(_field(player, "hand", []))


def _hand_count(player: Any) -> int:
    hand = _hand(player)
    return len(hand) if hand else max(0, _int(_field(player, "handCount", 0), 0))


def _deck_count(player: Any) -> int:
    return max(0, _int(_field(player, "deckCount", 0), 0))


def _prize_count(player: Any) -> int:
    return len(_as_list(_field(player, "prize", [])))


def _bench_slots(player: Any) -> int:
    return max(0, _int(_field(player, "benchMax", 5), 5) - len(_bench(player)))


def _statused(player: Any) -> bool:
    return any(bool(_field(player, key, False)) for key in ("asleep", "burned", "confused", "paralyzed", "poisoned"))


def _field_counts(player: Any) -> Counter[int]:
    return Counter(_card_id(x) for x in _board(player) if _card_id(x))


def _hand_counts(player: Any) -> Counter[int]:
    return Counter(_card_id(x) for x in _hand(player) if _card_id(x))


def _turn(obs: Any) -> int:
    return max(0, _int(_field(_field(obs, "current", {}), "turn", 0), 0))


def _turn_actions(obs: Any) -> int:
    return max(0, _int(_field(_field(obs, "current", {}), "turnActionCount", 0), 0))


def _stadium_id(obs: Any) -> int:
    stadium = _safe_index(_field(_field(obs, "current", {}), "stadium", []), 0)
    return _card_id(stadium)


def get_card(obs: Any, area: Any, index: Any, player_index: Any = None) -> Any:
    current = _field(obs, "current", {})
    select = _field(obs, "select", {})
    if not isinstance(player_index, int):
        player_index = _your_index(obs)
    player = _player(obs, player_index)

    if area == AREA_DECK:
        return _safe_index(_field(select, "deck", []), index)
    if area == AREA_HAND:
        return _safe_index(_field(player, "hand", []), index)
    if area == AREA_DISCARD:
        return _safe_index(_field(player, "discard", []), index)
    if area == AREA_ACTIVE:
        return _safe_index(_field(player, "active", []), index)
    if area == AREA_BENCH:
        return _safe_index(_field(player, "bench", []), index)
    if area == AREA_PRIZE:
        return _safe_index(_field(player, "prize", []), index)
    if area == AREA_STADIUM:
        return _safe_index(_field(current, "stadium", []), index)
    if area == AREA_LOOKING:
        return _safe_index(_field(current, "looking", []), index)
    return None


def _option_card(obs: Any, option: Any) -> Any:
    option_type = _int(_field(option, "type", -1), -1)
    owner = _field(option, "playerIndex", _your_index(obs))
    if not isinstance(owner, int):
        owner = _your_index(obs)

    if option_type == OPTION_PLAY:
        return get_card(obs, AREA_HAND, _field(option, "index"), _your_index(obs))
    if option_type in (OPTION_ATTACH, OPTION_EVOLVE):
        return get_card(obs, _field(option, "area", AREA_HAND), _field(option, "index"), owner)
    if option_type in (OPTION_CARD, OPTION_ABILITY, OPTION_DISCARD):
        return get_card(obs, _field(option, "area"), _field(option, "index"), owner)
    if option_type in (OPTION_TOOL_CARD, OPTION_ENERGY_CARD, OPTION_ENERGY):
        pokemon = get_card(obs, _field(option, "area"), _field(option, "index"), owner)
        if option_type == OPTION_TOOL_CARD:
            return _safe_index(_tools(pokemon), _field(option, "toolIndex"))
        return _safe_index(_energy_cards(pokemon), _field(option, "energyIndex"))
    return None


def _option_target(obs: Any, option: Any) -> Any:
    owner = _field(option, "playerIndex", _your_index(obs))
    if not isinstance(owner, int):
        owner = _your_index(obs)
    in_play_area = _field(option, "inPlayArea")
    in_play_index = _field(option, "inPlayIndex")
    if in_play_area is not None:
        return get_card(obs, in_play_area, in_play_index, owner)
    return None


def _effect_id(select: Any) -> int:
    return _card_id(_field(select, "effect")) or _card_id(_field(select, "contextCard"))


def _is_our_pokemon(card: Any) -> bool:
    return _card_id(card) in POKEMON_IDS


def _is_energy(card: Any) -> bool:
    return _card_id(card) in ENERGY_IDS


def _remaining_hp_ratio(card: Any) -> float:
    return _hp(card) / max(1, _max_hp(card))


def _prize_proxy(card: Any) -> int:
    cid = _card_id(card)
    max_hp = _max_hp(card)
    if cid == MEGA_KANGASKHAN_EX or max_hp >= 380:
        return 3
    if max_hp >= 190:
        return 2
    return 1


def _evolution_depth(card: Any) -> int:
    return len(_pre_evolutions(card))


def _is_spread_threat(card: Any) -> bool:
    return _card_id(card) in SPREAD_OR_BENCH_PRESSURE_IDS


def _target_value(card: Any, active: bool = False) -> float:
    if card is None:
        return -10000.0
    cid = _card_id(card)
    hp = _hp(card)
    max_hp = _max_hp(card)
    energies = _energy_count(card)
    score = 500.0
    score += _prize_proxy(card) * 900
    score += energies * 420
    score += _evolution_depth(card) * 260
    score += len(_tools(card)) * 170
    score += max(0, max_hp - hp) * 1.6
    if active:
        score += 250
    if _is_spread_threat(card):
        score += 2500
    if cid in SETUP_ENGINE_IDS and energies == 0:
        score += 450

    # Kill bypass engines before they mature.  These bonuses are intentionally
    # bounded so an immediate low-HP prize still wins the comparison.
    if cid == 848:      # Buneary
        score += 4800
    elif cid == 849:    # Mega Lopunny ex
        score += 6800
    elif cid == 741:    # Abra
        score += 4300
    elif cid == 742:    # Kadabra
        score += 5700
    elif cid in (743, 245):
        score += 6900
    elif cid == 708:    # Chikorita
        score += 3900
    elif cid == 709:    # Bayleef
        score += 5200
    elif cid == 710:    # Meganium
        score += 6600
    elif cid == DWEBBLE:
        score += 4200
    elif cid == MEGA_KANGASKHAN_EX:
        score += 2700

    if hp <= 120:
        score += 4300
    elif hp <= 180:
        score += 1800
    return score


def _attacker_value(card: Any) -> float:
    if card is None:
        return -10000.0
    cid = _card_id(card)
    score = _energy_count(card) * 900 + _hp(card) * 2.0
    if cid == CRUSTLE:
        score += 5300
    elif cid == MEGA_KANGASKHAN_EX:
        score += 3300
    elif cid == DWEBBLE:
        score += 1500
    if _has_attached(card, HEROS_CAPE):
        score += 900
    return score


def _spread_risk(obs: Any) -> bool:
    opponent = _opponent(obs)
    for pokemon in _board(opponent):
        if _is_spread_threat(pokemon) and (_energy_count(pokemon) >= 1 or pokemon is _active(opponent)):
            return True
    # Recent bench damage is also a generic warning even for an unknown card.
    active_serial = _int(_field(_active(_me(obs)), "serial", -1), -1)
    for log in _as_list(_field(obs, "logs", [])):
        if _int(_field(log, "type", -1), -1) == 16 and bool(_field(log, "putDamageCounter", False)):
            serial = _int(_field(log, "serial", -2), -2)
            if serial != active_serial and _int(_field(log, "playerIndex", -1), -1) == _your_index(obs):
                return True
    return False


def _gust_risk(obs: Any) -> bool:
    me = _me(obs)
    opponent = _opponent(obs)
    valuable_bench = any(_card_id(p) == MEGA_KANGASKHAN_EX for p in _bench(me))
    return valuable_bench and (_hand_count(opponent) >= 4 or _turn(obs) >= 5)


def _crustle_ready_on_bench(obs: Any) -> bool:
    return any(_card_id(p) == CRUSTLE and _energy_count(p) >= 2 for p in _bench(_me(obs)))


def _wall_established(obs: Any) -> bool:
    return _card_id(_active(_me(obs))) == CRUSTLE


def _missing_setup(obs: Any) -> bool:
    counts = _field_counts(_me(obs))
    return counts[DWEBBLE] + counts[CRUSTLE] == 0 or counts[CRUSTLE] == 0


def _good_supporter_in_hand(obs: Any) -> bool:
    hand_ids = set(_hand_counts(_me(obs)))
    return bool(hand_ids & SUPPORTER_IDS)


def _best_opponent_bench(obs: Any) -> Any:
    bench = _bench(_opponent(obs))
    return max(bench, key=lambda p: _target_value(p, False), default=None)


def _opponent_ids(obs: Any) -> set[int]:
    ids = {_card_id(p) for p in _board(_opponent(obs))}
    ids.update(_card_id(c) for c in _as_list(_field(_opponent(obs), "discard", [])))
    return {cid for cid in ids if cid}


def _is_mirror(obs: Any) -> bool:
    return bool(_opponent_ids(obs) & MIRROR_LINE_IDS)


def _known_bypass_ready(card: Any) -> bool:
    cid = _card_id(card)
    need = BYPASS_READY_ENERGY.get(cid)
    return need is not None and _energy_count(card) >= need


def _generic_non_ex_bypass_ready(card: Any) -> bool:
    if card is None or _card_id(card) in MIRROR_LINE_IDS:
        return False
    # Unknown low-HP evolved attackers with energy are treated as potential wall
    # bypasses.  Basics need two energy to avoid reacting to harmless pivots.
    max_hp = _max_hp(card)
    energy = _energy_count(card)
    depth = _evolution_depth(card)
    return max_hp < 190 and ((depth >= 1 and energy >= 1) or (max_hp >= 100 and energy >= 2))


def _opponent_active_bypass(obs: Any) -> bool:
    active = _active(_opponent(obs))
    return bool(active is not None and (
        _is_spread_threat(active)
        or _known_bypass_ready(active)
        or _generic_non_ex_bypass_ready(active)
    ))


def _opponent_has_bypass_pressure(obs: Any) -> bool:
    ids = _opponent_ids(obs)
    if ids & BYPASS_SETUP_IDS:
        return True
    for p in _board(_opponent(obs)):
        if _is_spread_threat(p) or _known_bypass_ready(p) or _generic_non_ex_bypass_ready(p):
            return True
    return False


def _kang_ready_on_bench(obs: Any) -> bool:
    return any(_card_id(p) == MEGA_KANGASKHAN_EX and _energy_count(p) >= 3 for p in _bench(_me(obs)))


def _wall_with_backup(obs: Any) -> bool:
    if not _wall_established(obs):
        return False
    counts = _field_counts(_me(obs))
    return counts[DWEBBLE] + counts[CRUSTLE] >= 2


def _mirror_draw_lock(obs: Any) -> bool:
    if not (_is_mirror(obs) and _wall_with_backup(obs)):
        return False
    if _opponent_active_bypass(obs):
        return False
    return _deck_count(_me(obs)) <= _deck_count(_opponent(obs)) + 2


def _opponent_alakazam_pressure(obs: Any) -> bool:
    return bool(_opponent_ids(obs) & {741, 742, 743, 245})


def _attack_damage_budget(obs: Any) -> int:
    select = _field(obs, "select", {})
    if not _has_attack_option(select):
        return 0
    active_id = _card_id(_active(_me(obs)))
    if active_id == CRUSTLE:
        return 120
    if active_id == MEGA_KANGASKHAN_EX:
        return 200
    return 0


def _boss_conversion_value(obs: Any, target: Any) -> float:
    if target is None:
        return -10000.0
    damage = _attack_damage_budget(obs)
    if damage <= 0:
        return -9000.0
    value = _target_value(target)
    if _hp(target) <= damage:
        value += 9800 + _prize_proxy(target) * 1600
    elif _card_id(target) in BYPASS_SETUP_IDS | KNOWN_BYPASS_ATTACKER_IDS | {DWEBBLE, MEGA_KANGASKHAN_EX}:
        value += 2600
    else:
        value -= 2400
    return value


def _best_boss_target(obs: Any) -> Any:
    return max(_bench(_opponent(obs)), key=lambda p: _boss_conversion_value(obs, p), default=None)


def _has_attack_option(select: Any, attack_id: int | None = None) -> bool:
    for option in _as_list(_field(select, "option", [])):
        if _int(_field(option, "type", -1), -1) != OPTION_ATTACK:
            continue
        if attack_id is None or _int(_field(option, "attackId", -1), -1) == attack_id:
            return True
    return False


def _has_option_type(select: Any, option_type: int) -> bool:
    return any(_int(_field(o, "type", -1), -1) == option_type for o in _as_list(_field(select, "option", [])))


# ---------------------------------------------------------------------------
# Action scoring
# ---------------------------------------------------------------------------


def _energy_target_score(obs: Any, target: Any, energy_id: int) -> float:
    if target is None:
        return -20000.0
    cid = _card_id(target)
    units = _energy_count(target)
    active = target is _active(_me(obs))
    spread = _spread_risk(obs)
    bypass = _opponent_has_bypass_pressure(obs)
    mirror = _is_mirror(obs)
    score = 0.0

    if cid == DWEBBLE:
        # One Energy is enough for Ascension.  Additional Energy belongs on the
        # evolved wall or on a real bypass attacker.
        score = 13000 if units == 0 else 850 if units == 1 else -4200
        if active and units == 0:
            score += 2800
        if _field_counts(_me(obs))[CRUSTLE] == 0:
            score += 1600
    elif cid == CRUSTLE:
        if units == 0:
            score = 14300
        elif units == 1:
            score = 15100
        elif units == 2:
            wants_jumbo = _hand_counts(_me(obs))[JUMBO_ICE_CREAM] > 0 or _damage(target) >= 50
            score = 9300 if wants_jumbo else 2500
        else:
            score = -1600 - (units - 3) * 3200
        if active:
            score += 1700
        if bypass and units < 2:
            score += 800
    elif cid == MEGA_KANGASKHAN_EX:
        if mirror:
            score = -5200
        else:
            score = (7300, 8800, 10800, 400)[min(units, 3)]
            if bypass:
                score += 4700
            if not _wall_established(obs):
                score += 800
            if _wall_established(obs) and not bypass:
                score -= 2500
            if spread or _gust_risk(obs):
                score -= 2600
            if active:
                score += 1000
    else:
        score = 1000

    if energy_id == SPIKY_ENERGY:
        score += 1000 if active and cid == CRUSTLE else 250
    elif energy_id == MIST_ENERGY:
        score += 900 if active or spread else 250
    elif energy_id == GROW_GRASS_ENERGY:
        score += 650 if cid in (DWEBBLE, CRUSTLE) else 100
    elif energy_id == BASIC_GRASS_ENERGY:
        score += 100

    return score


def _tool_target_score(obs: Any, target: Any, tool_id: int) -> float:
    if target is None or _tools(target):
        return -12000.0
    cid = _card_id(target)
    active = target is _active(_me(obs))
    if tool_id == HEROS_CAPE:
        if cid == CRUSTLE:
            return 16100 + (1800 if active else 0)
        if cid == DWEBBLE and active and _field_counts(_me(obs))[CRUSTLE] == 0:
            return 13300
        if cid == MEGA_KANGASKHAN_EX:
            score = 9400 + (1000 if active else 0)
            if _spread_risk(obs) or _gust_risk(obs):
                score -= 3000
            return score
    if tool_id == HANDHELD_FAN:
        if cid == CRUSTLE and active:
            return 11600
        if cid == MEGA_KANGASKHAN_EX and active:
            return 9000
        return 4200
    return 1000


def _score_attach(obs: Any, option: Any) -> float:
    card = _option_card(obs, option)
    target = _option_target(obs, option)
    cid = _card_id(card)
    if cid in ENERGY_IDS:
        return _energy_target_score(obs, target, cid)
    if cid in TOOL_IDS:
        return _tool_target_score(obs, target, cid)
    return 1000


def _score_evolve(obs: Any, option: Any) -> float:
    card = _option_card(obs, option)
    target = _option_target(obs, option)
    if _card_id(card) == CRUSTLE and _card_id(target) == DWEBBLE:
        score = 17200
        if target is _active(_me(obs)):
            score += 2700
        score += _energy_count(target) * 500
        if _spread_risk(obs) or _opponent_has_bypass_pressure(obs):
            score += 700
        return score
    return 2500


def _supporter_score(obs: Any, cid: int) -> float:
    me = _me(obs)
    opponent = _opponent(obs)
    hand = _hand_count(me)
    op_hand = _hand_count(opponent)
    deck = _deck_count(me)
    turn = _turn(obs)
    active = _active(me)
    counts = _field_counts(me)
    missing = _missing_setup(obs)
    spread = _spread_risk(obs)
    gust = _gust_risk(obs)
    mirror_lock = _mirror_draw_lock(obs)
    alakazam = _opponent_alakazam_pressure(obs)

    if cid == LILLIES_DETERMINATION:
        if deck <= 6 or mirror_lock:
            return -9800
        score = 10600 + max(0, 5 - hand) * 1300
        if missing:
            score += 2200
        if hand >= 8:
            score -= 5300
        return score

    if cid == BOSSS_ORDERS:
        target = _best_boss_target(obs)
        if target is None or _attack_damage_budget(obs) <= 0:
            return -8800
        value = _boss_conversion_value(obs, target)
        score = 3600 + value
        if _is_mirror(obs):
            if _card_id(target) == DWEBBLE:
                score += 5200
            elif _card_id(target) == MEGA_KANGASKHAN_EX:
                score += 3400
            elif _card_id(target) == CRUSTLE:
                score -= 4200
        # Avoid wasting gust when the current active is already at least as good.
        current = _active(opponent)
        current_value = _target_value(current, True)
        if _hp(current) <= _attack_damage_budget(obs):
            current_value += 9000
        if current_value >= value + 2200:
            score -= 6500
        return score

    if cid == TEAM_ROCKETS_PETREL:
        if mirror_lock:
            return -8600
        score = 8900
        if turn <= 4:
            score += 2300
        if missing:
            score += 2400
        if hand <= 3:
            score += 900
        if deck <= 8:
            score -= 7000
        return score

    if cid == HILDA:
        if mirror_lock and not missing:
            return -8200
        score = 8300
        if missing or counts[DWEBBLE] == 0:
            score += 3100
        if not any(_card_id(c) in ENERGY_IDS for c in _hand(me)):
            score += 1700
        if turn <= 4:
            score += 1000
        if deck <= 8:
            score -= 6500
        return score

    if cid == ERI:
        score = 3300 + max(0, op_hand - 3) * 1050
        if gust or spread:
            score += 1800
        if alakazam and op_hand >= 5:
            score += 2600
        if op_hand < 4:
            score -= 3800
        return score

    if cid == XEROSICS_MACHINATIONS:
        score = 2900 + max(0, op_hand - 4) * 1150
        if op_hand >= 7:
            score += 2600
        if alakazam and op_hand >= 5:
            score += 14300
        if op_hand <= 3:
            score -= 5200
        return score

    if cid == HEAL_SUPPORTER:
        damage = _damage(active)
        score = damage * 45
        if _card_id(active) == CRUSTLE:
            score += 3200
        if damage < 40:
            score -= 6500
        return score

    if cid == BIANCAS_DEVOTION:
        damage = _damage(active)
        ratio = 1.0 - _remaining_hp_ratio(active) if active is not None else 0.0
        score = damage * 38 + ratio * 5000
        if _card_id(active) in (CRUSTLE, MEGA_KANGASKHAN_EX):
            score += 1800
        if damage < 60:
            score -= 5200
        return score

    if cid == LISIAS_APPEAL:
        target = _best_boss_target(obs)
        if target is None or _attack_damage_budget(obs) <= 0:
            return -7200
        score = 3400 + _boss_conversion_value(obs, target)
        if target is not None and (_is_spread_threat(target) or _energy_count(target) >= 2):
            score += 1800
        return score

    return 2000

def _score_play(obs: Any, option: Any) -> float:
    card = _option_card(obs, option)
    cid = _card_id(card)
    me = _me(obs)
    counts = _field_counts(me)
    hand = _hand_count(me)
    deck = _deck_count(me)
    active = _active(me)
    turn = _turn(obs)
    spread = _spread_risk(obs)
    gust = _gust_risk(obs)
    missing = _missing_setup(obs)

    if cid == DWEBBLE:
        if _bench_slots(me) <= 0:
            return -9000
        return 12700 if counts[DWEBBLE] + counts[CRUSTLE] == 0 else 9800 if counts[DWEBBLE] + counts[CRUSTLE] == 1 else 2600

    if cid == MEGA_KANGASKHAN_EX:
        if _bench_slots(me) <= 0:
            return -9000
        if _is_mirror(obs) and counts[DWEBBLE] + counts[CRUSTLE] > 0:
            return -11200
        field_kang = counts[MEGA_KANGASKHAN_EX]
        if field_kang >= 1:
            return -9000 if (spread or gust or turn <= 6) else 700
        score = 7000
        if counts[DWEBBLE] + counts[CRUSTLE] == 0:
            score += 2600
        if _wall_established(obs):
            score -= 2000
        if spread or gust:
            score -= 3600
        if _opponent_has_bypass_pressure(obs):
            score += 3800
        return score

    if cid in SUPPORTER_IDS:
        return _supporter_score(obs, cid)

    if cid == JUMBO_ICE_CREAM:
        damage = _damage(active)
        score = damage * 52
        if _energy_count(active) >= 3:
            score += 4300
        else:
            score -= 6500
        if _card_id(active) == CRUSTLE:
            score += 1700
        if damage < 50:
            score -= 6000
        return score

    if cid == POKEGEAR_30:
        if deck <= 7 or _mirror_draw_lock(obs):
            return -8800
        score = 6100
        if not _good_supporter_in_hand(obs):
            score += 3900
        if missing:
            score += 1200
        if hand >= 8:
            score -= 2200
        return score

    if cid == BUDDY_BUDDY_POFFIN:
        if deck <= 7 or _bench_slots(me) <= 0:
            return -7500
        return 13100 if counts[DWEBBLE] + counts[CRUSTLE] == 0 else 9000 if counts[DWEBBLE] + counts[CRUSTLE] == 1 else 700

    if cid == ULTRA_BALL:
        if deck <= 7 or hand < 3:
            return -7200
        score = 7000
        if counts[DWEBBLE] == 0 and counts[CRUSTLE] == 0:
            score += 4200
        elif counts[DWEBBLE] > 0 and counts[CRUSTLE] == 0:
            score += 5000
        elif counts[MEGA_KANGASKHAN_EX] == 0 and _opponent_has_bypass_pressure(obs):
            score += 1300
        return score

    if cid == SWITCH:
        if active is None:
            return -8000
        score = -1200
        if _card_id(active) == MEGA_KANGASKHAN_EX and _crustle_ready_on_bench(obs):
            score += 12900
        if _card_id(active) == CRUSTLE and _opponent_active_bypass(obs) and _kang_ready_on_bench(obs):
            score += 17100
        if _statused(me):
            score += 7300
        if _remaining_hp_ratio(active) <= 0.35 and any(_attacker_value(p) > _attacker_value(active) for p in _bench(me)):
            score += 5800
        if _card_id(active) == CRUSTLE and not _opponent_active_bypass(obs):
            score -= 6900
        return score

    if cid == HAND_TRIMMER:
        op_hand = _hand_count(_opponent(obs))
        if op_hand < 6:
            return -6200
        score = 2500 + max(0, op_hand - 5) * 1650
        if _opponent_alakazam_pressure(obs):
            score += 17500
        return score

    if cid == HEROS_CAPE:
        if active is None or _tools(active):
            return -8000
        return _tool_target_score(obs, active, cid)

    if cid == HANDHELD_FAN:
        if active is None or _tools(active):
            return -6500
        opponent_active = _active(_opponent(obs))
        score = 3300 + _energy_count(opponent_active) * 1200
        if _card_id(active) == CRUSTLE:
            score += 1800
        return score

    if cid in STADIUM_IDS:
        current = _stadium_id(obs)
        if current == cid:
            return -8500
        score = 4300
        if current and current not in STADIUM_IDS:
            score += 2800
        if cid == TEAM_ROCKETS_FACTORY:
            score += 1900 if turn <= 7 and hand <= 6 else 300
        elif cid == COMMUNITY_CENTER:
            score += min(4000, sum(_damage(p) for p in _board(me)) * 16)
        elif cid == FESTIVAL_GROUNDS:
            score += 2800 if _statused(me) else 300
        return score

    return 800


def _score_ability(obs: Any, option: Any) -> float:
    card = _option_card(obs, option)
    cid = _card_id(card)
    if cid == MEGA_KANGASKHAN_EX:
        if _deck_count(_me(obs)) <= 6 or _mirror_draw_lock(obs):
            return -9800
        score = 6500 + max(0, 6 - _hand_count(_me(obs))) * 850
        if _missing_setup(obs):
            score += 1000
        # Immediate attacks and critical evolution should remain ahead of pure draw.
        if _has_attack_option(_field(obs, "select", {})):
            score -= 1800
        return score
    return 3000


def _score_attack(obs: Any, option: Any) -> float:
    attack_id = _int(_field(option, "attackId", -1), -1)
    me = _me(obs)
    opponent = _opponent(obs)
    active = _active(me)
    target = _active(opponent)
    target_hp = _hp(target)
    target_value = _target_value(target, True)

    if attack_id == ASCENSION or _card_id(active) == DWEBBLE:
        if _field_counts(me)[CRUSTLE] == 0:
            return 23500
        return 8300

    if attack_id == CRUSTLE_ATTACK or _card_id(active) == CRUSTLE:
        score = 15800 + target_value * 0.42
        if target_hp and target_hp <= 120:
            score += 9800 + _prize_proxy(target) * 1500
        if _is_spread_threat(target) or _card_id(target) in BYPASS_SETUP_IDS | KNOWN_BYPASS_ATTACKER_IDS:
            score += 2400
        return score

    if attack_id == RAPID_FIRE_COMBO or _card_id(active) == MEGA_KANGASKHAN_EX:
        if _card_id(target) == CRUSTLE:
            return -16000
        score = 14900 + target_value * 0.35
        # Conservative expected-damage threshold; coin upside remains a bonus.
        if target_hp and target_hp <= 200:
            score += 6500 + _prize_proxy(target) * 1300
        if _spread_risk(obs) and _crustle_ready_on_bench(obs):
            score -= 1900
        return score

    score = 11900 + target_value * 0.3
    if target_hp <= 120:
        score += 5000
    return score


def _score_retreat(obs: Any) -> float:
    me = _me(obs)
    active = _active(me)
    if active is None:
        return -10000
    score = -6500.0
    cid = _card_id(active)
    bench = _bench(me)
    best_bench = max((_attacker_value(p) for p in bench), default=-10000)

    if cid == MEGA_KANGASKHAN_EX and _crustle_ready_on_bench(obs):
        score += 17300
    if cid == CRUSTLE and _opponent_active_bypass(obs) and _kang_ready_on_bench(obs):
        score += 19600
    if _statused(me):
        score += 9300
    if _remaining_hp_ratio(active) <= 0.30 and best_bench > _attacker_value(active) + 800:
        score += 7200
    if cid == DWEBBLE and not _has_attack_option(_field(obs, "select", {}), ASCENSION) and _crustle_ready_on_bench(obs):
        score += 9800
    if cid == CRUSTLE and not _opponent_active_bypass(obs):
        score -= 8200
    return score


def _score_main_option(obs: Any, option: Any) -> float:
    option_type = _int(_field(option, "type", -1), -1)
    if option_type == OPTION_EVOLVE:
        return _score_evolve(obs, option)
    if option_type == OPTION_ATTACH:
        return _score_attach(obs, option)
    if option_type == OPTION_PLAY:
        return _score_play(obs, option)
    if option_type == OPTION_ABILITY:
        return _score_ability(obs, option)
    if option_type == OPTION_ATTACK:
        return _score_attack(obs, option)
    if option_type == OPTION_RETREAT:
        return _score_retreat(obs)
    if option_type == OPTION_END:
        return 0.0
    if option_type == OPTION_DISCARD:
        return 800.0
    return 100.0


# ---------------------------------------------------------------------------
# Subselection scoring
# ---------------------------------------------------------------------------


def _setup_score(obs: Any, card: Any, context: int, occurrence: int) -> float:
    cid = _card_id(card)
    if context == CONTEXT_SETUP_ACTIVE:
        if cid == DWEBBLE:
            return 12000 - occurrence * 100
        if cid == MEGA_KANGASKHAN_EX:
            return 5900 - occurrence * 100
        return 100
    if cid == DWEBBLE:
        return 10800 - occurrence * 2400
    if cid == MEGA_KANGASKHAN_EX:
        return 5000 if occurrence == 0 else -8000
    return 100


def _to_active_score(obs: Any, card: Any) -> float:
    cid = _card_id(card)
    score = _attacker_value(card)
    opponent_active = _active(_opponent(obs))
    if cid == CRUSTLE:
        score += 4800
        if _prize_proxy(opponent_active) >= 2:
            score += 2600
        if _opponent_active_bypass(obs):
            score -= 9200
    elif cid == MEGA_KANGASKHAN_EX:
        if _card_id(opponent_active) == CRUSTLE:
            score -= 9500
        if _opponent_active_bypass(obs) and _energy_count(card) >= 3:
            score += 12800
        elif _energy_count(card) < 3:
            score -= 5200
        if _spread_risk(obs):
            score -= 1800
    elif cid == DWEBBLE and _field_counts(_me(obs))[CRUSTLE] == 0:
        score += 1900
    return score


def _to_hand_score(obs: Any, card: Any, effect_id: int) -> float:
    cid = _card_id(card)
    counts = _field_counts(_me(obs))
    hand_counts = _hand_counts(_me(obs))
    score = 1000.0

    if cid == CRUSTLE:
        score = 12600 if counts[DWEBBLE] > 0 and counts[CRUSTLE] == 0 else 7200
    elif cid == DWEBBLE:
        score = 11900 if counts[DWEBBLE] + counts[CRUSTLE] == 0 else 6500
    elif cid == MEGA_KANGASKHAN_EX:
        if _is_mirror(obs):
            score = -5200
        else:
            score = 7800 if counts[cid] == 0 and _opponent_has_bypass_pressure(obs) else 1800
    elif cid in ENERGY_IDS:
        score = 8500 if not any(_card_id(x) in ENERGY_IDS for x in _hand(_me(obs))) else 3900
        if cid == MIST_ENERGY and _spread_risk(obs):
            score += 1700
    elif cid == LILLIES_DETERMINATION:
        score = -6500 if _mirror_draw_lock(obs) else (9400 if _hand_count(_me(obs)) <= 4 else 5100)
    elif cid == BOSSS_ORDERS:
        target = _best_boss_target(obs)
        score = 9200 if target is not None and _boss_conversion_value(obs, target) > 9000 else 4300
        if _is_mirror(obs) and target is not None and _card_id(target) in (DWEBBLE, MEGA_KANGASKHAN_EX):
            score += 3200
    elif cid == TEAM_ROCKETS_PETREL:
        score = -6200 if _mirror_draw_lock(obs) else (9000 if _missing_setup(obs) else 4300)
    elif cid == HILDA:
        score = -5600 if _mirror_draw_lock(obs) and not _missing_setup(obs) else (8800 if _missing_setup(obs) else 4700)
    elif cid == ERI:
        score = 7000 if _hand_count(_opponent(obs)) >= 5 else 2500
    elif cid == XEROSICS_MACHINATIONS:
        score = 7200 if _hand_count(_opponent(obs)) >= 6 else 2300
    elif cid == JUMBO_ICE_CREAM:
        score = 6600 if _damage(_active(_me(obs))) >= 60 else 1800
    elif cid == BUDDY_BUDDY_POFFIN:
        score = 9800 if counts[DWEBBLE] + counts[CRUSTLE] == 0 else 3400
    elif cid == ULTRA_BALL:
        score = 8000 if _missing_setup(obs) else 3600
    elif cid == SWITCH:
        score = 2500
        if _card_id(_active(_me(obs))) == MEGA_KANGASKHAN_EX and _crustle_ready_on_bench(obs):
            score = 6800
        if _card_id(_active(_me(obs))) == CRUSTLE and _opponent_active_bypass(obs) and _kang_ready_on_bench(obs):
            score = 9800
    elif cid == HEROS_CAPE:
        score = 8300 if not any(_tools(p) for p in _board(_me(obs))) else 2100
    elif cid in STADIUM_IDS:
        score = 4200

    if hand_counts[cid] >= 2 and cid not in ENERGY_IDS:
        score -= 1000
    if effect_id == BUDDY_BUDDY_POFFIN and cid != DWEBBLE:
        score -= 5000
    return score


def _discard_score(obs: Any, card: Any) -> float:
    """Higher means safer to discard."""
    cid = _card_id(card)
    hand_counts = _hand_counts(_me(obs))
    counts = _field_counts(_me(obs))

    if cid == MEGA_KANGASKHAN_EX:
        if _is_mirror(obs) and counts[DWEBBLE] + counts[CRUSTLE] > 0:
            return 12600
        return 10800 if hand_counts[cid] >= 2 or counts[cid] >= 1 else 2500
    if cid == DWEBBLE:
        return -10500 if counts[DWEBBLE] + counts[CRUSTLE] == 0 else -3500
    if cid == CRUSTLE:
        return -11500 if counts[DWEBBLE] > 0 and counts[CRUSTLE] == 0 else -3000
    if cid in ENERGY_IDS:
        total_energy = sum(hand_counts[x] for x in ENERGY_IDS)
        return 3200 if total_energy >= 3 else -6500
    if cid == HEROS_CAPE:
        return -10000
    if cid == SWITCH:
        return -6500 if _gust_risk(obs) or _statused(_me(obs)) else 2200
    if cid in STADIUM_IDS:
        return 6500 if _stadium_id(obs) in STADIUM_IDS else 2500
    if cid in SUPPORTER_IDS:
        if _mirror_draw_lock(obs) and cid in (LILLIES_DETERMINATION, TEAM_ROCKETS_PETREL, HILDA):
            return 9800
        if cid == BOSSS_ORDERS and _is_mirror(obs):
            return -4300
        return 6200 if hand_counts[cid] >= 2 else 900
    if cid in (POKEGEAR_30, BUDDY_BUDDY_POFFIN) and _turn(obs) >= 6:
        return 5600
    if cid == JUMBO_ICE_CREAM and _damage(_active(_me(obs))) < 40:
        return 5000
    return 1800


def _attached_discard_score(obs: Any, option: Any) -> float:
    owner = _field(option, "playerIndex", _your_index(obs))
    pokemon = get_card(obs, _field(option, "area"), _field(option, "index"), owner)
    card = _option_card(obs, option)
    score = 1000.0
    if _int(owner, _your_index(obs)) == _your_index(obs):
        # Preserve energy on the active wall and developed attackers.
        if _card_id(pokemon) in (CRUSTLE, MEGA_KANGASKHAN_EX):
            score -= 4500
        if pokemon is _active(_me(obs)):
            score -= 2500
        if _card_id(card) == MIST_ENERGY and _spread_risk(obs):
            score -= 1900
    else:
        score += _energy_count(pokemon) * 1300 + _target_value(pokemon) * 0.25
    return score


def _score_card_selection(obs: Any, option: Any, context: int, occurrence: int) -> float:
    card = _option_card(obs, option)
    cid = _card_id(card)
    owner = _int(_field(option, "playerIndex", _your_index(obs)), _your_index(obs))
    effect_id = _effect_id(_field(obs, "select", {}))

    if context in (CONTEXT_SETUP_ACTIVE, CONTEXT_SETUP_BENCH):
        return _setup_score(obs, card, context, occurrence)

    if context == CONTEXT_SWITCH:
        if owner != _your_index(obs):
            if effect_id in (BOSSS_ORDERS, LISIAS_APPEAL):
                score = _boss_conversion_value(obs, card)
            else:
                score = _target_value(card, False)
            if _is_spread_threat(card):
                score += 3200
            if _hp(card) <= 120:
                score += 4300
            if _is_mirror(obs) and cid == DWEBBLE:
                score += 3600
            return score
        return _to_active_score(obs, card)

    if context == CONTEXT_TO_ACTIVE:
        return _to_active_score(obs, card)

    if context in (CONTEXT_TO_BENCH, CONTEXT_TO_FIELD):
        if owner != _your_index(obs):
            return _target_value(card)
        if cid == DWEBBLE:
            return 11000 - occurrence * 2300
        if cid == MEGA_KANGASKHAN_EX:
            if _is_mirror(obs):
                return -10500
            return 5100 if _field_counts(_me(obs))[cid] == 0 and occurrence == 0 else -8500
        return 500

    if context == CONTEXT_TO_HAND:
        return _to_hand_score(obs, card, effect_id)

    if context in (CONTEXT_EVOLVES_FROM, CONTEXT_EVOLVES_TO, CONTEXT_EVOLVE):
        if cid == DWEBBLE:
            score = 11800 + _energy_count(card) * 800
            if card is _active(_me(obs)):
                score += 1800
            return score
        if cid == CRUSTLE:
            return 12500
        return 800

    if context == CONTEXT_ATTACH_FROM:
        # Some effects ask for a Pokémon target through a CARD option.
        return _energy_target_score(obs, card, effect_id if effect_id in ENERGY_IDS else BASIC_GRASS_ENERGY)

    if context == CONTEXT_ATTACH_TO:
        if owner != _your_index(obs):
            return _target_value(card)
        if effect_id in TOOL_IDS:
            return _tool_target_score(obs, card, effect_id)
        return _energy_target_score(obs, card, effect_id if effect_id in ENERGY_IDS else BASIC_GRASS_ENERGY)

    if context == CONTEXT_DETACH_FROM:
        if owner != _your_index(obs):
            return _target_value(card) + _energy_count(card) * 900
        return -_attacker_value(card)

    if context in (CONTEXT_HEAL, CONTEXT_REMOVE_DAMAGE_COUNTER):
        score = _damage(card) * 55
        if card is _active(_me(obs)):
            score += 3300
        if cid == CRUSTLE:
            score += 1700
        return score

    if context in (CONTEXT_DAMAGE, CONTEXT_DAMAGE_COUNTER, CONTEXT_DAMAGE_COUNTER_ANY, CONTEXT_EFFECT_TARGET):
        if owner != _your_index(obs):
            return _target_value(card, _field(option, "area") == AREA_ACTIVE)
        # Harmful effects on our own side: sacrifice the least valuable body.
        return -_attacker_value(card)

    if context in (CONTEXT_DISCARD, CONTEXT_DISCARD_CARD_OR_ATTACHED):
        return _discard_score(obs, card)

    if context in (CONTEXT_TO_DECK, CONTEXT_TO_DECK_BOTTOM, CONTEXT_NOT_MOVE):
        # Preserve critical resources by putting expendable cards back first.
        return _discard_score(obs, card)

    if context == CONTEXT_TO_PRIZE:
        return -_to_hand_score(obs, card, effect_id)

    if context == CONTEXT_LOOK:
        return _to_hand_score(obs, card, effect_id)

    if owner != _your_index(obs):
        return _target_value(card, _field(option, "area") == AREA_ACTIVE)
    return _to_hand_score(obs, card, effect_id)


def _score_number(obs: Any, option: Any, context: int) -> float:
    number = _int(_field(option, "number", 0), 0)
    if context == CONTEXT_DRAW_COUNT:
        reserve = 8 if _is_mirror(obs) else 4
        safe_draw = max(0, _deck_count(_me(obs)) - reserve)
        preferred = min(number, safe_draw)
        if _mirror_draw_lock(obs):
            preferred = 0
        return 10000 - abs(number - preferred) * 2400 + preferred * 100
    if context in (CONTEXT_DAMAGE_COUNTER_COUNT, CONTEXT_REMOVE_DAMAGE_COUNTER_COUNT):
        return number * 1000
    return number * 100


def _score_yes_no(obs: Any, option: Any, context: int) -> float:
    option_type = _int(_field(option, "type", -1), -1)
    yes = option_type == OPTION_YES
    if context == CONTEXT_IS_FIRST:
        # Going second enables turn-one Ascension and is the deck's preferred plan.
        return 10000 if not yes else -10000
    if context == CONTEXT_MULLIGAN:
        return 8000 if yes else -8000
    if context == CONTEXT_ACTIVATE:
        effect_id = _effect_id(_field(obs, "select", {}))
        if effect_id == MEGA_KANGASKHAN_EX and (_deck_count(_me(obs)) <= 6 or _mirror_draw_lock(obs)):
            return 9000 if not yes else -9000
        return 8000 if yes else -3000
    if context in (CONTEXT_FIRST_EFFECT, CONTEXT_COIN_HEAD):
        return 5000 if yes else 0
    if context == CONTEXT_MORE_DEVOLVE:
        return 4000 if yes else 0
    return 1000 if yes else 0


def _score_sub_option(obs: Any, option: Any, context: int, occurrence: int) -> float:
    option_type = _int(_field(option, "type", -1), -1)
    if option_type == OPTION_CARD:
        return _score_card_selection(obs, option, context, occurrence)
    if option_type in (OPTION_TOOL_CARD, OPTION_ENERGY_CARD, OPTION_ENERGY):
        if context in (CONTEXT_DISCARD_ENERGY_CARD, CONTEXT_DISCARD_TOOL_CARD, CONTEXT_DISCARD_ENERGY, CONTEXT_DISCARD_CARD_OR_ATTACHED):
            return _attached_discard_score(obs, option)
        card = _option_card(obs, option)
        if context in (CONTEXT_TO_HAND_ENERGY, CONTEXT_TO_DECK_ENERGY):
            return 4800 if _card_id(card) == MIST_ENERGY and _spread_risk(obs) else 3200
        return 1000
    if option_type == OPTION_NUMBER:
        return _score_number(obs, option, context)
    if option_type in (OPTION_YES, OPTION_NO):
        return _score_yes_no(obs, option, context)
    if option_type == OPTION_ATTACK:
        return _score_attack(obs, option)
    if option_type == OPTION_EVOLVE:
        return _score_evolve(obs, option)
    if option_type == OPTION_SKILL:
        return 1000
    if option_type == OPTION_SPECIAL_CONDITION:
        # Recovery: prefer poison/burn before sleep/confusion when forced; all legal.
        return 1000 - _int(_field(option, "specialConditionType", 0), 0) * 10
    return 100


# ---------------------------------------------------------------------------
# Legal selection and loop protection
# ---------------------------------------------------------------------------

_MEMORY = {"turn": -1, "signature": None, "repeat": 0}


def _reset_memory() -> None:
    _MEMORY["turn"] = -1
    _MEMORY["signature"] = None
    _MEMORY["repeat"] = 0


def _signature(obs: Any, select: Any) -> tuple[Any, ...]:
    option_sig = []
    for o in _as_list(_field(select, "option", [])):
        option_sig.append((
            _field(o, "type"), _field(o, "area"), _field(o, "index"),
            _field(o, "inPlayArea"), _field(o, "inPlayIndex"),
            _field(o, "attackId"), _field(o, "number"),
        ))
    return (_turn(obs), _turn_actions(obs), _field(select, "context"), tuple(option_sig))


def _update_repeat(obs: Any, select: Any) -> int:
    turn = _turn(obs)
    if turn < _int(_MEMORY.get("turn", -1), -1):
        _reset_memory()
    sig = _signature(obs, select)
    if sig == _MEMORY.get("signature"):
        _MEMORY["repeat"] = _int(_MEMORY.get("repeat", 0), 0) + 1
    else:
        _MEMORY["signature"] = sig
        _MEMORY["repeat"] = 0
    _MEMORY["turn"] = turn
    return _int(_MEMORY["repeat"], 0)


def _choose_indices(obs: Any, select: Any) -> list[int]:
    options = _as_list(_field(select, "option", []))
    if not options:
        return []

    context = _int(_field(select, "context", -1), -1)
    min_count = max(0, _int(_field(select, "minCount", 0), 0))
    max_count = min(len(options), max(min_count, _int(_field(select, "maxCount", min_count), min_count)))
    repeat = _update_repeat(obs, select)

    occurrences: Counter[int] = Counter()
    ranked: list[tuple[float, int]] = []
    for index, option in enumerate(options):
        card = _option_card(obs, option)
        cid = _card_id(card)
        occurrence = occurrences[cid]
        occurrences[cid] += 1
        score = _score_main_option(obs, option) if context == CONTEXT_MAIN else _score_sub_option(obs, option, context, occurrence)
        # Stable tie-breaking rewards earlier options only very slightly.
        ranked.append((float(score) - index * 0.001, index))

    ranked.sort(key=lambda x: x[0], reverse=True)

    if context == CONTEXT_MAIN and repeat >= 2:
        # If the environment returns the exact same choice repeatedly, force a
        # progress action (attack/end) rather than reselecting a stuck effect.
        progress = [
            (score, idx) for score, idx in ranked
            if _int(_field(options[idx], "type", -1), -1) in (OPTION_ATTACK, OPTION_END, OPTION_RETREAT)
        ]
        if progress:
            ranked = progress + [x for x in ranked if x not in progress]

    selected: list[int] = []
    for score, index in ranked:
        if len(selected) >= max_count:
            break
        if len(selected) < min_count or score > 0:
            selected.append(index)

    # Forced selections must always satisfy minCount.
    if len(selected) < min_count:
        for _, index in ranked:
            if index not in selected:
                selected.append(index)
                if len(selected) >= min_count:
                    break

    return sorted(selected[:max_count])


def read_deck_csv() -> list[int]:
    candidates: list[str] = []
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    except NameError:
        pass
    candidates.extend(["/kaggle_simulations/agent/deck.csv", "deck.csv"])

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as file:
                deck = [int(line.strip()) for line in file if line.strip()]
            if len(deck) == 60:
                return deck
        except (OSError, ValueError):
            continue
    return list(DECK)


def agent(observation: dict) -> list[int]:
    """Kaggle public agent: return the deck or legal option indices."""
    select = _field(observation, "select")
    if select is None:
        _reset_memory()
        deck = read_deck_csv()
        if len(deck) != 60:
            raise RuntimeError(f"Deck must contain 60 cards, got {len(deck)}")
        return deck
    return _choose_indices(observation, select)

# =============================================================================
# Universal v6 — focused replay learning with gated tactical search
# =============================================================================
# This layer keeps the strong v2 policy for unknown, mirror and Lucario states,
# and applies only the corrections repeatedly supported by actual loss replays:
#   1) ex spread attacks do not themselves bypass Crustle's damage wall;
#   2) Boss/Petrel should convert one-hit engine KOs against Marnie/Dragapult;
#   3) Mist Energy protects the Dragapult-exposed bench, not active Crustle;
#   4) shallow search is used only for concrete Boss/heal/switch + attack lines.

_V6_OLD_TARGET_VALUE = _target_value
_V6_OLD_ACTIVE_BYPASS = _opponent_active_bypass
_V6_OLD_BYPASS_PRESSURE = _opponent_has_bypass_pressure
_V6_OLD_ENERGY_SCORE = _energy_target_score
_V6_OLD_SUPPORTER_SCORE = _supporter_score
_V6_OLD_PLAY_SCORE = _score_play
_V6_OLD_TO_HAND_SCORE = _to_hand_score
_V6_OLD_ATTACK_SCORE = _score_attack
_V6_OLD_RETREAT_SCORE = _score_retreat

_V6_MARNIE = {646,647,648,860,104}
_V6_DRAG = {119,120,121}
_V6_DUDUN_EX = 306
_V6_ENGINE_BONUS = {
    104: 9000, 112: 7600, 860: 4300, 646: 5000, 647: 6900,
    119: 6500, 120: 8200,
}
_V6_LEARNED = {
    "marnie_boss_ko": 9800.0,
    "marnie_petrel_setup": 4600.0,
    "marnie_crustle_attack": 3300.0,
    "drag_mist_bench": 6200.0,
    "drag_kang_charge": 4500.0,
    "keep_wall_vs_ex_spread": 14000.0,
}


def _v6_arch(obs: Any) -> str:
    ids=_opponent_ids(obs)
    if ids & _V6_MARNIE:return "marnie"
    if ids & _V6_DRAG:return "dragapult"
    if ids & MIRROR_LINE_IDS:return "mirror"
    if ids & {333,677,678,675,676,305,306,65,66}:return "lucario"
    return "unknown"


def _v6_true_bypass(card: Any) -> bool:
    if card is None:return False
    cid=_card_id(card)
    if cid==_V6_DUDUN_EX:return _energy_count(card)>=3
    if cid in KNOWN_BYPASS_ATTACKER_IDS:return _known_bypass_ready(card)
    return _generic_non_ex_bypass_ready(card)


def _opponent_active_bypass(obs: Any) -> bool:  # noqa: F811
    return _v6_true_bypass(_active(_opponent(obs)))


def _opponent_has_bypass_pressure(obs: Any) -> bool:  # noqa: F811
    if _opponent_ids(obs) & BYPASS_SETUP_IDS:return True
    return any(_v6_true_bypass(p) for p in _board(_opponent(obs)))


def _target_value(card: Any, active: bool=False) -> float:  # noqa: F811
    return _V6_OLD_TARGET_VALUE(card,active)+_V6_ENGINE_BONUS.get(_card_id(card),0.0)


def _boss_conversion_value(obs: Any,target: Any) -> float:  # noqa: F811
    if target is None:return -10000.0
    dmg=_attack_damage_budget(obs)
    if dmg<=0:return -9000.0
    score=_target_value(target,False)
    if _hp(target)<=dmg:
        score+=12000+_prize_proxy(target)*1800
        if _card_id(target) in _V6_ENGINE_BONUS:score+=4300
    elif _card_id(target) in _V6_ENGINE_BONUS:
        score+=1800
    else:score-=2500
    return score


def _energy_target_score(obs: Any,target: Any,energy_id: int) -> float:  # noqa: F811
    score=_V6_OLD_ENERGY_SCORE(obs,target,energy_id)
    if target is None:return score
    arch=_v6_arch(obs);cid=_card_id(target);active=target is _active(_me(obs))
    if arch=="marnie":
        if cid==CRUSTLE and energy_id in {SPIKY_ENERGY,GROW_GRASS_ENERGY}:score+=1900
        if cid==MEGA_KANGASKHAN_EX and _field_counts(_me(obs))[CRUSTLE]>0:score-=4800
        if energy_id==MIST_ENERGY:score-=1000
    elif arch=="dragapult":
        if cid==MEGA_KANGASKHAN_EX:score+=_V6_LEARNED["drag_kang_charge"]
        if energy_id==MIST_ENERGY:
            if not active and cid in {DWEBBLE,CRUSTLE,MEGA_KANGASKHAN_EX}:score+=_V6_LEARNED["drag_mist_bench"]
            if active and cid==CRUSTLE:score-=3200
    return score


def _supporter_score(obs: Any,cid: int) -> float:  # noqa: F811
    score=_V6_OLD_SUPPORTER_SCORE(obs,cid)
    arch=_v6_arch(obs)
    if arch=="marnie":
        target=_best_boss_target(obs)
        engine=target is not None and _card_id(target) in {104,112,860,646,647}
        ko=engine and _attack_damage_budget(obs)>0 and _hp(target)<=_attack_damage_budget(obs)
        if cid==BOSSS_ORDERS and ko:score+=_V6_LEARNED["marnie_boss_ko"]
        if cid==TEAM_ROCKETS_PETREL and engine:score+=_V6_LEARNED["marnie_petrel_setup"]
        if cid in {ERI,XEROSICS_MACHINATIONS} and _hand_count(_opponent(obs))<8:score-=5200
        if cid==LILLIES_DETERMINATION and _wall_established(obs) and _hand_count(_me(obs))>=5:score-=3000
    elif arch=="dragapult" and cid==BOSSS_ORDERS:
        target=_best_boss_target(obs)
        if target is not None and _card_id(target) in {119,120} and _hp(target)<=_attack_damage_budget(obs):score+=6200
    return score


def _score_play(obs: Any,option: Any) -> float:  # noqa: F811
    score=_V6_OLD_PLAY_SCORE(obs,option)
    cid=_card_id(_option_card(obs,option));arch=_v6_arch(obs);counts=_field_counts(_me(obs));line=counts[DWEBBLE]+counts[CRUSTLE]
    if arch=="marnie":
        if cid==MEGA_KANGASKHAN_EX and line>0:return -12500.0
        if cid==DWEBBLE and line>=2:return -11200.0
        if cid in {HAND_TRIMMER,ERI,XEROSICS_MACHINATIONS} and _hand_count(_opponent(obs))<8:score-=4800
    if arch=="dragapult" and cid==MEGA_KANGASKHAN_EX and counts[MEGA_KANGASKHAN_EX]==0 and _bench_slots(_me(obs))>0:
        score=max(score,7600.0)
    return score


def _to_hand_score(obs: Any,card: Any,effect_id: int) -> float:  # noqa: F811
    score=_V6_OLD_TO_HAND_SCORE(obs,card,effect_id);cid=_card_id(card);arch=_v6_arch(obs)
    if arch=="marnie" and cid==BOSSS_ORDERS:
        t=_best_boss_target(obs)
        if t is not None and _card_id(t) in {104,112,860,646,647}:score+=7600
    if arch=="dragapult" and cid==MIST_ENERGY:score+=3000
    return score


def _score_attack(obs: Any,option: Any) -> float:  # noqa: F811
    score=_V6_OLD_ATTACK_SCORE(obs,option);arch=_v6_arch(obs)
    if arch=="marnie" and _card_id(_active(_me(obs)))==CRUSTLE:score+=_V6_LEARNED["marnie_crustle_attack"]
    if arch=="dragapult" and _card_id(_active(_me(obs)))==MEGA_KANGASKHAN_EX:score+=1600
    return score


def _score_retreat(obs: Any) -> float:  # noqa: F811
    score=_V6_OLD_RETREAT_SCORE(obs)
    if _card_id(_active(_me(obs)))==CRUSTLE and _card_id(_active(_opponent(obs))) in {121,648}:
        score-=_V6_LEARNED["keep_wall_vs_ex_spread"]
    return score

# Optional engine search imports.
try:
    from cg.api import to_observation_class as _v6_to_obs, search_begin as _v6_begin, search_step as _v6_step, search_end as _v6_end, search_release as _v6_release
    _V6_SEARCH=True
except Exception:
    _V6_SEARCH=False

_V6_TEMPLATES={
 "marnie":[7]*10+[104]*2+[112]*4+[646]*4+[647]*3+[648]*3+[860]*2+[1079]*3+[1080]+[1086]*4+[1097]*3+[1122]+[1137]+[1152]*4+[1182]*2+[1219]*4+[1227]*4+[1231]+[1259]*4,
 "dragapult":[119]*4+[120]*4+[121]*2+[112]*2+[235]*2+[31,140,343,1071,689]+[1227]*4+[1182]*3+[1198]*2+[1213,1240]+[1121]*4+[1086]*4+[1152]*4+[1097]*3+[1260,1080,1081,1137]+[1256]*2+[2]*4+[5]*3+[7]*2,
}

def _v6_known(player: Any) -> list[int]:
    out=[]
    for c in _hand(player)+_as_list(_field(player,"discard",[])):
        if _card_id(c):out.append(_card_id(c))
    for p in _board(player):
        if _card_id(p):out.append(_card_id(p))
        out += [_card_id(x) for x in _pre_evolutions(p) if _card_id(x)]
        out += [_card_id(x) for x in _energy_cards(p) if _card_id(x)]
        out += [_card_id(x) for x in _tools(p) if _card_id(x)]
    return out

def _v6_remove(pool,known):
    c=Counter(pool)
    for x in known:
        if c[x]>0:c[x]-=1
    out=[]
    for x in sorted(c):out.extend([x]*c[x])
    return out

def _v6_hidden(obs: Any):
    me,op=_me(obs),_opponent(obs);rem=_v6_remove(read_deck_csv(),_v6_known(me));pn=len(_as_list(_field(me,"prize",[])))
    yp=rem[:pn];yd=rem[pn:pn+_deck_count(me)]
    while len(yp)<pn:yp.append(BASIC_GRASS_ENERGY)
    while len(yd)<_deck_count(me):yd.append(BASIC_GRASS_ENERGY)
    arch=_v6_arch(obs);pool=_v6_remove(_V6_TEMPLATES.get(arch,[344]*4+[1]*56),_v6_known(op));need=_hand_count(op)+len(_as_list(_field(op,"prize",[])))+_deck_count(op)
    filler=7 if arch=="marnie" else 5
    pool += [filler]*max(0,need-len(pool));h=_hand_count(op);pp=len(_as_list(_field(op,"prize",[])))
    return yd,yp,pool[h+pp:h+pp+_deck_count(op)],pool[h:h+pp],pool[:h],[]

def _v6_rollout_choose(obs: Any) -> list[int]:
    s=_field(obs,"select");opts=_as_list(_field(s,"option",[]))
    if not opts:return []
    ctx=_int(_field(s,"context",-1),-1);mn=max(0,_int(_field(s,"minCount",0),0));mx=min(len(opts),max(mn,_int(_field(s,"maxCount",mn),mn)));occ=Counter();rank=[]
    for i,o in enumerate(opts):
        c=_option_card(obs,o);cid=_card_id(c);n=occ[cid];occ[cid]+=1;v=_score_main_option(obs,o) if ctx==CONTEXT_MAIN else _score_sub_option(obs,o,ctx,n);rank.append((v-i*.001,i))
    rank.sort(reverse=True);sel=[]
    for v,i in rank:
        if len(sel)>=mx:break
        if len(sel)<mn or v>0:sel.append(i)
    if len(sel)<mn:
        for _,i in rank:
            if i not in sel:sel.append(i)
            if len(sel)>=mn:break
    return sorted(sel[:mx])

def _v6_hp(player: Any) -> int:return sum(max(0,_hp(p)) for p in _board(player))

def _v6_tactical_choice(obs: Any,select: Any,heuristic: list[int]) -> list[int]:
    if not _V6_SEARCH or not heuristic or _v6_arch(obs) not in {"marnie","dragapult"}:return heuristic
    if _field(obs,"search_begin_input") is None or _int(_field(select,"context",-1),-1)!=CONTEXT_MAIN:return heuristic
    opts=_as_list(_field(select,"option",[]));att=[i for i,o in enumerate(opts) if _int(_field(o,"type",-1),-1)==OPTION_ATTACK];tac=[]
    for i,o in enumerate(opts):
        typ=_int(_field(o,"type",-1),-1);cid=_card_id(_option_card(obs,o))
        if typ==OPTION_PLAY and cid in {BOSSS_ORDERS,LISIAS_APPEAL,SWITCH,HEAL_SUPPORTER,BIANCAS_DEVOTION,JUMBO_ICE_CREAM}:tac.append(i)
        elif typ in {OPTION_RETREAT,OPTION_EVOLVE}:tac.append(i)
    if not att or not tac:return heuristic
    cand=[heuristic[0]]
    for i in tac+att:
        if i not in cand:cand.append(i)
    cand=cand[:5]
    try:yd,yp,od,opr,oh,oa=_v6_hidden(obs);root=_v6_begin(_v6_to_obs(obs),yd,yp,od,opr,oh,oa,False)
    except Exception:return heuristic
    yi=_your_index(obs);me0,op0=_me(obs),_opponent(obs);rt=_turn(obs);res=[]
    try:
        for idx in cand:
            leaf=None
            try:
                st=_v6_step(root.searchId,[idx]);leaf=st.searchId;d=0
                while (_field(st.observation,"select") is not None and _int(_field(_field(st.observation,"current",{}),"result",-1),-1)==-1 and _int(_field(_field(st.observation,"current",{}),"turn",rt),rt)==rt and d<10):
                    st=_v6_step(st.searchId,_v6_rollout_choose(st.observation));leaf=st.searchId;d+=1
                me=_player(st.observation,yi);op=_player(st.observation,1-yi);pg=len(_as_list(_field(op0,"prize",[])))-len(_as_list(_field(op,"prize",[])));dmg=max(0,_v6_hp(op0)-_v6_hp(op));spent=max(0,_deck_count(me0)-_deck_count(me));wall=1 if _card_id(_active(me))==CRUSTLE and not _opponent_active_bypass(st.observation) else 0
                val=pg*100000+dmg*22-spent*60+wall*2200+.01*_score_main_option(obs,opts[idx]);res.append((val,(pg,dmg,wall),idx))
            except Exception:pass
            finally:
                if leaf is not None:
                    try:_v6_release(leaf)
                    except Exception:pass
    finally:
        try:_v6_end()
        except Exception:pass
    if not res:return heuristic
    res.sort(reverse=True);best=res[0];cur=next((x for x in res if x[2]==heuristic[0]),None)
    if cur is None:return heuristic
    decisive=best[1][0]>cur[1][0] or best[1][1]>=cur[1][1]+60 or (best[1][2]>cur[1][2] and best[1][1]>=cur[1][1]-20)
    return [best[2]] if decisive else heuristic

_V6_BASE_AGENT=agent

def agent(observation: dict) -> list[int]:  # noqa: F811
    select=_field(observation,"select")
    if select is None:
        _reset_memory();deck=read_deck_csv()
        if len(deck)!=60:raise RuntimeError(f"Deck must contain 60 cards, got {len(deck)}")
        return deck
    h=_choose_indices(observation,select)
    return _v6_tactical_choice(observation,select,h)
# Evaluation switch: retain learned policy while disabling expensive branch search.
_V6_SEARCH = False
# v7c: matchup-gated replay learning.  Marnie's counter engine requires a
# three-Energy Crustle to begin taking one-hit KOs before Froslass/Munkidori
# damage-counter pressure accumulates.  Preserve the proven base curve elsewhere.
_V7C_PREV_ENERGY_SCORE = _energy_target_score
def _energy_target_score(obs: Any, target: Any, energy_id: int) -> float:  # noqa: F811
    score = _V7C_PREV_ENERGY_SCORE(obs, target, energy_id)
    if target is not None and _v6_arch(obs)=="marnie" and _card_id(target)==CRUSTLE:
        units=_energy_count(target)
        if units<3:
            score=max(score,(15800.0,16600.0,16200.0)[units])
            if energy_id==GROW_GRASS_ENERGY and not any(_card_id(e)==GROW_GRASS_ENERGY for e in _energy_cards(target)):
                score+=900.0
            if target is _active(_me(obs)):score+=1200.0
        else:
            score=min(score,-4600.0-(units-3)*2600.0)
    return score
# v8 replay lesson: at 30 HP or less, a legal full/large heal must precede a
# non-winning attack.  Multiple exact loss branches flipped after this choice.
_V8H_PREV_MAIN_SCORE = _score_main_option

def _v8h_attack_wins_now(obs: Any) -> bool:
    target=_active(_opponent(obs));dmg=_attack_damage_budget(obs)
    return bool(target is not None and dmg>0 and _hp(target)<=dmg and _prize_count(_me(obs))<=_prize_proxy(target))

def _score_main_option(obs: Any, option: Any) -> float:  # noqa: F811
    score=_V8H_PREV_MAIN_SCORE(obs,option)
    typ=_int(_field(option,"type",-1),-1)
    cid=_card_id(_option_card(obs,option))
    active=_active(_me(obs))
    if active is not None and _hp(active)<=30 and _damage(active)>0 and not _v8h_attack_wins_now(obs):
        legal_heal=(typ==OPTION_PLAY and cid in {JUMBO_ICE_CREAM,HEAL_SUPPORTER,BIANCAS_DEVOTION})
        if legal_heal:
            # Jumbo needs 3 Energy; Bianca's legal option already means its HP
            # condition is satisfied.  Legal option generation validates cards.
            score=max(score,52000.0+_damage(active)*25.0)
        elif typ==OPTION_ATTACK:
            score-=26000.0
    return score


# =============================================================================
# Universal v8 — expanded state sensing, replay-distilled residual policy, and
# concrete one-turn lookahead.
# =============================================================================
# The rule policy remains primary.  The learned residual was fitted offline to
# 1,761 counterfactual branches from 238 decision states reconstructed from 80
# real loss replays.  It is intentionally low-weight and confidence-gated.

_V8_MODEL = {'feature_count': 98, 'learning_rate': 0.07, 'init': -1.0, 'trees': [{'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 68, -2, -2, 13, -2, -2], 'threshold': [4.5, 0.5, -2.0, -2.0, 1.5, -2.0, -2.0], 'value': [0.46055741792929294, 0.15980629539951574, 0.1309823677581864, 0.875, 0.5666293339026474, 0.855460147601476, 0.4796591666666667]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [4, 12, -2, -2, 19, -2, -2], 'threshold': [13.5, 0.5, -2.0, -2.0, 275.0, -2.0, -2.0], 'value': [0.42297939472336377, 0.5225703939810193, 0.3088481354947994, 0.6284583757634159, 0.13874522895882863, 0.061771891182574964, 0.49212793841827396]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [4, 0, -2, -2, 68, -2, -2], 'threshold': [13.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.39339648486965906, 0.48634999875306, 0.38938529731368343, 0.7495045829232068, 0.1173315376219673, 0.0835384321430781, 0.8355556340658171]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 68, -2, -2, 13, -2, -2], 'threshold': [4.5, 0.5, -2.0, -2.0, 1.5, -2.0, -2.0], 'value': [0.37182480216887925, 0.11219586159203876, 0.08507110387562783, 0.7013104333521473, 0.45983089276170486, 0.6976821928492937, 0.36217937066426803]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 19, -2, -2, 12, -2, -2], 'threshold': [4.5, 225.0, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.33877225948633566, 0.09723188882072377, 0.006855076458027139, 0.2397069436322652, 0.4220196198345888, 0.2148762869246838, 0.4976221190563547]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [4, 12, -2, -2, 9, -2, -2], 'threshold': [13.5, 0.5, -2.0, -2.0, 6.5, -2.0, -2.0], 'value': [0.3142830132928851, 0.39512906893705263, 0.1864194280447144, 0.4513352160506364, 0.07255941619055051, 0.25745004460232, -0.0029151444869817585]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 15, -2, -2, 2, -2, -2], 'threshold': [5.5, 295.0, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.2998231323064234, 0.12907965412111636, 0.09023286216442139, 0.5715036113327776, 0.3876433916407391, 0.43281702273194544, 0.1277938387908184]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 68, -2, -2, 12, -2, -2], 'threshold': [4.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.2643486540158468, 0.06602600321975786, 0.040291331165594246, 0.5416744522398228, 0.33270094283011864, 0.14148415878004145, 0.36623642857721617]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [4, 7, -2, -2, 68, -2, -2], 'threshold': [13.5, 1.5, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.2601521690725732, 0.33238108926693966, 1.0581202862534833, 0.2653188791964541, 0.049186511079134174, 0.020466981827925085, 0.510344779022866]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [4, 7, -2, -2, 19, -2, -2], 'threshold': [13.5, 1.5, -2.0, -2.0, 275.0, -2.0, -2.0], 'value': [0.2455800096198509, 0.31013158165369464, 1.0474697556147952, 0.2376624419003514, 0.05450735639966059, -0.0037750573672705526, 0.31066517410071615]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 68, -2, -2, 2, -2, -2], 'threshold': [4.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.2397872244716074, 0.060862023792662236, 0.03312135754716196, 0.5413746817141872, 0.3012500796666684, 0.32494862529271795, 0.05211153048189493]}, {'children_left': [1, 2, -1, -1, 5, -1, -1], 'children_right': [4, 3, -1, -1, 6, -1, -1], 'feature': [11, 68, -2, -2, 12, -2, -2], 'threshold': [4.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0], 'value': [0.20418830768219148, 0.04149094313235061, 0.015442712432035224, 0.4540446304778274, 0.26063433211785086, 0.059884795013715186, 0.2757693301808186]}]}
_V8_MATCHUPS = ("mirror","dragapult","marnie","lucario")
_V8_IMPORTANT_OPP = (104,112,646,647,648,860,119,120,121,333,677,678,305,306,65,66,344,345,756)
_V8_IMPORTANT_CARDS = (344,345,756,1227,1182,1219,1225,1186,1197,1212,1190,1204,1147,1122,1086,1121,1123,1087,1159,1161,1257,1242,1245,14,18,11,1)
_V8_OPTION_TYPES = tuple(range(7,15))
_V8_ATTACKS = (478,479,1092)
_V8_SEARCH_CACHE: dict[tuple[Any,...], int] = {}


def _v8_phase(obs: Any) -> str:
    me=_me(obs);counts=_field_counts(me);ready=any(_card_id(p)==CRUSTLE and _energy_count(p)>=3 for p in _board(me))
    if counts[DWEBBLE]+counts[CRUSTLE]==0:return "no_line"
    if counts[CRUSTLE]==0:return "evolve"
    if not ready:return "charge"
    if _prize_count(me)<=2 or _prize_count(_opponent(obs))<=2:return "endgame"
    if _deck_count(me)<=8:return "deck_endgame"
    return "pressure"


def _v8_engine_count(player: Any, arch: str) -> int:
    ids=[_card_id(p) for p in _board(player)]
    if arch=="marnie":return sum(x in {104,112,646,647,860} for x in ids)
    if arch=="dragapult":return sum(x in {119,120} for x in ids)
    if arch=="lucario":return sum(x in {333,677,305,306,65,66} for x in ids)
    if arch=="mirror":return sum(x in {DWEBBLE,MEGA_KANGASKHAN_EX} for x in ids)
    return 0


def _v8_immediate_ko_target(obs: Any) -> Any:
    dmg=_attack_damage_budget(obs)
    if dmg<=0:return None
    return max((p for p in _bench(_opponent(obs)) if _hp(p)<=dmg),key=lambda p:_target_value(p,False),default=None)


def _v8_attack_wins_now(obs: Any) -> bool:
    t=_active(_opponent(obs));d=_attack_damage_budget(obs)
    return bool(t is not None and d>0 and _hp(t)<=d and _prize_count(_me(obs))<=_prize_proxy(t))


_V8_PREV_MAIN_SCORE = _score_main_option

def _score_main_option(obs: Any, option: Any) -> float:  # noqa: F811
    score=_V8_PREV_MAIN_SCORE(obs,option)
    typ=_int(_field(option,"type",-1),-1);card=_option_card(obs,option);cid=_card_id(card)
    arch=_v6_arch(obs);phase=_v8_phase(obs);me=_me(obs);active=_active(me);counts=_field_counts(me)
    attack_ready=_has_attack_option(_field(obs,"select",{}))
    target=_option_target(obs,option)

    # Emergency survival is universal and was the cleanest repeated
    # counterfactual correction: at <=30 HP, heal before a non-winning attack.
    if active is not None and _hp(active)<=30 and _damage(active)>0 and not _v8_attack_wins_now(obs):
        if typ==OPTION_PLAY and cid in {JUMBO_ICE_CREAM,HEAL_SUPPORTER,BIANCAS_DEVOTION}:
            score=max(score,56000.0+_damage(active)*28.0)
        elif typ==OPTION_ATTACK:
            score-=30000.0

    if arch=="marnie":
        # One Crustle line attacks while keeping the bench small.  The matchup
        # is won by starting 120-damage one-hit KOs before counters accumulate.
        if typ==OPTION_PLAY and cid==DWEBBLE and counts[DWEBBLE]+counts[CRUSTLE]>=1:score-=12500.0
        if typ==OPTION_PLAY and cid==MEGA_KANGASKHAN_EX and counts[DWEBBLE]+counts[CRUSTLE]>0:score-=9000.0
        if typ==OPTION_ATTACK and _card_id(active)==CRUSTLE:score+=2600.0
        if typ==OPTION_PLAY and cid==BOSSS_ORDERS:
            ko=_v8_immediate_ko_target(obs)
            if ko is not None and _card_id(ko) in {104,112,860,646,647}:score+=5200.0
        if typ==OPTION_PLAY and cid==TEAM_ROCKETS_PETREL and attack_ready and BOSSS_ORDERS not in _hand_counts(me) and _v8_engine_count(_opponent(obs),arch)>0:score+=1800.0
        if attack_ready and _card_id(active)==CRUSTLE and typ==OPTION_PLAY and cid in {LILLIES_DETERMINATION,HILDA,ERI,XEROSICS_MACHINATIONS,HAND_TRIMMER,POKEGEAR_30}:
            score-=2200.0
        if typ==OPTION_ATTACH and cid==HEROS_CAPE and target is not None and _card_id(target)==CRUSTLE:score+=5800.0

    elif arch=="dragapult":
        wall_ready=any(_card_id(p)==CRUSTLE and _energy_count(p)>=3 for p in _board(me))
        if typ==OPTION_ATTACK and _card_id(active)==CRUSTLE:score+=1800.0
        if typ==OPTION_ATTACH and cid==MIST_ENERGY and target is not None:
            if target is active and _card_id(target)==CRUSTLE:score-=4200.0
            elif target in _bench(me) and _card_id(target) in {DWEBBLE,CRUSTLE,MEGA_KANGASKHAN_EX}:score+=3400.0 if wall_ready else -900.0
        if typ==OPTION_ATTACH and target is not None and _card_id(target)==CRUSTLE and _energy_count(target)<3:score+=2600.0
        if typ==OPTION_PLAY and cid==MEGA_KANGASKHAN_EX and counts[MEGA_KANGASKHAN_EX]>=1:score-=6200.0
        if typ==OPTION_PLAY and cid==BOSSS_ORDERS:
            ko=_v8_immediate_ko_target(obs)
            if ko is not None and _card_id(ko) in {119,120}:score+=4300.0

    elif arch=="lucario":
        if typ==OPTION_PLAY and cid==BOSSS_ORDERS:
            ko=_v8_immediate_ko_target(obs)
            if ko is not None and _card_id(ko) in {333,677,305,306,65,66}:score+=3900.0
        if typ==OPTION_ATTACK and _card_id(active)==CRUSTLE and _card_id(_active(_opponent(obs))) in {677,678}:score+=1800.0

    elif arch=="mirror":
        if typ==OPTION_PLAY and cid in {LILLIES_DETERMINATION,TEAM_ROCKETS_PETREL,HILDA,POKEGEAR_30} and _wall_established(obs) and _deck_count(me)<=_deck_count(_opponent(obs))+4:score-=7500.0
        if typ==OPTION_PLAY and cid==BOSSS_ORDERS:
            ko=_v8_immediate_ko_target(obs)
            if ko is not None and _card_id(ko) in {DWEBBLE,MEGA_KANGASKHAN_EX}:score+=4200.0

    # Generic phase sense: finish setup/energy before optional disruption, but
    # do not suppress an immediate prize conversion or emergency heal.
    if phase in {"no_line","evolve","charge"} and typ==OPTION_PLAY and cid in {ERI,XEROSICS_MACHINATIONS,HAND_TRIMMER}:
        score-=2200.0
    return score


def _v8_model_features(obs: Any, option: Any) -> list[float]:
    arch=_v6_arch(obs);st=_field(obs,"current",{});yi=_your_index(obs);me=_me(obs);op=_opponent(obs)
    mb=_board(me);ob=_board(op);ma=_active(me);oa=_active(op);mc=_field_counts(me);oc=_field_counts(op)
    typ=_int(_field(option,"type",-1),-1);src=_option_card(obs,option);tar=_option_target(obs,option);scid=_card_id(src);tcid=_card_id(tar);aid=_int(_field(option,"attackId",0),0)
    x=[]
    x += [1.0 if arch==m else 0.0 for m in _V8_MATCHUPS]
    x += [float(_turn(obs)),float(_turn_actions(obs)),float(_hand_count(me)),float(_hand_count(op)),float(_deck_count(me)),float(_deck_count(op)),float(_prize_count(me)),float(_prize_count(op)),float(len(_bench(me))),float(len(_bench(op)))]
    x += [float(_card_id(ma)),float(_hp(ma)),float(_energy_count(ma)),float(_damage(ma)),float(_card_id(oa)),float(_hp(oa)),float(_energy_count(oa)),float(_damage(oa))]
    x += [float(sum(_energy_count(z) for z in mb)),float(sum(_energy_count(z) for z in ob)),float(sum(_damage(z) for z in mb)),float(sum(_damage(z) for z in ob))]
    x += [float(mc[i]) for i in (DWEBBLE,CRUSTLE,MEGA_KANGASKHAN_EX)]
    x += [float(oc[i]) for i in _V8_IMPORTANT_OPP]
    x += [1.0 if typ==i else 0.0 for i in _V8_OPTION_TYPES]
    x += [1.0 if scid==i else 0.0 for i in _V8_IMPORTANT_CARDS]
    x += [1.0 if aid==i else 0.0 for i in _V8_ATTACKS]
    x += [float(scid),float(tcid),float(_hp(tar)),float(_energy_count(tar)),float(_damage(tar))]
    x += [float(_has_attack_option(_field(obs,"select",{}))),float(mc[CRUSTLE]>0),float(any(_card_id(z)==CRUSTLE and _energy_count(z)>=3 for z in mb)),float(any(_card_id(z)==MEGA_KANGASKHAN_EX and _energy_count(z)>=3 for z in mb)),float(oc[104]+oc[112]),float(oc[119]+oc[120]),float(oc[305]+oc[306]+oc[65]+oc[66])]
    return x


def _v8_model_predict(x: list[float]) -> float:
    m=_V8_MODEL;v=float(m["init"])
    if len(x)!=int(m["feature_count"]):return v
    for t in m["trees"]:
        n=0
        while t["children_left"][n]!=-1:
            f=t["feature"][n];n=t["children_left"][n] if x[f]<=t["threshold"][n] else t["children_right"][n]
        v += float(m["learning_rate"])*float(t["value"][n])
    return v


def _v8_learned_choice(obs: Any, select: Any, heuristic: list[int]) -> list[int]:
    if not heuristic or _int(_field(select,"context",-1),-1)!=CONTEXT_MAIN or _turn(obs)<2:return heuristic
    arch=_v6_arch(obs)
    if arch not in _V8_MATCHUPS:return heuristic
    opts=_as_list(_field(select,"option",[]))
    if len(opts)<2:return heuristic
    h=heuristic[0]
    preds=[_v8_model_predict(_v8_model_features(obs,o)) for o in opts]
    learned=max(range(len(opts)),key=lambda i:preds[i])
    if learned==h:return heuristic
    margin=preds[learned]-preds[h]
    # Conservative residual: only act on a clear learned margin when the base
    # rule scores are already close.  Setup, END and unsafe retreat are excluded.
    if margin<0.115:return heuristic
    hs=_score_main_option(obs,opts[h]);ls=_score_main_option(obs,opts[learned])
    if hs-ls>2600:return heuristic
    ltyp=_int(_field(opts[learned],"type",-1),-1)
    if ltyp==OPTION_END and _deck_count(_me(obs))>2:return heuristic
    if ltyp==OPTION_RETREAT and _score_retreat(obs)<0:return heuristic
    return [learned]


def _v8_public_value(before: Any, after: Any, arch: str, yi: int) -> tuple[int,int,int,int,int,float]:
    cur=_field(after,"current",{});res=_int(_field(cur,"result",-1),-1);me0=_player(before,yi);op0=_player(before,1-yi);me=_player(after,yi);op=_player(after,1-yi)
    terminal=2 if res==yi else (-2 if res==1-yi else 0)
    prize_gain=_prize_count(op0)-_prize_count(op)
    engines=_v8_engine_count(op0,arch)-_v8_engine_count(op,arch)
    opp_damage=max(0,sum(_hp(p) for p in _board(op0))-sum(_hp(p) for p in _board(op)))
    heal=max(0,sum(_hp(p) for p in _board(me))-sum(_hp(p) for p in _board(me0)))
    wall=1 if _card_id(_active(me))==CRUSTLE and not _opponent_active_bypass(after) else 0
    ready=sum(_card_id(p)==CRUSTLE and _energy_count(p)>=3 for p in _board(me))+sum(_card_id(p)==MEGA_KANGASKHAN_EX and _energy_count(p)>=3 for p in _board(me))
    deck_spent=max(0,_deck_count(me0)-_deck_count(me))
    scalar=prize_gain*120000+engines*24000+opp_damage*35+heal*24+wall*2600+ready*1800-deck_spent*70
    return (terminal,prize_gain,engines,heal,wall,scalar)


def _v8_search_candidates(obs: Any, select: Any, heuristic: list[int]) -> list[int]:
    if not heuristic:return []
    opts=_as_list(_field(select,"option",[]));cand=[heuristic[0]];dmg=_attack_damage_budget(obs);active=_active(_me(obs))
    concrete=False
    for i,o in enumerate(opts):
        typ=_int(_field(o,"type",-1),-1);cid=_card_id(_option_card(obs,o))
        if typ==OPTION_PLAY and cid==BOSSS_ORDERS and dmg>0 and _v8_immediate_ko_target(obs) is not None:
            cand.append(i);concrete=True
        elif typ==OPTION_PLAY and cid in {JUMBO_ICE_CREAM,HEAL_SUPPORTER,BIANCAS_DEVOTION} and active is not None and _hp(active)<=50 and _damage(active)>=70:
            cand.append(i);concrete=True
        elif typ in {OPTION_RETREAT} and ((_card_id(active)==MEGA_KANGASKHAN_EX and _crustle_ready_on_bench(obs)) or (_card_id(active)==CRUSTLE and _opponent_active_bypass(obs) and _kang_ready_on_bench(obs))):
            cand.append(i);concrete=True
        elif typ==OPTION_PLAY and cid==SWITCH and ((_card_id(active)==MEGA_KANGASKHAN_EX and _crustle_ready_on_bench(obs)) or (_card_id(active)==CRUSTLE and _opponent_active_bypass(obs) and _kang_ready_on_bench(obs))):
            cand.append(i);concrete=True
        elif typ==OPTION_ATTACK:
            cand.append(i)
    if not concrete:return []
    out=[]
    for i in cand:
        if i not in out:out.append(i)
    return out[:3]


def _v8_lookahead(obs: Any, select: Any, heuristic: list[int]) -> list[int]:
    if _field(obs,"search_begin_input") is None or _int(_field(select,"context",-1),-1)!=CONTEXT_MAIN:return heuristic
    cand=_v8_search_candidates(obs,select,heuristic)
    if len(cand)<2:return heuristic
    sig=_signature(obs,select)+(tuple(cand),)
    cached=_V8_SEARCH_CACHE.get(sig)
    if cached is not None and cached in cand:return [cached]
    try:
        yd,yp,od,opr,oh,oa=_v6_hidden(obs);root=_v6_begin(_v6_to_obs(obs),yd,yp,od,opr,oh,oa,False)
    except Exception:return heuristic
    yi=_your_index(obs);arch=_v6_arch(obs);root_turn=_turn(obs);rank=[]
    try:
        for idx in cand:
            leaf=None
            try:
                st=_v6_step(root.searchId,[idx]);leaf=st.searchId;depth=0
                while (_field(st.observation,"select") is not None and _int(_field(_field(st.observation,"current",{}),"result",-1),-1)==-1 and _int(_field(_field(st.observation,"current",{}),"turn",root_turn),root_turn)==root_turn and depth<7):
                    st=_v6_step(st.searchId,_v6_rollout_choose(st.observation));leaf=st.searchId;depth+=1
                rank.append((_v8_public_value(obs,st.observation,arch,yi),idx))
            except Exception:
                pass
            finally:
                if leaf is not None:
                    try:_v6_release(leaf)
                    except Exception:pass
    finally:
        try:_v6_end()
        except Exception:pass
    if not rank:return heuristic
    rank.sort(reverse=True);bestv,besti=rank[0];cur=next((v for v,i in rank if i==heuristic[0]),None)
    if cur is None:return heuristic
    # Adopt only a concrete tactical improvement, never a small scalar fluctuation.
    decisive=(bestv[0]>cur[0] or bestv[1]>cur[1] or bestv[2]>cur[2] or (bestv[3]>=cur[3]+60 and bestv[1]>=cur[1]) or (bestv[4]>cur[4] and bestv[5]>=cur[5]-800))
    chosen=besti if decisive else heuristic[0]
    if len(_V8_SEARCH_CACHE)>256:_V8_SEARCH_CACHE.clear()
    _V8_SEARCH_CACHE[sig]=chosen
    return [chosen]


_V8_PREV_AGENT = agent

def agent(observation: dict) -> list[int]:  # noqa: F811
    select=_field(observation,"select")
    if select is None:
        _V8_SEARCH_CACHE.clear()
        return _V8_PREV_AGENT(observation)
    heuristic=_choose_indices(observation,select)
    learned=_v8_learned_choice(observation,select,heuristic)
    return _v8_lookahead(observation,select,learned)

# Lightweight diagnostics used by the offline league runner.  They do not alter
# decisions and are harmless in Kaggle execution.
_V8_DIAGNOSTICS = {"learned_evaluated":0,"learned_overrides":0,"search_triggered":0,"search_overrides":0}
_V8_LEARN_CORE = _v8_learned_choice
_V8_SEARCH_CORE = _v8_lookahead

def _v8_learned_choice(obs: Any, select: Any, heuristic: list[int]) -> list[int]:  # noqa: F811
    _V8_DIAGNOSTICS["learned_evaluated"] += 1
    out=_V8_LEARN_CORE(obs,select,heuristic)
    if out!=heuristic:_V8_DIAGNOSTICS["learned_overrides"] += 1
    return out

def _v8_lookahead(obs: Any, select: Any, heuristic: list[int]) -> list[int]:  # noqa: F811
    cand=_v8_search_candidates(obs,select,heuristic) if _field(obs,"select") is not None else []
    if len(cand)>=2:_V8_DIAGNOSTICS["search_triggered"] += 1
    out=_V8_SEARCH_CORE(obs,select,heuristic)
    if out!=heuristic:_V8_DIAGNOSTICS["search_overrides"] += 1
    return out

# ---------------------------------------------------------------------------
# v17 guarded-learning shadow sensor
# ---------------------------------------------------------------------------
# Training from 174 independent loss replays and 520 counterfactual decision
# states did not produce a confidence gate that improved unseen replay choices.
# The learned components therefore run in shadow/telemetry mode only.  The
# public action returned below is exactly the validated v8 champion action.

_V17_CHAMPION_AGENT = agent
_V17_SHADOW = {
    "observations": 0,
    "risk_observations": 0,
    "matchups": {"mirror": 0, "dragapult": 0, "marnie": 0, "lucario": 0, "unknown": 0},
    "phases": {"opening": 0, "development": 0, "pressure": 0, "endgame": 0},
    "flags": {
        "low_active_hp": 0,
        "deckout_risk": 0,
        "bypass_pressure": 0,
        "counter_engine_pressure": 0,
        "wall_missing": 0,
        "kangaskhan_liability": 0,
        "boss_exposure": 0,
        "wall_ready": 0,
    },
    "last": None,
}


def _v17_shadow_sense(obs: Any) -> dict[str, Any]:
    me = _me(obs)
    op = _opponent(obs)
    active = _active(me)
    turn = _turn(obs)
    prizes = _prize_count(me)
    deck = _deck_count(me)
    arch = _v6_arch(obs)
    if arch not in _V17_SHADOW["matchups"]:
        arch = "unknown"

    counts = _field_counts(me)
    wall_ready = any(
        _card_id(p) == CRUSTLE and _energy_count(p) >= 2
        for p in _board(me)
    )
    kang_liability = any(
        _card_id(p) == MEGA_KANGASKHAN_EX and _energy_count(p) < 2
        for p in _bench(me)
    )
    boss_exposure = any(
        _card_id(p) == MEGA_KANGASKHAN_EX
        for p in _bench(me)
    )
    counter_engine = bool(
        _opponent_ids(obs) & {104, 112, 646, 647, 648, 860, 305, 306}
    )
    bypass = _opponent_has_bypass_pressure(obs)
    low_hp = active is not None and _hp(active) <= 50
    deckout = deck <= max(4, prizes + 1)
    wall_missing = counts[DWEBBLE] + counts[CRUSTLE] == 0

    if turn <= 3:
        phase = "opening"
    elif prizes <= 2 or deck <= 6:
        phase = "endgame"
    elif bypass or counter_engine or (active is not None and _damage(active) >= 100):
        phase = "pressure"
    else:
        phase = "development"

    flags = {
        "low_active_hp": low_hp,
        "deckout_risk": deckout,
        "bypass_pressure": bypass,
        "counter_engine_pressure": counter_engine,
        "wall_missing": wall_missing,
        "kangaskhan_liability": kang_liability,
        "boss_exposure": boss_exposure,
        "wall_ready": wall_ready,
    }
    mask = 0
    for bit, key in enumerate(flags):
        if flags[key]:
            mask |= 1 << bit
    return {
        "turn": turn,
        "phase": phase,
        "matchup": arch,
        "risk_mask": mask,
        "flags": flags,
        "deck": deck,
        "prizes": prizes,
        "active_id": _card_id(active),
        "active_hp": _hp(active) if active is not None else 0,
        "opponent_active_id": _card_id(_active(op)),
    }


def agent(observation: dict) -> list[int]:  # noqa: F811
    try:
        sensed = _v17_shadow_sense(observation)
        _V17_SHADOW["observations"] += 1
        _V17_SHADOW["matchups"][sensed["matchup"]] += 1
        _V17_SHADOW["phases"][sensed["phase"]] += 1
        risky = False
        for key, value in sensed["flags"].items():
            if value:
                _V17_SHADOW["flags"][key] += 1
                if key != "wall_ready":
                    risky = True
        if risky:
            _V17_SHADOW["risk_observations"] += 1
        _V17_SHADOW["last"] = sensed
    except Exception:
        # Telemetry must never affect legal action generation.
        pass
    return _V17_CHAMPION_AGENT(observation)

# =============================================================================
# Universal v18 — disruption-aware control layer
# =============================================================================
# This layer improves the use of the deck's interaction cards.  It is deliberately
# state-gated: it never gives up an immediate winning/engine KO, emergency heal,
# or missing Crustle development merely to reduce the opponent's hand.

_V18_PREV_MAIN_SCORE = _score_main_option
_V18_PREV_CARD_SCORE = _score_card_selection
_V18_PREV_SUB_SCORE = _score_sub_option
_V18_PREV_TOOL_SCORE = _tool_target_score

_V18_MODE = "balanced"
_V18_CFG = {'self_trim_cost': 0.9, 'trim_base': 4300.0, 'trim_per': 1550.0, 'xero_base': 5000.0, 'xero_per': 1650.0, 'eri_base': 4300.0, 'eri_per': 2800.0, 'fan_attach': 3000.0, 'stadium_denial': 4800.0}

_V18_ITEM_PRIORITIES = {
    # generic setup / recovery / disruption items
    1079: 9800.0,  # Rare Candy
    1080: 9000.0,  # Unfair Stamp
    1081: 8600.0,  # Enhanced Hammer
    1086: 7600.0,  # Buddy-Buddy Poffin
    1087: 5400.0,  # Hand Trimmer
    1097: 9000.0,  # Night Stretcher
    1121: 8400.0,  # Ultra Ball
    1122: 5600.0,  # Pokegear 3.0
    1123: 9200.0,  # Switch
    1137: 8800.0,  # Tool Scrapper
    1141: 9900.0,  # Premium Power Pro
    1142: 10400.0, # Fighting Gong
    1147: 8200.0,  # Jumbo Ice Cream
    1152: 6100.0,  # Poke Pad
}
_V18_ARCH_ITEM_TOTAL = {"mirror": 12, "dragapult": 18, "marnie": 17, "lucario": 15}
_V18_ARCH_ITEM_IDS = {
    "mirror": {1086,1087,1121,1122,1123,1147},
    "dragapult": {1080,1081,1086,1097,1121,1137,1152},
    "marnie": {1079,1080,1086,1097,1122,1137,1152},
    "lucario": {1086,1141,1142,1152},
}


def _v18_visible_ids(player: Any) -> list[int]:
    ids=[]
    for c in _as_list(_field(player,"discard",[])):
        ids.append(_card_id(c))
    for p in _board(player):
        ids.append(_card_id(p))
        ids.extend(_card_id(x) for x in _pre_evolutions(p))
        ids.extend(_card_id(x) for x in _tools(p))
        ids.extend(_card_id(x) for x in _energy_cards(p))
    st=_stadium_id_from_player(player) if False else 0
    return [x for x in ids if x]


def _v18_expected_items_in_hand(obs: Any) -> float:
    arch=_v6_arch(obs);op=_opponent(obs);h=_hand_count(op)
    total=_V18_ARCH_ITEM_TOTAL.get(arch, 14)
    item_ids=_V18_ARCH_ITEM_IDS.get(arch,set(_V18_ITEM_PRIORITIES))
    seen=sum(1 for x in _v18_visible_ids(op) if x in item_ids)
    remaining=max(0,total-seen)
    unknown=max(1,_deck_count(op)+_hand_count(op)+_prize_count(op))
    return min(float(h), h*remaining/unknown)


def _v18_setup_complete(obs: Any) -> bool:
    me=_me(obs);counts=_field_counts(me)
    return counts[CRUSTLE]>0 and any(_card_id(p)==CRUSTLE and _energy_count(p)>=2 for p in _board(me))


def _v18_immediate_priority(obs: Any) -> bool:
    # Do not trade away a winning attack, a one-hit engine KO, or emergency life.
    if _v8_attack_wins_now(obs):
        return True
    active=_active(_me(obs))
    if active is not None and _hp(active)<=30 and _damage(active)>0:
        return True
    return _v8_immediate_ko_target(obs) is not None and _attack_damage_budget(obs)>0


def _v18_disruption_window(obs: Any) -> float:
    if not _v18_setup_complete(obs) or _v18_immediate_priority(obs):
        return -3.0
    arch=_v6_arch(obs);op=_opponent(obs);me=_me(obs)
    value=0.0
    if _wall_established(obs): value+=1.2
    if _card_id(_active(me))==CRUSTLE: value+=0.8
    if _turn(obs)<=7: value+=0.9
    if _opponent_has_bypass_pressure(obs): value+=0.8
    if _v8_engine_count(op,arch)>0: value+=0.7
    if _prize_count(me)<=2: value-=0.8
    if _deck_count(me)<=7: value-=1.2
    return value


def _v18_hand_disruption_value(obs: Any,cid:int) -> float:
    me=_me(obs);op=_opponent(obs);mh=_hand_count(me);oh=_hand_count(op)
    window=_v18_disruption_window(obs)
    if window<0:return -12000.0
    arch=_v6_arch(obs);score=0.0
    if cid==HAND_TRIMMER:
        opp_cut=max(0,oh-5);self_cut=max(0,mh-5)
        net=opp_cut-_V18_CFG["self_trim_cost"]*self_cut
        if opp_cut<2 or net<1:return -7600.0
        score=_V18_CFG["trim_base"]+net*_V18_CFG["trim_per"]+window*900.0
        # Item before supporter can be a strong tempo play, but avoid deleting
        # our own large hand merely to trim one extra opposing card.
        if mh<=5 and oh>=8:score+=2200.0
    elif cid==XEROSICS_MACHINATIONS:
        cut=max(0,oh-3)
        if cut<2:return -7200.0
        score=_V18_CFG["xero_base"]+cut*_V18_CFG["xero_per"]+window*950.0
        if oh>=7:score+=1800.0
        if arch=="marnie" and _v8_engine_count(op,arch)>=2:score+=1200.0
    elif cid==ERI:
        expected=min(2.0,_v18_expected_items_in_hand(obs))
        if expected<0.55 or oh<4:return -6800.0
        score=_V18_CFG["eri_base"]+expected*_V18_CFG["eri_per"]+window*850.0
        if arch in {"marnie","lucario"} and _turn(obs)<=7:score+=1000.0
        if oh>=7:score+=700.0
    return score


def _v18_opponent_item_score(obs: Any,card: Any) -> float:
    cid=_card_id(card);arch=_v6_arch(obs);score=_V18_ITEM_PRIORITIES.get(cid,1200.0)
    op=_opponent(obs);ids=_opponent_ids(obs)
    if cid==1079 and arch=="marnie" and ids & {646,647}:score+=4200.0
    if cid in {1141,1142} and arch=="lucario":score+=5200.0
    if cid in {1086,1121} and _turn(obs)<=5:score+=2400.0
    if cid==1097 and len(_as_list(_field(op,"discard",[])))>=4:score+=2800.0
    if cid==1137 and any(_tools(p) for p in _board(_me(obs))):score+=3200.0
    if cid==1081 and sum(_energy_count(p) for p in _board(_me(obs)))>=2:score+=2600.0
    if cid==1123 and (_card_id(_active(op)) not in {121,648,678,306}):score+=2200.0
    if cid==1147 and any(_damage(p)>=60 and _energy_count(p)>=3 for p in _board(op)):score+=2600.0
    return score


def _v18_fan_destination_score(obs: Any,card: Any) -> float:
    cid=_card_id(card);arch=_v6_arch(obs)
    # The moved Energy should become as unproductive as possible.  Penalize
    # evolved/ready attackers and known acceleration/conversion engines.
    score=7600.0-_attacker_value(card)*0.42-_energy_count(card)*900.0
    if cid in {121,648,678,306,849,743,245,710}:score-=9000.0
    if cid in {112,120,647,677,333,66}:score-=3500.0
    safe={"marnie":{860,646,104},"dragapult":{235,140,343,1071,119},"lucario":{305,675,676},"mirror":{DWEBBLE}}.get(arch,set())
    if cid in safe:score+=4300.0
    if _energy_count(card)==0:score+=1400.0
    return score


def _tool_target_score(obs: Any,target: Any,tool_id:int) -> float:  # noqa: F811
    score=_V18_PREV_TOOL_SCORE(obs,target,tool_id)
    if tool_id==HANDHELD_FAN and target is not None and target is _active(_me(obs)):
        oa=_active(_opponent(obs))
        if _card_id(target)==CRUSTLE and oa is not None and _energy_count(oa)>=2 and len(_bench(_opponent(obs)))>0:
            score+=_V18_CFG["fan_attach"]
            if _opponent_active_bypass(obs):score+=1200.0
    return score


def _score_main_option(obs: Any,option: Any) -> float:  # noqa: F811
    score=_V18_PREV_MAIN_SCORE(obs,option)
    typ=_int(_field(option,"type",-1),-1);cid=_card_id(_option_card(obs,option));arch=_v6_arch(obs)
    if typ==OPTION_PLAY and cid in {ERI,XEROSICS_MACHINATIONS,HAND_TRIMMER}:
        dv=_v18_hand_disruption_value(obs,cid)
        if dv>-7000:score=max(score,dv)
    # Stadium denial is interaction too: erase opposing damage/HP/lock stadiums
    # before they obtain another full turn of value.
    if typ==OPTION_PLAY and cid in STADIUM_IDS:
        cur=_stadium_id(obs)
        hostile=(arch=="marnie" and cur==1259) or (arch=="lucario" and cur==1252) or (arch=="dragapult" and cur in {1256,1260})
        if hostile:score+=_V18_CFG["stadium_denial"]
    # Fan is most valuable on the active Crustle just before a powered attacker
    # is forced to hit into it.
    if typ==OPTION_ATTACH and cid==HANDHELD_FAN:
        target=_option_target(obs,option);oa=_active(_opponent(obs))
        if target is _active(_me(obs)) and _card_id(target)==CRUSTLE and oa is not None and _energy_count(oa)>=2:
            score+=_V18_CFG["fan_attach"]
    return score


def _score_card_selection(obs: Any,option: Any,context:int,occurrence:int) -> float:  # noqa: F811
    card=_option_card(obs,option);owner=_int(_field(option,"playerIndex",_your_index(obs)),_your_index(obs));effect=_effect_id(_field(obs,"select",{}))
    if effect==ERI and context in {CONTEXT_DISCARD,CONTEXT_DISCARD_CARD_OR_ATTACHED} and owner!=_your_index(obs):
        return _v18_opponent_item_score(obs,card)-occurrence*0.01
    if effect==HANDHELD_FAN and context==CONTEXT_ATTACH_FROM and owner!=_your_index(obs):
        return _v18_fan_destination_score(obs,card)
    return _V18_PREV_CARD_SCORE(obs,option,context,occurrence)


def _score_sub_option(obs: Any,option: Any,context:int,occurrence:int) -> float:  # noqa: F811
    effect=_effect_id(_field(obs,"select",{}));typ=_int(_field(option,"type",-1),-1)
    if effect==HANDHELD_FAN and context==CONTEXT_SWITCH_ENERGY and typ in {OPTION_ENERGY,OPTION_ENERGY_CARD}:
        c=_option_card(obs,option);cid=_card_id(c)
        score=4200.0
        # Remove scarce/special Energy first.  Rock Fighting and other special
        # Energy are harder for the opponent to replace than basic Energy.
        if cid not in {1,2,3,4,5,6,7,8,9,10}:score+=4200.0
        if cid in {20,14,18,11}:score+=1800.0
        return score
    return _V18_PREV_SUB_SCORE(obs,option,context,occurrence)

# Diagnostics only; decisions remain deterministic.
_V18_DIAGNOSTICS={"mode":_V18_MODE,"disruption_options":0,"eri_discards":0,"fan_moves":0}
_V18_PREV_AGENT=agent

def agent(observation: dict) -> list[int]:  # noqa: F811
    sel=_field(observation,"select")
    if sel is not None:
        try:
            ctx=_int(_field(sel,"context",-1),-1);effect=_effect_id(sel)
            if ctx==CONTEXT_MAIN:
                for o in _as_list(_field(sel,"option",[])):
                    if _card_id(_option_card(observation,o)) in {ERI,XEROSICS_MACHINATIONS,HAND_TRIMMER}:
                        _V18_DIAGNOSTICS["disruption_options"]+=1;break
            elif effect==ERI:_V18_DIAGNOSTICS["eri_discards"]+=1
            elif effect==HANDHELD_FAN:_V18_DIAGNOSTICS["fan_moves"]+=1
        except Exception:pass
    return _V18_PREV_AGENT(observation)
