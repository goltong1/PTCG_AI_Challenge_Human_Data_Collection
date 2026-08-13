"""Guarded matchup heuristics retained through the v35 full-flow league.

The source deck shares 36/60 cards with Tera Box, so this engine transfers only
patterns that are expressed by cards present in the fixed Tera Box list.  Every
intervention is a legal public-state action and has a strict tactical guard:
complete an attack, establish a free in-hand attacker, or veto a Boss without
an immediate prize.  Unsupported states retain the v29 action.  The proposed
Lucario Kangaskhan opening was 162/1000 versus the retained 166/1000 baseline,
so Lucario keeps only the exact public-state Clefairy knockout conversion.
"""
from __future__ import annotations

import json
import os

from cg.api import all_attack, all_card_data


MAIN = 0
TO_BENCH = 5
TO_FIELD = 6
TO_HAND = 7
LOOK = 24
PLAY = 7
ATTACH = 8
ABILITY = 10
ATTACK = 13
RETREAT = 12
END = 14
ACTIVE = 4
BENCH = 5
RAINBOW = 10
COLORLESS = 0

TEAL = 96
WELLSPRING = 108
CORNERSTONE = 117
CLEFAIRY = 272
KANGASKHAN = 756
MEOWTH = 1071
TERA_ORB = 1127
ULTRA_BALL = 1121
BOSS = 1182
XEROSIC = 1197
LILLIE = 1227
AREA_ZERO = 1250
PRISM_CARD = 16
DRAG_LINE = {119, 120, 121}

DEFAULT_RULES = {
    # v30 retained behavior.
    "drag_old_search": True,
    "drag_exact_ko_switch": True,
    # v31 candidates are enabled per variant and only retained after gates.
    "drag_line_promotion": False,
    "drag_exact_bench_fill": False,
    "drag_prism_reserve": False,
    "drag_early_clefairy": False,
    "drag_core_clefairy": False,
    "drag_kanga_opening": False,
    "drag_clefairy_stage_guard": True,
    "drag_no_clef_overattach": False,
    "drag_preserve_second_prism": False,
    "drag_clef_survival_guard": False,
    "drag_latias_bridge": False,
    "drag_boss_denial": False,
    "drag_clefairy_near_ko_veto": True,
}
try:
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matchup_v32_config.json")
    RULES = dict(DEFAULT_RULES)
    RULES.update(json.load(open(_config_path, encoding="utf-8")))
except Exception:
    RULES = dict(DEFAULT_RULES)

CARDS = {int(card.cardId): card for card in all_card_data()}
ATTACKS = {int(attack.attackId): attack for attack in all_attack()}

STATS = {
    "decisions": 0,
    "overrides": 0,
    "boss_vetoes": 0,
    "forced_boss_kos": 0,
    "xerosic_forces": 0,
    "tempo_rescues": 0,
    "attachment_completions": 0,
    "search_overrides": 0,
    "attack_rescues": 0,
    "clefairy_deploys": 0,
    "clefairy_prism_reserves": 0,
    "clefairy_bench_fills": 0,
    "clefairy_line_promotions": 0,
    "clefairy_boss_denials": 0,
    "kanga_searches": 0,
    "kanga_progress_attachments": 0,
    "kanga_opening_switches": 0,
    "clefairy_stage_retreats": 0,
    "clefairy_overattach_redirects": 0,
    "clefairy_second_prism_saves": 0,
    "clefairy_near_ko_vetoes": 0,
    "records": [],
}

MEM = {"switch_target": None, "boss_target": None, "kanga_attacks": 0, "last_kanga_attack_turn": -1}


def reset():
    for key in STATS:
        STATS[key] = [] if key == "records" else 0
    MEM.update({"switch_target": None, "boss_target": None, "kanga_attacks": 0,
                "last_kanga_attack_turn": -1})


def _record(item):
    STATS["records"].append(item)
    if len(STATS["records"]) > 200:
        del STATS["records"][:-200]


def _cid(card):
    try:
        return int((card or {}).get("id") or 0)
    except Exception:
        return 0


def _players(raw):
    current = raw.get("current") or {}
    players = current.get("players") or []
    try:
        me = int(current.get("yourIndex") or 0)
    except Exception:
        me = 0
    return current, players, me


def _source(raw, option):
    current, players, me = _players(raw)
    try:
        owner = option.get("playerIndex")
        owner = me if owner is None else int(owner)
        area = int(option.get("area", 2))
        zones = {
            1: (raw.get("select") or {}).get("deck") or [],
            2: players[owner].get("hand") or [],
            3: players[owner].get("discard") or [],
            4: players[owner].get("active") or [],
            5: players[owner].get("bench") or [],
            7: current.get("stadium") or [],
            12: current.get("looking") or [],
        }
        return zones.get(area, [])[int(option.get("index"))]
    except Exception:
        return None


def _target(raw, option):
    _current, players, me = _players(raw)
    try:
        area = int(option.get("inPlayArea"))
        zone = players[me].get("active") or [] if area == ACTIVE else players[me].get("bench") or [] if area == BENCH else []
        return zone[int(option.get("inPlayIndex"))]
    except Exception:
        return None


def _board(raw):
    _current, players, me = _players(raw)
    if len(players) < 2:
        return {}, [], []
    mine = players[me]
    board = list(mine.get("active") or []) + list(mine.get("bench") or [])
    hand = list(mine.get("hand") or [])
    return mine, board, hand


def _opponent_board(raw):
    _current, players, me = _players(raw)
    if len(players) < 2:
        return [], []
    opponent = players[1 - me]
    return list(opponent.get("active") or []), list(opponent.get("bench") or [])


def _opponent_has(raw, card_id):
    active, bench = _opponent_board(raw)
    return any(_cid(card) == card_id for card in active + bench)


def _opponent_has_any(raw, card_ids):
    active, bench = _opponent_board(raw)
    wanted = set(card_ids)
    return any(_cid(card) in wanted for card in active + bench)


def _energy_type(card):
    cid = _cid(card)
    if cid == 16:
        return RAINBOW
    data = CARDS.get(cid)
    try:
        return int(data.energyType)
    except Exception:
        return 1 if cid == 1 else None


def _pool(pokemon, added=None):
    values = pokemon.get("energies")
    if values is None:
        values = [RAINBOW for _ in pokemon.get("energyCards") or []]
    result = [int(value) for value in values]
    if added is not None:
        result.append(int(added))
    return result


def _missing_for_attack(pokemon, attack, added=None):
    pool = _pool(pokemon, added)
    missing = 0
    for requirement in attack.energies:
        requirement = int(requirement)
        if requirement == COLORLESS:
            if pool:
                pool.pop(0)
            else:
                missing += 1
        else:
            index = next((i for i, value in enumerate(pool) if value in {requirement, RAINBOW}), None)
            if index is None:
                missing += 1
            else:
                pool.pop(index)
    return missing


def _missing(pokemon, added=None):
    data = CARDS.get(_cid(pokemon))
    if data is None:
        return 99
    values = [_missing_for_attack(pokemon, ATTACKS[int(aid)], added) for aid in data.attacks or [] if int(aid) in ATTACKS]
    return min(values) if values else 99


def _ready_damage(pokemon):
    data = CARDS.get(_cid(pokemon))
    if data is None:
        return 0
    damage = 0
    for attack_id in data.attacks or []:
        attack = ATTACKS.get(int(attack_id))
        if attack is not None and _missing_for_attack(pokemon, attack) == 0:
            damage = max(damage, int(attack.damage or 0))
    # Conservative public-state lower bounds for variable attacks.
    if _cid(pokemon) == TEAL and _missing(pokemon) == 0:
        damage = max(damage, 90)
    if _cid(pokemon) == CLEFAIRY and _missing(pokemon) == 0:
        damage = max(damage, 80)
    return damage


def _same_source(raw, left, right):
    return (
        int(left.get("area", 2)) == int(right.get("area", 2))
        and int(left.get("index", -1)) == int(right.get("index", -1))
        and _cid(_source(raw, left)) == _cid(_source(raw, right))
    )


def _immediate_boss_ko(raw):
    current, players, me = _players(raw)
    if len(players) < 2:
        return False
    mine = players[me]
    opponent = players[1 - me]
    active = (mine.get("active") or [None])[0]
    for target in opponent.get("bench") or []:
        if active and _can_knock_out(raw, active, target):
            return True
    return False


def _can_knock_out(raw, attacker, target):
    if _missing(attacker) != 0:
        return False
    damage = _ready_damage(attacker)
    attacker_id = _cid(attacker)
    target_id = _cid(target)
    own_active, own_bench = [], []
    _current, players, me = _players(raw)
    if len(players) == 2:
        mine, opponent = players[me], players[1 - me]
        own_active = list(mine.get("active") or [])
        own_bench = list(mine.get("bench") or [])
        opponent_bench = list(opponent.get("bench") or [])
    else:
        opponent_bench = []
    if attacker_id == CLEFAIRY:
        damage = 20 * (1 + len(own_bench) + len(opponent_bench))
        # Fairy Zone gives Dragon Pokémon Psychic weakness; Mega Lucario has
        # printed Psychic weakness already.
        target_data = CARDS.get(target_id)
        psychic_weak = bool(target_data is not None and int(target_data.weakness or -1) == 5)
        if target_id in {119, 120, 121} or psychic_weak:
            damage *= 2
    elif attacker_id == TEAL:
        target_energies = len(target.get("energyCards") or target.get("energies") or [])
        damage = 30 + 30 * (len(_pool(attacker)) + target_energies)
    try:
        return damage >= int(target.get("hp") or 0) > 0
    except Exception:
        return False


def _clefairy_damage(raw, target, extra_own_bench=0):
    """Exact public Full Moon Rondo damage for the current board."""
    _current, players, me = _players(raw)
    if len(players) != 2 or target is None:
        return 0
    own_bench = list(players[me].get("bench") or [])
    opponent_bench = list(players[1 - me].get("bench") or [])
    damage = 20 * (1 + len(own_bench) + len(opponent_bench) + int(extra_own_bench))
    target_id = _cid(target)
    target_data = CARDS.get(target_id)
    psychic_weak = bool(target_data is not None and int(target_data.weakness or -1) == 5)
    if target_id in DRAG_LINE or psychic_weak:
        damage *= 2
    return damage


def _own_has(raw, card_id):
    _mine, board, hand = _board(raw)
    return any(_cid(card) == card_id for card in board + hand)


def _drag_active_target(raw):
    active, _bench = _opponent_board(raw)
    return active[0] if active else None


def _ready_clefairy(raw):
    _mine, board, _hand = _board(raw)
    return next((card for card in board if _cid(card) == CLEFAIRY and _missing(card) == 0), None)


def _clefairy_conversion_safe(raw):
    """Avoid donating Clefairy into an already-ready second Phantom Dive."""
    _current, players, me = _players(raw)
    if len(players) != 2:
        return True
    mine, opponent = players[me], players[1 - me]
    if len(mine.get("prize") or []) <= 2:
        return True
    return not any(_cid(card) == 121 and _missing(card) == 0 for card in opponent.get("bench") or [])


def _high_hp_basic(card):
    data = CARDS.get(_cid(card))
    return bool(data is not None and data.basic and int(data.hp or 0) >= 190)


def _drag_exact_bench_fill(raw, base, matchup):
    """Add one safe body only when its +40 weakness damage creates a KO."""
    if matchup != "dragapult" or not RULES.get("drag_exact_bench_fill"):
        return base
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN:
        return base
    target = _drag_active_target(raw)
    clefairy = _ready_clefairy(raw)
    if target is None or _cid(target) not in DRAG_LINE or clefairy is None:
        return base
    _mine, board, _hand = _board(raw)
    active = board[0] if board else None
    # From the Bench this tactic is executable only with Latias free retreat.
    if _cid(active) != CLEFAIRY and not any(_cid(card) == 184 for card in board):
        return base
    now = _clefairy_damage(raw, target)
    hp = int(target.get("hp") or 0)
    if not (now < hp <= _clefairy_damage(raw, target, 1)):
        return base
    priorities = {TEAL: 10, 184: 9, WELLSPRING: 8, CORNERSTONE: 8, KANGASKHAN: 7, CLEFAIRY: 6, 140: 5}
    candidates = []
    for index, option in enumerate(select.get("option") or []):
        if int(option.get("type", -1)) != PLAY:
            continue
        card = _source(raw, option)
        if not _high_hp_basic(card):
            continue
        candidates.append((priorities.get(_cid(card), 1), -index, index, _cid(card)))
    if not candidates:
        return base
    _priority, _neg_index, index, card_id = max(candidates)
    if base == [index]:
        return base
    STATS["overrides"] += 1
    STATS["clefairy_bench_fills"] += 1
    _record({"matchup": matchup, "family": "clefairy_exact_bench_fill", "chosen": index,
             "body": card_id, "damage_before": now, "damage_after": now + 40, "target_hp": hp})
    return [index]


def _drag_prism_reserve(raw, base, matchup):
    """Reserve the first universal energy for Clefairy's Psychic requirement."""
    if matchup != "dragapult" or not RULES.get("drag_prism_reserve"):
        return base
    options = (raw.get("select") or {}).get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0]); chosen = options[base_index]
    except Exception:
        return base
    if int(chosen.get("type", -1)) != ATTACH or _cid(_source(raw, chosen)) != PRISM_CARD:
        return base
    _mine, board, _hand = _board(raw)
    clefairy = next((card for card in board if _cid(card) == CLEFAIRY), None)
    if clefairy is None or RAINBOW in _pool(clefairy) or _missing(clefairy) == 0:
        return base
    base_target = _target(raw, chosen)
    if base_target is None:
        return base
    # Never break an attack that this attachment completes immediately.
    base_before = _missing(base_target)
    base_after = _missing(base_target, RAINBOW)
    if base_before > 0 and base_after == 0:
        return base
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != ATTACH or not _same_source(raw, chosen, option):
            continue
        target = _target(raw, option)
        if _cid(target) != CLEFAIRY or _missing(target, RAINBOW) >= _missing(target):
            continue
        if index == base_index:
            return base
        STATS["overrides"] += 1
        STATS["clefairy_prism_reserves"] += 1
        _record({"matchup": matchup, "family": "clefairy_prism_reserve", "base": base_index,
                 "chosen": index, "clefairy_missing_before": _missing(clefairy)})
        return [index]
    return base


def _drag_early_deploy(raw, base, matchup):
    if matchup != "dragapult" or not RULES.get("drag_early_clefairy"):
        return base
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN or not _opponent_has_any(raw, DRAG_LINE):
        return base
    current = raw.get("current") or {}
    if int(current.get("turn") or 0) > 6:
        return base
    _mine, board, _hand = _board(raw)
    if any(_cid(card) == CLEFAIRY for card in board):
        return base
    # Require either the scarce Psychic out or free-retreat bridge to be known.
    if not (_own_has(raw, PRISM_CARD) or any(_cid(card) == 184 for card in board)):
        return base
    try:
        base_type = int((select.get("option") or [])[int(base[0])].get("type", -1))
    except Exception:
        return base
    if base_type != END:
        return base
    for index, option in enumerate(select.get("option") or []):
        if int(option.get("type", -1)) == PLAY and _cid(_source(raw, option)) == CLEFAIRY:
            STATS["overrides"] += 1
            STATS["clefairy_deploys"] += 1
            _record({"matchup": matchup, "family": "clefairy_early_deploy", "chosen": index})
            return [index]
    return base


def _drag_kanga_attachment(raw, base, matchup):
    """Charge the 300-HP opening attacker without breaking a completed attack."""
    if matchup != "dragapult" or not RULES.get("drag_kanga_opening"):
        return base
    current = raw.get("current") or {}
    if int(current.get("turn") or 0) > 5:
        return base
    options = (raw.get("select") or {}).get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0]); chosen = options[base_index]
    except Exception:
        return base
    if int(chosen.get("type", -1)) != ATTACH:
        return base
    source_card = _source(raw, chosen)
    # Kangaskhan pays only Colorless.  Keep the four universal cards available
    # for Clefairy's mandatory Psychic requirement and use abundant Grass for
    # opening progress instead.
    if _cid(source_card) == PRISM_CARD:
        return base
    energy_type = _energy_type(source_card)
    base_target = _target(raw, chosen)
    if energy_type is None or base_target is None:
        return base
    _mine, board, _hand = _board(raw)
    kanga = next((card for card in board if _cid(card) == KANGASKHAN and _missing(card) > 0), None)
    if kanga is None:
        return base
    base_before, base_after = _missing(base_target), _missing(base_target, energy_type)
    # An attack completed this turn is more valuable than opening progress.
    if base_before > 0 and base_after == 0:
        return base
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != ATTACH or not _same_source(raw, chosen, option):
            continue
        target = _target(raw, option)
        if _cid(target) != KANGASKHAN:
            continue
        before, after = _missing(target), _missing(target, energy_type)
        if after >= before or index == base_index:
            return base
        STATS["overrides"] += 1
        STATS["kanga_progress_attachments"] += 1
        _record({"matchup": matchup, "family": "kanga_opening_attach", "base": base_index,
                 "chosen": index, "missing_before": before, "missing_after": after})
        return [index]
    return base


def _drag_preserve_second_prism(raw, base, matchup):
    """Use Grass for Clefairy's colorless slot when Psychic is already live."""
    if matchup != "dragapult" or not RULES.get("drag_preserve_second_prism"):
        return base
    options = (raw.get("select") or {}).get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0]); chosen = options[base_index]
    except Exception:
        return base
    target = _target(raw, chosen)
    if (int(chosen.get("type", -1)) != ATTACH or _cid(_source(raw, chosen)) != PRISM_CARD
            or _cid(target) != CLEFAIRY or RAINBOW not in _pool(target) or _missing(target) != 1):
        return base
    for index, option in enumerate(options):
        alt_target = _target(raw, option)
        if (int(option.get("type", -1)) == ATTACH and _cid(_source(raw, option)) == 1
                and _cid(alt_target) == CLEFAIRY and _missing(alt_target, 1) == 0):
            STATS["overrides"] += 1
            STATS["clefairy_second_prism_saves"] += 1
            _record({"matchup": matchup, "family": "clefairy_second_prism_save",
                     "base": base_index, "chosen": index})
            return [index]
    return base


def _drag_clef_overattach(raw, base, matchup):
    """Do not spend a third+ energy on an already attack-ready Clefairy."""
    if matchup != "dragapult" or not RULES.get("drag_no_clef_overattach"):
        return base
    options = (raw.get("select") or {}).get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0]); chosen = options[base_index]
    except Exception:
        return base
    if int(chosen.get("type", -1)) != ATTACH:
        return base
    base_target = _target(raw, chosen)
    energy_type = _energy_type(_source(raw, chosen))
    if _cid(base_target) != CLEFAIRY or energy_type is None or _missing(base_target) != 0:
        return base
    priorities = {KANGASKHAN: 7, TEAL: 6, WELLSPRING: 5, CORNERSTONE: 5,
                  184: 4, 140: 3, 1071: 2}
    candidates = []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != ATTACH or not _same_source(raw, chosen, option):
            continue
        target = _target(raw, option)
        if target is None or _cid(target) == CLEFAIRY:
            continue
        before, after = _missing(target), _missing(target, energy_type)
        if after >= before:
            continue
        candidates.append((int(after == 0), priorities.get(_cid(target), 1), before - after,
                           -index, index, _cid(target)))
    if not candidates:
        return base
    best = max(candidates); index, target_id = best[4], best[5]
    STATS["overrides"] += 1
    STATS["clefairy_overattach_redirects"] += 1
    _record({"matchup": matchup, "family": "clefairy_overattach_redirect", "base": base_index,
             "chosen": index, "target": target_id, "energy": _cid(_source(raw, chosen))})
    return [index]


def _drag_kanga_switch(raw, base, matchup):
    if matchup != "dragapult" or not RULES.get("drag_kanga_opening"):
        return base
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN or int((raw.get("current") or {}).get("turn") or 0) > 6:
        return base
    _mine, board, _hand = _board(raw)
    active = board[0] if board else None
    kanga = next((card for card in board[1:] if _cid(card) == KANGASKHAN and _missing(card) == 0), None)
    target = _drag_active_target(raw)
    if active is None or kanga is None or target is None or _cid(active) == KANGASKHAN:
        return base
    # Preserve an immediate Clefairy conversion KO or any other current KO.
    if _can_knock_out(raw, active, target):
        return base
    if not _can_knock_out(raw, kanga, target) and _missing(active) == 0:
        return base
    for index, option in enumerate(select.get("option") or []):
        if int(option.get("type", -1)) == RETREAT:
            MEM["switch_target"] = KANGASKHAN
            STATS["overrides"] += 1
            STATS["kanga_opening_switches"] += 1
            _record({"matchup": matchup, "family": "kanga_opening_switch", "chosen": index,
                     "from": _cid(active), "target": _cid(target)})
            return [index]
    return base


def _drag_clefairy_stage_guard(raw, base, matchup):
    """Open with ready Kangaskhan and preserve Clefairy for Dragapult ex.

    In the 1,000-game v32 baseline, Kangaskhan as the first attacker converted
    far more often than Clefairy as the first attacker.  Clefairy's repeated
    attacks were valuable, but spending its 190-HP body on Dreepy/Drakloak was
    usually followed by Phantom Dive.  This rule acts only before Kangaskhan's
    first attack, only when both attackers are already legal, and never gives
    up a Clefairy-only knockout or the retained near-KO Fezandipiti line.
    """
    if (matchup != "dragapult" or not RULES.get("drag_kanga_opening")
            or not RULES.get("drag_clefairy_stage_guard") or MEM["kanga_attacks"] > 0):
        return base
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN or not isinstance(base, list) or len(base) != 1:
        return base
    options = select.get("option") or []
    try:
        base_type = int(options[int(base[0])].get("type", -1))
    except Exception:
        return base
    if base_type not in {ATTACK, RETREAT, END}:
        return base
    _mine, board, _hand = _board(raw)
    active = board[0] if board else None
    kanga = next((card for card in board[1:] if _cid(card) == KANGASKHAN and _missing(card) == 0), None)
    target = _drag_active_target(raw)
    if active is None or kanga is None or target is None or _cid(active) != CLEFAIRY or _missing(active) != 0:
        return base
    clefairy_ko = _can_knock_out(raw, active, target)
    kanga_ko = _can_knock_out(raw, kanga, target)
    if (_cid(target) == 121 and clefairy_ko) or (clefairy_ko and not kanga_ko):
        return base
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != RETREAT:
            continue
        MEM["switch_target"] = KANGASKHAN
        if base == [index]:
            return base
        STATS["overrides"] += 1
        STATS["kanga_opening_switches"] += 1
        STATS["clefairy_stage_retreats"] += 1
        _record({"matchup": matchup, "family": "clefairy_stage_retreat",
                 "base": int(base[0]), "chosen": index, "target": _cid(target),
                 "clefairy_ko": clefairy_ko, "kanga_ko": kanga_ko})
        return [index]
    return base


def _drag_clefairy_near_ko_veto(raw, base, matchup):
    """Keep a ready Clefairy Active when it sets Fezandipiti to 10--30 HP.

    The 2026-08-09 full win retreated a ready Clefairy, promoted a zero-energy
    Kangaskhan, and ended the turn against a 210-HP Fezandipiti ex.  Full Moon
    Rondo already dealt 200 in that public state, after which Munkidori could
    take the two-Prize knockout with Adrena-Brain.  This guard runs only after
    the base policy has finished its useful setup and proposes RETREAT or END.
    """
    if matchup != "dragapult" or not RULES.get("drag_clefairy_near_ko_veto"):
        return base
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN or not isinstance(base, list) or len(base) != 1:
        return base
    options = select.get("option") or []
    try:
        base_type = int(options[int(base[0])].get("type", -1))
    except Exception:
        return base
    if base_type not in {RETREAT, END}:
        return base
    _mine, board, _hand = _board(raw)
    active = board[0] if board else None
    target = _drag_active_target(raw)
    if (active is None or target is None or _cid(active) != CLEFAIRY
            or _cid(target) != 140 or _missing(active) != 0):
        return base
    damage = _clefairy_damage(raw, target)
    hp = int(target.get("hp") or 0)
    if hp <= 0 or damage >= hp or hp - damage > 30:
        return base
    for index, option in enumerate(options):
        if int(option.get("type", -1)) == ATTACK and int(option.get("attackId") or 0) == 371:
            STATS["overrides"] += 1
            STATS["clefairy_near_ko_vetoes"] += 1
            _record({"matchup": matchup, "family": "clefairy_near_ko_veto",
                     "base": int(base[0]), "chosen": index, "damage": damage,
                     "target_hp": hp})
            return [index]
    return base


def _select_attack(raw, options):
    current, players, me = _players(raw)
    opponent_hp = 0
    try:
        opponent_hp = int(((players[1 - me].get("active") or [None])[0] or {}).get("hp") or 0)
    except Exception:
        pass
    candidates = []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != ATTACK:
            continue
        attack = ATTACKS.get(int(option.get("attackId") or -1))
        damage = int(attack.damage or 0) if attack is not None else 0
        ko = damage >= opponent_hp > 0
        candidates.append((int(ko), damage, index))
    return max(candidates)[2] if candidates else None


def _alternative_after_boss_veto(raw, options, matchup):
    current, players, me = _players(raw)
    mine = players[me] if len(players) > me else {}
    opponent = players[1 - me] if len(players) == 2 else {}
    own_hand = len(mine.get("hand") or [])
    opp_hand = int(opponent.get("handCount") or len(opponent.get("hand") or []))
    scored = []
    for index, option in enumerate(options):
        typ = int(option.get("type", -1))
        source = _cid(_source(raw, option))
        score = -100000
        if typ == PLAY and source == LILLIE and own_hand <= 5:
            score = 9500 - own_hand * 300
        elif typ == PLAY and source == XEROSIC and opp_hand >= (7 if matchup == "marnie" else 9):
            score = 9000 + opp_hand * 100
        elif typ == ABILITY:
            score = 8200
        elif typ == ATTACH:
            target = _target(raw, option)
            energy = _energy_type(_source(raw, option))
            if target is not None and energy is not None:
                before, after = _missing(target), _missing(target, energy)
                if after < before:
                    score = 7600 + (before - after) * 800
        elif typ == ATTACK:
            attack = ATTACKS.get(int(option.get("attackId") or -1))
            score = 7000 + int(attack.damage or 0) if attack is not None else 7000
        elif typ == END:
            score = 100
        if source == BOSS:
            score = -100000
        scored.append((score, -index, index))
    best = max(scored) if scored else None
    return best[2] if best is not None and best[0] > -100000 else None


def _required_setup(raw, matchup):
    current, _players_, _me = _players(raw)
    _mine, board, hand = _board(raw)
    known = {_cid(card) for card in board + hand}
    turn = int(current.get("turn") or 0)
    if turn > 6:
        return None
    if TEAL not in known and matchup != "dragapult":
        return TEAL
    if matchup == "dragapult" and _opponent_has(raw, 121):
        teal = next((card for card in board if _cid(card) == TEAL), None)
        if teal is not None and _missing(teal) == 0 and CLEFAIRY not in known:
            return CLEFAIRY
    if matchup == "lucario":
        # The 60-game gate showed that forced setup/search did not raise the
        # matchup as a whole.  Keep the v29 setup and intervene only once a
        # Clefairy knockout is already available on the public board.
        return None
    if matchup in {"marnie", "lopunny", "zoroark"} and KANGASKHAN not in known:
        return KANGASKHAN
    return None


def _tempo_rescue(raw, base, matchup):
    select = raw.get("select") or {}
    options = select.get("option") or []
    if int(select.get("context", -1)) != MAIN or not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0])
        if int(options[base_index].get("type", -1)) != END:
            return base
    except Exception:
        return base
    attack_index = _select_attack(raw, options)
    if attack_index is not None:
        STATS["overrides"] += 1
        STATS["attack_rescues"] += 1
        _record({"matchup": matchup, "family": "attack_rescue", "chosen": attack_index})
        return [attack_index]
    target = _required_setup(raw, matchup)
    if target is None:
        return base
    candidates = []
    for index, option in enumerate(options):
        typ = int(option.get("type", -1))
        source = _cid(_source(raw, option))
        if typ == PLAY and source == target:
            candidates.append((12000, index, "play"))
        elif typ == PLAY and source == TERA_ORB and target in {TEAL, KANGASKHAN}:
            candidates.append((10000, index, "tera_orb"))
        elif typ == ATTACH:
            pokemon = _target(raw, option)
            energy = _energy_type(_source(raw, option))
            if pokemon is not None and energy is not None and _missing(pokemon, energy) < _missing(pokemon):
                priority = 2 if _cid(pokemon) == target else 1
                candidates.append((7000 + priority * 500, index, "progress_attach"))
    if not candidates:
        return base
    _score, index, reason = max(candidates)
    STATS["overrides"] += 1
    STATS["tempo_rescues"] += 1
    _record({"matchup": matchup, "family": "tempo", "target": target, "chosen": index, "reason": reason})
    return [index]


def _attachment_completion(raw, base, matchup):
    if matchup not in {"dragapult", "marnie", "lopunny", "zoroark"}:
        return base
    options = (raw.get("select") or {}).get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0])
        chosen = options[base_index]
        if int(chosen.get("type", -1)) != ATTACH:
            return base
    except Exception:
        return base
    source_energy = _energy_type(_source(raw, chosen))
    base_target = _target(raw, chosen)
    if source_energy is None or base_target is None:
        return base
    base_before = _missing(base_target)
    base_after = _missing(base_target, source_energy)
    candidates = []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != ATTACH or not _same_source(raw, chosen, option):
            continue
        target = _target(raw, option)
        if target is None:
            continue
        before, after = _missing(target), _missing(target, source_energy)
        if after >= before:
            continue
        cid = _cid(target)
        completion = before > 0 and after == 0
        priority = 0
        if matchup == "dragapult":
            priority = {CLEFAIRY: 7 if completion and _opponent_has(raw, 121) else 0}.get(cid, 0)
        elif matchup == "lucario":
            # Clefairy is promoted only when this exact attachment makes it
            # attack-ready; this avoids the failed v29 forced-setup behavior.
            priority = {CLEFAIRY: 6 if completion else 0}.get(cid, 0)
        else:
            priority = {TEAL: 5, KANGASKHAN: 4}.get(cid, 0)
        if priority <= 0:
            continue
        candidates.append((int(completion), priority, before - after, -index, index, cid, before, after))
    if not candidates:
        return base
    best = max(candidates)
    best_index, best_cid, best_before, best_after = best[4], best[5], best[6], best[7]
    if best_index == base_index:
        return base
    # Only redirect when the alternative completes an attack that the baseline
    # does not, or when baseline spends energy on an already-ready attacker.
    safe = (best_after == 0 and base_after > 0) or (base_before == 0 and best_after < best_before)
    if not safe:
        return base
    STATS["overrides"] += 1
    STATS["attachment_completions"] += 1
    _record({"matchup": matchup, "family": "attach", "base": base_index, "chosen": best_index, "target": best_cid})
    return [best_index]


def _search_choice(raw, base, matchup):
    select = raw.get("select") or {}
    context = int(select.get("context", -1))
    if context not in {TO_BENCH, TO_FIELD, TO_HAND, LOOK} or not isinstance(base, list) or len(base) != 1:
        return base
    options = select.get("option") or []
    _mine, board, hand = _board(raw)
    known = {_cid(card) for card in board + hand}
    turn = int(((raw.get("current") or {}).get("turn") or 0))
    if turn > 6:
        return base
    priorities = []
    if matchup == "dragapult":
        priorities = []
        teal = next((card for card in board if _cid(card) == TEAL), None)
        if RULES.get("drag_old_search") and _opponent_has(raw, 121) and teal is not None and _missing(teal) == 0:
            priorities = [CLEFAIRY]
        if RULES.get("drag_early_clefairy") and _opponent_has_any(raw, DRAG_LINE):
            has_core = any(_cid(card) in {TEAL, KANGASKHAN} for card in board)
            psychic_out = any(_cid(card) == PRISM_CARD for card in hand) or any(
                RAINBOW in _pool(card) for card in board if _cid(card) == CLEFAIRY
            )
            if has_core and (psychic_out or turn <= 4):
                priorities = [CLEFAIRY] + priorities
        if RULES.get("drag_core_clefairy") and _opponent_has_any(raw, DRAG_LINE):
            # Replay/policy residual: Kangaskhan attacks first; Clefairy is
            # searched only after the free-retreat and Teal draw core exists.
            board_ids = {_cid(card) for card in board}
            if {KANGASKHAN, 184, TEAL}.issubset(board_ids):
                priorities = [CLEFAIRY] + priorities
        if RULES.get("drag_kanga_opening") and _opponent_has_any(raw, DRAG_LINE):
            board_ids = {_cid(card) for card in board}
            if turn <= 4 and KANGASKHAN not in known:
                priorities = [KANGASKHAN] + priorities
            elif (MEM["kanga_attacks"] > 0 and KANGASKHAN in board_ids
                  and ({184, TEAL} & board_ids) and CLEFAIRY not in known):
                priorities = [CLEFAIRY] + priorities
        if RULES.get("drag_latias_bridge"):
            clefairy = next((card for card in board if _cid(card) == CLEFAIRY), None)
            if clefairy is not None and _missing(clefairy) <= 1 and not any(_cid(card) == 184 for card in board + hand):
                priorities = [184] + priorities
    elif matchup == "lucario":
        return base
    elif matchup in {"marnie", "lopunny", "zoroark"}:
        priorities = [TEAL, KANGASKHAN]
    else:
        return base
    desired = next((cid for cid in priorities if cid not in known), None)
    if desired is None:
        return base
    for index, option in enumerate(options):
        if _cid(_source(raw, option)) == desired:
            if int(base[0]) == index:
                return base
            STATS["overrides"] += 1
            STATS["search_overrides"] += 1
            if matchup == "dragapult" and desired == KANGASKHAN:
                STATS["kanga_searches"] += 1
            _record({"matchup": matchup, "family": "search", "chosen": index, "target": desired})
            return [index]
    return base


def _dragapult_promotion(raw, base, matchup):
    select = raw.get("select") or {}
    options = select.get("option") or []
    context = int(select.get("context", -1))
    if matchup == "dragapult" and context == 3 and MEM.get("boss_target") is not None:
        wanted = int(MEM["boss_target"])
        for index, option in enumerate(options):
            if _cid(_source(raw, option)) == wanted:
                MEM["boss_target"] = None
                if base != [index]:
                    STATS["overrides"] += 1
                    _record({"matchup": matchup, "family": "clefairy_boss_target",
                             "chosen": index, "target": wanted})
                return [index]
        MEM["boss_target"] = None
    if matchup == "dragapult" and RULES.get("drag_kanga_opening") and context in {3, 4}:
        target = _drag_active_target(raw)
        desired = MEM.get("switch_target") if context == 3 else None
        if context == 4 and MEM["kanga_attacks"] == 0:
            # After a knockout, Clefairy still wins priority for an immediate
            # two-Prize Dragapult conversion; otherwise open with Kangaskhan.
            _mine, board, _hand = _board(raw)
            ready_clef = next((c for c in board if _cid(c) == CLEFAIRY and _missing(c) == 0), None)
            if not (target is not None and _cid(target) == 121 and ready_clef is not None
                    and _can_knock_out(raw, ready_clef, target)):
                desired = KANGASKHAN
        if desired == KANGASKHAN:
            for index, option in enumerate(options):
                card = _source(raw, option)
                if _cid(card) == KANGASKHAN and _missing(card) == 0:
                    MEM["switch_target"] = None
                    if base != [index]:
                        STATS["overrides"] += 1
                        STATS["kanga_opening_switches"] += 1
                        _record({"matchup": matchup, "family": "kanga_promote", "chosen": index,
                                 "context": context, "target": _cid(target)})
                    return [index]
            MEM["switch_target"] = None
    if matchup == "dragapult":
        target_ids = DRAG_LINE if RULES.get("drag_line_promotion") else {121}
        enabled = bool(RULES.get("drag_exact_ko_switch"))
    elif matchup == "lucario":
        target_ids = {678}; enabled = True
    else:
        target_ids = set(); enabled = False
    if not enabled or not _opponent_has_any(raw, target_ids):
        return base
    # Nested retreat/switch/promote selection: select an attack-ready Clefairy.
    if context in {3, 4}:
        target = _drag_active_target(raw)
        for index, option in enumerate(options):
            card = _source(raw, option)
            if (_cid(card) == CLEFAIRY and _missing(card) == 0 and target is not None
                    and _cid(target) in target_ids and _can_knock_out(raw, card, target)):
                if (matchup == "dragapult" and RULES.get("drag_clef_survival_guard")
                        and not _clefairy_conversion_safe(raw)):
                    continue
                if not isinstance(base, list) or base != [index]:
                    STATS["overrides"] += 1
                    if matchup == "dragapult" and RULES.get("drag_line_promotion"):
                        STATS["clefairy_line_promotions"] += 1
                    _record({"matchup": matchup, "family": "clefairy_promote", "chosen": index,
                             "context": context, "target": _cid(target)})
                    return [index]
        return base
    if context != MAIN:
        return base
    _mine, board, _hand = _board(raw)
    active = board[0] if board else None
    clefairy = next((card for card in board[1:] if _cid(card) == CLEFAIRY and _missing(card) == 0), None)
    opponent_active, _opponent_bench = _opponent_board(raw)
    target = opponent_active[0] if opponent_active else None
    if active is None or clefairy is None or target is None or _cid(target) not in target_ids or _cid(active) == CLEFAIRY:
        return base
    if not _can_knock_out(raw, clefairy, target):
        return base
    if (matchup == "dragapult" and RULES.get("drag_clef_survival_guard")
            and not _clefairy_conversion_safe(raw)):
        return base
    for index, option in enumerate(options):
        if int(option.get("type", -1)) == RETREAT:
            STATS["overrides"] += 1
            _record({"matchup": matchup, "family": "clefairy_ko_switch", "chosen": index})
            return [index]
    return base


def _force_dragapult_boss_ko(raw, base, matchup):
    if (matchup != "dragapult" or not RULES.get("drag_boss_denial")
            or int(((raw.get("select") or {}).get("context", -1))) != MAIN):
        return base
    mine, board, _hand = _board(raw)
    active = (mine.get("active") or [None])[0] if mine else None
    if active is None or _cid(active) != CLEFAIRY or _missing(active) != 0:
        return base
    opponent_active, opponent_bench = _opponent_board(raw)
    current_target = opponent_active[0] if opponent_active else None
    current_ko = bool(current_target is not None and _can_knock_out(raw, active, current_target))
    prize_left = len(mine.get("prize") or [])
    targets = []
    for target in opponent_bench:
        cid = _cid(target)
        if cid not in DRAG_LINE or not _can_knock_out(raw, active, target):
            continue
        prizes = 2 if cid == 121 else 1
        # Boss only for a game-ending 2-Prize conversion, or when the Active
        # cannot be knocked out.  This excludes the failed broad v30 force.
        if cid == 121 and prize_left <= 2:
            score = 10000 + prizes * 1000
        elif not current_ko:
            score = 7000 + prizes * 1000 + (1000 if cid in {119, 120} else 0)
        else:
            continue
        targets.append((score, cid))
    if not targets:
        return base
    options = (raw.get("select") or {}).get("option") or []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) == PLAY and _cid(_source(raw, option)) == BOSS:
            if isinstance(base, list) and base == [index]:
                MEM["boss_target"] = max(targets)[1]
                return base
            MEM["boss_target"] = max(targets)[1]
            STATS["overrides"] += 1
            STATS["forced_boss_kos"] += 1
            STATS["clefairy_boss_denials"] += 1
            _record({"matchup": matchup, "family": "forced_boss_ko", "chosen": index})
            return [index]
    return base


def _force_alakazam_xerosic(raw, base, matchup):
    if matchup != "alakazam" or int(((raw.get("select") or {}).get("context", -1))) != MAIN:
        return base
    current, players, me = _players(raw)
    if len(players) < 2 or bool(current.get("supporterPlayed")):
        return base
    mine, opponent = players[me], players[1 - me]
    opp_hand = int(opponent.get("handCount") or len(opponent.get("hand") or []))
    opponent_active = (opponent.get("active") or [None])[0]
    turn = int(current.get("turn") or 0)
    if opp_hand < 7 or not opponent_active:
        return base
    if _cid(opponent_active) not in {245, 743} and turn < 6:
        return base
    options = (raw.get("select") or {}).get("option") or []
    try:
        base_index = int(base[0])
        base_option = options[base_index]
        base_type = int(base_option.get("type", -1))
        base_source = _cid(_source(raw, base_option))
    except Exception:
        return base
    # Finish a guaranteed knockout instead of spending the disruption card.
    active = (mine.get("active") or [None])[0]
    if base_type == ATTACK and active and _can_knock_out(raw, active, opponent_active):
        return base
    if base_type not in {ATTACK, END, PLAY}:
        return base
    if base_type == PLAY and base_source not in {BOSS, LILLIE, XEROSIC, 1188, 1221}:
        return base
    for index, option in enumerate(options):
        if int(option.get("type", -1)) == PLAY and _cid(_source(raw, option)) == XEROSIC:
            if base == [index]:
                return base
            STATS["overrides"] += 1
            STATS["xerosic_forces"] += 1
            _record({"matchup": matchup, "family": "xerosic_survival", "base": base_index, "chosen": index, "opponent_hand": opp_hand})
            return [index]
    return base


def choose(raw, base, matchup="generic"):
    STATS["decisions"] += 1
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN:
        searched = _search_choice(raw, base, matchup)
        return _dragapult_promotion(raw, searched, matchup)
    options = select.get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0])
        base_option = options[base_index]
    except Exception:
        return base
    base_source = _cid(_source(raw, base_option))
    if matchup == "dragapult" and int(base_option.get("type", -1)) == ATTACK:
        _mine, board, _hand = _board(raw)
        active = board[0] if board else None
        turn = int(((raw.get("current") or {}).get("turn") or 0))
        if _cid(active) == KANGASKHAN and MEM["last_kanga_attack_turn"] != turn:
            MEM["kanga_attacks"] += 1
            MEM["last_kanga_attack_turn"] = turn
    # The replay-inspired forced Xerosic branch changed only 1/100 Alakazam
    # games and failed its regression gate, so v30 keeps the retained v29
    # Alakazam choice instead of forcing an unsupported intervention.
    if matchup in {"marnie", "lopunny", "zoroark"} and base_source == BOSS and not _immediate_boss_ko(raw):
        alternative = _alternative_after_boss_veto(raw, options, matchup)
        if alternative is not None and alternative != base_index:
            STATS["overrides"] += 1
            STATS["boss_vetoes"] += 1
            _record({"matchup": matchup, "family": "boss_veto", "base": base_index, "chosen": alternative})
            return [alternative]
    updated = _force_dragapult_boss_ko(raw, base, matchup)
    if updated != base:
        return updated
    updated = _drag_clefairy_near_ko_veto(raw, base, matchup)
    if updated != base:
        return updated
    updated = _drag_clefairy_stage_guard(raw, base, matchup)
    if updated != base:
        return updated
    updated = _drag_kanga_switch(raw, base, matchup)
    if updated != base:
        return updated
    updated = _drag_exact_bench_fill(raw, base, matchup)
    if updated != base:
        return updated
    updated = _drag_early_deploy(raw, base, matchup)
    if updated != base:
        return updated
    updated = _tempo_rescue(raw, base, matchup)
    if updated != base:
        return updated
    updated = _attachment_completion(raw, base, matchup)
    updated = _drag_kanga_attachment(raw, updated, matchup)
    updated = _drag_preserve_second_prism(raw, updated, matchup)
    updated = _drag_clef_overattach(raw, updated, matchup)
    updated = _drag_prism_reserve(raw, updated, matchup)
    return _dragapult_promotion(raw, updated, matchup)
