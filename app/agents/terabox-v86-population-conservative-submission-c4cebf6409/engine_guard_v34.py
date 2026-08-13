"""Public-state safety and matchup-flow layer retained for v35.

The layer prevents attacks that Cornerstone Stance will reduce to zero,
validates Energy Switch against immediate completion or a bounded build, and
uses Cornerstone Mask Ogerpon as a Crustle-specific wall only after preserving
an available Teal Mask tempo attack.  The former Marnie Cornerstone build is
disabled because two independent 1,000-game leagues showed that fast Teal Mask
pressure is substantially stronger.  The deck is never changed.
"""
from __future__ import annotations

import copy
import json
import os

from cg.api import all_attack, all_card_data


MAIN = 0
PROMOTE = {3, 4}
SEARCH = {5, 6, 7, 24}
ENERGY_FROM = 28
ENERGY_TO = 21

PLAY = 7
ATTACH = 8
ABILITY = 10
RETREAT = 12
ATTACK = 13
END = 14

GRASS_ENERGY = 1
PRISM_ENERGY = 16
RAINBOW = 10
COLORLESS = 0

TEAL = 96
WELLSPRING = 108
CORNERSTONE = 117
CHI_YU = 31
CLEFAIRY = 272
KANGASKHAN = 756
ENERGY_SWITCH = 1116
TERA_ORB = 1127
ULTRA_BALL = 1121
BOSS = 1182
MARNIE_LINE = {646, 647, 648}
# Dragapult, Marnie, and Crustle retain their native energy sequencing.  In the
# 1,000-game Crustle gate this produced 431 wins versus 408 under the broad
# completion filter; three-plus-transfer lines remain rare and are audited in
# the league report.  Lucario keeps the replay-audited waste guard.
ENERGY_GUARD_MATCHUPS = {"lucario"}

CARDS = {int(card.cardId): card for card in all_card_data()}
ATTACKS = {int(attack.attackId): attack for attack in all_attack()}

DEFAULT_RULES = {
    "cornerstone_immunity_guard": True,
    "energy_switch_guard": True,
    "marnie_cornerstone_plan": False,
    "crustle_cornerstone_plan": True,
    "no_state_change_loop_guard": True,
}
try:
    RULES = dict(DEFAULT_RULES)
    RULES.update(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "guard_v34_config.json"), encoding="utf-8")))
except Exception:
    RULES = dict(DEFAULT_RULES)

STATS = {
    "decisions": 0,
    "overrides": 0,
    "cornerstone_zero_damage_vetoes": 0,
    "cornerstone_counter_retreats": 0,
    "counter_promotions": 0,
    "boss_escape_kos": 0,
    "energy_switch_vetoes": 0,
    "energy_switch_plans": 0,
    "energy_switch_source_fixes": 0,
    "energy_switch_target_fixes": 0,
    "marnie_cornerstone_plays": 0,
    "marnie_cornerstone_searches": 0,
    "marnie_cornerstone_attachments": 0,
    "marnie_cornerstone_promotions": 0,
    "marnie_demolish_forces": 0,
    "crustle_cornerstone_plays": 0,
    "crustle_cornerstone_searches": 0,
    "crustle_cornerstone_attachments": 0,
    "crustle_cornerstone_promotions": 0,
    "crustle_demolish_forces": 0,
    "loop_breaks": 0,
    "records": [],
}

MEM = {
    "energy_plan": None,
    "switch_target_serial": None,
    "switch_reason": None,
    "boss_target_serial": None,
    "search_target": None,
    "last_state": None,
    "last_action": None,
    "crustle_energy_switch_plays": 0,
}


def reset():
    for key in STATS:
        STATS[key] = [] if key == "records" else 0
    MEM.update({
        "energy_plan": None,
        "switch_target_serial": None,
        "switch_reason": None,
        "boss_target_serial": None,
        "search_target": None,
        "last_state": None,
        "last_action": None,
        "crustle_energy_switch_plays": 0,
    })


def _record(item):
    STATS["records"].append(item)
    if len(STATS["records"]) > 200:
        del STATS["records"][:-200]


def _cid(card):
    try:
        return int((card or {}).get("id") or 0)
    except Exception:
        return 0


def _serial(card):
    try:
        return int((card or {}).get("serial") or -1)
    except Exception:
        return -1


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
        zone = players[me].get("active") or [] if area == 4 else players[me].get("bench") or [] if area == 5 else []
        return zone[int(option.get("inPlayIndex"))]
    except Exception:
        return None


def _board(raw):
    _current, players, me = _players(raw)
    if len(players) != 2:
        return {}, [], [], {}, [], []
    mine, opponent = players[me], players[1 - me]
    own = [card for card in list(mine.get("active") or []) + list(mine.get("bench") or []) if card]
    theirs = [card for card in list(opponent.get("active") or []) + list(opponent.get("bench") or []) if card]
    return mine, own, list(mine.get("hand") or []), opponent, theirs, list(opponent.get("hand") or [])


def _pool(card):
    values = card.get("energies")
    if values is None:
        values = [RAINBOW if _cid(energy) == PRISM_ENERGY else GRASS_ENERGY for energy in card.get("energyCards") or []]
    return [int(value) for value in values]


def _missing_for_attack(card, attack):
    pool = _pool(card)
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


def _missing(card):
    data = CARDS.get(_cid(card))
    if data is None:
        return 99
    values = [_missing_for_attack(card, ATTACKS[int(aid)]) for aid in data.attacks or [] if int(aid) in ATTACKS]
    return min(values) if values else 99


def _with_energy(card, energy_type):
    result = copy.deepcopy(card)
    result["energies"] = list(_pool(result)) + [int(energy_type)]
    return result


def _without_basic_energy(card, energy_serial=None):
    result = copy.deepcopy(card)
    cards = list(result.get("energyCards") or [])
    index = next((i for i, energy in enumerate(cards)
                  if _cid(energy) == GRASS_ENERGY and (energy_serial is None or _serial(energy) == energy_serial)), None)
    if index is None:
        return None
    del cards[index]
    result["energyCards"] = cards
    values = list(result.get("energies") or _pool(card))
    try:
        values.remove(GRASS_ENERGY)
    except ValueError:
        return None
    result["energies"] = values
    return result


def _attack_blocked(attacker, attack_id, target):
    if _cid(target) != CORNERSTONE:
        return False
    data = CARDS.get(_cid(attacker))
    has_ability = bool(data is not None and data.skills)
    # Demolish explicitly ignores effects on the defending Pokémon.
    return has_ability and int(attack_id or 0) != 148


def _ready_attack_ids(card):
    data = CARDS.get(_cid(card))
    if data is None:
        return []
    return [int(aid) for aid in data.attacks or []
            if int(aid) in ATTACKS and _missing_for_attack(card, ATTACKS[int(aid)]) == 0]


def _safe_ready_attacker(card, target):
    return any(not _attack_blocked(card, attack_id, target) for attack_id in _ready_attack_ids(card))


def _has_safe_attack(card, target):
    data = CARDS.get(_cid(card))
    return bool(data is not None and any(not _attack_blocked(card, int(attack_id), target)
                                         for attack_id in data.attacks or []))


def _teal_damage(raw, target):
    _mine, own, _hand, _opponent, _theirs, _opp_hand = _board(raw)
    active = own[0] if own else None
    if _cid(active) != TEAL:
        return 0
    return 30 + 30 * (len(_pool(active)) + len(_pool(target)))


def _main_choice(raw, base):
    options = (raw.get("select") or {}).get("option") or []
    if not isinstance(base, list) or len(base) != 1:
        return None, options
    try:
        index = int(base[0])
        return (index, options[index]), options
    except Exception:
        return None, options


def _same_source(raw, left, right):
    return _serial(_source(raw, left)) == _serial(_source(raw, right)) and _cid(_source(raw, left)) == _cid(_source(raw, right))


def _state_fingerprint(raw):
    current, players, me = _players(raw)
    if len(players) != 2:
        return None
    mine, opponent = players[me], players[1 - me]

    def pokemon(player):
        return tuple((_serial(card), _cid(card), int(card.get("hp") or 0), tuple(_pool(card)))
                     for card in list(player.get("active") or []) + list(player.get("bench") or []) if card)

    return (
        int(current.get("turn") or 0),
        pokemon(mine), pokemon(opponent),
        tuple(sorted(_serial(card) for card in mine.get("hand") or [])),
        tuple(sorted(_serial(card) for card in mine.get("discard") or [])),
        int(mine.get("deckCount") or 0),
        bool(current.get("energyAttached")), bool(current.get("supporterPlayed")), bool(current.get("retreated")),
    )


def _action_signature(raw, option):
    return (int(option.get("type", -1)), _cid(_source(raw, option)), _serial(_source(raw, option)),
            _cid(_target(raw, option)), _serial(_target(raw, option)), int(option.get("attackId") or 0))


def _safe_attack_option(raw, options):
    _mine, own, _hand, _opponent, theirs, _opp_hand = _board(raw)
    active = own[0] if own else None
    target = theirs[0] if theirs else None
    choices = []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != ATTACK:
            continue
        attack_id = int(option.get("attackId") or 0)
        if _attack_blocked(active, attack_id, target):
            continue
        attack = ATTACKS.get(attack_id)
        choices.append((int((attack or object()).damage or 0) if attack is not None else 0, -index, index))
    return max(choices)[2] if choices else None


def _fallback(raw, options, excluded_signature=None):
    safe_attack = _safe_attack_option(raw, options)
    if safe_attack is not None:
        return safe_attack
    ranked = []
    for index, option in enumerate(options):
        signature = _action_signature(raw, option)
        if signature == excluded_signature:
            continue
        typ = int(option.get("type", -1))
        source_id = _cid(_source(raw, option))
        if typ == ATTACH:
            target = _target(raw, option)
            energy_type = RAINBOW if source_id == PRISM_ENERGY else GRASS_ENERGY if source_id == GRASS_ENERGY else None
            progress = 0 if target is None or energy_type is None else _missing(target) - _missing(_with_energy(target, energy_type))
            score = 7000 + 500 * progress
        elif typ == ABILITY:
            score = 6000
        elif typ == PLAY and source_id not in {ENERGY_SWITCH, BOSS}:
            score = 4500
        elif typ == RETREAT:
            score = 3500
        elif typ == END:
            score = 100
        else:
            continue
        ranked.append((score, -index, index))
    return max(ranked)[2] if ranked else None


def _energy_plans(raw, matchup):
    mine, own, hand, opponent, theirs, _opp_hand = _board(raw)
    if not own or not theirs:
        return []
    active_serial = _serial(own[0])
    target = theirs[0]
    switch_count = sum(_cid(card) == ENERGY_SWITCH for card in hand)
    basic_energy_count = sum(_cid(energy) == GRASS_ENERGY for card in own for energy in card.get("energyCards") or [])
    plans = []
    for donor in own:
        for energy in donor.get("energyCards") or []:
            if _cid(energy) != GRASS_ENERGY:
                continue
            donor_after = _without_basic_energy(donor, _serial(energy))
            if donor_after is None:
                continue
            for recipient in own:
                if _serial(recipient) == _serial(donor):
                    continue
                recipient_after = _with_energy(recipient, GRASS_ENERGY)
                recipient_before_missing = _missing(recipient)
                recipient_after_missing = _missing(recipient_after)
                donor_before_missing = _missing(donor)
                donor_after_missing = _missing(donor_after)
                completion = recipient_before_missing > 0 and recipient_after_missing == 0
                donor_stays_ready = donor_before_missing == 0 and donor_after_missing == 0
                recipient_is_active = _serial(recipient) == active_serial
                immunity_breaker = (_cid(target) == CORNERSTONE and _safe_ready_attacker(recipient_after, target))
                cornerstone_plan = (
                    (matchup == "marnie" and RULES.get("marnie_cornerstone_plan"))
                    or (matchup == "crustle" and RULES.get("crustle_cornerstone_plan"))
                )
                matchup_wall = bool(cornerstone_plan and _cid(recipient) == CORNERSTONE)
                donor_has_usable_ready_attack = _safe_ready_attacker(donor, target)
                stone_counter_progress = (
                    _cid(target) == CORNERSTONE
                    and _has_safe_attack(recipient, target)
                    and recipient_after_missing < recipient_before_missing
                    and (donor_stays_ready or not donor_has_usable_ready_attack)
                )
                # Permit a bounded two-card consolidation only when the full
                # sequence can ready the current attacker (or the Marnie wall)
                # this turn.  This retains legitimate Energy Switch chains but
                # still rejects open-ended shuffling among incomplete Benched
                # Pokémon.
                bounded_chain = (
                    not completion
                    and recipient_after_missing < recipient_before_missing
                    and recipient_before_missing <= min(2, switch_count)
                    and basic_energy_count >= recipient_before_missing
                    and (recipient_is_active or matchup_wall)
                )
                if not ((completion and (donor_stays_ready or recipient_is_active or immunity_breaker or matchup_wall))
                        or bounded_chain or stone_counter_progress):
                    continue
                score = 100
                score += 80 if recipient_is_active else 0
                score += 70 if immunity_breaker else 0
                score += 60 if matchup_wall else 0
                score += 65 if stone_counter_progress else 0
                score += 30 if donor_stays_ready else 0
                score += 35 if bounded_chain else 0
                score += 10 * max(0, donor_after_missing - donor_before_missing == 0)
                plans.append({
                    "score": score,
                    "energy_serial": _serial(energy),
                    "source_serial": _serial(donor),
                    "source_id": _cid(donor),
                    "target_serial": _serial(recipient),
                    "target_id": _cid(recipient),
                    "donor_before_missing": donor_before_missing,
                    "donor_after_missing": donor_after_missing,
                    "target_before_missing": recipient_before_missing,
                    "target_after_missing": recipient_after_missing,
                })
    return sorted(plans, key=lambda item: item["score"], reverse=True)


def _resolve_energy_plan(raw, base, context):
    plan = MEM.get("energy_plan")
    if not plan:
        return base
    options = (raw.get("select") or {}).get("option") or []
    if context == ENERGY_FROM:
        for index, option in enumerate(options):
            source = _source(raw, option)
            if _serial(source) != plan["source_serial"]:
                continue
            try:
                energy = (source.get("energyCards") or [])[int(option.get("energyIndex"))]
            except Exception:
                continue
            if _serial(energy) == plan["energy_serial"]:
                if base != [index]:
                    STATS["overrides"] += 1
                    STATS["energy_switch_source_fixes"] += 1
                return [index]
        # The planned donor/energy is no longer among the legal choices.  Do
        # not let a stale plan hijack the destination prompt of a later Item.
        MEM["energy_plan"] = None
    if context == ENERGY_TO:
        for index, option in enumerate(options):
            if _serial(_source(raw, option)) == plan["target_serial"]:
                MEM["energy_plan"] = None
                if base != [index]:
                    STATS["overrides"] += 1
                    STATS["energy_switch_target_fixes"] += 1
                return [index]
        MEM["energy_plan"] = None
    return base


def _resolve_promote_or_boss(raw, base, context):
    if context not in PROMOTE:
        return base
    options = (raw.get("select") or {}).get("option") or []
    boss_serial = MEM.get("boss_target_serial")
    if boss_serial is not None:
        for index, option in enumerate(options):
            if _serial(_source(raw, option)) == boss_serial:
                MEM["boss_target_serial"] = None
                if base != [index]:
                    STATS["overrides"] += 1
                return [index]
        MEM["boss_target_serial"] = None
    switch_serial = MEM.get("switch_target_serial")
    if switch_serial is not None:
        switch_reason = MEM.get("switch_reason")
        for index, option in enumerate(options):
            if _serial(_source(raw, option)) == switch_serial:
                MEM["switch_target_serial"] = None
                MEM["switch_reason"] = None
                if base != [index]:
                    STATS["overrides"] += 1
                    STATS["counter_promotions"] += 1
                    if switch_reason in {"marnie", "crustle"}:
                        STATS[f"{switch_reason}_cornerstone_promotions"] += 1
                return [index]
        MEM["switch_target_serial"] = None
        MEM["switch_reason"] = None
    return base


def _resolve_search(raw, base, matchup, context):
    if context not in SEARCH:
        return base
    target = MEM.get("search_target")
    stone_plan = ((matchup == "marnie" and RULES.get("marnie_cornerstone_plan"))
                  or (matchup == "crustle" and RULES.get("crustle_cornerstone_plan")))
    if target is None and stone_plan:
        _mine, own, hand, _opponent, _theirs, _opp_hand = _board(raw)
        if not any(_cid(card) == CORNERSTONE for card in own + hand):
            target = CORNERSTONE
    if target is None:
        return base
    for index, option in enumerate((raw.get("select") or {}).get("option") or []):
        if _cid(_source(raw, option)) == target:
            MEM["search_target"] = None
            if base != [index]:
                STATS["overrides"] += 1
                STATS[f"{matchup}_cornerstone_searches"] += 1
            return [index]
    # Cornerstone can be prized or already exhausted from the deck.  Clearing
    # the request here prevents it leaking into an unrelated later search.
    MEM["search_target"] = None
    return base


def _matchup_cornerstone(raw, base, options, matchup):
    mine, own, hand, _opponent, theirs, _opp_hand = _board(raw)
    if not own or not theirs:
        return base
    active, target = own[0], theirs[0]
    corner = next((card for card in own if _cid(card) == CORNERSTONE), None)
    known_corner = corner is not None or any(_cid(card) == CORNERSTONE for card in hand)

    if corner is not None and _missing(corner) == 0:
        if _cid(active) == CORNERSTONE:
            for index, option in enumerate(options):
                if int(option.get("type", -1)) == ATTACK and int(option.get("attackId") or 0) == 148:
                    if base != [index]:
                        STATS["overrides"] += 1
                        STATS[f"{matchup}_demolish_forces"] += 1
                        _record({"family": f"{matchup}_demolish", "turn": int((_players(raw)[0]).get("turn") or 0),
                                 "target": _cid(target)})
                    return [index]
        elif _safe_ready_attacker(corner, target) and (
                matchup != "crustle" or _cid(target) in {117, 345}
                or not _safe_ready_attacker(active, target)):
            for index, option in enumerate(options):
                if int(option.get("type", -1)) == RETREAT:
                    MEM["switch_target_serial"] = _serial(corner)
                    MEM["switch_reason"] = matchup
                    if base != [index]:
                        STATS["overrides"] += 1
                        STATS["cornerstone_counter_retreats"] += 1
                        STATS[f"{matchup}_cornerstone_promotions"] += 1
                    return [index]

    if corner is None:
        for index, option in enumerate(options):
            if int(option.get("type", -1)) == PLAY and _cid(_source(raw, option)) == CORNERSTONE:
                if base != [index]:
                    STATS["overrides"] += 1
                    STATS[f"{matchup}_cornerstone_plays"] += 1
                return [index]
        if not known_corner:
            for wanted in (TERA_ORB, ULTRA_BALL):
                for index, option in enumerate(options):
                    if int(option.get("type", -1)) == PLAY and _cid(_source(raw, option)) == wanted:
                        if wanted == TERA_ORB or (isinstance(base, list) and base == [index]):
                            MEM["search_target"] = CORNERSTONE
                            if base != [index]:
                                STATS["overrides"] += 1
                            return [index]

    choice, _unused = _main_choice(raw, base)
    if corner is not None and choice is not None and int(choice[1].get("type", -1)) == ATTACH:
        base_option = choice[1]
        source_id = _cid(_source(raw, base_option))
        energy_type = RAINBOW if source_id == PRISM_ENERGY else GRASS_ENERGY if source_id == GRASS_ENERGY else None
        base_target = _target(raw, base_option)
        if energy_type is not None and base_target is not None:
            base_completion = _missing(base_target) > 0 and _missing(_with_energy(base_target, energy_type)) == 0
            corner_before = _missing(corner)
            corner_after = _missing(_with_energy(corner, energy_type))
            if corner_after < corner_before and not base_completion:
                for index, option in enumerate(options):
                    if int(option.get("type", -1)) != ATTACH or not _same_source(raw, base_option, option):
                        continue
                    if _serial(_target(raw, option)) == _serial(corner):
                        if base != [index]:
                            STATS["overrides"] += 1
                            STATS[f"{matchup}_cornerstone_attachments"] += 1
                            _record({"family": f"{matchup}_cornerstone_attach", "source": source_id,
                                     "before": corner_before, "after": corner_after})
                        return [index]
    return base


def _cornerstone_immunity_guard(raw, base, options):
    choice, _unused = _main_choice(raw, base)
    if choice is None or int(choice[1].get("type", -1)) != ATTACK:
        return base
    _mine, own, _hand, _opponent, theirs, _opp_hand = _board(raw)
    if not own or not theirs:
        return base
    active, target = own[0], theirs[0]
    attack_id = int(choice[1].get("attackId") or 0)
    if not _attack_blocked(active, attack_id, target):
        return base
    STATS["cornerstone_zero_damage_vetoes"] += 1

    # First escape: take a real prize with Boss instead of striking immunity.
    if _cid(active) == TEAL:
        _current, players, me = _players(raw)
        opponent_bench = list((players[1 - me].get("bench") or [])) if len(players) == 2 else []
        ko_target = next((card for card in opponent_bench
                          if _cid(card) != CORNERSTONE and _teal_damage(raw, card) >= int(card.get("hp") or 0) > 0), None)
        if ko_target is not None:
            for index, option in enumerate(options):
                if int(option.get("type", -1)) == PLAY and _cid(_source(raw, option)) == BOSS:
                    MEM["boss_target_serial"] = _serial(ko_target)
                    STATS["overrides"] += 1
                    STATS["boss_escape_kos"] += 1
                    _record({"family": "cornerstone_boss_escape", "target": _cid(ko_target)})
                    return [index]

    counter = next((card for card in own[1:] if _safe_ready_attacker(card, target)), None)
    if counter is not None:
        for index, option in enumerate(options):
            if int(option.get("type", -1)) == RETREAT:
                MEM["switch_target_serial"] = _serial(counter)
                MEM["switch_reason"] = "cornerstone_counter"
                STATS["overrides"] += 1
                STATS["cornerstone_counter_retreats"] += 1
                _record({"family": "cornerstone_counter_retreat", "from": _cid(active), "to": _cid(counter)})
                return [index]

    # If no concrete counter is ready, end the turn and preserve resources.
    # Playing a random Item or moving more Energy merely replaces one zero-EV
    # action with another and caused the broad v34 draft to regress at 1,000.
    fallback = next((index for index, option in enumerate(options)
                     if int(option.get("type", -1)) == END), None)
    if fallback is None:
        fallback = _fallback(raw, options, _action_signature(raw, choice[1]))
    if fallback is not None and fallback != choice[0]:
        STATS["overrides"] += 1
        _record({"family": "cornerstone_zero_damage_veto", "attacker": _cid(active), "fallback": fallback})
        return [fallback]
    return base


def _energy_switch_guard(raw, base, options, matchup):
    choice, _unused = _main_choice(raw, base)
    if choice is None or int(choice[1].get("type", -1)) != PLAY or _cid(_source(raw, choice[1])) != ENERGY_SWITCH:
        return base
    plans = _energy_plans(raw, matchup)
    if plans:
        MEM["energy_plan"] = plans[0]
        STATS["energy_switch_plans"] += 1
        _record({"family": "energy_switch_plan", **plans[0]})
        return base
    fallback = _fallback(raw, options, _action_signature(raw, choice[1]))
    if fallback is not None and fallback != choice[0]:
        STATS["overrides"] += 1
        STATS["energy_switch_vetoes"] += 1
        _record({"family": "energy_switch_veto", "fallback": fallback})
        return [fallback]
    return base


def _crustle_energy_switch_cap(raw, base, options, matchup):
    """Preserve useful one/two-transfer lines and reject the third shuffle.

    In the v35 1,000-game flow gate, exactly one Energy Switch was the strongest
    bucket.  An unconditional completion filter also removed productive first
    moves, while the third transfer was the stable waste boundary.
    """
    if matchup != "crustle":
        return base
    choice, _unused = _main_choice(raw, base)
    if choice is None or int(choice[1].get("type", -1)) != PLAY or _cid(_source(raw, choice[1])) != ENERGY_SWITCH:
        return base
    if MEM["crustle_energy_switch_plays"] < 2:
        MEM["crustle_energy_switch_plays"] += 1
        return base
    fallback = _fallback(raw, options, _action_signature(raw, choice[1]))
    if fallback is not None and fallback != choice[0]:
        STATS["overrides"] += 1
        STATS["energy_switch_vetoes"] += 1
        _record({"family": "crustle_third_energy_switch_veto", "fallback": fallback})
        return [fallback]
    return base


def _loop_guard(raw, base, options):
    choice, _unused = _main_choice(raw, base)
    if choice is None:
        return base
    state = _state_fingerprint(raw)
    action = _action_signature(raw, choice[1])
    if state == MEM.get("last_state") and action == MEM.get("last_action") and int(choice[1].get("type", -1)) != END:
        fallback = _fallback(raw, options, action)
        if fallback is not None and fallback != choice[0]:
            STATS["overrides"] += 1
            STATS["loop_breaks"] += 1
            _record({"family": "no_state_change_loop", "action": action, "fallback": fallback})
            MEM["last_state"] = state
            MEM["last_action"] = _action_signature(raw, options[fallback])
            return [fallback]
    MEM["last_state"] = state
    MEM["last_action"] = action
    return base


def choose(raw, base, matchup="generic"):
    STATS["decisions"] += 1
    select = raw.get("select") or {}
    try:
        context = int(select.get("context", -1))
    except Exception:
        return base

    # A global completion-only rule lost the 1,000-game mirror gate (48.6%).
    # Keep the replay-supported deck specialists, and in an unclassified game
    # activate it only after Cornerstone is publicly visible.  This preserves
    # legitimate early consolidation while fixing the supplied stone-wall line.
    _mine, _own, _hand, _opponent, visible_opponents, _opp_hand = _board(raw)
    generic_cornerstone = matchup == "generic" and any(_cid(card) == CORNERSTONE for card in visible_opponents)
    energy_guard_active = (RULES.get("energy_switch_guard")
                           and (matchup in ENERGY_GUARD_MATCHUPS or generic_cornerstone))
    if energy_guard_active:
        updated = _resolve_energy_plan(raw, base, context)
        if updated != base:
            return updated
    updated = _resolve_promote_or_boss(raw, base, context)
    if updated != base:
        return updated
    stone_plan = ((matchup == "marnie" and RULES.get("marnie_cornerstone_plan"))
                  or (matchup == "crustle" and RULES.get("crustle_cornerstone_plan")))
    if stone_plan:
        updated = _resolve_search(raw, base, matchup, context)
        if updated != base:
            return updated
    if context != MAIN:
        return base

    options = select.get("option") or []
    if stone_plan:
        base = _matchup_cornerstone(raw, base, options, matchup)
    if RULES.get("cornerstone_immunity_guard"):
        base = _cornerstone_immunity_guard(raw, base, options)
    if energy_guard_active:
        base = _energy_switch_guard(raw, base, options, matchup)
    else:
        base = _crustle_energy_switch_cap(raw, base, options, matchup)
    return _loop_guard(raw, base, options) if RULES.get("no_state_change_loop_guard") else base
