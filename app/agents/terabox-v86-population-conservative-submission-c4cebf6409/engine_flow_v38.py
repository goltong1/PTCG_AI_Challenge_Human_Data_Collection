"""Terabox v38 replay-audited Crustle flow overlay.

Selected after repeated full-game gates against Crustle v130. The overlay runs
after the retained v34 guard, chooses only currently legal options, and reads no
opponent hand, prize identities, deck order, or hidden agent name. Matchup
routing is supplied by main.py from publicly seen cards.
"""
from __future__ import annotations

from cg.api import all_attack, all_card_data

ATTACH = 8
RETREAT = 12
ATTACK = 13
END = 14
MAIN = 0
PROMOTE = {3, 4}

CORNER = 117
CHIYU = 31
PECHARUNT = 230
MUNKIDORI = 112
WELLSPRING = 108
PRISM = 16
CRUSTLE = 345
DEMOLISH = 148
SOB = 135

CARDS = {int(card.cardId): card for card in all_card_data()}
ATTACKS = {int(attack.attackId): attack for attack in all_attack()}
MEMORY = {"promote_serial": None}
STATS = {
    "decisions": 0,
    "overrides": 0,
    "cornerstone_attachments": 0,
    "blocked_attack_vetoes": 0,
}


def reset() -> None:
    MEMORY["promote_serial"] = None
    for key in STATS:
        STATS[key] = 0


def _cid(card) -> int:
    try:
        return int((card or {}).get("id") or 0)
    except Exception:
        return 0


def _serial(card) -> int:
    try:
        return int((card or {}).get("serial") or -1)
    except Exception:
        return -1


def _state(raw):
    current = raw.get("current") or {}
    players = current.get("players") or []
    try:
        me = int(current.get("yourIndex") or 0)
    except Exception:
        me = 0
    if len(players) != 2:
        return current, {}, [], {}, []
    mine, opponent = players[me], players[1 - me]
    own = [
        card
        for card in list(mine.get("active") or []) + list(mine.get("bench") or [])
        if card
    ]
    theirs = [
        card
        for card in list(opponent.get("active") or []) + list(opponent.get("bench") or [])
        if card
    ]
    # Deliberately omit opponent hand and prize identities.
    return current, mine, own, opponent, theirs


def _source(raw, option):
    current, _, _, _, _ = _state(raw)
    players = current.get("players") or []
    try:
        owner = option.get("playerIndex")
        owner = int(current.get("yourIndex") or 0) if owner is None else int(owner)
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
    _, mine, _, _, _ = _state(raw)
    try:
        area = int(option.get("inPlayArea"))
        zone = (
            mine.get("active") or []
            if area == 4
            else mine.get("bench") or []
            if area == 5
            else []
        )
        return zone[int(option.get("inPlayIndex"))]
    except Exception:
        return None


def _base_index(base, options):
    try:
        if isinstance(base, list) and len(base) == 1 and 0 <= int(base[0]) < len(options):
            return int(base[0])
    except Exception:
        pass
    return None


def _energy_pool(card):
    values = (card or {}).get("energies")
    if values is not None:
        return [int(value) for value in values]
    return [10 if _cid(energy) == PRISM else 1 for energy in (card or {}).get("energyCards") or []]


def _missing_for(card, attack_id, extra_energy_type=None) -> int:
    attack = ATTACKS.get(int(attack_id or 0))
    pool = list(_energy_pool(card))
    if extra_energy_type is not None:
        pool.append(int(extra_energy_type))
    if not attack:
        return 99
    missing = 0
    for requirement in attack.energies:
        requirement = int(requirement)
        if requirement == 0:
            if pool:
                pool.pop(0)
            else:
                missing += 1
        else:
            index = next(
                (i for i, value in enumerate(pool) if value in {requirement, 10}),
                None,
            )
            if index is None:
                missing += 1
            else:
                pool.pop(index)
    return missing


def _missing_any(card) -> int:
    data = CARDS.get(_cid(card))
    values = [_missing_for(card, attack_id) for attack_id in (data.attacks or [])] if data else []
    return min(values) if values else 99


def _is_ex(card_id: int) -> bool:
    data = CARDS.get(int(card_id))
    return bool(data and (getattr(data, "ex", False) or getattr(data, "megaEx", False)))


def _has_ability(card_id: int) -> bool:
    data = CARDS.get(int(card_id))
    return bool(data and getattr(data, "skills", None))


def _safe_attack(attacker, target, attack_id: int) -> bool:
    attacker_id = _cid(attacker)
    target_id = _cid(target)
    attack_id = int(attack_id or 0)
    if target_id == CORNER and _has_ability(attacker_id) and attack_id != DEMOLISH:
        return False
    if target_id == CRUSTLE and _is_ex(attacker_id) and attack_id != DEMOLISH:
        return False
    return True


def _base_damage(attacker, attack_id: int, raw) -> int:
    attack = ATTACKS.get(int(attack_id or 0))
    damage = int(getattr(attack, "damage", 0) or 0) if attack else 0
    # Chi-Yu's Ground Melter gets the public Stadium bonus.
    if _cid(attacker) == CHIYU and int(attack_id) == 20 and (raw.get("current") or {}).get("stadium"):
        damage += 60
    return damage


def _record_override(base, index: int, stat_key: str) -> list[int]:
    choice = [index]
    if base != choice:
        STATS["overrides"] += 1
        STATS[stat_key] += 1
    return choice


def choose(raw, base, matchup: str = "generic"):
    STATS["decisions"] += 1
    select = raw.get("select") or {}
    options = select.get("option") or []
    try:
        context = int(select.get("context", -1))
    except Exception:
        return base

    if context in PROMOTE and MEMORY.get("promote_serial") is not None:
        wanted = MEMORY["promote_serial"]
        for index, option in enumerate(options):
            if _serial(_source(raw, option)) == wanted:
                MEMORY["promote_serial"] = None
                return _record_override(base, index, "blocked_attack_vetoes")
        MEMORY["promote_serial"] = None

    if context != MAIN or matchup != "crustle":
        return base

    _, _, own, _, theirs = _state(raw)
    if not own or not theirs:
        return base
    base_index = _base_index(base, options)
    base_option = options[base_index] if base_index is not None else None
    active, target = own[0], theirs[0]

    # When Cornerstone is already public, prioritize only a legal manual
    # attachment that makes actual progress toward Demolish. No discard/search
    # is forced; those broader candidates regressed in full-game gates.
    corner = next((card for card in own if _cid(card) == CORNER), None)
    if corner is not None and _missing_any(corner) > 0:
        candidates = []
        before = _missing_for(corner, DEMOLISH)
        for index, option in enumerate(options):
            if int(option.get("type", -1)) != ATTACH:
                continue
            if _serial(_target(raw, option)) != _serial(corner):
                continue
            energy = _source(raw, option)
            energy_type = 10 if _cid(energy) == PRISM else 1
            after = _missing_for(corner, DEMOLISH, energy_type)
            gain = before - after
            candidates.append((gain, 1 if _cid(energy) == PRISM else 0, -index, index))
        if candidates and max(candidates)[0] > 0:
            return _record_override(base, max(candidates)[3], "cornerstone_attachments")

    # Crustle prevents damage from Pokémon ex. Veto a truly blank ex attack at
    # the final action layer. Wellspring Sob remains legal because the public
    # retreat-lock effect is useful even when its damage is prevented.
    if base_option is not None and int(base_option.get("type", -1)) == ATTACK:
        attack_id = int(base_option.get("attackId") or 0)
        blocked = _cid(target) == CRUSTLE and _is_ex(_cid(active)) and attack_id != DEMOLISH
        allow_sob = _cid(active) == WELLSPRING and attack_id == SOB
        if blocked and not allow_sob:
            ready = []
            priority = {CORNER: 5, CHIYU: 4, PECHARUNT: 3, MUNKIDORI: 2}
            for card in own[1:]:
                data = CARDS.get(_cid(card))
                if not data:
                    continue
                attacks = [
                    int(aid)
                    for aid in (data.attacks or [])
                    if _missing_for(card, int(aid)) == 0
                    and _safe_attack(card, target, int(aid))
                ]
                if attacks:
                    ready.append(
                        (
                            priority.get(_cid(card), 1),
                            max(_base_damage(card, aid, raw) for aid in attacks),
                            -_serial(card),
                            card,
                        )
                    )
            if ready:
                chosen = max(ready)[3]
                for index, option in enumerate(options):
                    if int(option.get("type", -1)) == RETREAT:
                        MEMORY["promote_serial"] = _serial(chosen)
                        return _record_override(base, index, "blocked_attack_vetoes")
            for index, option in enumerate(options):
                if int(option.get("type", -1)) == END:
                    return _record_override(base, index, "blocked_attack_vetoes")

    return base
