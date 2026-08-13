"""Replay-audited energy attachment correction for Tera Box v27.

The overlay is deliberately narrow.  In the replay-supported Archaludon line,
it changes a manual attachment only when Cornerstone Mask Ogerpon can already
attack and the *same energy card* completes another attacker on the board.
Mega Kangaskhan ex receives one additional guard: a fourth energy is redirected
to a genuine energy-progress target even when that target is not completed
immediately.

This keeps total field energy (and therefore Myriad Leaf Shower scaling) the
same while reducing single-attacker over-investment found in official wins and
losses.
"""
from __future__ import annotations

from cg.api import all_attack, all_card_data


ATTACH = 8
MAIN = 0
ACTIVE = 4
BENCH = 5
RAINBOW = 10
COLORLESS = 0

TEAL_OGERPON = 96
MEGA_KANGASKHAN = 756

_CARDS = {int(card.cardId): card for card in all_card_data()}
_ATTACKS = {int(attack.attackId): attack for attack in all_attack()}

_PRIORITY = {
    "lucario": [272, 96, 756, 117, 108, 184, 230, 31],
    "dragapult": [272, 756, 108, 96, 117, 184],
    "archaludon": [117, 96, 108, 272, 756, 184],
    "crustle": [230, 31, 96, 756, 272, 117],
    "alakazam": [117, 96, 108, 272, 756, 184],
    "marnie": [117, 96, 108, 272, 756, 184],
    "generic": [756, 96, 272, 108, 117, 31, 230, 184],
}

STATS = {
    "decisions": 0,
    "manual_attachments": 0,
    "ready_target_attachments": 0,
    "completion_redirects": 0,
    "kangaskhan_fourth_redirects": 0,
    "records": [],
}


def reset():
    STATS["decisions"] = 0
    STATS["manual_attachments"] = 0
    STATS["ready_target_attachments"] = 0
    STATS["completion_redirects"] = 0
    STATS["kangaskhan_fourth_redirects"] = 0
    STATS["records"] = []


def _card_id(card):
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
        player = players[owner]
        area = int(option.get("area", 2))
        zones = {
            1: (raw.get("select") or {}).get("deck") or [],
            2: player.get("hand") or [],
            3: player.get("discard") or [],
            4: player.get("active") or [],
            5: player.get("bench") or [],
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
        player = players[me]
        cards = player.get("active") or [] if area == ACTIVE else player.get("bench") or [] if area == BENCH else []
        return cards[int(option.get("inPlayIndex"))]
    except Exception:
        return None


def _energy_type(source):
    # Deck card 16 is Prism Energy; battle observations encode it as rainbow 10.
    source_id = _card_id(source)
    if source_id == 16:
        return RAINBOW
    data = _CARDS.get(source_id)
    try:
        return int(data.energyType)
    except Exception:
        return 1 if source_id == 1 else None


def _energy_pool(pokemon):
    values = pokemon.get("energies")
    if values is not None:
        return [int(value) for value in values]
    # Old replay observations always contain energies.  This fallback only
    # preserves a conservative count if a future schema omits the type list.
    return [RAINBOW for _ in (pokemon.get("energyCards") or [])]


def _attack_missing(pokemon, added=None):
    data = _CARDS.get(_card_id(pokemon))
    if data is None:
        return 99
    pool0 = _energy_pool(pokemon)
    if added is not None:
        pool0.append(int(added))
    best = 99
    for attack_id in data.attacks or []:
        attack = _ATTACKS.get(int(attack_id))
        if attack is None:
            continue
        pool = list(pool0)
        missing = 0
        typed = [int(req) for req in attack.energies if int(req) != COLORLESS]
        colorless = sum(int(req) == COLORLESS for req in attack.energies)
        for requirement in typed:
            index = next((i for i, value in enumerate(pool) if value in (requirement, RAINBOW)), None)
            if index is None:
                missing += 1
            else:
                pool.pop(index)
        available = len(pool)
        if available < colorless:
            missing += colorless - available
        best = min(best, missing)
    return best


def _ready_damage(pokemon, added=None):
    data = _CARDS.get(_card_id(pokemon))
    if data is None:
        return 0
    pool0 = _energy_pool(pokemon)
    if added is not None:
        pool0.append(int(added))
    best = 0
    for attack_id in data.attacks or []:
        attack = _ATTACKS.get(int(attack_id))
        if attack is None:
            continue
        pool = list(pool0)
        payable = True
        typed = [int(req) for req in attack.energies if int(req) != COLORLESS]
        colorless = sum(int(req) == COLORLESS for req in attack.energies)
        for requirement in typed:
            index = next((i for i, value in enumerate(pool) if value in (requirement, RAINBOW)), None)
            if index is None:
                payable = False
                break
            pool.pop(index)
        if payable and len(pool) >= colorless:
            best = max(best, int(attack.damage or 0))
    return best


def _priority(matchup, card_id):
    order = _PRIORITY.get(matchup, _PRIORITY["generic"])
    try:
        return len(order) - order.index(card_id)
    except ValueError:
        return 0


def _same_source(raw, left, right):
    # Index/area identity matters: do not silently swap one copy of an energy
    # for another copy that may have different legality or reveal information.
    return (
        int(left.get("area", 2)) == int(right.get("area", 2))
        and int(left.get("index", -1)) == int(right.get("index", -1))
        and _card_id(_source(raw, left)) == _card_id(_source(raw, right))
    )


def choose(raw, base, matchup="generic"):
    STATS["decisions"] += 1
    select = raw.get("select") or {}
    if int(select.get("context", -1)) != MAIN or not isinstance(base, list) or len(base) != 1:
        return base
    options = select.get("option") or []
    try:
        chosen_index = int(base[0])
        chosen = options[chosen_index]
    except Exception:
        return base
    if int(chosen.get("type", -1)) != ATTACH:
        return base
    STATS["manual_attachments"] += 1

    selected = _target(raw, chosen)
    source = _source(raw, chosen)
    added = _energy_type(source)
    if selected is None or added is None or _card_id(selected) == TEAL_OGERPON:
        return base
    selected_missing = _attack_missing(selected)
    if selected_missing != 0:
        return base
    # A low-cost utility attack being ready does not make an attachment
    # redundant.  Preserve attachments that unlock a stronger damage tier
    # (for example Chi-Yu's Ground Melter after Allure is already available).
    if _ready_damage(selected, added) > _ready_damage(selected):
        return base
    STATS["ready_target_attachments"] += 1

    alternatives = []
    selected_id = _card_id(selected)
    selected_energy_count = len(selected.get("energyCards") or selected.get("energies") or [])
    kangaskhan_fourth = selected_id == MEGA_KANGASKHAN and selected_energy_count >= 3
    # Replay evidence and matched rollouts support this completion split for
    # the Cornerstone wall specifically.  Do not generalize it to Clefairy in
    # the Dragapult matchup: a 60-game screen correlated those redirects with
    # six losses in seven triggered games.
    completion_split = matchup == "archaludon" and selected_id == 117
    if not completion_split and not kangaskhan_fourth:
        return base
    for index, option in enumerate(options):
        if index == chosen_index or int(option.get("type", -1)) != ATTACH or not _same_source(raw, chosen, option):
            continue
        target = _target(raw, option)
        if target is None or _card_id(target) == selected_id and target is selected:
            continue
        before = _attack_missing(target)
        after = _attack_missing(target, added)
        if after >= before:
            continue
        completes = before > 0 and after == 0
        if completes and not completion_split and not kangaskhan_fourth:
            continue
        if not completes and not kangaskhan_fourth:
            continue
        target_id = _card_id(target)
        score = (100000 if completes else 0) + (before - after) * 1000 + _priority(matchup, target_id) * 10
        alternatives.append((score, -after, -before, -index, index, target_id, before, after, completes))

    if not alternatives:
        return base
    best = max(alternatives)
    replacement_index = best[4]
    reason = "completion" if best[8] else "kangaskhan_fourth"
    if best[8]:
        STATS["completion_redirects"] += 1
    else:
        STATS["kangaskhan_fourth_redirects"] += 1
    current = raw.get("current") or {}
    record = {
        "turn": int(current.get("turn") or 0),
        "matchup": matchup,
        "source": _card_id(source),
        "from": selected_id,
        "to": best[5],
        "before_missing": best[6],
        "after_missing": best[7],
        "reason": reason,
    }
    records = STATS["records"]
    records.append(record)
    if len(records) > 64:
        del records[:-64]
    return [replacement_index]
