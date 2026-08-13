"""Public-state replay-regret overlay for Tera Box v29.

The offline teacher branches mutually exclusive actions from identical official
replay states.  This runtime student sees only the legal observation: its own
hand, public fields, prize counts, and opponent hand count.  It intervenes only
in manual attachment and supporter decisions, where the new replays provide
causal evidence and where short action-order differences cannot contaminate the
label.  Unsupported states retain the v28 baseline.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from cg.api import all_attack, all_card_data

ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PRIOR_PATH = os.path.join(ROOT, "replay_policy_v29.json")

MAIN = 0
TO_BENCH = 5
TO_FIELD = 6
TO_HAND = 7
LOOK = 24
PLAY = 7
ATTACH = 8
ACTIVE = 4
BENCH = 5
RAINBOW = 10
COLORLESS = 0

SUPPORTERS = {1182, 1188, 1197, 1221, 1227}
PRISM = 16
GRASS = 1

CARDS = {int(card.cardId): card for card in all_card_data()}
ATTACKS = {int(attack.attackId): attack for attack in all_attack()}
try:
    with open(PRIOR_PATH, encoding="utf-8") as handle:
        PRIORS = json.load(handle).get("priors", {})
except Exception:
    PRIORS = {}

TECH = {
    "crustle": {117, 31, 230},
    "dragapult": {272, 96},
    "lucario": {272, 96},
    "alakazam": {117, 96},
    "cinderace": {96},
    "cynthia": {96, 184},
    "spidops": {96},
    "generic": {96},
}

STATS = {
    "decisions": 0,
    "attachment_windows": 0,
    "supporter_windows": 0,
    "search_windows": 0,
    "overrides": 0,
    "attachment_overrides": 0,
    "supporter_overrides": 0,
    "search_overrides": 0,
    "setup_rescue_overrides": 0,
    "records": [],
}


def reset():
    for key in list(STATS):
        STATS[key] = [] if key == "records" else 0


def _record(value):
    STATS["records"].append(value)
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
        zone = player.get("active") or [] if area == ACTIVE else player.get("bench") or [] if area == BENCH else []
        return zone[int(option.get("inPlayIndex"))]
    except Exception:
        return None


def _energy_type(source):
    source_id = _cid(source)
    if source_id == PRISM:
        return RAINBOW
    data = CARDS.get(source_id)
    try:
        return int(data.energyType)
    except Exception:
        return GRASS if source_id == GRASS else None


def _pool(pokemon, added=None):
    values = pokemon.get("energies")
    if values is None:
        values = [RAINBOW for _ in pokemon.get("energyCards") or []]
    result = [int(value) for value in values]
    if added is not None:
        result.append(int(added))
    return result


def _missing(pokemon, added=None):
    data = CARDS.get(_cid(pokemon))
    if data is None:
        return 99
    best = 99
    for attack_id in data.attacks or []:
        attack = ATTACKS.get(int(attack_id))
        if attack is None:
            continue
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
        best = min(best, missing)
    return best


def _ready_damage(pokemon):
    data = CARDS.get(_cid(pokemon))
    if data is None:
        return 0
    best = 0
    for attack_id in data.attacks or []:
        attack = ATTACKS.get(int(attack_id))
        if attack is not None and _missing_for_attack(pokemon, attack) == 0:
            best = max(best, int(attack.damage or 1))
    return best


def _missing_for_attack(pokemon, attack):
    pool = _pool(pokemon)
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


def _phase(raw):
    turn = int(((raw.get("current") or {}).get("turn") or 0))
    return "early" if turn <= 5 else "mid" if turn <= 10 else "late"


def _prior_keys(matchup, phase, family, source, target=0):
    if family == "attach":
        return [
            f"{matchup}|{phase}|attach|{source}|{target}",
            f"{matchup}|attach|{source}|{target}",
            f"{matchup}|{phase}|attach|target:{target}",
            f"{matchup}|attach|target:{target}",
            f"all|attach|{source}|{target}",
        ]
    return [
        f"{matchup}|{phase}|supporter|{source}",
        f"{matchup}|supporter|{source}",
        f"all|supporter|{source}",
    ]


def _learned(matchup, phase, family, source, target=0):
    weighted = 0.0
    weight = 0.0
    support = 0
    positive = 0.0
    for rank, key in enumerate(_prior_keys(matchup, phase, family, source, target)):
        item = PRIORS.get(key)
        if not item or int(item.get("n", 0)) < 2:
            continue
        n = int(item["n"])
        # Specific keys get more influence, but broad keys prevent a single
        # replay state from becoming a hard rule.
        w = min(6, n) * (1.0 if rank < 2 else 0.55 if rank < 4 else 0.35)
        weighted += float(item["mean"]) * w
        weight += w
        support = max(support, n)
        positive = max(positive, float(item.get("positive_rate", 0.0)))
    value = weighted / weight if weight else 0.0
    return max(-6500.0, min(6500.0, value)), support, positive


def _attachment_score(raw, option, matchup):
    current, players, me = _players(raw)
    mine = players[me]
    source = _source(raw, option)
    target = _target(raw, option)
    if source is None or target is None:
        return -1e9, {"reason": "missing_source_or_target"}
    source_id = _cid(source)
    target_id = _cid(target)
    energy = _energy_type(source)
    if energy is None:
        return -1e9, {"reason": "not_energy"}
    before = _missing(target)
    after = _missing(target, energy)
    active = (mine.get("active") or [None])[0]
    active_ready = bool(active and _ready_damage(active) > 0)
    target_active = target is active or (
        int(option.get("inPlayArea", -1)) == ACTIVE and int(option.get("inPlayIndex", -1)) == 0
    )
    score = max(0, before - after) * 2800.0
    hard = False
    reasons = []
    if before > 0 and after == 0:
        score += 5000.0
        hard = True
        reasons.append("completes_attacker")
    if before == 0:
        score -= 5200.0
        reasons.append("over_attachment")
    if not active_ready:
        if target_active:
            score += 5200.0
            hard = True
            reasons.append("enables_first_attacker")
        elif active is not None and _missing(active) < 99:
            score -= 3000.0
    else:
        if not target_active and before > 0:
            score += 3600.0
            reasons.append("builds_backup")
        if target_active and before == 0:
            score -= 3200.0
    if source_id == PRISM and target_id in {96, 756}:
        score -= 8500.0
        hard = True
        reasons.append("preserve_prism")
    if source_id == GRASS and target_id == 96:
        score += 1800.0
    if source_id == PRISM and target_id in TECH.get(matchup, set()) and before > 0:
        score += 2600.0
    if matchup in {"dragapult", "lucario"}:
        if target_id == 272 and (active_ready or target_active):
            score += 4200.0
            reasons.append("clefairy_chain")
        if target_id == 756 and not target_active:
            score -= 4800.0
    elif matchup == "crustle":
        if target_id == 117 and active_ready:
            score += 4600.0
            reasons.append("cornerstone_chain")
        if target_id == 756 and not target_active:
            score -= 3200.0
    elif matchup == "alakazam":
        if target_id in {96, 117}:
            score += 2600.0
        if target_id == 756 and not target_active:
            score -= 3500.0
    learned, support, positive = _learned(matchup, _phase(raw), "attach", source_id, target_id)
    score += learned * 0.65
    return score, {
        "source": source_id,
        "target": target_id,
        "before": before,
        "after": after,
        "learned": round(learned, 2),
        "support": support,
        "positive": positive,
        "hard": hard,
        "reasons": reasons,
    }


def _prize_value(card):
    data = CARDS.get(_cid(card))
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def _boss_ko_value(raw):
    _current, players, me = _players(raw)
    mine, opponent = players[me], players[1 - me]
    active = (mine.get("active") or [None])[0]
    damage = _ready_damage(active)
    if not damage:
        return 0
    best = 0
    for pokemon in opponent.get("bench") or []:
        if pokemon and int(pokemon.get("hp") or 0) <= damage:
            best = max(best, _prize_value(pokemon))
    return best


def _supporter_score(raw, option, matchup):
    current, players, me = _players(raw)
    mine, opponent = players[me], players[1 - me]
    source_id = _cid(_source(raw, option))
    own_hand = int(mine.get("handCount") or 0)
    opp_hand = int(opponent.get("handCount") or 0)
    own_prize = len(mine.get("prize") or [])
    opp_prize = len(opponent.get("prize") or [])
    score = 0.0
    hard = False
    reasons = []
    if source_id == 1227:
        score += max(0, 7 - own_hand) * 1500.0
        if own_hand <= 5:
            score += 3000.0
            hard = True
            reasons.append("low_hand_draw")
        if own_hand >= 9:
            score -= 3200.0
    elif source_id == 1197:
        score += max(0, opp_hand - 5) * (1450.0 if matchup == "alakazam" else 1000.0)
        if opp_hand >= 8:
            score += 3200.0
            hard = True
            reasons.append("large_opponent_hand")
        if opp_hand <= 5:
            score -= 3800.0
    elif source_id == 1182:
        prizes = _boss_ko_value(raw)
        if prizes:
            score += 7000.0 + prizes * 3500.0
            hard = True
            reasons.append(f"boss_ko_{prizes}")
        else:
            score -= 2800.0
    elif source_id == 1188:
        score += 1200.0 if own_hand >= 6 else -1800.0
    elif source_id == 1221:
        if own_prize > opp_prize:
            score += 2600.0
        if own_hand <= 4:
            score -= 1200.0
    learned, support, positive = _learned(matchup, _phase(raw), "supporter", source_id)
    score += learned * 0.55
    return score, {
        "source": source_id,
        "learned": round(learned, 2),
        "support": support,
        "positive": positive,
        "hard": hard,
        "reasons": reasons,
        "hands": [own_hand, opp_hand],
    }


def _board_and_hand(raw):
    _current, players, me = _players(raw)
    mine = players[me]
    board = [card for card in (mine.get("active") or []) + (mine.get("bench") or []) if card]
    hand = [card for card in mine.get("hand") or [] if card]
    return mine, board, hand


def _search_score(raw, option, matchup):
    source = _source(raw, option)
    source_id = _cid(source)
    if not source_id:
        return -1e9, {"reason": "missing_search_source"}
    mine, board, hand = _board_and_hand(raw)
    board_ids = [_cid(card) for card in board]
    hand_ids = [_cid(card) for card in hand]
    known = board_ids + hand_ids
    count = known.count(source_id)
    turn = int(((raw.get("current") or {}).get("turn") or 0))
    active = (mine.get("active") or [None])[0]
    active_ready = bool(active and _ready_damage(active) > 0)
    score = 0.0
    hard = False
    reasons = []
    data = CARDS.get(source_id)
    is_pokemon = bool(data and (data.basic or data.stage1 or data.stage2))
    if is_pokemon:
        if source_id == 96:
            if count == 0:
                score += 10000.0
                hard = True
                reasons.append("first_teal")
            elif turn <= 5 and count < 2:
                score += 5200.0
                hard = True
                reasons.append("second_teal")
            elif count >= 3:
                score -= 3500.0
        elif source_id == 272:
            if matchup in {"dragapult", "lucario"} and count == 0:
                score += 8200.0 if 96 in known else 6500.0
                hard = True
                reasons.append("clefairy_specialist")
            elif matchup not in {"dragapult", "lucario"}:
                score -= 1800.0
        elif source_id == 117:
            if matchup in {"crustle", "alakazam"} and count == 0:
                score += 7600.0 if (active_ready or 96 in known) else 5200.0
                hard = True
                reasons.append("cornerstone_specialist")
            elif matchup not in {"crustle", "alakazam"}:
                score -= 1200.0
        elif source_id == 184:
            if count == 0:
                score += 2800.0
            else:
                score -= 4800.0
        elif source_id == 756:
            if count == 0 and not board:
                score += 3200.0
            elif matchup in {"dragapult", "lucario", "alakazam"}:
                score -= 5200.0
            elif count >= 1:
                score -= 2800.0
        elif source_id == 1071:
            if count == 0 and not any(cid in hand_ids for cid in SUPPORTERS):
                score += 2200.0
            elif count:
                score -= 3500.0
        elif source_id in {31, 230}:
            if matchup == "crustle" and count == 0 and 117 in known:
                score += 3300.0
            elif count:
                score -= 2600.0
        elif source_id == 112:
            if matchup == "dragapult" and count == 0:
                score += 2600.0
            elif count:
                score -= 2500.0
        elif source_id == 140:
            if count == 0 and len(mine.get("prize") or []) <= 4:
                score += 1800.0
            elif count:
                score -= 3200.0
        if not active_ready and turn <= 6 and source_id in {117, 184, 272, 756} and 96 not in known:
            score -= 2800.0
    elif source_id == GRASS:
        grass_in_hand = hand_ids.count(GRASS)
        teal_count = board_ids.count(96)
        if teal_count and grass_in_hand == 0:
            score += 7200.0
            hard = True
            reasons.append("teal_energy")
        elif grass_in_hand >= 2:
            score -= 1800.0
    elif source_id == PRISM:
        tech_ids = TECH.get(matchup, set())
        if set(board_ids) & tech_ids and hand_ids.count(PRISM) == 0:
            score += 6200.0
            hard = True
            reasons.append("tech_prism")
        elif not (set(known) & tech_ids):
            score -= 2200.0
    elif source_id in SUPPORTERS:
        # The option is a search result rather than a PLAY option, but the same
        # public hand/field conditions determine which supporter is useful.
        fake = dict(option)
        fake["area"] = option.get("area", 1)
        support_score, detail = _supporter_score(raw, fake, matchup)
        score += support_score
        hard = bool(detail.get("hard"))
        reasons.extend(detail.get("reasons") or [])
    if source_id in hand_ids and source_id not in {GRASS, PRISM}:
        score -= 1800.0
    return score, {
        "source": source_id,
        "count_known": count,
        "active_ready": active_ready,
        "hard": hard,
        "reasons": reasons,
    }


def _choose_search(raw, base, matchup):
    select = raw.get("select") or {}
    options = select.get("option") or []
    context = int(select.get("context", -1))
    if context not in {TO_HAND, TO_BENCH, TO_FIELD, LOOK}:
        return base
    if not isinstance(base, list) or len(base) != 1 or int(select.get("minCount", 1) or 0) != 1 or int(select.get("maxCount", 1) or 0) != 1:
        return base
    STATS["search_windows"] += 1
    candidates = []
    for index, option in enumerate(options):
        score, detail = _search_score(raw, option, matchup)
        if score > -1e8:
            candidates.append((score, -index, index, detail))
    if len(candidates) < 2:
        return base
    candidates.sort(reverse=True)
    best_score, _, best_index, best_detail = candidates[0]
    try:
        base_index = int(base[0])
    except Exception:
        return base
    base_entry = next((entry for entry in candidates if entry[2] == base_index), None)
    if base_entry is None or best_index == base_index:
        return base
    base_score, _, __, base_detail = base_entry
    margin = best_score - base_score
    if margin < 5200.0 or not best_detail.get("hard"):
        return base
    STATS["overrides"] += 1
    STATS["search_overrides"] += 1
    _record({
        "matchup": matchup,
        "turn": int(((raw.get("current") or {}).get("turn") or 0)),
        "family": "search",
        "base": base_index,
        "chosen": best_index,
        "margin": round(margin, 2),
        "base_detail": base_detail,
        "chosen_detail": best_detail,
    })
    return [best_index]


def _setup_target(raw, matchup):
    mine, board, hand = _board_and_hand(raw)
    known = {_cid(card) for card in board + hand}
    active = (mine.get("active") or [None])[0]
    active_ready = bool(active and _ready_damage(active) > 0)
    if 96 not in known:
        return 96, "missing_first_attacker"
    if matchup in {"dragapult", "lucario"} and 272 not in known:
        return 272, "missing_clefairy"
    if matchup in {"crustle", "alakazam"} and 117 not in known and (active_ready or int(((raw.get("current") or {}).get("turn") or 0)) >= 4):
        return 117, "missing_cornerstone"
    if not active_ready:
        return 96, "first_attacker_not_ready"
    return None, None


def _setup_rescue(raw, base, matchup):
    select = raw.get("select") or {}
    options = select.get("option") or []
    if int(select.get("context", -1)) != MAIN or not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0])
        base_type = int(options[base_index].get("type", -1))
    except Exception:
        return base
    turn = int(((raw.get("current") or {}).get("turn") or 0))
    if base_type != 14 or turn > 8:
        return base
    mine, board, hand = _board_and_hand(raw)
    active = (mine.get("active") or [None])[0]
    if active and _ready_damage(active) > 0:
        return base
    target_id, reason = _setup_target(raw, matchup)
    if target_id is None:
        return base
    candidates = []
    # If the required Pokémon is already in hand, bench it before giving up the
    # turn.  This is stronger and cheaper than opening a search item.
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != PLAY:
            continue
        source_id = _cid(_source(raw, option))
        if source_id == target_id:
            candidates.append((12000, index, source_id, "play_target"))
    # Otherwise use the narrowest legal search item.  Tera Orb is preferred for
    # Teal/Cornerstone; Ultra Ball is required for Clefairy.
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != PLAY:
            continue
        source_id = _cid(_source(raw, option))
        if target_id in {96, 117} and source_id == 1127:
            candidates.append((10500, index, source_id, "tera_orb"))
        elif source_id == 1121:
            candidates.append((9000, index, source_id, "ultra_ball"))
    # A legal attachment or Teal Dance can itself rescue the attack clock.
    for index, option in enumerate(options):
        typ = int(option.get("type", -1))
        if typ == ATTACH:
            score, detail = _attachment_score(raw, option, matchup)
            if detail.get("after", 99) < detail.get("before", 99):
                candidates.append((7600 + max(0, score), index, detail.get("source"), "progress_attachment"))
        elif typ == 10 and _cid(_source(raw, option)) == 96:
            candidates.append((8000, index, 96, "teal_dance"))
    if not candidates:
        return base
    candidates.sort(reverse=True)
    value, best_index, source_id, action_reason = candidates[0]
    STATS["overrides"] += 1
    STATS["setup_rescue_overrides"] += 1
    _record({
        "matchup": matchup,
        "turn": turn,
        "family": "setup_rescue",
        "base": base_index,
        "chosen": best_index,
        "margin": value,
        "chosen_detail": {"source": source_id, "target": target_id, "reason": reason, "action": action_reason},
    })
    return [best_index]


def choose(raw, base, matchup="generic"):
    STATS["decisions"] += 1
    # Same-seed regression gates rejected both interventions: Dragapult fell in
    # the 100-game gate, and Lucario fell from 8/20 to 6/20 despite attacking
    # earlier.  Preserve their retained v28 policies until causal evidence is
    # positive on match wins as well as intermediate tempo.
    if matchup in {"dragapult", "lucario"}:
        return base
    select = raw.get("select") or {}
    options = select.get("option") or []
    if int(select.get("context", -1)) != MAIN:
        return _choose_search(raw, base, matchup)
    if not isinstance(base, list) or len(base) != 1:
        return base
    try:
        base_index = int(base[0])
        base_option = options[base_index]
    except Exception:
        return base
    rescued = _setup_rescue(raw, base, matchup)
    if rescued != base:
        return rescued
    base_type = int(base_option.get("type", -1))
    family = None
    if base_type == ATTACH:
        family = "attach"
        STATS["attachment_windows"] += 1
    elif base_type == PLAY and _cid(_source(raw, base_option)) in SUPPORTERS:
        family = "supporter"
        STATS["supporter_windows"] += 1
    if family is None:
        return base
    candidates = []
    for index, option in enumerate(options):
        typ = int(option.get("type", -1))
        if family == "attach" and typ == ATTACH:
            score, detail = _attachment_score(raw, option, matchup)
        elif family == "supporter" and typ == PLAY and _cid(_source(raw, option)) in SUPPORTERS:
            score, detail = _supporter_score(raw, option, matchup)
        else:
            continue
        candidates.append((score, -index, index, detail))
    if len(candidates) < 2:
        return base
    candidates.sort(reverse=True)
    best_score, _, best_index, best_detail = candidates[0]
    base_entry = next((entry for entry in candidates if entry[2] == base_index), None)
    if base_entry is None:
        return base
    base_score, _, __, base_detail = base_entry
    margin = best_score - base_score
    if best_index == base_index:
        return base
    supported = best_detail.get("support", 0) >= (4 if family == "attach" else 2) and best_detail.get("positive", 0.0) >= 0.55
    hard = bool(best_detail.get("hard"))
    threshold = 2600.0 if family == "attach" else 4800.0
    if margin < threshold or not (supported or hard):
        return base
    STATS["overrides"] += 1
    STATS[family + "_overrides"] += 1
    _record({
        "matchup": matchup,
        "turn": int(((raw.get("current") or {}).get("turn") or 0)),
        "family": family,
        "base": base_index,
        "chosen": best_index,
        "margin": round(margin, 2),
        "base_detail": base_detail,
        "chosen_detail": best_detail,
    })
    return [best_index]
