from __future__ import annotations

import math
from typing import Any, Iterable

ALAKAZAM_IDS = {743, 245}
LINE_IDS = {741, 742, 743, 245}
DRAW_SUPPORT_IDS = {66, 140}
EX_LIABILITY_IDS = {140, 272}
PROTECTIVE_ENERGY_IDS = {11, 20}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _id(card: Any) -> int:
    if card is None:
        return 0
    return int(_get(card, "id", _get(card, "cardId", 0)) or 0)


def _cards(zone: Any) -> list[Any]:
    return [card for card in (zone or []) if card is not None]


def _field(player: Any) -> list[Any]:
    return _cards(_get(player, "active", [])) + _cards(_get(player, "bench", []))


def _energy_cards(pokemon: Any) -> list[Any]:
    cards = _cards(_get(pokemon, "energyCards", []))
    if cards:
        return cards
    energies = _get(pokemon, "energies", []) or []
    expanded: list[dict[str, int]] = []
    for energy_id, count in enumerate(energies):
        expanded.extend({"id": energy_id} for _ in range(int(count or 0)))
    return expanded


def _energy_count(pokemon: Any) -> int:
    return len(_energy_cards(pokemon))


def _hp_ratio(pokemon: Any) -> float:
    if pokemon is None:
        return 0.0
    hp = float(_get(pokemon, "hp", 0) or 0)
    max_hp = float(_get(pokemon, "maxHp", 0) or 0)
    return hp / max_hp if max_hp > 0 else 0.0


def _ready_attackers(player: Any) -> float:
    ready = 0.0
    for pokemon in _field(player):
        cid = _id(pokemon)
        energy = _energy_count(pokemon)
        if cid in ALAKAZAM_IDS and energy >= 1:
            ready += 1.0
        elif cid == 272 and energy >= 2:
            ready += 1.0
        elif energy >= 2:
            ready += 0.45
        elif energy >= 1:
            ready += 0.15
    return ready


def _line_quality(player: Any) -> float:
    score = 0.0
    for pokemon in _field(player):
        cid = _id(pokemon)
        if cid == 741:
            score += 0.25
        elif cid == 742:
            score += 0.55
        elif cid in ALAKAZAM_IDS:
            score += 1.0
    return score


def _special_energy_count(player: Any) -> int:
    count = 0
    for pokemon in _field(player):
        count += sum(1 for energy in _energy_cards(pokemon) if _id(energy) in PROTECTIVE_ENERGY_IDS)
    return count


def _bench_liability(player: Any) -> float:
    liability = 0.0
    for pokemon in _cards(_get(player, "bench", [])):
        if _id(pokemon) in EX_LIABILITY_IDS and _energy_count(pokemon) == 0:
            liability += 1.0
    return liability


def _active(player: Any) -> Any:
    cards = _cards(_get(player, "active", []))
    return cards[0] if cards else None


def state_components(obs: Any, perspective: int) -> dict[str, float]:
    current = _get(obs, "current")
    if current is None:
        return {"total": 0.0}

    result = int(_get(current, "result", -1))
    if result != -1:
        terminal = 1.0 if result == perspective else -1.0
        return {"terminal": terminal, "total": terminal}

    players = _get(current, "players", [])
    me = players[perspective]
    op = players[1 - perspective]
    my_active = _active(me)
    op_active = _active(op)

    my_prizes = len(_get(me, "prize", []) or [])
    op_prizes = len(_get(op, "prize", []) or [])
    my_hand = float(_get(me, "handCount", len(_get(me, "hand", []) or [])) or 0)
    op_hand = float(_get(op, "handCount", len(_get(op, "hand", []) or [])) or 0)
    my_deck = float(_get(me, "deckCount", 0) or 0)
    op_deck = float(_get(op, "deckCount", 0) or 0)

    prize_adv = (op_prizes - my_prizes) / 6.0
    hand_adv = max(-1.0, min(1.0, (my_hand - op_hand) / 10.0))
    ready_adv = max(-1.5, min(1.5, _ready_attackers(me) - _ready_attackers(op))) / 1.5
    line_adv = max(-2.0, min(2.0, _line_quality(me) - _line_quality(op))) / 2.0
    hp_adv = _hp_ratio(my_active) - _hp_ratio(op_active)
    deck_safety = 0.0
    if my_deck <= 3:
        deck_safety -= (4.0 - my_deck) / 4.0
    if op_deck <= 3:
        deck_safety += (4.0 - op_deck) / 4.0
    protected_pressure = min(1.0, _special_energy_count(op) / 2.0)
    has_damage_attacker = any(_id(card) == 245 for card in _field(me))
    protection_answer = protected_pressure * (1.0 if has_damage_attacker else -1.0)
    liability = max(-1.0, min(1.0, _bench_liability(op) - _bench_liability(me)))

    raw = (
        2.80 * prize_adv
        + 0.35 * hand_adv
        + 1.25 * ready_adv
        + 0.65 * line_adv
        + 0.40 * hp_adv
        + 0.80 * deck_safety
        + 0.35 * protection_answer
        + 0.30 * liability
    )
    total = math.tanh(raw / 3.0)
    return {
        "prize_adv": prize_adv,
        "hand_adv": hand_adv,
        "ready_adv": ready_adv,
        "line_adv": line_adv,
        "hp_adv": hp_adv,
        "deck_safety": deck_safety,
        "protection_answer": protection_answer,
        "liability": liability,
        "total": total,
    }


def shaped_state_score(obs: Any, perspective: int) -> float:
    return float(state_components(obs, perspective)["total"])


def trajectory_target(
    outcome: float,
    state_score: float,
    turn: int,
    final_turn: int,
) -> float:
    """Blend local state quality and final result.

    Early states receive less final-outcome credit, while late states receive more.
    This reduces the old failure mode where one late mistake labels every earlier
    good state as entirely bad.
    """
    progress = 0.0 if final_turn <= 0 else max(0.0, min(1.0, turn / final_turn))
    outcome_weight = 0.35 + 0.45 * progress
    target = outcome_weight * outcome + (1.0 - outcome_weight) * state_score
    return max(-1.0, min(1.0, target))


def transition_utility(root_obs: Any, child_obs: Any, perspective: int) -> float:
    root = shaped_state_score(root_obs, perspective)
    child = shaped_state_score(child_obs, perspective)
    current = _get(child_obs, "current")
    result = int(_get(current, "result", -1)) if current is not None else -1
    terminal_bonus = 0.0
    if result != -1:
        terminal_bonus = 1.0 if result == perspective else -1.0
    utility = 0.55 * child + 0.35 * (child - root) + 0.10 * terminal_bonus
    return max(-1.0, min(1.0, utility))
