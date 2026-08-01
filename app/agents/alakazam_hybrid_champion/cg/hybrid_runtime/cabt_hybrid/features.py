from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

# Important IDs for the current deck and common opponents.
KEY_CARD_IDS = [
    741, 742, 743, 245, 305, 66, 140, 272,
    1079, 1081, 1120, 1086, 1097, 1129, 1152, 1156,
    1182, 1197, 1225, 1231, 1264, 5, 19, 13,
    119, 120, 121, 344, 345, 414, 677, 678, 11, 20,
]

OPTION_TYPES = list(range(17))
ACTION_OPTION_TYPES = list(range(17))


def _card_id(card: Any) -> int:
    if card is None:
        return 0
    if isinstance(card, dict):
        return int(card.get("id", card.get("cardId", 0)) or 0)
    return int(getattr(card, "id", getattr(card, "cardId", 0)) or 0)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_cards(zone: Any) -> Iterable[Any]:
    if not zone:
        return []
    return [card for card in zone if card is not None]


def _attached_energy_count(pokemon: Any) -> int:
    if pokemon is None:
        return 0
    cards = _get(pokemon, "energyCards", []) or []
    if cards:
        return len(cards)
    energies = _get(pokemon, "energies", []) or []
    return int(sum(int(x) for x in energies)) if energies else 0


def _hp_ratio(pokemon: Any) -> float:
    if pokemon is None:
        return 0.0
    hp = float(_get(pokemon, "hp", 0) or 0)
    max_hp = float(_get(pokemon, "maxHp", 0) or 0)
    return hp / max_hp if max_hp > 0 else 0.0


def _zone_counter(player: Any, zone_name: str) -> Counter[int]:
    return Counter(_card_id(c) for c in _iter_cards(_get(player, zone_name, [])))


def _field_counter(player: Any) -> Counter[int]:
    cards = list(_iter_cards(_get(player, "active", []))) + list(
        _iter_cards(_get(player, "bench", []))
    )
    return Counter(_card_id(c) for c in cards)


def feature_names() -> list[str]:
    names = [
        "turn", "turn_action_count", "is_first_player", "supporter_played",
        "stadium_played", "energy_attached", "retreated",
        "my_prizes", "opp_prizes", "my_deck", "opp_deck",
        "my_hand", "opp_hand", "my_bench", "opp_bench",
        "my_active_hp_ratio", "opp_active_hp_ratio",
        "my_active_energy", "opp_active_energy",
        "my_active_id_scaled", "opp_active_id_scaled",
    ]
    for prefix in ["my_hand", "my_field", "my_discard", "opp_field", "opp_discard"]:
        names.extend(f"{prefix}_id_{cid}" for cid in KEY_CARD_IDS)
    names.extend(f"option_type_{t}" for t in OPTION_TYPES)
    names.extend(["select_min", "select_max", "remain_damage_counter", "remain_energy_cost"])
    return names


FEATURE_DIM = len(feature_names())


def extract_features(obs: Any, perspective: int | None = None) -> np.ndarray:
    """Convert an observation dict/dataclass into a fixed-size float32 vector."""
    current = _get(obs, "current")
    if current is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    current_player = int(_get(current, "yourIndex", 0) or 0)
    me = current_player if perspective is None else int(perspective)
    opp = 1 - me
    players = _get(current, "players", [])
    my = players[me]
    op = players[opp]

    my_active = next(iter(_iter_cards(_get(my, "active", []))), None)
    op_active = next(iter(_iter_cards(_get(op, "active", []))), None)

    values: list[float] = [
        float(_get(current, "turn", 0) or 0) / 50.0,
        float(_get(current, "turnActionCount", 0) or 0) / 30.0,
        1.0 if int(_get(current, "firstPlayer", -1)) == me else 0.0,
        float(bool(_get(current, "supporterPlayed", False))),
        float(bool(_get(current, "stadiumPlayed", False))),
        float(bool(_get(current, "energyAttached", False))),
        float(bool(_get(current, "retreated", False))),
        len(_get(my, "prize", []) or []) / 6.0,
        len(_get(op, "prize", []) or []) / 6.0,
        float(_get(my, "deckCount", 0) or 0) / 60.0,
        float(_get(op, "deckCount", 0) or 0) / 60.0,
        float(_get(my, "handCount", len(_get(my, "hand", []) or [])) or 0) / 20.0,
        float(_get(op, "handCount", len(_get(op, "hand", []) or [])) or 0) / 20.0,
        len(_get(my, "bench", []) or []) / 8.0,
        len(_get(op, "bench", []) or []) / 8.0,
        _hp_ratio(my_active),
        _hp_ratio(op_active),
        _attached_energy_count(my_active) / 6.0,
        _attached_energy_count(op_active) / 6.0,
        _card_id(my_active) / 1300.0,
        _card_id(op_active) / 1300.0,
    ]

    counters = [
        _zone_counter(my, "hand"),
        _field_counter(my),
        _zone_counter(my, "discard"),
        _field_counter(op),
        _zone_counter(op, "discard"),
    ]
    for counter in counters:
        values.extend(min(counter[cid], 4) / 4.0 for cid in KEY_CARD_IDS)

    select = _get(obs, "select")
    option_counts = Counter()
    if select is not None:
        for option in _get(select, "option", []) or []:
            t = int(_get(option, "type", -1))
            option_counts[t] += 1
    values.extend(min(option_counts[t], 8) / 8.0 for t in OPTION_TYPES)
    values.extend(
        [
            float(_get(select, "minCount", 0) or 0) / 8.0 if select else 0.0,
            float(_get(select, "maxCount", 0) or 0) / 8.0 if select else 0.0,
            float(_get(select, "remainDamageCounter", 0) or 0) / 30.0 if select else 0.0,
            float(_get(select, "remainEnergyCost", 0) or 0) / 10.0 if select else 0.0,
        ]
    )

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"Feature size mismatch: {vector.shape}, expected {(FEATURE_DIM,)}")
    return vector


def action_feature_names() -> list[str]:
    names = [f"action_type_{t}" for t in ACTION_OPTION_TYPES]
    names.extend(
        [
            "action_count",
            "same_as_rule",
            "mean_attack_id",
            "mean_card_id",
            "mean_area",
            "mean_in_play_area",
            "mean_player_index",
            "contains_attack",
            "contains_end",
            "contains_evolve",
            "contains_attach",
            "contains_play",
            "contains_retreat",
        ]
    )
    return names


ACTION_FEATURE_DIM = len(action_feature_names())
ACTION_INPUT_DIM = FEATURE_DIM + ACTION_FEATURE_DIM


def extract_action_features(select: Any, action: list[int], rule_action: list[int] | None = None) -> np.ndarray:
    options = list(_get(select, "option", []) or [])
    chosen = [options[i] for i in action if 0 <= i < len(options)]
    counts = Counter(int(_get(option, "type", -1)) for option in chosen)
    values: list[float] = [min(counts[t], 4) / 4.0 for t in ACTION_OPTION_TYPES]

    def mean_field(name: str, scale: float) -> float:
        vals = [float(_get(option, name, 0) or 0) for option in chosen]
        return (sum(vals) / len(vals) / scale) if vals else 0.0

    action_types = set(counts)
    values.extend(
        [
            len(action) / 8.0,
            1.0 if rule_action is not None and sorted(action) == sorted(rule_action) else 0.0,
            mean_field("attackId", 1500.0),
            mean_field("cardId", 1500.0),
            mean_field("area", 10.0),
            mean_field("inPlayArea", 10.0),
            mean_field("playerIndex", 1.0),
            1.0 if 13 in action_types else 0.0,
            1.0 if 14 in action_types else 0.0,
            1.0 if 9 in action_types else 0.0,
            1.0 if 8 in action_types else 0.0,
            1.0 if 7 in action_types else 0.0,
            1.0 if 12 in action_types else 0.0,
        ]
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (ACTION_FEATURE_DIM,):
        raise RuntimeError(f"Action feature mismatch: {vector.shape}, expected {(ACTION_FEATURE_DIM,)}")
    return vector


def extract_action_input(
    obs: Any,
    action: list[int],
    perspective: int,
    rule_action: list[int] | None = None,
) -> np.ndarray:
    return np.concatenate(
        [
            extract_features(obs, perspective=perspective),
            extract_action_features(_get(obs, "select"), action, rule_action),
        ]
    ).astype(np.float32, copy=False)
