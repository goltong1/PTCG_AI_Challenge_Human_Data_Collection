from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .features import extract_action_input, extract_features
from .rewards import shaped_state_score
from .serialization import to_plain_dict


@dataclass
class SearchConfig:
    beam_width: int = 4
    max_depth: int = 7
    max_actions_per_node: int = 5
    time_budget_ms: int = 80
    manual_coin: bool = False
    mode: str = "hybrid"
    heuristic_weight: float = 1.0
    value_model_weight: float = 0.10
    action_model_weight: float = 0.10
    rule_action_bonus: float = 90.0
    override_margin: float = 35.0
    strong_rule_override_margin: float = 180.0
    min_ranker_confidence: float = 0.08


@dataclass
class BeamNode:
    search_id: int
    observation: Any
    first_action: list[int] | None
    score: float
    depth: int


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def option_heuristic(option: Any) -> float:
    t = int(_get(option, "type", -1))
    base = {
        9: 90.0,   # evolve
        8: 85.0,   # attach
        10: 78.0,  # ability
        13: 75.0,  # attack
        7: 65.0,   # play card
        12: 35.0,  # retreat
        11: 20.0,
        14: -30.0, # end
        1: 5.0,
        2: 0.0,
        3: 10.0,
        4: 10.0,
        5: 10.0,
        6: 10.0,
        0: 0.0,
        15: 10.0,
        16: 10.0,
    }.get(t, 0.0)
    attack_id = _get(option, "attackId")
    if attack_id == 338:  # Strange Hacking can create long selection loops.
        base -= 120.0
    return base


def candidate_actions(select: Any, rule_action: list[int] | None, limit: int) -> list[list[int]]:
    options = list(_get(select, "option", []) or [])
    minimum = int(_get(select, "minCount", 0) or 0)
    maximum = int(_get(select, "maxCount", 0) or 0)
    if not options:
        return [[]] if minimum == 0 else []

    candidates: list[list[int]] = []
    if rule_action is not None and minimum <= len(rule_action) <= maximum:
        candidates.append(list(rule_action))

    ranked = sorted(range(len(options)), key=lambda i: option_heuristic(options[i]), reverse=True)
    if minimum <= 1 <= maximum:
        candidates.extend([[i] for i in ranked[: max(limit, 1)]])
    if minimum == 0:
        candidates.append([])

    if minimum >= 2:
        pool = ranked[: min(len(ranked), 7)]
        for count in range(minimum, min(maximum, minimum + 1) + 1):
            for combo in itertools.combinations(pool, count):
                candidates.append(list(combo))
                if len(candidates) >= limit * 2:
                    break
            if len(candidates) >= limit * 2:
                break

    unique: list[list[int]] = []
    seen = set()
    for action in candidates:
        key = tuple(sorted(action))
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique[:limit]


def heuristic_value(obs: Any, perspective: int) -> float:
    current = _get(obs, "current")
    if current is None:
        return 0.0
    result = int(_get(current, "result", -1))
    if result != -1:
        return 1_000_000.0 if result == perspective else -1_000_000.0
    # Rich shaped score is bounded [-1, 1]; map it to a stable search scale.
    return shaped_state_score(obs, perspective) * 1000.0


def value_prediction(model: Any | None, obs: Any, perspective: int) -> float:
    if model is None:
        return 0.0
    raw = float(model.predict(extract_features(obs, perspective=perspective)))
    return math.tanh(raw) * 1000.0


def action_prediction(
    model: Any | None,
    obs: Any,
    action: list[int],
    perspective: int,
    rule_action: list[int] | None,
) -> float:
    if model is None:
        return 0.0
    raw = float(model.predict(
        extract_action_input(obs, action, perspective=perspective, rule_action=rule_action)
    ))
    return math.tanh(raw) * 1000.0


def _first_option_type(select: Any, action: list[int]) -> int:
    options = list(_get(select, "option", []) or [])
    if len(action) != 1 or not (0 <= action[0] < len(options)):
        return -1
    return int(_get(options[action[0]], "type", -1))


def choose_action_ranker_only(
    obs_dict: dict,
    rule_agent: Callable[[dict], list[int]],
    action_model: Any | None,
    max_actions: int,
    min_confidence: float,
) -> list[int]:
    fallback = rule_agent(obs_dict)
    if action_model is None:
        return fallback
    select = obs_dict.get("select") or {}
    current = obs_dict.get("current") or {}
    perspective = int(current.get("yourIndex", 0))
    actions = candidate_actions(select, fallback, max_actions)
    if len(actions) < 2:
        return fallback
    scored = [
        (action_prediction(action_model, obs_dict, action, perspective, fallback), action)
        for action in actions
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    confidence = (scored[0][0] - scored[1][0]) / 1000.0
    return scored[0][1] if confidence >= min_confidence else fallback


def choose_action_with_beam_search(
    obs_dict: dict,
    rule_agent: Callable[[dict], list[int]],
    value_model: Any | None,
    action_model: Any | None,
    own_deck: list[int],
    opponent_template: list[int],
    hidden_inputs: dict[str, list[int]],
    config: SearchConfig,
) -> list[int]:
    from cg.api import search_begin, search_end, search_release, search_step, to_observation_class

    fallback = rule_agent(obs_dict)
    obs = to_observation_class(obs_dict)
    if obs.current is None or obs.select is None:
        return fallback
    perspective = int(obs.current.yourIndex)
    root_turn = int(obs.current.turn)
    deadline = time.perf_counter() + config.time_budget_ms / 1000.0

    try:
        root = search_begin(
            obs,
            hidden_inputs["your_deck"],
            hidden_inputs["your_prize"],
            hidden_inputs["opponent_deck"],
            hidden_inputs["opponent_prize"],
            hidden_inputs["opponent_hand"],
            hidden_inputs["opponent_active"],
            manual_coin=config.manual_coin,
        )
    except Exception:
        return fallback

    first_action_scores: dict[tuple[int, ...], float] = {}
    root_plain = to_plain_dict(root.observation)
    root_action_scores: dict[tuple[int, ...], float] = {}
    root_actions = candidate_actions(obs.select, fallback, config.max_actions_per_node)
    ranker_values = []
    for action in root_actions:
        ranker = action_prediction(action_model, root_plain, action, perspective, fallback)
        root_action_scores[tuple(action)] = ranker
        ranker_values.append(ranker)
    ranker_values.sort(reverse=True)
    ranker_confidence = (
        (ranker_values[0] - ranker_values[1]) / 1000.0 if len(ranker_values) >= 2 else 0.0
    )
    if action_model is None:
        ranker_confidence = 1.0

    root_score = (
        config.heuristic_weight * heuristic_value(root.observation, perspective)
        + config.value_model_weight * value_prediction(value_model, root.observation, perspective)
    )
    beam = [BeamNode(root.searchId, root.observation, None, root_score, 0)]

    try:
        for depth in range(config.max_depth):
            if time.perf_counter() >= deadline:
                break
            children: list[BeamNode] = []
            for node in beam:
                if time.perf_counter() >= deadline:
                    break
                node_obs = node.observation
                current = node_obs.current
                select = node_obs.select
                if current is None or int(current.result) != -1 or select is None:
                    if node.first_action is not None:
                        key = tuple(node.first_action)
                        first_action_scores[key] = max(first_action_scores.get(key, -float("inf")), node.score)
                    continue

                if int(current.turn) != root_turn:
                    score = (
                        config.heuristic_weight * heuristic_value(node_obs, perspective)
                        + config.value_model_weight * value_prediction(value_model, node_obs, perspective)
                    )
                    if node.first_action is not None:
                        key = tuple(node.first_action)
                        first_action_scores[key] = max(first_action_scores.get(key, -float("inf")), score)
                    continue

                node_dict = to_plain_dict(node_obs)
                try:
                    rule_choice = rule_agent(node_dict)
                except Exception:
                    rule_choice = None
                actions = candidate_actions(select, rule_choice, config.max_actions_per_node)

                for action in actions:
                    if time.perf_counter() >= deadline:
                        break
                    try:
                        child = search_step(node.search_id, action)
                    except Exception:
                        continue
                    first = action if node.first_action is None else node.first_action
                    first_key = tuple(first)
                    score = (
                        config.heuristic_weight * heuristic_value(child.observation, perspective)
                        + config.value_model_weight * value_prediction(value_model, child.observation, perspective)
                        + config.action_model_weight * root_action_scores.get(first_key, 0.0)
                        + (config.rule_action_bonus if sorted(first) == sorted(fallback) else 0.0)
                    )
                    children.append(BeamNode(child.searchId, child.observation, first, score, depth + 1))

            if not children:
                break
            children.sort(key=lambda node: node.score, reverse=True)
            for pruned in children[config.beam_width:]:
                try:
                    search_release(pruned.search_id)
                except Exception:
                    pass
            beam = children[: config.beam_width]
            if depth == config.max_depth - 1 or time.perf_counter() >= deadline:
                for leaf in beam:
                    if leaf.first_action is not None:
                        key = tuple(leaf.first_action)
                        first_action_scores[key] = max(first_action_scores.get(key, -float("inf")), leaf.score)
    finally:
        try:
            search_end()
        except Exception:
            pass

    fallback_key = tuple(fallback)
    fallback_score = first_action_scores.get(fallback_key, -float("inf"))
    if not first_action_scores or not math.isfinite(fallback_score):
        return fallback
    best_key, best_score = max(first_action_scores.items(), key=lambda item: item[1])
    best_action = list(best_key)
    if best_key == fallback_key:
        return fallback

    margin = best_score - fallback_score
    fallback_type = _first_option_type(obs.select, fallback)
    required_margin = (
        config.strong_rule_override_margin if fallback_type in {8, 9, 13} else config.override_margin
    )
    # If the ranker is uncertain, only allow a very large search improvement.
    if ranker_confidence < config.min_ranker_confidence:
        required_margin = max(required_margin, config.strong_rule_override_margin)
    return best_action if margin >= required_margin else fallback
