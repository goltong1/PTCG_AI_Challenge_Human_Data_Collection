from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .beam_search import (
    SearchConfig,
    choose_action_ranker_only,
    choose_action_with_beam_search,
)
from .hidden_info import ArchetypeInference, ArchetypeTracker, infer_opponent_archetype, predict_search_inputs
from .import_utils import load_agent, read_deck
from .model import load_action_checkpoint, load_value_checkpoint


class HybridAgent:
    def __init__(
        self,
        project_root: Path,
        search_config_path: Path,
        mode_override: str | None = None,
        value_model_override: Path | None = None,
        action_model_override: Path | None = None,
    ) -> None:
        self.root = project_root.resolve()
        self.base_dir = self.root / "agents" / "base"
        self.opponent_dirs = sorted((self.root / "agents" / "opponents").iterdir())
        self.rule_module = load_agent(self.base_dir, "hybrid_base_rule")
        self.rule_agent = getattr(self.rule_module, "_rule_agent", None)
        if self.rule_agent is None:
            self.rule_agent = getattr(self.rule_module, "agent")
        self.own_deck = read_deck(self.base_dir)

        raw = json.loads(search_config_path.read_text(encoding="utf-8"))
        self.mode = str(mode_override or raw.get("mode", "hybrid"))
        self.config = SearchConfig(
            beam_width=int(raw["beam_width"]),
            max_depth=int(raw["max_depth"]),
            max_actions_per_node=int(raw["max_actions_per_node"]),
            time_budget_ms=int(raw["time_budget_ms"]),
            manual_coin=bool(raw.get("manual_coin", False)),
            mode=self.mode,
            heuristic_weight=float(raw.get("heuristic_weight", 1.0)),
            value_model_weight=float(raw.get("value_model_weight", 0.10)),
            action_model_weight=float(raw.get("action_model_weight", 0.10)),
            rule_action_bonus=float(raw.get("rule_action_bonus", 90.0)),
            override_margin=float(raw.get("override_margin", 35.0)),
            strong_rule_override_margin=float(raw.get("strong_rule_override_margin", 180.0)),
            min_ranker_confidence=float(raw.get("min_ranker_confidence", 0.08)),
        )
        self.use_search = bool(raw.get("use_search", True))

        archetype = raw.get("archetype_inference", {})
        self.archetype_high_confidence = float(archetype.get("high_confidence", 0.68))
        self.archetype_medium_confidence = float(archetype.get("medium_confidence", 0.45))
        self.archetype_minimum_evidence = float(archetype.get("minimum_evidence_weight", 2.0))
        self.mixed_model_scale = float(archetype.get("mixed_model_weight_scale", 0.55))
        self.unknown_model_scale = float(archetype.get("unknown_model_weight_scale", 0.0))
        self.unknown_rule_bonus = float(archetype.get("unknown_rule_action_bonus", 150.0))
        self.unknown_override_margin = float(archetype.get("unknown_override_margin", 70.0))
        self.unknown_strong_margin = float(archetype.get("unknown_strong_rule_override_margin", 260.0))
        self.mixed_override_scale = float(archetype.get("mixed_override_scale", 1.35))
        self.tracker = ArchetypeTracker()
        self.last_inference: ArchetypeInference | None = None

        value_path = value_model_override or (self.root / raw.get("value_model_path", "data/models/champion_value.npz"))
        action_path = action_model_override or (self.root / raw.get("action_model_path", "data/models/champion_action.npz"))
        self.value_model = load_value_checkpoint(value_path) if value_path.is_file() else None
        self.action_model = load_action_checkpoint(action_path) if action_path.is_file() else None

    def _infer(self, obs_dict: dict, perspective: int) -> ArchetypeInference:
        inference = infer_opponent_archetype(
            obs_dict,
            self.opponent_dirs,
            perspective,
            tracker=self.tracker,
            high_confidence=self.archetype_high_confidence,
            medium_confidence=self.archetype_medium_confidence,
            minimum_evidence_weight=self.archetype_minimum_evidence,
        )
        self.last_inference = inference
        return inference

    def _adaptive_search_setup(self, inference: ArchetypeInference):
        config = replace(self.config)
        value_model = self.value_model
        action_model = self.action_model

        if inference.is_unknown:
            # Models were trained on known archetypes. On an out-of-distribution
            # deck, use board-state heuristics and the proven generic rule plan.
            config.max_depth = min(config.max_depth, 5)
            config.max_actions_per_node = min(config.max_actions_per_node, 4)
            config.heuristic_weight = max(config.heuristic_weight, 1.10)
            config.value_model_weight *= self.unknown_model_scale
            config.action_model_weight *= self.unknown_model_scale
            config.rule_action_bonus = max(config.rule_action_bonus, self.unknown_rule_bonus)
            config.override_margin = max(config.override_margin, self.unknown_override_margin)
            config.strong_rule_override_margin = max(
                config.strong_rule_override_margin, self.unknown_strong_margin
            )
            config.min_ranker_confidence = max(config.min_ranker_confidence, 0.16)
            if self.unknown_model_scale <= 0.0:
                value_model = None
                action_model = None
        elif inference.is_mixed:
            # Partial evidence: keep learned knowledge, but do not let a guessed
            # matchup override strong setup/attack rules without a wider margin.
            config.value_model_weight *= self.mixed_model_scale
            config.action_model_weight *= self.mixed_model_scale
            config.rule_action_bonus *= 1.15
            config.override_margin *= self.mixed_override_scale
            config.strong_rule_override_margin *= self.mixed_override_scale
            config.min_ranker_confidence = max(config.min_ranker_confidence, 0.11)

        return config, value_model, action_model

    def act(self, obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return self.own_deck

        fallback = self.rule_agent(obs_dict)
        if self.mode == "rule":
            return fallback

        select = obs_dict.get("select") or {}
        select_type = select.get("type")
        if select_type not in ("Main", 0):
            return fallback

        if self.mode == "hybrid" and self.value_model is None and self.action_model is None:
            return fallback
        if self.mode == "value_search" and self.value_model is None:
            return fallback
        if self.mode == "action_search" and self.action_model is None:
            return fallback

        perspective = int(obs_dict["current"]["yourIndex"])
        try:
            inference = self._infer(obs_dict, perspective)
        except Exception:
            return fallback

        if self.mode == "ranker_only":
            if inference.is_unknown:
                return fallback
            confidence = self.config.min_ranker_confidence
            if inference.is_mixed:
                confidence = max(confidence, 0.12)
            return choose_action_ranker_only(
                obs_dict,
                self.rule_agent,
                self.action_model,
                self.config.max_actions_per_node,
                confidence,
            )

        if not self.use_search or not obs_dict.get("search_begin_input"):
            return fallback

        try:
            adaptive_config, value_model, action_model = self._adaptive_search_setup(inference)
            if self.mode == "heuristic_search":
                value_model = None
                action_model = None
            elif self.mode == "value_search":
                action_model = None
            elif self.mode == "action_search":
                value_model = None

            hidden = predict_search_inputs(obs_dict, self.own_deck, inference.template, perspective)
            return choose_action_with_beam_search(
                obs_dict,
                self.rule_agent,
                value_model,
                action_model,
                self.own_deck,
                inference.template,
                hidden,
                adaptive_config,
            )
        except Exception:
            return fallback
