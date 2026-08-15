"""Cross-vintage causal multi-head attention observer for Lucario v152.

The observer consumes the same ordered public event/decision tokens used during
training and exposes risk/phase telemetry.  It is deliberately incapable of
changing an action: ``ACTION_AUTHORITY`` is a source-level constant and
``choose`` returns the exact input object.
"""
from __future__ import annotations

import json
import math
import os


ACTION_AUTHORITY = False


def _int(value, default=0):
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _sigmoid(value):
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


class TemporalAttentionObserver:
    """Six-member causal decayed-attention ensemble with telemetry-only output."""

    def __init__(self, root, history, token_module, replay_module, model_file="temporal_attention_observer_model.json"):
        self.history = history
        self.token_module = token_module
        self.replay_module = replay_module
        self.load_error = None
        try:
            with open(os.path.join(root, model_file), encoding="utf-8") as handle:
                self.model = json.load(handle)
        except Exception as exc:
            self.model = {"enabled": False, "members": []}
            self.load_error = type(exc).__name__
        self.input_dim = max(1, _int(self.model.get("input_dim"), 32))
        self.action_dim = max(1, _int(self.model.get("action_dim"), 96))
        self.heads = max(1, _int(self.model.get("heads"), 2))
        self.width = max(1, _int(self.model.get("width"), 8))
        self.members = list(self.model.get("members") or [])
        self.enabled = bool(self.model.get("enabled", False) and self.members)
        # The model file cannot enable action authority.  This constant is the
        # only value consulted by runtime code and is intentionally False.
        self.action_authority = ACTION_AUTHORITY
        self.reset()

    def reset(self):
        self.event_cursor = 0
        self.decision_cursor = 0
        self.states = [
            {
                "sums": [[0.0] * self.width for _ in range(self.heads)],
                "norms": [0.0] * self.heads,
            }
            for _ in self.members
        ]
        self.stats = {
            "enabled": self.enabled,
            "action_authority": False,
            "calls": 0,
            "passthrough_calls": 0,
            "overrides": 0,
            "override_attempts": 0,
            "synced_events": 0,
            "synced_decisions": 0,
            "tokens": 0,
            "outcome_samples": 0,
            "win_sum": 0.0,
            "win_min": 1.0,
            "win_max": 0.0,
            "phase_counts": {},
            "sync_errors": 0,
            "feature_errors": 0,
        }
        self.last = {
            "turn": 0,
            "phase": "reset",
            "own_prizes": 6,
            "win_signal": 0.5,
            "loss_risk": 0.5,
            "outcome_logit": 0.0,
            "context_win_signal": 0.5,
            "context_outcome_logit": 0.0,
            "action_family": "unknown",
            "action_feature_count": 0,
            "base_action": [],
            "short_long_shift": 0.5,
            "context_norms": [0.0] * self.heads,
            "tokens": 0,
        }
        if self.enabled:
            # Training starts every trajectory with this structural sentinel.
            sentinel = self.token_module.event_vector({"type": -99, "playerIndex": -1, "turn": 0}, 0, self.input_dim)
            self._step(sentinel)
            self.last["tokens"] = self.stats["tokens"]

    def _step(self, sparse):
        if not sparse:
            return
        for member_index, member in enumerate(self.members):
            state = self.states[member_index]
            wa = member.get("wa") or []
            wv = member.get("Wv") or []
            raw_decay = member.get("raw_decay") or []
            for head in range(self.heads):
                try:
                    attention_logit = sum(float(wa[head][index]) * float(value) for index, value in sparse.items())
                    attention_mass = math.exp(max(-8.0, min(8.0, attention_logit)))
                    decay = _sigmoid(raw_decay[head])
                    value_vector = [
                        sum(float(wv[head][index][channel]) * float(value) for index, value in sparse.items())
                        for channel in range(self.width)
                    ]
                    old_sum = state["sums"][head]
                    state["sums"][head] = [
                        decay * old_sum[channel] + attention_mass * value_vector[channel]
                        for channel in range(self.width)
                    ]
                    state["norms"][head] = decay * state["norms"][head] + attention_mass
                except Exception:
                    self.stats["sync_errors"] += 1
        self.stats["tokens"] += 1

    def _contexts(self):
        result = []
        for state in self.states:
            context = []
            for head in range(self.heads):
                norm = max(1e-9, float(state["norms"][head]))
                context.extend(value / norm for value in state["sums"][head])
            result.append(context)
        return result

    def _outcome(self, contexts, action_features):
        context_logits = []
        conditioned_logits = []
        for member, context in zip(self.members, contexts):
            weights = member.get("wh") or []
            bias_value = member.get("bias") or [0.0]
            bias = float(bias_value[0] if isinstance(bias_value, list) else bias_value)
            context_logit = bias + sum(float(weight) * value for weight, value in zip(weights, context))
            action_weights = member.get("ws") or []
            interaction_weights = member.get("V") or []
            action_term = sum(float(action_weights[index]) * float(value) for index, value in action_features.items())
            interaction = 0.0
            for channel, context_value in enumerate(context):
                if context_value:
                    interaction += context_value * sum(
                        float(interaction_weights[channel][index]) * float(value)
                        for index, value in action_features.items()
                    )
            context_logits.append(context_logit)
            conditioned_logits.append(context_logit + action_term + interaction)
        context_logit = sum(context_logits) / max(1, len(context_logits))
        outcome_logit = sum(conditioned_logits) / max(1, len(conditioned_logits))
        return context_logit, _sigmoid(context_logit), outcome_logit, _sigmoid(outcome_logit)

    def _phase(self, obs):
        current = obs.get("current") or {}
        turn = _int(current.get("turn"), 0)
        me = self.history.me_index if self.history.me_index in (0, 1) else _int(current.get("yourIndex"), 0)
        players = current.get("players") or []
        prizes = 6
        try:
            prizes = len((players[me] or {}).get("prize") or [])
        except Exception:
            pass
        # Setup observations expose an empty prize list before six prizes are
        # dealt; do not mislabel that turn-zero state as a closeout.
        if turn <= 0:
            phase = "opening"
        elif prizes <= 2:
            phase = "closeout"
        elif turn <= 4:
            phase = "opening"
        elif turn >= 11:
            phase = "late"
        else:
            phase = "midgame"
        return turn, prizes, phase

    def sync(self, obs, action_features=None, family="unknown", base=None):
        """Synchronize causal state and update telemetry; never return an action."""
        self.stats["calls"] += 1
        if not self.enabled or not isinstance(obs, dict):
            return
        try:
            current = obs.get("current") or {}
            me = self.history.me_index if self.history.me_index in (0, 1) else _int(current.get("yourIndex"), 0)
            # A previously emitted decision precedes the new public logs.
            while self.decision_cursor < len(self.history.decisions):
                token = self.token_module.decision_vector(self.history.decisions[self.decision_cursor], me, self.input_dim)
                self._step(token)
                self.decision_cursor += 1
                self.stats["synced_decisions"] += 1
            while self.event_cursor < len(self.history.events):
                token = self.token_module.event_vector(self.history.events[self.event_cursor], me, self.input_dim)
                self._step(token)
                self.event_cursor += 1
                self.stats["synced_events"] += 1
            contexts = self._contexts()
            action_features = action_features or {}
            context_logit, context_win, outcome_logit, win_signal = self._outcome(contexts, action_features)
            norms = []
            for head in range(self.heads):
                per_member = []
                lo, hi = head * self.width, (head + 1) * self.width
                for context in contexts:
                    per_member.append(math.sqrt(sum(value * value for value in context[lo:hi])))
                norms.append(sum(per_member) / max(1, len(per_member)))
            short_long_shift = norms[0] / max(1e-9, sum(norms)) if norms else 0.5
            turn, prizes, phase = self._phase(obs)
            self.stats["outcome_samples"] += 1
            self.stats["win_sum"] += win_signal
            self.stats["win_min"] = min(self.stats["win_min"], win_signal)
            self.stats["win_max"] = max(self.stats["win_max"], win_signal)
            self.stats["phase_counts"][phase] = self.stats["phase_counts"].get(phase, 0) + 1
            self.last = {
                "turn": turn,
                "phase": phase,
                "own_prizes": prizes,
                # Training target y=1 means a Lucario win.  ``loss_risk`` is the
                # explicit inverse; neither signal has action authority.
                "win_signal": round(win_signal, 6),
                "loss_risk": round(1.0 - win_signal, 6),
                "outcome_logit": round(outcome_logit, 6),
                "context_win_signal": round(context_win, 6),
                "context_outcome_logit": round(context_logit, 6),
                "action_family": family,
                "action_feature_count": len(action_features),
                "base_action": list(base) if isinstance(base, list) else [],
                "short_long_shift": round(short_long_shift, 6),
                "context_norms": [round(value, 6) for value in norms],
                "tokens": self.stats["tokens"],
            }
        except Exception:
            self.stats["sync_errors"] += 1

    def choose(self, obs, base):
        """Compatibility passthrough with source-level disabled authority."""
        action_features = {}
        family = "unknown"
        try:
            if isinstance(obs, dict) and isinstance(base, list):
                family = self.replay_module.recognize(self.history, obs)
                description = self.replay_module.action_desc(self.history, obs, base)
                action_features = self.token_module.dense_action_features(
                    self.replay_module, self.history, obs, description, family, self.action_dim
                )
        except Exception:
            self.stats["feature_errors"] += 1
        self.sync(obs, action_features, family, base)
        self.stats["passthrough_calls"] += 1
        return base

    def get_stats(self):
        samples = self.stats["outcome_samples"]
        win_mean = self.stats["win_sum"] / samples if samples else 0.5
        return {
            "version": self.model.get("version", "unavailable"),
            "enabled": self.enabled,
            "action_authority": False,
            "members": len(self.members),
            "heads": self.heads,
            "width": self.width,
            "calls": self.stats["calls"],
            "passthrough_calls": self.stats["passthrough_calls"],
            "overrides": 0,
            "override_attempts": 0,
            "synced_events": self.stats["synced_events"],
            "synced_decisions": self.stats["synced_decisions"],
            "tokens": self.stats["tokens"],
            "outcome_samples": samples,
            "target_semantics": "win_signal=P(Lucario trajectory win); loss_risk=1-win_signal",
            "win_mean": round(win_mean, 6),
            "win_min": round(self.stats["win_min"] if samples else 0.5, 6),
            "win_max": round(self.stats["win_max"] if samples else 0.5, 6),
            "loss_risk_mean": round(1.0 - win_mean, 6),
            "phase_counts": dict(self.stats["phase_counts"]),
            "sync_errors": self.stats["sync_errors"],
            "feature_errors": self.stats["feature_errors"],
            "load_error": self.load_error,
            "last": dict(self.last),
        }
