"""Small causal history store for the submission policy.

The simulator already supplies the complete public board on every decision,
but ``logs`` are incremental.  This module keeps only facts that can be known
from those logs and from our own selected actions.  It deliberately contains
no opponent-name or deck-identity logic and has no third-party dependencies.
"""

from __future__ import annotations

from collections import Counter


# Numeric values are part of the CABT public API.  Keeping them local makes
# this module usable without importing the native ``cg`` package.
_HAND = 2
_DRAW = 4
_MOVE_CARD = 6
_MOVE_CARD_REVERSE = 7
_PLAY = 10
_ATTACH = 11
_EVOLVE = 12
_ATTACK = 15

_OPT_CARD = 3
_OPT_PLAY = 7
_OPT_ATTACH = 8
_OPT_EVOLVE = 9
_OPT_ABILITY = 10
_OPT_RETREAT = 12
_OPT_ATTACK = 13

_CTX_SWITCH = 3
_CTX_TO_ACTIVE = 4

_HAND_RESET_IDS = frozenset((1080, 1213, 1227))
_OPPONENT_HAND_RESET_IDS = frozenset((1080, 1213))
_SEARCH_IDS = frozenset((1086, 1097, 1121, 1152, 1231))
_BOSS_ID = 1182
_BUDEW_ID = 235
_BUDEW_ATTACK_ID = 323
_DRAW_ABILITY_IDS = frozenset((120, 140))


def _int(value, default=0):
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _enum_int(value, default=-1):
    return _int(value, default)


class CausalHistory:
    """Ordered, per-game public history with conservative hidden-hand facts."""

    def __init__(self):
        self.game_number = 0
        self.total_resets = 0
        self.reset()

    def reset(self):
        self.game_number += 1
        self.total_resets += 1
        self.initialized = False
        self.my_index = None
        self.turn = None
        self.last_turn = None
        self.last_result = -1
        self.last_ingest_key = None
        self.log_count = 0
        self.decisions = []
        self.public_events = []
        self.known_opponent_hand = {}
        self.known_opponent_counts = Counter()
        self.hand_reset_keys = set()
        self.search_keys = set()
        self.budew_attack_keys = set()
        self.pending_own_log_events = Counter()
        self.total_hand_resets = 0
        self.total_searches = 0
        self.total_budew_attacks = 0
        self.pending_retreat_serial = None
        self.pending_gust_serial = None
        self.pending_trap_boss_serial = None
        self.pending_trap_turn = None
        self.total_reset_sequences_armed = 0
        self.total_reset_sequence_setup_actions = 0
        self.total_reset_sequence_forced_resets = 0
        self.total_reset_sequence_post_draws = 0
        self.total_reset_sequence_aborts = 0
        self._reset_turn_facts()

    def _reset_turn_facts(self):
        self.hand_resets_this_turn = 0
        self.searches_this_turn = 0
        self.budew_attacks_this_turn = 0
        self.pending_retreat_serial = None
        self.pending_gust_serial = None
        # 0 idle, 1 evacuating deterministic resources, 2 reset committed,
        # 3 a post-reset draw engine has been used.
        self.reset_sequence_phase = 0
        self.reset_sequence_reset_id = 0
        self.reset_sequence_setup_actions_this_turn = 0

    def arm_reset_sequence(self, reset_id):
        """Start a within-turn reset plan without carrying it across turns."""
        reset_id = _int(reset_id, 0)
        if self.reset_sequence_phase == 0:
            self.reset_sequence_phase = 1
            self.reset_sequence_reset_id = reset_id
            self.total_reset_sequences_armed += 1
        return self.reset_sequence_phase == 1 and self.reset_sequence_reset_id == reset_id

    def mark_reset_sequence_setup(self):
        if self.reset_sequence_phase == 1:
            self.reset_sequence_setup_actions_this_turn += 1
            self.total_reset_sequence_setup_actions += 1

    def mark_reset_sequence_forced_reset(self):
        if self.reset_sequence_phase == 1:
            self.total_reset_sequence_forced_resets += 1

    def abort_reset_sequence(self):
        if self.reset_sequence_phase == 1:
            self.total_reset_sequence_aborts += 1
        self.reset_sequence_phase = 0
        self.reset_sequence_reset_id = 0
        self.reset_sequence_setup_actions_this_turn = 0

    def prepare(self, obs):
        """Ingest one observation exactly once.

        A null selection with a null state is the authoritative deck-request
        reset.  Null selections carrying a state can contain opponent logs and
        are ingested normally.  The decreasing-turn fallback covers harnesses
        that omit the deck request; repeated turn-0 setup selections do not
        reset history.
        """
        # Only the null-selection/null-state deck request is a game boundary.
        # Replay containers may expose null selections while the other player
        # resolves actions; their incremental logs must still be ingested.
        if obs is None or (
            getattr(obs, "select", None) is None and getattr(obs, "current", None) is None
        ):
            self.reset()
            return
        state = getattr(obs, "current", None)
        if state is None:
            return
        turn = _int(getattr(state, "turn", None), 0)
        result = _int(getattr(state, "result", None), -1)
        if self.initialized and self.last_turn is not None and turn < self.last_turn:
            self.reset()
        elif self.initialized and self.last_result >= 0 and result < 0:
            self.reset()
        if not self.initialized:
            self.initialized = True
            self.my_index = _int(getattr(state, "yourIndex", None), 0)
            self.turn = turn
            self.last_turn = turn
            self.last_result = result
            self._reset_turn_facts()
        elif turn != self.turn:
            self.turn = turn
            self._reset_turn_facts()
        if self.pending_trap_turn is not None and turn > self.pending_trap_turn + 2:
            self.clear_trap_boss_intent()

        logs = list(getattr(obs, "logs", None) or [])
        fingerprint = (
            turn,
            _int(getattr(state, "turnActionCount", None), 0),
            tuple(self._log_fingerprint(log) for log in logs),
        )
        if fingerprint != self.last_ingest_key:
            for log in logs:
                self._record_public_event(state, log)
                self._ingest_log(log)
            self.last_ingest_key = fingerprint
        self.last_turn = turn
        self.last_result = result

    @staticmethod
    def _log_fingerprint(log):
        return tuple(
            _int(getattr(log, name, None), -1)
            for name in (
                "type", "playerIndex", "cardId", "serial", "fromArea",
                "toArea", "cardIdBefore", "serialBefore", "cardIdAfter",
                "serialAfter", "cardIdTarget", "serialTarget", "attackId",
            )
        )

    def _record_public_event(self, state, log):
        """Append one received public log in causal observation order."""
        self.public_events.append({
            "sequence": len(self.public_events),
            "observed_turn": _int(getattr(state, "turn", None), 0),
            "observed_turn_action_count": _int(getattr(state, "turnActionCount", None), 0),
            "log_type": _enum_int(getattr(log, "type", None)),
            "player_index": _int(getattr(log, "playerIndex", None), -1),
            "card_id": _int(getattr(log, "cardId", None), 0),
            "serial": _int(getattr(log, "serial", None), -1),
            "from_area": _enum_int(getattr(log, "fromArea", None)),
            "to_area": _enum_int(getattr(log, "toArea", None)),
            "card_id_before": _int(getattr(log, "cardIdBefore", None), 0),
            "serial_before": _int(getattr(log, "serialBefore", None), -1),
            "card_id_after": _int(getattr(log, "cardIdAfter", None), 0),
            "serial_after": _int(getattr(log, "serialAfter", None), -1),
            "card_id_active": _int(getattr(log, "cardIdActive", None), 0),
            "serial_active": _int(getattr(log, "serialActive", None), -1),
            "card_id_bench": _int(getattr(log, "cardIdBench", None), 0),
            "serial_bench": _int(getattr(log, "serialBench", None), -1),
            "card_id_target": _int(getattr(log, "cardIdTarget", None), 0),
            "serial_target": _int(getattr(log, "serialTarget", None), -1),
            "attack_id": _int(getattr(log, "attackId", None), 0),
            "value": _int(getattr(log, "value", None), 0),
            "result": _int(getattr(log, "result", None), -1),
            "reason": _int(getattr(log, "reason", None), 0),
        })

    def recent_public_events(self, limit=64):
        limit = max(0, _int(limit, 64))
        events = self.public_events[-limit:] if limit else []
        return [dict(event) for event in events]

    def _fact_key(self, card_id, serial, suffix=0):
        serial = _int(serial, -1)
        if serial >= 0:
            return (self.turn, serial)
        return (self.turn, "unknown", _int(card_id), suffix)

    def _mark_reset(self, card_id, serial, suffix=0):
        key = self._fact_key(card_id, serial, suffix)
        if key not in self.hand_reset_keys:
            self.hand_reset_keys.add(key)
            self.hand_resets_this_turn += 1
            self.total_hand_resets += 1

    def _mark_search(self, card_id, serial, suffix=0):
        key = self._fact_key(card_id, serial, suffix)
        if key not in self.search_keys:
            self.search_keys.add(key)
            self.searches_this_turn += 1
            self.total_searches += 1

    def _mark_budew(self, serial, suffix=0):
        key = self._fact_key(_BUDEW_ID, serial, suffix)
        if key not in self.budew_attack_keys:
            self.budew_attack_keys.add(key)
            self.budew_attacks_this_turn += 1
            self.total_budew_attacks += 1

    def _add_known_hand(self, serial, card_id):
        serial = _int(serial, -1)
        card_id = _int(card_id, 0)
        if serial < 0 or card_id <= 0:
            return
        old = self.known_opponent_hand.get(serial)
        if old == card_id:
            return
        if old is not None:
            self.known_opponent_counts[old] -= 1
        self.known_opponent_hand[serial] = card_id
        self.known_opponent_counts[card_id] += 1

    def _remove_known_hand(self, serial):
        serial = _int(serial, -1)
        card_id = self.known_opponent_hand.pop(serial, None)
        if card_id is not None:
            self.known_opponent_counts[card_id] -= 1
            if self.known_opponent_counts[card_id] <= 0:
                del self.known_opponent_counts[card_id]

    def _clear_known_hand(self):
        self.known_opponent_hand.clear()
        self.known_opponent_counts.clear()

    def _ingest_log(self, log):
        self.log_count += 1
        kind = _enum_int(getattr(log, "type", None))
        player = _int(getattr(log, "playerIndex", None), -1)
        card_id = _int(getattr(log, "cardId", None), 0)
        serial = _int(getattr(log, "serial", None), -1)
        mine = player == self.my_index

        if mine:
            event_key = (
                kind, card_id, serial,
                _int(getattr(log, "attackId", None), 0),
            )
            if self.pending_own_log_events.get(event_key, 0) > 0:
                # Our semantic decision is authoritative.  Replay streams can
                # repeat the resulting public log in more than one observation,
                # so keep this key for the remainder of the game.
                return
            if kind == _PLAY:
                if card_id in _HAND_RESET_IDS:
                    self._mark_reset(card_id, serial)
                if card_id in _OPPONENT_HAND_RESET_IDS:
                    self._clear_known_hand()
                if card_id in _SEARCH_IDS:
                    self._mark_search(card_id, serial)
            elif kind == _ATTACK and (
                card_id == _BUDEW_ID or _int(getattr(log, "attackId", None)) == _BUDEW_ATTACK_ID
            ):
                self._mark_budew(serial)
            return

        # Opponent facts are retained only while the exact serial is publicly
        # known to remain in hand.
        if kind == _DRAW:
            self._add_known_hand(serial, card_id)
        elif kind == _MOVE_CARD:
            from_area = _enum_int(getattr(log, "fromArea", None))
            to_area = _enum_int(getattr(log, "toArea", None))
            if from_area == _HAND:
                self._remove_known_hand(serial)
            if to_area == _HAND:
                self._add_known_hand(serial, card_id)
        elif kind == _MOVE_CARD_REVERSE:
            # A face-down movement out of hand does not identify which known
            # serial moved, so retaining any exact-hand assertion is unsafe.
            if _enum_int(getattr(log, "fromArea", None)) == _HAND:
                self._clear_known_hand()
        elif kind in (_PLAY, _ATTACH, _EVOLVE):
            self._remove_known_hand(serial)
            if kind == _PLAY and card_id in _HAND_RESET_IDS:
                self._clear_known_hand()

    def known_opponent_has(self, card_ids):
        return any(self.known_opponent_counts.get(_int(card_id), 0) > 0 for card_id in card_ids)

    def set_retreat_intent(self, serial):
        self.pending_retreat_serial = _int(serial, -1)

    def set_gust_intent(self, serial):
        self.pending_gust_serial = _int(serial, -1)

    def set_trap_boss_intent(self, serial):
        """Remember a one-own-turn Boss follow-up after preserving a trap."""
        self.pending_trap_boss_serial = _int(serial, -1)
        self.pending_trap_turn = _int(self.turn, 0)

    def clear_trap_boss_intent(self):
        self.pending_trap_boss_serial = None
        self.pending_trap_turn = None

    def record_action(self, obs, indices, source_resolver=None):
        """Record the final, validated semantic action in decision order."""
        select = getattr(obs, "select", None)
        state = getattr(obs, "current", None)
        if select is None or state is None:
            return
        options = list(getattr(select, "option", None) or [])
        context = _enum_int(getattr(select, "context", None))
        effect = getattr(select, "effect", None)
        turn = _int(getattr(state, "turn", None), 0)
        action_count = _int(getattr(state, "turnActionCount", None), 0)
        for index in list(indices or []):
            if not isinstance(index, int) or not (0 <= index < len(options)):
                continue
            option = options[index]
            kind = _enum_int(getattr(option, "type", None))
            source = None
            if source_resolver is not None:
                try:
                    source = source_resolver(obs, options, index)
                except Exception:
                    source = None
            if source is None and kind in (_OPT_ATTACK, _OPT_RETREAT):
                try:
                    active = list(state.players[state.yourIndex].active or [])
                    source = active[0] if active else None
                except Exception:
                    source = None
            card_id = _int(getattr(source, "id", getattr(option, "cardId", None)), 0)
            serial = _int(getattr(source, "serial", getattr(option, "serial", None)), -1)
            target = self._option_target(obs, option)
            semantic = {
                "sequence": len(self.decisions),
                "turn": turn,
                "turn_action_count": action_count,
                "context": context,
                "effect_card_id": _int(getattr(effect, "id", None), 0),
                "option_index": index,
                "option_type": kind,
                "card_id": card_id,
                "serial": serial,
                "target_card_id": _int(getattr(target, "id", None), 0),
                "target_serial": _int(getattr(target, "serial", None), -1),
                "attack_id": _int(getattr(option, "attackId", None), 0),
                "reset_sequence_phase_before": self.reset_sequence_phase,
            }
            self.decisions.append(semantic)
            key_suffix = action_count
            if kind == _OPT_PLAY:
                if card_id in _HAND_RESET_IDS or card_id in _SEARCH_IDS:
                    self.pending_own_log_events[(_PLAY, card_id, serial, 0)] += 1
                if card_id in _HAND_RESET_IDS:
                    self._mark_reset(card_id, serial, key_suffix)
                if card_id in _OPPONENT_HAND_RESET_IDS:
                    self._clear_known_hand()
                if card_id in _SEARCH_IDS:
                    self._mark_search(card_id, serial, key_suffix)
                if self.reset_sequence_phase == 1 and card_id == self.reset_sequence_reset_id:
                    self.reset_sequence_phase = 2
            elif kind == _OPT_ATTACK and semantic["attack_id"] == _BUDEW_ATTACK_ID:
                self.pending_own_log_events[(_ATTACK, card_id, serial, semantic["attack_id"])] += 1
                self._mark_budew(serial, key_suffix)
            elif kind == _OPT_ABILITY and card_id in _DRAW_ABILITY_IDS and self.reset_sequence_phase == 2:
                self.reset_sequence_phase = 3
                self.total_reset_sequence_post_draws += 1
            if kind == _OPT_CARD and context in (_CTX_SWITCH, _CTX_TO_ACTIVE):
                self.pending_retreat_serial = None
            if kind == _OPT_CARD and self.pending_gust_serial is not None:
                if (
                    self.pending_trap_boss_serial is not None
                    and self.pending_trap_boss_serial == self.pending_gust_serial
                ):
                    self.clear_trap_boss_intent()
                self.pending_gust_serial = None
            if kind == _OPT_PLAY and card_id != _BOSS_ID:
                # A gust intent is meaningful only immediately after Boss.
                self.pending_gust_serial = None

    def _option_target(self, obs, option):
        try:
            state = obs.current
            me = state.yourIndex
            area = _enum_int(getattr(option, "inPlayArea", None))
            index = _int(getattr(option, "inPlayIndex", None), -1)
            player = state.players[me]
            cards = list(player.active) if area == 4 else list(player.bench) if area == 5 else []
            return cards[index] if 0 <= index < len(cards) else None
        except Exception:
            return None

    def stats(self):
        return {
            "history_games": max(0, self.game_number - 1),
            "history_resets": max(0, self.total_resets - 1),
            "history_decisions": len(self.decisions),
            "history_logs": self.log_count,
            "history_public_events": len(self.public_events),
            "known_opponent_hand": len(self.known_opponent_hand),
            "hand_resets_this_turn": self.hand_resets_this_turn,
            "searches_this_turn": self.searches_this_turn,
            "budew_attacks_this_turn": self.budew_attacks_this_turn,
            "total_hand_resets": self.total_hand_resets,
            "total_searches": self.total_searches,
            "total_budew_attacks": self.total_budew_attacks,
            "pending_retreat": int(self.pending_retreat_serial is not None and self.pending_retreat_serial >= 0),
            "pending_gust": int(self.pending_gust_serial is not None and self.pending_gust_serial >= 0),
            "pending_trap_boss": int(self.pending_trap_boss_serial is not None and self.pending_trap_boss_serial >= 0),
            "reset_sequence_phase": self.reset_sequence_phase,
            "reset_sequence_setup_actions_this_turn": self.reset_sequence_setup_actions_this_turn,
            "total_reset_sequences_armed": self.total_reset_sequences_armed,
            "total_reset_sequence_setup_actions": self.total_reset_sequence_setup_actions,
            "total_reset_sequence_forced_resets": self.total_reset_sequence_forced_resets,
            "total_reset_sequence_post_draws": self.total_reset_sequence_post_draws,
            "total_reset_sequence_aborts": self.total_reset_sequence_aborts,
        }
