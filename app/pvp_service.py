from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from game_service import GameError
from worker_service import GameWorkerProxy, WorkerCapacity, WorkerError


ROOM_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass
class PlayerEntry:
    session_id: str
    name: str
    deck: list[int]
    deck_label: str
    joined_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)


@dataclass
class WaitingRoom:
    code: str
    host: PlayerEntry
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class MatchRoom:
    match_id: str
    players: list[PlayerEntry]
    worker: GameWorkerProxy
    room_code: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    last_seen: dict[str, float] = field(default_factory=dict)
    departed: set[str] = field(default_factory=set)

    def seat_of(self, session_id: str) -> int:
        for seat, player in enumerate(self.players):
            if player.session_id == session_id:
                return seat
        raise GameError("이 온라인 대전의 참가자가 아닙니다.")


class PvpHub:
    """In-memory authoritative matchmaking for a single CABT server instance.

    Each match owns one isolated CABT worker process. Browser sessions only hold
    opaque cookies; decks and native game state never leave the server except
    through each player's redacted public view.
    """

    def __init__(
        self,
        max_matches: int = 8,
        max_waiting: int = 64,
        queue_timeout: int = 600,
        match_idle_seconds: int = 7200,
        online_grace_seconds: int = 25,
        client_timeout_seconds: int = 45,
        disconnect_grace_seconds: int = 15,
        capacity: WorkerCapacity | None = None,
    ) -> None:
        self.max_matches = max(1, int(max_matches))
        self.max_waiting = max(2, int(max_waiting))
        self.queue_timeout = max(60, int(queue_timeout))
        self.match_idle_seconds = max(600, int(match_idle_seconds))
        self.online_grace_seconds = max(10, int(online_grace_seconds))
        self.client_timeout_seconds = max(5, int(client_timeout_seconds))
        self.disconnect_grace_seconds = max(2, int(disconnect_grace_seconds))
        self.capacity = capacity
        self.lock = threading.RLock()
        self.quick_queue: list[PlayerEntry] = []
        self.waiting_rooms: dict[str, WaitingRoom] = {}
        self.matches: dict[str, MatchRoom] = {}
        self.session_match: dict[str, str] = {}
        self.disconnect_hints: dict[str, float] = {}

    @staticmethod
    def _clean_name(raw: str) -> str:
        value = " ".join(str(raw or "").strip().split())[:32]
        return value or "Anonymous"

    def _hint_expired_locked(self, session_id: str, now: float) -> bool:
        hinted_at = self.disconnect_hints.get(session_id)
        return hinted_at is not None and now - hinted_at > self.disconnect_grace_seconds

    def _touch_session_locked(self, session_id: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        touched = False
        for entry in self.quick_queue:
            if entry.session_id == session_id:
                entry.last_seen = current
                touched = True
        for waiting in self.waiting_rooms.values():
            if waiting.host.session_id == session_id:
                waiting.host.last_seen = current
                touched = True
        match_id = self.session_match.get(session_id)
        room = self.matches.get(match_id or "")
        if room is not None:
            room.last_seen[session_id] = current
            touched = True
        if touched:
            self.disconnect_hints.pop(session_id, None)
        return touched

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        removed_waiters = {
            entry.session_id
            for entry in self.quick_queue
            if now - entry.joined_at > self.queue_timeout
            or now - entry.last_seen > self.client_timeout_seconds
            or self._hint_expired_locked(entry.session_id, now)
        }
        self.quick_queue = [
            entry for entry in self.quick_queue if entry.session_id not in removed_waiters
        ]

        expired_codes = []
        for code, room in self.waiting_rooms.items():
            host = room.host
            if (
                now - room.created_at > self.queue_timeout
                or now - host.last_seen > self.client_timeout_seconds
                or self._hint_expired_locked(host.session_id, now)
            ):
                expired_codes.append(code)
                removed_waiters.add(host.session_id)
        for code in expired_codes:
            self.waiting_rooms.pop(code, None)

        expired_matches: list[str] = []
        for match_id, room in self.matches.items():
            latest = max(room.last_seen.values(), default=room.created_at)
            player_stale = any(
                now - room.last_seen.get(player.session_id, room.created_at)
                > self.client_timeout_seconds
                for player in room.players
            )
            disconnect_expired = any(
                self._hint_expired_locked(player.session_id, now)
                for player in room.players
            )
            if (
                not room.worker.process.is_alive()
                or room.departed
                or player_stale
                or disconnect_expired
                or now - latest > self.match_idle_seconds
            ):
                expired_matches.append(match_id)
        for match_id in expired_matches:
            self._close_match_locked(match_id)

        active_sessions = set(self.session_match)
        active_sessions.update(entry.session_id for entry in self.quick_queue)
        active_sessions.update(room.host.session_id for room in self.waiting_rooms.values())
        for session_id in list(self.disconnect_hints):
            if session_id not in active_sessions:
                self.disconnect_hints.pop(session_id, None)

    def cleanup(self) -> None:
        with self.lock:
            self._cleanup_locked()

    def _remove_waiting_locked(self, session_id: str) -> None:
        self.quick_queue = [entry for entry in self.quick_queue if entry.session_id != session_id]
        codes = [
            code for code, room in self.waiting_rooms.items() if room.host.session_id == session_id
        ]
        for code in codes:
            self.waiting_rooms.pop(code, None)
        self.disconnect_hints.pop(session_id, None)

    def _ensure_waiting_capacity_locked(self) -> None:
        waiting_count = len(self.quick_queue) + len(self.waiting_rooms)
        if waiting_count >= self.max_waiting:
            raise GameError(
                f"현재 매칭 대기 한도({self.max_waiting}명)에 도달했습니다. 잠시 후 다시 시도하세요."
            )

    def _new_room_code_locked(self) -> str:
        for _ in range(100):
            code = "".join(secrets.choice(ROOM_ALPHABET) for _ in range(6))
            if code not in self.waiting_rooms:
                return code
        raise GameError("방 코드를 생성하지 못했습니다. 다시 시도하세요.")

    def _start_match_locked(
        self,
        first: PlayerEntry,
        second: PlayerEntry,
        room_code: str | None = None,
    ) -> MatchRoom:
        if len(self.matches) >= self.max_matches:
            raise GameError(
                f"현재 온라인 대전 한도({self.max_matches}게임)에 도달했습니다. 잠시 후 다시 시도하세요."
            )
        players = [first, second]
        if secrets.randbelow(2):
            players.reverse()
        match_id = secrets.token_urlsafe(12)
        worker = GameWorkerProxy(f"pvp-{match_id}", capacity=self.capacity)
        try:
            worker.call(
                "pvp_start",
                player0_deck=players[0].deck,
                player1_deck=players[1].deck,
                deck_labels=[players[0].deck_label, players[1].deck_label],
                player_names=[players[0].name, players[1].name],
            )
        except Exception:
            worker.terminate()
            raise
        now = time.monotonic()
        room = MatchRoom(
            match_id=match_id,
            players=players,
            worker=worker,
            room_code=room_code,
            last_seen={players[0].session_id: now, players[1].session_id: now},
        )
        self.matches[match_id] = room
        for player in players:
            self.session_match[player.session_id] = match_id
        return room

    def _close_match_locked(self, match_id: str) -> None:
        room = self.matches.pop(match_id, None)
        if room is None:
            return
        for player in room.players:
            if self.session_match.get(player.session_id) == match_id:
                self.session_match.pop(player.session_id, None)
            self.disconnect_hints.pop(player.session_id, None)
        room.worker.terminate()

    def _entry(self, session_id: str, name: str, deck: list[int], deck_label: str) -> PlayerEntry:
        if len(deck) != 60:
            raise GameError("온라인 대전 덱은 정확히 60장이어야 합니다.")
        return PlayerEntry(
            session_id=session_id,
            name=self._clean_name(name),
            deck=list(deck),
            deck_label=str(deck_label or "온라인 덱")[:80],
        )

    def join_quick(self, session_id: str, name: str, deck: list[int], deck_label: str) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            if session_id in self.session_match:
                return self._status_locked(session_id)
            self._remove_waiting_locked(session_id)
            entry = self._entry(session_id, name, deck, deck_label)
            opponent = next(
                (queued for queued in self.quick_queue if queued.session_id != session_id),
                None,
            )
            if opponent is None:
                self._ensure_waiting_capacity_locked()
                self.quick_queue.append(entry)
                return self._status_locked(session_id)
            self.quick_queue.remove(opponent)
            try:
                self._start_match_locked(opponent, entry)
            except Exception:
                self.quick_queue.insert(0, opponent)
                raise
            return self._status_locked(session_id)

    def create_private(self, session_id: str, name: str, deck: list[int], deck_label: str) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            if session_id in self.session_match:
                return self._status_locked(session_id)
            self._remove_waiting_locked(session_id)
            self._ensure_waiting_capacity_locked()
            code = self._new_room_code_locked()
            self.waiting_rooms[code] = WaitingRoom(
                code=code,
                host=self._entry(session_id, name, deck, deck_label),
            )
            return self._status_locked(session_id)

    def join_private(
        self,
        session_id: str,
        code: str,
        name: str,
        deck: list[int],
        deck_label: str,
    ) -> dict[str, Any]:
        normalized = "".join(str(code or "").upper().split())
        with self.lock:
            self._cleanup_locked()
            if session_id in self.session_match:
                return self._status_locked(session_id)
            waiting = self.waiting_rooms.get(normalized)
            if waiting is None:
                raise GameError("존재하지 않거나 만료된 방 코드입니다.")
            if waiting.host.session_id == session_id:
                return self._status_locked(session_id)
            self._remove_waiting_locked(session_id)
            guest = self._entry(session_id, name, deck, deck_label)
            self.waiting_rooms.pop(normalized, None)
            try:
                self._start_match_locked(waiting.host, guest, room_code=normalized)
            except Exception:
                self.waiting_rooms[normalized] = waiting
                raise
            return self._status_locked(session_id)

    def cancel(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            self._remove_waiting_locked(session_id)
            return self._status_locked(session_id)

    def _match_for_locked(self, session_id: str) -> tuple[MatchRoom, int]:
        match_id = self.session_match.get(session_id)
        room = self.matches.get(match_id or "")
        if room is None:
            self.session_match.pop(session_id, None)
            raise GameError("매칭된 온라인 게임이 없습니다.")
        seat = room.seat_of(session_id)
        room.last_seen[session_id] = time.monotonic()
        self.disconnect_hints.pop(session_id, None)
        return room, seat

    def _status_locked(self, session_id: str) -> dict[str, Any]:
        self._touch_session_locked(session_id)
        match_id = self.session_match.get(session_id)
        room = self.matches.get(match_id or "")
        if room is not None:
            seat = room.seat_of(session_id)
            room.last_seen[session_id] = time.monotonic()
            opponent = room.players[1 - seat]
            return {
                "status": "matched",
                "match_id": room.match_id,
                "seat": seat,
                "player_name": room.players[seat].name,
                "opponent_name": opponent.name,
                "room_code": room.room_code,
            }
        for entry in self.quick_queue:
            if entry.session_id == session_id:
                return {
                    "status": "searching",
                    "player_name": entry.name,
                    "queue_seconds": int(time.monotonic() - entry.joined_at),
                }
        for room_code, waiting in self.waiting_rooms.items():
            if waiting.host.session_id == session_id:
                return {
                    "status": "waiting_room",
                    "player_name": waiting.host.name,
                    "room_code": room_code,
                    "queue_seconds": int(time.monotonic() - waiting.created_at),
                }
        return {"status": "idle"}

    def status(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            return self._status_locked(session_id)

    def state(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            room, seat = self._match_for_locked(session_id)
            worker = room.worker
            opponent = room.players[1 - seat]
            now = time.monotonic()
            opponent_online = (
                opponent.session_id not in room.departed
                and now - room.last_seen.get(opponent.session_id, room.created_at)
                <= self.online_grace_seconds
            )
            opponent_left = opponent.session_id in room.departed
            match_id = room.match_id
            room_code = room.room_code
        state = worker.call("pvp_state", viewer_seat=seat)
        state.update(
            {
                "match_id": match_id,
                "matchmaking_status": "matched",
                "opponent_online": opponent_online,
                "opponent_left": opponent_left,
                "room_code": room_code,
            }
        )
        return state

    def action(self, session_id: str, indices: list[int]) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            room, seat = self._match_for_locked(session_id)
            if session_id in room.departed:
                raise GameError("이미 온라인 대전에서 나갔습니다.")
            worker = room.worker
        state = worker.call("pvp_action", seat=seat, indices=indices)
        opponent = room.players[1 - seat]
        now = time.monotonic()
        opponent_left = opponent.session_id in room.departed
        opponent_online = (
            not opponent_left
            and now - room.last_seen.get(opponent.session_id, room.created_at)
            <= self.online_grace_seconds
        )
        state.update(
            {
                "match_id": room.match_id,
                "matchmaking_status": "matched",
                "room_code": room.room_code,
                "opponent_left": opponent_left,
                "opponent_online": opponent_online,
            }
        )
        return state

    def game_file(self, session_id: str, command: str, **kwargs: Any) -> str:
        with self.lock:
            self._cleanup_locked()
            room, _seat = self._match_for_locked(session_id)
            worker = room.worker
        return str(worker.call(command, **kwargs))

    def heartbeat(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self._touch_session_locked(session_id)
            self._cleanup_locked()
            return self._status_locked(session_id)

    def mark_disconnected(self, session_id: str) -> None:
        with self.lock:
            active = session_id in self.session_match
            if not active:
                active = any(entry.session_id == session_id for entry in self.quick_queue)
            if not active:
                active = any(
                    waiting.host.session_id == session_id
                    for waiting in self.waiting_rooms.values()
                )
            if active:
                self.disconnect_hints[session_id] = time.monotonic()

    def leave(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self._cleanup_locked()
            self._remove_waiting_locked(session_id)
            match_id = self.session_match.get(session_id)
            room = self.matches.get(match_id or "")
            if room is None:
                self.session_match.pop(session_id, None)
                self.disconnect_hints.pop(session_id, None)
                return {"status": "idle"}
            # An explicit lobby/leave action is final. Close the shared worker
            # immediately so a one-game Railway deployment frees its slot.
            self._close_match_locked(room.match_id)
            return {"status": "idle"}

    def stats(self) -> dict[str, int]:
        with self.lock:
            self._cleanup_locked()
            return {
                "quick_queue": len(self.quick_queue),
                "private_rooms": len(self.waiting_rooms),
                "active_matches": len(self.matches),
                "max_matches": self.max_matches,
                "max_waiting": self.max_waiting,
            }

    def shutdown(self) -> None:
        with self.lock:
            rooms = list(self.matches.values())
            self.matches.clear()
            self.session_match.clear()
            self.quick_queue.clear()
            self.waiting_rooms.clear()
            self.disconnect_hints.clear()
        for room in rooms:
            try:
                room.worker.terminate()
            except (WorkerError, OSError):
                pass
