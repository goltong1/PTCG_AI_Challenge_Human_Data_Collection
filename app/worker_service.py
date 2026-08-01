from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import threading
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


class WorkerError(RuntimeError):
    pass


class WorkerCapacity:
    """Process-wide capacity shared by AI sessions and PvP matches."""

    def __init__(self, limit: int = 1) -> None:
        self.limit = max(1, int(limit))
        self.active = 0
        self.lock = threading.RLock()

    def acquire(self) -> None:
        with self.lock:
            if self.active >= self.limit:
                raise WorkerError(
                    f"현재 서버 게임 처리 한도({self.limit}게임)에 도달했습니다. "
                    "진행 중인 게임이 끝난 뒤 다시 시도하세요."
                )
            self.active += 1

    def release(self) -> None:
        with self.lock:
            self.active = max(0, self.active - 1)

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {"active": self.active, "limit": self.limit}


def worker_main(connection: Connection, engine_temp_path: str) -> None:
    # Import inside the spawned process so every session gets an independent
    # native CABT engine and independent agent module namespace.
    from agent_loader import AgentRepository
    from game_service import GameError, GameManager, ROOT
    from replay_service import ReplayManager
    from result_service import ResultCollector

    # The bundled CABT adapter writes the selected AI deck beside cg/api.py.
    # Give every browser session its own 5 MB engine copy so concurrent users
    # cannot overwrite each other's deck.csv or native-engine state files.
    engine_temp = Path(engine_temp_path)
    session_engine = engine_temp / "engine"
    shutil.copytree(ROOT / "engine", session_engine)

    agents = AgentRepository(ROOT / "agents", engine_root=session_engine)
    manager = GameManager(agents)
    replay = ReplayManager(manager)
    collector = ResultCollector()
    try:
        while True:
            request = connection.recv()
            if not isinstance(request, dict):
                connection.send({"ok": False, "error": "잘못된 워커 요청입니다."})
                continue
            command = str(request.get("command") or "")
            args = request.get("args") if isinstance(request.get("args"), dict) else {}
            if command == "shutdown":
                connection.send({"ok": True, "result": {"ok": True}})
                break
            try:
                if command == "state":
                    result = manager.public_state()
                elif command == "start":
                    result = manager.start(
                        int(args.get("human_seat", 0)),
                        list(args.get("human_deck") or []),
                        str(args.get("deck_label") or ""),
                        str(args.get("agent_id") or ""),
                    )
                elif command == "human_action":
                    result = manager.act_human(list(args.get("indices") or []))
                elif command == "pvp_start":
                    result = manager.start_pvp(
                        list(args.get("player0_deck") or []),
                        list(args.get("player1_deck") or []),
                        list(args.get("deck_labels") or []),
                        list(args.get("player_names") or []),
                        str(args.get("engine_agent_id") or ""),
                    )
                elif command == "pvp_state":
                    result = manager.public_state_for(int(args.get("viewer_seat", 0)))
                elif command == "pvp_action":
                    result = manager.act_player(
                        int(args.get("seat", 0)),
                        list(args.get("indices") or []),
                    )
                elif command == "ai_step":
                    result = manager.act_ai_once()
                elif command == "close_game":
                    manager.close()
                    result = {"ok": True}
                elif command == "record_zip":
                    result = str(manager.record_zip())
                elif command == "official_replay":
                    result = str(manager.official_replay_file())
                elif command == "viewer":
                    result = str(manager.viewer_launcher(int(args.get("player", 0))))
                elif command == "replay_load":
                    result = replay.load(bytes(args.get("data") or b""), str(args.get("filename") or "replay.zip"))
                elif command == "replay_state":
                    result = replay.state(int(args.get("index", 0)), int(args.get("view_seat", 0)))
                elif command == "replay_close":
                    replay.close()
                    result = {"ok": True}
                elif command == "submit_result":
                    result = collector.submit(
                        manager,
                        str(args.get("player_name") or ""),
                        str(args.get("note") or ""),
                    )
                else:
                    raise GameError(f"지원하지 않는 워커 명령입니다: {command}")
                connection.send({"ok": True, "result": result})
            except Exception as exc:
                connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except EOFError:
        pass
    finally:
        replay.close()
        manager.close()
        connection.close()


class GameWorkerProxy:
    def __init__(self, session_id: str, timeout: float = 180.0, capacity: WorkerCapacity | None = None) -> None:
        self.session_id = session_id
        self.timeout = timeout
        self.capacity = capacity
        self.capacity_acquired = False
        if self.capacity is not None:
            self.capacity.acquire()
            self.capacity_acquired = True
        self.lock = threading.RLock()
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.engine_temp = Path(tempfile.mkdtemp(prefix=f"cabt-session-{session_id[:8]}-"))
        self.process = context.Process(
            target=worker_main,
            args=(child, str(self.engine_temp)),
            name=f"cabt-session-{session_id[:8]}",
            daemon=True,
        )
        try:
            self.process.start()
        except Exception:
            parent.close()
            child.close()
            shutil.rmtree(self.engine_temp, ignore_errors=True)
            if self.capacity is not None and self.capacity_acquired:
                self.capacity.release()
                self.capacity_acquired = False
            raise
        child.close()
        self.last_access = time.monotonic()

    def call(self, command: str, **args: Any) -> Any:
        with self.lock:
            if not self.process.is_alive():
                raise WorkerError("게임 세션 프로세스가 종료되었습니다. 새 게임을 시작하세요.")
            try:
                self.connection.send({"command": command, "args": args})
                if not self.connection.poll(self.timeout):
                    self.terminate()
                    raise WorkerError("게임 처리 시간이 제한을 초과했습니다.")
                response = self.connection.recv()
            except WorkerError:
                raise
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.terminate()
                raise WorkerError("게임 세션 프로세스와 연결이 끊어졌습니다. 새 게임을 시작하세요.") from exc
            self.last_access = time.monotonic()
            if not isinstance(response, dict) or not response.get("ok"):
                message = response.get("error") if isinstance(response, dict) else "워커 응답 오류"
                raise WorkerError(str(message))
            return response.get("result")

    def terminate(self) -> None:
        with self.lock:
            if self.process.is_alive():
                try:
                    self.connection.send({"command": "shutdown", "args": {}})
                    if self.connection.poll(1.5):
                        self.connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(timeout=1.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
            try:
                self.connection.close()
            except OSError:
                pass
            shutil.rmtree(self.engine_temp, ignore_errors=True)
            if self.capacity is not None and self.capacity_acquired:
                self.capacity.release()
                self.capacity_acquired = False


class SessionHub:
    def __init__(self, max_sessions: int = 8, idle_seconds: int = 3600, capacity: WorkerCapacity | None = None) -> None:
        self.max_sessions = max(1, max_sessions)
        self.idle_seconds = max(300, idle_seconds)
        self.capacity = capacity
        self.lock = threading.RLock()
        self.sessions: dict[str, GameWorkerProxy] = {}

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, worker in self.sessions.items()
            if not worker.process.is_alive() or now - worker.last_access > self.idle_seconds
        ]
        for session_id in expired:
            worker = self.sessions.pop(session_id, None)
            if worker is not None:
                worker.terminate()

    def peek(self, session_id: str) -> GameWorkerProxy | None:
        with self.lock:
            self._cleanup_locked()
            worker = self.sessions.get(session_id)
            if worker is not None:
                worker.last_access = time.monotonic()
            return worker

    def get(self, session_id: str) -> GameWorkerProxy:
        with self.lock:
            self._cleanup_locked()
            worker = self.sessions.get(session_id)
            if worker is not None:
                worker.last_access = time.monotonic()
                return worker
            if len(self.sessions) >= self.max_sessions:
                raise WorkerError(
                    f"현재 동시 접속 한도({self.max_sessions}게임)에 도달했습니다. 잠시 후 다시 시도하세요."
                )
            worker = GameWorkerProxy(session_id, capacity=self.capacity)
            self.sessions[session_id] = worker
            return worker

    def close(self, session_id: str) -> None:
        with self.lock:
            worker = self.sessions.pop(session_id, None)
        if worker is not None:
            worker.terminate()

    def shutdown(self) -> None:
        with self.lock:
            workers = list(self.sessions.values())
            self.sessions.clear()
        for worker in workers:
            worker.terminate()
