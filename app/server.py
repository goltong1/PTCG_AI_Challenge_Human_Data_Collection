from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import secrets
import signal
import shutil
import sys
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from agent_loader import AgentLoadError, AgentRepository, MAX_ZIP_BYTES
from deck_store import DeckStore, DeckStoreError
from game_service import DATA_ROOT, GameError, GameManager, ROOT, parse_uploaded_deck, read_deck
from image_service import CardImageIndex
from replay_service import REPLAY_UPLOAD_LIMIT, ReplayManager
from result_service import ResultCollector, admin_token
from pvp_service import PvpHub
from worker_service import SessionHub, WorkerCapacity, WorkerError


APP_VERSION = "5.1.3-railway"
JSON_BODY_LIMIT = 2 * 1024 * 1024
PUBLIC_MODE = os.environ.get("CABT_PUBLIC_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
RESULT_SUBMISSION_ENABLED = os.environ.get(
    "CABT_ENABLE_RESULT_SUBMISSION", "1" if PUBLIC_MODE else "0"
).strip().lower() in {"1", "true", "yes", "on"}
MAX_SESSIONS = max(1, int(os.environ.get("CABT_MAX_SESSIONS", "8")))
SESSION_IDLE_SECONDS = max(300, int(os.environ.get("CABT_SESSION_IDLE_SECONDS", "3600")))
ONLINE_MATCHING_ENABLED = os.environ.get(
    "CABT_ENABLE_ONLINE_MATCHING", "1" if PUBLIC_MODE else "0"
).strip().lower() in {"1", "true", "yes", "on"}
MAX_PVP_MATCHES = max(1, int(os.environ.get("CABT_MAX_PVP_MATCHES", "8")))
MAX_PVP_WAITERS = max(2, int(os.environ.get("CABT_MAX_PVP_WAITERS", "64")))
PVP_QUEUE_TIMEOUT = max(60, int(os.environ.get("CABT_PVP_QUEUE_TIMEOUT", "600")))
PVP_MATCH_IDLE_SECONDS = max(600, int(os.environ.get("CABT_PVP_MATCH_IDLE_SECONDS", "7200")))
CLIENT_DISCONNECT_SECONDS = max(5, int(os.environ.get("CABT_CLIENT_DISCONNECT_SECONDS", "45")))
CLIENT_DISCONNECT_GRACE_SECONDS = max(2, int(os.environ.get("CABT_CLIENT_DISCONNECT_GRACE_SECONDS", "15")))
MAX_ACTIVE_WORKERS = max(1, int(os.environ.get("CABT_MAX_ACTIVE_WORKERS", "1")))
SESSION_COOKIE = "CABT_SESSION"

agents = AgentRepository(ROOT / "agents")
manager = GameManager(agents)
replay_manager = ReplayManager(manager)
deck_store = DeckStore(DATA_ROOT / "user_data" / "decks.json")
image_index = CardImageIndex(ROOT, manager.card_ids())
result_collector = ResultCollector()
worker_capacity = WorkerCapacity(MAX_ACTIVE_WORKERS) if PUBLIC_MODE else None
session_hub = (
    SessionHub(
        MAX_SESSIONS,
        SESSION_IDLE_SECONDS,
        capacity=worker_capacity,
        disconnect_seconds=CLIENT_DISCONNECT_SECONDS,
        disconnect_grace_seconds=CLIENT_DISCONNECT_GRACE_SECONDS,
    )
    if PUBLIC_MODE
    else None
)
pvp_hub = (
    PvpHub(
        max_matches=MAX_PVP_MATCHES,
        max_waiting=MAX_PVP_WAITERS,
        queue_timeout=PVP_QUEUE_TIMEOUT,
        match_idle_seconds=PVP_MATCH_IDLE_SECONDS,
        client_timeout_seconds=CLIENT_DISCONNECT_SECONDS,
        disconnect_grace_seconds=CLIENT_DISCONNECT_GRACE_SECONDS,
        capacity=worker_capacity,
    )
    if ONLINE_MATCHING_ENABLED
    else None
)


class CABTServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class CABTHandler(BaseHTTPRequestHandler):
    server_version = f"CABTWeb/{APP_VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(f"[CABT] {self.client_address[0]} · {fmt % args}\n")
        sys.stdout.flush()

    def _send_headers(
        self,
        status: int,
        content_type: str,
        length: int,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        session_cookie = getattr(self, "_pending_session_cookie", None)
        if session_cookie:
            secure = self.headers.get("X-Forwarded-Proto", "").lower() == "https" or os.environ.get(
                "CABT_COOKIE_SECURE", "0"
            ).lower() in {"1", "true", "yes", "on"}
            value = f"{SESSION_COOKIE}={session_cookie}; Path=/; Max-Age=86400; HttpOnly; SameSite=Lax"
            if secure:
                value += "; Secure"
            self.send_header("Set-Cookie", value)
            self._pending_session_cookie = None
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        extra: dict[str, str] | None = None,
    ) -> None:
        self._send_headers(status, content_type, len(data), extra)
        self.wfile.write(data)

    def _send_json(self, data, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status, {"Cache-Control": "no-store"})

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"detail": message}, status)

    def _send_file(
        self,
        path: Path,
        content_type: str | None = None,
        download_name: str | None = None,
        cache: str = "no-cache",
    ) -> None:
        if not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "파일을 찾을 수 없습니다.")
            return
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {"Cache-Control": cache}
        if download_name:
            safe_name = download_name.replace('"', "")
            headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'
        size = path.stat().st_size
        self._send_headers(HTTPStatus.OK, mime, size, headers)
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)

    def _read_body(self, limit: int) -> bytes:
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            raise GameError("Content-Length가 없는 요청은 처리할 수 없습니다.")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise GameError("잘못된 Content-Length입니다.") from exc
        if length < 0 or length > limit:
            raise GameError(f"요청 크기가 제한을 초과했습니다: {limit // (1024 * 1024)}MB")
        data = self.rfile.read(length)
        if len(data) != length:
            raise GameError("요청 본문을 끝까지 받지 못했습니다.")
        return data

    def _read_json(self) -> dict:
        raw = self._read_body(JSON_BODY_LIMIT)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GameError("올바른 JSON 요청이 아닙니다.") from exc
        if not isinstance(data, dict):
            raise GameError("JSON 객체가 필요합니다.")
        return data

    @staticmethod
    def _safe_child(base: Path, raw_relative: str) -> Path | None:
        try:
            candidate = (base / unquote(raw_relative)).resolve()
            candidate.relative_to(base.resolve())
            return candidate
        except (OSError, ValueError):
            return None

    def _cookie_session_id(self) -> str | None:
        raw = self.headers.get("Cookie", "")
        try:
            cookie = SimpleCookie()
            cookie.load(raw)
            value = cookie.get(SESSION_COOKIE)
            token = value.value if value is not None else ""
        except Exception:
            token = ""
        if len(token) == 43 and all(char.isalnum() or char in "-_" for char in token):
            return token
        return None

    def _session_id(self, create: bool = True) -> str | None:
        session_id = self._cookie_session_id()
        if session_id is None and create:
            session_id = secrets.token_urlsafe(32)
            self._pending_session_cookie = session_id
        return session_id

    def _worker(self, create: bool = True):
        if not PUBLIC_MODE or session_hub is None:
            return None
        session_id = self._session_id(create=create)
        if session_id is None:
            return None
        if create and pvp_hub is not None:
            pvp_hub.cleanup()
        worker = session_hub.get(session_id) if create else session_hub.peek(session_id)
        return worker

    def _pvp_session_id(self, create: bool = True) -> str:
        if not ONLINE_MATCHING_ENABLED or pvp_hub is None:
            raise GameError("이 서버에서는 온라인 플레이어 매칭이 비활성화되어 있습니다.")
        session_id = self._session_id(create=create)
        if session_id is None:
            raise GameError("온라인 세션이 없습니다. 다시 매칭을 시작하세요.")
        return session_id

    def _prepare_pvp_session(self) -> str:
        """Return the browser session id after releasing its AI/replay worker.

        Matchmaking waiters must not consume the process-wide game-worker slot.
        A browser may still own an AI or replay worker from an earlier screen,
        so entering any PvP queue explicitly closes that worker before the
        player is added to the queue or room. The PvP match itself acquires one
        shared worker only after two distinct players have been paired.
        """
        session_id = self._pvp_session_id(create=True)
        if PUBLIC_MODE and session_hub is not None:
            session_hub.cleanup()
            session_hub.close(session_id)
        if pvp_hub is not None:
            pvp_hub.cleanup()
        return session_id

    def _game_call(self, command: str, create: bool = True, **kwargs):
        if PUBLIC_MODE:
            worker = self._worker(create=create)
            if worker is None:
                if command == "state":
                    return {"active": False, "presets": manager.presets(), "message": "게임을 시작하세요."}
                raise GameError("게임 세션이 없습니다. 새 게임을 시작하세요.")
            result = worker.call(command, **kwargs)
            if command in {"close_game", "replay_close"}:
                session_id = self._session_id(create=False)
                if session_id is not None:
                    session_hub.close(session_id)
            return result
        if command == "state":
            return manager.public_state()
        if command == "start":
            return manager.start(kwargs["human_seat"], kwargs["human_deck"], kwargs["deck_label"], kwargs["agent_id"])
        if command == "human_action":
            return manager.act_human(kwargs["indices"])
        if command == "ai_step":
            return manager.act_ai_once()
        if command == "close_game":
            manager.close()
            return {"ok": True}
        if command == "record_zip":
            return str(manager.record_zip())
        if command == "official_replay":
            return str(manager.official_replay_file())
        if command == "viewer":
            return str(manager.viewer_launcher(kwargs["player"]))
        if command == "replay_load":
            return replay_manager.load(kwargs["data"], kwargs["filename"])
        if command == "replay_state":
            return replay_manager.state(kwargs["index"], kwargs["view_seat"])
        if command == "replay_close":
            replay_manager.close()
            return {"ok": True}
        if command == "submit_result":
            return result_collector.submit(manager, kwargs.get("player_name", ""), kwargs.get("note", ""))
        raise GameError(f"지원하지 않는 게임 명령입니다: {command}")

    def _require_admin(self) -> None:
        expected = admin_token()
        if not expected:
            raise PermissionError("CABT_ADMIN_TOKEN이 설정되지 않았습니다.")
        authorization = self.headers.get("Authorization", "")
        supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise PermissionError("관리자 토큰이 올바르지 않습니다.")

    def _resolve_human_deck(self, request: dict) -> tuple[list[int], str]:
        # Browser-local deck builder decks are sent as card IDs. This keeps
        # public users' custom decks isolated without storing them in a shared
        # Railway volume or requiring accounts.
        if isinstance(request.get("deck_cards"), list):
            cards = manager.validate_deck_cards(request["deck_cards"], exact=True)
            raw_name = str(request.get("deck_name") or "브라우저 저장 덱")
            deck_name = " ".join(raw_name.split()).strip()[:80] or "브라우저 저장 덱"
            return cards, deck_name
        deck_ref = str(request.get("deck_ref") or "")
        if deck_ref.startswith("saved:"):
            if PUBLIC_MODE:
                raise GameError("공개 서버에서는 저장 덱을 사용할 수 없습니다.")
            saved = deck_store.get(deck_ref.split(":", 1)[1])
            return manager.validate_deck_cards(saved["cards"], exact=True), saved["name"]
        if deck_ref.startswith("preset:"):
            preset = deck_ref.split(":", 1)[1]
            if preset not in {row["id"] for row in manager.presets()}:
                raise GameError(f"프리셋 덱을 찾을 수 없습니다: {preset}")
            return read_deck(ROOT / "decks" / f"{preset}.csv"), preset
        if isinstance(request.get("deck_text"), str) and request["deck_text"].strip():
            deck_text = request["deck_text"]
            filename = str(request.get("deck_filename") or "uploaded_deck.csv")
            cards = manager.validate_deck_cards(parse_uploaded_deck(deck_text, filename), exact=True)
            return cards, filename
        preset = str(request.get("preset") or "alakazam_mirror")
        deck_path = ROOT / "decks" / f"{preset}.csv"
        if not deck_path.is_file():
            raise GameError(f"프리셋 덱을 찾을 수 없습니다: {preset}")
        return read_deck(deck_path), preset

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self._send_file(ROOT / "templates" / "index.html", "text/html; charset=utf-8")
                return
            if path == "/admin":
                self._send_file(ROOT / "templates" / "admin.html", "text/html; charset=utf-8")
                return
            if path == "/api/health":
                if session_hub is not None:
                    session_hub.cleanup()
                if pvp_hub is not None:
                    pvp_hub.cleanup()
                self._send_json(
                    {
                        "ok": True,
                        "version": APP_VERSION,
                        "public_mode": PUBLIC_MODE,
                        "data_root": str(DATA_ROOT),
                        "online_matching": ONLINE_MATCHING_ENABLED,
                        "pvp": pvp_hub.stats() if pvp_hub is not None else None,
                        "workers": worker_capacity.stats() if worker_capacity is not None else None,
                        "railway_volume": os.environ.get("RAILWAY_VOLUME_MOUNT_PATH"),
                    }
                )
                return
            if path == "/api/config":
                agent_rows = [info.public() for info in agents.list()]
                self._send_json(
                    {
                        "version": APP_VERSION,
                        "presets": manager.presets(),
                        "saved_decks": [] if PUBLIC_MODE else deck_store.list(),
                        "card_types": manager.card_types(),
                        "agents": agent_rows,
                        "default_agent": agent_rows[0]["id"] if agent_rows else None,
                        "card_images": image_index.status(),
                        "ai_upload_limit_mb": MAX_ZIP_BYTES // (1024 * 1024),
                        "replay_upload_limit_mb": REPLAY_UPLOAD_LIMIT // (1024 * 1024),
                        "public_mode": PUBLIC_MODE,
                        "allow_agent_upload": not PUBLIC_MODE,
                        "allow_deck_save": True,
                        "deck_save_scope": "browser" if PUBLIC_MODE else "server",
                        "result_submission_enabled": RESULT_SUBMISSION_ENABLED,
                        "online_matching_enabled": ONLINE_MATCHING_ENABLED,
                        "max_sessions": MAX_SESSIONS if PUBLIC_MODE else 1,
                        "max_pvp_matches": MAX_PVP_MATCHES if ONLINE_MATCHING_ENABLED else 0,
                        "max_pvp_waiters": MAX_PVP_WAITERS if ONLINE_MATCHING_ENABLED else 0,
                        "max_active_workers": MAX_ACTIVE_WORKERS if PUBLIC_MODE else 1,
                        "client_disconnect_seconds": CLIENT_DISCONNECT_SECONDS,
                        "client_disconnect_grace_seconds": CLIENT_DISCONNECT_GRACE_SECONDS,
                    }
                )
                return
            if path == "/api/cards":
                query = parse_qs(parsed.query)
                search = (query.get("q") or [""])[0]
                card_type = (query.get("type") or [""])[0]
                offset = int((query.get("offset") or ["0"])[0])
                limit = int((query.get("limit") or ["48"])[0])
                self._send_json(manager.search_cards(search, card_type, offset, limit))
                return
            if path == "/api/decks":
                self._send_json({"decks": [] if PUBLIC_MODE else deck_store.list()})
                return
            if path.startswith("/api/decks/saved/"):
                if PUBLIC_MODE:
                    raise GameError("공개 서버에서는 저장 덱을 제공하지 않습니다.")
                deck_id = unquote(path[len("/api/decks/saved/") :])
                deck = deck_store.get(deck_id)
                deck["source"] = "saved"
                deck["details"] = manager.deck_details(deck["cards"])
                self._send_json(deck)
                return
            if path.startswith("/api/decks/preset/"):
                preset_id = unquote(path[len("/api/decks/preset/") :])
                available = {row["id"]: row["label"] for row in manager.presets()}
                if preset_id not in available:
                    raise GameError("프리셋 덱을 찾을 수 없습니다.")
                cards = read_deck(ROOT / "decks" / f"{preset_id}.csv")
                self._send_json(
                    {
                        "id": preset_id,
                        "name": available[preset_id],
                        "source": "preset",
                        "card_count": len(cards),
                        "cards": cards,
                        "details": manager.deck_details(cards),
                    }
                )
                return
            if path == "/api/state":
                self._send_json(self._game_call("state", create=False))
                return
            if path == "/api/matchmaking/status":
                session_id = self._pvp_session_id(create=True)
                self._send_json(pvp_hub.status(session_id))
                return
            if path == "/api/pvp/state":
                session_id = self._pvp_session_id(create=False)
                self._send_json(pvp_hub.state(session_id))
                return
            if path == "/api/replay/state":
                query = parse_qs(parsed.query)
                index = int((query.get("index") or ["0"])[0])
                view_seat = int((query.get("view_seat") or ["0"])[0])
                self._send_json(self._game_call("replay_state", create=False, index=index, view_seat=view_seat))
                return
            if path == "/api/card-images/status":
                self._send_json(image_index.status())
                return
            if path == "/api/game/record.zip":
                record_path = Path(self._game_call("record_zip", create=False))
                self._send_file(record_path, "application/zip", download_name=record_path.name, cache="no-store")
                return
            if path == "/api/pvp/record.zip":
                session_id = self._pvp_session_id(create=False)
                record_path = Path(pvp_hub.game_file(session_id, "record_zip"))
                self._send_file(record_path, "application/zip", download_name=record_path.name, cache="no-store")
                return
            if path == "/api/pvp/official.json":
                session_id = self._pvp_session_id(create=False)
                official_path = Path(pvp_hub.game_file(session_id, "official_replay"))
                self._send_file(
                    official_path,
                    "application/json; charset=utf-8",
                    download_name=official_path.name,
                    cache="no-store",
                )
                return
            if path.startswith("/api/pvp/viewer/"):
                session_id = self._pvp_session_id(create=False)
                player = int(path.rsplit("/", 1)[-1])
                viewer = Path(pvp_hub.game_file(session_id, "viewer", player=player))
                self._send_file(viewer, "text/html; charset=utf-8")
                return
            if path == "/api/game/official.json":
                official_path = Path(self._game_call("official_replay", create=False))
                self._send_file(
                    official_path,
                    "application/json; charset=utf-8",
                    download_name=official_path.name,
                    cache="no-store",
                )
                return
            if path.startswith("/api/game/viewer/"):
                player = int(path.rsplit("/", 1)[-1])
                self._send_file(Path(self._game_call("viewer", create=False, player=player)), "text/html; charset=utf-8")
                return
            if path == "/api/admin/submissions":
                self._require_admin()
                query = parse_qs(parsed.query)
                limit = int((query.get("limit") or ["200"])[0])
                self._send_json({"submissions": result_collector.list(limit)})
                return
            if path.startswith("/api/admin/submissions/") and path.endswith("/zip"):
                self._require_admin()
                submission_id = unquote(path[len("/api/admin/submissions/") : -len("/zip")]).strip("/")
                zip_path = result_collector.zip_path(submission_id)
                self._send_file(zip_path, "application/zip", download_name=zip_path.name, cache="no-store")
                return
            if path.startswith("/api/admin/submissions/") and path.endswith("/summary"):
                self._require_admin()
                submission_id = unquote(path[len("/api/admin/submissions/") : -len("/summary")]).strip("/")
                summary_path = result_collector.summary_path(submission_id)
                self._send_file(summary_path, "application/json; charset=utf-8", cache="no-store")
                return
            if path.startswith("/api/card-image/"):
                card_id = int(path.rsplit("/", 1)[-1])
                image_path = image_index.find(card_id)
                if image_path is None:
                    self._send_error_json(HTTPStatus.NOT_FOUND, f"카드 이미지를 찾지 못했습니다: {card_id}")
                    return
                self._send_file(image_path, cache="public, max-age=86400")
                return
            if path.startswith("/static/"):
                static_root = ROOT / "static"
                target = self._safe_child(static_root, path[len("/static/") :])
                if target is None:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "잘못된 정적 파일 경로입니다.")
                    return
                cache = "no-cache" if target.suffix.lower() in {".js", ".css", ".html"} else "public, max-age=86400"
                self._send_file(target, cache=cache)
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "지원하지 않는 경로입니다.")
        except PermissionError as exc:
            self._send_error_json(HTTPStatus.UNAUTHORIZED, str(exc))
        except WorkerError as exc:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (GameError, AgentLoadError, DeckStoreError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except BrokenPipeError:
            pass
        except Exception as exc:
            traceback.print_exc()
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"서버 오류 · {type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/replay/load":
                query = parse_qs(parsed.query)
                filename = (query.get("filename") or ["replay.zip"])[0]
                data = self._read_body(REPLAY_UPLOAD_LIMIT)
                self._send_json(self._game_call("replay_load", data=data, filename=filename))
                return
            if path == "/api/replay/close":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                self._send_json(self._game_call("replay_close", create=False))
                return
            if path == "/api/client/heartbeat":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                session_id = self._session_id(create=False)
                if session_id is not None:
                    if session_hub is not None:
                        session_hub.touch(session_id)
                    if pvp_hub is not None:
                        pvp_hub.heartbeat(session_id)
                self._send_json({"ok": True})
                return
            if path == "/api/client/disconnect":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                session_id = self._session_id(create=False)
                if session_id is not None:
                    if session_hub is not None:
                        session_hub.mark_disconnected(session_id)
                    if pvp_hub is not None:
                        pvp_hub.mark_disconnected(session_id)
                self._send_json({"ok": True})
                return
            if path == "/api/game/start":
                request = self._read_json()
                human_seat = int(request.get("human_seat", 0))
                agent_id = str(request.get("agent_id") or "")
                human_deck, deck_label = self._resolve_human_deck(request)
                self._send_json(
                    self._game_call(
                        "start",
                        human_seat=human_seat,
                        human_deck=human_deck,
                        deck_label=deck_label,
                        agent_id=agent_id,
                    )
                )
                return
            if path == "/api/matchmaking/quick":
                request = self._read_json()
                player_deck, deck_label = self._resolve_human_deck(request)
                session_id = self._prepare_pvp_session()
                self._send_json(
                    pvp_hub.join_quick(
                        session_id,
                        str(request.get("player_name") or ""),
                        player_deck,
                        deck_label,
                    )
                )
                return
            if path == "/api/matchmaking/create-room":
                request = self._read_json()
                player_deck, deck_label = self._resolve_human_deck(request)
                session_id = self._prepare_pvp_session()
                self._send_json(
                    pvp_hub.create_private(
                        session_id,
                        str(request.get("player_name") or ""),
                        player_deck,
                        deck_label,
                    )
                )
                return
            if path == "/api/matchmaking/join-room":
                request = self._read_json()
                player_deck, deck_label = self._resolve_human_deck(request)
                session_id = self._prepare_pvp_session()
                self._send_json(
                    pvp_hub.join_private(
                        session_id,
                        str(request.get("room_code") or ""),
                        str(request.get("player_name") or ""),
                        player_deck,
                        deck_label,
                    )
                )
                return
            if path == "/api/matchmaking/cancel":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                session_id = self._pvp_session_id(create=False)
                self._send_json(pvp_hub.cancel(session_id))
                return
            if path == "/api/pvp/action":
                request = self._read_json()
                indices = request.get("indices")
                if not isinstance(indices, list):
                    raise GameError("indices 배열이 필요합니다.")
                session_id = self._pvp_session_id(create=False)
                self._send_json(pvp_hub.action(session_id, indices))
                return
            if path == "/api/pvp/leave":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                session_id = self._pvp_session_id(create=False)
                self._send_json(pvp_hub.leave(session_id))
                return
            if path == "/api/decks/inspect":
                request = self._read_json()
                exact = bool(request.get("exact", False))
                cards = manager.validate_deck_cards(request.get("cards"), exact=exact)
                raw_name = str(request.get("name") or "브라우저 저장 덱")
                name = " ".join(raw_name.split()).strip()[:80] or "브라우저 저장 덱"
                self._send_json(
                    {
                        "name": name,
                        "card_count": len(cards),
                        "cards": cards,
                        "details": manager.deck_details(cards),
                    }
                )
                return
            if path == "/api/decks/save":
                if PUBLIC_MODE:
                    raise GameError("공개 서버에서는 덱을 서버에 저장할 수 없습니다.")
                request = self._read_json()
                cards = manager.validate_deck_cards(request.get("cards"), exact=False)
                raw_id = request.get("id")
                deck_id = str(raw_id) if raw_id else None
                saved = deck_store.save(str(request.get("name") or ""), cards, deck_id)
                saved["source"] = "saved"
                saved["details"] = manager.deck_details(saved["cards"])
                self._send_json({"deck": saved, "decks": deck_store.list()})
                return
            if path == "/api/decks/delete":
                if PUBLIC_MODE:
                    raise GameError("공개 서버에서는 덱을 삭제할 수 없습니다.")
                request = self._read_json()
                deck_id = str(request.get("id") or "")
                if not deck_id:
                    raise GameError("삭제할 덱 ID가 필요합니다.")
                deck_store.delete(deck_id)
                self._send_json({"ok": True, "decks": deck_store.list()})
                return
            if path == "/api/game/action":
                request = self._read_json()
                indices = request.get("indices")
                if not isinstance(indices, list):
                    raise GameError("indices 배열이 필요합니다.")
                self._send_json(self._game_call("human_action", create=False, indices=indices))
                return
            if path == "/api/game/ai-step":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                self._send_json(self._game_call("ai_step", create=False))
                return
            if path == "/api/game/submit":
                if not RESULT_SUBMISSION_ENABLED:
                    raise GameError("이 서버에서는 결과 수집 기능이 비활성화되어 있습니다.")
                request = self._read_json()
                if not bool(request.get("consent", False)):
                    raise GameError("결과 전송 동의가 필요합니다.")
                self._send_json(
                    self._game_call(
                        "submit_result",
                        create=False,
                        player_name=str(request.get("player_name") or ""),
                        note=str(request.get("note") or ""),
                    )
                )
                return
            if path == "/api/game/close":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                self._send_json(self._game_call("close_game", create=False))
                return
            if path == "/api/card-images/rescan":
                if PUBLIC_MODE:
                    raise GameError("공개 서버에서는 이미지 재검색을 실행할 수 없습니다.")
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length:
                    self._read_body(JSON_BODY_LIMIT)
                self._send_json(image_index.rescan())
                return
            if path == "/api/agents/install":
                if PUBLIC_MODE:
                    raise GameError("공개 서버에서는 보안을 위해 AI ZIP 업로드가 비활성화되어 있습니다.")
                query = parse_qs(parsed.query)
                filename = (query.get("filename") or ["submission.zip"])[0]
                data = self._read_body(MAX_ZIP_BYTES)
                installed = agents.install_zip(data, filename)
                self._send_json({"installed": installed.public(), "agents": [info.public() for info in agents.list()]})
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "지원하지 않는 경로입니다.")
        except PermissionError as exc:
            self._send_error_json(HTTPStatus.UNAUTHORIZED, str(exc))
        except WorkerError as exc:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except (GameError, AgentLoadError, DeckStoreError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except BrokenPipeError:
            pass
        except Exception as exc:
            traceback.print_exc()
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"서버 오류 · {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CABT Human vs AI web arena and replay viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    server = CABTServer((options.host, options.port), CABTHandler)

    def handle_termination(_signum, _frame) -> None:
        raise KeyboardInterrupt

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_termination)
    local_url = f"http://127.0.0.1:{options.port}"
    print("")
    print("CABT Human vs AI · Web Arena")
    print(f"브라우저 주소: {local_url}")
    print(f"모드: {'PUBLIC MULTI-SESSION' if PUBLIC_MODE else 'LOCAL'}")
    print(f"데이터 폴더: {DATA_ROOT}")
    if options.host == "0.0.0.0":
        print("외부 접속 모드: 리버스 프록시 또는 같은 네트워크에서 접속할 수 있습니다.")
    print("종료하려면 이 창에서 Ctrl+C를 누르세요.")
    print("")
    if not options.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(local_url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nCABT 서버를 종료합니다.")
    finally:
        if session_hub is not None:
            session_hub.shutdown()
        if pvp_hub is not None:
            pvp_hub.shutdown()
        replay_manager.close()
        manager.close()
        server.server_close()


if __name__ == "__main__":
    main()
