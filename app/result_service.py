from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game_service import DATA_ROOT, GameError, GameManager


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ResultCollector:
    """Store completed user-vs-AI records selected for server-side collection."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or (DATA_ROOT / "submissions")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    @staticmethod
    def _clean_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        text = " ".join(text.split())
        return text[:limit]

    def submit(
        self,
        manager: GameManager,
        player_name: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        with self.lock, manager.lock:
            if manager.record_dir is None or manager.game_id is None:
                raise GameError("제출할 대전 기록이 없습니다.")
            if manager.observation is None:
                raise GameError("게임 상태를 찾을 수 없습니다.")
            result = int((manager.observation.get("current") or {}).get("result", -1))
            if result == -1:
                raise GameError("게임이 끝난 뒤 결과를 제출할 수 있습니다.")

            manager._finalize_record()  # Ensure metadata/replay files are current.
            source_zip = manager.record_zip()
            game_id = manager.game_id
            summary_path = self.root / f"{game_id}.json"
            zip_path = self.root / f"{game_id}.zip"

            existing: dict[str, Any] | None = None
            if summary_path.is_file():
                try:
                    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except (OSError, json.JSONDecodeError):
                    existing = None
            if existing is not None and zip_path.is_file():
                return {"ok": True, "duplicate": True, "submission": existing}

            metadata_path = manager.record_dir / "metadata.json"
            metadata: dict[str, Any] = {}
            if metadata_path.is_file():
                try:
                    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metadata = loaded
                except (OSError, json.JSONDecodeError):
                    metadata = {}

            submitted_at = datetime.now(timezone.utc).isoformat()
            human_seat = int(metadata.get("human_seat", manager.human_seat))
            result_text = "draw" if result == 2 else "human_win" if result == human_seat else "ai_win"
            summary = {
                "submission_id": game_id,
                "game_id": game_id,
                "submitted_at": submitted_at,
                "player_name": self._clean_text(player_name, 48) or "Anonymous",
                "note": self._clean_text(note, 240),
                "human_seat": human_seat,
                "ai_seat": int(metadata.get("ai_seat", manager.ai_seat)),
                "agent_id": str(metadata.get("agent_id") or ""),
                "agent_name": str(metadata.get("agent_name") or "AI"),
                "human_deck_label": str(metadata.get("human_deck_label") or manager.human_deck_label),
                "result": result,
                "result_text": result_text,
                "decision_count": int(manager.decision_count),
                "duration_seconds": round(max(0.0, (time.time() - manager.started_at) if manager.started_at else 0.0), 2),
                "record_zip": zip_path.name,
                "source": "cabt_web",
            }

            temp_zip = zip_path.with_suffix(".zip.tmp")
            temp_json = summary_path.with_suffix(".json.tmp")
            shutil.copyfile(source_zip, temp_zip)
            temp_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_zip.replace(zip_path)
            temp_json.replace(summary_path)
            return {"ok": True, "duplicate": False, "submission": summary}

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.lock:
            for path in self.root.glob("*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(item, dict):
                    item = dict(item)
                    item["zip_available"] = (self.root / f"{path.stem}.zip").is_file()
                    rows.append(item)
        rows.sort(key=lambda item: str(item.get("submitted_at") or ""), reverse=True)
        return rows[: max(1, min(int(limit), 1000))]

    def zip_path(self, submission_id: str) -> Path:
        if not SAFE_ID_RE.fullmatch(submission_id):
            raise GameError("잘못된 제출 ID입니다.")
        path = (self.root / f"{submission_id}.zip").resolve()
        if path.parent != self.root or not path.is_file():
            raise GameError("제출 ZIP을 찾을 수 없습니다.")
        return path

    def summary_path(self, submission_id: str) -> Path:
        if not SAFE_ID_RE.fullmatch(submission_id):
            raise GameError("잘못된 제출 ID입니다.")
        path = (self.root / f"{submission_id}.json").resolve()
        if path.parent != self.root or not path.is_file():
            raise GameError("제출 요약을 찾을 수 없습니다.")
        return path


def admin_token() -> str:
    return os.environ.get("CABT_ADMIN_TOKEN", "").strip()
