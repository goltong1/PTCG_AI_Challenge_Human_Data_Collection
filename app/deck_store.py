from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DeckStoreError(RuntimeError):
    pass


class DeckStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": 1, "decks": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeckStoreError(f"저장된 덱 파일을 읽을 수 없습니다: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("decks"), list):
            raise DeckStoreError("저장된 덱 파일의 형식이 올바르지 않습니다.")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError as exc:
            raise DeckStoreError(f"덱을 저장하지 못했습니다: {exc}") from exc

    @staticmethod
    def _public(deck: dict[str, Any], include_cards: bool = False) -> dict[str, Any]:
        result = {
            "id": str(deck["id"]),
            "name": str(deck["name"]),
            "card_count": len(deck.get("cards") or []),
            "created_at": str(deck.get("created_at") or ""),
            "updated_at": str(deck.get("updated_at") or ""),
        }
        if include_cards:
            result["cards"] = [int(card_id) for card_id in deck.get("cards") or []]
        return result

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            decks = self._read()["decks"]
            rows = [self._public(deck) for deck in decks]
            return sorted(rows, key=lambda deck: (deck["name"].casefold(), deck["id"]))

    def get(self, deck_id: str) -> dict[str, Any]:
        with self.lock:
            for deck in self._read()["decks"]:
                if str(deck.get("id")) == deck_id:
                    return self._public(deck, include_cards=True)
        raise DeckStoreError("저장된 덱을 찾을 수 없습니다.")

    def save(self, name: str, cards: list[int], deck_id: str | None = None) -> dict[str, Any]:
        clean_name = " ".join(str(name).split()).strip()
        if not clean_name:
            raise DeckStoreError("덱 이름을 입력하세요.")
        if len(clean_name) > 80:
            raise DeckStoreError("덱 이름은 80자 이하여야 합니다.")

        with self.lock:
            data = self._read()
            now = self._now()
            target = None
            if deck_id:
                target = next(
                    (deck for deck in data["decks"] if str(deck.get("id")) == deck_id),
                    None,
                )
                if target is None:
                    raise DeckStoreError("수정할 덱을 찾을 수 없습니다.")
            if target is None:
                target = {
                    "id": uuid.uuid4().hex[:16],
                    "created_at": now,
                }
                data["decks"].append(target)
            target.update(
                {
                    "name": clean_name,
                    "cards": [int(card_id) for card_id in cards],
                    "updated_at": now,
                }
            )
            self._write(data)
            return self._public(target, include_cards=True)

    def delete(self, deck_id: str) -> None:
        with self.lock:
            data = self._read()
            original_count = len(data["decks"])
            data["decks"] = [
                deck for deck in data["decks"] if str(deck.get("id")) != deck_id
            ]
            if len(data["decks"]) == original_count:
                raise DeckStoreError("삭제할 덱을 찾을 수 없습니다.")
            self._write(data)
