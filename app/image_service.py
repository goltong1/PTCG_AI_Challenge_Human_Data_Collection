from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif"}
EXTENSION_PRIORITY = {".webp": 0, ".png": 1, ".jpg": 2, ".jpeg": 3, ".avif": 4, ".gif": 5}
NUMBER_TOKEN = re.compile(r"(?<!\d)(\d{1,7})(?!\d)")


@dataclass(frozen=True)
class ImageEntry:
    card_id: int
    path: Path
    exact_name: bool


class CardImageIndex:
    """Recursively discovers card images anywhere below the project root.

    Supported examples:
      card_images/741.png
      card_images/set_a/card_741.webp
      images/0741 - Abra.jpg

    A numeric token in the filename is treated as a card ID only when it is
    present in ``valid_card_ids``. Exact numeric stems receive priority.
    """

    def __init__(self, root: Path, valid_card_ids: Iterable[int]) -> None:
        self.root = root.resolve()
        self.valid_card_ids = {int(v) for v in valid_card_ids}
        self._lock = threading.RLock()
        self._images: dict[int, Path] = {}
        self._scanned_at = 0.0
        self._revision = 0
        self.rescan()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._images)

    @property
    def scanned_at(self) -> float:
        with self._lock:
            return self._scanned_at

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "count": len(self._images),
                "revision": self._revision,
                "scanned_at": self._scanned_at,
                "search_root": str(self.root),
                "supported_extensions": sorted(IMAGE_SUFFIXES),
            }

    def _candidate_entries(self) -> list[ImageEntry]:
        entries: list[ImageEntry] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # Ignore generated game records and virtual environments.
            lowered_parts = {part.lower() for part in path.relative_to(self.root).parts[:-1]}
            if lowered_parts.intersection({"records", ".venv", "__pycache__", ".git"}):
                continue

            stem = path.stem.strip()
            exact_id: int | None = None
            if stem.isdigit():
                exact_id = int(stem)
                if exact_id in self.valid_card_ids:
                    entries.append(ImageEntry(exact_id, path, True))
                    continue

            tokens = [int(match.group(1)) for match in NUMBER_TOKEN.finditer(stem)]
            for card_id in tokens:
                if card_id in self.valid_card_ids:
                    entries.append(ImageEntry(card_id, path, False))
        return entries

    def _rank(self, entry: ImageEntry) -> tuple[int, int, int, str]:
        relative = entry.path.relative_to(self.root)
        return (
            0 if entry.exact_name else 1,
            EXTENSION_PRIORITY.get(entry.path.suffix.lower(), 99),
            len(relative.parts),
            str(relative).lower(),
        )

    def rescan(self) -> dict[str, object]:
        with self._lock:
            chosen: dict[int, ImageEntry] = {}
            for entry in self._candidate_entries():
                previous = chosen.get(entry.card_id)
                if previous is None or self._rank(entry) < self._rank(previous):
                    chosen[entry.card_id] = entry
            self._images = {card_id: entry.path for card_id, entry in chosen.items()}
            self._scanned_at = time.time()
            self._revision += 1
            return self.status()

    def find(self, card_id: int) -> Path | None:
        card_id = int(card_id)
        with self._lock:
            path = self._images.get(card_id)
            if path is not None and path.is_file():
                return path

        # Images may have been copied while the server was running. Rescan once
        # on a miss so no restart is required.
        self.rescan()
        with self._lock:
            path = self._images.get(card_id)
            return path if path is not None and path.is_file() else None
