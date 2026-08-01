from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator


MAX_ZIP_BYTES = 256 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 20_000
SLUG_RE = re.compile(r"[^a-z0-9]+")


class AgentLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentInfo:
    id: str
    name: str
    path: Path
    source: str
    original_filename: str
    installed_at: str
    deck_cards: int
    has_native_engine: bool

    def public(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("path", None)
        return result


@contextmanager
def agent_working_directory(root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


def read_deck(path: Path) -> list[int]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AgentLoadError(f"{path.name}: UTF-8 형식의 덱 파일이 아닙니다.") from exc
    values: list[int] = []
    for token in raw.replace(",", " ").split():
        if token.lstrip("+-").isdigit():
            values.append(int(token))
    if len(values) != 60:
        raise AgentLoadError(f"{path.name}: 카드 ID가 60개여야 합니다. 현재 {len(values)}개입니다.")
    return values


class AgentRuntime:
    def __init__(
        self,
        info: AgentInfo,
        module: ModuleType,
        agent: Callable[[dict[str, Any]], Any],
        cg_api: ModuleType,
        cg_game: ModuleType,
        loaded_module_names: set[str],
    ) -> None:
        self.info = info
        self.module = module
        self.agent_callable = agent
        self.cg_api = cg_api
        self.cg_game = cg_game
        self.loaded_module_names = loaded_module_names

    def act(self, observation: dict[str, Any]) -> list[int]:
        with agent_working_directory(self.info.path):
            result = self.agent_callable(observation)
        if not isinstance(result, (list, tuple)):
            raise AgentLoadError(
                f"{self.info.name}: agent() 반환값이 list[int]가 아닙니다: {type(result).__name__}"
            )
        try:
            return [int(value) for value in result]
        except (TypeError, ValueError) as exc:
            raise AgentLoadError(f"{self.info.name}: 행동 배열에 정수가 아닌 값이 있습니다.") from exc

    def deck(self) -> list[int]:
        # CABT's initial agent call contains all Observation keys even though
        # ``current`` and ``select`` are None. Some submissions convert the
        # dictionary to the official dataclass before checking ``select``.
        result = self.act({"select": None, "logs": [], "current": None})
        if len(result) != 60:
            raise AgentLoadError(
                f"{self.info.name}: 초기 agent()가 60장 덱을 반환하지 않았습니다. 현재 {len(result)}장입니다."
            )
        return result

    def battle_start(self, deck0: list[int], deck1: list[int]):
        return self.cg_game.battle_start(deck0, deck1)

    def battle_select(self, action: list[int]) -> dict[str, Any]:
        return self.cg_game.battle_select(action)

    def battle_finish(self) -> None:
        self.cg_game.battle_finish()

    def visualize_data(self) -> str:
        return self.cg_game.visualize_data()


class AgentRepository:
    def __init__(self, root: Path, engine_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine_root = (engine_root or (self.root.parent / "engine")).resolve()
        self.engine_cg = self.engine_root / "cg"
        self._loaded_module_names: set[str] = set()
        self._active_runtime: AgentRuntime | None = None

    @staticmethod
    def _pretty_name(value: str) -> str:
        value = re.sub(r"\(\d+\)", "", value)
        value = re.sub(r"[_\-]+", " ", value)
        value = re.sub(r"\bsubmission\b", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip()
        return value.title() or "CABT Agent"

    @staticmethod
    def _slug(value: str) -> str:
        slug = SLUG_RE.sub("-", value.lower()).strip("-")
        return slug[:48] or "agent"

    @staticmethod
    def _manifest_path(root: Path) -> Path:
        return root / "cabt_agent.json"

    def _read_manifest(self, root: Path) -> dict[str, Any]:
        path = self._manifest_path(root)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _info_from_root(self, root: Path) -> AgentInfo | None:
        if not (root / "main.py").is_file() or not (root / "deck.csv").is_file():
            return None
        try:
            deck_cards = len(read_deck(root / "deck.csv"))
        except AgentLoadError:
            deck_cards = 0
        manifest = self._read_manifest(root)
        return AgentInfo(
            id=root.name,
            name=str(manifest.get("name") or self._pretty_name(root.name)),
            path=root.resolve(),
            source=str(manifest.get("source") or "local"),
            original_filename=str(manifest.get("original_filename") or root.name),
            installed_at=str(manifest.get("installed_at") or ""),
            deck_cards=deck_cards,
            has_native_engine=(self.engine_cg / "api.py").is_file()
            or (root / "cg" / "api.py").is_file(),
        )

    def list(self) -> list[AgentInfo]:
        items = [
            info
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
            for info in [self._info_from_root(path)]
            if info is not None
        ]
        return sorted(items, key=lambda item: (item.source != "bundled", item.name.lower()))

    def get(self, agent_id: str) -> AgentInfo:
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", agent_id):
            raise AgentLoadError("잘못된 AI ID입니다.")
        root = (self.root / agent_id).resolve()
        if root.parent != self.root:
            raise AgentLoadError("AI 경로가 올바르지 않습니다.")
        info = self._info_from_root(root)
        if info is None:
            raise AgentLoadError(f"AI를 찾을 수 없습니다: {agent_id}")
        if info.deck_cards != 60:
            raise AgentLoadError(f"{info.name}: deck.csv가 올바른 60장 덱이 아닙니다.")
        return info

    def _purge_previous_modules(self) -> None:
        names = {
            name
            for name in self._loaded_module_names
            if name != "cg" and not name.startswith("cg.")
        }
        names.update(
            name
            for name in sys.modules
            if name == "cabt_hybrid"
            or name.startswith("cabt_hybrid.")
            or name.startswith("_cabt_agent_")
        )
        for name in sorted(names, key=len, reverse=True):
            sys.modules.pop(name, None)
        self._loaded_module_names.clear()
        self._active_runtime = None
        importlib.invalidate_caches()
        gc.collect()

    def _shared_engine(self, info: AgentInfo) -> tuple[ModuleType, ModuleType]:
        if not (self.engine_cg / "api.py").is_file():
            raise AgentLoadError("공용 CABT 엔진의 cg/api.py를 찾을 수 없습니다.")
        # A few Kaggle agents locate deck.csv from cg.api.__file__. Keep a
        # selected-deck copy beside the shared engine for those submissions.
        shutil.copyfile(info.path / "deck.csv", self.engine_root / "deck.csv")
        engine_text = str(self.engine_root)
        inserted = engine_text not in sys.path
        if inserted:
            sys.path.insert(0, engine_text)
        try:
            cg_api = importlib.import_module("cg.api")
            cg_game = importlib.import_module("cg.game")
        finally:
            if inserted:
                try:
                    sys.path.remove(engine_text)
                except ValueError:
                    pass
        return cg_api, cg_game

    def load(self, agent_id: str) -> AgentRuntime:
        info = self.get(agent_id)
        self._purge_previous_modules()
        cg_api, cg_game = self._shared_engine(info)
        before_modules = set(sys.modules)
        module_name = f"_cabt_agent_{self._slug(agent_id).replace('-', '_')}_{time.time_ns()}"
        try:
            with agent_working_directory(info.path):
                spec = importlib.util.spec_from_file_location(module_name, info.path / "main.py")
                if spec is None or spec.loader is None:
                    raise AgentLoadError(f"{info.name}: main.py를 불러올 수 없습니다.")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception as exc:
            self._purge_previous_modules()
            if isinstance(exc, AgentLoadError):
                raise
            raise AgentLoadError(f"{info.name} 로드 실패 · {type(exc).__name__}: {exc}") from exc

        agent = getattr(module, "agent", None)
        if not callable(agent):
            self._purge_previous_modules()
            raise AgentLoadError(f"{info.name}: main.py에 호출 가능한 agent(observation)가 없습니다.")

        loaded_names: set[str] = {module_name}
        for name in set(sys.modules) - before_modules:
            loaded = sys.modules.get(name)
            file_name = getattr(loaded, "__file__", None)
            if not file_name:
                continue
            try:
                Path(file_name).resolve().relative_to(info.path)
            except (OSError, ValueError):
                continue
            loaded_names.add(name)

        runtime = AgentRuntime(info, module, agent, cg_api, cg_game, loaded_names)
        try:
            runtime.deck()
        except Exception:
            self._loaded_module_names = loaded_names
            self._purge_previous_modules()
            raise
        self._loaded_module_names = loaded_names
        self._active_runtime = runtime
        return runtime

    @staticmethod
    def _validate_archive_member(member: zipfile.ZipInfo) -> None:
        name = member.filename.replace("\\", "/")
        path = Path(name)
        if not name or name.startswith("/") or path.is_absolute() or ".." in path.parts:
            raise AgentLoadError(f"안전하지 않은 ZIP 경로입니다: {member.filename}")
        if path.parts and ":" in path.parts[0]:
            raise AgentLoadError(f"안전하지 않은 ZIP 경로입니다: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise AgentLoadError("심볼릭 링크가 포함된 ZIP은 불러올 수 없습니다.")

    @staticmethod
    def _find_submission_root(extracted: Path) -> Path:
        candidates = []
        for main_path in extracted.rglob("main.py"):
            root = main_path.parent
            if (root / "deck.csv").is_file():
                score = (
                    0 if (root / "cg" / "api.py").is_file() else 1,
                    len(root.relative_to(extracted).parts),
                )
                candidates.append((score, root))
        if not candidates:
            raise AgentLoadError("ZIP 안에서 main.py와 deck.csv가 함께 있는 Kaggle 제출 루트를 찾지 못했습니다.")
        candidates.sort(key=lambda item: item[0])
        best_score = candidates[0][0]
        best = [root for score, root in candidates if score == best_score]
        if len(best) > 1:
            raise AgentLoadError("ZIP 안에 제출 루트가 여러 개 있어 AI를 특정할 수 없습니다.")
        return best[0]

    def install_zip(self, data: bytes, filename: str, source: str = "uploaded") -> AgentInfo:
        if not filename.lower().endswith(".zip"):
            raise AgentLoadError("Kaggle 제출 ZIP 파일만 불러올 수 있습니다.")
        if not data:
            raise AgentLoadError("빈 ZIP 파일입니다.")
        if len(data) > MAX_ZIP_BYTES:
            raise AgentLoadError("AI ZIP은 256MB 이하여야 합니다.")

        digest = hashlib.sha256(data).hexdigest()
        base_name = Path(filename).stem
        agent_id = f"{self._slug(base_name)}-{digest[:10]}"
        existing = self._info_from_root(self.root / agent_id)
        if existing is not None:
            return existing

        with tempfile.TemporaryDirectory(prefix=".cabt-agent-", dir=self.root) as tmp_text:
            tmp = Path(tmp_text)
            zip_path = tmp / "submission.zip"
            zip_path.write_bytes(data)
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    members = archive.infolist()
                    if len(members) > MAX_ARCHIVE_FILES:
                        raise AgentLoadError("ZIP 안의 파일 수가 너무 많습니다.")
                    unpacked_size = sum(max(0, member.file_size) for member in members)
                    if unpacked_size > MAX_UNPACKED_BYTES:
                        raise AgentLoadError("압축 해제 크기가 1GB를 초과합니다.")
                    for member in members:
                        self._validate_archive_member(member)
                    extracted = tmp / "extracted"
                    extracted.mkdir()
                    archive.extractall(extracted)
            except zipfile.BadZipFile as exc:
                raise AgentLoadError("손상되었거나 지원하지 않는 ZIP 파일입니다.") from exc

            submission_root = self._find_submission_root(extracted)
            read_deck(submission_root / "deck.csv")
            if not (self.engine_cg / "api.py").is_file():
                raise AgentLoadError("공용 CABT 엔진을 찾을 수 없습니다.")

            destination = self.root / agent_id
            shutil.copytree(submission_root, destination)
            manifest = {
                "id": agent_id,
                "name": self._pretty_name(base_name),
                "source": source,
                "original_filename": filename,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "sha256": digest,
            }
            self._manifest_path(destination).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        info = self._info_from_root(self.root / agent_id)
        if info is None:
            raise AgentLoadError("AI를 저장한 뒤 다시 찾지 못했습니다.")
        return info
