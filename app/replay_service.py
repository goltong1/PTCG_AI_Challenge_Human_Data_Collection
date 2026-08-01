from __future__ import annotations

import copy
import io
import json
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from game_service import (
    AREA_NAMES,
    CONTEXT_KO,
    CONTEXT_NAMES,
    ENERGY_NAMES,
    GameError,
    GameManager,
)


REPLAY_UPLOAD_LIMIT = 128 * 1024 * 1024
REPLAY_JSON_LIMIT = 256 * 1024 * 1024
REPLAY_UNCOMPRESSED_LIMIT = 512 * 1024 * 1024
MAX_REPLAY_FRAMES = 5000


def _pascal(name: str) -> str:
    return "".join(part.title() for part in name.lower().split("_"))


def _enum_aliases(rows: dict[int, str]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for value, name in rows.items():
        for alias in (name, name.replace("_", ""), _pascal(name)):
            aliases[alias.casefold()] = value
    return aliases


CONTEXT_ALIASES = _enum_aliases({value: name for value, name in CONTEXT_NAMES.items()})
OPTION_ALIASES = {
    "number": 0,
    "yes": 1,
    "no": 2,
    "card": 3,
    "toolcard": 4,
    "tool": 4,
    "energycard": 5,
    "energy": 6,
    "play": 7,
    "attach": 8,
    "evolve": 9,
    "ability": 10,
    "discard": 11,
    "retreat": 12,
    "attack": 13,
    "end": 14,
    "skill": 15,
    "effectorder": 15,
    "specialcondition": 16,
}
LOG_ALIASES = {
    "shuffle": 0,
    "hasbasicpokemon": 1,
    "turnstart": 2,
    "turnend": 3,
    "draw": 4,
    "drawreverse": 5,
    "movecard": 6,
    "movecardreverse": 7,
    "switch": 8,
    "change": 9,
    "play": 10,
    "attach": 11,
    "evolve": 12,
    "devolve": 13,
    "moveattached": 14,
    "attack": 15,
    "hpchange": 16,
    "poisoned": 17,
    "burned": 18,
    "asleep": 19,
    "paralyzed": 20,
    "confused": 21,
    "coin": 22,
    "result": 23,
}


def _enum_value(value: Any, aliases: dict[str, int], default: int = -1) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+-").isdigit():
            return int(text)
        return aliases.get(text.replace("_", "").replace("-", "").casefold(), default)
    return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_observation(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise GameError("리플레이 장면이 JSON 객체가 아닙니다.")
    frame = copy.deepcopy(raw)
    select = frame.get("select")
    if isinstance(select, dict):
        select["context"] = _enum_value(select.get("context"), CONTEXT_ALIASES, 0)
        options = select.get("option")
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict):
                    option["type"] = _enum_value(option.get("type"), OPTION_ALIASES, -1)
    logs = frame.get("logs")
    if isinstance(logs, list):
        for log in logs:
            if isinstance(log, dict):
                log["type"] = _enum_value(log.get("type"), LOG_ALIASES, -1)
    current = frame.get("current")
    if not isinstance(current, dict):
        raise GameError("리플레이 장면에 current 상태가 없습니다.")
    players = current.get("players")
    if not isinstance(players, list) or len(players) < 2:
        raise GameError("리플레이 장면에 두 플레이어 상태가 없습니다.")
    return frame


def _decode_json(data: bytes, label: str) -> Any:
    if len(data) > REPLAY_JSON_LIMIT:
        raise GameError(f"{label}: JSON 파일이 너무 큽니다.")
    try:
        return json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise GameError(f"{label}: UTF-8 JSON 파일이 아닙니다.") from exc
    except json.JSONDecodeError as exc:
        raise GameError(f"{label}: JSON 구문 오류 · line {exc.lineno}, column {exc.colno}") from exc


def _official_steps_frames(payload: dict[str, Any]) -> list[Any] | None:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, list):
            continue
        for state in step:
            if isinstance(state, dict) and isinstance(state.get("visualize"), list):
                return state["visualize"]

    frames: list[Any] = []
    for step in steps:
        if not isinstance(step, list):
            continue
        observations = [
            state.get("observation")
            for state in step
            if isinstance(state, dict) and isinstance(state.get("observation"), dict)
        ]
        selected = next(
            (obs for obs in observations if isinstance(obs.get("select"), dict)),
            observations[0] if observations else None,
        )
        if isinstance(selected, dict) and isinstance(selected.get("current"), dict):
            frame = copy.deepcopy(selected)
            frame.pop("step", None)
            frame.pop("remainingOverageTime", None)
            frame.pop("search_begin_input", None)
            frames.append(frame)
    return frames or None


def _metadata_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return copy.deepcopy(metadata)
    info = payload.get("info")
    result: dict[str, Any] = {}
    if isinstance(info, dict):
        names = info.get("TeamNames")
        if not isinstance(names, list) or len(names) < 2:
            agents = info.get("Agents")
            if isinstance(agents, list) and len(agents) >= 2:
                names = [
                    str(agent.get("Name") or f"Player {index}")
                    if isinstance(agent, dict)
                    else f"Player {index}"
                    for index, agent in enumerate(agents[:2])
                ]
        if isinstance(names, list) and len(names) >= 2:
            result["player_names"] = [str(names[0]), str(names[1])]
            result["players"] = [{"name": str(names[0])}, {"name": str(names[1])}]
        if info.get("EpisodeId") is not None:
            result["episode_id"] = info.get("EpisodeId")
    rewards = payload.get("rewards")
    if isinstance(rewards, list) and len(rewards) >= 2:
        result["rewards"] = rewards[:2]
        if rewards[0] > rewards[1]:
            result["result"] = 0
        elif rewards[1] > rewards[0]:
            result["result"] = 1
        else:
            result["result"] = 2
    return result


def _list_from_payload(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("frames", "replay", "observations", "replay_visualize"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return _official_steps_frames(payload)
    return None


def _parse_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GameError(f"{label}: UTF-8 JSONL 파일이 아닙니다.") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GameError(f"{label}: {line_number}번째 줄 JSON 오류") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


@dataclass
class LoadedReplay:
    filename: str
    frames: list[dict[str, Any]]
    transitions: list[dict[str, Any]]
    metadata: dict[str, Any]
    source_kind: str


class ReplayManager:
    def __init__(self, game_manager: GameManager) -> None:
        self.lock = threading.RLock()
        self.game_manager = game_manager
        self.loaded: LoadedReplay | None = None

    def close(self) -> None:
        with self.lock:
            self.loaded = None

    def load(self, data: bytes, filename: str) -> dict[str, Any]:
        with self.lock:
            if not data:
                raise GameError("리플레이 파일이 비어 있습니다.")
            if len(data) > REPLAY_UPLOAD_LIMIT:
                raise GameError(
                    f"리플레이 파일은 {REPLAY_UPLOAD_LIMIT // (1024 * 1024)}MB 이하만 불러올 수 있습니다."
                )
            safe_name = Path(filename or "replay").name
            suffix = Path(safe_name).suffix.lower()
            if suffix == ".zip" or data[:4] == b"PK\x03\x04":
                loaded = self._load_zip(data, safe_name)
            elif suffix in {".json", ".jsonl"} or data.lstrip()[:1] in {b"[", b"{"}:
                loaded = self._load_plain(data, safe_name, suffix)
            else:
                raise GameError("지원 형식은 CABT 기록 ZIP, 공식 에피소드 JSON, replay_visualize.json, transitions.jsonl입니다.")
            if not loaded.frames:
                raise GameError("재생 가능한 장면이 없습니다.")
            if len(loaded.frames) > MAX_REPLAY_FRAMES:
                raise GameError(f"리플레이 장면 수가 제한({MAX_REPLAY_FRAMES})을 초과했습니다.")
            self.game_manager.card_ids()
            self.loaded = loaded
            default_seat = _int(loaded.metadata.get("human_seat"), 0)
            if default_seat not in (0, 1):
                default_seat = 0
            return self.state(0, default_seat)

    def _load_zip(self, data: bytes, filename: str) -> LoadedReplay:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise GameError("손상되었거나 올바르지 않은 ZIP 파일입니다.") from exc
        with archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if not infos:
                raise GameError("ZIP 안에 파일이 없습니다.")
            if any(info.flag_bits & 0x1 for info in infos):
                raise GameError("암호화된 ZIP은 불러올 수 없습니다.")
            total_size = sum(info.file_size for info in infos)
            if total_size > REPLAY_UNCOMPRESSED_LIMIT:
                raise GameError("ZIP 압축 해제 크기가 안전 제한을 초과했습니다.")

            by_basename: dict[str, list[zipfile.ZipInfo]] = {}
            for info in infos:
                by_basename.setdefault(Path(info.filename).name.casefold(), []).append(info)

            def read_named(name: str) -> bytes | None:
                matches = by_basename.get(name.casefold()) or []
                if not matches:
                    return None
                info = sorted(matches, key=lambda item: (item.filename.count("/"), item.filename))[0]
                return archive.read(info)

            metadata: dict[str, Any] = {}
            metadata_bytes = read_named("metadata.json")
            if metadata_bytes:
                parsed = _decode_json(metadata_bytes, "metadata.json")
                if isinstance(parsed, dict):
                    metadata = parsed

            transitions: list[dict[str, Any]] = []
            transitions_bytes = read_named("transitions.jsonl")
            if transitions_bytes:
                transitions = _parse_jsonl(transitions_bytes, "transitions.jsonl")

            frames: list[Any] | None = None
            replay_bytes = read_named("replay_visualize.json")
            if replay_bytes:
                frames = _list_from_payload(_decode_json(replay_bytes, "replay_visualize.json"))

            if frames is None:
                for preferred in ("replay.json", "frames.json", "observations.json"):
                    candidate = read_named(preferred)
                    if candidate:
                        frames = _list_from_payload(_decode_json(candidate, preferred))
                        if frames is not None:
                            break

            if frames is None and transitions:
                frames = [row.get("observation") for row in transitions if isinstance(row.get("observation"), dict)]

            if frames is None:
                json_infos = [info for info in infos if info.filename.lower().endswith(".json")]
                for info in sorted(json_infos, key=lambda item: item.file_size):
                    parsed = _decode_json(archive.read(info), info.filename)
                    frames = _list_from_payload(parsed)
                    if frames is not None:
                        break

            if frames is None:
                raise GameError("ZIP 안에서 replay_visualize.json 또는 재생 가능한 JSON을 찾지 못했습니다.")

            normalized = [_normalize_observation(frame) for frame in frames]
            return LoadedReplay(filename, normalized, transitions, metadata, "zip")

    def _load_plain(self, data: bytes, filename: str, suffix: str) -> LoadedReplay:
        if suffix == ".jsonl":
            transitions = _parse_jsonl(data, filename)
            frames = [row.get("observation") for row in transitions if isinstance(row.get("observation"), dict)]
            if not frames:
                raise GameError("JSONL에 observation 장면이 없습니다.")
            return LoadedReplay(
                filename,
                [_normalize_observation(frame) for frame in frames],
                transitions,
                {},
                "jsonl",
            )

        payload = _decode_json(data, filename)
        rows = _list_from_payload(payload)
        if rows is None:
            raise GameError("JSON에서 리플레이 장면 배열을 찾지 못했습니다.")
        if rows and isinstance(rows[0], dict) and isinstance(rows[0].get("observation"), dict):
            transitions = [row for row in rows if isinstance(row, dict)]
            frames = [row["observation"] for row in transitions]
            return LoadedReplay(
                filename,
                [_normalize_observation(frame) for frame in frames],
                transitions,
                _metadata_from_payload(payload),
                "json-transitions",
            )
        metadata = _metadata_from_payload(payload)
        source_kind = "official-json" if isinstance(payload, dict) and isinstance(payload.get("steps"), list) else "json"
        return LoadedReplay(
            filename,
            [_normalize_observation(frame) for frame in rows],
            [],
            metadata,
            source_kind,
        )

    def _player_names(self) -> list[str]:
        if self.loaded is None:
            return ["Player 0", "Player 1"]
        metadata = self.loaded.metadata
        names = metadata.get("player_names")
        if isinstance(names, list) and len(names) >= 2:
            return [str(names[0] or "Player 0"), str(names[1] or "Player 1")]
        players = metadata.get("players")
        if isinstance(players, list) and len(players) >= 2:
            return [
                str(players[0].get("name") or "Player 0") if isinstance(players[0], dict) else "Player 0",
                str(players[1].get("name") or "Player 1") if isinstance(players[1], dict) else "Player 1",
            ]
        agent_name = metadata.get("agent_name")
        human_seat = _int(metadata.get("human_seat"), -1)
        if agent_name and human_seat in (0, 1):
            result = ["Player 0", "Player 1"]
            result[human_seat] = "Human"
            result[1 - human_seat] = str(agent_name)
            return result
        return ["Player 0", "Player 1"]

    def _serialize_player(self, player: dict[str, Any]) -> dict[str, Any]:
        hand_raw = player.get("hand")
        reveal_hand = isinstance(hand_raw, list)
        hand = hand_raw if reveal_hand else []
        return {
            "active": [self.game_manager._serialize_pokemon(p) for p in player.get("active") or [] if p],
            "bench": [self.game_manager._serialize_pokemon(p) for p in player.get("bench") or [] if p],
            "bench_max": _int(player.get("benchMax"), 5),
            "deck_count": _int(player.get("deckCount"), len(player.get("deck") or [])),
            "discard": [
                self.game_manager._serialize_card(card) for card in (player.get("discard") or [])[-16:]
            ],
            "discard_count": len(player.get("discard") or []),
            "prize_count": len(player.get("prize") or []),
            "hand_count": _int(player.get("handCount"), len(hand)),
            "hand": [self.game_manager._serialize_card(card) for card in hand] if reveal_hand else None,
            "conditions": {
                "poisoned": bool(player.get("poisoned", False)),
                "burned": bool(player.get("burned", False)),
                "asleep": bool(player.get("asleep", False)),
                "paralyzed": bool(player.get("paralyzed", False)),
                "confused": bool(player.get("confused", False)),
            },
        }

    def _enrich_log(self, log: dict[str, Any], names: list[str]) -> dict[str, Any]:
        catalog = self.game_manager.catalog
        event_type = _int(log.get("type"), -1)
        player_index = _int(log.get("playerIndex"), -1)
        who = names[player_index] if player_index in (0, 1) else "Game"
        if catalog is None:
            return {"type": event_type, "text": f"{who}: event {event_type}", "raw": log}
        text = f"{who}: event {event_type}"
        if event_type == 0:
            text = f"{who} · 덱 셔플"
        elif event_type == 1:
            text = f"{who} · 기본 포켓몬 {'확인' if log.get('hasBasicPokemon') else '없음'}"
        elif event_type == 2:
            text = f"{who} 턴 시작"
        elif event_type == 3:
            text = f"{who} 턴 종료"
        elif event_type == 10:
            text = f"{who} · {catalog.card_name(log.get('cardId'))} 사용"
        elif event_type == 11:
            text = f"{who} · {catalog.card_name(log.get('cardId'))}를 {catalog.card_name(log.get('cardIdTarget'))}에게 부착"
        elif event_type == 12:
            text = f"{who} · {catalog.card_name(log.get('cardIdTarget'))} → {catalog.card_name(log.get('cardId'))} 진화"
        elif event_type == 15:
            text = f"{who} · {catalog.attack_name(log.get('attackId'))} 사용"
        elif event_type == 16:
            text = f"{catalog.card_name(log.get('cardId'))} HP {_int(log.get('value')):+d}"
        elif event_type == 8:
            text = f"{who} · {catalog.card_name(log.get('cardIdBench'))}를 배틀필드로 교체"
        elif event_type == 22:
            text = f"{who} · 동전 {'앞면' if log.get('head') else '뒷면'}"
        elif event_type == 23:
            text = "게임 종료"
        elif event_type == 4:
            text = f"{who} · {catalog.card_name(log.get('cardId'))} 드로우"
        elif event_type == 5:
            text = f"{who} · 카드 1장 드로우"
        elif event_type == 6 and log.get("cardId") is not None:
            text = (
                f"{who} · {catalog.card_name(log.get('cardId'))} "
                f"{AREA_NAMES.get(_int(log.get('fromArea'), -1), '?')} → "
                f"{AREA_NAMES.get(_int(log.get('toArea'), -1), '?')}"
            )
        elif event_type == 7:
            text = f"{who} · 비공개 카드 이동"
        return {"type": event_type, "text": text, "raw": log}

    def _transition_for(self, index: int) -> dict[str, Any] | None:
        if self.loaded is None or index < 0 or index >= len(self.loaded.transitions):
            return None
        row = self.loaded.transitions[index]
        return row if isinstance(row, dict) else None

    def _action_summary(self, transition_index: int, view_seat: int, names: list[str]) -> dict[str, Any]:
        assert self.loaded is not None
        transition = self._transition_for(transition_index)
        before = self.loaded.frames[min(transition_index, len(self.loaded.frames) - 1)]
        select = before.get("select") or {}
        actor = _int((before.get("current") or {}).get("yourIndex"), -1)
        indices: list[int] = []
        think_ms = 0
        timestamp = ""
        if transition:
            actor = _int(transition.get("actor"), actor)
            raw_indices = transition.get("action_indices")
            if isinstance(raw_indices, list):
                indices = [_int(value, -1) for value in raw_indices if _int(value, -1) >= 0]
            think_ms = _int(transition.get("think_ms"), 0)
            timestamp = str(transition.get("timestamp") or "")
        options = select.get("option") or []
        option_rows = []
        for option_index in indices:
            if 0 <= option_index < len(options):
                option_rows.append(
                    self.game_manager._enrich_option(before, select, options[option_index], option_index)
                )
        labels = [row["label"] for row in option_rows]
        details = [row["detail"] for row in option_rows if row.get("detail")]
        context = _int(select.get("context"), 0)
        prompt = CONTEXT_KO.get(context, "선택 처리")
        if labels:
            text = " · ".join(labels)
        elif transition:
            text = prompt
        else:
            text = f"{prompt} 완료"
        after_index = min(transition_index + 1, len(self.loaded.frames) - 1)
        events = [
            self._enrich_log(log, names)
            for log in (self.loaded.frames[after_index].get("logs") or [])
            if isinstance(log, dict)
        ]
        return {
            "sequence": transition_index + 1,
            "frame_index": after_index,
            "source": "human" if actor == view_seat else "ai",
            "actor": actor,
            "actor_name": names[actor] if actor in (0, 1) else "Game",
            "turn": _int((before.get("current") or {}).get("turn"), 0),
            "context": CONTEXT_NAMES.get(context, "SELECTION"),
            "prompt": prompt,
            "indices": indices,
            "labels": labels,
            "details": details,
            "text": text,
            "think_ms": think_ms,
            "events": events[-8:],
            "timestamp": timestamp,
        }

    def state(self, index: int, view_seat: int = 0) -> dict[str, Any]:
        with self.lock:
            if self.loaded is None:
                raise GameError("불러온 리플레이가 없습니다.")
            if view_seat not in (0, 1):
                raise GameError("관전 좌석은 Player 0 또는 Player 1이어야 합니다.")
            total = len(self.loaded.frames)
            index = max(0, min(_int(index), total - 1))
            frame = self.loaded.frames[index]
            current = frame.get("current") or {}
            players = current.get("players") or [{}, {}]
            names = self._player_names()
            result = _int(current.get("result"), -1)
            actor = _int(current.get("yourIndex"), -1)
            top_seat = 1 - view_seat

            result_text = None
            if result != -1:
                result_text = "무승부" if result == 2 else f"{names[result]} 승리" if result in (0, 1) else "게임 종료"

            select = frame.get("select") if result == -1 else None
            enriched_options = []
            if isinstance(select, dict):
                for option_index, option in enumerate(select.get("option") or []):
                    if isinstance(option, dict):
                        enriched_options.append(
                            self.game_manager._enrich_option(frame, select, option, option_index)
                        )

            upcoming = self._transition_for(index)
            selected_indices = []
            if upcoming and isinstance(upcoming.get("action_indices"), list):
                selected_indices = [
                    _int(value, -1) for value in upcoming["action_indices"] if _int(value, -1) >= 0
                ]

            history_length = min(index, max(0, len(self.loaded.frames) - 1))
            action_history = [
                self._action_summary(action_index, view_seat, names)
                for action_index in range(history_length)
            ]
            last_action = action_history[-1] if action_history else None
            context = _int(select.get("context"), 0) if isinstance(select, dict) else 0

            return {
                "active": True,
                "mode": "replay",
                "game_id": str(self.loaded.metadata.get("game_id") or self.loaded.filename),
                "agent": {"name": names[top_seat], "id": "replay-player"},
                "human_deck_label": str(self.loaded.metadata.get("human_deck_label") or "Replay"),
                "human_seat": view_seat,
                "ai_seat": top_seat,
                "player_names": names,
                "human_turn": False,
                "ai_pending": False,
                "actor": actor,
                "turn": _int(current.get("turn"), 0),
                "turn_action_count": _int(current.get("turnActionCount"), 0),
                "first_player": _int(current.get("firstPlayer"), -1),
                "result": result,
                "result_text": result_text,
                "decision_count": index,
                "human": self._serialize_player(players[view_seat]),
                "ai": self._serialize_player(players[top_seat]),
                "stadium": [
                    self.game_manager._serialize_card(card) for card in current.get("stadium") or []
                ],
                "looking": [
                    self.game_manager._serialize_card(card) if card else None
                    for card in current.get("looking") or []
                ],
                "flags": {
                    "supporter_played": bool(current.get("supporterPlayed", False)),
                    "stadium_played": bool(current.get("stadiumPlayed", False)),
                    "energy_attached": bool(current.get("energyAttached", False)),
                    "retreated": bool(current.get("retreated", False)),
                },
                "selection": None
                if not isinstance(select, dict)
                else {
                    "type": select.get("type"),
                    "context": context,
                    "context_name": CONTEXT_NAMES.get(context, str(context)),
                    "prompt": CONTEXT_KO.get(context, "이 장면의 선택지"),
                    "min_count": _int(select.get("minCount"), 0),
                    "max_count": _int(select.get("maxCount"), 0),
                    "remain_damage_counter": _int(select.get("remainDamageCounter"), 0),
                    "remain_energy_cost": _int(select.get("remainEnergyCost"), 0),
                    "options": enriched_options,
                    "selected_indices": selected_indices,
                },
                "logs": [
                    self._enrich_log(log, names)
                    for log in (frame.get("logs") or [])[-24:]
                    if isinstance(log, dict)
                ],
                "last_action": last_action,
                "action_history": action_history[-160:],
                "last_error": None,
                "record_available": False,
                "replay": {
                    "filename": self.loaded.filename,
                    "source_kind": self.loaded.source_kind,
                    "index": index,
                    "total": total,
                    "has_transitions": bool(self.loaded.transitions),
                    "view_seat": view_seat,
                    "next_selected_indices": selected_indices,
                    "metadata": {
                        "created_at": self.loaded.metadata.get("created_at"),
                        "updated_at": self.loaded.metadata.get("updated_at"),
                        "duration_seconds": self.loaded.metadata.get("duration_seconds"),
                        "status": self.loaded.metadata.get("status"),
                    },
                },
            }
