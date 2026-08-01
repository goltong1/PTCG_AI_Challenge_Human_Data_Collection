from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_loader import AgentLoadError, AgentRepository, AgentRuntime, read_deck


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("CABT_DATA_DIR")
    or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    or str(ROOT)
).expanduser().resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

AREA_NAMES = {
    1: "덱",
    2: "패",
    3: "트래시",
    4: "배틀필드",
    5: "벤치",
    6: "프라이즈",
    7: "스타디움",
    8: "에너지",
    9: "도구",
    10: "진화 전",
    11: "플레이어",
    12: "확인 중인 카드",
}

OPTION_NAMES = {
    0: "수 선택",
    1: "예",
    2: "아니오",
    3: "카드 선택",
    4: "도구 선택",
    5: "에너지 카드 선택",
    6: "에너지 선택",
    7: "카드 사용",
    8: "에너지 부착",
    9: "진화",
    10: "특성 사용",
    11: "버리기",
    12: "후퇴",
    13: "기술 사용",
    14: "턴 종료",
    15: "효과 순서",
    16: "특수 상태",
}

CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE_POKEMON",
    2: "SETUP_BENCH_POKEMON",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    9: "TO_DECK",
    10: "TO_DECK_BOTTOM",
    11: "TO_PRIZE",
    12: "NOT_MOVE",
    13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY",
    15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER",
    17: "HEAL",
    18: "EVOLVES_FROM",
    19: "EVOLVES_TO",
    20: "DEVOLVE",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    23: "DETACH_FROM",
    24: "LOOK",
    25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD",
    27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY",
    33: "SWITCH_ENERGY",
    34: "SKILL_ORDER",
    35: "ATTACK",
    36: "DISABLE_ATTACK",
    37: "EVOLVE",
    38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST",
    42: "MULLIGAN",
    43: "ACTIVATE",
    44: "FIRST_EFFECT",
    45: "MORE_DEVOLVE",
    46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}

CONTEXT_KO = {
    0: "행동을 선택하세요",
    1: "배틀필드에 놓을 포켓몬을 선택하세요",
    2: "벤치에 놓을 포켓몬을 선택하세요",
    3: "교체할 포켓몬을 선택하세요",
    4: "배틀필드에 낼 포켓몬을 선택하세요",
    5: "벤치에 낼 포켓몬을 선택하세요",
    6: "필드에 낼 카드를 선택하세요",
    7: "패에 넣을 카드를 선택하세요",
    8: "버릴 카드를 선택하세요",
    9: "덱으로 되돌릴 카드를 선택하세요",
    10: "덱 아래로 되돌릴 카드를 선택하세요",
    11: "프라이즈로 보낼 카드를 선택하세요",
    12: "그대로 둘 카드를 선택하세요",
    13: "데미지 카운터를 놓을 대상을 선택하세요",
    14: "데미지 카운터를 배분하세요",
    15: "데미지를 줄 대상을 선택하세요",
    16: "데미지 카운터를 제거할 대상을 선택하세요",
    17: "회복할 대상을 선택하세요",
    18: "진화할 포켓몬을 선택하세요",
    19: "진화 카드를 선택하세요",
    20: "퇴화할 포켓몬을 선택하세요",
    21: "부착할 카드를 선택하세요",
    22: "부착 대상을 선택하세요",
    23: "분리할 카드를 선택하세요",
    24: "확인할 카드를 선택하세요",
    25: "효과 대상을 선택하세요",
    26: "버릴 에너지 카드를 선택하세요",
    27: "버릴 도구를 선택하세요",
    28: "교체할 에너지 카드를 선택하세요",
    29: "버릴 카드를 선택하세요",
    30: "버릴 에너지를 선택하세요",
    31: "패로 되돌릴 에너지를 선택하세요",
    32: "덱으로 되돌릴 에너지를 선택하세요",
    33: "옮길 에너지를 선택하세요",
    34: "효과 처리 순서를 선택하세요",
    35: "사용할 기술을 선택하세요",
    36: "사용할 수 없게 할 기술을 선택하세요",
    37: "진화 조합을 선택하세요",
    38: "드로우할 장수를 선택하세요",
    39: "놓을 데미지 카운터 수를 선택하세요",
    40: "제거할 데미지 카운터 수를 선택하세요",
    41: "선공을 선택하시겠습니까?",
    42: "다시 뽑으시겠습니까?",
    43: "효과를 사용하시겠습니까?",
    44: "먼저 처리할 효과를 선택하시겠습니까?",
    45: "더 퇴화시키겠습니까?",
    46: "동전의 앞면을 선택하시겠습니까?",
    47: "부여할 특수 상태를 선택하세요",
    48: "회복할 특수 상태를 선택하세요",
}

ENERGY_NAMES = {
    0: "Colorless",
    1: "Grass",
    2: "Fire",
    3: "Water",
    4: "Lightning",
    5: "Psychic",
    6: "Fighting",
    7: "Darkness",
    8: "Metal",
    9: "Dragon",
    10: "Rainbow",
    11: "Team Rocket",
}

SPECIAL_NAMES = {0: "독", 1: "화상", 2: "잠듦", 3: "마비", 4: "혼란"}


class GameError(RuntimeError):
    pass


def parse_uploaded_deck(text: str, filename: str) -> list[int]:
    values = [
        int(token)
        for token in text.replace(",", " ").split()
        if token.strip().lstrip("+-").isdigit()
    ]
    if len(values) != 60:
        raise GameError(f"{filename}: 카드 ID가 정확히 60개 필요합니다. 현재 {len(values)}개입니다.")
    return values


class CardCatalog:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.api = runtime.cg_api
        self.cards = {int(card.cardId): card for card in self.api.all_card_data()}
        self.attacks = {int(attack.attackId): attack for attack in self.api.all_attack()}

    def card_name(self, card_id: int | None) -> str:
        if card_id is None:
            return "알 수 없는 카드"
        card = self.cards.get(int(card_id))
        return card.name if card else f"Card {card_id}"

    def attack_name(self, attack_id: int | None) -> str:
        if attack_id is None:
            return "알 수 없는 기술"
        attack = self.attacks.get(int(attack_id))
        return attack.name if attack else f"Attack {attack_id}"

    def card_summary(self, card_id: int) -> dict[str, Any]:
        data = self.cards.get(int(card_id))
        if data is None:
            return {
                "id": int(card_id),
                "name": f"Card {card_id}",
                "type": "UNKNOWN",
                "hp": 0,
                "energy_type": "COLORLESS",
                "text": "",
                "attacks": [],
            }
        attacks = []
        for attack_id in data.attacks:
            attack = self.attacks.get(int(attack_id))
            if attack:
                attacks.append(
                    {
                        "id": int(attack.attackId),
                        "name": attack.name,
                        "text": attack.text,
                        "damage": int(attack.damage),
                        "energies": [ENERGY_NAMES.get(int(e), str(e)) for e in attack.energies],
                    }
                )
        skill_text = "\n".join(f"{skill.name}: {skill.text}" for skill in data.skills)
        try:
            card_type = self.api.CardType(int(data.cardType)).name
        except Exception:
            card_type = str(data.cardType)
        return {
            "id": int(card_id),
            "name": data.name,
            "type": card_type,
            "hp": int(data.hp),
            "energy_type": ENERGY_NAMES.get(int(data.energyType), str(data.energyType)),
            "retreat_cost": int(data.retreatCost),
            "ex": bool(data.ex),
            "mega_ex": bool(data.megaEx),
            "basic": bool(data.basic),
            "stage1": bool(data.stage1),
            "stage2": bool(data.stage2),
            "text": skill_text,
            "attacks": attacks,
        }


class GameManager:
    def __init__(self, agents: AgentRepository) -> None:
        self.lock = threading.RLock()
        self.agents = agents
        self.runtime: AgentRuntime | None = None
        self.catalog: CardCatalog | None = None
        self.observation: dict[str, Any] | None = None
        self.human_seat = 0
        self.ai_seat = 1
        self.ai_deck: list[int] = []
        self.human_deck: list[int] = []
        self.record_dir: Path | None = None
        self.game_id: str | None = None
        self.decision_count = 0
        self.started_at: float | None = None
        self.finished = False
        self.last_error: str | None = None
        self.last_action: dict[str, Any] | None = None
        self.action_history: list[dict[str, Any]] = []
        self.human_deck_label = ""
        self.official_observations: list[dict[str, Any]] = []
        self.official_actions: list[list[int]] = []
        self.official_think_ms: list[int] = []
        self.game_mode = "ai"
        self.player_names = ["Human", "AI"]
        self.player_deck_labels = ["", ""]

    def presets(self) -> list[dict[str, str]]:
        labels = {
            "alakazam_mirror": "후딘 미러",
            "initial_alakazam": "초기 후딘",
            "lucario": "메가 루카리오 ex",
            "dragapult": "드래펄트 ex",
            "crustle": "암팰리스",
            "rocket": "로켓단·프리저",
            "marnie_grimmsnarl": "마리의 오롱털",
        }
        return [
            {"id": path.stem, "label": labels.get(path.stem, path.stem)}
            for path in sorted((ROOT / "decks").glob("*.csv"))
        ]

    def card_ids(self) -> set[int]:
        with self.lock:
            if self.catalog is None:
                default = self.agents.list()
                if not default:
                    return set()
                runtime = self.agents.load(default[0].id)
                self.runtime = runtime
                self.catalog = CardCatalog(runtime)
            return set(self.catalog.cards)

    def card_types(self) -> list[dict[str, Any]]:
        with self.lock:
            self.card_ids()
            if self.catalog is None:
                return []
            counts = Counter(
                self.catalog.card_summary(card_id)["type"] for card_id in self.catalog.cards
            )
            order = [
                "POKEMON",
                "ITEM",
                "TOOL",
                "SUPPORTER",
                "STADIUM",
                "BASIC_ENERGY",
                "SPECIAL_ENERGY",
            ]
            return [
                {"id": card_type, "count": counts[card_type]}
                for card_type in order
                if counts[card_type]
            ]

    def search_cards(
        self,
        query: str = "",
        card_type: str = "",
        offset: int = 0,
        limit: int = 48,
    ) -> dict[str, Any]:
        with self.lock:
            self.card_ids()
            if self.catalog is None:
                raise GameError("카드 카탈로그가 준비되지 않았습니다.")
            offset = max(0, int(offset))
            limit = max(1, min(96, int(limit)))
            wanted_type = card_type.strip().upper()
            tokens = query.casefold().split()
            rows = []
            for card_id in self.catalog.cards:
                summary = self.catalog.card_summary(card_id)
                if wanted_type and wanted_type != "ALL" and summary["type"] != wanted_type:
                    continue
                haystack = " ".join(
                    [
                        str(summary["id"]),
                        summary["name"],
                        summary["text"],
                        " ".join(attack["name"] for attack in summary["attacks"]),
                    ]
                ).casefold()
                if tokens and not all(token in haystack for token in tokens):
                    continue
                summary["image_url"] = f"/api/card-image/{card_id}"
                rows.append(summary)
            rows.sort(key=lambda card: (card["name"].casefold(), card["id"]))
            total = len(rows)
            return {
                "cards": rows[offset : offset + limit],
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total,
            }

    def validate_deck_cards(self, cards: list[Any], exact: bool = False) -> list[int]:
        if not isinstance(cards, list):
            raise GameError("cards 배열이 필요합니다.")
        if any(isinstance(card_id, bool) or not isinstance(card_id, int) for card_id in cards):
            raise GameError("카드 ID는 정수여야 합니다.")
        if exact and len(cards) != 60:
            raise GameError(f"대전 덱은 정확히 60장이어야 합니다. 현재 {len(cards)}장입니다.")
        if len(cards) > 60:
            raise GameError(f"덱은 최대 60장까지 저장할 수 있습니다. 현재 {len(cards)}장입니다.")

        with self.lock:
            self.card_ids()
            if self.catalog is None:
                raise GameError("카드 카탈로그가 준비되지 않았습니다.")
            unknown = sorted(set(cards) - set(self.catalog.cards))
            if unknown:
                preview = ", ".join(str(card_id) for card_id in unknown[:8])
                raise GameError(f"카탈로그에 없는 카드 ID가 있습니다: {preview}")

            name_counts: Counter[str] = Counter()
            display_names: dict[str, str] = {}
            for card_id in cards:
                summary = self.catalog.card_summary(card_id)
                if summary["type"] == "BASIC_ENERGY":
                    continue
                key = summary["name"].casefold()
                name_counts[key] += 1
                display_names[key] = summary["name"]
            over_limit = [
                f"{display_names[key]} {count}장"
                for key, count in name_counts.items()
                if count > 4
            ]
            if over_limit:
                raise GameError(
                    "기본 에너지를 제외한 같은 이름의 카드는 최대 4장입니다: "
                    + ", ".join(over_limit[:6])
                )
        return list(cards)

    def deck_details(self, cards: list[int]) -> list[dict[str, Any]]:
        with self.lock:
            self.card_ids()
            if self.catalog is None:
                raise GameError("카드 카탈로그가 준비되지 않았습니다.")
            counts = Counter(cards)
            rows = []
            for card_id, quantity in counts.items():
                summary = self.catalog.card_summary(card_id)
                summary["image_url"] = f"/api/card-image/{card_id}"
                summary["quantity"] = quantity
                rows.append(summary)
            type_order = {
                "POKEMON": 0,
                "ITEM": 1,
                "TOOL": 2,
                "SUPPORTER": 3,
                "STADIUM": 4,
                "BASIC_ENERGY": 5,
                "SPECIAL_ENERGY": 6,
            }
            rows.sort(
                key=lambda card: (
                    type_order.get(card["type"], 99),
                    card["name"].casefold(),
                    card["id"],
                )
            )
            return rows

    def _finish_native(self) -> None:
        if self.observation is not None and self.runtime is not None:
            try:
                self.runtime.battle_finish()
            except Exception:
                pass
        self.observation = None

    def close(self) -> None:
        with self.lock:
            self._finalize_record()
            self._finish_native()

    def start(
        self,
        human_seat: int,
        human_deck: list[int],
        deck_label: str,
        agent_id: str,
    ) -> dict[str, Any]:
        with self.lock:
            self._finalize_record()
            self._finish_native()
            if human_seat not in (0, 1):
                raise GameError("사람 좌석은 Player 0 또는 Player 1이어야 합니다.")
            if len(human_deck) != 60:
                raise GameError("사람 덱은 정확히 60장이어야 합니다.")

            try:
                runtime = self.agents.load(agent_id)
                ai_deck = runtime.deck()
            except AgentLoadError as exc:
                raise GameError(str(exc)) from exc

            self.runtime = runtime
            self.catalog = CardCatalog(runtime)
            self.game_mode = "ai"
            self.human_seat = human_seat
            self.ai_seat = 1 - human_seat
            self.ai_deck = ai_deck
            self.human_deck = list(human_deck)
            self.human_deck_label = deck_label
            decks: list[list[int] | None] = [None, None]
            decks[self.human_seat] = self.human_deck
            decks[self.ai_seat] = self.ai_deck

            try:
                observation, start_data = runtime.battle_start(decks[0], decks[1])
            except Exception as exc:
                raise GameError(f"BattleStart 실행 실패 · {type(exc).__name__}: {exc}") from exc
            if observation is None or int(start_data.errorType) != 0:
                raise GameError(
                    f"BattleStart 실패: errorType={start_data.errorType}, errorPlayer={start_data.errorPlayer}"
                )

            self.observation = observation
            self.decision_count = 0
            self.started_at = time.time()
            self.finished = False
            self.last_error = None
            self.last_action = None
            self.action_history = []
            self.official_observations = [copy.deepcopy(observation)]
            self.official_actions = []
            self.official_think_ms = []
            self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            self.record_dir = DATA_ROOT / "records" / self.game_id
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._write_deck(self.record_dir / f"deck_player{self.human_seat}.csv", self.human_deck)
            self._write_deck(self.record_dir / f"deck_player{self.ai_seat}.csv", self.ai_deck)
            names = ["", ""]
            names[self.human_seat] = "Human"
            names[self.ai_seat] = runtime.info.name
            self.player_names = names
            self.player_deck_labels = ["", ""]
            self.player_deck_labels[self.human_seat] = deck_label
            self.player_deck_labels[self.ai_seat] = runtime.info.name
            metadata = {
                "game_id": self.game_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "human_seat": self.human_seat,
                "ai_seat": self.ai_seat,
                "human_deck_label": deck_label,
                "agent_id": runtime.info.id,
                "agent_name": runtime.info.name,
                "player_names": names,
                "players": [{"name": names[0]}, {"name": names[1]}],
                "status": "active",
                "result": -1,
            }
            self._write_json(self.record_dir / "metadata.json", metadata)
            self._save_replay()
            return self.public_state()

    def start_pvp(
        self,
        player0_deck: list[int],
        player1_deck: list[int],
        deck_labels: list[str],
        player_names: list[str],
        engine_agent_id: str = "",
    ) -> dict[str, Any]:
        """Start one authoritative human-vs-human match.

        A bundled agent runtime is used only as a CABT native-engine provider;
        neither player's decisions are delegated to its ``agent()`` function.
        """
        with self.lock:
            self._finalize_record()
            self._finish_native()
            if len(player0_deck) != 60 or len(player1_deck) != 60:
                raise GameError("온라인 대전 덱은 양쪽 모두 정확히 60장이어야 합니다.")
            if not isinstance(player_names, list) or len(player_names) != 2:
                raise GameError("온라인 플레이어 이름 두 개가 필요합니다.")
            available = self.agents.list()
            if not available:
                raise GameError("CABT 게임 엔진을 제공할 기본 AI가 없습니다.")
            runtime_id = engine_agent_id or available[0].id
            try:
                runtime = self.agents.load(runtime_id)
            except AgentLoadError as exc:
                raise GameError(str(exc)) from exc

            clean_names = []
            for seat, raw_name in enumerate(player_names):
                name = " ".join(str(raw_name or "").strip().split())[:32]
                clean_names.append(name or f"Player {seat}")
            labels = [str(value or "온라인 덱")[:80] for value in (deck_labels or [])[:2]]
            while len(labels) < 2:
                labels.append("온라인 덱")

            self.runtime = runtime
            self.catalog = CardCatalog(runtime)
            self.game_mode = "pvp"
            # Keep the legacy fields populated so replay export remains fully
            # compatible: human_deck maps to P0, ai_deck maps to P1.
            self.human_seat = 0
            self.ai_seat = 1
            self.human_deck = list(player0_deck)
            self.ai_deck = list(player1_deck)
            self.human_deck_label = labels[0]
            self.player_deck_labels = labels
            self.player_names = clean_names

            try:
                observation, start_data = runtime.battle_start(self.human_deck, self.ai_deck)
            except Exception as exc:
                raise GameError(f"BattleStart 실행 실패 · {type(exc).__name__}: {exc}") from exc
            if observation is None or int(start_data.errorType) != 0:
                raise GameError(
                    f"BattleStart 실패: errorType={start_data.errorType}, errorPlayer={start_data.errorPlayer}"
                )

            self.observation = observation
            self.decision_count = 0
            self.started_at = time.time()
            self.finished = False
            self.last_error = None
            self.last_action = None
            self.action_history = []
            self.official_observations = [copy.deepcopy(observation)]
            self.official_actions = []
            self.official_think_ms = []
            self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            self.record_dir = DATA_ROOT / "records" / self.game_id
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._write_deck(self.record_dir / "deck_player0.csv", self.human_deck)
            self._write_deck(self.record_dir / "deck_player1.csv", self.ai_deck)
            metadata = {
                "game_id": self.game_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": "pvp",
                "player_names": clean_names,
                "players": [{"name": clean_names[0]}, {"name": clean_names[1]}],
                "deck_labels": labels,
                "engine_agent_id": runtime.info.id,
                "engine_agent_name": runtime.info.name,
                "status": "active",
                "result": -1,
            }
            self._write_json(self.record_dir / "metadata.json", metadata)
            self._save_replay()
            return self.public_state_for(0)

    def act_human(self, indices: list[int]) -> dict[str, Any]:
        with self.lock:
            self._ensure_active()
            current = self.observation.get("current") or {}
            if int(current.get("result", -1)) != -1:
                return self.public_state()
            if int(current.get("yourIndex", -1)) != self.human_seat:
                raise GameError("현재는 AI의 선택 차례입니다.")
            return self._apply_action(indices, "human", think_ms=0)

    def act_player(self, seat: int, indices: list[int]) -> dict[str, Any]:
        with self.lock:
            self._ensure_active()
            if self.game_mode != "pvp":
                raise GameError("현재 게임은 온라인 플레이어 대전이 아닙니다.")
            if seat not in (0, 1):
                raise GameError("플레이어 좌석은 0 또는 1이어야 합니다.")
            current = self.observation.get("current") or {}
            if int(current.get("result", -1)) != -1:
                return self.public_state_for(seat)
            if int(current.get("yourIndex", -1)) != seat:
                raise GameError("현재는 상대 플레이어의 선택 차례입니다.")
            self._apply_action(indices, f"player{seat}", think_ms=0)
            return self.public_state_for(seat)

    def act_ai_once(self) -> dict[str, Any]:
        with self.lock:
            self._ensure_active()
            current = self.observation.get("current") or {}
            if int(current.get("result", -1)) != -1:
                return self.public_state()
            if int(current.get("yourIndex", -1)) != self.ai_seat:
                raise GameError("현재는 사람의 선택 차례입니다.")
            if self.runtime is None:
                raise GameError("AI 런타임이 초기화되지 않았습니다.")
            if self.decision_count >= 2000:
                self.last_error = "한 게임의 선택 횟수가 안전 제한 2,000회를 초과했습니다."
                self._finalize_record(error=self.last_error)
                raise GameError(self.last_error)

            started = time.perf_counter()
            try:
                action = self.runtime.act(self.observation)
            except Exception as exc:
                self.last_error = f"AI 실행 오류 · {type(exc).__name__}: {exc}"
                self._finalize_record(error=self.last_error)
                raise GameError(self.last_error) from exc
            think_ms = int(round((time.perf_counter() - started) * 1000))
            return self._apply_action(action, "ai", think_ms=think_ms)

    def _ensure_active(self) -> None:
        if self.observation is None:
            raise GameError("진행 중인 게임이 없습니다.")

    def _apply_action(self, action: list[int], source: str, think_ms: int) -> dict[str, Any]:
        if self.observation is None or self.runtime is None:
            raise GameError("게임이 초기화되지 않았습니다.")
        self._validate_action(self.observation, action)
        before = copy.deepcopy(self.observation)
        try:
            after = self.runtime.battle_select(action)
        except Exception as exc:
            self.last_error = f"게임 엔진 선택 오류 · {type(exc).__name__}: {exc}"
            self._finalize_record(error=self.last_error)
            raise GameError(self.last_error) from exc

        self._record_transition(before, action, source, after, think_ms)
        self.observation = after
        self.official_actions.append(list(action))
        self.official_think_ms.append(max(0, int(think_ms)))
        self.official_observations.append(copy.deepcopy(after))
        self.decision_count += 1
        self.last_action = self._action_summary(before, action, source, after, think_ms)
        self.action_history.append(self.last_action)
        self._save_replay()
        if int((after.get("current") or {}).get("result", -1)) != -1:
            self.finished = True
            self._finalize_record()
        return self.public_state()

    @staticmethod
    def _validate_action(obs: dict[str, Any], action: list[int]) -> None:
        if not isinstance(action, list) or not all(isinstance(value, int) for value in action):
            raise GameError("행동은 option index의 정수 배열이어야 합니다.")
        if len(action) != len(set(action)):
            raise GameError("같은 option을 중복 선택할 수 없습니다.")
        select = obs.get("select")
        if select is None:
            raise GameError("현재 선택 데이터가 없습니다.")
        minimum = int(select.get("minCount", 0))
        maximum = int(select.get("maxCount", 0))
        if not minimum <= len(action) <= maximum:
            raise GameError(f"{minimum}~{maximum}개를 선택해야 합니다. 현재 {len(action)}개입니다.")
        option_count = len(select.get("option") or [])
        for index in action:
            if index < 0 or index >= option_count:
                raise GameError(f"잘못된 option index입니다: {index}")

    def public_state(self) -> dict[str, Any]:
        with self.lock:
            if self.observation is None:
                return {
                    "active": False,
                    "presets": self.presets(),
                    "message": "게임을 시작하세요.",
                }
            obs = self.observation
            current = obs.get("current") or {}
            players = current.get("players") or [{}, {}]
            result = int(current.get("result", -1))
            actor = int(current.get("yourIndex", -1))
            human_turn = result == -1 and actor == self.human_seat
            ai_pending = result == -1 and actor == self.ai_seat
            select = obs.get("select") if human_turn else None
            enriched_options = []
            if select:
                for index, option in enumerate(select.get("option") or []):
                    enriched_options.append(self._enrich_option(obs, select, option, index))

            result_text = None
            if result != -1:
                if result == 2:
                    result_text = "무승부"
                elif result == self.human_seat:
                    result_text = "사람 승리"
                else:
                    result_text = "AI 승리"

            context = int(select.get("context", 0)) if select else None
            return {
                "active": True,
                "game_id": self.game_id,
                "agent": self.runtime.info.public() if self.runtime else None,
                "human_deck_label": self.human_deck_label,
                "human_seat": self.human_seat,
                "ai_seat": self.ai_seat,
                "human_turn": human_turn,
                "ai_pending": ai_pending,
                "actor": actor,
                "turn": int(current.get("turn", 0)),
                "turn_action_count": int(current.get("turnActionCount", 0)),
                "first_player": int(current.get("firstPlayer", -1)),
                "result": result,
                "result_text": result_text,
                "decision_count": self.decision_count,
                "human": self._serialize_player(players[self.human_seat], reveal_hand=True),
                "ai": self._serialize_player(players[self.ai_seat], reveal_hand=False),
                "stadium": [self._serialize_card(card) for card in current.get("stadium") or []],
                "looking": [
                    self._serialize_card(card) if card else None for card in current.get("looking") or []
                ],
                "flags": {
                    "supporter_played": bool(current.get("supporterPlayed", False)),
                    "stadium_played": bool(current.get("stadiumPlayed", False)),
                    "energy_attached": bool(current.get("energyAttached", False)),
                    "retreated": bool(current.get("retreated", False)),
                },
                "selection": None
                if not select
                else {
                    "type": int(select.get("type", 0)),
                    "context": context,
                    "context_name": CONTEXT_NAMES.get(context, str(context)),
                    "prompt": CONTEXT_KO.get(context, "선택지를 고르세요"),
                    "min_count": int(select.get("minCount", 0)),
                    "max_count": int(select.get("maxCount", 0)),
                    "remain_damage_counter": int(select.get("remainDamageCounter", 0)),
                    "remain_energy_cost": int(select.get("remainEnergyCost", 0)),
                    "options": enriched_options,
                },
                "logs": [self._enrich_log(log) for log in (obs.get("logs") or [])][-24:],
                "last_action": self.last_action,
                "action_history": self.action_history[-80:],
                "last_error": self.last_error,
                "record_available": self.record_dir is not None,
            }

    def public_state_for(self, viewer_seat: int) -> dict[str, Any]:
        """Serialize a PVP game from exactly one player's private viewpoint."""
        with self.lock:
            if viewer_seat not in (0, 1):
                raise GameError("관전 좌석은 0 또는 1이어야 합니다.")
            if self.observation is None:
                return {"active": False, "mode": "pvp", "message": "매칭된 게임이 없습니다."}
            if self.game_mode != "pvp":
                raise GameError("현재 게임은 온라인 플레이어 대전이 아닙니다.")
            obs = self.observation
            current = obs.get("current") or {}
            players = current.get("players") or [{}, {}]
            result = int(current.get("result", -1))
            actor = int(current.get("yourIndex", -1))
            viewer_turn = result == -1 and actor == viewer_seat
            opponent_seat = 1 - viewer_seat
            select = obs.get("select") if viewer_turn else None
            enriched_options = []
            if select:
                for index, option in enumerate(select.get("option") or []):
                    enriched_options.append(self._enrich_option(obs, select, option, index))

            result_text = None
            if result != -1:
                if result == 2:
                    result_text = "무승부"
                elif result == viewer_seat:
                    result_text = "승리"
                else:
                    result_text = "패배"

            def viewer_action(row: dict[str, Any] | None) -> dict[str, Any] | None:
                if not row:
                    return None
                copied = copy.deepcopy(row)
                row_actor = int(copied.get("actor", -1))
                copied["source"] = "human" if row_actor == viewer_seat else "ai"
                copied["actor_name"] = (
                    self.player_names[row_actor]
                    if 0 <= row_actor < len(self.player_names)
                    else f"Player {row_actor}"
                )
                return copied

            context = int(select.get("context", 0)) if select else None
            return {
                "active": True,
                "mode": "pvp",
                "game_id": self.game_id,
                "human_seat": viewer_seat,
                "ai_seat": opponent_seat,
                "viewer_seat": viewer_seat,
                "opponent_seat": opponent_seat,
                "player_names": list(self.player_names),
                "human_deck_label": self.player_deck_labels[viewer_seat],
                "opponent_deck_label": self.player_deck_labels[opponent_seat],
                "human_turn": viewer_turn,
                "ai_pending": False,
                "opponent_pending": result == -1 and actor == opponent_seat,
                "actor": actor,
                "turn": int(current.get("turn", 0)),
                "turn_action_count": int(current.get("turnActionCount", 0)),
                "first_player": int(current.get("firstPlayer", -1)),
                "result": result,
                "result_text": result_text,
                "decision_count": self.decision_count,
                "human": self._serialize_player(players[viewer_seat], reveal_hand=True),
                "ai": self._serialize_player(players[opponent_seat], reveal_hand=False),
                "stadium": [self._serialize_card(card) for card in current.get("stadium") or []],
                "looking": [
                    self._serialize_card(card) if card else None for card in current.get("looking") or []
                ],
                "flags": {
                    "supporter_played": bool(current.get("supporterPlayed", False)),
                    "stadium_played": bool(current.get("stadiumPlayed", False)),
                    "energy_attached": bool(current.get("energyAttached", False)),
                    "retreated": bool(current.get("retreated", False)),
                },
                "selection": None
                if not select
                else {
                    "type": int(select.get("type", 0)),
                    "context": context,
                    "context_name": CONTEXT_NAMES.get(context, str(context)),
                    "prompt": CONTEXT_KO.get(context, "선택지를 고르세요"),
                    "min_count": int(select.get("minCount", 0)),
                    "max_count": int(select.get("maxCount", 0)),
                    "remain_damage_counter": int(select.get("remainDamageCounter", 0)),
                    "remain_energy_cost": int(select.get("remainEnergyCost", 0)),
                    "options": enriched_options,
                },
                "logs": [self._enrich_log(log) for log in (obs.get("logs") or [])][-24:],
                "last_action": viewer_action(self.last_action),
                "action_history": [
                    viewer_action(row) for row in self.action_history[-80:] if row is not None
                ],
                "last_error": self.last_error,
                "record_available": self.record_dir is not None,
            }

    def _serialize_player(self, player: dict[str, Any], reveal_hand: bool) -> dict[str, Any]:
        hand = player.get("hand") if reveal_hand else None
        return {
            "active": [self._serialize_pokemon(p) for p in player.get("active") or [] if p],
            "bench": [self._serialize_pokemon(p) for p in player.get("bench") or [] if p],
            "bench_max": int(player.get("benchMax", 5)),
            "deck_count": int(player.get("deckCount", 0)),
            "discard": [self._serialize_card(card) for card in (player.get("discard") or [])[-16:]],
            "discard_count": len(player.get("discard") or []),
            "prize_count": len(player.get("prize") or []),
            "hand_count": int(player.get("handCount", len(hand or []))),
            "hand": [self._serialize_card(card) for card in hand or []] if reveal_hand else None,
            "conditions": {
                "poisoned": bool(player.get("poisoned", False)),
                "burned": bool(player.get("burned", False)),
                "asleep": bool(player.get("asleep", False)),
                "paralyzed": bool(player.get("paralyzed", False)),
                "confused": bool(player.get("confused", False)),
            },
        }

    def _serialize_card(self, card: dict[str, Any]) -> dict[str, Any]:
        if self.catalog is None:
            raise GameError("카드 카탈로그가 준비되지 않았습니다.")
        card_id = int(card.get("id", 0))
        summary = self.catalog.card_summary(card_id)
        summary.update(
            {
                "serial": int(card.get("serial", -1)),
                "player_index": int(card.get("playerIndex", -1)),
                "image_url": f"/api/card-image/{card_id}",
            }
        )
        return summary

    def _serialize_pokemon(self, pokemon: dict[str, Any]) -> dict[str, Any]:
        result = self._serialize_card(pokemon)
        result.update(
            {
                "hp_current": int(pokemon.get("hp", 0)),
                "hp_max": int(pokemon.get("maxHp", result.get("hp", 0))),
                "appeared_this_turn": bool(pokemon.get("appearThisTurn", False)),
                "energies": [
                    ENERGY_NAMES.get(int(energy), str(energy))
                    for energy in pokemon.get("energies") or []
                ],
                "energy_cards": [
                    self._serialize_card(card) for card in pokemon.get("energyCards") or []
                ],
                "tools": [self._serialize_card(card) for card in pokemon.get("tools") or []],
                "pre_evolution": [
                    self._serialize_card(card) for card in pokemon.get("preEvolution") or []
                ],
            }
        )
        return result

    def _card_from_area(
        self,
        obs: dict[str, Any],
        select: dict[str, Any],
        player_index: int,
        area: int,
        index: int,
    ) -> dict[str, Any] | None:
        current = obs.get("current") or {}
        players = current.get("players") or [{}, {}]
        if area == 12:
            items = current.get("looking") or []
        elif area == 1 and select.get("deck") is not None:
            items = select.get("deck") or []
        else:
            player = players[player_index] if 0 <= player_index < len(players) else {}
            key = {2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}.get(area)
            if area == 7:
                items = current.get("stadium") or []
            elif key:
                items = player.get(key) or []
            else:
                items = []
        if 0 <= index < len(items):
            return items[index]
        return None

    def _resolve_option_card(
        self,
        obs: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
    ) -> dict[str, Any] | None:
        option_type = int(option.get("type", -1))
        current = obs.get("current") or {}
        actor = int(current.get("yourIndex", 0))
        if option_type == 7:
            hand = (current.get("players") or [{}, {}])[actor].get("hand") or []
            index = int(option.get("index", -1))
            return hand[index] if 0 <= index < len(hand) else None
        if option_type in (8, 9, 10, 11):
            area = int(option.get("area", 2))
            index = int(option.get("index", -1))
            return self._card_from_area(obs, select, actor, area, index)
        if option_type in (3, 4, 5, 6):
            player_index = int(option.get("playerIndex", actor))
            area = int(option.get("area", 2))
            index = int(option.get("index", -1))
            return self._card_from_area(obs, select, player_index, area, index)
        if option_type == 15 and option.get("cardId") is not None:
            return {
                "id": int(option["cardId"]),
                "serial": int(option.get("serial", -1)),
                "playerIndex": actor,
            }
        return None

    def _target_text(
        self,
        obs: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
    ) -> str:
        if self.catalog is None:
            return ""
        option_type = int(option.get("type", -1))
        if option_type not in (8, 9):
            return ""
        current = obs.get("current") or {}
        actor = int(current.get("yourIndex", 0))
        target_area = int(option.get("inPlayArea", 4))
        target_index = int(option.get("inPlayIndex", 0))
        target = self._card_from_area(obs, select, actor, target_area, target_index)
        return self.catalog.card_name(target.get("id")) if target else AREA_NAMES.get(target_area, "대상")

    def _enrich_option(
        self,
        obs: dict[str, Any],
        select: dict[str, Any],
        option: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        if self.catalog is None:
            raise GameError("카드 카탈로그가 준비되지 않았습니다.")
        option_type = int(option.get("type", -1))
        card = self._resolve_option_card(obs, select, option)
        serialized = self._serialize_card(card) if card else None
        label = OPTION_NAMES.get(option_type, f"Option {option_type}")
        detail = ""
        if serialized:
            label = serialized["name"]
            detail = OPTION_NAMES.get(option_type, "선택")
            target = self._target_text(obs, select, option)
            if target:
                detail += f" → {target}"
        elif option_type == 13:
            attack = self.catalog.attacks.get(int(option.get("attackId", -1)))
            if attack:
                label = attack.name
                cost = " ".join(ENERGY_NAMES.get(int(e), str(e)) for e in attack.energies)
                detail = f"기술 · {attack.damage} damage · {cost}".strip(" ·")
            else:
                label = self.catalog.attack_name(option.get("attackId"))
                detail = "기술 사용"
        elif option_type == 0:
            label = str(option.get("number", 0))
            detail = "수 선택"
        elif option_type == 1:
            label, detail = "예", "효과를 진행합니다"
        elif option_type == 2:
            label, detail = "아니오", "효과를 진행하지 않습니다"
        elif option_type == 12:
            label, detail = "후퇴", "배틀 포켓몬을 벤치와 교체"
        elif option_type == 14:
            label, detail = "턴 종료", "현재 턴을 끝냅니다"
        elif option_type == 16:
            label = SPECIAL_NAMES.get(int(option.get("specialConditionType", -1)), "특수 상태")
            detail = "특수 상태 선택"
        elif option_type == 6:
            label = ENERGY_NAMES.get(int(option.get("energyIndex", -1)), "에너지")
            detail = f"{option.get('count', 1)}개 에너지"
        area = option.get("area")
        if area is not None and not detail:
            detail = AREA_NAMES.get(int(area), "")
        return {
            "index": index,
            "type": option_type,
            "type_name": OPTION_NAMES.get(option_type, str(option_type)),
            "label": label,
            "detail": detail,
            "card": serialized,
            "raw": option,
        }

    def _enrich_log(self, log: dict[str, Any]) -> dict[str, Any]:
        if self.catalog is None:
            return {"type": int(log.get("type", -1)), "text": "게임 이벤트", "raw": log}
        event_type = int(log.get("type", -1))
        player_index = log.get("playerIndex")
        if self.game_mode == "pvp" and isinstance(player_index, int) and 0 <= player_index < 2:
            who = self.player_names[player_index]
        else:
            who = (
                "Human"
                if player_index == self.human_seat
                else self.runtime.info.name
                if self.runtime and player_index == self.ai_seat
                else "Game"
            )
        text = f"{who}: event {event_type}"
        if event_type == 2:
            text = f"{who} 턴 시작"
        elif event_type == 3:
            text = f"{who} 턴 종료"
        elif event_type == 10:
            text = f"{who} · {self.catalog.card_name(log.get('cardId'))} 사용"
        elif event_type == 11:
            text = (
                f"{who} · {self.catalog.card_name(log.get('cardId'))}를 "
                f"{self.catalog.card_name(log.get('cardIdTarget'))}에게 부착"
            )
        elif event_type == 12:
            text = (
                f"{who} · {self.catalog.card_name(log.get('cardIdTarget'))} → "
                f"{self.catalog.card_name(log.get('cardId'))} 진화"
            )
        elif event_type == 15:
            text = f"{who} · {self.catalog.attack_name(log.get('attackId'))} 사용"
        elif event_type == 16:
            value = int(log.get("value", 0))
            text = f"{self.catalog.card_name(log.get('cardId'))} HP {value:+d}"
        elif event_type == 8:
            text = f"{who} · {self.catalog.card_name(log.get('cardIdBench'))}를 배틀필드로 교체"
        elif event_type == 22:
            text = f"{who} · 동전 {'앞면' if log.get('head') else '뒷면'}"
        elif event_type == 23:
            text = "게임 종료"
        elif event_type == 4:
            text = f"{who} · {self.catalog.card_name(log.get('cardId'))} 드로우"
        elif event_type == 5:
            text = f"{who} · 카드 1장 드로우"
        elif event_type == 6 and log.get("cardId") is not None:
            text = (
                f"{who} · {self.catalog.card_name(log.get('cardId'))} "
                f"{AREA_NAMES.get(int(log.get('fromArea', -1)), '?')} → "
                f"{AREA_NAMES.get(int(log.get('toArea', -1)), '?')}"
            )
        elif event_type == 7:
            text = f"{who} · 비공개 카드 이동"
        return {"type": event_type, "text": text, "raw": log}

    def _action_summary(
        self,
        before: dict[str, Any],
        action: list[int],
        source: str,
        after: dict[str, Any],
        think_ms: int,
    ) -> dict[str, Any]:
        select = before.get("select") or {}
        option_rows = []
        for index in action:
            options = select.get("option") or []
            if 0 <= index < len(options):
                option_rows.append(self._enrich_option(before, select, options[index], index))
        labels = [row["label"] for row in option_rows]
        details = [row["detail"] for row in option_rows if row.get("detail")]
        actor = int((before.get("current") or {}).get("yourIndex", -1))
        if self.game_mode == "pvp" and 0 <= actor < len(self.player_names):
            actor_name = self.player_names[actor]
        else:
            actor_name = "Human" if source == "human" else self.runtime.info.name if self.runtime else "AI"
        events = [self._enrich_log(log) for log in (after.get("logs") or [])]
        return {
            "sequence": self.decision_count,
            "source": source,
            "actor": actor,
            "actor_name": actor_name,
            "turn": int((before.get("current") or {}).get("turn", 0)),
            "context": CONTEXT_NAMES.get(int(select.get("context", -1)), "SELECTION"),
            "prompt": CONTEXT_KO.get(int(select.get("context", -1)), "선택"),
            "indices": list(action),
            "labels": labels,
            "details": details,
            "text": " · ".join(labels) if labels else "선택 없음",
            "think_ms": think_ms,
            "events": events[-8:],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _record_transition(
        self,
        before: dict[str, Any],
        action: list[int],
        source: str,
        after: dict[str, Any],
        think_ms: int,
    ) -> None:
        if self.record_dir is None:
            return
        select = before.get("select") or {}
        options = select.get("option") or []
        row = {
            "step": self.decision_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "actor": int((before.get("current") or {}).get("yourIndex", -1)),
            "action_indices": list(action),
            "selected_options": [options[index] for index in action if 0 <= index < len(options)],
            "think_ms": think_ms,
            "observation": self._strip_private_transport(before),
            "after_result": int((after.get("current") or {}).get("result", -1)),
        }
        with (self.record_dir / "transitions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _strip_private_transport(obs: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(obs)
        result.pop("search_begin_input", None)
        return result

    @staticmethod
    def _strip_card_names(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: GameManager._strip_card_names(item)
                for key, item in value.items()
                if key != "name"
            }
        if isinstance(value, list):
            return [GameManager._strip_card_names(item) for item in value]
        return value

    def _observation_for_official_player(
        self,
        raw_observation: dict[str, Any],
        visual_frame: dict[str, Any] | None,
        seat: int,
        step: int,
        remaining_time: float,
    ) -> dict[str, Any]:
        observation = copy.deepcopy(raw_observation)
        current = observation.get("current")
        actor = int((current or {}).get("yourIndex", -1))
        if isinstance(current, dict):
            current["yourIndex"] = seat
            players = current.get("players")
            visual_players = ((visual_frame or {}).get("current") or {}).get("players")
            if isinstance(players, list) and len(players) >= 2:
                for player_index, player in enumerate(players[:2]):
                    if not isinstance(player, dict):
                        continue
                    if player_index == seat:
                        if isinstance(visual_players, list) and player_index < len(visual_players):
                            visual_player = visual_players[player_index]
                            if isinstance(visual_player, dict):
                                visual_hand = visual_player.get("hand")
                                if isinstance(visual_hand, list):
                                    player["hand"] = self._strip_card_names(visual_hand)
                                    player["handCount"] = len(visual_hand)
                    else:
                        player["hand"] = None
        if seat != actor:
            observation["select"] = None
            observation["search_begin_input"] = None
        observation["step"] = step
        observation["remainingOverageTime"] = round(max(0.0, remaining_time), 6)
        observation.setdefault("logs", [])
        observation.setdefault("select", None)
        observation.setdefault("search_begin_input", None)
        return observation

    def _official_replay_payload(
        self,
        frames: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        names = metadata.get("player_names")
        if not isinstance(names, list) or len(names) < 2:
            names = ["Player 0", "Player 1"]
        names = [str(names[0]), str(names[1])]

        official_frames = copy.deepcopy(frames)
        if official_frames and isinstance(official_frames[0], dict):
            official_frames[0].setdefault("action", [list(self.human_deck if self.human_seat == 0 else self.ai_deck), list(self.human_deck if self.human_seat == 1 else self.ai_deck)])
            official_frames[0]["ps"] = names

        observations = self.official_observations or ([copy.deepcopy(self.observation)] if self.observation else [])
        actions = self.official_actions
        result = int(metadata.get("result", -1))
        finished = result in (0, 1, 2)
        rewards = [0, 0]
        if result == 0:
            rewards = [1, -1]
        elif result == 1:
            rewards = [-1, 1]

        remaining = [600.0, 600.0]
        pregame_observation = {
            "current": None,
            "logs": [],
            "remainingOverageTime": 600,
            "search_begin_input": None,
            "select": None,
            "step": 0,
        }
        steps: list[list[dict[str, Any]]] = [
            [
                {
                    "action": [],
                    "info": {},
                    "observation": copy.deepcopy(pregame_observation),
                    "reward": 0,
                    "status": "ACTIVE",
                    "visualize": official_frames,
                },
                {
                    "action": [],
                    "info": {},
                    "observation": copy.deepcopy(pregame_observation),
                    "reward": 0,
                    "status": "ACTIVE",
                },
            ]
        ]
        total_states = len(observations)
        deck_actions = [
            list(self.human_deck if self.human_seat == 0 else self.ai_deck),
            list(self.human_deck if self.human_seat == 1 else self.ai_deck),
        ]
        for observation_index, raw in enumerate(observations):
            if not isinstance(raw, dict):
                continue
            frame = official_frames[min(observation_index, len(official_frames) - 1)] if official_frames else None
            previous_action: list[int] = []
            previous_actor = -1
            if observation_index > 0 and observation_index - 1 < len(actions):
                previous_action = list(actions[observation_index - 1])
                previous_raw = observations[observation_index - 1]
                previous_actor = int(((previous_raw.get("current") or {}).get("yourIndex", -1)))
                if 0 <= previous_actor <= 1 and observation_index - 1 < len(self.official_think_ms):
                    remaining[previous_actor] -= self.official_think_ms[observation_index - 1] / 1000.0

            is_last = observation_index == total_states - 1
            step_status = "DONE" if is_last and finished else "ACTIVE"
            step_reward = rewards if is_last and finished else [0, 0]
            states: list[dict[str, Any]] = []
            for seat in (0, 1):
                if observation_index == 0:
                    state_action = deck_actions[seat]
                else:
                    state_action = previous_action if seat == previous_actor else []
                state = {
                    "action": state_action,
                    "info": {},
                    "observation": self._observation_for_official_player(
                        raw, frame, seat, observation_index + 1, remaining[seat]
                    ),
                    "reward": step_reward[seat],
                    "status": step_status,
                }
                states.append(state)
            steps.append(states)

        seed_text = str(metadata.get("game_id") or self.game_id or "cabt-local")
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        episode_id = int.from_bytes(digest[4:8], "big") % 1_000_000_000
        episode_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"cabt-local:{seed_text}"))
        statuses = ["DONE", "DONE"] if finished else ["ACTIVE", "ACTIVE"]

        return {
            "configuration": {
                "actTimeout": 0,
                "episodeSteps": 10000000,
                "runTimeout": 2000,
                "seed": seed,
            },
            "description": "Limited Card Battle.",
            "id": episode_uuid,
            "info": {
                "Agents": [
                    {"Name": names[0], "ThumbnailUrl": None},
                    {"Name": names[1], "ThumbnailUrl": None},
                ],
                "EpisodeId": episode_id,
                "LiveVideoPath": None,
                "TeamNames": names,
            },
            "module_version": "1.32.2",
            "name": "cabt",
            "rewards": rewards,
            "schema_version": 1,
            "specification": {
                "action": {
                    "default": [],
                    "description": "List of option index.",
                    "type": "array",
                },
                "agents": [2],
                "configuration": {
                    "actTimeout": {
                        "default": 0,
                        "description": "Maximum runtime (seconds) to obtain an action from an agent.",
                        "minimum": 0,
                        "type": "number",
                    },
                    "episodeSteps": {
                        "default": 10000000,
                        "description": "Maximum number of steps in the episode.",
                        "minimum": 1,
                        "type": "integer",
                    },
                    "runTimeout": {
                        "default": 2000,
                        "description": "Maximum runtime (seconds) of an episode (not necessarily DONE).",
                        "minimum": 0,
                        "type": "number",
                    },
                },
                "info": {},
                "observation": {
                    "remainingOverageTime": {
                        "default": 600,
                        "description": "Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0.",
                        "minimum": 0,
                        "shared": False,
                        "type": "number",
                    },
                    "step": {
                        "default": 0,
                        "description": "Current step within the episode.",
                        "minimum": 0,
                        "shared": True,
                        "type": "integer",
                    },
                },
                "reward": {
                    "default": 0,
                    "description": "Lost:-1, Won:1, Draw:0",
                    "enum": [-1, 0, 1],
                    "type": ["number", "null"],
                },
            },
            "statuses": statuses,
            "steps": steps,
            "title": "Card Battle",
            "version": "1.0.0",
        }

    def _save_replay(self) -> None:
        if self.record_dir is None or self.observation is None or self.runtime is None:
            return
        try:
            frames = json.loads(self.runtime.visualize_data())
            self._write_json(self.record_dir / "replay_visualize.json", frames)
            metadata_path = self.record_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            official = self._official_replay_payload(frames, metadata)
            self._write_compact_json(self.record_dir / "official_replay.json", official)
        except Exception:
            pass

    def _finalize_record(self, error: str | None = None) -> None:
        if self.record_dir is None:
            return
        metadata_path = self.record_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        result = -1
        if self.observation is not None:
            result = int((self.observation.get("current") or {}).get("result", -1))
        metadata.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "error" if error else ("finished" if result != -1 else "active"),
                "result": result,
                "decision_count": self.decision_count,
                "duration_seconds": round(time.time() - self.started_at, 3)
                if self.started_at
                else None,
                "error": error,
            }
        )
        self._write_json(metadata_path, metadata)
        self._save_replay()

    def record_zip(self) -> Path:
        with self.lock:
            if self.record_dir is None or not self.record_dir.exists():
                raise GameError("저장된 게임 기록이 없습니다.")
            self._finalize_record()
            zip_path = self.record_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(self.record_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(self.record_dir.parent))
            return zip_path

    def official_replay_file(self) -> Path:
        with self.lock:
            if self.record_dir is None or not self.record_dir.exists():
                raise GameError("저장된 게임 기록이 없습니다.")
            self._finalize_record()
            path = self.record_dir / "official_replay.json"
            if not path.is_file():
                raise GameError("공식 형식 JSON을 생성하지 못했습니다.")
            return path

    def viewer_launcher(self, player: int) -> Path:
        with self.lock:
            if player not in (0, 1):
                raise GameError("player는 0 또는 1이어야 합니다.")
            if self.record_dir is None:
                raise GameError("게임 기록이 없습니다.")
            replay_path = self.record_dir / "replay_visualize.json"
            if not replay_path.exists():
                raise GameError("replay_visualize.json이 아직 생성되지 않았습니다.")
            frames = json.loads(replay_path.read_text(encoding="utf-8"))
            metadata = json.loads((self.record_dir / "metadata.json").read_text(encoding="utf-8"))
            names = metadata.get("player_names", ["Player 0", "Player 1"])
            if frames and isinstance(frames[0], dict):
                frames[0]["ps"] = names
            import base64

            payload = json.dumps(frames, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            encoded = base64.b64encode(payload).decode("ascii")
            url = f"https://ptcgvis.heroz.jp/Visualizer/Replay/{player}"
            html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>CABT Replay</title></head><body style="font-family:system-ui;background:#111827;color:#fff;
display:grid;place-items:center;min-height:100vh"><form id="f" method="post" action="{url}">
<input id="j" type="hidden" name="json"><button type="submit">공식 뷰어 열기</button></form>
<script>const b="{encoded}";const x=Uint8Array.from(atob(b),c=>c.charCodeAt(0));
document.getElementById('j').value=new TextDecoder('utf-8').decode(x);
document.getElementById('f').submit();</script></body></html>"""
            path = self.record_dir / f"official_replay_player{player}.html"
            path.write_text(html, encoding="utf-8")
            return path

    @staticmethod
    def _write_deck(path: Path, deck: list[int]) -> None:
        path.write_text("\n".join(str(value) for value in deck) + "\n", encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_compact_json(path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


__all__ = [
    "AgentRepository",
    "GameError",
    "GameManager",
    "ROOT",
    "parse_uploaded_deck",
    "read_deck",
]
