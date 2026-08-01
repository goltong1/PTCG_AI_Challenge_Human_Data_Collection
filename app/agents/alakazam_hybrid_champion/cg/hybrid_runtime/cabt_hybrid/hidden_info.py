from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .import_utils import read_deck


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _id(card: Any) -> int:
    if card is None:
        return 0
    return int(_get(card, "id", _get(card, "cardId", 0)) or 0)


def _serial(card: Any) -> int | None:
    value = _get(card, "serial", None)
    return int(value) if value is not None else None


def _player_index(card: Any) -> int | None:
    value = _get(card, "playerIndex", None)
    return int(value) if value is not None else None


def _is_energy_id(card_id: int) -> bool:
    # CABT card IDs 1-20 are Energy cards in the current card pool.
    return 1 <= card_id <= 20


def _is_pokemon_id(card_id: int) -> bool:
    return 21 <= card_id < 1000


def _copy_limit(card_id: int) -> int:
    # Basic Energy is unrestricted. Keeping all Energy unrestricted is safer for
    # hidden-state reconstruction than accidentally truncating a legal deck.
    return 60 if _is_energy_id(card_id) else 4


def _iter_nested_cards(card: Any) -> Iterable[tuple[Any, str]]:
    if card is None:
        return
    yield card, "card"
    for child in _get(card, "preEvolution", []) or []:
        yield from _iter_nested_cards(child)
    for child in _get(card, "energyCards", []) or []:
        yield child, "attached"
    for child in _get(card, "tools", []) or []:
        yield child, "attached"


def _consume_card(counter: Counter[int], card: Any) -> None:
    if card is None:
        return
    cid = _id(card)
    if cid and counter[cid] > 0:
        counter[cid] -= 1
    for child in _get(card, "preEvolution", []) or []:
        _consume_card(counter, child)
    for child in _get(card, "energyCards", []) or []:
        _consume_card(counter, child)
    for child in _get(card, "tools", []) or []:
        _consume_card(counter, child)


def remaining_from_template(template: list[int], visible_cards: list[Any], count: int) -> list[int]:
    counter = Counter(template)
    for card in visible_cards:
        _consume_card(counter, card)
    remaining: list[int] = []
    for cid, n in counter.items():
        remaining.extend([cid] * max(0, n))
    if not remaining:
        remaining = list(template)
    while len(remaining) < count:
        remaining.extend(template)
    return remaining[:count]


def visible_player_cards(player: Any) -> list[Any]:
    cards: list[Any] = []
    # Opponent hand and prizes are hidden in normal observations, but the lists
    # are harmless when the engine supplies public/revealed entries.
    for zone in ["hand", "discard", "active", "bench", "prize"]:
        cards.extend([x for x in (_get(player, zone, []) or []) if x is not None])
    return cards


@dataclass
class ArchetypeInference:
    name: str
    confidence: float
    novelty: float
    coverage: float
    evidence_weight: float
    probabilities: dict[str, float]
    template: list[int]
    mode: str
    observed_cards: dict[int, int]

    @property
    def is_unknown(self) -> bool:
        return self.mode == "unknown"

    @property
    def is_mixed(self) -> bool:
        return self.mode == "mixed"


@dataclass
class ArchetypeTracker:
    """Accumulates every public card revealed by the opponent during a game.

    A card may leave the field, be shuffled back, or have a Stadium replaced.
    Tracking its serial number prevents that information from being forgotten
    and prevents repeated observations from counting as additional copies.
    """

    seen_serial_weights: dict[int, float] = field(default_factory=dict)
    seen_serial_ids: dict[int, int] = field(default_factory=dict)
    no_serial_max_counts: Counter[int] = field(default_factory=Counter)
    no_serial_weights: dict[int, float] = field(default_factory=dict)
    last_turn: int | None = None
    last_perspective: int | None = None

    def reset(self) -> None:
        self.seen_serial_weights.clear()
        self.seen_serial_ids.clear()
        self.no_serial_max_counts.clear()
        self.no_serial_weights.clear()
        self.last_turn = None
        self.last_perspective = None

    def _maybe_reset(self, obs: Any, perspective: int) -> None:
        current = _get(obs, "current")
        turn = int(_get(current, "turn", 0) or 0)
        if self.last_turn is not None:
            if turn < self.last_turn:
                self.reset()
            elif self.last_perspective is not None and perspective != self.last_perspective and turn <= 1:
                self.reset()
        self.last_turn = turn
        self.last_perspective = perspective

    def _record(self, card: Any, weight: float, snapshot_counts: Counter[int]) -> None:
        for nested, relation in _iter_nested_cards(card):
            cid = _id(nested)
            if not cid:
                continue
            nested_weight = weight
            if relation == "attached":
                nested_weight = min(weight, 2.25)
            elif nested is not card:
                nested_weight = min(weight, 3.25)
            serial = _serial(nested)
            if serial is None:
                snapshot_counts[cid] += 1
                self.no_serial_weights[cid] = max(self.no_serial_weights.get(cid, 0.0), nested_weight)
                continue
            old_weight = self.seen_serial_weights.get(serial, 0.0)
            if nested_weight > old_weight:
                self.seen_serial_weights[serial] = nested_weight
                self.seen_serial_ids[serial] = cid

    def update(self, obs: Any, perspective: int) -> tuple[Counter[int], dict[int, float]]:
        self._maybe_reset(obs, perspective)
        current = _get(obs, "current")
        players = _get(current, "players", []) or []
        opponent = players[1 - perspective]
        snapshot_counts: Counter[int] = Counter()

        for card in _get(opponent, "active", []) or []:
            self._record(card, 5.0, snapshot_counts)
        for card in _get(opponent, "bench", []) or []:
            self._record(card, 4.5, snapshot_counts)
        for card in _get(opponent, "discard", []) or []:
            self._record(card, 2.25, snapshot_counts)
        for card in _get(opponent, "hand", []) or []:
            self._record(card, 3.0, snapshot_counts)

        # Stadium and temporary reveal/search windows live at the current-state level.
        for card in _get(current, "stadium", []) or []:
            owner = _player_index(card)
            if owner is None or owner == 1 - perspective:
                self._record(card, 2.75, snapshot_counts)
        looking = _get(current, "looking", None)
        if looking is not None:
            candidates = looking if isinstance(looking, list) else [looking]
            for card in candidates:
                if _player_index(card) == 1 - perspective:
                    self._record(card, 3.25, snapshot_counts)

        for cid, count in snapshot_counts.items():
            self.no_serial_max_counts[cid] = max(self.no_serial_max_counts[cid], count)

        counts: Counter[int] = Counter(self.seen_serial_ids.values())
        counts.update(self.no_serial_max_counts)
        weights: dict[int, float] = defaultdict(float)
        for serial, cid in self.seen_serial_ids.items():
            weights[cid] += self.seen_serial_weights.get(serial, 1.0)
        for cid, count in self.no_serial_max_counts.items():
            weights[cid] += self.no_serial_weights.get(cid, 1.0) * count
        return counts, dict(weights)


def _softmax(scores: dict[str, float], temperature: float = 2.25) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    exp_values = {name: math.exp((score - maximum) / max(0.1, temperature)) for name, score in scores.items()}
    total = sum(exp_values.values()) or 1.0
    return {name: value / total for name, value in exp_values.items()}


def _build_template(
    decks: dict[str, list[int]],
    probabilities: dict[str, float],
    observed: Counter[int],
    *,
    generic: bool,
) -> list[int]:
    deck_counts = {name: Counter(deck) for name, deck in decks.items()}
    document_frequency = Counter()
    for counter in deck_counts.values():
        document_frequency.update(counter.keys())

    expected: dict[int, float] = defaultdict(float)
    for name, probability in probabilities.items():
        for cid, count in deck_counts[name].items():
            if generic and _is_pokemon_id(cid) and cid not in observed and document_frequency[cid] < 2:
                continue
            expected[cid] += probability * count

    # Public cards must be representable in the inferred 60-card template even
    # when the archetype has never been seen during training.
    forced: Counter[int] = Counter()
    for cid, count in observed.items():
        forced[cid] = min(_copy_limit(cid), max(0, count))
        if generic and _is_pokemon_id(cid):
            # A revealed Basic or attacker is commonly played in multiple copies.
            expected[cid] = max(expected.get(cid, 0.0), float(min(4, count + 2)))

    result_counts = Counter(forced)
    total = sum(result_counts.values())
    ranked = sorted(
        expected,
        key=lambda cid: (expected[cid] - result_counts[cid], document_frequency[cid], -cid),
        reverse=True,
    )

    # Allocate expected cards one copy at a time. This avoids rounding a large
    # number of low-probability cards into an oversized mixture.
    while total < 60:
        best_cid = None
        best_need = -float("inf")
        for cid in ranked:
            if result_counts[cid] >= _copy_limit(cid):
                continue
            need = expected[cid] - result_counts[cid]
            if need > best_need:
                best_need = need
                best_cid = cid
        if best_cid is None or best_need <= 0:
            break
        result_counts[best_cid] += 1
        total += 1

    # Fill any remaining slots with broadly shared Trainers and Energy rather
    # than inventing an unobserved matchup-specific Pokémon engine.
    generic_pool = sorted(
        document_frequency,
        key=lambda cid: (
            1 if cid >= 1000 else 0,
            document_frequency[cid],
            sum(counter[cid] for counter in deck_counts.values()),
            -cid,
        ),
        reverse=True,
    )
    if generic:
        generic_pool = [cid for cid in generic_pool if cid >= 1000 or _is_energy_id(cid)] + generic_pool

    cursor = 0
    while total < 60 and generic_pool:
        cid = generic_pool[cursor % len(generic_pool)]
        cursor += 1
        if result_counts[cid] >= _copy_limit(cid):
            if cursor > len(generic_pool) * 20:
                break
            continue
        result_counts[cid] += 1
        total += 1

    # Last-resort Basic Energy padding. Prefer an Energy already revealed or
    # common in the known template pool.
    if total < 60:
        energy_candidates = [cid for cid in observed if _is_energy_id(cid)]
        if not energy_candidates:
            energy_candidates = [cid for cid in generic_pool if _is_energy_id(cid)] or [5]
        energy_id = energy_candidates[0]
        result_counts[energy_id] += 60 - total

    template: list[int] = []
    for cid, count in sorted(result_counts.items()):
        template.extend([cid] * count)
    return template[:60]


def infer_opponent_archetype(
    obs: Any,
    opponent_dirs: list[Path],
    perspective: int,
    tracker: ArchetypeTracker | None = None,
    *,
    high_confidence: float = 0.68,
    medium_confidence: float = 0.45,
    minimum_evidence_weight: float = 2.0,
) -> ArchetypeInference:
    if not opponent_dirs:
        raise ValueError("No opponent deck templates are available")

    decks = {directory.name: read_deck(directory) for directory in opponent_dirs}
    deck_counts = {name: Counter(deck) for name, deck in decks.items()}
    document_frequency = Counter()
    for counter in deck_counts.values():
        document_frequency.update(counter.keys())

    tracker = tracker or ArchetypeTracker()
    observed, evidence_weights = tracker.update(obs, perspective)
    total_weight = sum(evidence_weights.values())
    template_count = len(decks)

    scores: dict[str, float] = {}
    coverages: dict[str, float] = {}
    for name, counts in deck_counts.items():
        matched_weight = 0.0
        score = 0.0
        for cid, seen_count in observed.items():
            weight = evidence_weights.get(cid, float(seen_count))
            per_copy_weight = weight / max(1, seen_count)
            df = document_frequency[cid]
            idf = math.log((template_count + 1.0) / (df + 0.5)) + 0.20
            matched = min(seen_count, counts[cid])
            unmatched = max(0, seen_count - counts[cid])
            matched_part = per_copy_weight * matched
            unmatched_part = per_copy_weight * unmatched
            matched_weight += matched_part
            score += matched_part * idf
            score -= unmatched_part * idf * 1.35
        coverages[name] = matched_weight / total_weight if total_weight > 0 else 0.0
        scores[name] = score

    probabilities = _softmax(scores)
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    best_name, best_probability = ordered[0]
    second_probability = ordered[1][1] if len(ordered) > 1 else 0.0
    coverage = coverages[best_name]
    novelty = max(0.0, min(1.0, 1.0 - coverage))
    evidence_factor = min(1.0, total_weight / 6.0)
    margin_factor = max(0.0, min(1.0, (best_probability - second_probability) / 0.55))
    confidence = best_probability * (0.55 + 0.45 * evidence_factor) * (0.55 + 0.45 * coverage)
    confidence *= 0.70 + 0.30 * margin_factor

    if total_weight < minimum_evidence_weight:
        mode = "unknown"
    elif confidence >= high_confidence and novelty <= 0.30:
        mode = "known"
    elif confidence >= medium_confidence and novelty <= 0.55:
        mode = "mixed"
    else:
        mode = "unknown"

    if mode == "known":
        template = list(decks[best_name])
        name = best_name
    elif mode == "mixed":
        top = ordered[: min(3, len(ordered))]
        total_probability = sum(value for _, value in top) or 1.0
        blend = {name: value / total_probability for name, value in top}
        template = _build_template(decks, blend, observed, generic=False)
        name = "+".join(name for name, _ in top[:2])
    else:
        # Retain weak evidence, but use a broad public-card-driven hidden deck.
        # This prevents an unseen archetype from being forced into the nearest
        # known matchup and prevents known-matchup models from oversteering play.
        weak = {name: 0.35 * probabilities[name] + 0.65 / len(decks) for name in decks}
        normalizer = sum(weak.values()) or 1.0
        weak = {name: value / normalizer for name, value in weak.items()}
        template = _build_template(decks, weak, observed, generic=True)
        name = "unknown_generic"

    return ArchetypeInference(
        name=name,
        confidence=max(0.0, min(1.0, confidence)),
        novelty=novelty,
        coverage=coverage,
        evidence_weight=total_weight,
        probabilities=dict(ordered),
        template=template,
        mode=mode,
        observed_cards=dict(observed),
    )


def choose_opponent_template(obs: Any, opponent_dirs: list[Path], perspective: int) -> list[int]:
    """Backward-compatible wrapper used by older scripts."""
    return infer_opponent_archetype(obs, opponent_dirs, perspective).template


def predict_search_inputs(
    obs: Any,
    own_deck_template: list[int],
    opponent_deck_template: list[int],
    perspective: int,
) -> dict[str, list[int]]:
    current = _get(obs, "current")
    players = _get(current, "players")
    me = players[perspective]
    op = players[1 - perspective]

    your_deck_count = int(_get(me, "deckCount", 0) or 0)
    your_prize_count = len(_get(me, "prize", []) or [])
    op_deck_count = int(_get(op, "deckCount", 0) or 0)
    op_prize_count = len(_get(op, "prize", []) or [])
    op_hand_count = int(_get(op, "handCount", 0) or 0)

    your_visible = visible_player_cards(me)
    op_visible = visible_player_cards(op)

    your_remaining = remaining_from_template(own_deck_template, your_visible, your_deck_count + your_prize_count)
    op_remaining = remaining_from_template(
        opponent_deck_template,
        op_visible,
        op_deck_count + op_prize_count + op_hand_count,
    )

    your_prize = [_id(c) for c in (_get(me, "prize", []) or []) if c is not None]
    if len(your_prize) < your_prize_count:
        your_prize.extend(your_remaining[your_deck_count:your_deck_count + your_prize_count])

    op_hand = op_remaining[:op_hand_count]
    op_prize = op_remaining[op_hand_count:op_hand_count + op_prize_count]
    op_deck = op_remaining[op_hand_count + op_prize_count:op_hand_count + op_prize_count + op_deck_count]

    opponent_active: list[int] = []
    active = _get(op, "active", []) or []
    if active and active[0] is None:
        basic_fallbacks = [741, 305, 119, 344, 673, 646, 463, 1]
        opponent_active = [next((cid for cid in basic_fallbacks if cid in opponent_deck_template), opponent_deck_template[0])]

    return {
        "your_deck": your_remaining[:your_deck_count],
        "your_prize": your_prize[:your_prize_count],
        "opponent_deck": op_deck,
        "opponent_prize": op_prize,
        "opponent_hand": op_hand,
        "opponent_active": opponent_active,
    }
