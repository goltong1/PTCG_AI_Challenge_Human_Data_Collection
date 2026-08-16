"""Pure deterministic safety guards for the Dragapult v22 candidate.

The runtime adapter in :mod:`main` supplies only facts known at the current
causal prefix.  This module deliberately knows nothing about CABT option
indices: actions and commitments are identified by card/target serials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional, Tuple


class ActionKind(str, Enum):
    BENCH_BASIC = "bench_basic"
    HAND_RESET = "hand_reset"
    BASIC_SEARCH_ITEM = "basic_search_item"
    RECOVERY_TARGET = "recovery_target"
    OTHER = "other"


class Disposition(str, Enum):
    ALLOW = "allow"
    DEMOTE = "demote"
    DEFER = "defer"
    BLOCK = "block"


class Proof(str, Enum):
    UNKNOWN = "unknown"
    ESTIMATED = "estimated"
    PROVEN = "proven"


class CommitmentKind(str, Enum):
    SELECTED_SERIAL = "selected_serial"
    ATTACK_CONTINUITY_ATTACH = "attack_continuity_attach"


class BenchRole(str, Enum):
    RESERVED_GOAL = "reserved_goal"
    CORE_ATTACKER = "core_attacker"
    HIGH_VALUE_SUPPORT = "high_value_support"
    CONDITIONAL_DUPLICATE = "conditional_duplicate"
    OTHER = "other"


class GuardId(str, Enum):
    P0_TERMINAL_LOW_HP_BENCH = "P0_terminal_low_hp_bench"
    P1_SELECTED_SERIAL_BEFORE_RESET = "P1_selected_serial_before_reset"
    P1_ATTACH_BEFORE_RESET = "P1_attach_before_reset"
    P2_RESERVED_BENCH_SLOT = "P2_reserved_bench_slot"
    P2_DEAD_RECOVERY_TARGET = "P2_dead_recovery_target"
    P2_FULL_BENCH_RECOVERY_DEMOTION = "P2_full_bench_recovery_demotion"
    P3_NO_TARGET_BASIC_SEARCH = "P3_no_target_basic_search"
    P3_REDUNDANT_EMPTY_SEARCH = "P3_redundant_empty_search"


@dataclass(frozen=True)
class ActionRef:
    kind: ActionKind
    serial: Optional[int] = None
    owner: Optional[int] = None
    card_id: Optional[int] = None
    card_name: str = ""
    target_serial: Optional[int] = None
    hp: Optional[int] = None
    prize_value: int = 0
    consumes_bench_slot: bool = False
    bench_role: BenchRole = BenchRole.OTHER
    consumes_commitment_serials: FrozenSet[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TerminalPrizeThreat:
    opponent_prizes_remaining: int
    pending_non_attack_ko_prizes: int = 0
    pending_ko_proof: Proof = Proof.UNKNOWN
    attack_ready_now: bool = False
    attack_available_after_pending_ko: bool = False
    observed_bench_damage: int = 0
    bench_damage_proof: Proof = Proof.UNKNOWN
    candidate_targetable_on_bench: bool = True
    candidate_stays_at_or_below_hp_until_opponent_attack: bool = True
    safe_alternative_exists: bool = False
    same_turn_win_available: bool = False


@dataclass(frozen=True)
class PreResetCommitment:
    serial: int
    kind: CommitmentKind
    selected_this_turn: bool = False
    unconsumed: bool = True
    in_hand: bool = True
    will_be_lost_by_reset: bool = True
    legal_use_now: bool = False
    reset_remains_legal_after_use: bool = False
    productive_use_proof: Proof = Proof.UNKNOWN
    continuity_gain_proof: Proof = Proof.UNKNOWN
    enables_next_own_turn_attack: bool = False
    description: str = ""


@dataclass(frozen=True)
class BenchPlan:
    capacity: int = 5
    occupied: int = 0
    explicitly_reserved_slots: int = 0
    reservation_proof: Proof = Proof.UNKNOWN
    reserved_for: Tuple[str, ...] = ()
    candidate_is_reserved_goal: bool = False
    candidate_may_consume_reserved_slot: bool = False
    recovery_target_requires_bench_slot: bool = False
    can_free_bench_slot_before_target_use: bool = False
    target_has_nonbench_productive_use: bool = False
    must_consume_target_this_turn: bool = False
    productive_alternative_serials: Tuple[int, ...] = ()

    @property
    def free_slots(self):
        return max(0, self.capacity - self.occupied)


@dataclass(frozen=True)
class SearchFacts:
    eligible_target_upper_bound: Optional[int] = None
    productive_target_upper_bound: Optional[int] = None
    target_bound_proof: Proof = Proof.UNKNOWN
    same_search_resolved_this_turn: bool = False
    deck_order_information_since_last_shuffle: bool = False
    explicit_shuffle_benefit: bool = False
    item_has_other_effect: bool = False


@dataclass(frozen=True)
class GuardInput:
    action: ActionRef
    terminal_threat: Optional[TerminalPrizeThreat] = None
    pre_reset_commitments: Tuple[PreResetCommitment, ...] = ()
    bench_plan: Optional[BenchPlan] = None
    search_facts: Optional[SearchFacts] = None


@dataclass(frozen=True)
class GuardHit:
    guard_id: GuardId
    disposition: Disposition
    reason: str
    required_serials: Tuple[int, ...] = ()
    alternative_serials: Tuple[int, ...] = ()


@dataclass(frozen=True)
class GuardVerdict:
    disposition: Disposition
    hits: Tuple[GuardHit, ...]

    @property
    def blocked(self):
        return self.disposition in (Disposition.BLOCK, Disposition.DEFER)


def _p0(inp):
    a, t = inp.action, inp.terminal_threat
    if a.kind is not ActionKind.BENCH_BASIC or t is None:
        return None
    if a.hp is None or a.hp <= 0 or a.prize_value <= 0:
        return None
    if t.same_turn_win_available or not t.safe_alternative_exists:
        return None
    if t.pending_ko_proof is not Proof.PROVEN or t.bench_damage_proof is not Proof.PROVEN:
        return None
    if not t.attack_ready_now or not t.attack_available_after_pending_ko:
        return None
    if not t.candidate_targetable_on_bench or not t.candidate_stays_at_or_below_hp_until_opponent_attack:
        return None
    if t.observed_bench_damage < a.hp:
        return None
    after = t.opponent_prizes_remaining - t.pending_non_attack_ko_prizes
    if after <= 0 or after > a.prize_value:
        return None
    return GuardHit(
        GuardId.P0_TERMINAL_LOW_HP_BENCH,
        Disposition.BLOCK,
        "proven non-attack KO plus revealed bench damage closes the prize map",
    )


def _p1(inp):
    if inp.action.kind is not ActionKind.HAND_RESET:
        return ()
    hits = []
    for c in inp.pre_reset_commitments:
        if c.serial in inp.action.consumes_commitment_serials:
            continue
        if not (c.unconsumed and c.in_hand and c.will_be_lost_by_reset and c.legal_use_now and c.reset_remains_legal_after_use):
            continue
        if c.kind is CommitmentKind.SELECTED_SERIAL:
            if c.selected_this_turn and c.productive_use_proof is Proof.PROVEN:
                hits.append(GuardHit(GuardId.P1_SELECTED_SERIAL_BEFORE_RESET, Disposition.DEFER, c.description or "consume selected serial before reset", (c.serial,)))
        elif c.kind is CommitmentKind.ATTACK_CONTINUITY_ATTACH:
            if c.continuity_gain_proof is Proof.PROVEN and c.enables_next_own_turn_attack:
                hits.append(GuardHit(GuardId.P1_ATTACH_BEFORE_RESET, Disposition.DEFER, c.description or "use continuity attachment before reset", (c.serial,)))
    return tuple(hits)


def _p2(inp):
    a, p = inp.action, inp.bench_plan
    if p is None:
        return ()
    hits = []
    if a.consumes_bench_slot and a.kind is ActionKind.BENCH_BASIC:
        protected = a.bench_role in (BenchRole.RESERVED_GOAL, BenchRole.CORE_ATTACKER)
        if (p.reservation_proof is Proof.PROVEN and p.explicitly_reserved_slots > 0 and p.free_slots <= p.explicitly_reserved_slots and not p.candidate_is_reserved_goal and not p.candidate_may_consume_reserved_slot and not protected):
            hits.append(GuardHit(GuardId.P2_RESERVED_BENCH_SLOT, Disposition.DEFER, "preserve proven bench reservation"))
    if a.kind is ActionKind.RECOVERY_TARGET and p.recovery_target_requires_bench_slot:
        no_slot = p.free_slots == 0 and not p.can_free_bench_slot_before_target_use
        alternatives = tuple(sorted(set(p.productive_alternative_serials)))
        if no_slot and not p.target_has_nonbench_productive_use and alternatives:
            if p.must_consume_target_this_turn:
                hits.append(GuardHit(GuardId.P2_DEAD_RECOVERY_TARGET, Disposition.BLOCK, "recovered Basic cannot use a slot before expiry", alternative_serials=alternatives))
            else:
                hits.append(GuardHit(GuardId.P2_FULL_BENCH_RECOVERY_DEMOTION, Disposition.DEMOTE, "full bench delays recovered Basic", alternative_serials=alternatives))
    return tuple(hits)


def _p3(inp):
    a, f = inp.action, inp.search_facts
    if a.kind is not ActionKind.BASIC_SEARCH_ITEM or f is None:
        return None
    if f.target_bound_proof is not Proof.PROVEN or f.explicit_shuffle_benefit or f.item_has_other_effect:
        return None
    if f.eligible_target_upper_bound == 0:
        return GuardHit(GuardId.P3_NO_TARGET_BASIC_SEARCH, Disposition.BLOCK, "all eligible targets proven outside deck")
    if f.productive_target_upper_bound == 0 and f.same_search_resolved_this_turn and not f.deck_order_information_since_last_shuffle:
        return GuardHit(GuardId.P3_REDUNDANT_EMPTY_SEARCH, Disposition.BLOCK, "equivalent search already resolved and no productive target remains")
    return None


_ORDER = {Disposition.ALLOW: 0, Disposition.DEMOTE: 1, Disposition.DEFER: 2, Disposition.BLOCK: 3}


def evaluate(inp):
    hits = []
    p0 = _p0(inp)
    if p0:
        hits.append(p0)
    hits.extend(_p1(inp))
    hits.extend(_p2(inp))
    p3 = _p3(inp)
    if p3:
        hits.append(p3)
    if not hits:
        return GuardVerdict(Disposition.ALLOW, ())
    return GuardVerdict(max((h.disposition for h in hits), key=_ORDER.get), tuple(hits))
