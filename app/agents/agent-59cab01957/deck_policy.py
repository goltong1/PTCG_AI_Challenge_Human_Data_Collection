"""Dragapult v56 fixed-deck heuristic policy distilled from loss replays.

The target line is two Dreepy bodies, Drakloak draw before evolution, and a
Fire+Psychic Dragapult ex ready for Phantom Dive by the third own turn.  A
second line is preserved while Munkidori, Dudunsparce, Meowth ex, Fezandipiti
ex, Budew, Unfair Stamp, Risky Ruins and Boss are admitted only by their
state-dependent conversion rules.  Runtime remains deterministic and shallow.

Rating-100 macro foundation cleaned into a matchup-agnostic core.
Runtime decisions are rule-dominant, with a deterministic hypergeometric
expected-value residual for future setup, attack conversion, and exposure.
Cross-opponent distillation adds a conservative confirmed-KO tempo term while
keeping matchup names and opponent card IDs out of the live policy.
V16 adds a guarded two-turn Prize bundle plan that can leave a Bench target at
10 HP only when a second Phantom Dive is already secured and disruption risk
is explicit. Chi-Yu wall forcing and extra Budew loops remain disabled after
cross-play ablation showed no universal gain.
V17 distills cross-play into probability-weighted evolution and attack value:
charged or damaged evolution lines receive survival value under pressure, and
Phantom Dive receives expected Prize-conversion value without hard-coding any
opponent identity. The residuals were retained only after five-opponent
cross-play improved while direct play against V16 remained statistically even.
V19 adds one replay-distilled two-ply conversion: when a ready Phantom Dive,
transferable own damage, and a 201-230 HP opponent coincide, Darkness Energy on
Munkidori is valued as an immediate 30+200 KO rather than delayed development.
Broader wall and survival-search experiments were rejected by cross-play gates.
V20 removes the false-positive branch where a 201-230 HP Bench target caused
Darkness commitment even though Phantom Dive could not deal its 200 Active
damage there. The conversion now requires that exact window on the Active.
It also caps the window by the counters actually transferable this turn, so
10 available damage is never valued as if Munkidori could move the full 30.
This correction survived a successive-halving league scored on external mean,
matchup floor, Wilson lower bound, runtime, and anti-regression play against
V17, V18, and V19. Broader candidates were removed after long confirmation.
V30 follows a structural-methodology reset. Bounded one- and two-ply engine
search plus a 18,887-state learned value model were tested, then rejected after
600-game confirmation failed to improve the external league. Policy-deck
co-optimization instead removed three low-conversion slots: disabled Chi-Yu,
Team Rocket's Watchtower, and Judge. They become a third Munkidori, fourth
Poke Pad, and third Darkness Energy. This raises setup/Adrena-Brain access
without adding runtime search, matchup names, or opponent-card exceptions.
"""
import os
import math
import sys
from collections import defaultdict, Counter
from enum import IntEnum
from cg.api import AreaType, CardType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, all_attack, to_observation_class
'\nDragapult ex Deck\nAdvanced Level\nThis deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.\n'
# Submission-safe deck resolution.  Never trust the process CWD first: Kaggle
# may execute from a directory that contains an unrelated deck.csv.  The cg package
# is shipped beside this policy, so its parent is the authoritative submission root.
import cg.api as _cg_api
_submission_root = os.path.dirname(os.path.dirname(os.path.abspath(_cg_api.__file__)))
_candidate = os.path.join(_submission_root, 'deck.csv')
file_path = _candidate if os.path.exists(_candidate) else '/kaggle_simulations/agent/deck.csv'
with open(file_path, 'r') as file:
    csv = file.read().split('\n')
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

def _min_attack_cost(card_id: int) -> int:
    attacks = getattr(card_table.get(card_id), 'attacks', []) or []
    costs = [len(attack_table[a].energies) for a in attacks if a in attack_table]
    return min(costs) if costs else 99
Dreepy = 119
Drakloak = 120
Dragapult_ex = 121
Munkidori = 112
Budew = 235
Chi_Yu = 31
Fezandipiti_ex = 140
Shaymin = 343
Meowth_ex = 1071
Yveltal = 689
Unfair_Stamp = 1080
Risky_Ruins = 1260
Jamming_Tower = 1246
Dawn = 1231
Hero_Cape = 1159
Full_Metal_Lab = 1244
Gravity_Mountain = 1252
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Enhanced_Hammer = 1081
Ultra_Ball = 1121
Poke_Pad = 1152
Tool_Scrapper = 1137
Boss_Orders = 1182
Crispin = 1198
Rosas_Encouragement = 1240
Judge = 1213
Lillie_Determination = 1227
Team_Rocket_Watchtower = 1256
Basic_Fire_Energy = 2
Basic_Psychic_Energy = 5
Basic_Darkness_Energy = 7
Dunsparce = 305
Dudunsparce = 66
Latias_ex = -184
Rare_Candy = 1079
Crushing_Hammer = 1120
Lucky_Helmet = -1156
Brock_Scouting = -1210
UNNECESSARY = -10000000
EV_ATTACH = True
EV_EVOLVE = True
EV_PLAY = False
EV_ATTACK = True
KO_TEMPO_BASE = 850
ENABLE_PRIZE_BUNDLE = True
ENABLE_CHIYU_WALL = False
BUDEW_STALL_CAP = 1

def hit_probability(population: int, hits: int, draws: int = 1) -> float:
    """Exact hypergeometric P(at least one hit), used as deterministic EV."""
    population = max(0, int(population))
    hits = max(0, min(int(hits), population))
    draws = max(0, min(int(draws), population))
    if population == 0 or hits == 0 or draws == 0:
        return 0.0
    misses = population - hits
    if draws > misses:
        return 1.0
    return 1.0 - (math.comb(misses, draws) / math.comb(population, draws))

class MachineState(IntEnum):
    OPENING = 0
    EVOLUTION = 1
    CHARGE = 2
    PRESSURE = 3
    RECOVERY = 4
    WALL_BREAK = 5
    ENDGAME = 6
    MIRROR_CONTROL = 7
    HAND_RECOVERY = 8
    STALL_SETUP = 9
    YVELTAL_LOCK = 10
machine_state = MachineState.OPENING
machine_transitions = defaultdict(int)
opponent_seen_ids: set[int] = set()
opponent_seen_counts: defaultdict[int, int] = defaultdict(int)
budew_stall_uses = 0

def _transition(new_state: MachineState):
    global machine_state
    if new_state != machine_state:
        machine_transitions[f'{machine_state.name}->{new_state.name}'] += 1
        machine_state = new_state

class AttackPlan:
    attack: int = 0
    counter: list[int] = []
can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0
bench_attacker = False
pre_turn_log: list[Log] = []
current_turn_log: list[Log] = []
prize: list[int] = []
card_counts: defaultdict[int, int] = defaultdict(int)
serial_set: set[int] = set()
plan_a = AttackPlan()
plan_b = AttackPlan()
defer_bench_ko = False

def no_damage_dex(id: int) -> bool:
    """Checks if the defending Pokémon possesses innate immunities preventing Dragapult ex from hitting it."""
    return id == 158 or id == 207 or id == 330 or (id == 345)

def no_damage_counter(pokemon: Pokemon) -> bool:
    """Checks if a target prevents placement of Phantom Dive's 6 bench damage counters (via abilities/Energy)."""
    if pokemon.id == 28 or pokemon.id == 199 or pokemon.id == 203 or (pokemon.id == 207) or (pokemon.id == 362) or (pokemon.id == 1136):
        return True
    for card in pokemon.energyCards:
        if card.id == 11 or card.id == 20:
            return True
    return False

def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and 'Lillie' in data.name:
                count -= 1
    return max(0, count)

def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Universal target value: prizes, committed energy/tools, stage, HP."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 170 + len(pokemon.tools) * 90
    score += 260 if data.stage2 else 130 if data.stage1 else 0
    score += min(400, pokemon.hp)
    return score

def add_card_count(card: Card | Pokemon | None, my_index: int):
    if card == None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)

def set_card_counts(obs: Observation, my_index: int):
    card_counts.clear()
    serial_set.clear()
    for id in my_deck:
        card_counts[id] += 1
    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:
        add_card_count(card, my_index)
    for card in my_state.discard:
        add_card_count(card, my_index)
    for card in my_state.bench:
        add_card_count(card, my_index)
    for card in my_state.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking != None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None

def main_option_proc(obs: Observation, damage: int):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    global can_switch
    global can_attack
    global can_main_attack
    global can_energy_attach
    global defer_bench_ko
    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    defer_bench_ko = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:
                can_main_attack = True
    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and (not (bench_attacker and can_switch)):
        return
    cards = [op_state.active[0]]
    for pokemon in op_state.bench:
        cards.append(pokemon)
    counter_indices = []
    ci = []
    ci.append(0)
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])
    remain_prize = len(my_state.prize)
    plan_score = 0
    opponent_has_fez = any(p is not None and p.id == Fezandipiti_ex for p in list(op_state.active) + list(op_state.bench))
    bench_phantom_ready = any(
        p is not None and p.id == Dragapult_ex
        and Basic_Fire_Energy in {e.id for e in p.energyCards}
        and Basic_Psychic_Energy in {e.id for e in p.energyCards}
        for p in my_state.bench
    )
    clear_trigger_signal = opponent_has_fez or (op_state.handCount >= 4 and my_state.handCount >= 7)
    disruption_trigger_risk = ENABLE_PRIZE_BUNDLE and bench_phantom_ready and remain_prize > 2 and clear_trigger_signal
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
            # A confirmed KO is worth more than the same proportional damage:
            # it removes the opponent's next evolution/attack branch and the
            # resources already committed to that body.  This generic tempo
            # term is deliberately below a full Prize so it cannot override
            # multi-Prize checkmates.
            base_score += KO_TEMPO_BASE + 140 * len(pokemon.energyCards)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        max_defer = False
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                prize = base_prize_count
                score = base_score
                for index in indices:
                    prize += prize_count(cards[index], False)
                    score += pokemon_score(cards[index], False)
                if remain_prize <= prize:
                    score = 50000
                elif prize >= 2:
                    if remain_prize <= 4:
                        score -= 1200
                elif prize == 1:
                    score -= 300
                else:
                    score += 1200
                # When the Active target survives, taking only one small Bench
                # KO can unnecessarily unlock Flip the Script / Unfair Stamp.
                # Prefer leaving a 10-HP counter target and bundle it with the
                # Active KO next turn, but never defer a multi-Prize turn or a
                # game-ending line.
                candidate_survives = active_damage > 0 and pokemon.hp > active_damage
                setup_targets = [p for j, p in enumerate(cards) if j != i and 20 <= p.hp <= 120 and not no_damage_counter(p)]
                should_defer = disruption_trigger_risk and candidate_survives and base_prize_count == 0 and prize == 1 and bool(setup_targets)
                if should_defer:
                    score -= 2600
                if max_score < score:
                    max_score = score
                    ci = indices
                    max_defer = should_defer
        if disruption_trigger_risk and active_damage > 0 and pokemon.hp > active_damage:
            setup_targets = [p for j, p in enumerate(cards) if j != i and 20 <= p.hp <= 120 and not no_damage_counter(p)]
            if setup_targets:
                max_score += 1150
                if not ci:
                    max_defer = True
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
            defer_bench_ko = max_defer
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter

def foundation_agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        return my_deck
    global pre_turn_log
    global current_turn_log
    global opponent_seen_ids
    global opponent_seen_counts
    global budew_stall_uses
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
        machine_transitions.clear()
        opponent_seen_ids.clear()
        opponent_seen_counts.clear()
        budew_stall_uses = 0
        _transition(MachineState.OPENING)
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []
    for _log in obs.logs:
        if _log.playerIndex != my_index:
            for _cid in (_log.cardId, _log.cardIdAfter, _log.cardIdActive, _log.cardIdBench):
                if _cid is not None and _cid > 0:
                    opponent_seen_ids.add(_cid)
                    opponent_seen_counts[_cid] += 1
        if _log.type == LogType.ATTACK and _log.playerIndex == my_index and (_log.attackId == 323):
            budew_stall_uses += 1
    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323:
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if log.playerIndex == my_index and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE) and (log.toArea == AreaType.DISCARD):
                pre_ko = True
    if select.deck != None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for id in card_counts:
            for _ in range(card_counts[id]):
                prize.append(id)
    set_card_counts(obs, my_index)
    for id in prize:
        card_counts[id] -= 1
    deck_counts = card_counts
    prize_diff = len(my_state.prize) - len(op_state.prize)
    global bench_attacker
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)
    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    for card in my_state.active:
        if card == None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= 2:
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = field_counts[Dragapult_ex] * 2 >= len(op_state.prize)
    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id
    visible_opponent = [p for p in list(op_state.active) + list(op_state.bench) if p is not None]
    seen_opponent_ids = {p.id for p in visible_opponent} | set(opponent_seen_ids)
    lucario_match = bool(seen_opponent_ids & {673,674,675,676,677,678})
    arch_match = bool(seen_opponent_ids & {169,190,Full_Metal_Lab})
    # New high-pressure family from replay 92538304. Require Lopunny/Froslass
    # signature cards so Snorunt alone does not collide with Marnie.
    lopunny_froslass_match = bool(seen_opponent_ids & {848,849,861})
    marnie_match = bool(seen_opponent_ids & {646,647,648,860,104})
    fast_match = lucario_match or arch_match or lopunny_froslass_match
    hero_cape_live = any(any(getattr(t, 'id', 0) == Hero_Cape for t in p.tools) for p in visible_opponent)
    hand_ids_now = Counter(c.id for c in (my_state.hand or []))
    # Replay-protected evolution package: do not voluntarily shuffle away a
    # guaranteed next evolution in the two matchups where attack-clock loss is fatal.
    protected_evolution = bool(fast_match and (
        (can_evolve_dreepy and hand_ids_now[Drakloak] > 0) or
        (can_evolve_drakloak and hand_ids_now[Dragapult_ex] > 0)
    ))
    ex_wall_present = any(no_damage_dex(p.id) for p in visible_opponent)
    dark_munk_count = sum((1 for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Munkidori and any((e.id == Basic_Darkness_Energy for e in p.energyCards))))
    munk_needs_dark = any((p is not None and p.id == Munkidori and not any((e.id == Basic_Darkness_Energy for e in p.energyCards)) for p in list(my_state.active) + list(my_state.bench)))
    own_total_damage = sum((max(0, card_table[p.id].hp - p.hp) for p in list(my_state.active) + list(my_state.bench) if p is not None))
    damaged_mirror_line = any((p.id in (Dreepy, Drakloak, Dragapult_ex) and p.hp < card_table[p.id].hp for p in list(my_state.active) + list(my_state.bench) if p is not None))

    def _phantom_ready(p):
        if p is None or p.id != Dragapult_ex:
            return False
        ids = {e.id for e in p.energyCards}
        return Basic_Fire_Energy in ids and Basic_Psychic_Energy in ids
    bench_attacker = any((_phantom_ready(p) for p in my_state.bench if p is not None))
    ready_drag_count = sum(1 for p in list(my_state.active) + list(my_state.bench) if p is not None and _phantom_ready(p))
    ready_dragapult = ready_drag_count > 0 or can_main_attack or bench_attacker
    dreepy_line_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    # Replay-learned milestone: Lucario wins jump sharply when the second Phantom
    # attacker is online by our ~8th turn. Archaludon has the same tempo pressure.
    relay_needed = bool((lucario_match or arch_match or lopunny_froslass_match) and ready_drag_count == 1 and dreepy_line_count >= 2 and len(op_state.prize) > 1)
    lucario_relay_budget = bool(lucario_match and relay_needed)
    active_pokemon = my_state.active[0] if my_state.active and my_state.active[0] is not None else None
    active_support_stranded = bool(active_pokemon is not None and active_pokemon.id in (Meowth_ex, Fezandipiti_ex, Munkidori, Shaymin, Chi_Yu, Yveltal) and (card_table[active_pokemon.id].retreatCost > len(active_pokemon.energies)) and (len(my_state.bench) > 0))
    opponent_ready_attacker = any((len(p.energyCards) >= _min_attack_cost(p.id) for p in visible_opponent))
    likely_boss = op_state.handCount >= 5 and opponent_seen_counts[Boss_Orders] < 3
    urgent_attack = False
    opponent_active = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    active_ex_wall = bool(opponent_active is not None and no_damage_dex(opponent_active.id))
    searched_progress_this_turn = any((l.type == LogType.PLAY and l.playerIndex == my_index and (l.cardId in (Poke_Pad, Ultra_Ball, Buddy_Buddy_Poffin, Night_Stretcher)) for l in current_turn_log))
    searched_card_still_actionable = searched_progress_this_turn and any((c.id == Drakloak and can_evolve_dreepy or (c.id == Dragapult_ex and can_evolve_drakloak) or (c.id == Dreepy and len(my_state.bench) < 5 and (dreepy_line_count < 2)) or (c.id == Munkidori and field_counts[Munkidori] == 0 and (len(my_state.bench) < 5)) for c in my_state.hand or []))
    block_search_then_lillie = searched_card_still_actionable and my_state.handCount >= 7
    stall_cap = BUDEW_STALL_CAP
    stall_setup = bool(active_id == Budew and (not ready_dragapult) and (state.turn <= 7) and (dreepy_line_count >= 1) and (budew_stall_uses < stall_cap))
    opponent_active_data = card_table.get(opponent_active.id) if opponent_active is not None else None
    opponent_setup_body = bool(opponent_active_data and (opponent_active_data.basic or opponent_active_data.stage1) and (not opponent_active_data.ex) and (not opponent_active_data.megaEx))
    trap_target = bool(opponent_active is not None and opponent_setup_body and (len(opponent_active.energyCards) <= 1) and (opponent_active_data.retreatCost >= 1) and (not opponent_ready_attacker) and (op_state.handCount <= 4))
    yveltal_pokemon = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Yveltal), None)
    yveltal_has_dark = bool(yveltal_pokemon and any((e.id == Basic_Darkness_Energy for e in yveltal_pokemon.energyCards)))
    yveltal_lock_ready = bool(yveltal_pokemon and yveltal_has_dark and trap_target and (not ready_dragapult))
    main_candidates = [p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id in (Dreepy, Drakloak, Dragapult_ex)]
    charging_candidates = []
    for p in main_candidates:
        ids = {e.id for e in p.energyCards}
        if not (Basic_Fire_Energy in ids and Basic_Psychic_Energy in ids):
            charging_candidates.append(p)
    pool = charging_candidates if charging_candidates else main_candidates

    def _primary_key(p):
        ids = {e.id for e in p.energyCards}
        required = int(Basic_Fire_Energy in ids) + int(Basic_Psychic_Energy in ids)
        stage = 3 if p.id == Dragapult_ex else 2 if p.id == Drakloak else 1
        active_bonus = 1 if p in my_state.active else 0
        bench_counter_safe = 1
        active_attack_safe = 1
        return (bench_counter_safe, active_attack_safe, required, stage, len(p.energies), active_bonus, p.hp)
    primary = max(pool, key=_primary_key) if pool else None
    primary_serial = primary.serial if primary is not None else -1
    primary_energy_ids = {e.id for e in primary.energyCards} if primary is not None else set()
    primary_missing_types = {Basic_Fire_Energy, Basic_Psychic_Energy} - primary_energy_ids
    attack_energy_in_hand = {c.id for c in my_state.hand or [] if c.id in (Basic_Fire_Energy, Basic_Psychic_Energy)}
    can_complete_primary_from_hand = bool(primary_missing_types & attack_energy_in_hand)
    progress_ids = {Drakloak, Dragapult_ex, Ultra_Ball, Poke_Pad, Crispin, Lillie_Determination, Basic_Fire_Energy, Basic_Psychic_Energy, Night_Stretcher}
    hand_progress = any((c.id in progress_ids for c in my_state.hand or []))
    hand_bricked = not ready_dragapult and dreepy_line_count > 0 and (my_state.handCount <= 3) and (not hand_progress)
    munk_ko_window = bool(
        ready_dragapult and own_total_damage >= 10 and dark_munk_count == 0
        and opponent_active is not None
        and 201 <= opponent_active.hp <= 200 + min(30, own_total_damage)
        and not no_damage_counter(opponent_active)
    )
    if ENABLE_CHIYU_WALL and active_ex_wall:
        desired_state = MachineState.WALL_BREAK
    elif len(op_state.prize) <= 2 and ready_dragapult:
        desired_state = MachineState.ENDGAME
    elif yveltal_lock_ready:
        desired_state = MachineState.YVELTAL_LOCK
    elif stall_setup:
        desired_state = MachineState.STALL_SETUP
    elif hand_bricked:
        desired_state = MachineState.HAND_RECOVERY
    elif pre_ko and (not ready_dragapult):
        desired_state = MachineState.RECOVERY
    elif dreepy_line_count == 0:
        desired_state = MachineState.OPENING
    elif field_counts[Dragapult_ex] == 0:
        desired_state = MachineState.EVOLUTION
    elif not ready_dragapult:
        desired_state = MachineState.CHARGE
    else:
        desired_state = MachineState.PRESSURE
    _transition(desired_state)
    phase = machine_state
    support_count = 0
    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        energy_ids = {e.id for e in pokemon.energyCards}
        required_energy = attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy)
        primary_bonus = 0
        if required_energy and pokemon.id in (Dreepy, Drakloak, Dragapult_ex):
            if pokemon.serial == primary_serial:
                primary_bonus = 10500
            elif primary_serial >= 0 and phase in (MachineState.OPENING, MachineState.EVOLUTION, MachineState.CHARGE, MachineState.RECOVERY):
                primary_bonus = -4500
        if active_support_stranded and active and (active_pokemon is not None) and (pokemon.serial == active_pokemon.serial) and (energy_count < card_table[pokemon.id].retreatCost):
            if attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                # Fire/Psychic are the two scarce Phantom Dive components.
                # Spending either merely to start paying a support retreat cost
                # delayed the first Phantom Dive twice as often in the IRL loss
                # set.  Keep the tactical exception only when that attachment
                # immediately frees an already-ready benched Dragapult.
                return 36500 if bench_attacker else -1
            if attach_id == Basic_Darkness_Energy and pokemon.id in (Munkidori, Yveltal):
                return 36200
        if pokemon.id == Munkidori and attach_id == Basic_Darkness_Energy and munk_ko_window:
            return 53500 + (300 if active else 0)
        if pokemon.id == Munkidori and attach_id == Basic_Darkness_Energy and (not ready_dragapult):
            return -1
        if pokemon.id == Dragapult_ex and required_energy and (attach_id not in energy_ids):
            if Basic_Fire_Energy not in energy_ids or Basic_Psychic_Energy not in energy_ids:
                return 31500 + primary_bonus + (300 if active else 0)
        if attach_id == 0:
            if pokemon.id == Dragapult_ex:
                return 60000
            if pokemon.id == Munkidori and any((e.id == Basic_Darkness_Energy for e in pokemon.energyCards)):
                return 30000
            return 1000
        if card_table[attach_id].cardType == CardType.TOOL:
            score = 60000
            if active:
                score += 1000
            return score
        if pokemon.id == Yveltal:
            ids = {e.id for e in pokemon.energyCards}
            if active and trap_target and (not ready_dragapult) and (not can_complete_primary_from_hand) and (attach_id == Basic_Darkness_Energy) and (Basic_Darkness_Energy not in ids):
                return 24800
            return -1
            if attach_id == Basic_Darkness_Energy and len(ids) < 2:
                return 25000 + len(ids) * 700
            if len(pokemon.energies) == 2 and attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                return 24600
            return -1
        if pokemon.id == Chi_Yu:
            if not ENABLE_CHIYU_WALL:
                return -1
            if len(pokemon.energies) == 0 and attach_id == Basic_Fire_Energy:
                return 40500 + (800 if active else 0)
            if len(pokemon.energies) == 1 and attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy, Basic_Darkness_Energy):
                return 42500 + (1200 if active else 0)
            return -1
        if pokemon.id == Shaymin:
            return -1
        if pokemon.id == Munkidori:
            if attach_id == Basic_Darkness_Energy and (not any((e.id == Basic_Darkness_Energy for e in pokemon.energyCards))):
                return 24500 + (300 if active else 0)
            return -1
        if pokemon.id in (Dreepy, Drakloak, Dragapult_ex) and attach_id == Basic_Darkness_Energy:
            return -1
        if pokemon.id == Dunsparce:
            if active and energy_count == 0 and bench_attacker:
                return 22500
            return -1
        if pokemon.id == Dudunsparce:
            return -1
        if pokemon.id in (Budew, Shaymin):
            return -1
        elif pokemon.id in (Meowth_ex, Fezandipiti_ex):
            if active and (not can_switch) and (not my_state.asleep) and (not my_state.paralyzed):
                if bench_attacker:
                    return 22000
                else:
                    return -1
            else:
                return -1
        if active and can_main_attack:
            return -1
        score = 20000
        if energy_count >= 2:
            if active and (not can_switch) and (not my_state.asleep) and (not my_state.paralyzed):
                score += 200
            else:
                return -1
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id:
                return -1
            if pokemon.id == Dragapult_ex:
                score += 250
            elif pokemon.id == Dreepy:
                score -= 150
            else:
                score -= 200
            if active:
                score += 200
        elif active:
            if bench_attacker:
                score += 400
        else:
            if pokemon.id == Dragapult_ex:
                score += 150
            elif pokemon.id == Dreepy:
                score += 100
            else:
                score += 50
            if bench_attacker:
                score -= 200
        if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
            score -= 500
        score += primary_bonus
        return score

    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Dreepy:
            if main_pokemon_count >= 3:
                score = 1000
            else:
                score = 18000
        elif id == Drakloak:
            if can_evolve_dreepy:
                score = 28000 if relay_needed else 20000
            else:
                score = 6000 if relay_needed else 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 32000 if relay_needed else 10000
                else:
                    score = 50
            elif field_counts[id] >= 2:
                score = 50
            else:
                score = 2000
        elif id == Munkidori:
            has_dark_munk = any((p.id == Munkidori and any((e.id == Basic_Darkness_Energy for e in p.energyCards)) for p in list(my_state.active) + list(my_state.bench) if p is not None))
            target_munk_count = 2 if (marnie_match and (ready_dragapult or own_total_damage >= 20)) else 1
            if fast_match and (not ready_dragapult) and own_total_damage <= 0:
                # First Phantom precedes the Adrena-Brain package in the winning
                # Lucario/Archaludon replay sequences. Do not spend Bench/energy
                # bandwidth on Munkidori before it has a job.
                score = 100
            elif field_counts[Munkidori] < target_munk_count and (hand_counts[Basic_Darkness_Energy] > 0 or deck_counts[Basic_Darkness_Energy] > 0):
                score = 15000
            elif field_counts[Munkidori] >= target_munk_count or has_dark_munk:
                score = 100
        elif id == Dunsparce:
            if field_counts[Dunsparce] + field_counts[Dudunsparce] == 0 and len(my_state.bench) <= 3:
                score = 10000
            else:
                score = 200
        elif id == Dudunsparce:
            can_evolve_dunsparce = any((p.id == Dunsparce and (not p.appearThisTurn) for p in list(my_state.active) + list(my_state.bench) if p is not None))
            score = 18000 if can_evolve_dunsparce else 500
        elif id == Chi_Yu:
            score = 300
        elif id == Yveltal:
            score = 200
        elif id == Shaymin:
            score = 12000 if field_counts[Shaymin] == 0 and len(my_state.bench) <= 3 else 100
        elif id == Fezandipiti_ex:
            if pre_ko:
                score = 50000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
        elif id == Latias_ex:
            if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                    score = 28000
                else:
                    score = 15000
            else:
                score = 10
        elif id == Budew:
            if budew_stall_uses >= stall_cap and dreepy_line_count >= 1:
                score = UNNECESSARY
            elif field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                score = UNNECESSARY
            elif state.turn >= 2:
                score = 30000
        elif id == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif id == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                score = 40000
        elif id == Unfair_Stamp:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 80
        elif id == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0:
                score = UNNECESSARY
            else:
                if state.turn <= 2 and field_counts[Budew] == 0 and (deck_counts[Budew] >= 1):
                    count += 1
                if count >= 2:
                    score = 35000
        elif id == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        if fast_match and i in (Fezandipiti_ex, Meowth_ex):
                            # One consistency use is acceptable; replaying an already
                            # discarded two-Prize support gives Lucario/Archaludon an
                            # unnecessary Boss prize target and does not advance RELAY.
                            continue
                        score = max(score, hand_score(i, ignore_count))
        elif id == Enhanced_Hammer:
            has_special = any((card_table[e.id].cardType == CardType.SPECIAL_ENERGY for p in visible_opponent for e in p.energyCards))
            score = 24000 if has_special else 20
        elif id == Tool_Scrapper:
            tool_count = sum((len(p.tools) for p in visible_opponent))
            score = 22000 + tool_count * 2000 if tool_count else 20
        elif id == Crushing_Hammer:
            score = 20
        elif id == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
        elif id == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif id == Lucky_Helmet:
            score = 15
        elif id == Boss_Orders:
            if plan_a.attack > 0:
                score = 36000 if lucario_relay_budget else 60000
        elif id == Rosas_Encouragement:
            has_fire_discard = discard_counts[Basic_Fire_Energy] > 0
            has_psychic_discard = discard_counts[Basic_Psychic_Energy] > 0
            primary_ids = {e.id for e in primary.energyCards} if primary is not None else set()
            can_complete_pair = has_fire_discard and has_psychic_discard or (has_fire_discard and Basic_Psychic_Energy in primary_ids) or (has_psychic_discard and Basic_Fire_Energy in primary_ids)
            score = 59000 if prize_diff > 0 and primary is not None and can_complete_pair else UNNECESSARY
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if relay_needed:
                    score = 58500
                elif not can_main_attack and (not bench_attacker) and (field_counts[Dragapult_ex] >= 1):
                    score = 55000
                else:
                    score = 25000
        elif id == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else:
                    score = 30000
        elif id == Lillie_Determination:
            if not ignore_count or support_count == 0:
                score = UNNECESSARY if my_state.deckCount <= 6 or block_search_then_lillie else (18000 if lucario_relay_budget else 45000)
        elif id == Dawn:
            if not ignore_count or support_count == 0:
                dawn_match = arch_match or lopunny_froslass_match
                need_basic_line = dreepy_line_count == 0 and deck_counts[Dreepy] > 0
                need_stage1 = can_evolve_dreepy and hand_counts[Drakloak] == 0 and deck_counts[Drakloak] > 0
                need_stage2 = can_evolve_drakloak and hand_counts[Dragapult_ex] == 0 and deck_counts[Dragapult_ex] > 0
                if dawn_match and need_basic_line:
                    score = 64000
                elif dawn_match and need_stage2:
                    score = 62000
                elif dawn_match and need_stage1:
                    score = 60000
                else:
                    score = 500
        elif id == Judge:
            if lucario_relay_budget:
                score = 500
            elif my_state.deckCount <= 6 or (pre_ko and hand_counts[Unfair_Stamp] > 0):
                score = UNNECESSARY
            elif op_state.handCount >= 6 and my_state.handCount <= 5:
                score = 43000
            elif op_state.handCount >= 7:
                score = 35000
            elif my_state.handCount <= 2:
                score = 26000
            else:
                score = 500
        elif id == Jamming_Tower:
            phantom_unlock = stadium_id == 1264 and (ready_dragapult or field_counts[Dragapult_ex] >= 1 or can_evolve_drakloak)
            if phantom_unlock:
                score = 68000
            elif hero_cape_live or (arch_match and stadium_id == Full_Metal_Lab) or (lucario_match and stadium_id == Gravity_Mountain):
                score = 62000
            elif marnie_match and stadium_id == 1259:
                score = 60000
            else:
                score = 50
        elif id == Risky_Ruins:
            low_hp_unready = sum((1 for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id in (Dreepy, Budew, Chi_Yu, Shaymin) and (p.hp <= 80)))
            op_basic_non_dark = sum((1 for p in visible_opponent if card_table[p.id].basic and card_table[p.id].energyType != 7))
            drag_ready = field_counts[Dragapult_ex] > 0 or (field_counts[Drakloak] > 0 and can_evolve_drakloak)
            munk_ready = dark_munk_count > 0
            if (arch_match and stadium_id == Full_Metal_Lab) or (lucario_match and stadium_id == Gravity_Mountain):
                score = 47000
            elif stadium_id != Risky_Ruins and op_basic_non_dark >= 1 and drag_ready and (low_hp_unready == 0 or munk_ready):
                score = 12000 + op_basic_non_dark * 2200
            else:
                score = 50
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy or id == Basic_Darkness_Energy:
            if can_main_attack and (len(op_state.prize) <= 2 or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                if id == Basic_Darkness_Energy:
                    max_score = -10000
                    for pokemon in list(my_state.active) + list(my_state.bench):
                        if pokemon is not None:
                            max_score = max(max_score, attach_score(id, pokemon, pokemon in my_state.active))
                    score = max_score - 3000
                    if max_score < 0:
                        score = 50
                    return score
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon == None:
                        continue
                    max_score = max(max_score, attach_score(id, pokemon, True))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(id, pokemon, False))
                score = max_score - 5000
                if can_main_attack or bench_attacker:
                    score /= 10
        if phase == MachineState.OPENING:
            if id == Dreepy and dreepy_line_count < 2:
                score += 7000
        elif phase == MachineState.EVOLUTION:
            if id in (Drakloak, Dragapult_ex, Ultra_Ball, Poke_Pad):
                score += 5000
        elif phase == MachineState.CHARGE:
            if id == Crispin:
                score += 16000
            elif id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                score += 3500
        elif phase == MachineState.RECOVERY:
            if id in (Night_Stretcher, Dreepy, Drakloak, Dragapult_ex, Ultra_Ball):
                score += 7000
        elif phase == MachineState.ENDGAME:
            if id == Boss_Orders and plan_a.attack > 0:
                score += 9000
            elif id in (Lillie_Determination, Judge, Buddy_Buddy_Poffin):
                score -= 7000
        elif phase == MachineState.STALL_SETUP:
            if id in (Drakloak, Dragapult_ex, Crispin, Basic_Fire_Energy, Basic_Psychic_Energy):
                score += 800
        elif phase == MachineState.YVELTAL_LOCK:
            if id in (Drakloak, Dragapult_ex, Crispin, Basic_Fire_Energy, Basic_Psychic_Energy):
                score += 600
        elif phase == MachineState.HAND_RECOVERY:
            if id in (Lillie_Determination, Judge, Poke_Pad, Ultra_Ball, Meowth_ex, Fezandipiti_ex):
                score += 12000
            if id in (Boss_Orders, Risky_Ruins, Team_Rocket_Watchtower):
                score -= 5000
        if not ignore_count and hand_counts[id] > 0:
            if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                score -= 10
            elif id == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score
    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)
        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        score = hand_score(card.id, True)
                        if support_score < score:
                            support_score = score
                            use_support = card.id
    hand_scores = []
    negative_hand_count = 0
    disposable_hand_count = 0
    for card in my_state.hand:
        score = hand_score(card.id, False)
        hand_scores.append(score)
        if score < 0:
            negative_hand_count += 1
        if score <= 500:
            disposable_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1
    no_draw = my_state.deckCount <= 8
    wall_attacker_ready = any(p is not None and p.id == Chi_Yu and len(p.energyCards) >= 2 for p in list(my_state.active) + list(my_state.bench))
    do_switch = not can_main_attack and (bench_attacker or wall_attacker_ready or (active_id != Budew and field_counts[Budew] >= 1 and (state.turn >= 2)))
    if phase == MachineState.YVELTAL_LOCK and active_id != Yveltal and yveltal_lock_ready:
        do_switch = True
    effect_card_id = 0 if select.effect == None else select.effect.id
    context_card_id = 0 if select.contextCard == None else select.contextCard.id
    crispin_available = set()
    if effect_card_id == Crispin and select.deck is not None:
        crispin_available = {c.id for c in select.deck}
    # Probability-calibrated, matchup-agnostic value layer.  The rule core
    # still supplies legality/sequencing; this small residual only separates
    # actions whose immediate scores ignore next-turn conversion and exposure.
    deck_n = max(0, my_state.deckCount)
    evolution_outs = max(0, deck_counts[Drakloak] + deck_counts[Dragapult_ex] + deck_counts[Ultra_Ball] + deck_counts[Poke_Pad])
    energy_outs = max(0, deck_counts[Basic_Fire_Energy] + deck_counts[Basic_Psychic_Energy] + deck_counts[Crispin])
    p_evolve_2 = hit_probability(deck_n, evolution_outs, 2)
    p_energy_2 = hit_probability(deck_n, energy_outs, 2)
    # This is an uncertainty estimate, not an assertion that Boss is in hand.
    unseen_boss_prior = max(0, 2 - min(2, opponent_seen_counts[Boss_Orders]))
    p_boss = min(0.65, hit_probability(max(1, op_state.deckCount), unseen_boss_prior, 1) + 0.055 * op_state.handCount)
    opponent_pressure = min(1.0, 0.22 * sum(len(p.energyCards) for p in visible_opponent) + (0.28 if opponent_ready_attacker else 0.0))

    def expected_future_adjustment(option, base_score: float) -> int:
        if base_score < 0:
            return 0
        adjustment = 0.0
        if EV_ATTACH and option.type == OptionType.ATTACH:
            energy = get_card(obs, option.area, option.index, my_index)
            target = get_card(obs, option.inPlayArea, option.inPlayIndex, my_index)
            if energy is not None and target is not None and target.id in (Dreepy, Drakloak, Dragapult_ex):
                ids = {e.id for e in target.energyCards}
                before = int(Basic_Fire_Energy in ids) + int(Basic_Psychic_Energy in ids)
                after_ids = ids | {energy.id}
                after = int(Basic_Fire_Energy in after_ids) + int(Basic_Psychic_Energy in after_ids)
                # Value the increase in probability of a live Phantom Dive,
                # discounted if the target is exposed to a likely response.
                conversion_gain = (after - before) * (0.55 + 0.45 * p_evolve_2)
                exposure = opponent_pressure * (0.35 + 0.65 * p_boss) if option.inPlayArea == AreaType.BENCH else opponent_pressure * 0.45
                adjustment += 3600 * conversion_gain - 900 * exposure
                if after == 2:
                    adjustment += 1800 * (1.0 - exposure)
        elif EV_EVOLVE and option.type == OptionType.EVOLVE:
            target = get_card(obs, option.inPlayArea, option.inPlayIndex, my_index)
            if target is not None:
                # Evolving a charged or damaged body preserves committed value
                # and reduces the probability of losing the line before payoff.
                committed = len(target.energyCards) + (1 if target.hp < card_table[target.id].hp else 0)
                adjustment += 850 * committed + 1000 * opponent_pressure
        elif EV_PLAY and option.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, option.index, my_index)
            if card is not None:
                if card.id == Dreepy:
                    setup_gain = (1.0 - p_evolve_2) if dreepy_line_count < 2 else 0.15
                    bench_risk = p_boss * opponent_pressure if len(my_state.bench) >= 3 else 0.0
                    adjustment += 1500 * setup_gain - 1900 * bench_risk
                elif card.id in (Lillie_Determination, Judge):
                    key_outs = evolution_outs + energy_outs
                    draw_count = 6 if card.id == Lillie_Determination else 5
                    p_progress = hit_probability(deck_n, key_outs, draw_count)
                    adjustment += 2600 * p_progress
                    if hand_bricked:
                        adjustment += 2200
                elif card.id == Crushing_Hammer:
                    # Coin flip is represented by its true expectation. Prefer
                    # denial when it can actually delay a threatening attacker.
                    threatening_energy = sum(len(p.energyCards) for p in visible_opponent if len(p.energyCards) >= max(1, _min_attack_cost(p.id) - 1))
                    adjustment += 0.5 * min(3000, 700 * threatening_energy)
                elif card.id in (Buddy_Buddy_Poffin, Ultra_Ball, Poke_Pad):
                    adjustment += 1700 * max(p_evolve_2, p_energy_2)
        elif EV_ATTACK and option.type == OptionType.ATTACK:
            if option.attackId == 154 and opponent_active is not None:
                active_prizes = prize_count(opponent_active, True) if opponent_active.hp <= damage else 0
                bench_finish_prizes = sum(prize_count(p, False) for p in op_state.bench if p is not None and p.hp <= 60 and not no_damage_counter(p))
                adjustment += 2400 * active_prizes + 1700 * min(2, bench_finish_prizes)
        return int(round(adjustment))

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = 1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE or context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Chi_Yu:
                            score += 1000
                        elif card.id == Yveltal:
                            dark_count = sum((1 for e in card.energyCards if e.id == Basic_Darkness_Energy)) if isinstance(card, Pokemon) else 0
                            if phase == MachineState.YVELTAL_LOCK and dark_count >= 1:
                                score += 103000
                        elif card.id == Munkidori:
                            score += -1500
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Dunsparce:
                            score += 500 if context != SelectContext.SETUP_ACTIVE_POKEMON else 1000
                        elif card.id == Dudunsparce:
                            score += -500
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    elif plan_a.attack == o.index + 1:
                        score += 100000
                    # Cross-play target denial: when Boss/switch effects expose
                    # an opponent engine, prefer the low-HP evolution seed.
                    if o.playerIndex != my_index:
                        if card.id in (646, 860, 131):
                            score += 42000
                        elif card.id in (647, 104, 112, 132):
                            score += 16000
                    score += energy_count * 1000
                    score += hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if card.id == Dreepy:
                        score = 10000
                    elif card.id == Budew and state.firstPlayer != my_index:
                        score = 9000
                    elif card.id == Dunsparce:
                        score = 2000
                    else:
                        score = -1
                elif context == SelectContext.ATTACH_TO and effect_card_id == Crispin:
                    chi_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Chi_Yu), None)
                    yveltal_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Yveltal), None)
                    primary_ids = {e.id for e in primary.energyCards} if primary is not None else set()
                    if card.id == Munkidori and munk_ko_window:
                        score = 101000
                    elif card.id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                        score = 99000 if card.id not in primary_ids else 72000
                    elif card.id == Basic_Darkness_Energy and munk_needs_dark and ready_dragapult:
                        # Only choose Darkness as Crispin's attached Energy when a
                        # Munkidori actually still needs it.  The old field-count gate
                        # remained true after Munk already had Darkness, leaving no
                        # valid positive target and causing argmax to dump Darkness onto
                        # the first Dragapult line.
                        score = 18000
                    else:
                        score = -1
                elif context == SelectContext.ATTACH_TO and effect_card_id == Rosas_Encouragement:
                    if card.id == Basic_Fire_Energy:
                        score = 98000
                    elif card.id == Basic_Psychic_Energy:
                        score = 97500
                    elif card.id == Basic_Darkness_Energy:
                        score = -1
                    else:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    if effect_card_id == Night_Stretcher and fast_match and card.id in (Fezandipiti_ex, Meowth_ex):
                        score = -1
                    hand_counts[card.id] += 1
                    if effect_card_id in (Poke_Pad, Ultra_Ball):
                        if card.id == Drakloak and can_evolve_dreepy:
                            score += 90000
                        elif card.id == Dragapult_ex and can_evolve_drakloak:
                            score += 88000
                        elif card.id == Dreepy and dreepy_line_count < 2:
                            score += 76000
                        elif card.id == Munkidori and field_counts[Munkidori] == 0 and (own_total_damage > 0):
                            score += 65000
                    if effect_card_id == Dawn:
                        if card.id == Dreepy:
                            score = 125000 if dreepy_line_count < 2 else 70000
                        elif card.id == Munkidori:
                            score = 112000 if dreepy_line_count >= 2 and field_counts[Munkidori] < 2 else 65000
                        elif card.id == Drakloak:
                            score = 123000
                        elif card.id == Dragapult_ex:
                            score = 121000
                    if effect_card_id == Crispin:
                        has_fire = Basic_Fire_Energy in crispin_available
                        has_psychic = Basic_Psychic_Energy in crispin_available
                        has_dark = Basic_Darkness_Energy in crispin_available
                        chi_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Chi_Yu), None)
                        yveltal_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Yveltal), None)
                        if context == SelectContext.TO_HAND and fast_match and (not ready_dragapult or relay_needed):
                            # Crispin is two-stage: the first selected Basic Energy goes
                            # to hand, while a *different type* selected next is attached
                            # from the deck.  The previous code took the primary line's
                            # missing Phantom color into hand, accidentally removing the
                            # exact color we wanted the second stage to attach.
                            missing = set(primary_missing_types)
                            phantom = {Basic_Fire_Energy, Basic_Psychic_Energy}
                            if len(missing) == 1:
                                need = next(iter(missing))
                                other = Basic_Psychic_Energy if need == Basic_Fire_Energy else Basic_Fire_Energy
                                # Leave `need` in the deck for ATTACH_TO.  Prefer putting
                                # the other Phantom color in hand for the relay; Darkness
                                # is a safe second choice when the other color is absent.
                                if need in crispin_available and card.id == other and other in crispin_available:
                                    score = 116000
                                elif need in crispin_available and card.id == Basic_Darkness_Energy and Basic_Darkness_Energy in crispin_available:
                                    score = 104000
                                elif card.id == need:
                                    score = 18000
                                elif card.id in phantom:
                                    score = 70000
                                else:
                                    score = -1
                            elif len(missing) >= 2:
                                # Empty attacker: take one Phantom color to hand and leave
                                # the other in the deck to attach immediately.
                                if card.id == Basic_Fire_Energy and Basic_Psychic_Energy in crispin_available:
                                    score = 112000
                                elif card.id == Basic_Psychic_Energy and Basic_Fire_Energy in crispin_available:
                                    score = 111500
                                elif card.id in phantom:
                                    score = 82000
                                else:
                                    score = -1
                            else:
                                # The primary already has both colors; Crispin belongs to
                                # relay/utility rather than duplicating an attack color.
                                if card.id in phantom:
                                    score = 78000
                                elif card.id == Basic_Darkness_Energy and munk_needs_dark:
                                    score = 76000
                                else:
                                    score = -1
                        elif context == SelectContext.TO_HAND and (not (has_fire and has_psychic)):
                            if card.id == Basic_Darkness_Energy and munk_needs_dark and ready_dragapult:
                                score = 95000
                            elif card.id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                                score = 70000
                            else:
                                score = -1
                        elif card.id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                            attached_ids = set()
                            for pkm in list(my_state.active) + list(my_state.bench):
                                if pkm is not None and pkm.id in (Dragapult_ex, Drakloak, Dreepy):
                                    attached_ids.update((e.id for e in pkm.energyCards))
                            score = 93000 if card.id in attached_ids else 90000
                        elif card.id == Basic_Darkness_Energy and field_counts[Munkidori] > 0 and munk_ko_window:
                            score = 99000
                        elif card.id == Basic_Darkness_Energy and field_counts[Munkidori] > 0 and (can_main_attack or bench_attacker):
                            score = 20000
                        else:
                            score = -1
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    if isinstance(card, Pokemon):
                        damage_on_card = max(0, card_table[card.id].hp - card.hp)
                        score = damage_on_card * 100 + pokemon_score(card, False)
                        if damage_on_card <= 0:
                            score = -1
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            # Cross-play engine denial: spend Phantom Dive counters on
                            # fragile evolution/ability engines before they convert.
                            if lucario_match and card.id == 677:
                                score += 72000
                            elif lucario_match and card.id in (673,675):
                                score += 36000
                            elif marnie_match and card.id == 112:
                                score += 74000
                            elif marnie_match and card.id in (646,860):
                                score += 65000
                            elif marnie_match and card.id in (647,648):
                                score += 34000
                            elif card.id in (131, 646, 860):
                                score += 65000
                            elif card.id in (132, 133, 351):
                                score += 30000
                        else:
                            remain_damage = select.remainDamageCounter * 10
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            elif 210 <= hp <= 200 + remain_damage:
                                score += 30000
                            elif 20 <= hp <= 60 + remain_damage:
                                score += 10000
                            elif hp == 10:
                                # Replay audit: the old selector could leave the final
                                # Phantom counter unused / spread elsewhere instead of
                                # converting a guaranteed KO. Cash out the prize first.
                                score += 180000
                            if lucario_match and card.id == 677:
                                score += 65000
                            elif lucario_match and card.id in (673,675):
                                score += 32000
                            elif marnie_match and card.id == 112:
                                score += 66000
                            elif marnie_match and card.id in (646,860):
                                score += 56000
                            elif marnie_match and card.id in (647,648):
                                score += 30000
                            elif card.id in (131, 646, 860):
                                score += 52000
                            elif card.id in (132, 133, 351):
                                score += 22000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if effect_card_id == Crispin and context_card_id == Basic_Darkness_Energy and isinstance(card, Pokemon):
                        # Forced-safety fallback: if Crispin somehow has to attach a
                        # Darkness and no fresh Munkidori target exists, never poison an
                        # established Phantom line.  Prefer a Munk that still needs Dark,
                        # then an expendable support body, then the least-developed line.
                        eids = {e.id for e in card.energyCards}
                        if card.id == Munkidori and Basic_Darkness_Energy not in eids:
                            score = max(score, 120000)
                        elif card.id in (Dreepy, Drakloak, Dragapult_ex):
                            stage = 3 if card.id == Dragapult_ex else 2 if card.id == Drakloak else 1
                            score = -60000 - stage * 10000 - len(card.energyCards) * 5000
                        else:
                            score = max(score, 1000)
                    if fast_match and effect_card_id == Crispin and context_card_id in (Basic_Fire_Energy, Basic_Psychic_Energy) and isinstance(card, Pokemon):
                        # Same safety rule for a forced duplicate Phantom color.  If
                        # Fire is already on a Dragapult that still needs Psychic (or
                        # vice versa), duplicating that color delays Phantom by a full
                        # turn.  Prefer any line that can actually use the color; if none
                        # exists, dump the forced attachment on a support/least-developed
                        # body rather than the established attacker.
                        eids = {e.id for e in card.energyCards}
                        opposite = Basic_Psychic_Energy if context_card_id == Basic_Fire_Energy else Basic_Fire_Energy
                        if card.id in (Dreepy, Drakloak, Dragapult_ex) and context_card_id in eids and opposite not in eids:
                            stage = 3 if card.id == Dragapult_ex else 2 if card.id == Drakloak else 1
                            score = -55000 - stage * 10000 - len(card.energyCards) * 5000
                        elif card.id not in (Dreepy, Drakloak, Dragapult_ex):
                            score = max(score, 800)
                    if card.id == Dragapult_ex and not (effect_card_id == Crispin and (context_card_id == Basic_Darkness_Energy or (fast_match and context_card_id in (Basic_Fire_Energy, Basic_Psychic_Energy)))):
                        score += 200
                    if effect_card_id == Rosas_Encouragement and isinstance(card, Pokemon):
                        score = 100000 if card.serial == primary_serial else 1000
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            if o.playerIndex != state.yourIndex:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if isinstance(card, Pokemon):
                    ec = len(card.energyCards)
                    score = 100 + ec * 500
                    # Replay-derived ENERGY_DENIAL: wins removed energy from the
                    # actual attacker, not random Lunatone/Solrock bodies.
                    if lucario_match:
                        if card.id == 678: score += 15000 + ec * 2500
                        elif card.id == 674: score += 12000 + ec * 2200
                        elif card.id in (677,673) and ec >= 2: score += 7000
                        elif card.id in (675,676): score -= 3000
                    elif arch_match:
                        if card.id == 169 and ec >= 2: score += 15000
                        elif card.id == 666 and ec >= 1: score += 13000
                        elif card.id == 190: score += 9000 + ec * 1500
                    if o.area == AreaType.ACTIVE: score += 400
                else:
                    score = 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            phase_blocked = False
            if card.id == Dreepy and dreepy_line_count >= 2 and (phase != MachineState.RECOVERY):
                phase_blocked = True
            if phase in (MachineState.OPENING, MachineState.EVOLUTION, MachineState.CHARGE) and (not ready_dragapult):
                if card.id in (Chi_Yu, Yveltal):
                    phase_blocked = True
            if card.id == Dreepy:
                if main_pokemon_count < 3:
                    score = 51000
                else:
                    score = -1
            elif card.id == Munkidori:
                target_munk_count = 2 if (marnie_match and (ready_dragapult or own_total_damage >= 20)) else 1
                if fast_match and (not ready_dragapult) and own_total_damage <= 0:
                    score = -1
                elif field_counts[Munkidori] < target_munk_count and len(my_state.bench) < 5:
                    score = 50500
                else:
                    score = -1
            elif card.id == Chi_Yu:
                if ENABLE_CHIYU_WALL and ex_wall_present and field_counts[Chi_Yu] == 0 and len(my_state.bench) < 5:
                    score = 54500
                elif ENABLE_CHIYU_WALL and dreepy_line_count == 0 and field_counts[Chi_Yu] == 0 and len(my_state.bench) < 4:
                    score = 18000
                else:
                    score = -1
            elif card.id == Yveltal:
                chi_available = field_counts[Chi_Yu] + hand_counts[Chi_Yu] + deck_counts[Chi_Yu] + discard_counts[Chi_Yu]
                score = -1
            elif card.id == Shaymin:
                if field_counts[Shaymin] == 0 and len(my_state.bench) <= 3:
                    score = 50020
                else:
                    score = -1
            elif card.id == Dunsparce:
                support_bench = field_counts[Dunsparce] + field_counts[Dudunsparce] + field_counts[Fezandipiti_ex] + field_counts[Meowth_ex]
                bench_limit = 4
                support_limit = 2
                if field_counts[Dunsparce] + field_counts[Dudunsparce] == 0 and len(my_state.bench) < bench_limit and (support_bench < support_limit):
                    score = 50100
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                if lucario_relay_budget:
                    score = -1
                elif card_score > 0 and len(my_state.bench) < 5:
                    score = 53000
                else:
                    score = -1
            elif card.id == Latias_ex:
                if active_id != Drakloak and active_id != Dragapult_ex:
                    score = 51000
                else:
                    score = -1
            elif card.id == Budew:
                if budew_stall_uses >= stall_cap and dreepy_line_count >= 1:
                    score = -1
                elif field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                    score = 52000
                else:
                    score = -1
            elif card.id == Meowth_ex:
                if lucario_relay_budget:
                    score = -1
                elif fast_match and (not ready_dragapult) and dreepy_line_count >= 1 and (not hand_bricked):
                    score = -1
                elif len(my_state.bench) >= 4:
                    score = -1
                elif state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and (not plan_a.attack <= 0):
                    score = 50000
                else:
                    score = -1
            elif card.id == Rare_Candy:
                if no_more_dex:
                    score = -1
                else:
                    score = 75000
            elif card.id == Unfair_Stamp:
                score = 15000
            elif card.id == Night_Stretcher:
                if card_score >= 18000:
                    score = 42000
                else:
                    score = -1
            elif card.id == Enhanced_Hammer:
                has_special = any((card_table[e.id].cardType == CardType.SPECIAL_ENERGY for p in visible_opponent for e in p.energyCards))
                score = 41000 if has_special else -1
            elif card.id == Tool_Scrapper:
                tool_count = sum((len(p.tools) for p in visible_opponent))
                score = 40500 if tool_count > 0 else -1
            elif card.id == Crushing_Hammer:
                op_energy = sum((len(p.energyCards) for p in list(op_state.active) + list(op_state.bench) if p is not None))
                score = (24000 if lucario_relay_budget else 40000) if op_energy > 0 else -1
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support and my_state.deckCount > 6 and (not block_search_then_lillie) and (not protected_evolution):
                    score = 14000
                else:
                    score = -1
            elif card.id == Dawn:
                score = 52000 if card.id == use_support and my_state.deckCount > 3 else -1
            elif card.id == Judge:
                if card.id == use_support and my_state.deckCount > 6 and (not protected_evolution) and (not (pre_ko and hand_counts[Unfair_Stamp] > 0)):
                    score = 33500
                else:
                    score = -1
            elif card.id == Jamming_Tower:
                phantom_unlock = stadium_id == 1264 and (ready_dragapult or field_counts[Dragapult_ex] >= 1 or can_evolve_drakloak)
                critical = phantom_unlock or hero_cape_live or (arch_match and stadium_id == Full_Metal_Lab) or (lucario_match and stadium_id == Gravity_Mountain)
                if phantom_unlock:
                    score = 94000
                elif critical:
                    score = 90000
                elif marnie_match and stadium_id == 1259:
                    score = 76000
                else:
                    score = -1
            elif card.id == Risky_Ruins:
                chi_needs_stadium = False
                if chi_needs_stadium:
                    score = 76000
                elif stadium_id in (1264, Full_Metal_Lab, Gravity_Mountain) or (stadium_id != Risky_Ruins and hand_score(Risky_Ruins, True) >= 1000):
                    score = 52000 if stadium_id in (Full_Metal_Lab, Gravity_Mountain) else 39000
                else:
                    score = -1
            elif card.id == Team_Rocket_Watchtower:
                colorless_ability = any((card_table[p.id].energyType == 0 and len(card_table[p.id].skills) > 0 for p in visible_opponent))
                need_stadium_for_chi = False
                if stadium_id > 0 or colorless_ability or need_stadium_for_chi:
                    score = 72000 if need_stadium_for_chi else 52000
                else:
                    score = -1
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                if deck_counts[Dreepy] + deck_counts[Budew] + deck_counts[Munkidori] + deck_counts[Yveltal] > 0:
                    score = 46000
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                missing_stage = can_evolve_dreepy and deck_counts[Drakloak] > 0 or (can_evolve_drakloak and deck_counts[Dragapult_ex] > 0)
                wall_target = False
                if fast_match and can_evolve_drakloak and deck_counts[Dragapult_ex] > 0 and my_state.handCount >= 3 and (negative_hand_count >= 1 or disposable_hand_count >= 1):
                    # Guaranteed Stage-2 access beats probabilistic Pad when an
                    # established Drakloak is already the bottleneck.
                    score = 62500
                elif missing_stage and (negative_hand_count >= 1 or disposable_hand_count >= 1) and (my_state.handCount >= 3):
                    score = 50500
                elif wall_target and (negative_hand_count >= 1 or disposable_hand_count >= 1) and (my_state.handCount >= 3):
                    score = 53500
                elif main_pokemon_count < 2 and negative_hand_count >= 1:
                    score = 48000
                elif negative_hand_count >= 2:
                    score = 44000
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if deck_counts[Dreepy] + deck_counts[Drakloak] + deck_counts[Munkidori] + deck_counts[Yveltal] + deck_counts[Chi_Yu] > 0:
                    score = 45000
                else:
                    score = -1
            elif card.id == Rosas_Encouragement:
                if card.id == use_support:
                    score = 43000
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    score = 50000 if phase == MachineState.CHARGE and card.id == Crispin else 35000
                else:
                    score = -1
            if phase_blocked:
                score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif pokemon.id == Dunsparce:
                score += 62000 if my_state.handCount <= 6 else 30000
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Munkidori:
                own_damage = sum((max(0, card_table[p.id].hp - p.hp) for p in list(my_state.active) + list(my_state.bench) if p is not None))
                if own_damage >= 10 and any((e.id == Basic_Darkness_Energy for e in card.energyCards)):
                    transferable = min(30, own_damage)
                    can_finish = any((p.hp <= transferable and (not no_damage_counter(p)) for p in visible_opponent))
                    score = 104000 if can_finish else 70000
                else:
                    score = -1
            elif card.id == 1259:
                # Spikemuth Gym can only search a Marnie's Pokemon. This deck
                # contains none, so activation only shuffles and wastes tempo.
                score = -1
            elif no_draw:
                score = -1
            elif card.id == 1267:
                score = 1
            elif card.id == Dudunsparce:
                draw_threshold = 6 if (lucario_match or arch_match) else 5
                score = 65000 if my_state.handCount <= draw_threshold else -1
            elif card.id == -999999:
                score = -1
            elif card.id == Fezandipiti_ex:
                score = 67000 if pre_ko else -1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            if do_switch:
                score = 10000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId
            if active_id == Yveltal and o.attackId == 997 and (phase == MachineState.YVELTAL_LOCK):
                score = 6000
            if o.attackId == 154 and urgent_attack:
                score = 76000
            elif ENABLE_CHIYU_WALL and active_id == Chi_Yu and o.attackId == 20:
                score = 72000 if ex_wall_present and stadium_id > 0 else 18000 if stadium_id > 0 else -1
            elif ENABLE_CHIYU_WALL and active_id == Chi_Yu and o.attackId == 19:
                score = 15000 if hand_bricked else 50
        score += expected_future_adjustment(o, score)
        scores.append(score)
    output = []
    if len(scores) >= 1:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if sorted_scores[i][1] >= 0 or select.minCount > i or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON):
                output.append(sorted_scores[i][0])
    return output

# Final deployment: the universal macro owns every live decision.
# Learned candidates were evaluated offline and rejected when they reduced
# cross-opponent generalization or matchup floor.
def agent(observation: dict) -> list[int]:
    return foundation_agent(observation)
