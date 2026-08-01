import os
import sys
from collections import defaultdict
from enum import IntEnum

from cg.api import AreaType, CardType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, to_observation_class

"""
Dragapult ex Deck
Advanced Level
This deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.
"""

# Load deck.csv from the submission directory without relying on main.py.__file__.
file_path = "deck.csv"
if not os.path.exists(file_path):
    import cg.api as _cg_api
    _submission_root = os.path.dirname(os.path.dirname(os.path.abspath(_cg_api.__file__)))
    _candidate = os.path.join(_submission_root, "deck.csv")
    file_path = _candidate if os.path.exists(_candidate) else "/kaggle_simulations/agent/deck.csv"
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))
    
# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}

# Card IDs used by the rule engine. The active 60-card list is loaded from deck.csv.
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
# Inactive compatibility aliases retained for inherited branches.
Dunsparce = -305
Dudunsparce = -66
Latias_ex = -184
Rare_Candy = -1079
Crushing_Hammer = -1120
Lucky_Helmet = -1156
Brock_Scouting = -1210

UNNECESSARY = -10000000

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
opponent_phantom_seen = False
mirror_profile = 0  # 0 unknown, 1 Munkidori/Judge, 2 fast Dragapult/Lopunny
opponent_seen_ids: set[int] = set()
opponent_seen_counts: defaultdict[int, int] = defaultdict(int)
budew_stall_uses = 0

def _transition(new_state: MachineState):
    global machine_state
    if new_state != machine_state:
        machine_transitions[f"{machine_state.name}->{new_state.name}"] += 1
        machine_state = new_state


class AttackPlan:
    attack: int = 0
    counter: list[int] = []

can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0  # The Supporter card planned for use.
bench_attacker = False  # Whether there is a Benched Pokémon that is ready to attack
pre_turn_log: list[Log] = []
current_turn_log: list[Log] = []

prize: list[int] = []
card_counts: defaultdict[int, int] = defaultdict(int)
serial_set: set[int] = set()
plan_a = AttackPlan()
plan_b = AttackPlan()


def no_damage_dex(id: int) -> bool:
    """Checks if the defending Pokémon possesses innate immunities preventing Dragapult ex from hitting it."""
    # Drednaw, Milotic ex, Sylveon, Crustle
    return id == 158 or id == 207 or id == 330 or id == 345


def no_damage_counter(pokemon: Pokemon) -> bool:
    """Checks if a target prevents placement of Phantom Dive's 6 bench damage counters (via abilities/Energy)."""
    # Poltchageist, Empoleon ex, Skeledirge, Milotic ex, Misty's Magikarp, Antique Cover Fossil
    if pokemon.id == 28 or pokemon.id == 199 or pokemon.id == 203 or pokemon.id == 207 or pokemon.id == 362 or pokemon.id == 1136:
        return True
    for card in pokemon.energyCards:
        # Mist Energy, Rock Fighting Energy
        if card.id == 11 or card.id == 20:
            return True
    return False


def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    
    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 650
    if id in (272,849):
        score += 850
    if id in (742,647,848):
        score += 450
    if id == 344:  # Dwebble before Ascension
        score += 2600
    if id == 120:  # draw engine and next Dragapult
        score += 850
    elif id == 119:
        score += 500
    score += pokemon.hp
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

    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:  # Phantom Dive
                can_main_attack = True
    
    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
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
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
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
                else:
                    if prize >= 2:
                        if remain_prize <= 4:
                            score -= 1200
                    elif prize == 1:
                        score -= 300
                    else:
                        score += 1200
                if max_score < score:
                    max_score = score
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return my_deck

    global pre_turn_log
    global current_turn_log
    global opponent_phantom_seen
    global mirror_profile
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
        opponent_phantom_seen = False
        mirror_profile = 0
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
        if _log.type == LogType.ATTACK and _log.playerIndex != my_index and _log.attackId == 154:
            opponent_phantom_seen = True
        if _log.type == LogType.ATTACK and _log.playerIndex == my_index and _log.attackId == 323:
            budew_stall_uses += 1

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323:  # Itchy Pollen
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                and log.toArea == AreaType.DISCARD):
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

    # Number of cards per card ID on the Bench and in the Active Spot
    field_counts = defaultdict(int)
    # Number of cards per card ID in hand
    hand_counts = defaultdict(int)
    # Number of cards per card ID in discard pile
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
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    visible_opponent = [p for p in list(op_state.active) + list(op_state.bench) if p is not None]
    seen_opponent_ids = {p.id for p in visible_opponent} | set(opponent_seen_ids)
    is_crustle_match = bool(seen_opponent_ids & {344,345})
    is_lucario_match = bool(seen_opponent_ids & {673,674,675,676,677,678})
    is_rocket_match = bool(seen_opponent_ids & {15,17,414,463,473,474,891})
    is_dragapult_match = bool(seen_opponent_ids & {119,120,121})
    is_alakazam_match = bool(seen_opponent_ids & {741,742,743})
    is_marnie_match = bool(seen_opponent_ids & {646,647,648})
    is_starmie_match = bool(seen_opponent_ids & {860,861,1030,1031})
    opponent_profile = ('crustle' if is_crustle_match else 'lucario' if is_lucario_match else
                        'dragapult' if is_dragapult_match else 'alakazam' if is_alakazam_match else
                        'marnie' if is_marnie_match else 'starmie' if is_starmie_match else
                        'rocket' if is_rocket_match else 'unknown')
    efficiency_profile = is_alakazam_match or is_marnie_match or is_lucario_match or is_rocket_match
    opponent_has_munkidori = any(p.id == Munkidori for p in visible_opponent)
    opponent_dragapult_ready = any(
        p.id == Dragapult_ex
        and Basic_Fire_Energy in {e.id for e in p.energyCards}
        and Basic_Psychic_Energy in {e.id for e in p.energyCards}
        for p in visible_opponent
    )
    opponent_dreepy = sum(1 for p in visible_opponent if p.id == Dreepy)
    opponent_drakloak = sum(1 for p in visible_opponent if p.id == Drakloak)
    opponent_crustle_active = bool(op_state.active and op_state.active[0] is not None and op_state.active[0].id == 345)
    opponent_dwebble_bench = sum(1 for p in op_state.bench if p is not None and p.id == 344)
    if seen_opponent_ids & {112,305,66,306}:
        mirror_profile = 1
    elif seen_opponent_ids & {184,848,849,272}:
        mirror_profile = 2
    needs_deep_bench = is_lucario_match or is_dragapult_match
    dark_munk_count = sum(1 for p in list(my_state.active) + list(my_state.bench)
                          if p is not None and p.id == Munkidori and any(e.id == Basic_Darkness_Energy for e in p.energyCards))
    own_total_damage = sum(max(0, card_table[p.id].hp - p.hp) for p in list(my_state.active) + list(my_state.bench) if p is not None)
    damaged_mirror_line = any(
        p.id in (Dreepy, Drakloak, Dragapult_ex) and p.hp < card_table[p.id].hp
        for p in list(my_state.active) + list(my_state.bench) if p is not None
    )
    mirror_damage_emergency = is_dragapult_match and (
        opponent_phantom_seen or opponent_dragapult_ready or own_total_damage >= 20 or damaged_mirror_line
    )

    def _phantom_ready(p):
        if p is None or p.id != Dragapult_ex:
            return False
        ids = {e.id for e in p.energyCards}
        return Basic_Fire_Energy in ids and Basic_Psychic_Energy in ids

    # Correct the inherited two-energy shortcut: Darkness + one attack Energy is
    # not enough for Phantom Dive. The FSM uses exact readiness.
    bench_attacker = any(_phantom_ready(p) for p in my_state.bench if p is not None)
    ready_dragapult = can_main_attack or bench_attacker or any(_phantom_ready(p) for p in my_state.active if p is not None)
    dreepy_line_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    active_pokemon = my_state.active[0] if my_state.active and my_state.active[0] is not None else None
    active_support_stranded = bool(active_pokemon is not None and active_pokemon.id in
        (Meowth_ex, Fezandipiti_ex, Munkidori, Shaymin, Chi_Yu, Yveltal)
        and card_table[active_pokemon.id].retreatCost > len(active_pokemon.energies)
        and len(my_state.bench) > 0)
    opponent_ready_attacker = opponent_dragapult_ready or any(card_table[p.id].ex and len(p.energyCards) >= 2 for p in visible_opponent)
    likely_boss = op_state.handCount >= 5 and opponent_seen_counts[Boss_Orders] < 3
    urgent_attack = (is_lucario_match or is_rocket_match) and (len(op_state.prize) <= 2 or (opponent_ready_attacker and likely_boss))
    target_priority_ids = {'dragapult': {Dreepy: 7000, Drakloak: 10000, Munkidori: 8000}, 'lucario': {677:8500,673:6500,675:4500,676:4500}, 'alakazam': {741:7500,742:10500}, 'marnie': {646:7000,647:9500}, 'starmie': {1030:8500,860:7500,Munkidori:7000}, 'crustle': {344:12000}}.get(opponent_profile,{})
    opponent_active = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    searched_progress_this_turn = any(
        l.type == LogType.PLAY and l.playerIndex == my_index and
        l.cardId in (Poke_Pad, Ultra_Ball, Buddy_Buddy_Poffin, Night_Stretcher)
        for l in current_turn_log
    )
    searched_card_still_actionable = searched_progress_this_turn and any(
        (c.id == Drakloak and can_evolve_dreepy)
        or (c.id == Dragapult_ex and can_evolve_drakloak)
        or (c.id == Dreepy and len(my_state.bench) < 5 and dreepy_line_count < 2)
        or (c.id == Munkidori and field_counts[Munkidori] == 0 and len(my_state.bench) < 5)
        or (c.id == Chi_Yu and is_crustle_match and field_counts[Chi_Yu] == 0)
        for c in (my_state.hand or [])
    )
    # Only block search -> Lillie when the hand is already large and the searched
    # card can be used immediately. Small hands retain the recovery option.
    block_search_then_lillie = searched_card_still_actionable and my_state.handCount >= 7

    stall_caps = {'dragapult': 4 if mirror_profile == 1 else 3, 'lucario': 3,
                  'alakazam': 2, 'marnie': 2, 'starmie': 2, 'crustle': 1,
                  'rocket': 1, 'unknown': 1}
    stall_cap = stall_caps.get(opponent_profile, 1)
    stall_setup = bool(active_id == Budew and not ready_dragapult and state.turn <= 7
                       and dreepy_line_count >= 1 and budew_stall_uses < stall_cap)

    # Yveltal lock is a contingency line only when Yveltal is already in play.
    # It never steals a search from the Dragapult line.
    trap_ids = {Dreepy, Drakloak, Munkidori, 741,742,646,647,1030,860,344,675,676,677,673}
    trap_target = bool(opponent_active is not None and opponent_active.id in trap_ids
                       and len(opponent_active.energyCards) <= 1
                       and card_table[opponent_active.id].retreatCost >= 1
                       and not opponent_ready_attacker and op_state.handCount <= 4)
    yveltal_pokemon = next((p for p in list(my_state.active)+list(my_state.bench) if p is not None and p.id == Yveltal), None)
    yveltal_has_dark = bool(yveltal_pokemon and any(e.id == Basic_Darkness_Energy for e in yveltal_pokemon.energyCards))
    yveltal_lock_ready = bool(yveltal_pokemon and yveltal_has_dark and trap_target and not ready_dragapult)

    # Lock Fire/Psychic attachments onto one charging line. If one attacker is
    # already online, the lock moves to the best backup line.
    main_candidates = [p for p in list(my_state.active) + list(my_state.bench)
                       if p is not None and p.id in (Dreepy, Drakloak, Dragapult_ex)]
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
        bench_counter_safe = 0 if (is_dragapult_match and p not in my_state.active and p.hp <= 60) else 1
        active_attack_safe = 0 if (is_dragapult_match and p in my_state.active and opponent_dragapult_ready and p.hp <= 200) else 1
        return (bench_counter_safe, active_attack_safe, required, stage, len(p.energies), active_bonus, p.hp)
    primary = max(pool, key=_primary_key) if pool else None
    primary_serial = primary.serial if primary is not None else -1
    primary_energy_ids = {e.id for e in primary.energyCards} if primary is not None else set()
    primary_missing_types = {Basic_Fire_Energy, Basic_Psychic_Energy} - primary_energy_ids
    attack_energy_in_hand = {c.id for c in (my_state.hand or []) if c.id in (Basic_Fire_Energy, Basic_Psychic_Energy)}
    can_complete_primary_from_hand = bool(primary_missing_types & attack_energy_in_hand)

    progress_ids = {Drakloak, Dragapult_ex, Ultra_Ball, Poke_Pad, Crispin, Lillie_Determination,
                    Basic_Fire_Energy, Basic_Psychic_Energy, Night_Stretcher}
    hand_progress = any(c.id in progress_ids for c in (my_state.hand or []))
    hand_bricked = (not ready_dragapult and dreepy_line_count > 0 and my_state.handCount <= 3 and not hand_progress)
    if is_crustle_match:
        desired_state = MachineState.WALL_BREAK
    elif len(op_state.prize) <= 2 and ready_dragapult:
        desired_state = MachineState.ENDGAME
    elif yveltal_lock_ready:
        desired_state = MachineState.YVELTAL_LOCK
    elif stall_setup:
        desired_state = MachineState.STALL_SETUP
    elif is_dragapult_match:
        desired_state = MachineState.MIRROR_CONTROL
    elif hand_bricked:
        desired_state = MachineState.HAND_RECOVERY
    elif pre_ko and not ready_dragapult:
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
        if required_energy and pokemon.id in (Dreepy, Drakloak, Dragapult_ex) and phase != MachineState.WALL_BREAK:
            if pokemon.serial == primary_serial:
                primary_bonus = 10500
            elif primary_serial >= 0 and phase in (MachineState.OPENING, MachineState.EVOLUTION, MachineState.CHARGE, MachineState.RECOVERY):
                primary_bonus = -4500
        if (active_support_stranded and active and active_pokemon is not None
            and pokemon.serial == active_pokemon.serial
            and energy_count < card_table[pokemon.id].retreatCost):
            if attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                return 36500
            if attach_id == Basic_Darkness_Energy and pokemon.id in (Munkidori, Yveltal):
                return 36200
        # Before the first Phantom Dive attacker is ready, Darkness on Munkidori
        # costs the deck a full turn of tempo. Defer it outside wall-breaking.
        if (pokemon.id == Munkidori and attach_id == Basic_Darkness_Energy
            and not ready_dragapult and phase != MachineState.WALL_BREAK
            and not (phase == MachineState.MIRROR_CONTROL and mirror_damage_emergency)):
            return -1
        # Recover a Dragapult that has two energies but is missing one required type.
        if pokemon.id == Dragapult_ex and required_energy and attach_id not in energy_ids:
            if Basic_Fire_Energy not in energy_ids or Basic_Psychic_Energy not in energy_ids:
                return 31500 + primary_bonus + (300 if active else 0)
        if attach_id == 0:
            if pokemon.id == Dragapult_ex:
                return 60000
            if pokemon.id == Yveltal and is_crustle_match:
                return 52000
            if pokemon.id == Munkidori and any(e.id == Basic_Darkness_Energy for e in pokemon.energyCards):
                return 30000
            return 1000
        if card_table[attach_id].cardType == CardType.TOOL:
            # Attach tool
            score = 60000
            if active:
                score += 1000
            return score
        
        # Attach energy
        if (mirror_profile == 2 and active and bench_attacker
            and pokemon.id in (Shaymin, Munkidori, Chi_Yu, Yveltal, Fezandipiti_ex, Meowth_ex, Dreepy)
            and energy_count < card_table[pokemon.id].retreatCost):
            return 38500
        if pokemon.id == Yveltal:
            ids = {e.id for e in pokemon.energyCards}
            if (active and trap_target and not ready_dragapult and not can_complete_primary_from_hand
                and attach_id == Basic_Darkness_Energy and Basic_Darkness_Energy not in ids):
                return 24800
            if not is_crustle_match or field_counts[Chi_Yu] > 0:
                return -1
            if attach_id == Basic_Darkness_Energy and len(ids) < 2:
                return 25000 + len(ids) * 700
            if len(pokemon.energies) == 2 and attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                return 24600
            return -1
        if pokemon.id == Chi_Yu:
            if not is_crustle_match:
                return -1
            if len(pokemon.energies) == 0 and attach_id == Basic_Fire_Energy:
                return 40500 + (800 if active else 0)
            if len(pokemon.energies) == 1 and attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy, Basic_Darkness_Energy):
                return 42500 + (1200 if active else 0)
            return -1
        if pokemon.id == Shaymin:
            return -1
        if pokemon.id == Munkidori:
            if attach_id == Basic_Darkness_Energy and not any(e.id == Basic_Darkness_Energy for e in pokemon.energyCards):
                if is_crustle_match:
                    chi_ready = any(p.id == Chi_Yu and len(p.energies) >= 2 for p in list(my_state.active)+list(my_state.bench) if p is not None)
                    return (25200 if chi_ready else 18500) + (200 if active else 0)
                if is_dragapult_match and mirror_damage_emergency:
                    if mirror_profile == 2 and can_complete_primary_from_hand:
                        return 29200 + (300 if active else 0)
                    return 33700 + (300 if active else 0)
                if is_lucario_match or (is_dragapult_match and opponent_has_munkidori):
                    return 26300 + (300 if active else 0)
                return 24500 + (300 if active else 0)
            if is_crustle_match and attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy) and len(pokemon.energies) < 2:
                return 19400
            return -1
        if pokemon.id in (Dreepy, Drakloak, Dragapult_ex) and attach_id == Basic_Darkness_Energy:
            return -1
        if pokemon.id == Dunsparce:
            if active and energy_count == 0 and bench_attacker:
                return 22500
            return -1
        if pokemon.id == Dudunsparce:
            if is_crustle_match and energy_count < 3:
                return 22200 + energy_count * 300
            return -1
        if pokemon.id in (Budew, Shaymin):
            return -1
        elif pokemon.id in (Meowth_ex, Fezandipiti_ex):
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                if bench_attacker or field_counts[Budew] >= 1:
                    return 22000
                else:
                    return 18000
            else:
                return -1
        if active and can_main_attack:
            return -1
        score = 20000
        if is_crustle_match and pokemon.id == Drakloak:
            score += 1800
        if is_crustle_match and pokemon.id == Dragapult_ex:
            score -= 800
        if energy_count >= 2:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
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
        else:  # energy_count == 0
            if active:
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
                score = 20000
            else:
                score = 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 10000
                else:
                    score = 50
            else:
                if field_counts[id] >= 2:
                    score = 50
                else:
                    score = 2000
        elif id == Munkidori:
            has_dark_munk = any(p.id == Munkidori and any(e.id == Basic_Darkness_Energy for e in p.energyCards) for p in list(my_state.active) + list(my_state.bench) if p is not None)
            target_munk_count = 2 if (is_lucario_match or (is_dragapult_match and opponent_has_munkidori and own_total_damage >= 40)) else 1
            if field_counts[Munkidori] < target_munk_count and (hand_counts[Basic_Darkness_Energy] > 0 or deck_counts[Basic_Darkness_Energy] > 0):
                score = 26000 if is_crustle_match else 15000
            elif field_counts[Munkidori] >= target_munk_count or has_dark_munk:
                score = 100
        elif id == Dunsparce:
            if field_counts[Dunsparce] + field_counts[Dudunsparce] == 0 and len(my_state.bench) <= (4 if needs_deep_bench else 3):
                score = 10000
            else:
                score = 200
        elif id == Dudunsparce:
            can_evolve_dunsparce = any(p.id == Dunsparce and not p.appearThisTurn for p in list(my_state.active) + list(my_state.bench) if p is not None)
            score = 18000 if can_evolve_dunsparce else 500
        elif id == Chi_Yu:
            score = 43000 if is_crustle_match and field_counts[Chi_Yu] == 0 else 300
        elif id == Yveltal:
            if is_crustle_match and field_counts[Chi_Yu] == 0 and field_counts[Yveltal] == 0:
                score = 24000
            else:
                score = 200
        elif id == Shaymin:
            score = 12000 if not is_dragapult_match and field_counts[Shaymin] == 0 and len(my_state.bench) <= 3 else 100
        elif id == Fezandipiti_ex:
            if pre_ko:
                if (is_dragapult_match or is_crustle_match) and (my_state.handCount >= 4 or len(my_state.bench) >= 4):
                    score = 100
                else:
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
            if is_dragapult_match and (my_state.handCount >= 4 or len(my_state.bench) >= 3):
                score = 5
            elif support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
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
                if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                    count += 1
                if count >= 2:
                    score = 35000
        elif id == Night_Stretcher:
            if is_crustle_match and discard_counts[Chi_Yu] > 0 and field_counts[Chi_Yu] == 0:
                score = 62000
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        score = max(score, hand_score(i, ignore_count))
        elif id == Enhanced_Hammer:
            has_special = any(card_table[e.id].cardType == CardType.SPECIAL_ENERGY for p in visible_opponent for e in p.energyCards)
            score = 24000 if has_special else 20
        elif id == Tool_Scrapper:
            tool_count = sum(len(p.tools) for p in visible_opponent)
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
            if is_crustle_match and opponent_dwebble_bench > 0 and (can_attack or bench_attacker):
                score = 90000
            elif plan_a.attack > 0:
                score = 60000
        elif id == Rosas_Encouragement:
            has_fire_discard = discard_counts[Basic_Fire_Energy] > 0
            has_psychic_discard = discard_counts[Basic_Psychic_Energy] > 0
            primary_ids = {e.id for e in primary.energyCards} if primary is not None else set()
            can_complete_pair = ((has_fire_discard and has_psychic_discard) or
                                 (has_fire_discard and Basic_Psychic_Energy in primary_ids) or
                                 (has_psychic_discard and Basic_Fire_Energy in primary_ids))
            score = 59000 if prize_diff > 0 and primary is not None and can_complete_pair else UNNECESSARY
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
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
                score = UNNECESSARY if my_state.deckCount <= 6 or block_search_then_lillie else 45000
        elif id == Judge:
            # Never spend Judge immediately before an available Unfair Stamp or near deck-out.
            if my_state.deckCount <= 6 or (pre_ko and hand_counts[Unfair_Stamp] > 0):
                score = UNNECESSARY
            elif op_state.handCount >= 6 and my_state.handCount <= 5:
                score = 43000
            elif op_state.handCount >= 7:
                score = 35000
            elif my_state.handCount <= 2:
                score = 26000
            else:
                score = 500
        elif id == Risky_Ruins:
            low_hp_unready = sum(1 for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id in (Dreepy, Budew, Chi_Yu, Shaymin) and p.hp <= 80)
            op_basic_non_dark = sum(1 for p in visible_opponent if card_table[p.id].basic and card_table[p.id].energyType != 7)
            drag_ready = field_counts[Dragapult_ex] > 0 or (field_counts[Drakloak] > 0 and can_evolve_drakloak)
            munk_ready = dark_munk_count > 0
            if stadium_id != Risky_Ruins and op_basic_non_dark >= 1 and drag_ready and (low_hp_unready == 0 or munk_ready):
                score = 12000 + op_basic_non_dark * 2200 + (6500 if is_lucario_match else 0)
                if is_dragapult_match and munk_ready and (opponent_dreepy + opponent_drakloak) >= 1:
                    score += 12000
            else:
                score = 50
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy or id == Basic_Darkness_Energy:
            if can_main_attack and (len(op_state.prize) <= 2
                or (bench_attacker and len(op_state.prize) <= 4)):
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
            if id == Dreepy and dreepy_line_count < 2: score += 7000
        elif phase == MachineState.EVOLUTION:
            if id in (Drakloak, Dragapult_ex, Ultra_Ball, Poke_Pad): score += 5000
        elif phase == MachineState.CHARGE:
            if id == Crispin: score += 16000
            elif id in (Basic_Fire_Energy, Basic_Psychic_Energy): score += 3500
        elif phase == MachineState.RECOVERY:
            if id in (Night_Stretcher, Dreepy, Drakloak, Dragapult_ex, Ultra_Ball): score += 7000
        elif phase == MachineState.ENDGAME:
            if id == Boss_Orders and plan_a.attack > 0: score += 9000
            elif id in (Lillie_Determination, Judge, Buddy_Buddy_Poffin): score -= 7000
        elif phase == MachineState.MIRROR_CONTROL:
            if id in (Dreepy, Drakloak, Dragapult_ex, Night_Stretcher, Ultra_Ball, Poke_Pad): score += 6500
            if id == Munkidori and field_counts[Munkidori] == 0: score += 8000
            if id == Basic_Darkness_Energy and mirror_damage_emergency and dark_munk_count == 0: score += 9000
            if id == Crispin and not ready_dragapult: score += 9000
            if id == Boss_Orders and plan_a.attack > 0: score += 7000
            if id in (Chi_Yu, Yveltal, Shaymin): score -= 9000
        elif phase == MachineState.STALL_SETUP:
            if id in (Drakloak, Dragapult_ex, Crispin, Basic_Fire_Energy, Basic_Psychic_Energy): score += 800
        elif phase == MachineState.YVELTAL_LOCK:
            if id in (Drakloak, Dragapult_ex, Crispin, Basic_Fire_Energy, Basic_Psychic_Energy): score += 600
        elif phase == MachineState.HAND_RECOVERY:
            if id in (Lillie_Determination, Judge, Poke_Pad, Ultra_Ball, Meowth_ex, Fezandipiti_ex): score += 12000
            if id in (Boss_Orders, Risky_Ruins, Team_Rocket_Watchtower): score -= 5000

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

    no_draw = (my_state.deckCount <= 8)  # Whether to restrict actions that reduce the deck
    wall_attacker_ready = is_crustle_match and any(
        (p.id == Chi_Yu and len(p.energies) >= 2)
        or (p.id == Yveltal and len(p.energies) >= 3 and sum(1 for e in p.energyCards if e.id == Basic_Darkness_Energy) >= 2)
        or (p.id == Munkidori and len(p.energies) >= 2)
        or (p.id == Drakloak and Basic_Fire_Energy in {e.id for e in p.energyCards} and Basic_Psychic_Energy in {e.id for e in p.energyCards})
        for p in my_state.bench if p is not None
    )
    do_switch = (not can_main_attack and (bench_attacker or wall_attacker_ready or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    if phase == MachineState.YVELTAL_LOCK and active_id != Yveltal and yveltal_lock_ready:
        do_switch = True
    if is_crustle_match and active_id == Dragapult_ex:
        if opponent_crustle_active and wall_attacker_ready:
            do_switch = True
        active_crustle_hp = op_state.active[0].hp if op_state.active and op_state.active[0] is not None and op_state.active[0].id == 345 else 999
        for pkm in my_state.bench:
            ready = (pkm.id == Drakloak and len(pkm.energies) >= 2) or (pkm.id == Dudunsparce and len(pkm.energies) >= 3) or (pkm.id == Munkidori and len(pkm.energies) >= 2)
            if ready and (active_crustle_hp <= 130 or (dark_munk_count >= 2 and own_total_damage >= 50)):
                do_switch = True
    effect_card_id = 0 if select.effect == None else select.effect.id
    context_card_id = 0 if select.contextCard == None else select.contextCard.id
    crispin_available = set()
    if effect_card_id == Crispin and select.deck is not None:
        crispin_available = {c.id for c in select.deck}
    
    scores = []  # Score for each action
    for o in select.option:
        score = 0  # The default and baseline score is 0.
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
                if (context == SelectContext.SWITCH
                    or context == SelectContext.TO_ACTIVE
                    or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    # Selection of the Pokémon to send to the Active Spot
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                            if is_dragapult_match and hp <= 60:
                                score -= 12000
                        elif card.id == Drakloak:
                            if is_crustle_match and energy_count >= 2:
                                score += 92000
                            elif energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                            if is_crustle_match and opponent_crustle_active:
                                score -= 90000
                        elif card.id == Chi_Yu:
                            score += 120000 if is_crustle_match and energy_count >= 2 else 1000
                        elif card.id == Yveltal:
                            dark_count = sum(1 for e in card.energyCards if e.id == Basic_Darkness_Energy) if isinstance(card, Pokemon) else 0
                            if phase == MachineState.YVELTAL_LOCK and dark_count >= 1:
                                score += 103000
                            elif is_crustle_match and energy_count >= 3 and dark_count >= 2:
                                score += 110000
                            elif is_crustle_match and energy_count >= 1:
                                score += 25000
                        elif card.id == Munkidori:
                            score += 95000 if is_crustle_match and energy_count >= 2 else -1500
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Dunsparce:
                            score += 500 if context != SelectContext.SETUP_ACTIVE_POKEMON else 1000
                        elif card.id == Dudunsparce:
                            score += 90000 if is_crustle_match and energy_count >= 3 else -500
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    else:
                        if card.id in target_priority_ids: score += target_priority_ids[card.id]
                        if plan_a.attack == o.index + 1:
                            score += 100000
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
                    if phase == MachineState.WALL_BREAK and chi_target is not None:
                        chi_ids = {e.id for e in chi_target.energyCards}
                        if Basic_Fire_Energy not in chi_ids:
                            score = 110000 if card.id == Basic_Fire_Energy else (25000 if card.id in (Basic_Psychic_Energy, Basic_Darkness_Energy) else -1)
                        elif len(chi_target.energies) < 2:
                            score = 106000 if card.id == Basic_Psychic_Energy else (101000 if card.id == Basic_Fire_Energy else 70000 if card.id == Basic_Darkness_Energy else -1)
                        else:
                            score = -1
                    elif phase == MachineState.WALL_BREAK and yveltal_target is not None:
                        dark_count_y = sum(1 for e in yveltal_target.energyCards if e.id == Basic_Darkness_Energy)
                        if dark_count_y < 2:
                            score = 110000 if card.id == Basic_Darkness_Energy else (30000 if card.id in (Basic_Fire_Energy, Basic_Psychic_Energy) else -1)
                        elif len(yveltal_target.energies) < 3:
                            score = 105000 if card.id in (Basic_Fire_Energy, Basic_Psychic_Energy) else -1
                        else:
                            score = -1
                    else:
                        primary_ids = {e.id for e in primary.energyCards} if primary is not None else set()
                        if card.id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                            score = 99000 if card.id not in primary_ids else 72000
                        elif card.id == Basic_Darkness_Energy and field_counts[Munkidori] > 0 and ready_dragapult:
                            score = 18000
                        else:
                            score = -1
                elif context == SelectContext.ATTACH_TO and effect_card_id == Rosas_Encouragement:
                    if card.id == Basic_Fire_Energy:
                        score = 98000
                    elif card.id == Basic_Psychic_Energy:
                        score = 97500
                    elif card.id == Basic_Darkness_Energy:
                        score = 1000 if phase == MachineState.WALL_BREAK else -1
                    else:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id in (Poke_Pad, Ultra_Ball):
                        if card.id == Drakloak and can_evolve_dreepy: score += 90000
                        elif card.id == Dragapult_ex and can_evolve_drakloak: score += 88000
                        elif card.id == Dreepy and dreepy_line_count < 2: score += 76000
                        elif card.id == Munkidori and field_counts[Munkidori] == 0 and (own_total_damage > 0 or is_dragapult_match or is_lucario_match): score += 65000
                        elif card.id == Chi_Yu and is_crustle_match and field_counts[Chi_Yu] == 0: score += 85000
                    if effect_card_id == Crispin:
                        has_fire = Basic_Fire_Energy in crispin_available
                        has_psychic = Basic_Psychic_Energy in crispin_available
                        has_dark = Basic_Darkness_Energy in crispin_available
                        chi_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Chi_Yu), None)
                        yveltal_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Yveltal), None)
                        if context == SelectContext.TO_HAND and phase == MachineState.WALL_BREAK and chi_target is not None:
                            chi_ids = {e.id for e in chi_target.energyCards}
                            # Leave the Energy Chi-Yu needs for the automatic attachment.
                            if Basic_Fire_Energy not in chi_ids and has_fire:
                                score = 108000 if card.id in (Basic_Psychic_Energy, Basic_Darkness_Energy) else 20000
                            elif len(chi_target.energies) < 2 and has_psychic:
                                score = 108000 if card.id in (Basic_Fire_Energy, Basic_Darkness_Energy) else 20000
                            else:
                                score = 50000 if card.id in (Basic_Fire_Energy, Basic_Psychic_Energy) else 10000
                        elif context == SelectContext.TO_HAND and phase == MachineState.WALL_BREAK and yveltal_target is not None and has_dark:
                            score = 108000 if card.id in (Basic_Fire_Energy, Basic_Psychic_Energy) else 20000
                        # If only one attack type remains, take Darkness into hand so the useful type can be attached.
                        elif context == SelectContext.TO_HAND and not (has_fire and has_psychic):
                            if card.id == Basic_Darkness_Energy:
                                score = 95000
                            elif card.id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                                score = 70000
                            else:
                                score = -1
                        elif card.id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                            # Prefer putting the already-attached type into hand, leaving the missing type to attach.
                            attached_ids = set()
                            for pkm in list(my_state.active) + list(my_state.bench):
                                if pkm is not None and pkm.id in (Dragapult_ex, Drakloak, Dreepy):
                                    attached_ids.update(e.id for e in pkm.energyCards)
                            score = 93000 if card.id in attached_ids else 90000
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
                        if is_crustle_match and card.id == Dragapult_ex:
                            score += 15000
                        if damage_on_card <= 0:
                            score = -1
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if is_crustle_match and o.area == AreaType.ACTIVE and card.id == 345:
                                score += 200000
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            if card.id == 133 or card.id == 351:
                                score += 30000
                            if is_dragapult_match:
                                if card.id in (Dreepy, Drakloak) and hp <= 60:
                                    score += 180000
                                elif card.id == Munkidori and hp <= 60:
                                    score += 120000
                                elif card.id in (Fezandipiti_ex, Meowth_ex) and hp <= 60:
                                    score += 90000
                        else:
                            remain_damage = select.remainDamageCounter * 10
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                            if is_dragapult_match:
                                if card.id in (Dreepy, Drakloak) and hp <= remain_damage:
                                    score += 200000
                                elif card.id == Munkidori and hp <= remain_damage:
                                    score += 130000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
                    if effect_card_id == Rosas_Encouragement and isinstance(card, Pokemon):
                        score = 100000 if card.serial == primary_serial else 1000
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            # Discarding energy (Retreat or Crushing Hammer)
            if o.playerIndex != state.yourIndex:
                if o.area == AreaType.BENCH:
                    score = 20
                else:
                    score = 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            phase_blocked = False
            if card.id == Dreepy and dreepy_line_count >= 2 and phase not in (MachineState.RECOVERY, MachineState.WALL_BREAK):
                phase_blocked = True
            if phase in (MachineState.OPENING, MachineState.EVOLUTION, MachineState.CHARGE) and not ready_dragapult:
                if card.id in (Chi_Yu, Yveltal):
                    phase_blocked = True
            if card.id == Dreepy:
                if main_pokemon_count < 3:
                    score = 51000
                else:
                    score = -1
            elif card.id == Munkidori:
                target_munk_count = 2 if (is_lucario_match or (is_dragapult_match and opponent_has_munkidori and own_total_damage >= 40)) else 1
                if field_counts[Munkidori] < target_munk_count and len(my_state.bench) < 5:
                    score = 54500 if is_crustle_match else 50500
                else:
                    score = -1
            elif card.id == Chi_Yu:
                if is_crustle_match and field_counts[Chi_Yu] == 0 and len(my_state.bench) < 5:
                    score = 59000
                else:
                    score = -1
            elif card.id == Yveltal:
                chi_available = field_counts[Chi_Yu] + hand_counts[Chi_Yu] + deck_counts[Chi_Yu] + discard_counts[Chi_Yu]
                if is_crustle_match and chi_available == 0 and field_counts[Yveltal] == 0 and len(my_state.bench) < 5:
                    score = 53500
                else:
                    score = -1
            elif card.id == Shaymin:
                if not is_dragapult_match and field_counts[Shaymin] == 0 and len(my_state.bench) <= 3:
                    score = 50020
                else:
                    score = -1
            elif card.id == Dunsparce:
                support_bench = field_counts[Dunsparce] + field_counts[Dudunsparce] + field_counts[Fezandipiti_ex] + field_counts[Meowth_ex]
                bench_limit = 5 if needs_deep_bench else 4
                support_limit = 3 if needs_deep_bench else 2
                if field_counts[Dunsparce] + field_counts[Dudunsparce] == 0 and len(my_state.bench) < bench_limit and support_bench < support_limit:
                    score = 50100
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                if (is_dragapult_match or is_crustle_match) and (my_state.handCount >= 4 or len(my_state.bench) >= 4):
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
                if is_dragapult_match and (my_state.handCount >= 4 or len(my_state.bench) >= 3):
                    score = -1
                elif len(my_state.bench) >= (5 if needs_deep_bench else 4):
                    score = -1
                elif state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and not plan_a.attack <= 0:
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
                has_special = any(card_table[e.id].cardType == CardType.SPECIAL_ENERGY for p in visible_opponent for e in p.energyCards)
                score = 41000 if has_special else -1
            elif card.id == Tool_Scrapper:
                tool_count = sum(len(p.tools) for p in visible_opponent)
                score = 40500 if tool_count > 0 else -1
            elif card.id == Crushing_Hammer:
                op_energy = sum(len(p.energyCards) for p in list(op_state.active) + list(op_state.bench) if p is not None)
                score = 40000 if op_energy > 0 else -1
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support and my_state.deckCount > 6 and not block_search_then_lillie:
                    score = 14000
                else:
                    score = -1
            elif card.id == Judge:
                if card.id == use_support and my_state.deckCount > 6 and not (pre_ko and hand_counts[Unfair_Stamp] > 0):
                    score = 33500
                else:
                    score = -1
            elif card.id == Risky_Ruins:
                chi_needs_stadium = is_crustle_match and field_counts[Chi_Yu] > 0 and stadium_id == 0
                if mirror_profile == 1 and stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif chi_needs_stadium:
                    score = 76000
                elif stadium_id == 1264 or (stadium_id != Risky_Ruins and hand_score(Risky_Ruins, True) >= 1000):
                    score = 47000 if is_crustle_match else (50000 if is_dragapult_match and dark_munk_count > 0 else 39000)
                else:
                    score = -1
            elif card.id == Team_Rocket_Watchtower:
                colorless_ability = any(card_table[p.id].energyType == 0 and len(card_table[p.id].skills) > 0 for p in visible_opponent)
                need_stadium_for_chi = is_crustle_match and field_counts[Chi_Yu] > 0
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
                missing_stage = (can_evolve_dreepy and deck_counts[Drakloak] > 0) or (can_evolve_drakloak and deck_counts[Dragapult_ex] > 0)
                wall_target = is_crustle_match and field_counts[Chi_Yu] == 0 and deck_counts[Chi_Yu] > 0
                if missing_stage and (negative_hand_count >= 1 or disposable_hand_count >= 1) and my_state.handCount >= 3:
                    score = 53000 if phase == MachineState.MIRROR_CONTROL else 50500
                elif wall_target and (negative_hand_count >= 1 or disposable_hand_count >= 1) and my_state.handCount >= 3:
                    score = 53500
                elif main_pokemon_count < 2 and negative_hand_count >= 1:
                    score = 48000
                elif negative_hand_count >= 2:
                    score = 44000
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if efficiency_profile:
                    useful = ((deck_counts[Drakloak] > 0 and field_counts[Dreepy] > 0)
                              or (deck_counts[Dreepy] > 0 and dreepy_line_count < 2)
                              or (deck_counts[Munkidori] > 0 and field_counts[Munkidori] == 0))
                    score = 45000 if useful else -1
                elif deck_counts[Dreepy] + deck_counts[Drakloak] + deck_counts[Munkidori] + deck_counts[Yveltal] + deck_counts[Chi_Yu] > 0:
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
            elif is_crustle_match and pokemon.id == Drakloak and field_counts[Dragapult_ex] >= 1:
                score = -1
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Munkidori:
                own_damage = sum(max(0, card_table[p.id].hp - p.hp) for p in list(my_state.active) + list(my_state.bench) if p is not None)
                if own_damage >= 10 and any(e.id == Basic_Darkness_Energy for e in card.energyCards):
                    transferable = min(30, own_damage)
                    can_finish = any(p.hp <= transferable and not no_damage_counter(p) for p in visible_opponent)
                    score = 104000 if can_finish else (78000 if is_crustle_match else 70000)
                else:
                    score = -1
            elif no_draw:
                score = -1
            elif card.id == 1267:  # Lumiose City
                score = 1
            elif card.id == Dudunsparce:
                draw_threshold = 7 if needs_deep_bench else 5
                score = -1 if is_crustle_match else (65000 if my_state.handCount <= draw_threshold else -1)
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
            if active_id == Yveltal and o.attackId == 997 and phase == MachineState.YVELTAL_LOCK:
                score = 6000
            if o.attackId == 154 and urgent_attack:
                score = 76000
            if is_crustle_match and opponent_crustle_active and active_id == Dragapult_ex:
                score = -1
            elif is_crustle_match and active_id == Chi_Yu:
                score += 82000 if o.attackId == 20 and stadium_id != 0 else 30000
            elif is_crustle_match and active_id == Yveltal:
                score += 70000 if o.attackId == 998 else 36000
            elif is_crustle_match and active_id == Munkidori:
                score += 70000 if o.attackId == 141 else 30000
            elif is_crustle_match and active_id == Drakloak:
                score += 60000
            elif is_crustle_match and active_id == Dragapult_ex and o.attackId == 154:
                # Keep bench pressure, but switching to a lethal non-ex attacker takes precedence.
                score = 140

        scores.append(score)

    output = []
    if len(scores) >= 1:
        # Select in descending order of score
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            # If the score is negative, do not select it if skipping is possible
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])
                
    return output
