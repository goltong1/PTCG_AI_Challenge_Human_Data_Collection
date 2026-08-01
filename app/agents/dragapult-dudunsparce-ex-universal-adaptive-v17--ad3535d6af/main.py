"""Universal Adaptive v17.

Dudunsparce-ex Crustle Adapter v16.

Architecture reset:
- speculative opponent-turn lookahead is disabled;
- the stable league-distilled execution policy remains the action prior;
- damage-counter placement uses an exact remaining-budget prize optimizer;
- Active damage immunity never suppresses a legal Phantom Dive Bench effect;
- Munkidori + Phantom Dive is linked only when public HP/immunity makes the
  nine-counter sequence objectively available.
"""
import os
import sys
# Make the bundled cg package importable even when the host executes main.py
# from another working directory or through raw exec().
_submission_dir_hint = os.path.dirname(os.path.abspath(globals().get("__file__", "/kaggle_simulations/agent/main.py")))
for _path_hint in (_submission_dir_hint, "/kaggle_simulations/agent", os.getcwd()):
    if _path_hint and _path_hint not in sys.path:
        sys.path.insert(0, _path_hint)
from collections import Counter, defaultdict
from dataclasses import asdict
from enum import IntEnum

from cg.api import (AreaType, CardType, Log, LogType, Observation, SelectContext,
                    OptionType, Card, Pokemon, State, all_card_data, all_attack,
                    to_observation_class, search_begin, search_step, search_end,
                    search_release)

"""
Dragapult ex Deck
League-distilled adaptive mirror policy v9
Advanced Level
Offline league distillation: go-first tempo, mirror Budew cap=2, early first Munkidori when a Dreepy line is established and Bench usage is low.
This deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.
"""

# Resolve deck.csv from the extracted submission directory.  Kaggle executes
# main.py without guaranteeing that the current working directory is the agent
# directory, so the bundled cg package is used as a stable location anchor.
import cg.api as _cg_api
_submission_root = os.path.dirname(os.path.dirname(os.path.abspath(_cg_api.__file__)))
_deck_candidates = [
    os.path.join(_submission_root, "deck.csv"),
    "/kaggle_simulations/agent/deck.csv",
    os.path.join(os.getcwd(), "deck.csv"),
]
file_path = next((p for p in _deck_candidates if os.path.isfile(p)), None)
if file_path is None:
    raise FileNotFoundError("deck.csv was not found in the submission directory")
with open(file_path, "r", encoding="utf-8") as file:
    csv = [line.strip() for line in file if line.strip()]
if len(csv) != 60:
    raise ValueError(f"deck.csv must contain exactly 60 card IDs, got {len(csv)}")
my_deck = [int(card_id) for card_id in csv]
    
# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
attack_table = {a.attackId:a for a in all_attack()}

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
Dunsparce = 305
Dudunsparce = 66
Dudunsparce_ex = 306
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
    TEMPO_RACE = 11
    BACKUP_SETUP = 12
    ANTI_ENGINE = 13
    FAT_WALL = 14

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


class TurnIntent(IntEnum):
    """Single-turn objective used to prevent unrelated, low-tempo actions."""
    SETUP = 0
    COMPLETE_EVOLUTION = 1
    COMPLETE_ATTACKER = 2
    TAKE_PRIZES = 3
    RECOVER_BOARD = 4
    BREAK_WALL = 5
    DISRUPT = 6
    STALL = 7
    LOCK = 8


turn_intent = TurnIntent.SETUP
_lookahead_active = False
_lookahead_turn_key: tuple[int, int] | None = None
_lookahead_calls_this_turn = 0
LOOKAHEAD_MAX_CALLS_PER_TURN = 0
LOOKAHEAD_BRANCHES = 3
LOOKAHEAD_MAX_STEPS = 14
LOOKAHEAD_MAX_MAIN_ACTIONS = 4
LOOKAHEAD_SCENARIOS = 1
LOOKAHEAD_MAX_TURN_DELTA = 1


def _snapshot_runtime_globals() -> dict:
    """Save mutable rule-engine state before speculative search rollouts."""
    return {
        "machine_state": machine_state,
        "machine_transitions": defaultdict(int, machine_transitions),
        "opponent_phantom_seen": opponent_phantom_seen,
        "mirror_profile": mirror_profile,
        "opponent_seen_ids": set(opponent_seen_ids),
        "opponent_seen_counts": defaultdict(int, opponent_seen_counts),
        "budew_stall_uses": budew_stall_uses,
        "can_switch": can_switch,
        "can_attack": can_attack,
        "can_main_attack": can_main_attack,
        "can_energy_attach": can_energy_attach,
        "use_support": use_support,
        "bench_attacker": bench_attacker,
        "pre_turn_log": list(pre_turn_log),
        "current_turn_log": list(current_turn_log),
        "prize": list(prize),
        "card_counts": defaultdict(int, card_counts),
        "serial_set": set(serial_set),
        "plan_a": (plan_a.attack, list(plan_a.counter)),
        "plan_b": (plan_b.attack, list(plan_b.counter)),
        "turn_intent": turn_intent,
        "lookahead_turn_key": _lookahead_turn_key,
        "lookahead_calls": _lookahead_calls_this_turn,
    }


def _restore_runtime_globals(saved: dict) -> None:
    global machine_state, machine_transitions, opponent_phantom_seen, mirror_profile
    global opponent_seen_ids, opponent_seen_counts, budew_stall_uses
    global can_switch, can_attack, can_main_attack, can_energy_attach, use_support
    global bench_attacker, pre_turn_log, current_turn_log, prize, card_counts, serial_set
    global turn_intent, _lookahead_turn_key, _lookahead_calls_this_turn
    machine_state = saved["machine_state"]
    machine_transitions = saved["machine_transitions"]
    opponent_phantom_seen = saved["opponent_phantom_seen"]
    mirror_profile = saved["mirror_profile"]
    opponent_seen_ids = saved["opponent_seen_ids"]
    opponent_seen_counts = saved["opponent_seen_counts"]
    budew_stall_uses = saved["budew_stall_uses"]
    can_switch = saved["can_switch"]
    can_attack = saved["can_attack"]
    can_main_attack = saved["can_main_attack"]
    can_energy_attach = saved["can_energy_attach"]
    use_support = saved["use_support"]
    bench_attacker = saved["bench_attacker"]
    pre_turn_log = saved["pre_turn_log"]
    current_turn_log = saved["current_turn_log"]
    prize = saved["prize"]
    card_counts = saved["card_counts"]
    serial_set = saved["serial_set"]
    plan_a.attack, plan_a.counter = saved["plan_a"]
    plan_b.attack, plan_b.counter = saved["plan_b"]
    turn_intent = saved["turn_intent"]
    _lookahead_turn_key = saved["lookahead_turn_key"]
    _lookahead_calls_this_turn = saved["lookahead_calls"]


# Replay-trained generic value model.
# The weights were fitted offline from both-player perspectives in the ten supplied
# loss replays.  Card-specific identity features were intentionally excluded so the
# evaluator learns board quality rather than memorising a matchup name.
REPLAY_VALUE_WEIGHTS = (
    -0.00000505, 0.702975413, 0.438452154, 0.408285297, 1.73549055,
    1.60617637, 1.42813972, 1.26356506, -0.0881423166,
    1.06154334, -1.10419419, 2.23238534, -4.03523271,
    0.111734883, -0.295282980, -0.218001006,
)

# Opponent deck priors learned from the uploaded replays.  They are used only to
# create legal hidden-card determinizations for search; visible information always
# overrides the prior.
REPLAY_OPPONENT_MODELS = {
    "archaludon": {8:14,57:1,169:4,190:4,666:4,1097:3,1121:4,1122:4,1147:4,1152:4,1159:1,1182:2,1185:4,1227:4,1244:3},
    "alakazam": {5:2,19:4,66:2,140:1,305:3,343:1,741:4,742:4,743:3,1079:3,1081:3,1086:4,1097:1,1129:2,1152:4,1182:2,1184:1,1197:1,1225:4,1231:4,1266:2},
    "hop": {11:4,12:1,19:4,65:4,66:3,272:1,304:2,311:1,343:1,878:4,879:3,1086:4,1097:3,1122:2,1123:1,1152:4,1171:4,1182:2,1194:2,1210:2,1227:4,1255:4},
    "lucario": {6:13,673:3,674:3,675:2,676:3,677:4,678:4,1102:4,1123:3,1141:2,1142:3,1152:4,1159:1,1182:2,1192:4,1227:3,1252:2},
    "marnie": {7:10,104:2,112:4,646:4,647:3,648:3,860:2,1079:3,1080:1,1086:4,1097:3,1122:1,1137:1,1152:4,1182:2,1219:4,1227:4,1231:1,1259:4},
    "venusaur": {1:12,96:4,140:1,650:2,651:2,652:2,709:2,710:2,917:2,1071:2,1094:4,1097:2,1121:4,1122:2,1159:1,1182:3,1197:1,1213:2,1227:4,1229:2,1261:4},
    # Learned from the two newly supplied Crustle loss replays.  This is a
    # conservative 60-card shell used only for hidden-zone determinizations.
    "crustle": {1:11,11:4,344:4,345:4,1086:4,1117:4,1120:4,1121:4,
                 1097:3,1213:3,1227:3,1225:3,1123:2,1122:2,1081:2,
                 1182:2,1159:1},
}
PROFILE_ENERGY = {"archaludon":8, "alakazam":19, "hop":19, "lucario":6,
                  "marnie":7, "venusaur":1, "crustle":1, "unknown":5}
PROFILE_DAMAGE_CEILING = {"archaludon":160, "alakazam":300, "hop":130,
                          "lucario":270, "marnie":210, "venusaur":240,
                          "dragapult":260, "crustle":120, "unknown":180}


def _profile_from_ids(ids: set[int]) -> str:
    if ids & {741,742,743}: return "alakazam"
    if ids & {646,647,648}: return "marnie"
    if ids & {673,674,675,676,677,678}: return "lucario"
    if ids & {878,879,304,311,65}: return "hop"
    if ids & {57,169,190,666}: return "archaludon"
    if ids & {650,651,652,917,709,710,96}: return "venusaur"
    if ids & {119,120,121}: return "dragapult"
    if ids & {344,345}: return "crustle"
    return "unknown"


def _remove_known_card(pool: Counter, card: Card | Pokemon | None, owner: int) -> None:
    if card is None:
        return
    # Pokemon observations do not expose playerIndex in the public dataclass.
    # They are reached through an owner's field, so treat the supplied owner as
    # authoritative.  The previous direct attribute access disabled the hidden
    # determinization path whenever a Pokémon was already in play.
    card_owner = getattr(card, "playerIndex", owner)
    if card_owner == owner and pool[card.id] > 0:
        pool[card.id] -= 1
    if isinstance(card, Pokemon):
        for attached in card.energyCards:
            _remove_known_card(pool, attached, owner)
        for attached in card.tools:
            _remove_known_card(pool, attached, owner)
        for previous in card.preEvolution:
            _remove_known_card(pool, previous, owner)


def _remaining_pool_from_model(state: State, player: int, model: Counter) -> list[int]:
    p = state.players[player]
    pool = Counter(model)
    for card in (p.hand or []):
        _remove_known_card(pool, card, player)
    for card in (p.discard or []):
        _remove_known_card(pool, card, player)
    for card in (p.active or []):
        _remove_known_card(pool, card, player)
    for card in (p.bench or []):
        _remove_known_card(pool, card, player)
    for card in (state.stadium or []):
        _remove_known_card(pool, card, player)
    remaining: list[int] = []
    for cid, count in pool.items():
        if count > 0:
            remaining.extend([cid] * count)
    return remaining


def _own_hidden_cards(obs: Observation, scenario: int) -> tuple[list[int], list[int]]:
    state = obs.current
    me = state.yourIndex
    mine = state.players[me]
    remaining = _remaining_pool_from_model(state, me, Counter(my_deck))
    needed = mine.deckCount + len(mine.prize)
    if len(remaining) < needed:
        remaining.extend([Basic_Psychic_Energy] * (needed - len(remaining)))
    elif len(remaining) > needed:
        remaining = remaining[:needed]
    priority = {Dragapult_ex:100,Drakloak:96,Dreepy:90,Basic_Fire_Energy:88,
                Basic_Psychic_Energy:87,Crispin:83,Ultra_Ball:80,Poke_Pad:76,
                Night_Stretcher:72,Boss_Orders:68,Munkidori:60,Basic_Darkness_Energy:55}
    if scenario == 0:
        remaining.sort(key=lambda cid:(priority.get(cid,10),-cid), reverse=True)
    else:
        # Risk determinization: one critical piece can be prized and the draw order
        # is less favourable.  This prevents the search from assuming perfect draws.
        remaining.sort(key=lambda cid:(priority.get(cid,10),-cid))
        rotate = (state.turn * 7 + mine.handCount) % max(1, len(remaining))
        remaining = remaining[rotate:] + remaining[:rotate]
    deck = remaining[:mine.deckCount]
    prize_cards = remaining[mine.deckCount:mine.deckCount + len(mine.prize)]
    while len(prize_cards) < len(mine.prize):
        prize_cards.append(Basic_Psychic_Energy)
    return deck, prize_cards


def _opponent_hidden_cards(obs: Observation, profile: str, scenario: int) -> tuple[list[int], list[int], list[int], list[int]]:
    state = obs.current
    me = state.yourIndex
    op_index = 1 - me
    other = state.players[op_index]
    model = Counter(REPLAY_OPPONENT_MODELS.get(profile, {}))
    energy_id = PROFILE_ENERGY.get(profile, Basic_Psychic_Energy)
    if not model:
        # Unknown matchup: preserve all visible identities and use a conservative
        # basic-energy shell for the hidden zones.
        model = Counter({energy_id:60})
    remaining = _remaining_pool_from_model(state, op_index, model)
    needed = other.deckCount + len(other.prize) + other.handCount
    if len(remaining) < needed:
        remaining.extend([energy_id] * (needed - len(remaining)))
    elif len(remaining) > needed:
        remaining = remaining[:needed]

    threat_ids = {
        "archaludon": {666:100,169:90,1185:88,1227:75,1182:70,8:60},
        "alakazam": {743:100,742:94,1079:92,1231:84,1225:82,1182:76,19:70},
        "hop": {879:100,1171:88,1194:84,1210:80,1182:75,19:65},
        "lucario": {678:100,677:92,1142:88,1192:84,1182:78,6:70},
        "marnie": {648:100,647:94,1079:90,1219:86,1182:76,7:70,112:68},
        "venusaur": {652:100,710:94,96:90,1121:86,1227:82,1182:78,1:70},
        "crustle": {345:100,344:96,1121:90,1120:88,1225:84,1213:82,
                    1182:78,1159:76,11:72,1:68},
    }.get(profile, {})
    if scenario == 1:
        remaining.sort(key=lambda cid:(threat_ids.get(cid,10),-cid), reverse=True)
    else:
        remaining.sort(key=lambda cid:(threat_ids.get(cid,10),-cid))
        rotate = (state.turn * 5 + other.handCount * 3) % max(1, len(remaining))
        remaining = remaining[rotate:] + remaining[:rotate]

    hand = remaining[:other.handCount]
    tail = remaining[other.handCount:]
    deck = tail[:other.deckCount]
    prize_cards = tail[other.deckCount:other.deckCount + len(other.prize)]
    while len(hand) < other.handCount: hand.append(energy_id)
    while len(deck) < other.deckCount: deck.append(energy_id)
    while len(prize_cards) < len(other.prize): prize_cards.append(energy_id)
    active: list[int] = []
    if other.active and other.active[0] is None:
        basic_candidates = [cid for cid in hand + deck if card_table[cid].cardType == CardType.POKEMON and card_table[cid].basic]
        active = [basic_candidates[0] if basic_candidates else Dreepy]
    return deck, prize_cards, hand, active


def _hidden_prediction_scenarios(obs: Observation) -> list[tuple[list[int], list[int], list[int], list[int], list[int], list[int]]]:
    state = obs.current
    me = state.yourIndex
    other = state.players[1-me]
    visible_ids = {p.id for p in list(other.active or []) + list(other.bench or []) if p is not None}
    visible_ids |= set(opponent_seen_ids)
    profile = _profile_from_ids(visible_ids)
    out = []
    for scenario in range(LOOKAHEAD_SCENARIOS):
        your_deck, your_prize = _own_hidden_cards(obs, scenario)
        op_deck, op_prize, op_hand, op_active = _opponent_hidden_cards(obs, profile, scenario)
        out.append((your_deck, your_prize, op_deck, op_prize, op_hand, op_active))
    return out


def _field(state: State, player: int) -> list[Pokemon]:
    p = state.players[player]
    return [pkm for pkm in list(p.active or []) + list(p.bench or []) if pkm is not None]


def _learned_state_margin(state: State, player: int) -> float:
    mine = state.players[player]
    other = state.players[1-player]
    my_field = _field(state, player)
    op_field = _field(state, 1-player)
    def total_damage(field):
        return sum(max(0, card_table[p.id].hp - p.hp) for p in field)
    def total_hp(field): return sum(max(0,p.hp) for p in field)
    def max_hp(field): return sum(card_table[p.id].hp for p in field)
    def ready(field): return sum(1 for p in field if len(p.energyCards) >= 2)
    def stage2(field): return sum(1 for p in field if card_table[p.id].stage2)
    def ex_count(field): return sum(1 for p in field if card_table[p.id].ex)
    def low(field): return sum(1 for p in field if p.hp <= 70)
    def active_frac(owner):
        if not owner.active or owner.active[0] is None: return 0.0
        p=owner.active[0]
        return p.hp / max(1,card_table[p.id].hp)
    my_bench = [p for p in (mine.bench or []) if p is not None]
    op_bench = [p for p in (other.bench or []) if p is not None]
    features = (
        1.0,
        (len(other.prize)-len(mine.prize))/6.0,
        (len(my_field)-len(op_field))/6.0,
        (len(my_bench)-len(op_bench))/5.0,
        (mine.handCount-other.handCount)/15.0,
        (mine.deckCount-other.deckCount)/60.0,
        (total_hp(my_field)-total_hp(op_field))/1000.0,
        (max_hp(my_field)-max_hp(op_field))/1000.0,
        (total_damage(op_field)-total_damage(my_field))/1000.0,
        (sum(len(p.energyCards) for p in my_field)-sum(len(p.energyCards) for p in op_field))/10.0,
        (ready(my_field)-ready(op_field))/4.0,
        (stage2(my_field)-stage2(op_field))/4.0,
        (ex_count(my_field)-ex_count(op_field))/4.0,
        (low(op_field)-low(my_field))/5.0,
        active_frac(mine)-active_frac(other),
        (sum(max(0,card_table[p.id].hp-p.hp) for p in op_bench)-sum(max(0,card_table[p.id].hp-p.hp) for p in my_bench))/500.0,
    )
    return sum(w*x for w,x in zip(REPLAY_VALUE_WEIGHTS,features))


def _attack_energy_ok(pkm: Pokemon, attack_id: int, extra: int = 0) -> bool:
    attack = attack_table.get(attack_id)
    if attack is None: return False
    available = Counter(pkm.energies)
    flexible = extra
    for required in attack.energies:
        if required == 0:
            if sum(available.values()) > 0:
                any_id = next(iter(available))
                available[any_id] -= 1
                if available[any_id] <= 0: del available[any_id]
            elif flexible > 0:
                flexible -= 1
            else:
                return False
        elif available[required] > 0:
            available[required] -= 1
            if available[required] <= 0: del available[required]
        elif flexible > 0:
            flexible -= 1
        else:
            return False
    return True


def _effective_attack_damage(pkm: Pokemon, attack_id: int, hand_count: int) -> int:
    attack = attack_table.get(attack_id)
    if attack is None: return 0
    if attack_id == 1072: return hand_count * 20  # Alakazam: Powerful Hand
    if attack_id == 1267: return 130
    damage = attack.damage
    if attack_id == 154: damage += 60
    if attack_id == 937: damage += 30
    return damage


def _opponent_threat(state: State, root_player: int) -> tuple[int,float]:
    other = state.players[1-root_player]
    visible = _field(state,1-root_player)
    ids={p.id for p in visible}|set(opponent_seen_ids)
    profile=_profile_from_ids(ids)
    max_damage=0
    for pkm in visible:
        for aid in card_table[pkm.id].attacks:
            if _attack_energy_ok(pkm,aid,extra=1):
                max_damage=max(max_damage,_effective_attack_damage(pkm,aid,other.handCount))
    max_damage=max(max_damage, int(PROFILE_DAMAGE_CEILING.get(profile,180)*0.62))
    if profile == "alakazam":
        # The new replay showed that evaluating only the current hand misses the
        # Psychic Draw / Dudunsparce refill before Powerful Hand.  Reserve three
        # additional cards when a live Alakazam engine is visible.
        live_draw_engine = any(p.id in (66, 742, 743) for p in visible)
        projected_hand = other.handCount + (3 if live_draw_engine else 1)
        max_damage = max(max_damage, min(300, projected_hand * 20))
    model=REPLAY_OPPONENT_MODELS.get(profile,{})
    boss_total=model.get(Boss_Orders,2)
    seen=min(boss_total,opponent_seen_counts[Boss_Orders])
    remaining=max(0,boss_total-seen)
    unknown=max(1,other.deckCount+other.handCount)
    boss_prob=min(0.85, remaining*other.handCount/unknown)
    return max_damage,boss_prob


def _search_board_value(obs: Observation, root_player: int, root_my_prize: int,
                        root_op_prize: int, steps: int) -> float:
    state=obs.current
    if state is None: return -1e9
    mine=state.players[root_player]
    other=state.players[1-root_player]
    my_field=_field(state,root_player)
    op_field=_field(state,1-root_player)
    if not op_field or len(other.prize)==0: return 1e9-steps*100
    if not my_field or len(mine.prize)==0: return -1e9
    value=0.0
    value+=(root_my_prize-len(mine.prize))*310000
    value-=(root_op_prize-len(other.prize))*310000
    value+=_learned_state_margin(state,root_player)*4500
    value+=(mine.handCount-other.handCount)*260
    if mine.deckCount<=3: value-=(4-mine.deckCount)*22000

    ready_drag=0
    backup_lines=0
    for pkm in my_field:
        ids={e.id for e in pkm.energyCards}
        damage=max(0,card_table[pkm.id].hp-pkm.hp)
        value-=damage*52
        if pkm.id==Dragapult_ex:
            value+=33000
            ready=Basic_Fire_Energy in ids and Basic_Psychic_Energy in ids
            if ready:
                ready_drag+=1; value+=52000
            elif len(ids & {Basic_Fire_Energy,Basic_Psychic_Energy})==1:
                value+=18000
            duplicate=sum(1 for e in pkm.energyCards if e.id in (Basic_Fire_Energy,Basic_Psychic_Energy))-len(ids & {Basic_Fire_Energy,Basic_Psychic_Energy})
            value-=duplicate*14000
        elif pkm.id==Drakloak:
            backup_lines+=1; value+=23000+len(pkm.energyCards)*8000
        elif pkm.id==Dreepy:
            backup_lines+=1; value+=11500+len(pkm.energyCards)*5500
        elif pkm.id==Munkidori:
            value+=6500+(12000 if Basic_Darkness_Energy in ids else 0)
        elif pkm.id==Budew:
            value+=2500 if state.turn<=4 else -11000
        elif card_table[pkm.id].ex:
            value-=6500
    value+=min(2,backup_lines)*15000
    if ready_drag==0 and backup_lines==0: value-=90000
    if len(my_field)==1 and my_field[0].hp<=80: value-=125000

    for pkm in op_field:
        damage=max(0,card_table[pkm.id].hp-pkm.hp)
        value+=damage*58
        if pkm.hp<=60: value+=prize_count(pkm,False)*10500

    threat,boss_prob=_opponent_threat(state,root_player)
    active_hp=mine.active[0].hp if mine.active and mine.active[0] is not None else 0
    if active_hp and threat>=active_hp: value-=18000
    vulnerable_ex=sum(1 for p in (mine.bench or []) if p is not None and card_table[p.id].ex and p.hp<=threat)
    value-=boss_prob*vulnerable_ex*12000
    value-=steps*720
    return value


def _lookahead_candidate_indices(obs: Observation, scores: list[float]) -> list[int]:
    select=obs.select
    ranked=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    candidates=[]
    for i in ranked:
        option=select.option[i]
        if scores[i]<0 and option.type not in (OptionType.ATTACK,OptionType.END): continue
        if option.type in (OptionType.PLAY,OptionType.ATTACH,OptionType.EVOLVE,OptionType.ABILITY,OptionType.RETREAT,OptionType.ATTACK,OptionType.END):
            candidates.append(i)
        if len(candidates)>=LOOKAHEAD_BRANCHES: break
    for wanted in (OptionType.ATTACK,OptionType.END,OptionType.EVOLVE,OptionType.ATTACH):
        for i,option in enumerate(select.option):
            if option.type==wanted and i not in candidates:
                candidates.append(i); break
    return candidates[:LOOKAHEAD_BRANCHES+2]


def _generic_card_value(card: Card | Pokemon | None) -> float:
    if card is None: return -1e6
    data=card_table[card.id]
    if data.cardType==CardType.POKEMON:
        value=data.hp+(900 if data.stage2 else 500 if data.stage1 else 250)
        if data.ex: value+=250
        if isinstance(card,Pokemon): value+=len(card.energyCards)*180+card.hp
        return value
    if data.cardType==CardType.BASIC_ENERGY: return 380
    if data.cardType==CardType.SPECIAL_ENERGY: return 430
    if data.cardType==CardType.SUPPORTER: return 520
    if data.cardType==CardType.ITEM: return 470
    if data.cardType==CardType.STADIUM: return 300
    return 250


# Tabular opponent action policy fitted from 743 decisions in the supplied loss replays.
REPLAY_POLICY_GLOBAL = {'1|3|0': 2613.64, '0|7|1122': -2276.79, '0|7|0': -4453.46, '0|7|1152': -1822.93, '0|7|1244': -8520.28, '0|14|0': -8892.54, '7|3|0': -3934.57, '0|7|169': 1446.35, '0|12|0': -12731.66, '0|7|1185': -3806.67, '0|8|8': -1446.35, '0|8|0': -10580.01, '0|13|0': -4362.58, '22|3|0': 798.6, '21|3|169': 0.0, '21|3|0': -1620.55, '7|3|1122': -4319.77, '7|3|1152': 474.94, '41|1|0': 5374.85, '41|2|0': -5374.85, '0|7|1086': -2488.65, '0|8|19': -11883.43, '5|3|0': 663.01, '30|6|0': 3703.85, '3|3|741': 0.0, '3|3|0': -1925.75, '0|7|1225': -9015.69, '0|9|742': -4245.99, '0|9|0': -3637.57, '0|7|1079': 611.38, '43|1|0': 7822.82, '43|2|0': -7822.82, '0|8|5': -9806.0, '0|7|1231': -2601.94, '0|13|11071': -5496.48, '37|9|743': 3806.67, '37|9|0': 3806.67, '0|7|305': -1446.35, '0|7|741': -2301.19, '0|9|743': -5884.84, '0|13|11072': -3433.5, '7|3|1086': -1566.55, '7|3|19': -5906.65, '7|3|1079': 955.18, '7|3|1231': -6354.93, '0|7|1266': 992.5, '0|7|140': -5808.93, '4|3|0': -3557.12, '4|3|743': 2438.85, '4|3|742': -3195.29, '0|7|1146': -7031.12, '0|7|1097': -2466.26, '0|7|1227': -3776.17, '0|7|1255': -756.44, '0|8|11': -4560.03, '0|7|1182': -7755.32, '0|7|304': -5150.2, '3|3|119': -3315.49, '0|9|879': -611.38, '0|7|65': -6142.7, '0|9|66': -3483.56, '0|7|1210': -4362.58, '0|7|1123': -6821.21, '0|13|11267': -3445.99, '0|10|0': -1789.7, '7|3|1182': 0.0, '7|3|11': -5150.2, '7|3|1227': -2002.26, '0|7|1194': -4761.85, '0|7|311': -8207.84, '0|8|1171': -10893.51, '3|3|121': 1446.35, '4|3|343': -7031.12, '4|3|304': -5906.65, '4|3|879': 2438.85, '0|8|12': -6518.02, '1|3|741': 1446.35, '2|3|0': 756.44, '0|13|11070': -3806.67, '0|7|343': -8477.47, '0|7|1264': -1748.94, '0|8|1156': -10300.4, '4|3|305': -6518.02, '0|8|13': -9165.13, '7|3|1225': 1446.35, '3|3|112': -992.5, '0|8|6': -10366.5, '0|7|676': -4319.77, '8|3|6': 1880.92, '8|3|0': 1880.92, '0|7|1102': 3195.29, '0|7|1142': -756.44, '0|7|677': -3806.67, '0|9|674': -3703.85, '0|9|678': -3315.49, '0|13|10981': -5150.2, '0|13|10982': -6404.74, '21|3|675': -5906.65, '21|3|676': -5906.65, '21|3|678': 0.0, '21|3|674': -992.5, '21|3|673': -5150.2, '7|3|1102': -6518.02, '7|3|1142': -5150.2, '0|7|1192': -8063.99, '0|7|1252': -10300.4, '0|7|1141': -4362.58, '3|3|140': 1446.35, '0|13|10983': -4761.85, '0|7|1184': -3806.67, '7|3|1197': 5906.65, '0|8|7': -9142.59, '0|7|112': -3606.99, '0|7|646': -1689.82, '0|9|647': -955.18, '0|7|860': -6821.21, '16|3|0': -2328.81, '40|0|30001': -10125.02, '40|0|0': -1543.21, '40|0|30002': -1773.79, '40|0|30003': 8804.91, '13|3|119': -3835.83, '13|3|0': -4557.75, '0|7|1219': -6105.38, '13|3|235': -4319.77, '13|3|120': -3421.43, '13|3|343': 3195.29, '13|3|112': -5150.2, '0|9|648': -955.18, '0|7|1259': -513.1, '21|3|648': -1689.82, '21|3|646': 1367.82, '0|13|10937': -5592.28, '15|3|120': -7473.2, '15|3|0': -3093.9, '15|3|112': 0.0, '7|3|1219': -2438.85, '4|3|648': 1446.35, '16|3|860': 0.0, '13|3|121': -8804.91, '13|3|140': -5496.48, '0|7|1080': -1446.35, '15|3|140': 0.0, '16|3|648': -5781.75, '16|3|112': -658.73, '15|3|121': -5906.65, '16|3|104': -4761.85, '13|3|1071': -4761.85, '21|3|647': -1446.35, '7|3|1259': -2438.85, '38|0|0': -1446.35, '0|7|1094': -1748.94, '0|8|1': -10437.08, '0|7|1261': 0.0, '0|7|96': -8063.99, '22|3|1': 756.44, '0|9|709': -3806.67, '33|6|0': -2438.85, '0|13|10941': -1773.79, '7|3|1094': 513.1, '0|7|1121': -10125.02, '7|3|1261': -5150.2, '0|8|1159': -1446.35}
REPLAY_POLICY_PROFILE = {'archaludon': {'42|1|0': -4157.71, '42|2|0': 4157.71, '0|7|1122': -2711.35, '0|7|0': -3916.08, '0|7|1152': -2360.32, '0|7|1244': -8520.28, '0|14|0': -10624.73, '7|3|0': -2243.37, '0|7|169': 1446.35, '0|12|0': -9532.56, '0|7|1185': -3806.67, '0|8|8': -1446.35, '0|8|0': -1446.35, '22|3|0': -3269.28, '21|3|169': 0.0, '21|3|0': 0.0, '7|3|1122': -1446.35, '7|3|1152': -4157.71}, 'alakazam': {'41|1|0': 7031.12, '41|2|0': -7031.12, '1|3|0': 1748.94, '0|7|1086': -1047.08, '0|7|0': -4202.32, '0|7|1152': -1320.1, '0|8|19': -12526.67, '0|8|0': -11575.71, '0|14|0': -9066.28, '7|3|0': -3119.23, '5|3|0': -230.73, '0|12|0': -13498.14, '30|6|0': 4157.71, '3|3|741': 0.0, '3|3|0': -1343.53, '0|7|1225': -9015.69, '0|9|742': -4245.99, '0|9|0': -4722.11, '0|7|1079': 611.38, '43|1|0': 12937.76, '43|2|0': -12937.76, '0|8|5': -9806.0, '0|7|1231': -2750.02, '0|13|11071': -5496.48, '0|13|0': -3819.44, '37|9|743': 3806.67, '37|9|0': 3806.67, '0|7|305': -1446.35, '0|7|741': -2301.19, '0|9|743': -5884.84, '0|13|11072': -3433.5, '7|3|1152': -442.08, '7|3|1086': -1566.55, '7|3|19': -5906.65, '7|3|1079': 955.18, '7|3|1231': -6354.93, '0|7|1266': 992.5, '0|7|140': -8520.28, '4|3|0': -3421.43, '4|3|743': 2438.85, '4|3|742': -3195.29, '0|7|1146': -7031.12, '0|7|1097': -1367.82, '1|3|741': 1446.35, '2|3|741': 0.0, '2|3|0': 1446.35, '0|13|11070': -3806.67, '0|9|66': -3795.74, '0|7|1182': -7787.56, '0|10|0': -5111.76, '0|7|343': -8477.47, '1|3|305': 0.0, '0|7|1264': -1748.94, '0|8|1156': -10300.4, '4|3|305': -6518.02, '4|3|741': 0.0, '0|8|13': -9165.13, '7|3|1225': 1446.35, '3|3|112': 0.0, '3|3|120': -4157.71, '4|3|343': -4157.71, '0|7|1184': -3806.67, '7|3|1197': 5906.65}, 'hop': {'41|1|0': -4157.71, '41|2|0': 4157.71, '0|7|1227': -5408.93, '0|7|0': -4705.56, '0|7|1255': -756.44, '0|8|11': -4560.03, '0|8|0': -8543.1, '0|7|1152': -3057.64, '0|7|1182': -4362.58, '0|14|0': -9010.02, '7|3|0': -3764.79, '0|7|304': -5150.2, '0|12|0': -7031.12, '0|7|878': 0.0, '3|3|119': -2438.85, '3|3|0': -1954.91, '30|6|0': 4157.71, '3|3|878': -4157.71, '0|9|879': -611.38, '0|9|0': -1446.35, '0|7|1122': -2360.32, '0|8|19': -7124.56, '0|7|65': -6142.7, '5|3|0': 2501.44, '0|9|66': -1954.91, '0|7|1210': -4362.58, '0|7|1123': 0.0, '3|3|879': 0.0, '0|13|11267': -3445.99, '0|13|0': -3583.65, '0|10|0': -2763.18, '7|3|1152': -611.38, '7|3|1182': 1446.35, '7|3|11': -5150.2, '7|3|1255': -4157.71, '0|7|1194': -4761.85, '0|7|311': -8207.84, '0|7|1086': -3370.08, '0|8|1171': -10893.51, '4|3|343': -5906.65, '4|3|0': -3159.64, '4|3|304': -5906.65, '4|3|879': 2438.85, '0|7|1097': -2438.85, '4|3|878': -4157.71, '0|8|12': -6518.02}, 'lucario': {'41|1|0': 4157.71, '41|2|0': -4157.71, '0|8|6': -10366.5, '0|8|0': -10366.5, '0|7|1152': -611.38, '0|7|0': -5068.32, '0|14|0': -11117.51, '7|3|0': -4492.21, '0|7|676': -4319.77, '0|10|0': 9066.28, '8|3|6': 1880.92, '8|3|0': 1880.92, '0|7|1102': 3195.29, '0|7|1142': -756.44, '0|7|677': -3806.67, '0|7|673': 4157.71, '0|9|674': -3703.85, '0|9|0': -3835.83, '0|7|1227': -3654.71, '0|9|678': -3315.49, '0|7|1182': -8063.99, '0|13|10981': -5150.2, '0|13|0': -6021.71, '0|13|10982': -6404.74, '43|1|0': 5906.65, '43|2|0': -5906.65, '3|3|119': -1446.35, '3|3|0': -1566.55, '0|7|1123': -9307.91, '22|3|0': 7861.55, '21|3|675': -5906.65, '21|3|0': -4032.81, '21|3|676': -5906.65, '21|3|678': 0.0, '21|3|674': -992.5, '21|3|673': -5150.2, '7|3|1152': 1880.92, '7|3|1102': -6518.02, '7|3|1142': -5150.2, '0|7|1192': -8063.99, '0|7|1252': -10300.4, '0|7|1141': -4362.58, '0|13|10983': -4761.85, '0|12|0': -5374.85, '30|6|0': 1748.94, '3|3|674': 0.0, '4|3|0': -3195.29, '4|3|678': 0.0}, 'marnie': {'41|1|0': 5906.65, '41|2|0': -5906.65, '1|3|860': 0.0, '1|3|0': 0.0, '2|3|860': 0.0, '2|3|0': 0.0, '2|3|112': 0.0, '0|7|1152': -1566.55, '0|7|0': -3850.75, '0|8|7': -9142.59, '0|8|0': -9142.59, '0|7|112': -3606.99, '0|14|0': -7031.12, '7|3|0': -5343.0, '0|7|646': -1689.82, '0|9|647': -955.18, '0|9|0': -727.65, '0|7|860': -6821.21, '0|10|0': 242.77, '16|3|0': -2328.81, '40|0|30001': -10125.02, '40|0|0': -1543.21, '40|0|30002': -1773.79, '40|0|30003': 8804.91, '13|3|119': -3835.83, '13|3|0': -4557.75, '0|7|1219': -6105.38, '16|3|647': 4157.71, '13|3|235': -4319.77, '13|3|120': -3421.43, '13|3|343': 3195.29, '0|7|1227': -1343.53, '13|3|112': -5150.2, '0|9|648': -955.18, '0|7|1259': -513.1, '43|1|0': 1124.47, '43|2|0': -1124.47, '22|3|0': 2398.91, '21|3|648': -1689.82, '21|3|0': -727.65, '21|3|646': 1367.82, '0|13|10937': -5592.28, '0|13|0': -5592.28, '0|12|0': -15275.22, '15|3|120': -7473.2, '15|3|0': -3093.9, '15|3|343': 0.0, '15|3|112': 0.0, '7|3|1152': 3806.67, '7|3|1219': -2438.85, '7|3|1227': -1343.53, '4|3|648': 1446.35, '4|3|0': -2873.41, '4|3|860': -4157.71, '4|3|112': -4157.71, '0|7|1086': -3315.49, '0|7|1097': -2711.35, '16|3|860': 0.0, '13|3|121': -8804.91, '13|3|140': -5496.48, '5|3|0': 4157.71, '0|7|1080': -1446.35, '15|3|140': 0.0, '0|7|1182': -6142.7, '16|3|648': -5781.75, '16|3|112': -658.73, '7|3|112': -4157.71, '3|3|112': -4157.71, '3|3|0': -2360.32, '3|3|140': 0.0, '15|3|121': -5906.65, '0|7|1231': 0.0, '16|3|104': -4761.85, '13|3|1071': -4761.85, '21|3|647': -1446.35, '7|3|1259': -2438.85, '7|3|1122': -5906.65}, 'venusaur': {'41|1|0': 4157.71, '41|2|0': -4157.71, '38|0|0': -1446.35, '0|7|1094': -1748.94, '0|7|0': -5267.11, '0|8|1': -10437.08, '0|8|0': -9532.56, '0|7|1261': 0.0, '0|14|0': -10008.18, '7|3|0': -3284.53, '0|7|917': 0.0, '0|7|96': -8063.99, '0|10|0': -2335.3, '22|3|1': 756.44, '22|3|0': 756.44, '0|9|651': 0.0, '0|9|0': -1124.47, '0|9|709': -3806.67, '0|13|10939': -4157.71, '0|13|0': -2276.79, '33|6|0': -2438.85, '21|3|652': 4157.71, '21|3|0': 0.0, '21|3|917': -4157.71, '0|13|10941': -1773.79, '0|12|0': -12274.76, '21|3|710': 4157.71, '21|3|96': -4157.71, '7|3|1094': 513.1, '0|7|1121': -10125.02, '7|3|1261': -5150.2, '0|7|1182': -7473.2, '0|8|1159': -1446.35, '43|1|0': 4157.71, '43|2|0': -4157.71, '0|7|1213': 0.0}}

def _rollout_option_identity(obs: Observation, option) -> int:
    try:
        state=obs.current; me=state.yourIndex
        if option.type == OptionType.PLAY:
            return state.players[me].hand[option.index].id
        if option.type in (OptionType.ATTACH, OptionType.EVOLVE):
            card=get_card(obs, option.area, option.index, me)
            return card.id if card is not None else 0
        if option.type == OptionType.CARD:
            card=get_card(obs, option.area, option.index, option.playerIndex)
            return card.id if card is not None else 0
        if option.type == OptionType.ATTACK:
            return 10000 + option.attackId
        if option.type == OptionType.NUMBER:
            return 30000 + option.number
    except Exception:
        pass
    return 0

def _replay_policy_bonus(obs: Observation, option) -> float:
    state=obs.current
    if state is None or obs.select is None:
        return 0.0
    me=state.yourIndex
    ids={p.id for p in list(state.players[1-me].active or [])+list(state.players[1-me].bench or []) if p is not None}
    # During opponent rollout, classify the acting deck from its own visible board.
    own_ids={p.id for p in list(state.players[me].active or [])+list(state.players[me].bench or []) if p is not None}
    profile=_profile_from_ids(own_ids)
    identity=_rollout_option_identity(obs,option)
    key=f"{int(obs.select.context)}|{int(option.type)}|{identity}"
    fallback=f"{int(obs.select.context)}|{int(option.type)}|0"
    g=REPLAY_POLICY_GLOBAL.get(key,REPLAY_POLICY_GLOBAL.get(fallback,0.0))
    table=REPLAY_POLICY_PROFILE.get(profile,{})
    p=table.get(key,table.get(fallback,0.0))
    return 0.35*g+0.65*p

def _generic_rollout_action(obs: Observation) -> list[int]:
    select=obs.select
    state=obs.current
    if select is None or state is None: return []
    me=state.yourIndex
    mine=state.players[me]
    other=state.players[1-me]
    scored=[]
    for i,o in enumerate(select.option):
        score=0.0
        if o.type==OptionType.YES: score=10
        elif o.type==OptionType.NO: score=0
        elif o.type==OptionType.NUMBER: score=float(o.number)
        elif o.type==OptionType.ATTACK:
            active=mine.active[0] if mine.active and mine.active[0] is not None else None
            damage=_effective_attack_damage(active,o.attackId,mine.handCount) if active else 0
            target=other.active[0] if other.active and other.active[0] is not None else None
            score=80000+damage*120+(180000 if target and damage>=target.hp else 0)
        elif o.type==OptionType.EVOLVE:
            evo=get_card(obs,o.area,o.index,me)
            score=72000+(25000 if evo and card_table[evo.id].stage2 else 0)
        elif o.type==OptionType.ATTACH:
            target=get_card(obs,o.inPlayArea,o.inPlayIndex,me)
            score=54000+(_generic_card_value(target) if target else 0)+(10000 if o.inPlayArea==AreaType.ACTIVE else 0)
        elif o.type==OptionType.ABILITY: score=61000
        elif o.type==OptionType.PLAY:
            card=get_card(obs,AreaType.HAND,o.index,me)
            data=card_table[card.id] if card else None
            if data and data.cardType==CardType.SUPPORTER: score=58000
            elif data and data.cardType==CardType.ITEM: score=52000
            elif data and data.cardType==CardType.POKEMON: score=36000+(8000 if data.basic else 0)
            elif data and data.cardType==CardType.STADIUM: score=24000
            else: score=18000
        elif o.type==OptionType.RETREAT:
            bench_ready=any(len(p.energyCards)>=2 for p in (mine.bench or []) if p is not None)
            score=46000 if bench_ready else -1000
        elif o.type==OptionType.END: score=0
        elif o.type==OptionType.CARD:
            card=get_card(obs,o.area,o.index,o.playerIndex)
            value=_generic_card_value(card)
            if select.context in (SelectContext.DISCARD,SelectContext.TO_DECK,SelectContext.TO_DECK_BOTTOM): score=-value
            elif select.context in (SelectContext.TO_HAND,SelectContext.EVOLVES_TO): score=value
            elif select.context in (SelectContext.SWITCH,SelectContext.TO_ACTIVE):
                score=value+(20000 if isinstance(card,Pokemon) and len(card.energyCards)>=2 else 0)
            elif select.context in (SelectContext.DAMAGE_COUNTER,SelectContext.DAMAGE_COUNTER_ANY,SelectContext.DAMAGE):
                remain=max(10,select.remainDamageCounter*10)
                score=(150000 if isinstance(card,Pokemon) and card.hp<=remain else 0)+(10000-card.hp if isinstance(card,Pokemon) else 0)
            elif select.context in (SelectContext.ATTACH_FROM,SelectContext.ATTACH_TO): score=value
            else: score=value
        elif o.type in (OptionType.ENERGY,OptionType.ENERGY_CARD,OptionType.TOOL_CARD): score=10
        elif o.type==OptionType.SKILL: score=10
        score += _replay_policy_bonus(obs, o)
        scored.append((score,i))
    scored.sort(reverse=True)
    count=select.maxCount
    out=[]
    for score,i in scored:
        if len(out)>=count: break
        if score>=0 or len(out)<select.minCount:
            out.append(i)
    if len(out)<select.minCount:
        out=[i for _,i in scored[:select.minCount]]
    return out


def _force_end(obs: Observation) -> list[int] | None:
    if obs.select is None: return None
    for i,o in enumerate(obs.select.option):
        if o.type==OptionType.END: return [i]
    return None


def _rerank_with_lookahead(obs: Observation, scores: list[float]) -> list[float]:
    """Replay-trained, opponent-aware two-ply determinized search."""
    global _lookahead_active
    if _lookahead_active or obs.select is None or obs.current is None: return scores
    if obs.select.context!=SelectContext.MAIN or obs.select.maxCount!=1: return scores
    candidates=_lookahead_candidate_indices(obs,scores)
    if len(candidates)<2: return scores
    ranked_scores=sorted((scores[i] for i in candidates), reverse=True)
    other_ids={p.id for p in list(obs.current.players[1-obs.current.yourIndex].active or []) + list(obs.current.players[1-obs.current.yourIndex].bench or []) if p is not None}
    search_profile=_profile_from_ids(other_ids | set(opponent_seen_ids))
    # Opponent-turn rollouts in mirror, Crustle, and Alakazam repeatedly
    # reduced A/B win rate.  Their replay corrections are therefore expressed
    # as bounded score terms, while deeper search remains active elsewhere.
    repetitive_profile = search_profile in ("dragapult", "crustle", "alakazam")
    if repetitive_profile:
        return scores
    local_max_steps = LOOKAHEAD_MAX_STEPS
    local_main_actions = LOOKAHEAD_MAX_MAIN_ACTIONS
    close_limit = 30000 if search_profile in ("marnie", "venusaur") else 52000
    if len(ranked_scores) >= 2 and ranked_scores[0] - ranked_scores[1] > close_limit:
        return scores
    saved=_snapshot_runtime_globals()
    created_ids:set[int]=set()
    branch_samples:defaultdict[int,list[float]]=defaultdict(list)
    try:
        root_player=obs.current.yourIndex
        root_turn=obs.current.turn
        root_my_prize=len(obs.current.players[root_player].prize)
        root_op_prize=len(obs.current.players[1-root_player].prize)
        for predictions in _hidden_prediction_scenarios(obs):
            root=search_begin(obs,*predictions)
            created_ids.add(root.searchId)
            for candidate in candidates:
                _restore_runtime_globals(saved)
                _lookahead_active=True
                steps=1
                main_counts=defaultdict(int)
                try:
                    node=search_step(root.searchId,[candidate])
                    created_ids.add(node.searchId)
                    while steps<local_max_steps:
                        sim_obs=node.observation
                        sim_state=sim_obs.current
                        if sim_state is None or sim_obs.select is None: break
                        mine=sim_state.players[root_player]; other=sim_state.players[1-root_player]
                        if (not _field(sim_state,root_player)) or (not _field(sim_state,1-root_player)): break
                        if sim_state.turn-root_turn>LOOKAHEAD_MAX_TURN_DELTA: break
                        if sim_obs.select.context==SelectContext.MAIN:
                            key=(sim_state.turn,sim_state.yourIndex)
                            main_counts[key]+=1
                            limit=local_main_actions if sim_state.yourIndex==root_player else local_main_actions+1
                            if main_counts[key]>limit:
                                action=_force_end(sim_obs)
                                if action is None: break
                            elif sim_state.yourIndex==root_player:
                                action=agent(asdict(sim_obs))
                            else:
                                action=_generic_rollout_action(sim_obs)
                        else:
                            action=agent(asdict(sim_obs)) if sim_state.yourIndex==root_player else _generic_rollout_action(sim_obs)
                        if not action and sim_obs.select.minCount>0: break
                        node=search_step(node.searchId,action)
                        created_ids.add(node.searchId)
                        steps+=1
                    branch_samples[candidate].append(_search_board_value(node.observation,root_player,root_my_prize,root_op_prize,steps)+scores[candidate]*0.02)
                except Exception:
                    branch_samples[candidate].append(-1e12)
        robust={}
        for i,vals in branch_samples.items():
            valid=[v for v in vals if v>-1e11]
            if valid:
                mean=sum(valid)/len(valid)
                robust[i]=0.68*mean+0.32*min(valid)
        if robust:
            best=max(robust,key=robust.get)
            ordered=sorted(robust.values(),reverse=True)
            second=ordered[1] if len(ordered)>1 else ordered[0]
            margin=max(2500.0,min(26000.0,robust[best]-second+4500.0))
            out=list(scores)
            out[best]=max(out)+margin
            return out
    except Exception:
        return scores
    finally:
        _lookahead_active=True
        _restore_runtime_globals(saved)
        for sid in sorted(created_ids,reverse=True):
            try: search_release(sid)
            except Exception: pass
        try: search_end()
        except Exception: pass
        _lookahead_active=False
    return scores


def _counter_allocation_values(obs: Observation, select, my_index: int) -> dict[int, tuple[int,int,int]]:
    """Exact marginal value of each 10-damage-counter placement.

    Returns option -> (maximum immediate Prizes obtainable with the full remaining
    counter budget after choosing it, Prizes of the chosen target, residual focus).
    No card names or matchup labels are used.
    """
    remaining=max(1,int(select.remainDamageCounter or 1))
    targets=[]
    for i,o in enumerate(select.option):
        if o.type!=OptionType.CARD:
            continue
        try:p=get_card(obs,o.area,o.index,o.playerIndex)
        except Exception:continue
        if p is None or p.hp<=0 or no_damage_counter(p):
            continue
        targets.append((i,p,o.area))
    out={}
    for chosen,pchosen,_ in targets:
        states=[]
        for i,p,area in targets:
            hp=max(0,p.hp-(10 if i==chosen else 0))
            need=(hp+9)//10 if hp>0 else 0
            pr=prize_count(p,False)
            states.append((need,pr,p.hp,hp,i,area))
        budget=max(0,remaining-1)
        dp=[0]*(budget+1)
        for need,pr,oldhp,hp,i,area in states:
            if hp<=0:
                continue
            if need<=budget:
                for b in range(budget,need-1,-1):
                    dp[b]=max(dp[b],dp[b-need]+pr)
        immediate=prize_count(pchosen,False) if pchosen.hp<=10 else 0
        total=immediate+(max(dp) if dp else 0)
        # Tie-break toward finishing valuable, already-damaged targets.
        chosen_pr=prize_count(pchosen,False)
        focus=chosen_pr*1000-max(0,pchosen.hp-10)
        out[chosen]=(total,immediate,focus)
    return out

def _bench_prize_budget(bench: list[Pokemon], counters: int=6) -> int:
    items=[]
    for p in bench:
        if p is None or p.hp<=0 or no_damage_counter(p):continue
        need=(p.hp+9)//10
        if need<=counters:items.append((need,prize_count(p,False)))
    dp=[0]*(counters+1)
    for need,pr in items:
        for b in range(counters,need-1,-1):dp[b]=max(dp[b],dp[b-need]+pr)
    return max(dp)

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
    if obs_dict.get("select") is None:
        return my_deck
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
    raw_hand_counts = defaultdict(int)
    for _c in (my_state.hand or []):
        raw_hand_counts[_c.id] += 1
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
    is_crustle_match = 345 in seen_opponent_ids
    # Public, card-effect based wall detection.  A lone Dwebble tech no longer
    # activates the entire Crustle package; actual anti-ex walls share the same
    # Dudunsparce ex response.
    is_wall_match = bool(seen_opponent_ids & {158,207,330,345})
    is_lucario_match = bool(seen_opponent_ids & {673,674,675,676,677,678})
    is_rocket_match = bool(seen_opponent_ids & {15,17,400,401,414,431,432,463,473,474,891})
    is_dragapult_match = bool(seen_opponent_ids & {119,120,121})
    is_alakazam_match = bool(seen_opponent_ids & {741,742,743})
    is_marnie_match = bool(seen_opponent_ids & {646,647,648})
    is_starmie_match = bool(seen_opponent_ids & {860,861,1030,1031})
    is_archaludon_match = bool(seen_opponent_ids & {57,169,190,666})
    is_iono_match = bool(seen_opponent_ids & {265,268,269,270,271})
    is_cynthia_match = bool(seen_opponent_ids & {379,380,381,341,342,387})
    opponent_profile = ('crustle' if is_crustle_match else 'lucario' if is_lucario_match else
                        'dragapult' if is_dragapult_match else 'alakazam' if is_alakazam_match else
                        'marnie' if is_marnie_match else 'starmie' if is_starmie_match else
                        'archaludon' if is_archaludon_match else 'iono' if is_iono_match else
                        'cynthia' if is_cynthia_match else 'rocket' if is_rocket_match else 'unknown')
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
    opponent_ex_count = sum(1 for p in visible_opponent if card_table[p.id].ex or card_table[p.id].megaEx)
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
    opponent_threat_damage, opponent_boss_probability = _opponent_threat(state, my_index)
    active_ex_lethal = bool(
        active_pokemon is not None and card_table[active_pokemon.id].ex
        and len(op_state.prize) <= 2 and opponent_threat_damage >= active_pokemon.hp
    )
    target_priority_ids = {
        'dragapult': {Dreepy:7000,Drakloak:10000,Munkidori:8000},
        'lucario': {677:9500,673:8500,675:5500,676:5500},
        'alakazam': {741:11000,742:12000,743:6500},
        'marnie': {646:8000,647:10500,648:6500,Munkidori:8500},
        'starmie': {1030:9000,860:8000,Munkidori:7500},
        'crustle': {344:13000},
        'archaludon': {666:14000,57:11500,169:7500,190:4000},
        'iono': {265:10000,268:9500,269:8500,270:6500,271:10000},
        'cynthia': {379:10000,380:12000,381:7000,341:10000,342:12000,387:7000},
        'rocket': {400:10000,401:12000,15:7000,414:7500,463:7500},
    }.get(opponent_profile,{})
    opponent_active = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    wall_active_now = bool(opponent_active is not None and no_damage_dex(opponent_active.id))
    wall_bench_prizes_now = _bench_prize_budget(op_state.bench or [], 6)
    # Enter the dedicated wall-break line only when Phantom Dive cannot take a
    # Bench Prize right now.  Otherwise keep normal Dragapult pressure.
    wall_break_now = wall_active_now and wall_bench_prizes_now <= 0
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
        or (c.id == Dudunsparce_ex and is_wall_match and any(p.id == Dunsparce and not p.appearThisTurn for p in list(my_state.active)+list(my_state.bench) if p is not None))
        or (c.id == Dunsparce and is_wall_match and field_counts[Dunsparce] + field_counts[Dudunsparce_ex] == 0 and len(my_state.bench) < 5)
        or (c.id == Chi_Yu and is_crustle_match and field_counts[Chi_Yu] == 0)
        for c in (my_state.hand or [])
    )
    searched_attack_piece_actionable = searched_progress_this_turn and any(
        c.id == Dragapult_ex and can_evolve_drakloak for c in (my_state.hand or [])
    )
    searched_actionable_ids = set()
    if searched_progress_this_turn:
        for _c in (my_state.hand or []):
            if _c.id == Drakloak and can_evolve_dreepy:
                searched_actionable_ids.add(Drakloak)
            elif _c.id == Dragapult_ex and can_evolve_drakloak:
                searched_actionable_ids.add(Dragapult_ex)
            elif _c.id == Dreepy and len(my_state.bench) < 5 and dreepy_line_count < 2:
                searched_actionable_ids.add(Dreepy)
            elif (_c.id == Munkidori and field_counts[Munkidori] == 0 and len(my_state.bench) < 5
                  and (own_total_damage > 0 or field_counts[Dragapult_ex] > 0
                       or is_crustle_match or mirror_damage_emergency
                       or (is_dragapult_match and dreepy_line_count >= 1 and len(my_state.bench) <= 3))):
                searched_actionable_ids.add(Munkidori)
            elif (_c.id == Dunsparce and is_wall_match and len(my_state.bench) < 5
                  and field_counts[Dunsparce] + field_counts[Dudunsparce_ex] == 0):
                searched_actionable_ids.add(Dunsparce)
            elif (_c.id == Dudunsparce_ex and is_wall_match
                  and any(p.id == Dunsparce and not p.appearThisTurn for p in list(my_state.active)+list(my_state.bench) if p is not None)):
                searched_actionable_ids.add(Dudunsparce_ex)
            elif _c.id == Chi_Yu and is_crustle_match and field_counts[Chi_Yu] == 0 and len(my_state.bench) < 5:
                searched_actionable_ids.add(Chi_Yu)
    resolve_searched_action = bool(searched_actionable_ids)
    block_search_then_lillie = resolve_searched_action

    stall_caps = {'dragapult': 2, 'lucario': 3,
                  'alakazam': 2, 'marnie': 2, 'starmie': 2, 'crustle': 1,
                  'rocket': 1, 'unknown': 1}
    stall_cap = stall_caps.get(opponent_profile, 1)
    stall_setup = bool(active_id == Budew and not ready_dragapult and state.turn <= 7
                       and dreepy_line_count >= 1 and budew_stall_uses < stall_cap
                       and not opponent_ready_attacker)

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
    _visible_energy_threat = any(len(p.energyCards) >= 1 for p in visible_opponent)
    def _primary_key(p):
        ids = {e.id for e in p.energyCards}
        required = int(Basic_Fire_Energy in ids) + int(Basic_Psychic_Energy in ids)
        stage = 3 if p.id == Dragapult_ex else 2 if p.id == Drakloak else 1
        active_bonus = 1 if p in my_state.active else 0
        bench_counter_safe = 0 if (is_dragapult_match and p not in my_state.active and p.hp <= 60) else 1
        active_attack_safe = 0 if (is_dragapult_match and p in my_state.active and opponent_dragapult_ready and p.hp <= 200) else 1
        # Do not finish the two-energy pair on a critically damaged Active
        # pre-evolution when a healthier line exists and a public attacker is
        # already being charged.  This avoids attach-then-retreat resource loss.
        generic_survival = 0 if (p in my_state.active and p.id in (Dreepy, Drakloak)
                                 and p.hp <= 70 and _visible_energy_threat and len(main_candidates) >= 2) else 1
        return (bench_counter_safe, active_attack_safe, generic_survival, required, stage, len(p.energies), active_bonus, p.hp)
    primary = max(pool, key=_primary_key) if pool else None
    primary_serial = primary.serial if primary is not None else -1
    primary_energy_ids = {e.id for e in primary.energyCards} if primary is not None else set()
    primary_missing_types = {Basic_Fire_Energy, Basic_Psychic_Energy} - primary_energy_ids
    attack_energy_in_hand = {c.id for c in (my_state.hand or []) if c.id in (Basic_Fire_Energy, Basic_Psychic_Energy)}
    can_complete_primary_from_hand = bool(primary_missing_types & attack_energy_in_hand)
    yveltal_two_dark_needs_third = any(
        p is not None and p.id == Yveltal
        and sum(1 for e in p.energyCards if e.id == Basic_Darkness_Energy) >= 2
        and len(p.energyCards) == 2
        for p in list(my_state.active) + list(my_state.bench)
    )

    # Deterministic same-turn Ultra Ball -> Dragapult ex -> Phantom Dive line.
    _hand_ids_now = [c.id for c in (my_state.hand or [])]
    _retreat_option_now = context == SelectContext.MAIN and any(o.type == OptionType.RETREAT for o in select.option)
    _eligible_drakloaks_now = [p for p in list(my_state.active) + list(my_state.bench)
                               if p is not None and p.id == Drakloak and not p.appearThisTurn
                               and (p in my_state.active or _retreat_option_now)]
    _support_open_now = not state.supporterPlayed
    _manual_open_now = not state.energyAttached
    _crispin_in_hand_now = Crispin in _hand_ids_now and _support_open_now
    _meowth_crispin_access_now = bool(Meowth_ex in _hand_ids_now and field_counts[Meowth_ex] == 0
                                       and len(my_state.bench) < 5 and _support_open_now
                                       and deck_counts[Crispin] > 0 and stadium_id != Team_Rocket_Watchtower)
    _crispin_access_now = _crispin_in_hand_now or _meowth_crispin_access_now
    def _can_finish_energy_now(p):
        ids = {e.id for e in p.energyCards}
        missing = {Basic_Fire_Energy, Basic_Psychic_Energy} - ids
        if not missing:
            return True
        hand_energy = {c.id for c in (my_state.hand or []) if c.id in missing}
        if len(missing) == 1 and _manual_open_now and hand_energy:
            return True
        if _crispin_access_now:
            fire_left = deck_counts[Basic_Fire_Energy] > 0
            psychic_left = deck_counts[Basic_Psychic_Energy] > 0
            if len(missing) == 1:
                return (Basic_Fire_Energy not in missing or fire_left) and (Basic_Psychic_Energy not in missing or psychic_left)
            return _manual_open_now and fire_left and psychic_left
        return False
    same_turn_existing_dragapult = next((p for p in list(my_state.active)
                                             if p is not None and p.id == Dragapult_ex and _can_finish_energy_now(p)), None)
    same_turn_ultra_target = next((p for p in sorted(_eligible_drakloaks_now, key=lambda q:(len(q.energyCards),q.hp), reverse=True)
                                   if _can_finish_energy_now(p)), None)
    same_turn_ultra_phantom = bool(same_turn_ultra_target is not None and Ultra_Ball in _hand_ids_now
                                   and deck_counts[Dragapult_ex] > 0 and my_state.handCount >= 3)
    same_turn_pokepad_phantom = bool(same_turn_ultra_target is not None and Poke_Pad in _hand_ids_now
                                     and deck_counts[Dragapult_ex] > 0
                                     and (opponent_ready_attacker or prize_diff >= 1 or state.turn >= 5))
    ultra_phantom_resolving = bool(select.effect is not None and select.effect.id == Ultra_Ball
                                    and same_turn_ultra_target is not None and deck_counts[Dragapult_ex] > 0)
    pokepad_phantom_resolving = bool(select.effect is not None and select.effect.id == Poke_Pad and same_turn_pokepad_phantom)
    dragapult_in_hand_now = Dragapult_ex in _hand_ids_now
    same_turn_hand_target = next((p for p in sorted(_eligible_drakloaks_now, key=lambda q:(len(q.energyCards),q.hp), reverse=True)
                                  if dragapult_in_hand_now and _can_finish_energy_now(p)), None)
    same_turn_hand_phantom = bool(same_turn_hand_target is not None)
    phantom_combo_target = same_turn_existing_dragapult or same_turn_hand_target or same_turn_ultra_target
    if same_turn_existing_dragapult is not None:
        primary = same_turn_existing_dragapult
        primary_serial = primary.serial
        primary_energy_ids = {e.id for e in primary.energyCards}
        primary_missing_types = {Basic_Fire_Energy, Basic_Psychic_Energy} - primary_energy_ids
    meowth_crispin_phantom = bool(same_turn_existing_dragapult is not None and _meowth_crispin_access_now)
    phantom_combo_active = bool(same_turn_existing_dragapult is not None or same_turn_ultra_phantom or same_turn_pokepad_phantom or same_turn_hand_phantom or ultra_phantom_resolving or pokepad_phantom_resolving)

    previous_disruption_this_turn = any(l.type == LogType.PLAY and l.playerIndex == my_index and l.cardId in (Judge, Unfair_Stamp) for l in current_turn_log)
    natural_stage_action = any((c.id == Drakloak and can_evolve_dreepy) or (c.id == Dragapult_ex and can_evolve_drakloak) for c in (my_state.hand or []))
    block_search_then_lillie = bool(block_search_then_lillie or previous_disruption_this_turn or natural_stage_action or phantom_combo_active)
    progress_ids = {Drakloak, Dragapult_ex, Ultra_Ball, Poke_Pad, Crispin, Lillie_Determination,
                    Basic_Fire_Energy, Basic_Psychic_Energy, Night_Stretcher}
    hand_progress = any(c.id in progress_ids for c in (my_state.hand or []))
    hand_bricked = (not ready_dragapult and dreepy_line_count > 0 and my_state.handCount <= 3 and not hand_progress)
    if wall_break_now:
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
    global turn_intent
    if phase == MachineState.WALL_BREAK:
        turn_intent = TurnIntent.BREAK_WALL
    elif phase == MachineState.ENDGAME or urgent_attack:
        turn_intent = TurnIntent.TAKE_PRIZES
    elif phase == MachineState.RECOVERY:
        turn_intent = TurnIntent.RECOVER_BOARD
    elif phase == MachineState.STALL_SETUP:
        turn_intent = TurnIntent.STALL
    elif phase == MachineState.YVELTAL_LOCK:
        turn_intent = TurnIntent.LOCK
    elif resolve_searched_action or field_counts[Dragapult_ex] == 0:
        turn_intent = TurnIntent.COMPLETE_EVOLUTION
    elif not ready_dragapult:
        turn_intent = TurnIntent.COMPLETE_ATTACKER
    elif op_state.handCount >= 7 and not state.supporterPlayed:
        turn_intent = TurnIntent.DISRUPT
    else:
        turn_intent = TurnIntent.TAKE_PRIZES
    support_count = 0

    for card in my_state.discard:
        discard_counts[card.id] += 1

    # Counterfactual preference learned from replay 88554750: when a damaged
    # two-Prize Active is in projected Alakazam range, do not spend the Supporter
    # on raw draw or a one-Prize gust.  Build and charge a fresh Stage 2 first.
    backup_drakloaks = [p for p in my_state.bench if p is not None and p.id == Drakloak]
    backup_dragapults = [p for p in my_state.bench if p is not None and p.id == Dragapult_ex]
    discard_attack_types = {cid for cid in (Basic_Fire_Energy, Basic_Psychic_Energy)
                            if discard_counts[cid] > 0}
    rosa_ready_targets = []
    for _p in backup_dragapults:
        _energy_ids = {e.id for e in _p.energyCards}
        _missing = {Basic_Fire_Energy, Basic_Psychic_Energy} - _energy_ids
        if _missing and (_missing & discard_attack_types):
            rosa_ready_targets.append(_p)
    rosa_ready_target_serials = {p.serial for p in rosa_ready_targets}
    hand_disruption_survival = bool(
        is_alakazam_match and active_ex_lethal and raw_hand_counts[Judge] > 0
        and not state.supporterPlayed
    )
    rosa_survival_plan = bool(
        is_alakazam_match and active_ex_lethal and prize_diff > 0
        and not hand_disruption_survival
        and raw_hand_counts[Rosas_Encouragement] > 0
        and (backup_drakloaks or backup_dragapults) and len(discard_attack_types) >= 1
    )
    # If the opponent needs only two Prizes and can KO the Active ex next turn,
    # preserving the two-Prize liability is worth more than a routine one-Prize
    # attack.  This is learned from replay 88554750 and expressed as a general
    # prize-map feature rather than an opponent-specific hard-coded sequence.
    prize_denial_bench = [p for p in my_state.bench if p is not None and not card_table[p.id].ex]
    available_attack_prizes = 0
    if active_pokemon is not None and opponent_active is not None:
        for _option in select.option:
            if _option.type != OptionType.ATTACK:
                continue
            _damage = _effective_attack_damage(active_pokemon, _option.attackId, my_state.handCount)
            if _damage >= opponent_active.hp:
                available_attack_prizes = max(available_attack_prizes, prize_count(opponent_active, True))
    immediate_win_available = available_attack_prizes >= len(op_state.prize) > 0
    high_value_attack_available = available_attack_prizes >= min(2, len(op_state.prize))
    emergency_prize_denial = bool(
        is_alakazam_match and active_ex_lethal
        and len(op_state.prize) <= 2 and len(my_state.prize) > 1
        and prize_denial_bench and not immediate_win_available
        and not high_value_attack_available
    )
    # v7 used +/-180k to 260k hard overrides.  v8 converts the same replay
    # regret into a bounded expected-loss term, so a tactical win or valuable
    # attack can still outrank retreat/disruption.
    threat_overkill = max(0, opponent_threat_damage - (active_pokemon.hp if active_pokemon is not None else 0))
    prize_denial_bonus = min(42000, int(18000 + threat_overkill * 80 + opponent_boss_probability * 10000)) if emergency_prize_denial else 0

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
        if (phantom_combo_active and phantom_combo_target is not None
            and pokemon.serial == phantom_combo_target.serial and required_energy
            and attach_id not in energy_ids):
            return 118000 + (500 if active else 0)
        if (active_support_stranded and active and active_pokemon is not None
            and pokemon.serial == active_pokemon.serial
            and energy_count < card_table[pokemon.id].retreatCost):
            # Preserve the two Phantom Dive types while the attack line is still
            # charging. Darkness is the preferred generic retreat resource.
            if attach_id == Basic_Darkness_Energy:
                return 40500
            if is_crustle_match and pokemon.id == Chi_Yu and attach_id == Basic_Fire_Energy:
                return 41000
            if bench_attacker or ready_dragapult:
                if attach_id in (Basic_Fire_Energy, Basic_Psychic_Energy):
                    return 38200 if attach_id not in primary_missing_types else 33500
            return -1
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
                return 108000 if yveltal_two_dark_needs_third else 24600
            return -1
        if pokemon.id == Chi_Yu:
            if not is_crustle_match:
                return -1
            # Replay-regret correction: after Crushing Hammer removed Fire, the
            # old policy treated a P+D Chi-Yu as "full" and refused to reattach
            # Fire, resulting in repeated Allure turns.  Fire is mandatory for
            # Ground Melter regardless of how many off-type energies are present.
            if Basic_Fire_Energy not in energy_ids and attach_id == Basic_Fire_Energy:
                return 62000 + (1200 if active else 0)
            if Basic_Fire_Energy in energy_ids and energy_count < 2 and attach_id not in energy_ids:
                return 51000 + (900 if active else 0)
            if Basic_Fire_Energy in energy_ids and energy_count == 2 and attach_id not in energy_ids:
                return 18000
            return -1
        if pokemon.id == Shaymin:
            return -1
        if pokemon.id == Munkidori:
            if attach_id == Basic_Darkness_Energy and not any(e.id == Basic_Darkness_Energy for e in pokemon.energyCards):
                if is_crustle_match:
                    chi_ready = any(p.id == Chi_Yu and Basic_Fire_Energy in {e.id for e in p.energyCards} and len(p.energies) >= 2 for p in list(my_state.active)+list(my_state.bench) if p is not None)
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
            if wall_break_now and field_counts[Dudunsparce_ex] == 0 and energy_count < 3:
                return (42000 if opponent_crustle_active else 25500) + energy_count * 900
            if active and energy_count == 0 and bench_attacker:
                return 22500
            return -1
        if pokemon.id == Dudunsparce_ex:
            if wall_break_now and energy_count < 3:
                return 50000 + energy_count * 1500
            return -1
        if pokemon.id == Dudunsparce:
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
            target_munk_count = 2 if ((is_lucario_match and own_total_damage >= 50) or (is_dragapult_match and opponent_has_munkidori and own_total_damage >= 40)) else 1
            munkidori_live = (own_total_damage > 0 or field_counts[Dragapult_ex] > 0
                              or is_crustle_match or mirror_damage_emergency
                              or (is_dragapult_match and dreepy_line_count >= 1 and len(my_state.bench) <= 3))
            if (munkidori_live and field_counts[Munkidori] < target_munk_count
                and (hand_counts[Basic_Darkness_Energy] > 0 or deck_counts[Basic_Darkness_Energy] > 0)):
                score = 26000 if is_crustle_match else 15000
            elif not munkidori_live or field_counts[Munkidori] >= target_munk_count or has_dark_munk:
                score = 100
        elif id == Dunsparce:
            if field_counts[Dunsparce] + field_counts[Dudunsparce] + field_counts[Dudunsparce_ex] == 0 and len(my_state.bench) <= (4 if needs_deep_bench else 3):
                score = 23000 if is_wall_match else 10000
            else:
                score = 200
        elif id == Dudunsparce:
            can_evolve_dunsparce = any(p.id == Dunsparce and not p.appearThisTurn for p in list(my_state.active) + list(my_state.bench) if p is not None)
            if is_wall_match and field_counts[Dudunsparce_ex] == 0 and field_counts[Dunsparce] <= 1:
                score = 1000
            else:
                score = 18000 if can_evolve_dunsparce else 500
        elif id == Dudunsparce_ex:
            can_evolve_dunsparce = any(p.id == Dunsparce and not p.appearThisTurn for p in list(my_state.active) + list(my_state.bench) if p is not None)
            score = 50000 if is_wall_match and can_evolve_dunsparce and field_counts[Dudunsparce_ex] == 0 else 200
        elif id == Chi_Yu:
            score = 76000 if is_crustle_match and field_counts[Chi_Yu] == 0 else 300
        elif id == Yveltal:
            if is_crustle_match and field_counts[Chi_Yu] == 0 and field_counts[Yveltal] == 0:
                score = 24000
            else:
                score = 200
        elif id == Shaymin:
            score = 12000 if not is_dragapult_match and not is_crustle_match and field_counts[Shaymin] == 0 and len(my_state.bench) <= 3 else 100
        elif id == Fezandipiti_ex:
            if pre_ko:
                if is_crustle_match and (my_state.handCount >= 3 or len(my_state.bench) >= 3):
                    score = 100
                elif is_dragapult_match and (my_state.handCount >= 4 or len(my_state.bench) >= 4):
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
            elif meowth_crispin_phantom:
                score = 90000
            elif my_state.handCount <= 2 and not hand_progress:
                score = 9000
            else:
                score = 80
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
            if is_wall_match and field_counts[Dunsparce] + field_counts[Dudunsparce_ex] == 0 and deck_counts[Dunsparce] > 0:
                count += 1
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
            if same_turn_ultra_phantom or ultra_phantom_resolving:
                score = 120000
            elif main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
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
            if rosa_survival_plan:
                score = 132000
            else:
                score = 59000 if prize_diff > 0 and primary is not None and can_complete_pair else UNNECESSARY
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if phantom_combo_active and phantom_combo_target is not None:
                    score = 116000
                elif yveltal_two_dark_needs_third and is_crustle_match:
                    score = 115000
                elif deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
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
            if natural_stage_action:
                score = UNNECESSARY
            # Never spend Judge immediately before an available Unfair Stamp or near deck-out.
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
    _active_immune = bool(op_state.active and op_state.active[0] is not None and no_damage_dex(op_state.active[0].id))
    _phantom_bench_prizes = _bench_prize_budget(list(op_state.bench or []), 6)
    wall_attacker_ready = is_wall_match and any(
        (p.id == Dudunsparce_ex and len(p.energyCards) >= 3)
        or (p.id == Munkidori and len(p.energies) >= 2)
        or (p.id == Drakloak and Basic_Fire_Energy in {e.id for e in p.energyCards} and Basic_Psychic_Energy in {e.id for e in p.energyCards})
        for p in my_state.bench if p is not None
    )
    do_switch = (not can_main_attack and (bench_attacker or wall_attacker_ready or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    if phase == MachineState.YVELTAL_LOCK and active_id != Yveltal and yveltal_lock_ready:
        do_switch = True
    if is_crustle_match and active_id == Dragapult_ex:
        if opponent_crustle_active and wall_attacker_ready and _phantom_bench_prizes == 0:
            do_switch = True
        active_crustle_hp = op_state.active[0].hp if op_state.active and op_state.active[0] is not None and op_state.active[0].id == 345 else 999
        for pkm in my_state.bench:
            ready = (pkm.id == Drakloak and len(pkm.energies) >= 2) or (pkm.id == Dudunsparce_ex and len(pkm.energies) >= 3) or (pkm.id == Munkidori and len(pkm.energies) >= 2)
            if ready and (active_crustle_hp <= 130 or (dark_munk_count >= 2 and own_total_damage >= 50)):
                do_switch = True
    effect_card_id = 0 if select.effect == None else select.effect.id
    context_card_id = 0 if select.contextCard == None else select.contextCard.id
    crispin_available = set()
    if effect_card_id == Crispin and select.deck is not None:
        crispin_available = {c.id for c in select.deck}
    
    counter_values = (_counter_allocation_values(obs, select, my_index)
                      if context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE) else {})
    scores = []  # Score for each action
    for _option_index, o in enumerate(select.option):
        score = 0  # The default and baseline score is 0.
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = 100000
            else:
                score = 1
        elif o.type == OptionType.NO:
            score = -100000 if context == SelectContext.IS_FIRST else 0
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
                        if emergency_prize_denial and context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                            if not card_table[card.id].ex:
                                # Prefer expendable one-Prize pivots; preserve an
                                # evolved Dragapult line unless no other pivot exists.
                                _sacrifice_bonus = {
                                    Budew: 24000, Munkidori: 22000, Shaymin: 20000,
                                    Yveltal: 18000, Chi_Yu: 16000, Dreepy: 8000,
                                    Drakloak: -12000,
                                }.get(card.id, 4000)
                                score += prize_denial_bonus // 2 + _sacrifice_bonus // 4
                            else:
                                score -= 12000
                        if card.id == Dreepy:
                            if context == SelectContext.SETUP_ACTIVE_POKEMON:
                                score += 125000 if state.firstPlayer == my_index else 108000
                            else:
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
                                score -= 18000
                        elif card.id == Chi_Yu:
                            if context == SelectContext.SETUP_ACTIVE_POKEMON and not is_crustle_match:
                                score -= 40000
                            else:
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
                            if context == SelectContext.SETUP_ACTIVE_POKEMON:
                                score -= 30000
                            else:
                                score += 95000 if is_crustle_match and energy_count >= 2 else -1500
                        elif card.id == Budew:
                            if context == SelectContext.SETUP_ACTIVE_POKEMON:
                                score += 126000 if state.firstPlayer != my_index else 96000
                            elif context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Dunsparce:
                            score += 500 if context != SelectContext.SETUP_ACTIVE_POKEMON else 1000
                        elif card.id == Dudunsparce_ex:
                            score += 125000 if wall_break_now and energy_count >= 3 else -1500
                        elif card.id == Dudunsparce:
                            score -= 500
                        elif card.id == Fezandipiti_ex:
                            if context == SelectContext.SETUP_ACTIVE_POKEMON and is_archaludon_match:
                                score += 26000
                            else:
                                score -= 1000
                        elif card.id == Shaymin and context == SelectContext.SETUP_ACTIVE_POKEMON:
                            score += 30000 if not is_archaludon_match else -30000
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
                        score = 9000 if is_wall_match and field_counts[Dunsparce] + field_counts[Dudunsparce_ex] == 0 else 2000
                    elif card.id == Fezandipiti_ex and active_id == Shaymin and field_counts[Dreepy] == 0:
                        score = 1200
                    else:
                        score = -1
                elif context == SelectContext.ATTACH_TO and effect_card_id == Crispin:
                    wall_target = next((p for p in list(my_state.active) + list(my_state.bench)
                                        if p is not None and p.id in (Dunsparce, Dudunsparce_ex) and len(p.energyCards) < 3), None) if wall_break_now else None
                    chi_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Chi_Yu), None)
                    yveltal_target = next((p for p in list(my_state.active) + list(my_state.bench) if p is not None and p.id == Yveltal), None)
                    if phase == MachineState.WALL_BREAK and wall_target is not None:
                        score = 120000 if card.serial == wall_target.serial else -1
                    elif phase == MachineState.WALL_BREAK and chi_target is not None:
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
                            score = 125000 if card.id in (Basic_Fire_Energy, Basic_Psychic_Energy) else -1
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
                    if (effect_card_id == Night_Stretcher and is_lucario_match
                        and card.id in (Fezandipiti_ex, Meowth_ex)):
                        score = -250000
                    if (effect_card_id == Drakloak and is_lucario_match and can_evolve_drakloak
                        and deck_counts[Dragapult_ex] > 0):
                        if card.id == Ultra_Ball:
                            score += 260000
                        elif card.id == Poke_Pad:
                            score -= 180000
                    if effect_card_id in (Poke_Pad, Ultra_Ball):
                        if card.id == Drakloak and can_evolve_dreepy: score += 90000
                        elif card.id == Dragapult_ex and can_evolve_drakloak: score += 210000 if (ultra_phantom_resolving or pokepad_phantom_resolving) else 88000
                        elif card.id == Dreepy and dreepy_line_count < 2: score += 76000
                        elif card.id == Munkidori and field_counts[Munkidori] == 0 and (own_total_damage > 0 or is_dragapult_match or is_lucario_match): score += 65000
                        elif (card.id == Dunsparce and is_wall_match and field_counts[Dunsparce] + field_counts[Dudunsparce_ex] == 0):
                            score += 82000
                        elif (card.id == Dudunsparce_ex and is_wall_match and field_counts[Dudunsparce_ex] == 0
                              and any(p.id == Dunsparce and not p.appearThisTurn for p in list(my_state.active)+list(my_state.bench) if p is not None)):
                            score += 92000
                        elif card.id == Chi_Yu and is_crustle_match and field_counts[Chi_Yu] == 0:
                            # New Crustle replays: searching Drakloak first led to
                            # repeated one-prize sacrifices while the wall set up.
                            score += 76000 if state.turn <= 4 else 56000
                        elif (card.id == Yveltal and is_crustle_match
                              and field_counts[Chi_Yu] == 0 and deck_counts[Chi_Yu] == 0
                              and field_counts[Yveltal] == 0):
                            # If Chi-Yu is prized, learn the replay's only viable
                            # non-ex fallback immediately instead of feeding
                            # multiple Drakloak into Superb Scissors.
                            score += 96000 if state.turn <= 4 else 62000
                        elif card.id in (Shaymin, Yveltal, Chi_Yu) and not is_crustle_match:
                            # These are matchup techs, not generic Ultra Ball/Poké Pad targets.
                            score -= 140000
                        elif card.id == Meowth_ex and not is_crustle_match:
                            if not meowth_crispin_phantom:
                                score -= 120000
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
                    same_id_indices = [i for i, h in enumerate(my_state.hand or []) if h.id == card.id]
                    protected_copy = bool(same_id_indices and o.index == same_id_indices[-1])
                    if (effect_card_id == Ultra_Ball and not ready_dragapult and card.id in
                        (Basic_Fire_Energy, Basic_Psychic_Energy)
                        and protected_copy and (raw_hand_counts[card.id] <= 1 or deck_counts[card.id] <= 1)):
                        score = -230000
                    if (effect_card_id == Ultra_Ball and card.id == Boss_Orders and protected_copy
                        and len(my_state.prize) <= 3
                        and any(prize_count(p, True) >= 2 for p in op_state.bench if p is not None)):
                        score = -250000
                    if ultra_phantom_resolving and same_turn_ultra_target is not None:
                        _ids = {e.id for e in same_turn_ultra_target.energyCards}
                        _missing = {Basic_Fire_Energy, Basic_Psychic_Energy} - _ids
                        if card.id in _missing or (card.id == Crispin and len(_missing) >= 1):
                            score = -250000
                        elif card.id in (Enhanced_Hammer, Tool_Scrapper, Judge, Team_Rocket_Watchtower, Risky_Ruins, Shaymin, Yveltal):
                            score = 220000
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
                            # Munkidori transfers the selected amount at once.
                            # When Phantom Dive is already legal against a damage-
                            # immune Active, prepare the best Bench target for the
                            # following six counters instead of feeding the wall.
                            if (_active_immune and active_id == Dragapult_ex
                                and active_pokemon is not None and _phantom_ready(active_pokemon)
                                and o.area == AreaType.BENCH and not no_damage_counter(card)):
                                _transfer = 30 if (select.effect is not None and select.effect.id == Munkidori) else max(10, int(select.remainDamageCounter or 1) * 10)
                                if 0 < hp - _transfer <= 60:
                                    score += prize_count(card,False) * 280000
                            if is_crustle_match and o.area == AreaType.ACTIVE and card.id == 345:
                                score += 8000
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
                            if hp <= remain_damage and card.id in target_priority_ids:
                                score += 220000 + prize_count(card, False) * 50000 + target_priority_ids.get(card.id, 0) * 2
                            elif index in plan_b.counter:
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
                    if _option_index in counter_values and not no_damage_counter(card):
                        _total_pr, _immediate_pr, _focus = counter_values[_option_index]
                        # Lexicographic objective: total Prizes first, then immediate
                        # conversion, then concentration.  This replaces brittle
                        # target-name bonuses without banning any legal placement.
                        score += _total_pr * 1000000 + _immediate_pr * 180000 + _focus
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
                    if effect_card_id == Rosas_Encouragement and isinstance(card, Pokemon):
                        if card.serial in rosa_ready_target_serials:
                            score = 72000
                        elif (card in my_state.active and active_ex_lethal):
                            score = 500
                        else:
                            score = 5000 if card.serial == primary_serial else 1000
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
                if card.id in searched_actionable_ids:
                    score = 119000
                elif main_pokemon_count < 3:
                    score = 51000
                else:
                    score = -1
            elif card.id == Munkidori:
                target_munk_count = 2 if ((is_lucario_match and own_total_damage >= 50) or (is_dragapult_match and opponent_has_munkidori and own_total_damage >= 40)) else 1
                munkidori_live = (own_total_damage > 0 or field_counts[Dragapult_ex] > 0
                                  or is_crustle_match or mirror_damage_emergency
                              or (is_dragapult_match and dreepy_line_count >= 1 and len(my_state.bench) <= 3))
                if (munkidori_live and card.id in searched_actionable_ids
                    and field_counts[Munkidori] < target_munk_count and len(my_state.bench) < 5):
                    score = 118000
                elif munkidori_live and field_counts[Munkidori] < target_munk_count and len(my_state.bench) < 5:
                    score = 54500 if is_crustle_match else 50500
                else:
                    score = -1
            elif card.id == Chi_Yu:
                if card.id in searched_actionable_ids and is_crustle_match and field_counts[Chi_Yu] == 0 and len(my_state.bench) < 5:
                    score = 118500
                elif is_crustle_match and field_counts[Chi_Yu] == 0 and len(my_state.bench) < 5:
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
                if not is_dragapult_match and not is_crustle_match and field_counts[Shaymin] == 0 and len(my_state.bench) <= 3:
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
                if is_lucario_match and not pre_ko:
                    score = -1
                elif is_crustle_match and (my_state.handCount >= 3 or len(my_state.bench) >= 3):
                    score = -1
                elif is_dragapult_match and (my_state.handCount >= 4 or len(my_state.bench) >= 4):
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
                if meowth_crispin_phantom:
                    score = 126000
                elif is_dragapult_match and (my_state.handCount >= 4 or len(my_state.bench) >= 3):
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
                support_only_recovery = bool(is_lucario_match and discard_counts[Fezandipiti_ex] + discard_counts[Meowth_ex] > 0
                                             and not any(discard_counts[cid] > 0 for cid in (Dreepy, Drakloak, Dragapult_ex,
                                                                                           Basic_Fire_Energy, Basic_Psychic_Energy)))
                if resolve_searched_action or support_only_recovery:
                    score = -1
                elif card_score >= 18000:
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
                    score = 35000 - (12000 if (rosa_survival_plan or hand_disruption_survival) else 0)
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support and my_state.deckCount > 6 and not block_search_then_lillie and not phantom_combo_active:
                    score = 14000 - (9000 if (rosa_survival_plan or hand_disruption_survival) else 0)
                else:
                    score = -1
            elif card.id == Judge:
                if hand_disruption_survival and card.id == use_support:
                    # Alakazam damage scales with hand size.  A disruption turn
                    # is preferred over feeding a damaged two-Prize Active.
                    score = 65000
                elif card.id == use_support and my_state.deckCount > 6 and not natural_stage_action and not phantom_combo_active and not resolve_searched_action and not (pre_ko and hand_counts[Unfair_Stamp] > 0):
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
                if resolve_searched_action:
                    score = -1
                elif deck_counts[Dreepy] + deck_counts[Budew] + deck_counts[Munkidori] + deck_counts[Yveltal] > 0:
                    score = 46000
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                missing_stage = (can_evolve_dreepy and deck_counts[Drakloak] > 0) or (can_evolve_drakloak and deck_counts[Dragapult_ex] > 0)
                wall_target = is_crustle_match and field_counts[Chi_Yu] == 0 and deck_counts[Chi_Yu] > 0
                if same_turn_ultra_phantom:
                    score = 125000
                elif resolve_searched_action:
                    score = -1
                elif (is_lucario_match and can_evolve_drakloak and deck_counts[Dragapult_ex] > 0
                      and my_state.handCount >= 3):
                    score = 56500
                elif missing_stage and (negative_hand_count >= 1 or disposable_hand_count >= 1) and my_state.handCount >= 3:
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
                if same_turn_pokepad_phantom:
                    score = 123000
                elif resolve_searched_action:
                    score = -1
                else:
                    useful = ((deck_counts[Drakloak] > 0 and can_evolve_dreepy)
                              or (deck_counts[Dreepy] > 0 and dreepy_line_count < 2 and len(my_state.bench) < 5)
                              or (deck_counts[Munkidori] > 0 and field_counts[Munkidori] == 0
                                  and (own_total_damage > 0 or is_dragapult_match or is_lucario_match))
                              or (is_crustle_match and deck_counts[Chi_Yu] > 0 and field_counts[Chi_Yu] == 0))
                    score = 45000 if useful else -1
            elif card.id == Rosas_Encouragement:
                if card.id == use_support:
                    if rosa_survival_plan and rosa_ready_targets:
                        score = 68000
                    elif rosa_survival_plan:
                        score = 26000
                    else:
                        score = 43000
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    if card.id == Crispin and phantom_combo_active:
                        score = 132000
                    elif card.id == Crispin and yveltal_two_dark_needs_third and is_crustle_match:
                        score = 122000
                    else:
                        score = 50000 if phase == MachineState.CHARGE and card.id == Crispin else 35000
                else:
                    score = -1
            if card_table[card.id].cardType == CardType.STADIUM and card.id == stadium_id:
                score = -1
            if phase_blocked:
                score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
            chi_missing_fire = any(
                p.id == Chi_Yu and Basic_Fire_Energy not in {e.id for e in p.energyCards}
                for p in list(my_state.active) + list(my_state.bench) if p is not None
            )
            if is_crustle_match and card.id == Basic_Fire_Energy and chi_missing_fire and pokemon.id != Chi_Yu:
                score -= 16000
            if emergency_prize_denial and pokemon in my_state.active:
                _retreat_cost = card_table[pokemon.id].retreatCost
                _retreat_deficit = max(0, _retreat_cost - len(pokemon.energyCards))
                if _retreat_deficit == 1:
                    score = max(score, 42000 + prize_denial_bonus // 3)
                elif _retreat_deficit == 0:
                    score -= 14000
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if rosa_survival_plan and pokemon.id == Drakloak:
                score += 26000
            if pokemon.id == Dreepy:
                if Drakloak in searched_actionable_ids:
                    score = 121000
                else:
                    score += 108000
            elif pokemon.id == Dunsparce:
                evo_id = o.cardId or 0
                if is_wall_match and field_counts[Dudunsparce_ex] == 0:
                    score += 120000 if evo_id == Dudunsparce_ex else (18000 if evo_id == Dudunsparce else 70000)
                else:
                    score += 62000 if my_state.handCount <= 6 else 30000
            elif phantom_combo_target is not None and pokemon.id == Drakloak and pokemon.serial == phantom_combo_target.serial:
                score = 124000
            elif pokemon.id == Drakloak and Dragapult_ex in searched_actionable_ids:
                score = 121500
            elif is_crustle_match and pokemon.id == Drakloak and field_counts[Dragapult_ex] >= 1:
                score -= 12000
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
            elif rosa_survival_plan and card.id == Drakloak:
                score = 61000
            elif card.id == 1267:  # Lumiose City
                score = 1
            elif card.id == Dudunsparce:
                draw_threshold = 7 if needs_deep_bench else 5
                preserve_only_wall_line = wall_break_now and field_counts[Dudunsparce_ex] == 0 and field_counts[Dunsparce] == 0
                score = -1 if preserve_only_wall_line else (65000 if my_state.handCount <= draw_threshold else -1)
            elif card.id == -999999:
                score = -1
            elif card.id == Fezandipiti_ex:
                score = 67000 if pre_ko else -1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            fresh_ready_backup = any(
                p.id == Dragapult_ex and _phantom_ready(p) and p.hp >= 260
                for p in my_state.bench if p is not None
            )
            if emergency_prize_denial:
                score = max(score, 36000 + prize_denial_bonus)
            elif active_ex_lethal and fresh_ready_backup:
                score = 155000
            elif phantom_combo_active and bench_attacker and active_id != Dragapult_ex:
                score = 112000
            elif do_switch:
                score = 10000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId
            if emergency_prize_denial:
                score -= max(10000, prize_denial_bonus - available_attack_prizes * 8000)
            elif active_ex_lethal and rosa_survival_plan and not immediate_win_available:
                score -= 18000
            if active_id == Yveltal and o.attackId == 997 and phase == MachineState.YVELTAL_LOCK:
                score = 6000
            if o.attackId == 154 and urgent_attack:
                score = 76000
            if o.attackId == 153 and active_pokemon is not None and _phantom_ready(active_pokemon):
                score = -1
            if active_id == Dudunsparce_ex:
                if o.attackId == 426:
                    active_prizes = prize_count(opponent_active, True) if opponent_active is not None else 0
                    if opponent_active is not None and opponent_active.hp <= 150:
                        score = max(score, 150000 + active_prizes * 80000)
                    elif _active_immune:
                        score = max(score, 82000)
                elif o.attackId == 425:
                    tail_damage = 60 * opponent_ex_count
                    if opponent_active is not None and tail_damage >= opponent_active.hp:
                        score = max(score, 110000 + prize_count(opponent_active, True) * 70000)
            if _active_immune and active_id == Dragapult_ex:
                if o.attackId == 154 and any(p is not None and p.hp > 0 and not no_damage_counter(p) for p in (op_state.bench or [])):
                    # Bench counters remain useful even when direct Active damage
                    # is blocked.  A small floor only beats END; setup, retreat to
                    # a ready wall-breaker, and immediate winning actions keep
                    # their normal higher scores.
                    score = max(score, 500 + _phantom_bench_prizes * 150000)
                else:
                    score -= 18000
            elif is_crustle_match and active_id == Chi_Yu:
                score += 82000 if o.attackId == 20 and stadium_id != 0 else 30000
            elif is_crustle_match and active_id == Yveltal:
                score += 70000 if o.attackId == 998 else 36000
            elif is_crustle_match and active_id == Munkidori:
                score += 70000 if o.attackId == 141 else 30000
            elif is_crustle_match and active_id == Drakloak:
                score += 60000
            elif is_crustle_match and active_id == Dragapult_ex and o.attackId == 154:
                # Bench pressure remains a legal fallback; it is only mildly
                # discounted when a dedicated wall breaker is ready.
                score += 12000

        scores.append(score)

    # The simulator-backed search is intentionally shallow: it resolves the
    # current tactical sequence, compares attack-now against setup, and stops
    # before guessing the opponent's full turn.
    global _lookahead_turn_key, _lookahead_calls_this_turn
    if context == SelectContext.MAIN and not _lookahead_active:
        key = (state.turn, my_index)
        if key != _lookahead_turn_key:
            _lookahead_turn_key = key
            _lookahead_calls_this_turn = 0
        tactical_ambiguity = bool(
            phantom_combo_active or same_turn_ultra_phantom or same_turn_pokepad_phantom
            or resolve_searched_action or urgent_attack
            or (can_main_attack and any(o.type in (OptionType.PLAY, OptionType.ATTACH,
                                                   OptionType.EVOLVE, OptionType.ABILITY,
                                                   OptionType.RETREAT) and scores[i] >= 0
                                    for i, o in enumerate(select.option)))
            or (can_evolve_drakloak and deck_counts[Dragapult_ex] > 0)
        )
        if (tactical_ambiguity
            and _lookahead_calls_this_turn < LOOKAHEAD_MAX_CALLS_PER_TURN):
            _lookahead_calls_this_turn += 1
            scores = _rerank_with_lookahead(obs, scores)

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
