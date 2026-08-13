from __future__ import annotations

import os
import math
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    KYOGRE = 721
    SNOVER = 722
    MEGA_ABOMASNOW_EX = 723

    ABRA = 741
    KADABRA = 742
    ALAKAZAM = 743

    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678
    DWEBBLE = 344
    CRUSTLE = 345

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    MEGA_SIGNAL = 1145
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    JUDGE = 1213
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10

# Abra/Kadabra-kill priority: deny the Psychic/Alakazam line (Lucario is x2 weak to Psychic).
# Moderate value tuned by sweep — high values over-commit and skip better KOs. Gated to Psychic decks.
_ABRA_BONUS = 400
_KADABRA_BONUS = 400


# Kaggle executes the submitted source with exec(), so main.py may not receive a
# usable __file__ and the process working directory may be outside the agent folder.
# cg.api, however, is imported from the submission's cg directory; use its module
# path to locate the sibling deck.csv. This changes packaging/path resolution only.
import cg.api as _cg_api

_THIS_FILE = globals().get("__file__")
if _THIS_FILE:
    BASE_DIR = os.path.dirname(os.path.abspath(_THIS_FILE))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_cg_api.__file__)))
DECK_PATH = os.path.join(BASE_DIR, "deck.csv")
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    player = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return player.hand[index]
        case AreaType.DISCARD:
            return player.discard[index]
        case AreaType.ACTIVE:
            return player.active[index]
        case AreaType.BENCH:
            return player.bench[index]
        case AreaType.PRIZE:
            return player.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in {144, 322, 323, 337}:  # low-value support Pokemon
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950   # KO Snover before it evolves into Mega Abomasnow (Fighting wall); ported from 1084
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.ABRA:
        score += _ABRA_BONUS    # deny the Alakazam (Psychic) line before it OHKOs our Lucario
    elif pokemon.id == C.KADABRA:
        score += _KADABRA_BONUS
    if pokemon.id == C.RIOLU:
        score += 800   # deny opponent's Lucario line by KOing Riolu (mirror edge, ported from 1084)
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


class LucarioPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked[: self.select.maxCount]

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has_crustle_axis(self) -> bool:
        return any(
            pokemon is not None and pokemon.id in {C.DWEBBLE, C.CRUSTLE}
            for pokemon in self._opponent_board()
        )

    def _opponent_is_water_deck(self) -> bool:
        # Abomasnow / water archetype: walls Fighting, so bulk up the Lucario line. Ported from 1084.
        return any(
            pokemon is not None and pokemon.id in {C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX}
            for pokemon in self._opponent_board()
        )

    def _opponent_is_psychic_engine(self) -> bool:
        # Alakazam hand-size combo deck (Psychic Draw engine). Alakazam's attack places
        # 2 damage counters per card in their hand, so a big hand = big damage.
        return any(
            pokemon is not None and pokemon.id in {C.ABRA, C.KADABRA, C.ALAKAZAM}
            for pokemon in self._opponent_board()
        )

    def _should_preserve_hariyama(self) -> bool:
        return (
            self._opponent_has_crustle_axis()
            and self.hand_counts[C.HARIYAMA] >= 1
            and any(pokemon is not None and pokemon.id == C.MAKUHITA for pokemon in self._my_board())
        )

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self.my_prizes_left in {2, 3}:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required = 3
            base_damage = 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required = 1
            base_damage = 70

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    if my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == C.CRUSTLE:
                        damage = 0
                    else:
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == EnergyType.FIGHTING:
                            damage *= 2
                        elif op_data.resistance == EnergyType.FIGHTING:
                            damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.me.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            score += 1 if pokemon.id == C.HARIYAMA else 0
            score += 100 if energy_count < 3 else 0
            score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            score += 1 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            score += 8 if self.my_prizes_left in {2, 3} else 20
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.MAKUHITA:
            score += -10 if self.field_counts[card.id] >= 1 else 10
        elif card.id == C.HARIYAMA:
            score += 20 if self.field_counts[C.MAKUHITA] >= 1 else -20
        elif card.id == C.LUNATONE:
            score += -250 if self.field_counts[card.id] >= 1 else 60
        elif card.id == C.SOLROCK:
            score += -250 if self.field_counts[card.id] >= 1 else 50
        elif card.id == C.RIOLU:
            lucario_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += -150 if lucario_line >= 2 else -3 if lucario_line >= 1 else 40
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = 20000
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
            return -1
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050 if can_bridge_draw else -1
            return 5000
        if card.id == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            if self._should_preserve_hariyama():
                return -1
            return -1 if self._low_deck() else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self._low_deck() else 3100
        if card.id == C.JUDGE:
            # Anti-control tech: reset the opponent's hand to 4 to gut Alakazam's hand-size
            # damage. Only worth a supporter when they run the Psychic engine and are hoarding
            # cards (and have more than us, so the reset is net-favorable). Otherwise dead.
            if (
                self._opponent_is_psychic_engine()
                and self.opponent.handCount >= 6
                and self.opponent.handCount >= self.me.handCount + 1
            ):
                return 3300
            return -1
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        return 10000

    def _score_gravity_mountain(self) -> float:
        opponent_has_stage2 = any(
            pokemon is not None and card_table[pokemon.id].stage2 for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500
        return 1200 if self.stadium_id else -1

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                if pokemon.id == C.RIOLU:
                    return 12200
                if pokemon.id == C.MEGA_LUCARIO_EX:
                    return 12800
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        evolved = get_card(self.obs, option.area, option.index, self.my_index)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not (
            evolved is not None and evolved.id == C.HARIYAMA and board_index == plan.attacker
        ):
            return -1
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self._low_deck():
            return -1
        return 30000

    def _remember_lunatone_ability(self, ranked: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True


def agent(obs_dict: dict) -> list[int]:
    global pre_turn
    global ability_used
    global plan

    if obs_dict.get("select") is None and "current" not in obs_dict:
        pre_turn = -1
        ability_used = False
        plan = AttackPlan()
        return my_deck

    obs = to_observation_class(obs_dict)
    if obs.select is None:
        pre_turn = -1
        ability_used = False
        plan = AttackPlan()
        return my_deck

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    return LucarioPolicy(obs).choose()


_base_tr=LucarioPolicy._score_play_trainer
_base_choose=LucarioPolicy.choose
_ppp_turn=-1
_ppp_used=0

def _track_choose(self):
 global _ppp_turn,_ppp_used
 if _ppp_turn!=self.state.turn:_ppp_turn=self.state.turn;_ppp_used=0
 r=_base_choose(self)
 if self.context==SelectContext.MAIN and r:
  o=self.select.option[r[0]]
  if o.type==OptionType.PLAY:
   c=get_card(self.obs,AreaType.HAND,o.index,self.my_index)
   if c is not None and c.id==C.PREMIUM_POWER_PRO:_ppp_used+=1
 return r

def _need_search(self,cid):
 if cid==C.MEGA_SIGNAL:return self.field_counts[C.RIOLU]>0 and self.hand_counts[C.MEGA_LUCARIO_EX]==0
 if cid==C.POKE_PAD:return self.field_counts[C.RIOLU]+self.field_counts[C.MEGA_LUCARIO_EX]<2 or self.field_counts[C.SOLROCK]==0 or self.field_counts[C.LUNATONE]==0 or (self._opponent_has_crustle_axis() and self.field_counts[C.MAKUHITA]+self.field_counts[C.HARIYAMA]==0)
 return True

def _tr(self,card):
 if card.id in {C.MEGA_SIGNAL,C.POKE_PAD}:return 10000 if _need_search(self,card.id) else -1
 return _base_tr(self,card)
LucarioPolicy._score_play_trainer=_tr
LucarioPolicy.choose=_track_choose


# tournament-learned archetype router: keep original v63 search behavior where it was stronger
_router_base_tr=_base_tr
_router_min_tr=LucarioPolicy._score_play_trainer

def _router_arch(self):
 ids={p.id for p in self._opponent_board() if p is not None}
 if ids & {344,345,756}:return 'crustle'
 if ids & {677,678,673,674,675,676}:return 'lucario'
 if ids & {119,120,121}:return 'dragapult'
 if ids & {741,742,743}:return 'alakazam'
 if ids & {463,473,474,414,891}:return 'rocket'
 if ids & {169,190,666}:return 'archaludon'
 if ids & {646,647,648}:return 'grimmsnarl'
 if ids & {506,507}:return 'articuno'
 return 'unknown'

def _router_tr(self,card):
 arch=_router_arch(self)
 # Replay/tournament learning showed that strict exact-search gating hurts the
 # one-prize Crustle line and Lucario mirrors, so retain the original v63 policy there.
 if arch in {'crustle','lucario','alakazam','unknown'}:
  return _router_base_tr(self,card)
 return _router_min_tr(self,card)
LucarioPolicy._score_play_trainer=_router_tr

# Keep the Kaggle raw-exec entry point as the final callable in this file.
_learned_internal_agent = agent
del agent
def agent(obs_dict: dict) -> list[int]:
    return _learned_internal_agent(obs_dict)


# --- Carmine large-hand hard stop ---
_carmine_hard_base_trainer = LucarioPolicy._score_play_trainer

def _carmine_hard_trainer(self, card):
    if card.id == C.CARMINE:
        if self._low_deck() or self.me.handCount >= 6:
            return -1
        if self.can_attack and plan.remain_hp <= 0 and self.me.handCount >= 4:
            return -1
    return _carmine_hard_base_trainer(self, card)

LucarioPolicy._score_play_trainer = _carmine_hard_trainer
_carmine_hard_internal_agent = agent
del agent
def agent(obs_dict: dict) -> list[int]:
    return _carmine_hard_internal_agent(obs_dict)
# Minimal disruption-aware patch, preserving the original Lucario decisions.
C.RIOLU_70=333; C.DUNSPARCE=305; C.DUDUNSPARCE=66; C.DUDUNSPARCE_EX=306
C.HILDA=1225; C.WALLY=1229; C.XEROSIC=1197; C.ROCK_FIGHTING_ENERGY=20; C.AIR_BALLOON=1174
ATTACK_DUDUN_TAIL=425; ATTACK_DUDUN_DRILL=426; ATTACK_DUDUN_LAND=76

def _d_opp(self): return [p for p in self.opponent.active+self.opponent.bench if p is not None]
def _d_board(self): return [p for p in self.me.active+self.me.bench if p is not None]
def _d_arch(self):
 ids={p.id for p in _d_opp(self)}|{c.id for c in self.opponent.discard}
 if ids&{741,742,743}:return 'alakazam'
 if ids&{119,120,121}:return 'dragapult'
 if ids&{344,345,756}:return 'crustle'
 if ids&{333,677,678,675,676}:return 'lucario'
 return 'other'
def _d_ex(self):return sum(1 for p in _d_opp(self) if card_table[p.id].ex or card_table[p.id].megaEx)

_d_prev_tr=LucarioPolicy._score_play_trainer
def _d_tr(self,c):
 arch=_d_arch(self)
 if c.id==C.XEROSIC:
  ready=any(p.id==C.MEGA_LUCARIO_EX and len(p.energies)>=1 for p in _d_board(self))
  return 3300 if ready and self.opponent.handCount>=(5 if arch=='alakazam' else 7) else -1
 if c.id==C.JUDGE:
  return 3275 if self.opponent.handCount>=(5 if arch=='alakazam' else 7) and self.me.handCount<=4 else -1
 if c.id==C.HILDA:
  need_evo=(any(p.id in {C.RIOLU,C.RIOLU_70,C.DUNSPARCE} for p in _d_board(self)) and (self.hand_counts[C.MEGA_LUCARIO_EX]+self.hand_counts[C.DUDUNSPARCE]+self.hand_counts[C.DUDUNSPARCE_EX])==0)
  need_energy=not self.state.energyAttached and self.hand_counts[C.BASIC_FIGHTING_ENERGY]+self.hand_counts[C.ROCK_FIGHTING_ENERGY]==0
  return 3225 if need_evo or need_energy else -1
 if c.id==C.WALLY:
  return 3260 if any(p.id==C.MEGA_LUCARIO_EX and p.maxHp-p.hp>=150 for p in _d_board(self)) else -1
 return _d_prev_tr(self,c)

_d_prev_hand=LucarioPolicy._score_to_hand
def _d_hand(self,c):
 s=_d_prev_hand(self,c)
 if c.id in {C.BASIC_FIGHTING_ENERGY,C.ROCK_FIGHTING_ENERGY} and not self.state.energyAttached and self.hand_counts[C.BASIC_FIGHTING_ENERGY]+self.hand_counts[C.ROCK_FIGHTING_ENERGY]==0:s+=700
 if c.id==C.DUDUNSPARCE_EX and self.field_counts[C.DUNSPARCE]>0 and (_d_arch(self)=='crustle' or _d_ex(self)>=2):s+=400
 if c.id==C.DUDUNSPARCE and self.field_counts[C.DUNSPARCE]>0:s+=250
 return s

_d_prev_energy=LucarioPolicy._energy_target_score
def _d_energy(self,p,active):
 s=_d_prev_energy(self,p,active)
 if _d_arch(self)=='crustle' and p.id in {C.DUNSPARCE,C.DUDUNSPARCE,C.DUDUNSPARCE_EX}:s+=500 if len(p.energies)<3 else -300
 return s

_d_prev_plan=LucarioPolicy._plan_attack
def _d_plan(self):
 global plan
 _d_prev_plan(self)
 if _d_arch(self)!='crustle':return
 opp=self._opponent_board();board=self._my_board()
 if not opp:return
 for i,p in enumerate(board):
  if p is None or (i>0 and not self.can_switch):continue
  if p.id==C.DUDUNSPARCE_EX and len(p.energies)>=3:plan=AttackPlan(i,0,1,opp[0].hp-150,False);return
  if p.id==C.DUDUNSPARCE and len(p.energies)>=3:plan=AttackPlan(i,0,0,opp[0].hp-90,False);return

_d_prev_opt=LucarioPolicy._score_option
def _d_opt(self,o):
 if o.type==OptionType.ATTACK and self.me.active:
  a=self.me.active[0]
  if a.id==C.DUDUNSPARCE_EX:
   aid=ATTACK_DUDUN_TAIL if plan.attack_index==0 else ATTACK_DUDUN_DRILL
   return 1200 if o.attackId==aid else 900
  if a.id==C.DUDUNSPARCE:return 1200 if o.attackId==ATTACK_DUDUN_LAND else 900
 return _d_prev_opt(self,o)

LucarioPolicy._score_play_trainer=_d_tr
LucarioPolicy._score_to_hand=_d_hand
LucarioPolicy._energy_target_score=_d_energy
LucarioPolicy._plan_attack=_d_plan
LucarioPolicy._score_option=_d_opt
# Learned Dragapult-only refinement; other matchups retain the d1 champion policy.
_d3_prev_attach=LucarioPolicy._score_attach
def _d3_attach(self,o):
 s=_d3_prev_attach(self,o)
 c=get_card(self.obs,AreaType.HAND,o.index,self.my_index);p=get_card(self.obs,o.inPlayArea,o.inPlayIndex,self.my_index)
 if c is not None and p is not None and c.id==C.ROCK_FIGHTING_ENERGY and _d_arch(self)=='dragapult':
  if p.id in {C.MEGA_LUCARIO_EX,C.RIOLU,C.RIOLU_70}:s+=600
  else:s-=300
 return s

_d3_prev_plan=LucarioPolicy._plan_attack
def _d3_plan(self):
 global plan
 _d3_prev_plan(self)
 if _d_arch(self)!='dragapult':return
 exn=_d_ex(self)
 if exn<3:return
 opp=self._opponent_board();board=self._my_board()
 if not opp:return
 dmg=60*exn
 for i,p in enumerate(board):
  if p is not None and p.id==C.DUDUNSPARCE_EX and len(p.energies)>=1 and (i==0 or self.can_switch):
   if dmg>=opp[0].hp:plan=AttackPlan(i,0,0,opp[0].hp-dmg,False);return

LucarioPolicy._score_attach=_d3_attach
LucarioPolicy._plan_attack=_d3_plan

C.BLACK_BELT_TRAINING=1211;C.TOOL_SCRAPPER=1137;C.CRUSHING_HAMMER=1120;C.NIGHT_STRETCHER=1097
_m3_prev_choose=LucarioPolicy.choose;_m3_prev_tr=LucarioPolicy._score_play_trainer;_m3_prev_choice=LucarioPolicy._score_card_choice
_m3_turn=-1;_m3_ppp=0;_m3_bb=False

def _m3_tgt(self):
 b=self._opponent_board();return b[plan.target] if plan.target>=0 and plan.target<len(b) else None
def _m3_tool(p,cid):return p is not None and any(getattr(t,'id',-1)==cid for t in p.tools)
def _m3_choose(self):
 global _m3_turn,_m3_ppp,_m3_bb
 if _m3_turn!=self.state.turn:_m3_turn=self.state.turn;_m3_ppp=0;_m3_bb=False
 r=_m3_prev_choose(self)
 if self.context==SelectContext.MAIN and r:
  o=self.select.option[r[0]]
  if o.type==OptionType.PLAY:
   c=get_card(self.obs,AreaType.HAND,o.index,self.my_index)
   if c is not None:
    if c.id==C.PREMIUM_POWER_PRO:_m3_ppp+=1
    elif c.id==C.BLACK_BELT_TRAINING:_m3_bb=True
 return r

def _m3_tr(self,c):
 arch=_d_arch(self);t=_m3_tgt(self)
 if c.id==C.BLACK_BELT_TRAINING:
  if arch!='lucario' or not self.can_attack or plan.target!=0 or t is None:return -1
  td=card_table[t.id]
  if not(td.ex or td.megaEx) or plan.remain_hp<=0:return -1
  rem=plan.remain_hp-30*_m3_ppp;ppp=self.hand_counts[C.PREMIUM_POWER_PRO]
  before=(max(0,rem)+29)//30;after=(max(0,rem-40)+29)//30
  return 5350 if after<=ppp and after<before else -1
 if c.id==C.PREMIUM_POWER_PRO and arch=='lucario' and self.can_attack and t is not None:
  b=40 if _m3_bb and (card_table[t.id].ex or card_table[t.id].megaEx) else 0
  return 5000 if plan.remain_hp-b-30*_m3_ppp>0 else -1
 if c.id==C.TOOL_SCRAPPER:
  return 5550 if arch=='lucario' and any(_m3_tool(p,C.HERO_CAPE) for p in self._opponent_board()) else -1
 if c.id==C.CRUSHING_HAMMER:
  return 4100 if arch=='lucario' and any(p is not None and len(p.energies)>0 for p in self._opponent_board()) else -1
 if c.id==C.NIGHT_STRETCHER:
  need=any(p is not None and p.id in {C.RIOLU,C.RIOLU_70} for p in self._my_board()) and self.hand_counts[C.MEGA_LUCARIO_EX]==0 and any(x.id==C.MEGA_LUCARIO_EX for x in self.me.discard)
  return 5200 if need else -1
 return _m3_prev_tr(self,c)

def _m3_choice(self,o):
 ctx=getattr(self.select,'contextCard',None);cid=getattr(ctx,'id',-1) if ctx is not None else -1
 pi=getattr(o,'playerIndex',self.my_index)
 try:c=get_card(self.obs,o.area,o.index,pi)
 except Exception:c=None
 if cid==C.TOOL_SCRAPPER and c is not None:return 10000 if c.id==C.HERO_CAPE and pi==self.opponent_index else 100
 if cid==C.NIGHT_STRETCHER and c is not None:return 10000 if c.id==C.MEGA_LUCARIO_EX else 100
 return _m3_prev_choice(self,o)
LucarioPolicy.choose=_m3_choose;LucarioPolicy._score_play_trainer=_m3_tr;LucarioPolicy._score_card_choice=_m3_choice
_m3_internal=agent
del agent
def agent(obs_dict):return _m3_internal(obs_dict)

# --- mirror learned policy candidate ---
_m4_old_target=target_score
_M4_EXTRA=800
def target_score(pokemon):
 s=_m4_old_target(pokemon)
 if pokemon.id in {C.RIOLU,C.RIOLU_70}:s+=_M4_EXTRA
 return s
_m4_prev_evolve=LucarioPolicy._score_evolve
_m4_prev_hand=LucarioPolicy._score_to_hand
_m4_prev_tr=LucarioPolicy._score_play_trainer
_M4_NOEX=True
_M4_DISRUPT=False
def _m4_evolve(self,o):
 e=get_card(self.obs,o.area,o.index,self.my_index)
 if _M4_NOEX and _d_arch(self)=='lucario' and e is not None and e.id==C.DUDUNSPARCE_EX:return -1
 return _m4_prev_evolve(self,o)
def _m4_hand(self,c):
 s=_m4_prev_hand(self,c)
 if _M4_NOEX and _d_arch(self)=='lucario' and c.id==C.DUDUNSPARCE_EX:s-=800
 return s
def _m4_tr(self,c):
 if _M4_DISRUPT and _d_arch(self)=='lucario':
  if c.id==C.XEROSIC:
   ready=any(p.id==C.MEGA_LUCARIO_EX and len(p.energies)>=1 for p in _d_board(self))
   return 3300 if ready and self.opponent.handCount>=6 else -1
  if c.id==C.JUDGE:return 3275 if self.opponent.handCount>=6 and self.me.handCount<=4 else -1
 return _m4_prev_tr(self,c)
LucarioPolicy._score_evolve=_m4_evolve;LucarioPolicy._score_to_hand=_m4_hand;LucarioPolicy._score_play_trainer=_m4_tr
_m4_internal=agent
del agent
def agent(obs_dict):return _m4_internal(obs_dict)

# --- Aura Jab resource-allocation learning for Lucario mirror ---
_M7_R70=False;_M7_CAP=99
_m7_prev_choose=LucarioPolicy.choose
_m7_prev_choice=LucarioPolicy._score_card_choice
_m7_prev_attach=LucarioPolicy._score_attach
_m7_prev_setup=LucarioPolicy._score_setup_active
_m7_prev_hand=LucarioPolicy._score_to_hand
_m7_prev_play=LucarioPolicy._score_play_pokemon
_m7_prev_active=LucarioPolicy._score_active_choice

def _m7_lines(self):return self.field_counts[C.RIOLU]+self.field_counts[C.RIOLU_70]+self.field_counts[C.MEGA_LUCARIO_EX]
def _m7_is_luc(p):return p is not None and p.id in {C.RIOLU,C.RIOLU_70,C.MEGA_LUCARIO_EX}
def _m7_needed(self):
 n=0
 for p in self.me.bench:
  if _m7_is_luc(p):n+=max(0,2-len(p.energies))
 return n

def _m7_choose(self):
 # Aura Jab first selects up to three Basic Fighting Energy cards from discard.
 # Select only as many as can be usefully placed on benched Lucario lines.
 if self.context==SelectContext.ATTACH_TO and self.me.active and self.me.active[0].id==C.MEGA_LUCARIO_EX and _d_arch(self)=='lucario':
  cards=[]
  for i,o in enumerate(self.select.option):
   try:c=get_card(self.obs,o.area,o.index,getattr(o,'playerIndex',self.my_index))
   except Exception:c=None
   if c is not None and c.id==C.BASIC_FIGHTING_ENERGY:cards.append(i)
  if cards and len(cards)==len(self.select.option):
   n=max(self.select.minCount,min(self.select.maxCount,_m7_needed(self)))
   return cards[:n]
 return _m7_prev_choose(self)

def _m7_choice(self,o):
 if self.context==SelectContext.ATTACH_FROM and _d_arch(self)=='lucario':
  try:p=get_card(self.obs,o.area,o.index,getattr(o,'playerIndex',self.my_index))
  except Exception:p=None
  if p is not None:
   if _m7_is_luc(p):return 10000-1000*len(p.energies)+(100 if p.id==C.MEGA_LUCARIO_EX else 0)
   return -10000
 return _m7_prev_choice(self,o)

def _m7_attach(self,o):
 c=get_card(self.obs,AreaType.HAND,o.index,self.my_index);p=get_card(self.obs,o.inPlayArea,o.inPlayIndex,self.my_index)
 if _d_arch(self)=='lucario' and c is not None and p is not None and c.id in {C.BASIC_FIGHTING_ENERGY,C.ROCK_FIGHTING_ENERGY} and p.id in {C.DUNSPARCE,C.DUDUNSPARCE,C.DUDUNSPARCE_EX}:return -1
 return _m7_prev_attach(self,o)
def _m7_setup(self,c):
 if _M7_R70 and c.id==C.RIOLU_70:return 3
 return _m7_prev_setup(self,c)
def _m7_hand(self,c):
 if _M7_R70 and c.id==C.RIOLU_70:
  lines=_m7_lines(self);return 240-self.hand_counts[c.id]*100+(-150 if lines>=2 else -3 if lines>=1 else 40)
 return _m7_prev_hand(self,c)
def _m7_play(self,c):
 if _M7_R70 and c.id==C.RIOLU_70 and _m7_lines(self)>=2:return -1
 if _d_arch(self)=='lucario' and c.id==C.DUNSPARCE and self.field_counts[C.DUNSPARCE]+self.field_counts[C.DUDUNSPARCE]>=_M7_CAP:return -1
 return _m7_prev_play(self,c)
def _m7_active(self,o,c):
 s=_m7_prev_active(self,o,c)
 return s+4 if _M7_R70 and c.id==C.RIOLU_70 else s
LucarioPolicy.choose=_m7_choose;LucarioPolicy._score_card_choice=_m7_choice;LucarioPolicy._score_attach=_m7_attach
LucarioPolicy._score_setup_active=_m7_setup;LucarioPolicy._score_to_hand=_m7_hand;LucarioPolicy._score_play_pokemon=_m7_play;LucarioPolicy._score_active_choice=_m7_active
_m7_internal=agent
del agent
def agent(obs_dict):return _m7_internal(obs_dict)


# === self-play learned Crustle route: preserve and charge Dudunsparce ex ===
_c1_evo=LucarioPolicy._score_evolve
_c1_energy=LucarioPolicy._energy_target_score
_c1_ability=LucarioPolicy._score_ability
_c1_tr=LucarioPolicy._score_play_trainer
_c1_hand=LucarioPolicy._score_to_hand

def _c1_has_ex(self):
 return any(p is not None and p.id==C.DUDUNSPARCE_EX for p in self._my_board())
def _c1_duns(self):
 return sum(1 for p in self._my_board() if p is not None and p.id==C.DUNSPARCE)

def _c1_evolve(self,o):
 s=_c1_evo(self,o)
 if _d_arch(self)!='crustle':return s
 e=get_card(self.obs,o.area,o.index,self.my_index)
 if e is None:return s
 if e.id==C.DUDUNSPARCE_EX:return s+4200
 if e.id==C.DUDUNSPARCE and not _c1_has_ex(self) and _c1_duns(self)<=1:return -1
 return s

def _c1_energy_score(self,p,active):
 s=_c1_energy(self,p,active)
 if _d_arch(self)=='crustle':
  if p.id==C.DUDUNSPARCE_EX:return s+(2600 if len(p.energies)<3 else -800)
  if p.id in {C.DUNSPARCE,C.DUDUNSPARCE}:return s-1600
 return s

def _c1_ability_score(self,o):
 c=get_card(self.obs,o.area,o.index,self.my_index)
 if _d_arch(self)=='crustle' and c is not None and c.id==C.DUDUNSPARCE:
  # Never shuffle away energy or the final evolution base needed for the ex attacker.
  if len(c.energies)>0 or (not _c1_has_ex(self) and _c1_duns(self)==0):return -1
 return _c1_ability(self,o)

def _c1_trainer(self,c):
 if _d_arch(self)=='crustle' and c.id==C.HILDA:
  need=any(p is not None and p.id==C.DUNSPARCE for p in self._my_board()) and not _c1_has_ex(self) and self.hand_counts[C.DUDUNSPARCE_EX]==0
  if need:return 5150
 return _c1_tr(self,c)

def _c1_to_hand(self,c):
 s=_c1_hand(self,c)
 if _d_arch(self)=='crustle':
  if c.id==C.DUDUNSPARCE_EX:s+=1400
  elif c.id==C.DUDUNSPARCE and not _c1_has_ex(self):s-=900
  elif c.id==C.DUNSPARCE and _c1_duns(self)==0:s+=500
 return s

LucarioPolicy._score_evolve=_c1_evolve
LucarioPolicy._energy_target_score=_c1_energy_score
LucarioPolicy._score_ability=_c1_ability_score
LucarioPolicy._score_play_trainer=_c1_trainer
LucarioPolicy._score_to_hand=_c1_to_hand


# Crustle prize-race learning: Hero's Cape belongs on the bypass attacker.
_c2_attach=LucarioPolicy._score_attach
_c2_active=LucarioPolicy._score_active_choice

def _c2_attach_score(self,o):
 s=_c2_attach(self,o)
 if _d_arch(self)=='crustle':
  c=get_card(self.obs,AreaType.HAND,o.index,self.my_index)
  p=get_card(self.obs,o.inPlayArea,o.inPlayIndex,self.my_index)
  if c is not None and p is not None:
   if c.id==C.HERO_CAPE:
    if p.id==C.DUDUNSPARCE_EX:return 14500
    if p.id in {C.RIOLU,C.RIOLU_70,C.MEGA_LUCARIO_EX}:return 3000
   if c.id==C.AIR_BALLOON and p.id==C.DUDUNSPARCE_EX:return 3500
 return s

def _c2_active_score(self,o,c):
 s=_c2_active(self,o,c)
 if _d_arch(self)=='crustle' and c is not None:
  if c.id==C.DUDUNSPARCE_EX and len(c.energies)>=3:s+=3000
  elif c.id==C.MEGA_LUCARIO_EX:s-=1000
 return s
LucarioPolicy._score_attach=_c2_attach_score
LucarioPolicy._score_active_choice=_c2_active_score


# Generalized exact Premium Power Pro conservation learned from self-play losses.
_c3_tr=LucarioPolicy._score_play_trainer
def _c3_trainer(self,c):
 if c.id==C.PREMIUM_POWER_PRO and self.can_attack and plan.target>=0:
  t=_m3_tgt(self)
  bb=40 if (_m3_bb and t is not None and (card_table[t.id].ex or card_table[t.id].megaEx)) else 0
  remaining=plan.remain_hp-bb-30*_m3_ppp
  return 5000 if remaining>0 else -1
 return _c3_tr(self,c)
LucarioPolicy._score_play_trainer=_c3_trainer

# PBT-selected bench composition
_M7_R70=True
_M7_CAP=2


# Late-game anti-loop gate learned from rare Dudunsparce mirror step-limit games.
_c6_ability=LucarioPolicy._score_ability
def _c6_ability_score(self,o):
 c=get_card(self.obs,o.area,o.index,self.my_index)
 if c is not None and c.id==C.DUDUNSPARCE:
  if self.me.deckCount<=5:return -1
  if self.state.turn>=24 and self.me.handCount>=5:return -1
 return _c6_ability(self,o)
LucarioPolicy._score_ability=_c6_ability_score

# ============================================================================
# Learned turn-planning engine v2
# - Recomputes tactics after every action (no stale global attack plan).
# - Uses exact-KO sequence constraints before generic action scoring.
# - Uses a shallow deterministic simulator with a learned linear value function
#   for non-forced choices. The value weights are tuned by population self-play.
# ============================================================================
from collections import Counter as _Counter
from cg.api import search_begin as _search_begin, search_step as _search_step, search_end as _search_end, search_release as _search_release

_AURA_JAB = 982
_MEGA_BRAVE = 983

# These weights are replaced by the offline population trainer when a stronger
# candidate is selected. Keeping them inline preserves Kaggle/raw-exec support.
_LEARNED_VALUE_WEIGHTS = {
    "prize_gain": 12000.0,
    "damage_gain": 7.0,
    "ready_lucario": 430.0,
    "charged_bench": 170.0,
    "hand": 22.0,
    "deck_safety": 18.0,
    "bad_energy": -150.0,
    "bad_cape": -2200.0,
    "stadium_control": 260.0,
    "dudun_engine": 105.0,
    "ppp_spent": -210.0,
    "boss_spent": -90.0,
    "supporter_spent": -35.0,
    "bench_lock": -170.0,
}


def _lt_cards_in_play(player):
    out=[]
    for p in (player.active or []) + (player.bench or []):
        if p is None: continue
        out.append(p)
    return out


def _lt_known_ids(player):
    ids=[]
    for c in (player.hand or []): ids.append(c.id)
    for c in (player.discard or []): ids.append(c.id)
    for p in _lt_cards_in_play(player):
        ids.append(p.id)
        ids.extend(getattr(x,"id",-1) for x in (p.preEvolution or []))
        ids.extend(getattr(x,"id",-1) for x in (p.energyCards or []))
        ids.extend(getattr(x,"id",-1) for x in (p.tools or []))
    return [x for x in ids if x >= 0]


def _lt_remove_known(deck, known):
    cnt=_Counter(deck)
    for cid in known:
        if cnt[cid] > 0: cnt[cid]-=1
    out=[]
    for cid in sorted(cnt): out.extend([cid]*cnt[cid])
    return out


def _lt_generic_opponent_pool(policy):
    # A legal exact opponent list is not required by SearchBegin; it needs card
    # identities for hidden zones. Prefer public/seen cards and fill any shortfall
    # with basic energy plus a basic Pokemon so setup determinizations stay valid.
    op=policy.opponent
    seen=_lt_known_ids(op)
    pool=list(seen)
    # Add known archetype staples when identified. Only affects our-turn draw/
    # shuffle simulations; opponent turns are not rolled out.
    arch=_d_arch(policy)
    templates={
        'alakazam':[741]*4+[742]*4+[743]*3+[305]*3+[66]*2+[5]*8+[19]*4,
        'dragapult':[119]*4+[120]*4+[121]*3+[6]*8,
        'crustle':[344]*4+[345]*4+[6]*10,
        'lucario':[333]*3+[677]+[678]*4+[676]*2+[675]*2+[6]*10,
    }
    pool.extend(templates.get(arch,[]))
    pool.extend([6]*60)
    return pool


def _lt_hidden_lists(policy):
    st=policy.state; me=policy.me; op=policy.opponent
    rem=_lt_remove_known(my_deck,_lt_known_ids(me))
    # Unknown prizes first, then deck. Rotate by turn to avoid a fixed top-deck bias.
    if rem:
        k=(st.turn*7 + me.deckCount) % len(rem)
        rem=rem[k:]+rem[:k]
    yp=rem[:len(me.prize)]
    yd=rem[len(me.prize):len(me.prize)+me.deckCount]
    while len(yp)<len(me.prize): yp.append(C.BASIC_FIGHTING_ENERGY)
    while len(yd)<me.deckCount: yd.append(C.BASIC_FIGHTING_ENERGY)

    pool=_lt_generic_opponent_pool(policy)
    # Remove public opponent cards where possible and then deterministically rotate.
    pool=_lt_remove_known(pool,_lt_known_ids(op))
    need=op.handCount+len(op.prize)+op.deckCount
    if len(pool)<need: pool.extend([6]*(need-len(pool)))
    if pool:
        k=(st.turn*11 + op.deckCount) % len(pool)
        pool=pool[k:]+pool[:k]
    oh=pool[:op.handCount]
    opr=pool[op.handCount:op.handCount+len(op.prize)]
    od=pool[op.handCount+len(op.prize):op.handCount+len(op.prize)+op.deckCount]
    return yd,yp,od,opr,oh,[]


def _lt_fighting_damage(attacker, attack_id, target, ppp=0, black_belt=False):
    if attacker is None or target is None: return 0
    if attacker.id==C.MEGA_LUCARIO_EX:
        dmg=130 if attack_id==_AURA_JAB else 270 if attack_id==_MEGA_BRAVE else 0
    elif attacker.id==C.DUDUNSPARCE_EX:
        if attack_id==ATTACK_DUDUN_TAIL:
            # Tenacious Tail: 60 per opposing Pokemon ex.
            dmg=0  # Filled by caller because it needs the full opposing board.
        elif attack_id==ATTACK_DUDUN_DRILL: dmg=150
        else: dmg=0
    elif attacker.id==C.SOLROCK: dmg=70
    else: dmg=0
    if dmg<=0:return 0
    dmg += 30*ppp
    if black_belt and (card_table[target.id].ex or card_table[target.id].megaEx): dmg += 40
    td=card_table[target.id]
    if td.weakness==EnergyType.FIGHTING:dmg*=2
    elif td.resistance==EnergyType.FIGHTING:dmg=max(0,dmg-30)
    return dmg


def _lt_available_attack_ids(policy):
    out=[]
    for o in policy.select.option:
        if o.type==OptionType.ATTACK: out.append(o.attackId)
    return out


def _lt_option_card(policy,o):
    try:
        if o.type in {OptionType.PLAY,OptionType.ATTACH,OptionType.EVOLVE,OptionType.ABILITY}:
            return get_card(policy.obs, o.area if o.type!=OptionType.PLAY else AreaType.HAND, o.index, policy.my_index)
    except Exception:
        return None
    return None


class _ExactTurnTactics:
    def __init__(self,policy):
        self.p=policy; self.me=policy.me; self.op=policy.opponent
        self.active=self.me.active[0] if self.me.active else None
        self.target=self.op.active[0] if self.op.active else None
        self.attack_ids=_lt_available_attack_ids(policy)
        self.ppp_in_hand=sum(1 for c in self.me.hand if c.id==C.PREMIUM_POWER_PRO)
        self.bb_in_hand=sum(1 for c in self.me.hand if c.id==C.BLACK_BELT_TRAINING)
        self.energy_in_hand=sum(1 for c in self.me.hand if c.id in {C.BASIC_FIGHTING_ENERGY,C.ROCK_FIGHTING_ENERGY})

    def _energy_ready_after_attach(self,attack_id):
        if self.active is None:return False
        need=1 if attack_id==_AURA_JAB else 2 if attack_id==_MEGA_BRAVE else 1
        have=len(self.active.energies)
        return have>=need or (not self.p.state.energyAttached and self.energy_in_hand>0 and have+1>=need)

    def active_exact_ko(self):
        if self.active is None or self.target is None:return None
        candidates=[]
        # Include attacks that become available after the one normal attachment.
        aids=set(self.attack_ids)
        if self.active.id==C.MEGA_LUCARIO_EX:
            aids.update([_AURA_JAB,_MEGA_BRAVE])
        for aid in aids:
            if aid not in {_AURA_JAB,_MEGA_BRAVE}:continue
            if not self._energy_ready_after_attach(aid):continue
            for bb in range(0,1+min(1,self.bb_in_hand if not self.p.state.supporterPlayed else 0)):
                base=_lt_fighting_damage(self.active,aid,self.target,0,bb>0)
                deficit=max(0,self.target.hp-base)
                need_ppp=(deficit+29)//30
                if need_ppp<=self.ppp_in_hand:
                    # Lower resource/attack cost wins ties. Aura Jab is preferred when both KO.
                    cost=need_ppp*100 + bb*75 + (0 if aid==_AURA_JAB else 25)
                    candidates.append((cost,aid,need_ppp,bb))
        return min(candidates) if candidates else None

    def forced_main_action(self):
        ko=self.active_exact_ko()
        if ko is None:return None
        _,aid,need_ppp,need_bb=ko
        # Do not burn rare damage modifiers merely to take a routine one-prize KO.
        # Free/low-cost exact KOs are always forced; boosted KOs are forced for ex,
        # critical engines, or when they finish the game.
        target_prizes=prize_count(self.target) if self.target is not None else 0
        critical=self.target is not None and self.target.id in {C.ABRA,C.KADABRA,C.ALAKAZAM,C.RIOLU,C.RIOLU_70}
        game_winning=self.target is not None and len(self.me.prize)<=target_prizes
        if (need_ppp or need_bb) and target_prizes<=1 and not critical and not game_winning:
            return None
        # One normal energy attachment may be mandatory before the attack.
        need_energy=1 if aid==_AURA_JAB else 2
        if self.active is not None and len(self.active.energies)<need_energy and not self.p.state.energyAttached:
            best=None
            for i,o in enumerate(self.p.select.option):
                if o.type!=OptionType.ATTACH:continue
                c=_lt_option_card(self.p,o)
                tgt=get_card(self.p.obs,o.inPlayArea,o.inPlayIndex,self.p.my_index)
                if c is not None and tgt is self.active and c.id in {C.BASIC_FIGHTING_ENERGY,C.ROCK_FIGHTING_ENERGY}:
                    best=i;break
            if best is not None:return best
        # Black Belt must precede item boosts.
        if need_bb and not self.p.state.supporterPlayed:
            for i,o in enumerate(self.p.select.option):
                if o.type==OptionType.PLAY:
                    c=_lt_option_card(self.p,o)
                    if c is not None and c.id==C.BLACK_BELT_TRAINING:return i
        # Count already-played PPP from discard changes is awkward; recompute current
        # damage. If not yet enough, play exactly one more PPP.
        cur=_lt_fighting_damage(self.active,aid,self.target,0,False)
        # Black Belt effect is active after supporter play but not explicitly stored;
        # the legacy tracker records it for this turn.
        if globals().get('_m3_bb',False):cur+=40 if (card_table[self.target.id].ex or card_table[self.target.id].megaEx) else 0
        remaining=max(0,self.target.hp-cur-30*globals().get('_m3_ppp',0))
        if remaining>0:
            for i,o in enumerate(self.p.select.option):
                if o.type==OptionType.PLAY:
                    c=_lt_option_card(self.p,o)
                    if c is not None and c.id==C.PREMIUM_POWER_PRO:return i
        for i,o in enumerate(self.p.select.option):
            if o.type==OptionType.ATTACK and o.attackId==aid:return i
        return None


def _lt_snapshot(obs):
    st=obs.current; yi=st.yourIndex; me=st.players[yi]; op=st.players[1-yi]
    def board_hp(pl):return sum(p.hp for p in _lt_cards_in_play(pl))
    def counts(pl,cid):return sum(1 for p in _lt_cards_in_play(pl) if p.id==cid)
    return {
        'my_prize':len(me.prize),'op_prize':len(op.prize),
        'op_hp':board_hp(op),'my_hp':board_hp(me),
        'my_hand':me.handCount,'my_deck':me.deckCount,
        'ppp_discard':sum(1 for c in me.discard if c.id==C.PREMIUM_POWER_PRO),
        'boss_discard':sum(1 for c in me.discard if c.id==C.BOSS_ORDERS),
        'supporter_played':1 if st.supporterPlayed else 0,
        'stadium':st.stadium[0].id if st.stadium else 0,
        'bench_count':len(me.bench),
    }


def _lt_state_value(root_obs,end_obs,weights=None):
    w=weights or _LEARNED_VALUE_WEIGHTS
    r=_lt_snapshot(root_obs); e=_lt_snapshot(end_obs)
    st=end_obs.current; yi=st.yourIndex; me=st.players[yi]; op=st.players[1-yi]
    prize_gain=max(0,r['my_prize']-e['my_prize'])
    damage_gain=max(0,r['op_hp']-e['op_hp'])
    ready=0; charged=0; bad_energy=0; dudun=0; bad_cape=0
    arch_ids={p.id for p in _lt_cards_in_play(op)}
    crustle=bool(arch_ids & {C.DWEBBLE,C.CRUSTLE,756})
    for p in _lt_cards_in_play(me):
        if p.id==C.MEGA_LUCARIO_EX:
            ready += 1 if len(p.energies)>=1 else 0
            charged += min(2,len(p.energies))
        elif p.id in {C.RIOLU,C.RIOLU_70}:
            charged += min(2,len(p.energies))*.6
        elif p.id in {C.SOLROCK,C.LUNATONE,C.DUNSPARCE,C.DUDUNSPARCE}:
            bad_energy += max(0,len(p.energies)-(1 if p.id==C.SOLROCK else 0))
        if p.id==C.DUDUNSPARCE:dudun+=1
        for t in p.tools:
            if t.id==C.HERO_CAPE and not (p.id==C.MEGA_LUCARIO_EX or (crustle and p.id==C.DUDUNSPARCE_EX)):
                bad_cape+=1
    stadium_control=1 if (r['stadium'] and e['stadium']!=r['stadium']) else 0
    ppp_spent=max(0,e['ppp_discard']-r['ppp_discard'])
    boss_spent=max(0,e['boss_discard']-r['boss_discard'])
    supporter_spent=max(0,e['supporter_played']-r['supporter_played'])
    deck_safety=min(e['my_deck'],12)
    bench_lock=1 if e['bench_count']>=5 else 0
    val=(w['prize_gain']*prize_gain+w['damage_gain']*damage_gain+w['ready_lucario']*ready+
         w['charged_bench']*charged+w['hand']*e['my_hand']+w['deck_safety']*deck_safety+
         w['bad_energy']*bad_energy+w['bad_cape']*bad_cape+w['stadium_control']*stadium_control+
         w['dudun_engine']*dudun+w['ppp_spent']*ppp_spent+w['boss_spent']*boss_spent+
         w['supporter_spent']*supporter_spent+w['bench_lock']*bench_lock)
    if st.result==yi: val+=1000000
    elif st.result==1-yi: val-=1000000
    return float(val)


def _lt_rollout_choice(obs):
    # Search observations are dataclasses and contain no search_begin_input.
    # Recompute the legacy policy from the simulated state; never call the outer
    # search wrapper recursively.
    try:return LucarioPolicy(obs).choose()
    except Exception:
        if obs.select is None:return []
        return list(range(min(obs.select.maxCount,len(obs.select.option))))


def _lt_rollout_main(policy, fallback_index):
    if policy.obs.search_begin_input is None:return fallback_index
    options=policy.select.option
    if len(options)<=1:return fallback_index
    # Preserve setup/time by evaluating at most six serious candidates. The base
    # score supplies a strong prior; attacks and tactical resources are always kept.
    raw=[]
    for i,o in enumerate(options):
        s=policy._score_option(o)
        c=_lt_option_card(policy,o)
        important=o.type in {OptionType.ATTACK,OptionType.ATTACH,OptionType.EVOLVE,OptionType.ABILITY}
        if c is not None and c.id in {C.PREMIUM_POWER_PRO,C.BOSS_ORDERS,C.BLACK_BELT_TRAINING,C.GRAVITY_MOUNTAIN,C.LILLIE_DETERMINATION,C.HILDA,C.JUDGE,C.XEROSIC}:important=True
        raw.append((1 if important else 0,s,i))
    raw.sort(reverse=True)
    cand=[]
    for _,_,i in raw:
        if i not in cand:cand.append(i)
        if len(cand)>=6:break
    if fallback_index not in cand:cand[-1]=fallback_index
    try:
        yd,yp,od,opr,oh,oa=_lt_hidden_lists(policy)
        root=_search_begin(policy.obs,yd,yp,od,opr,oh,oa,manual_coin=False)
    except Exception:
        return fallback_index
    best=(float('-inf'),fallback_index)
    root_turn=policy.state.turn
    try:
        for idx in cand:
            sid=None
            try:
                st=_search_step(root.searchId,[idx]);sid=st.searchId
                depth=0
                while st.observation.select is not None and st.observation.current.result==-1 and st.observation.current.turn==root_turn and depth<10:
                    act=_lt_rollout_choice(st.observation)
                    if len(act)<st.observation.select.minCount:
                        act=list(range(st.observation.select.minCount))
                    st=_search_step(st.searchId,act);sid=st.searchId;depth+=1
                val=_lt_state_value(policy.obs,st.observation)
                # Tiny base-policy prior stabilizes equally valued lines.
                val+=0.015*policy._score_option(options[idx])
                if val>best[0]:best=(val,idx)
            except Exception:
                pass
            finally:
                if sid is not None:
                    try:_search_release(sid)
                    except Exception:pass
    finally:
        try:_search_end()
        except Exception:pass
    return best[1]


_LT_USE_ROLLOUT=False
_LT_EXACT_KO=True
_LT_AURA_GENERAL=True
_LT_CAPE_STRICT=True
_LT_SEARCH_BLOCK=True
_LT_PPP_ONLY_KO=False

_lt_previous_agent=agent
_lt_previous_choose=LucarioPolicy.choose
_lt_previous_card_choice=LucarioPolicy._score_card_choice
_lt_previous_attach=LucarioPolicy._score_attach
_lt_previous_play_pokemon=LucarioPolicy._score_play_pokemon
_lt_previous_trainer=LucarioPolicy._score_play_trainer
_lt_last_turn=-1
_lt_main_calls=0


def _lt_safe_card_choice(self,o):
    # General Aura Jab allocation, not merely Lucario-mirror allocation.
    if _LT_AURA_GENERAL and _d_arch(self) in {'alakazam','crustle'} and self.context==SelectContext.ATTACH_FROM and getattr(getattr(self.select,'effect',None),'id',-1)==C.MEGA_LUCARIO_EX:
        try:p=get_card(self.obs,o.area,o.index,getattr(o,'playerIndex',self.my_index))
        except Exception:p=None
        if p is not None:
            arch=_d_arch(self)
            if arch=='crustle' and p.id==C.DUDUNSPARCE_EX:
                return 26000-2500*len(p.energies)
            if p.id in {C.MEGA_LUCARIO_EX,C.RIOLU,C.RIOLU_70}:
                return 20000-2500*len(p.energies)+(300 if p.id==C.MEGA_LUCARIO_EX else 0)
            if p.id==C.SOLROCK and len(p.energies)==0:return 3000
            return -20000
    return _lt_previous_card_choice(self,o)


def _lt_safe_attach(self,o):
    c=get_card(self.obs,AreaType.HAND,o.index,self.my_index)
    p=get_card(self.obs,o.inPlayArea,o.inPlayIndex,self.my_index)
    if _LT_CAPE_STRICT and _d_arch(self) in {'alakazam','crustle'} and c is not None and p is not None and c.id==C.HERO_CAPE:
        if p.id in {C.MEGA_LUCARIO_EX,C.RIOLU,C.RIOLU_70}:return 16000 if p.id==C.MEGA_LUCARIO_EX else 15000
        if _d_arch(self)=='crustle' and p.id==C.DUDUNSPARCE_EX:return 15800
        return -1
    return _lt_previous_attach(self,o)


def _lt_safe_pokemon(self,c):
    # Do not fill the final bench slot with an unusable search target.
    if len(self.me.bench)>=5:return -1
    return _lt_previous_play_pokemon(self,c)


def _lt_aura_active_ko(self):
    if _d_arch(self)!='alakazam' or not self.me.active or not self.opponent.active:return False
    a=self.me.active[0];t=self.opponent.active[0]
    if a.id!=C.MEGA_LUCARIO_EX:return False
    if not any(o.type==OptionType.ATTACK and o.attackId==_AURA_JAB for o in self.select.option):return False
    return t.hp<=_lt_fighting_damage(a,_AURA_JAB,t,0,False)

def _lt_safe_trainer(self,c):
    # When Aura Jab already KOs, never spend PPP or gust away the target.
    if _lt_aura_active_ko(self) and c.id in {C.PREMIUM_POWER_PRO,C.BOSS_ORDERS}:
        return -1
    # Premium Power Pro is a tactical KO resource, not generic chip damage.
    if _LT_PPP_ONLY_KO and c.id==C.PREMIUM_POWER_PRO:
        tact=_ExactTurnTactics(self).active_exact_ko()
        if tact is None or tact[2]<=0:
            return -1
    # Search is illegal strategically when no bench/evolution target can be used.
    if _LT_SEARCH_BLOCK and _d_arch(self)=='alakazam' and c.id==C.POKE_PAD:
        bench_full=len(self.me.bench)>=self.me.benchMax
        need_evo=(
            (self.field_counts[C.RIOLU]+self.field_counts[C.RIOLU_70]>0 and self.hand_counts[C.MEGA_LUCARIO_EX]==0) or
            (self.field_counts[C.DUNSPARCE]>0 and self.hand_counts[C.DUDUNSPARCE]+self.hand_counts[C.DUDUNSPARCE_EX]==0)
        )
        useful=(
            need_evo or
            (not bench_full and (
                self.field_counts[C.RIOLU]+self.field_counts[C.RIOLU_70]+self.field_counts[C.MEGA_LUCARIO_EX]<2 or
                self.field_counts[C.SOLROCK]==0 or self.field_counts[C.LUNATONE]==0 or
                self.field_counts[C.DUNSPARCE]+self.field_counts[C.DUDUNSPARCE]+self.field_counts[C.DUDUNSPARCE_EX]<2
            ))
        )
        if not useful:return -1
    return _lt_previous_trainer(self,c)



def _lt_replay_guard(self):
    """Conservative learned tactical guard from the human Alakazam loss.

    It does not replace the whole turn policy. It only intervenes when the current
    Active Mega Lucario has an immediate low-cost KO, or when a late-game two/three
    prize KO is fully guaranteed with the Premium Power Pro cards already in hand.
    """
    if self.context!=SelectContext.MAIN or _d_arch(self)!='alakazam' or not self.me.active or not self.opponent.active:
        return None
    a=self.me.active[0];t=self.opponent.active[0]
    if a.id!=C.MEGA_LUCARIO_EX:return None
    attacks={o.attackId:i for i,o in enumerate(self.select.option) if o.type==OptionType.ATTACK}
    if _AURA_JAB not in attacks:return None
    aura=_lt_fighting_damage(a,_AURA_JAB,t,0,False)
    # Low-cost KOs are handled as constraints in trainer/attack scoring so safe
    # setup actions may still happen before the attack.
    if t.hp<=aura:
        return None
    # Late-game ex KO, exactly the Clefairy-ex failure from the uploaded replay.
    pr=prize_count(t)
    if len(self.me.prize)<=3 and pr>=2:
        already=globals().get('_m3_ppp',0)
        need=max(0,(t.hp-aura-30*already+29)//30)
        have=sum(1 for c in self.me.hand if c.id==C.PREMIUM_POWER_PRO)
        if need<=have:
            if need>0:
                for i,o in enumerate(self.select.option):
                    if o.type==OptionType.PLAY:
                        c=_lt_option_card(self,o)
                        if c is not None and c.id==C.PREMIUM_POWER_PRO:return i
            return attacks[_AURA_JAB]
    return None

def _lt_choose(self):
    global _lt_last_turn,_lt_main_calls
    if self.state.turn!=_lt_last_turn:
        _lt_last_turn=self.state.turn;_lt_main_calls=0
    # Select only the useful number of Aura Jab energies.
    if _LT_AURA_GENERAL and _d_arch(self) in {'alakazam','crustle'} and self.context==SelectContext.ATTACH_TO and getattr(getattr(self.select,'effect',None),'id',-1)==C.MEGA_LUCARIO_EX and self.me.active and self.me.active[0].id==C.MEGA_LUCARIO_EX:
        energy_opts=[]
        for i,o in enumerate(self.select.option):
            try:c=get_card(self.obs,o.area,o.index,getattr(o,'playerIndex',self.my_index))
            except Exception:c=None
            if c is not None and c.id==C.BASIC_FIGHTING_ENERGY:energy_opts.append(i)
        need=0;arch=_d_arch(self)
        for p in self.me.bench:
            if p.id in {C.MEGA_LUCARIO_EX,C.RIOLU,C.RIOLU_70}:need+=max(0,2-len(p.energies))
            elif arch=='crustle' and p.id==C.DUDUNSPARCE_EX:need+=max(0,3-len(p.energies))
        n=max(self.select.minCount,min(self.select.maxCount,need))
        if energy_opts:return energy_opts[:n]

    base=_lt_previous_choose(self)
    if self.context!=SelectContext.MAIN or not base:return base
    _lt_main_calls+=1
    # Preserve setup order, but when the base policy finally attacks, use the
    # cheapest attack that still secures the KO.
    if _lt_aura_active_ko(self):
        bo=self.select.option[base[0]]
        if bo.type==OptionType.ATTACK:
            for i,o in enumerate(self.select.option):
                if o.type==OptionType.ATTACK and o.attackId==_AURA_JAB:return [i]
    forced=_lt_replay_guard(self) if _LT_EXACT_KO else None
    if forced is not None:return [forced]
    # Search on the first few consequential main decisions only; exact tactical
    # constraints remain active on every call.
    idx=base[0]
    if _LT_USE_ROLLOUT and _lt_main_calls<=5:
        idx=_lt_rollout_main(self,idx)
    return [idx]

LucarioPolicy._score_card_choice=_lt_safe_card_choice
LucarioPolicy._score_attach=_lt_safe_attach
LucarioPolicy._score_play_pokemon=_lt_safe_pokemon
LucarioPolicy._score_play_trainer=_lt_safe_trainer
LucarioPolicy.choose=_lt_choose

_lt_engine_internal=agent
del agent
def agent(obs_dict:dict)->list[int]:
    return _lt_engine_internal(obs_dict)

C.TEAM_ROCKET_WATCHTOWER=1256

_v_at=LucarioPolicy._score_attach
def _v_atf(self,o):
 c=get_card(self.obs,AreaType.HAND,o.index,self.my_index);p=get_card(self.obs,o.inPlayArea,o.inPlayIndex,self.my_index)
 if c is not None and p is not None and c.id==C.AIR_BALLOON:
  if o.inPlayArea==AreaType.ACTIVE:
   ready=any((q.id==C.MEGA_LUCARIO_EX and len(q.energies)>=1) or (q.id==C.DUDUNSPARCE_EX and len(q.energies)>=3) or (q.id==C.SOLROCK and len(q.energies)>=1) for q in self.me.bench if q is not None)
   if ready and p.id not in {C.MEGA_LUCARIO_EX,C.DUDUNSPARCE_EX}:return 25000
  if _d_arch(self)=='crustle' and p.id==C.DUDUNSPARCE_EX:return 15000
  if o.inPlayArea==AreaType.BENCH and p.id in {C.RIOLU,C.RIOLU_70,C.MEGA_LUCARIO_EX}:return -1
  if p.id in {C.DUNSPARCE,C.DUDUNSPARCE}:return -1
 return _v_at(self,o)
LucarioPolicy._score_attach=_v_atf

_v_tr=LucarioPolicy._score_play_trainer
def _v_trf(self,c):
 if _d_arch(self)=='dragapult' and len(self.opponent.prize)<=1:
  if c.id==C.JUDGE and self.opponent.handCount>=5:return 20000
  if c.id==C.XEROSIC and self.hand_counts[C.JUDGE]>0:return -1
 return _v_tr(self,c)
LucarioPolicy._score_play_trainer=_v_trf

_v_tr2=LucarioPolicy._score_play_trainer
def _v_tr2f(self,c):
 if c.id==C.PREMIUM_POWER_PRO and _d_arch(self)=='crustle' and self.me.active:
  a=self.me.active[0]
  if a.id==C.DUDUNSPARCE_EX:return -1
  if a.id==C.MEGA_LUCARIO_EX:
   ready_d=any(p is not None and p.id==C.DUDUNSPARCE_EX and len(p.energies)>=3 for p in self.me.bench)
   ready_s=any(p is not None and p.id==C.SOLROCK and len(p.energies)>=1 for p in self.me.bench) and any(p is not None and p.id==C.LUNATONE for p in self._my_board())
   if ready_d or not ready_s:return -1
 return _v_tr2(self,c)
LucarioPolicy._score_play_trainer=_v_tr2f

_vint=agent
del agent
def agent(obs_dict):return _vint(obs_dict)

# Replay-learned: do not spend a setup action before a damaged Dudunsparce ex takes a guaranteed Drill KO.
_v2_choose=LucarioPolicy.choose
def _v2_choosef(self):
 if self.context==SelectContext.MAIN and self.me.active and self.opponent.active:
  a=self.me.active[0];t=self.opponent.active[0]
  if a.id==C.DUDUNSPARCE_EX and a.hp<=120 and t.hp<=150:
   for i,o in enumerate(self.select.option):
    if o.type==OptionType.ATTACK and o.attackId==ATTACK_DUDUN_DRILL:return [i]
 return _v2_choose(self)
LucarioPolicy.choose=_v2_choosef
# Replay-learned emergency exact KO: a damaged Lucario facing a nearly-KO'd Dragapult ex
# attacks before bench/search actions when its bench is already nearly full.
_v3_choose=LucarioPolicy.choose
def _v3_choosef(self):
 if self.context==SelectContext.MAIN and _d_arch(self)=='dragapult' and self.me.active and self.opponent.active:
  a=self.me.active[0];t=self.opponent.active[0]
  if a.id==C.MEGA_LUCARIO_EX and a.hp<=150 and len(self.me.bench)>=4 and t.hp<=130:
   for i,o in enumerate(self.select.option):
    if o.type==OptionType.ATTACK and o.attackId==_AURA_JAB and _lt_fighting_damage(a,_AURA_JAB,t,0,False)>=t.hp:return [i]
 return _v3_choose(self)
LucarioPolicy.choose=_v3_choosef
_v3int=agent
del agent
def agent(obs_dict):return _v3int(obs_dict)

# ============================================================================
# v16 Cornerstone Mask Ogerpon value specialist
#
# This is a value-gated route, not a global "always play Ogerpon" rule.  The
# card stays in hand in unknown matchups, then becomes the primary wall/attacker
# after the Archaludon/Cinderace axis is observed.  Dudunsparce ex is retained
# because it is still the Crustle bypass attacker.
# ============================================================================
C.CORNERSTONE_OGERPON_EX = 117
ATTACK_DEMOLISH = 148

_v16_previous_arch = _d_arch
def _d_arch(self):
    ids = {p.id for p in _d_opp(self)} | {c.id for c in self.opponent.discard}
    if ids & {169, 190, 666}:
        return 'archaludon'
    return _v16_previous_arch(self)

def _v16_has_ogerpon(self):
    return any(p is not None and p.id == C.CORNERSTONE_OGERPON_EX for p in self._my_board())

def _v16_ready_ogerpon(self):
    return any(
        p is not None and p.id == C.CORNERSTONE_OGERPON_EX and len(p.energies) >= 3
        for p in self._my_board()
    )

_v16_previous_play_pokemon = LucarioPolicy._score_play_pokemon
def _v16_play_pokemon(self, card):
    if card.id == C.CORNERSTONE_OGERPON_EX:
        if _v16_has_ogerpon(self):
            return -1
        # Preserve the generic Lucario setup until the opposing engine is known.
        return 26000 if _d_arch(self) == 'archaludon' else -1
    return _v16_previous_play_pokemon(self, card)

_v16_previous_to_hand = LucarioPolicy._score_to_hand
def _v16_to_hand(self, card):
    score = _v16_previous_to_hand(self, card)
    if card.id == C.CORNERSTONE_OGERPON_EX:
        if _d_arch(self) == 'archaludon' and not _v16_has_ogerpon(self):
            return 3200
        return -900
    return score

_v16_previous_setup = LucarioPolicy._score_setup_active
def _v16_setup_active(self, card):
    if card.id == C.CORNERSTONE_OGERPON_EX:
        # Avoid exposing the two-prize wall before the matchup is identified.
        return -20
    return _v16_previous_setup(self, card)

_v16_previous_energy = LucarioPolicy._energy_target_score
def _v16_energy_target(self, pokemon, active):
    score = _v16_previous_energy(self, pokemon, active)
    if pokemon.id == C.CORNERSTONE_OGERPON_EX:
        if _d_arch(self) != 'archaludon':
            return score - 2400
        missing = max(0, 3 - len(pokemon.energies))
        # Charging the wall is valuable until Demolish is online; the fourth
        # energy is pure waste and must not displace a Lucario attachment.
        return score + (5200 + 500 * missing if missing else -2600)
    if _d_arch(self) == 'archaludon' and _v16_ready_ogerpon(self):
        # Once the wall attacks, resume normal Lucario development.
        return score + (200 if pokemon.id in {C.RIOLU, C.RIOLU_70, C.MEGA_LUCARIO_EX} else 0)
    return score

_v16_previous_active = LucarioPolicy._score_active_choice
def _v16_active_choice(self, option, card):
    score = _v16_previous_active(self, option, card)
    if card.id == C.CORNERSTONE_OGERPON_EX and _d_arch(self) == 'archaludon':
        # Archaludon ex and the setup Cinderace both have Abilities, so the
        # Cornerstone Stance converts this slot into the safest prize-race pivot.
        return score + (9000 if len(card.energies) >= 3 else 2600)
    return score

_v16_previous_plan = LucarioPolicy._plan_attack
def _v16_plan_attack(self):
    global plan
    _v16_previous_plan(self)
    if _d_arch(self) != 'archaludon' or not self.opponent.active:
        return
    target = self.opponent.active[0]
    board = self._my_board()
    for index, pokemon in enumerate(board):
        if pokemon is None or pokemon.id != C.CORNERSTONE_OGERPON_EX:
            continue
        if index > 0 and not self.can_switch:
            continue
        if len(pokemon.energies) >= 3:
            # Demolish ignores Metal Defender and every effect on the target.
            plan = AttackPlan(index, 0, 0, target.hp - 140, False)
            return

_v16_previous_option = LucarioPolicy._score_option
def _v16_option(self, option):
    if option.type == OptionType.ATTACK and self.me.active:
        active = self.me.active[0]
        if active.id == C.CORNERSTONE_OGERPON_EX:
            return 1800 if option.attackId == ATTACK_DEMOLISH else -1
    return _v16_previous_option(self, option)

_v16_previous_attach = LucarioPolicy._score_attach
def _v16_attach(self, option):
    card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
    pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
    if card is not None and pokemon is not None and _d_arch(self) == 'archaludon':
        if pokemon.id == C.CORNERSTONE_OGERPON_EX:
            if card.id == C.HERO_CAPE:
                # Damage prevention, rather than raw HP, is the point of this wall.
                return -1
            if card.id in {C.BASIC_FIGHTING_ENERGY, C.ROCK_FIGHTING_ENERGY}:
                return 14500 if len(pokemon.energies) < 3 else -1
    return _v16_previous_attach(self, option)

LucarioPolicy._score_play_pokemon = _v16_play_pokemon
LucarioPolicy._score_to_hand = _v16_to_hand
LucarioPolicy._score_setup_active = _v16_setup_active
LucarioPolicy._energy_target_score = _v16_energy_target
LucarioPolicy._score_active_choice = _v16_active_choice
LucarioPolicy._plan_attack = _v16_plan_attack
LucarioPolicy._score_option = _v16_option
LucarioPolicy._score_attach = _v16_attach

# Aura Jab/Fighting Gong attachment choices must value Ogerpon in the same state
# evaluator.  This patch applies only after Archaludon has been recognized.
_v16_previous_card_choice = LucarioPolicy._score_card_choice
def _v16_card_choice(self, option):
    if (
        _d_arch(self) == 'archaludon'
        and self.context == SelectContext.ATTACH_FROM
    ):
        try:
            pokemon = get_card(self.obs, option.area, option.index, getattr(option, 'playerIndex', self.my_index))
        except Exception:
            pokemon = None
        if pokemon is not None and pokemon.id == C.CORNERSTONE_OGERPON_EX:
            return 28000 - 3500 * len(pokemon.energies)
    return _v16_previous_card_choice(self, option)

LucarioPolicy._score_card_choice = _v16_card_choice

_v16_internal = agent
del agent
def agent(obs_dict):
    return _v16_internal(obs_dict)

# v18: Archaludon-only state-value correction selected by ablation.
# Broad exact-KO and pre-evolution target overrides regressed other matchups, so
# this layer changes no decision unless Archaludon/Cinderace is publicly seen.
_v18_play = LucarioPolicy._score_play_pokemon
def _v18_playf(self, card):
    if _d_arch(self) == 'archaludon' and card.id == C.DUNSPARCE:
        n = sum(1 for p in self._my_board() if p is not None and p.id in {C.DUNSPARCE,C.DUDUNSPARCE,C.DUDUNSPARCE_EX})
        if n >= 1:return -1
    return _v18_play(self, card)

_v18_evolve = LucarioPolicy._score_evolve
def _v18_evolvef(self, option):
    if _d_arch(self) == 'archaludon':
        evolved=get_card(self.obs,option.area,option.index,self.my_index)
        if evolved is not None and evolved.id in {C.DUDUNSPARCE,C.DUDUNSPARCE_EX}:return -1
    return _v18_evolve(self, option)

def _v18_planf(self):
    global plan
    _v16_previous_plan(self)
    if _d_arch(self) != 'archaludon' or not self.opponent.active:return
    target=self.opponent.active[0];base=plan
    exact=base.attacker>=0 and base.target==0 and base.remain_hp<=0
    if base.attacker>=0 and base.target==0:
        ppp=self.hand_counts[C.PREMIUM_POWER_PRO]
        belt=self.hand_counts[C.BLACK_BELT_TRAINING]>0 and (card_table[target.id].ex or card_table[target.id].megaEx)
        exact=exact or base.remain_hp<=30*ppp+(40 if belt else 0)
    for i,pokemon in enumerate(self._my_board()):
        if pokemon is None or pokemon.id!=C.CORNERSTONE_OGERPON_EX or len(pokemon.energies)<3:continue
        if i>0 and not self.can_switch:continue
        if i==0 or not exact:plan=AttackPlan(i,0,0,target.hp-140,False)
        return

_v18_trainer = LucarioPolicy._score_play_trainer
def _v18_trainerf(self, card):
    if _d_arch(self)=='archaludon' and card.id==C.PREMIUM_POWER_PRO and self.me.active and self.me.active[0].id==C.CORNERSTONE_OGERPON_EX:
        if not self.opponent.active:return -1
        remaining=self.opponent.active[0].hp-140-30*globals().get('_m3_ppp',0)
        return 5400 if 0<remaining<=30 else -1
    return _v18_trainer(self, card)

LucarioPolicy._score_play_pokemon=_v18_playf
LucarioPolicy._score_evolve=_v18_evolvef
LucarioPolicy._plan_attack=_v18_planf
LucarioPolicy._score_play_trainer=_v18_trainerf

_v18_internal=agent
del agent
def agent(obs_dict):return _v18_internal(obs_dict)

# v26 Tera-box value route.  Tenacious Tail needs only one Colorless Energy and
# scales with every opposing Pokemon ex, so it is the efficient answer to the
# supplied wide Area Zero board.  No decision changes before that public board
# is identified.
_v26_arch=_d_arch
def _d_arch(self):
 ids={p.id for p in _d_opp(self)}|{c.id for c in self.opponent.discard}
 if ids&{96,108,272,184,1071,230,140,31,756}:return 'terabox'
 return _v26_arch(self)

_v26_energy=LucarioPolicy._energy_target_score
def _v26_energyf(self,pokemon,active):
 s=_v26_energy(self,pokemon,active)
 if _d_arch(self)=='terabox':
  if pokemon.id==C.DUDUNSPARCE_EX:return s+(4200 if len(pokemon.energies)<1 else -1200)
  if pokemon.id in {C.DUNSPARCE,C.DUDUNSPARCE}:return s-900
 return s

_v26_hand=LucarioPolicy._score_to_hand
def _v26_handf(self,card):
 s=_v26_hand(self,card)
 if _d_arch(self)=='terabox':
  if card.id==C.DUDUNSPARCE_EX and self.field_counts[C.DUNSPARCE]>0:s+=2100
  elif card.id==C.DUDUNSPARCE and self.hand_counts[C.DUDUNSPARCE_EX]==0:s-=900
  elif card.id==C.DUNSPARCE and self.field_counts[C.DUNSPARCE]==0:s+=500
 return s

_v26_evolve=LucarioPolicy._score_evolve
def _v26_evolvef(self,option):
 s=_v26_evolve(self,option)
 if _d_arch(self)=='terabox':
  evolved=get_card(self.obs,option.area,option.index,self.my_index)
  if evolved is not None:
   if evolved.id==C.DUDUNSPARCE_EX:return s+5600
   if evolved.id==C.DUDUNSPARCE and not _c1_has_ex(self):return -1
 return s

_v26_attach=LucarioPolicy._score_attach
def _v26_attachf(self,option):
 c=get_card(self.obs,AreaType.HAND,option.index,self.my_index);p=get_card(self.obs,option.inPlayArea,option.inPlayIndex,self.my_index)
 if _d_arch(self)=='terabox' and c is not None and p is not None and p.id==C.DUDUNSPARCE_EX:
  if c.id in {C.BASIC_FIGHTING_ENERGY,C.ROCK_FIGHTING_ENERGY}:return 18000 if len(p.energies)<1 else -1
  if c.id==C.HERO_CAPE:return 17500
 return _v26_attach(self,option)

_v26_active=LucarioPolicy._score_active_choice
def _v26_activef(self,option,card):
 s=_v26_active(self,option,card)
 if _d_arch(self)=='terabox' and card.id==C.DUDUNSPARCE_EX and len(card.energies)>=1:return s+9000
 return s

_v26_plan=LucarioPolicy._plan_attack
def _v26_planf(self):
 global plan
 _v26_plan(self)
 if _d_arch(self)!='terabox' or not self.opponent.active:return
 exn=_d_ex(self);damage=60*exn
 if exn<3:return
 target=self.opponent.active[0]
 for i,pokemon in enumerate(self._my_board()):
  if pokemon is None or pokemon.id!=C.DUDUNSPARCE_EX or len(pokemon.energies)<1:continue
  if i>0 and not self.can_switch:continue
  if damage>=target.hp or plan.attacker<0 or plan.remain_hp>target.hp-damage:
   plan=AttackPlan(i,0,0,target.hp-damage,False)
  return

_v26_trainer=LucarioPolicy._score_play_trainer
def _v26_trainerf(self,card):
 if _d_arch(self)=='terabox' and card.id==C.PREMIUM_POWER_PRO and self.me.active and self.me.active[0].id==C.DUDUNSPARCE_EX:return -1
 return _v26_trainer(self,card)

LucarioPolicy._energy_target_score=_v26_energyf
LucarioPolicy._score_to_hand=_v26_handf
LucarioPolicy._score_evolve=_v26_evolvef
LucarioPolicy._score_attach=_v26_attachf
LucarioPolicy._score_active_choice=_v26_activef
LucarioPolicy._plan_attack=_v26_planf
LucarioPolicy._score_play_trainer=_v26_trainerf
_v26_internal=agent
del agent
def agent(obs_dict):return _v26_internal(obs_dict)

# ============================================================================
# v50 recovered public-information archetype precedence
#
# Card 756 (Mega Kangaskhan ex) appears in Tera-box lists but the older trainer
# router also treated it as sufficient evidence for Crustle.  That split the
# policy: attack/energy logic saw Tera-box while trainer/search logic saw
# Crustle.  Use one precedence rule everywhere.  Actual Dwebble/Crustle cards
# remain authoritative; otherwise stable archetypes are recognized before the
# broad Tera-box signature.  No hidden cards or fixed opponent index are used.
# ============================================================================
_V50_CRUSTLE_IDS = {C.DWEBBLE, C.CRUSTLE}
_V50_DRAGAPULT_IDS = {119, 120, 121}
_V50_ALAKAZAM_IDS = {245, C.ABRA, C.KADABRA, C.ALAKAZAM}
_V50_LUCARIO_IDS = {C.RIOLU_70, C.RIOLU, C.MEGA_LUCARIO_EX, C.MAKUHITA, C.HARIYAMA, C.LUNATONE, C.SOLROCK}
_V50_ARCHALUDON_IDS = {169, 190, 666}
_V50_TERABOX_IDS = {96, 108, 272, 184, 1071, 230, 140, 31, 756}

_v50_previous_d_arch = _d_arch
_v50_previous_router_arch = _router_arch

def _v50_revealed_opponent_ids(self):
    return {p.id for p in _d_opp(self)} | {c.id for c in self.opponent.discard}

def _v50_public_arch(self):
    ids = _v50_revealed_opponent_ids(self)
    if ids & _V50_CRUSTLE_IDS:
        return 'crustle'
    if ids & _V50_DRAGAPULT_IDS:
        return 'dragapult'
    if ids & _V50_ALAKAZAM_IDS:
        return 'alakazam'
    if ids & _V50_LUCARIO_IDS:
        return 'lucario'
    if ids & _V50_ARCHALUDON_IDS:
        return 'archaludon'
    if ids & _V50_TERABOX_IDS:
        return 'terabox'
    return _v50_previous_d_arch(self)

def _d_arch(self):
    return _v50_public_arch(self)

def _router_arch(self):
    arch = _v50_public_arch(self)
    if arch in {'crustle', 'dragapult', 'alakazam', 'lucario', 'archaludon', 'terabox'}:
        return arch
    return _v50_previous_router_arch(self)

# v53 promoted route: the replay-supported evolution-before-Lunatone ordering
# is enabled only in the four revealed matchups where held-out behavior improved.
# Lucario mirror, Crustle, Archaludon and unknown states retain recovered v50.
_V53_EVOLVE_FIRST_ARCHES = {'terabox', 'dragapult', 'alakazam', 'grimmsnarl'}
_v53_previous_choose = LucarioPolicy.choose

def _v53_choose(self):
    base = _v53_previous_choose(self)
    if self.context != SelectContext.MAIN or not base:
        return base
    if _router_arch(self) not in _V53_EVOLVE_FIRST_ARCHES:
        return base
    first = self.select.option[base[0]]
    if first.type != OptionType.ABILITY:
        return base
    ability_card = get_card(self.obs, first.area, first.index, self.my_index)
    if ability_card is None or ability_card.id != C.LUNATONE:
        return base
    candidates = []
    for index, option in enumerate(self.select.option):
        if option.type != OptionType.EVOLVE:
            continue
        evolved = get_card(self.obs, option.area, option.index, self.my_index)
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if (
            evolved is not None
            and evolved.id == C.MEGA_LUCARIO_EX
            and target is not None
            and target.id in {C.RIOLU, C.RIOLU_70}
            and not target.appearThisTurn
        ):
            candidates.append((self._score_evolve(option), -index, index))
    if candidates:
        return [max(candidates)[2]]
    return base

LucarioPolicy.choose = _v53_choose

# v56 candidate: after correcting prize perspective, immediately take a
# currently powered, zero-resource KO that removes all of our remaining prizes.
_v56_previous_choose = LucarioPolicy.choose

def _v56_choose(self):
    if self.context == SelectContext.MAIN and self.me.active and self.opponent.active:
        active = self.me.active[0]
        target = self.opponent.active[0]
        target_prizes = prize_count(target)
        if target_prizes > 0 and len(self.me.prize) <= target_prizes:
            exact = _ExactTurnTactics(self).active_exact_ko()
            if exact is not None:
                _, attack_id, need_ppp, need_black_belt = exact
                energy_needed = 1 if attack_id == _AURA_JAB else 2
                if need_ppp == 0 and need_black_belt == 0 and len(active.energies) >= energy_needed:
                    for index, option in enumerate(self.select.option):
                        if option.type == OptionType.ATTACK and option.attackId == attack_id:
                            return [index]
    return _v56_previous_choose(self)

LucarioPolicy.choose = _v56_choose

# v72 candidate: all eleven uploaded Alakazam Aura Jab allocation decisions
# selected Hariyama, while v56 selected another target every time.  Keep this
# correction strictly matchup/context gated.
_v72_previous_card_choice = LucarioPolicy._score_card_choice

def _v72_card_choice(self, option):
    effect = getattr(self.select, 'effect', None)
    if (
        self.context == SelectContext.ATTACH_FROM
        and _d_arch(self) == 'alakazam'
        and getattr(effect, 'id', -1) == C.MEGA_LUCARIO_EX
    ):
        try:
            target = get_card(self.obs, option.area, option.index, getattr(option, 'playerIndex', self.my_index))
        except Exception:
            target = None
        if target is not None and target.id == C.HARIYAMA:
            return 30000 - 2500 * len(target.energies)
    return _v72_previous_card_choice(self, option)

LucarioPolicy._score_card_choice = _v72_card_choice

# v79 candidate: Dragapult Aura Jab allocation by concrete next-attacker
# thresholds.  Build Lucario lines to two, then Hariyama/Makuhita, then a
# one-energy Solrock.  Fully prepared targets are explicitly capped.
_v79_previous_card_choice = LucarioPolicy._score_card_choice

def _v79_card_choice(self, option):
    effect = getattr(self.select, 'effect', None)
    if (
        self.context == SelectContext.ATTACH_FROM
        and _d_arch(self) == 'dragapult'
        and getattr(effect, 'id', -1) == C.MEGA_LUCARIO_EX
    ):
        try:
            target = get_card(self.obs, option.area, option.index, getattr(option, 'playerIndex', self.my_index))
        except Exception:
            target = None
        if target is not None:
            energy = len(target.energies)
            if target.id in {C.MEGA_LUCARIO_EX, C.RIOLU, C.RIOLU_70}:
                return 32000 - 3000 * energy if energy < 2 else -20000
            if target.id in {C.HARIYAMA, C.MAKUHITA}:
                return 22000 - 1500 * energy if energy < 3 else -20000
            if target.id == C.SOLROCK:
                return 18000 if energy < 1 else -20000
            return -20000
    return _v79_previous_card_choice(self, option)

LucarioPolicy._score_card_choice = _v79_card_choice

# v107a replay-912 candidate: public Active Crustle -> Cornerstone route.
# The gate is deliberately narrower than the rejected broad Crustle/Ogerpon
# experiment: Dwebble in the deck/Bench is not enough.  The route activates only
# when Crustle itself is the publicly visible Active Pokemon, whose Ability both
# blocks Pokemon-ex damage and makes its attack damage preventable by
# Cornerstone Stance.  Demolish ignores the blocking effect.
_V107_OGERPON_ACTIVE_IDS = {C.CRUSTLE}

def _v107_oger_window(self):
    return bool(
        self.opponent.active
        and self.opponent.active[0] is not None
        and self.opponent.active[0].id in _V107_OGERPON_ACTIVE_IDS
    )

_v107_prev_play_pokemon = LucarioPolicy._score_play_pokemon
def _v107_play_pokemon(self, card):
    if card.id == C.CORNERSTONE_OGERPON_EX and _v107_oger_window(self):
        if _v16_has_ogerpon(self) or len(self.me.bench) >= self.me.benchMax:
            return -1
        # Replay 91211615, turn 3: with a four-card hand, deploying Ogerpon
        # before Lillie's Determination discarded the only large early draw.
        # Draw first in the opening turns; the public Crustle gate will search
        # and deploy Ogerpon again after the hand is rebuilt.
        if (
            self.state.turn <= 5
            and not self.state.supporterPlayed
            and self.hand_counts[C.LILLIE_DETERMINATION] > 0
            and self.me.deckCount > LOW_DECK_COUNT
        ):
            return 2500
        return 44000
    return _v107_prev_play_pokemon(self, card)

_v107_prev_to_hand = LucarioPolicy._score_to_hand
def _v107_to_hand(self, card):
    if card.id == C.CORNERSTONE_OGERPON_EX and _v107_oger_window(self):
        energy_in_hand = self.hand_counts[C.BASIC_FIGHTING_ENERGY] + self.hand_counts[C.ROCK_FIGHTING_ENERGY]
        energy_is_selectable = False
        for option in self.select.option:
            try:
                selectable = get_card(self.obs, option.area, option.index, getattr(option, 'playerIndex', self.my_index))
            except Exception:
                selectable = None
            if selectable is not None and selectable.id in {C.BASIC_FIGHTING_ENERGY, C.ROCK_FIGHTING_ENERGY}:
                energy_is_selectable = True
                break
        if not self.state.energyAttached and energy_in_hand == 0 and energy_is_selectable:
            return _v107_prev_to_hand(self, card)
        return 42000 if not _v16_has_ogerpon(self) else -1
    return _v107_prev_to_hand(self, card)

_v107_prev_energy = LucarioPolicy._energy_target_score
def _v107_energy_target(self, pokemon, active):
    score = _v107_prev_energy(self, pokemon, active)
    if pokemon.id == C.CORNERSTONE_OGERPON_EX and _v107_oger_window(self):
        missing = max(0, 3 - len(pokemon.energies))
        return score + (15000 + 1200 * missing if missing else -6000)
    return score

_v107_prev_attach = LucarioPolicy._score_attach
def _v107_attach(self, option):
    card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
    pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
    if card is not None and pokemon is not None and _v107_oger_window(self) and pokemon.id == C.CORNERSTONE_OGERPON_EX:
        if card.id == C.HERO_CAPE:
            return -1
        if card.id in {C.BASIC_FIGHTING_ENERGY, C.ROCK_FIGHTING_ENERGY}:
            return 46000 if len(pokemon.energies) < 3 else -1
    return _v107_prev_attach(self, option)

_v107_prev_card_choice = LucarioPolicy._score_card_choice
def _v107_card_choice(self, option):
    if _v107_oger_window(self) and self.context == SelectContext.ATTACH_FROM:
        try:
            pokemon = get_card(self.obs, option.area, option.index, getattr(option, 'playerIndex', self.my_index))
        except Exception:
            pokemon = None
        if pokemon is not None and pokemon.id == C.CORNERSTONE_OGERPON_EX:
            return 60000 - 5000 * len(pokemon.energies) if len(pokemon.energies) < 3 else -30000
    return _v107_prev_card_choice(self, option)

_v107_prev_active = LucarioPolicy._score_active_choice
def _v107_active(self, option, card):
    score = _v107_prev_active(self, option, card)
    if card.id == C.CORNERSTONE_OGERPON_EX and _v107_oger_window(self):
        return score + (22000 if len(card.energies) >= 3 else 7000)
    return score

_v107_prev_plan = LucarioPolicy._plan_attack
def _v107_plan(self):
    global plan
    _v107_prev_plan(self)
    if not _v107_oger_window(self):
        return
    target = self.opponent.active[0]
    for index, pokemon in enumerate(self._my_board()):
        if pokemon is None or pokemon.id != C.CORNERSTONE_OGERPON_EX or len(pokemon.energies) < 3:
            continue
        if index > 0 and not self.can_switch:
            continue
        plan = AttackPlan(index, 0, 0, target.hp - 140, False)
        return

_v107_prev_choose = LucarioPolicy.choose
def _v107_choose(self):
    if self.context == SelectContext.MAIN and _v107_oger_window(self) and self.me.active:
        active = self.me.active[0]
        target = self.opponent.active[0]
        # Once the protected attacker is online, cash exact Demolish KOs before
        # exposing another action window.
        if active.id == C.CORNERSTONE_OGERPON_EX and len(active.energies) >= 3 and target.hp <= 140:
            for index, option in enumerate(self.select.option):
                if option.type == OptionType.ATTACK and option.attackId == ATTACK_DEMOLISH:
                    return [index]
    return _v107_prev_choose(self)

LucarioPolicy._score_play_pokemon = _v107_play_pokemon
LucarioPolicy._score_to_hand = _v107_to_hand
LucarioPolicy._energy_target_score = _v107_energy_target
LucarioPolicy._score_attach = _v107_attach
LucarioPolicy._score_card_choice = _v107_card_choice
LucarioPolicy._score_active_choice = _v107_active
LucarioPolicy._plan_attack = _v107_plan
LucarioPolicy.choose = _v107_choose

# Submission identity; no behavior depends on this label.
POLICY_RELEASE = "v109_crustle_ogerpon_draw_guarded"

# v110c resource-commitment hybrid.  Prefer Dudunsparce ex at equal/early
# investment, but finish an already two-energy Cornerstone line unless a ready
# three-energy Dudunsparce ex can attack immediately.
def _v110c_find(self, card_id):
    return next((p for p in self._my_board() if p is not None and p.id == card_id), None)

def _v110c_prefer_oger(self):
    oger = _v110c_find(self, C.CORNERSTONE_OGERPON_EX)
    dudun = _v110c_find(self, C.DUDUNSPARCE_EX)
    return bool(oger is not None and len(oger.energies) >= 2 and (dudun is None or len(dudun.energies) < 3))

_v110c_previous_play = LucarioPolicy._score_play_pokemon
def _v110c_play(self, card):
    if _v107_oger_window(self) and card.id == C.CORNERSTONE_OGERPON_EX:
        if _v110c_find(self, C.DUDUNSPARCE_EX) is not None or self.field_counts[C.DUNSPARCE] > 0:
            return -1
    return _v110c_previous_play(self, card)

_v110c_previous_hand = LucarioPolicy._score_to_hand
def _v110c_hand(self, card):
    if _v107_oger_window(self):
        if card.id == C.DUDUNSPARCE_EX and self.field_counts[C.DUNSPARCE] > 0:
            return 62000
        if card.id == C.CORNERSTONE_OGERPON_EX and (
            _v110c_find(self, C.DUDUNSPARCE_EX) is not None or self.field_counts[C.DUNSPARCE] > 0
        ):
            return 4000
    return _v110c_previous_hand(self, card)

_v110c_previous_energy = LucarioPolicy._energy_target_score
def _v110c_energy(self, pokemon, active):
    score = _v110c_previous_energy(self, pokemon, active)
    if not _v107_oger_window(self):
        return score
    if _v110c_prefer_oger(self):
        if pokemon.id == C.CORNERSTONE_OGERPON_EX:
            return 76000 - 7000 * len(pokemon.energies) if len(pokemon.energies) < 3 else -30000
        if pokemon.id == C.DUDUNSPARCE_EX:
            return score - 16000
    elif _v110c_find(self, C.DUDUNSPARCE_EX) is not None:
        if pokemon.id == C.DUDUNSPARCE_EX:
            return 74000 - 7000 * len(pokemon.energies) if len(pokemon.energies) < 3 else -30000
        if pokemon.id == C.CORNERSTONE_OGERPON_EX:
            return -30000
    return score

_v110c_previous_choice = LucarioPolicy._score_card_choice
def _v110c_choice(self, option):
    if _v107_oger_window(self) and self.context == SelectContext.ATTACH_FROM:
        try:
            pokemon = get_card(self.obs, option.area, option.index, getattr(option, 'playerIndex', self.my_index))
        except Exception:
            pokemon = None
        if pokemon is not None:
            if _v110c_prefer_oger(self) and pokemon.id == C.CORNERSTONE_OGERPON_EX and len(pokemon.energies) < 3:
                return 78000 - 6000 * len(pokemon.energies)
            if not _v110c_prefer_oger(self) and pokemon.id == C.DUDUNSPARCE_EX and len(pokemon.energies) < 3:
                return 78000 - 6000 * len(pokemon.energies)
    return _v110c_previous_choice(self, option)

_v110c_previous_active = LucarioPolicy._score_active_choice
def _v110c_active(self, option, card):
    score = _v110c_previous_active(self, option, card)
    if _v107_oger_window(self):
        if card.id == C.DUDUNSPARCE_EX and len(card.energies) >= 3 and not _v110c_prefer_oger(self):
            return score + 52000
        if card.id == C.CORNERSTONE_OGERPON_EX and len(card.energies) >= 3 and _v110c_prefer_oger(self):
            return score + 52000
    return score

_v110c_previous_plan = LucarioPolicy._plan_attack
def _v110c_plan(self):
    global plan
    _v110c_previous_plan(self)
    if not _v107_oger_window(self) or _v110c_prefer_oger(self):
        return
    target = self.opponent.active[0]
    dudun = _v110c_find(self, C.DUDUNSPARCE_EX)
    if dudun is None or len(dudun.energies) < 3:
        return
    for index, pokemon in enumerate(self._my_board()):
        if pokemon is dudun and (index == 0 or self.can_switch):
            plan = AttackPlan(index, 0, 1, target.hp - 150, False)
            return

LucarioPolicy._score_play_pokemon = _v110c_play
LucarioPolicy._score_to_hand = _v110c_hand
LucarioPolicy._energy_target_score = _v110c_energy
LucarioPolicy._score_card_choice = _v110c_choice
LucarioPolicy._score_active_choice = _v110c_active
LucarioPolicy._plan_attack = _v110c_plan
POLICY_RELEASE = "v110_crustle_commitment_hybrid_champion"


# === v111 cross-league Marnie evolution-line denial ========================
# Public Marnie's Impidimp/Morgrem are low-HP, one-Prize bridge pieces.  The
# baseline often allowed them to mature into a 320-HP Grimmsnarl ex, turning a
# cheap Aura Jab/Boss conversion into a two-hit prize race.  This local target
# residual applies only when those exact public card IDs are offered.
_v111_previous_target_score = target_score

def target_score(pokemon):
    score = _v111_previous_target_score(pokemon)
    if pokemon is None:
        return score
    if pokemon.id == 646:       # Marnie's Impidimp
        score += 1050
    elif pokemon.id == 647:     # Marnie's Morgrem
        score += 1250
    elif pokemon.id == 648:     # Marnie's Grimmsnarl ex
        score += 250
    return score

POLICY_RELEASE = "v111_marnie_line_denial_cross_league_final"

# === v115 exact Crustle attack replacement ================================
# Preserve the complete retained turn sequence.  Only after the retained policy
# itself selects Mega Brave into a public Active Crustle do we replace that
# zero-damage attack with Aura Jab, whose damage is also blocked but whose
# Fighting-Energy acceleration remains live.
_v115_previous_choose = LucarioPolicy.choose

def _v115_choose(self):
    base = _v115_previous_choose(self)
    if not (self.context == SelectContext.MAIN and base and len(base) == 1
            and self.me.active and self.opponent.active
            and self.me.active[0].id == C.MEGA_LUCARIO_EX
            and self.opponent.active[0].id == C.CRUSTLE):
        return base
    try:
        chosen = self.select.option[base[0]]
    except Exception:
        return base
    if chosen.type != OptionType.ATTACK or chosen.attackId != _MEGA_BRAVE:
        return base
    for index, option in enumerate(self.select.option):
        if option.type == OptionType.ATTACK and option.attackId == _AURA_JAB:
            return [index]
    return base

LucarioPolicy.choose = _v115_choose
POLICY_RELEASE = "v115_crustle_aura_replace_only"
