from __future__ import annotations

import os
import time
from collections import defaultdict, Counter
from enum import IntEnum

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    LogType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    all_attack,
    search_begin,
    search_end,
    search_step,
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
    ULTRA_BALL = 1121
    MEGA_SIGNAL = 1145
    BUDDY_BUDDY_POFFIN = 1086
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_cg_api.__file__)))
DECK_PATH = os.path.join(BASE_DIR, "deck.csv")
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}
attack_table = {attack.attackId: attack for attack in all_attack()}


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
                    if len(self.opponent.prize) <= prize:
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
        if self.context == SelectContext.DISCARD:
            return self._score_discard_choice(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_discard_choice(self, card: Pokemon | Card) -> float:
        cid = card.id
        score = 0.0
        draw_count = self.hand_counts[C.CARMINE] + self.hand_counts[C.LILLIE_DETERMINATION] + self.hand_counts[C.JUDGE]
        if cid == C.BASIC_FIGHTING_ENERGY:
            # One discarded Energy enables Aura Jab; protect the last attachable copy.
            if self.hand_counts[cid] >= 3:
                score += 500
            elif self.hand_counts[cid] >= 2 and self.discard_counts[cid] == 0:
                score += 350
            else:
                score -= 650
        elif cid == C.JUDGE:
            score += -850 if self._opponent_is_psychic_engine() else 600
        elif cid in {C.CARMINE, C.LILLIE_DETERMINATION}:
            score += 460 if draw_count >= 2 else -300
        elif cid == C.BOSS_ORDERS:
            score += 260 if self.hand_counts[cid] >= 2 and self.my_prizes_left > 2 else -600
        elif cid == C.SWITCH:
            score += 220 if self.hand_counts[cid] >= 2 else -700
        elif cid == C.HERO_CAPE:
            score -= 1800
        elif cid == C.PREMIUM_POWER_PRO:
            already = game_memory.played_this_turn[C.PREMIUM_POWER_PRO]
            needed = max(0, (max(0, plan.remain_hp - 30 * already) + 29) // 30)
            score += 280 if self.hand_counts[cid] > needed + 1 else 80 if self.hand_counts[cid] > needed else -900
        elif cid == C.GRAVITY_MOUNTAIN:
            opponent_stage2 = any(p is not None and card_table[p.id].stage2 for p in self._opponent_board())
            score += -900 if opponent_stage2 or self.stadium_id else 180
        elif cid in {C.DUSK_BALL, C.ULTRA_BALL, C.MEGA_SIGNAL, C.BUDDY_BUDDY_POFFIN, C.FIGHTING_GONG, C.POKE_PAD}:
            score += 320 if self.hand_counts[cid] >= 2 else 60
        elif cid == C.MEGA_LUCARIO_EX:
            unevolved = sum(1 for p in self._my_board() if p is not None and p.id == C.RIOLU)
            score += 320 if unevolved == 0 or self.hand_counts[cid] >= 2 else -1300
        elif cid == C.RIOLU:
            lines = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += 350 if lines >= 2 or self.hand_counts[cid] >= 2 else -800
        elif cid == C.HARIYAMA:
            score += -1500 if self._should_preserve_hariyama() else 300 if self.field_counts[C.MAKUHITA] == 0 or self.hand_counts[cid] >= 2 else -650
        elif cid == C.MAKUHITA:
            score += 300 if self.field_counts[cid] >= 1 or self.hand_counts[cid] >= 2 else -450
        elif cid in {C.LUNATONE, C.SOLROCK}:
            score += 380 if self.field_counts[cid] >= 1 or self.hand_counts[cid] >= 2 else -220
        else:
            score += 100
        return score

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
        if card.id == C.MEGA_SIGNAL:
            known = self.field_counts[C.MEGA_LUCARIO_EX] + self.hand_counts[C.MEGA_LUCARIO_EX] + self.discard_counts[C.MEGA_LUCARIO_EX]
            unevolved = sum(1 for p in self._my_board() if p is not None and p.id == C.RIOLU)
            if known >= 4 or self.hand_counts[C.MEGA_LUCARIO_EX] >= max(1, unevolved):
                return -1
            return 10150 if unevolved > 0 else 8900
        if card.id == C.ULTRA_BALL:
            if self.me.handCount < 3:
                return -1
            unevolved = sum(1 for p in self._my_board() if p is not None and p.id == C.RIOLU)
            need_lucario = unevolved > self.hand_counts[C.MEGA_LUCARIO_EX]
            missing_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] < 2
            if need_lucario:
                return 10200
            if missing_line and len(self.me.bench) < self.me.benchMax:
                return 9400
            return 7200 if self.phase in {Phase.OPENING, Phase.BUILD} and self.me.handCount >= 6 else -1
        if card.id == C.BUDDY_BUDDY_POFFIN:
            if len(self.me.bench) >= self.me.benchMax:
                return -1
            missing_riolu = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] < 2
            missing_makuhita = self.field_counts[C.MAKUHITA] + self.field_counts[C.HARIYAMA] < 1
            return 10400 if self.state.turn <= 2 and (missing_riolu or missing_makuhita) else 8700 if missing_riolu else -1
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



class Phase(IntEnum):
    OPENING = 0
    BUILD = 1
    ACCELERATE = 2
    PRESSURE = 3
    DEFEND = 4
    ENDGAME = 5


# Exact recipes are used only to build hidden-card determinizations for the search
# engine. The public-board policy remains the original v63 policy.
KNOWN_RECIPES = {
    "v63": (673, 673, 674, 674, 675, 675, 676, 676, 676, 677, 677, 677, 677, 678, 678, 678, 678, 1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142, 1152, 1159, 1182, 1182, 1182, 1192, 1192, 1192, 1192, 1227, 1227, 1227, 1227, 1213, 1252, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    "dragapult": (119,119,119,119,120,120,120,120,121,121,121,131,131,132,133,112,140,235,235,1071,31,1227,1227,1227,1227,1182,1182,1182,1198,1198,1240,1231,1086,1086,1086,1152,1152,1152,1152,1121,1121,1121,1121,1079,1079,1097,1097,1080,1256,1256,1246,2,2,2,5,5,5,5,7,7),
    "alakazam": (741,741,741,741,742,742,742,742,743,743,743,305,305,305,66,66,140,142,858,343,1152,1152,1152,1152,1086,1086,1086,1086,1079,1079,1079,1097,1129,1156,1156,1156,1081,1081,1081,1182,1182,1231,1231,1231,1231,1225,1225,1225,1225,1264,1264,1264,1264,5,5,19,19,19,19,13),
    "crustle": (344,344,344,344,345,345,345,345,1147,1147,1147,1147,1159,1264,1264,1264,1264,1212,1212,1212,1212,1224,1224,1224,1224,18,18,18,18,11,11,11,11,1086,1086,1086,1086,14,14,14,14,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1),
    "rocket": (463,463,463,463,891,891,891,473,473,474,414,414,1216,1216,1216,1216,1217,1217,1217,1217,1218,1218,1218,1218,1220,1220,1220,1220,1219,1219,1219,1219,1152,1152,1152,1152,1134,1134,1134,1134,1077,1077,1077,1077,1097,1097,1097,1121,1109,1257,1257,1257,15,15,15,15,17,17,17,17),
    "hop": (11,11,11,11,12,19,19,19,19,65,65,65,65,66,66,66,304,304,878,878,878,878,879,879,1086,1086,1086,1086,1097,1097,1097,1115,1115,1115,1122,1122,1122,1122,1152,1152,1152,1152,1171,1171,1171,1171,1182,1182,1194,1194,1210,1210,1227,1227,1227,1227,1255,1255,1255,1255),
}

ARCHETYPE_MARKERS = {
    "v63": {673,674,675,676,677,678,1102,1141,1142},
    "dragapult": {119,120,121,131,132,133,235},
    "alakazam": {741,742,743,305,66,858},
    "crustle": {344,345,1147,1212,1224},
    "rocket": {463,891,473,474,414,1216,1217,1218,1220},
    "hop": {878,879,304,1115,1171,1255},
}

SHUFFLE_SUPPORTERS = {C.CARMINE, C.LILLIE_DETERMINATION, C.JUDGE}
SEARCH_ITEMS = {C.DUSK_BALL, C.ULTRA_BALL, C.MEGA_SIGNAL, C.BUDDY_BUDDY_POFFIN, C.FIGHTING_GONG, C.POKE_PAD}


class GameMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = -1
        self.my_index = None
        self.known_opponent_hand = Counter()
        self.seen_opponent = Counter()
        self.searched_this_turn = False
        self.played_this_turn = Counter()
        self.deep_search_used = False

    def update(self, obs: Observation):
        if obs.current is None:
            return
        if self.my_index is None:
            self.my_index = obs.current.yourIndex
        if obs.current.turn < self.turn or obs.current.turn == 0 and self.turn > 0:
            self.reset()
            self.my_index = obs.current.yourIndex
        if obs.current.turn != self.turn:
            self.turn = obs.current.turn
            self.searched_this_turn = False
            self.played_this_turn.clear()
            self.deep_search_used = False
        my_i = self.my_index
        op_i = 1 - my_i
        for log in obs.logs:
            if log.playerIndex == op_i and log.cardId:
                self.seen_opponent[log.cardId] += 1
            if log.type == LogType.MOVE_CARD and log.cardId:
                if log.playerIndex == op_i:
                    if log.toArea == AreaType.HAND and log.fromArea != AreaType.HAND:
                        self.known_opponent_hand[log.cardId] += 1
                    if log.fromArea == AreaType.HAND and self.known_opponent_hand[log.cardId] > 0:
                        self.known_opponent_hand[log.cardId] -= 1
                elif log.playerIndex == my_i and log.fromArea == AreaType.HAND:
                    pass
            elif log.type == LogType.PLAY and log.cardId:
                if log.playerIndex == op_i:
                    if self.known_opponent_hand[log.cardId] > 0:
                        self.known_opponent_hand[log.cardId] -= 1
                    if log.cardId in SHUFFLE_SUPPORTERS:
                        self.known_opponent_hand.clear()
                elif log.playerIndex == my_i:
                    self.played_this_turn[log.cardId] += 1
                    if log.cardId in SEARCH_ITEMS:
                        self.searched_this_turn = True
            elif log.type == LogType.ATTACH and log.cardId and log.playerIndex == op_i:
                if self.known_opponent_hand[log.cardId] > 0:
                    self.known_opponent_hand[log.cardId] -= 1


game_memory = GameMemory()


def _board_cards(player) -> list[int]:
    ids = []
    for pokemon in player.active + player.bench:
        if pokemon is None:
            continue
        ids.append(pokemon.id)
        ids.extend(card.id for card in pokemon.preEvolution)
        ids.extend(card.id for card in pokemon.energyCards)
        ids.extend(card.id for card in pokemon.tools)
    return ids


def _card_utility(card_id: int, player, threat: bool = True) -> float:
    data = card_table.get(card_id)
    if data is None:
        return 0
    score = 0.0
    if data.cardType == CardType.BASIC_ENERGY or data.cardType == CardType.SPECIAL_ENERGY:
        score = 75
        if any(p is not None and len(p.energies) < len(card_table[p.id].attacks and attack_table[card_table[p.id].attacks[0]].energies or []) for p in player.active + player.bench):
            score += 45
    elif data.cardType == CardType.POKEMON:
        score = 80
        if data.stage1 or data.stage2:
            score += 80
        if data.ex or data.megaEx:
            score += 45
        if any(p is not None and data.evolvesFrom == card_table[p.id].name for p in player.active + player.bench):
            score += 160
    elif data.cardType == CardType.SUPPORTER:
        score = 135
        if card_id == C.BOSS_ORDERS:
            score += 80
        elif card_id == C.JUDGE:
            score += 25
    elif data.cardType == CardType.ITEM:
        score = 105
        if card_id in {C.DUSK_BALL, C.ULTRA_BALL, C.MEGA_SIGNAL, C.BUDDY_BUDDY_POFFIN, C.FIGHTING_GONG, C.POKE_PAD, 1079}:
            score += 55
        if card_id in {C.PREMIUM_POWER_PRO, 1120, 1081}:
            score += 35
    elif data.cardType == CardType.TOOL:
        score = 100
    elif data.cardType == CardType.STADIUM:
        score = 95
    return score if threat else -score


class OpponentBelief:
    def __init__(self, obs: Observation, memory: GameMemory):
        self.obs = obs
        self.state = obs.current
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.op = self.state.players[self.op_index]
        public = Counter(_board_cards(self.op))
        public.update(card.id for card in self.op.discard)
        if self.state.stadium and self.state.stadium[0].playerIndex == self.op_index:
            public[self.state.stadium[0].id] += 1
        public.update(memory.seen_opponent)
        self.public = public
        self.archetype = self._detect_archetype()
        self.recipe = list(KNOWN_RECIPES.get(self.archetype, ()))
        self.known_hand = Counter(memory.known_opponent_hand)

    def _detect_archetype(self) -> str | None:
        visible = set(self.public)
        best_name = None
        best = 0
        for name, markers in ARCHETYPE_MARKERS.items():
            score = len(visible & markers)
            if score > best:
                best_name, best = name, score
        return best_name if best >= 1 else None

    def _generic_energy(self) -> int:
        for p in self.op.active + self.op.bench:
            if p is not None:
                et = card_table[p.id].energyType
                if et in {EnergyType.FIRE, EnergyType.PSYCHIC, EnergyType.FIGHTING, EnergyType.DARKNESS, EnergyType.GRASS, EnergyType.WATER, EnergyType.LIGHTNING, EnergyType.METAL}:
                    return int(et)
        return C.BASIC_FIGHTING_ENERGY

    @staticmethod
    def _remove(counter: Counter, card_id: int, count: int = 1):
        if count <= 0:
            return
        counter[card_id] -= count
        if counter[card_id] <= 0:
            counter.pop(card_id, None)

    def scenarios(self) -> list[tuple[list[int], list[int], list[int]]]:
        if not self.recipe:
            filler = self._generic_energy()
            return [([filler] * self.op.deckCount, [filler] * len(self.op.prize), [filler] * self.op.handCount)]
        remaining = Counter(self.recipe)
        # Subtract currently public cards once. seen_opponent may include repeats already in discard/board,
        # so cap by recipe rather than driving counts negative.
        for card_id, count in self.public.items():
            self._remove(remaining, card_id, min(count, remaining.get(card_id, 0)))
        known = []
        for card_id, count in self.known_hand.items():
            take = min(count, remaining.get(card_id, 0), self.op.handCount - len(known))
            known.extend([card_id] * take)
            self._remove(remaining, card_id, take)
        result = []
        for style in (0, 1):
            pool = list(remaining.elements())
            if style == 0:
                pool.sort(key=lambda cid: _card_utility(cid, self.op, True), reverse=True)
            else:
                # Balanced scenario alternates high-impact and low-impact cards instead of assuming nuts.
                pool.sort(key=lambda cid: (_card_utility(cid, self.op, True), cid), reverse=True)
                mixed = []
                lo, hi = len(pool) - 1, 0
                while hi <= lo:
                    mixed.append(pool[hi]); hi += 1
                    if hi <= lo:
                        mixed.append(pool[lo]); lo -= 1
                pool = mixed
            hand = known + pool[: max(0, self.op.handCount - len(known))]
            rest = pool[max(0, self.op.handCount - len(known)):]
            # Keep strong cards in the deck in the balanced scenario; put low-impact cards in prizes.
            rest.sort(key=lambda cid: _card_utility(cid, self.op, True), reverse=(style == 1))
            prize_n = len(self.op.prize)
            prize = rest[-prize_n:] if style == 0 and prize_n else rest[:prize_n]
            if style == 0 and prize_n:
                deck = rest[:-prize_n]
            else:
                deck = rest[prize_n:]
            filler = self._generic_energy()
            hand = (hand + [filler] * self.op.handCount)[:self.op.handCount]
            prize = (prize + [filler] * prize_n)[:prize_n]
            deck = (deck + [filler] * self.op.deckCount)[:self.op.deckCount]
            result.append((deck, prize, hand))
        return result


def _our_hidden(obs: Observation) -> tuple[list[int], list[int]]:
    state = obs.current
    me = state.players[state.yourIndex]
    remaining = Counter(my_deck)
    visible = []
    if me.hand:
        visible.extend(card.id for card in me.hand)
    visible.extend(_board_cards(me))
    visible.extend(card.id for card in me.discard)
    if state.stadium and state.stadium[0].playerIndex == state.yourIndex:
        visible.append(state.stadium[0].id)
    prize = []
    for card in me.prize:
        if card is not None:
            prize.append(card.id)
            visible.append(card.id)
    for card_id in visible:
        if remaining.get(card_id, 0) > 0:
            remaining[card_id] -= 1
            if remaining[card_id] <= 0:
                remaining.pop(card_id, None)
    filler = C.BASIC_FIGHTING_ENERGY
    prize = (prize + [filler] * len(me.prize))[:len(me.prize)]
    deck = list(remaining.elements())
    deck = (deck + [filler] * me.deckCount)[:me.deckCount]
    return deck, prize


def _is_rule_box(pokemon: Pokemon) -> bool:
    data = card_table[pokemon.id]
    return bool(data.ex or data.megaEx)


def _generic_attack_damage(attacker: Pokemon, target: Pokemon | None, hand_count: int, stadium_id: int = 0) -> int:
    best = 0
    for attack_id in card_table[attacker.id].attacks:
        attack = attack_table.get(attack_id)
        if attack is None or len(attack.energies) > len(attacker.energies):
            continue
        damage = attack.damage
        counter_effect = False
        if attacker.id == C.ALAKAZAM and attack_id == 1072:
            damage = hand_count * 20
            counter_effect = True
        elif attacker.id == 121 and attack_id == 154:
            damage = 200
        if target is not None:
            if attacker.id == C.MEGA_LUCARIO_EX and target.id == C.CRUSTLE:
                damage = 0
            if stadium_id == 1247 and _is_rule_box(attacker) and not _is_rule_box(target):
                damage = 0
            if not counter_effect and damage > 0:
                adata, tdata = card_table[attacker.id], card_table[target.id]
                if "isn’t affected by Weakness" not in attack.text and tdata.weakness == adata.energyType:
                    damage *= 2
                elif "isn’t affected by" not in attack.text and tdata.resistance == adata.energyType:
                    damage = max(0, damage - 30)
        best = max(best, damage)
    return best


class SmartLucarioPolicy(LucarioPolicy):
    def __init__(self, obs: Observation):
        super().__init__(obs)
        self.phase = self._phase()

    def _incoming(self) -> int:
        if not self.me.active:
            return 0
        target = self.me.active[0]
        return max((_generic_attack_damage(p, target, self.opponent.handCount, self.stadium_id) for p in self.opponent.active + self.opponent.bench if p is not None), default=0)

    def _phase(self) -> Phase:
        if self.state.turn <= 2:
            return Phase.OPENING
        if self.my_prizes_left <= 2 or len(self.opponent.prize) <= 2:
            return Phase.ENDGAME
        ready_lucario = any(p is not None and p.id == C.MEGA_LUCARIO_EX and len(p.energies) >= 1 for p in self._my_board())
        ready_hariyama = any(p is not None and p.id == C.HARIYAMA and len(p.energies) >= 3 for p in self._my_board())
        if self.me.active and self._incoming() >= self.me.active[0].hp and (ready_lucario or ready_hariyama):
            return Phase.DEFEND
        if self.me.active and self.me.active[0].id == C.MEGA_LUCARIO_EX and len(self.me.active[0].energies) >= 1:
            bench_needs = any(p is not None and p is not self.me.active[0] and p.id in {C.RIOLU,C.MEGA_LUCARIO_EX,C.MAKUHITA,C.HARIYAMA} and len(p.energies) < (2 if p.id in {C.RIOLU,C.MEGA_LUCARIO_EX} else 3) for p in self._my_board())
            if bench_needs and self.discard_counts[C.BASIC_FIGHTING_ENERGY] > 0:
                return Phase.ACCELERATE
        if ready_lucario or ready_hariyama:
            return Phase.PRESSURE
        return Phase.BUILD

    def _score_play_trainer(self, card: Card) -> float:
        # Keep the original v63 trainer priorities, but do not burn Premium Power
        # Pro after the planned attack already reaches the target's remaining HP.
        # The original rule gave every copy a fixed 5000 score and could consume
        # all four before an already-lethal attack.
        if card.id == C.PREMIUM_POWER_PRO and self.can_attack:
            already = game_memory.played_this_turn[C.PREMIUM_POWER_PRO]
            remaining = plan.remain_hp - 30 * already
            if remaining <= 0:
                return -1
        return super()._score_play_trainer(card)

    def _attack_damage_now(self, option) -> int:
        if option.type != OptionType.ATTACK or not self.me.active or not self.opponent.active:
            return 0
        attacker, target = self.me.active[0], self.opponent.active[0]
        attack = attack_table.get(option.attackId)
        if attack is None:
            return 0
        damage = attack.damage + 30 * game_memory.played_this_turn[C.PREMIUM_POWER_PRO]
        counter_effect = False
        if attacker.id == C.ALAKAZAM and option.attackId == 1072:
            damage = self.me.handCount * 20
            counter_effect = True
        if attacker.id == C.MEGA_LUCARIO_EX and target.id == C.CRUSTLE:
            return 0
        if self.stadium_id == 1247 and _is_rule_box(attacker) and not _is_rule_box(target):
            return 0
        if not counter_effect:
            adata, tdata = card_table[attacker.id], card_table[target.id]
            if "isn’t affected by Weakness" not in attack.text and tdata.weakness == adata.energyType:
                damage *= 2
            elif "isn’t affected by" not in attack.text and tdata.resistance == adata.energyType:
                damage = max(0, damage - 30)
        return damage

    def _exact_attack_action(self, baseline_action: int) -> int | None:
        attacks = [(i, self._attack_damage_now(o)) for i, o in enumerate(self.select.option) if o.type == OptionType.ATTACK]
        if not attacks or not self.opponent.active:
            return None
        hp = self.opponent.active[0].hp
        kos = [(i, dmg) for i, dmg in attacks if dmg >= hp]
        if kos:
            # Preserve the original attack plan. Aura Jab is often better than
            # Mega Brave when both take the same Prize because it accelerates the bench.
            if self.select.option[baseline_action].type == OptionType.ATTACK and self._attack_damage_now(self.select.option[baseline_action]) >= hp:
                return baseline_action
            preferred = [i for i, dmg in kos if (self.select.option[i].attackId == MEGA_BRAVE) == (plan.attack_index == 1)]
            if preferred:
                return preferred[0]
            aura = [i for i, dmg in kos if self.select.option[i].attackId == 982]
            return aura[0] if aura else kos[0][0]
        return None

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context != SelectContext.MAIN:
            return super().choose()
        self._plan_attack()
        baseline = [self._score_option(option) for option in self.select.option]
        ranked = sorted(range(len(baseline)), key=lambda i: baseline[i], reverse=True)
        baseline_action = ranked[0]
        if game_memory.deep_search_used:
            self._remember_lunatone_ability([baseline_action])
            return [baseline_action]
        planner = DeepPlanner(self, baseline, game_memory)
        selected = planner.select_action(baseline_action)
        if planner.did_search:
            game_memory.deep_search_used = True
        self._remember_lunatone_ability([selected])
        return [selected]


class DeepPlanner:
    def __init__(self, root_policy: SmartLucarioPolicy, baseline: list[float], memory: GameMemory):
        self.root_policy = root_policy
        self.root_obs = root_policy.obs
        self.root_index = root_policy.my_index
        self.op_index = root_policy.op_index
        self.root_turn = root_policy.state.turn
        self.phase = root_policy.phase
        self.baseline = baseline
        self.memory = memory
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + 0.16
        self.nodes = 0
        self.max_nodes = 500
        self.archetype = None
        self.did_search = False

    def _timed_out(self) -> bool:
        return self.nodes >= self.max_nodes or time.perf_counter() >= self.deadline

    def _root_candidates(self, baseline_action: int) -> list[int]:
        ranked = sorted(range(len(self.baseline)), key=lambda i: self.baseline[i], reverse=True)
        candidates = ranked[:6]
        for i, option in enumerate(self.root_obs.select.option):
            if option.type in {OptionType.ATTACK, OptionType.RETREAT, OptionType.ATTACH, OptionType.EVOLVE}:
                candidates.append(i)
            elif option.type == OptionType.END and not any(x.type == OptionType.ATTACK for x in self.root_obs.select.option):
                candidates.append(i)
            elif option.type == OptionType.PLAY:
                card = get_card(self.root_obs, AreaType.HAND, option.index, self.root_index)
                if card is not None and card.id in {C.PREMIUM_POWER_PRO,C.SWITCH,C.BOSS_ORDERS,C.CARMINE,C.LILLIE_DETERMINATION,C.JUDGE,C.GRAVITY_MOUNTAIN,C.DUSK_BALL,C.ULTRA_BALL,C.MEGA_SIGNAL,C.BUDDY_BUDDY_POFFIN,C.FIGHTING_GONG,C.POKE_PAD}:
                    candidates.append(i)
        out=[]
        root_has_attack = any(o.type == OptionType.ATTACK for o in self.root_obs.select.option)
        for i in candidates:
            if root_has_attack and self.root_obs.select.option[i].type == OptionType.END:
                continue
            if i not in out and self.baseline[i] > -1:
                out.append(i)
        if baseline_action not in out:
            out.insert(0, baseline_action)
        return out[:6]

    def select_action(self, baseline_action: int) -> int:
        if self.root_turn <= 1 or self.root_obs.search_begin_input is None or len(self.root_obs.select.option) <= 1:
            return baseline_action
        options = self.root_obs.select.option
        has_attack = any(o.type == OptionType.ATTACK for o in options)
        has_retreat = any(o.type == OptionType.RETREAT for o in options)
        has_switch = any(o.type == OptionType.PLAY and (get_card(self.root_obs, AreaType.HAND, o.index, self.root_index) or Card(-1,-1,-1)).id == C.SWITCH for o in options)
        has_ppp = any(o.type == OptionType.PLAY and (get_card(self.root_obs, AreaType.HAND, o.index, self.root_index) or Card(-1,-1,-1)).id == C.PREMIUM_POWER_PRO for o in options)
        has_judge = any(o.type == OptionType.PLAY and (get_card(self.root_obs, AreaType.HAND, o.index, self.root_index) or Card(-1,-1,-1)).id == C.JUDGE for o in options)
        active = self.root_policy.me.active[0] if self.root_policy.me.active else None
        locked_lucario = bool(active is not None and active.id == C.MEGA_LUCARIO_EX and len(active.energies) >= 2 and not any(o.type == OptionType.ATTACK and o.attackId == MEGA_BRAVE for o in options))
        trigger = (has_attack and has_ppp) or (self.phase == Phase.DEFEND and (has_retreat or has_switch)) or (self.phase == Phase.ENDGAME and has_attack) or locked_lucario or (has_judge and self.root_policy._opponent_is_psychic_engine()) or game_memory.searched_this_turn
        if not trigger:
            return baseline_action
        candidates = self._root_candidates(baseline_action)
        if len(candidates) <= 1:
            return baseline_action
        belief = OpponentBelief(self.root_obs, self.memory)
        self.archetype = belief.archetype
        # Search is enabled only where repeated evaluation showed a benefit. For
        # Alakazam, Crustle and Rocket, the original v63 policy is already stronger
        # and remains the exact fallback.
        if self.archetype in {"alakazam", "crustle", "rocket", "hop"}:
            return baseline_action
        if self.archetype == "v63" and not (self.phase in {Phase.DEFEND, Phase.ENDGAME} or locked_lucario):
            return baseline_action
        self.did_search = True
        if self.archetype != "v63":
            self.deadline = min(self.deadline, self.start_time + 0.065)
            self.max_nodes = min(self.max_nodes, 180)
        our_deck, our_prize = _our_hidden(self.root_obs)
        scenario_values = {i: [] for i in candidates}
        scenarios = belief.scenarios()[:1]
        for op_deck, op_prize, op_hand in scenarios:
            if self._timed_out():
                break
            try:
                root = search_begin(self.root_obs, our_deck, our_prize, op_deck, op_prize, op_hand, [])
                for idx in candidates:
                    if self._timed_out():
                        break
                    try:
                        self.nodes += 1
                        child = search_step(root.searchId, [idx])
                        value = self._rollout(child, max_steps=(130 if self.archetype == "v63" else 48))
                        scenario_values[idx].append(value)
                    except Exception:
                        continue
            except Exception:
                pass
            finally:
                try:
                    search_end()
                except Exception:
                    pass
        if not scenario_values.get(baseline_action):
            return baseline_action
        rank_pos = {idx: pos for pos, idx in enumerate(sorted(range(len(self.baseline)), key=lambda i:self.baseline[i], reverse=True))}
        combined={}
        for idx, vals in scenario_values.items():
            if not vals:
                continue
            mean=sum(vals)/len(vals)
            worst=min(vals)
            prior=max(0, 6-rank_pos.get(idx,6))*180
            combined[idx]=0.72*mean+0.28*worst+prior
        if baseline_action not in combined:
            return baseline_action
        best=max(combined,key=combined.get)
        base_opt=self.root_obs.select.option[baseline_action]
        best_opt=self.root_obs.select.option[best]
        base_raw = scenario_values.get(baseline_action, [])
        best_raw = scenario_values.get(best, [])
        # Search is a verifier, not a replacement policy. Override the original
        # v63 action only for a demonstrated terminal swing, a locked Mega Brave
        # reset, or a decisive survival line. This prevents heuristic leaf values
        # from producing the unnatural sequencing seen in the supplied replays.
        terminal_win = bool(best_raw and max(best_raw) >= 5e8 and not (base_raw and max(base_raw) >= 5e8))
        avoids_terminal_loss = bool(base_raw and min(base_raw) <= -5e8 and best_raw and min(best_raw) > -5e8)
        # The original v63 attack selector is already highly tuned. Search may decide
        # when to attack, but it must not replace one legal attack with a different one.
        if base_opt.type == OptionType.ATTACK and best_opt.type in {OptionType.ATTACK, OptionType.END}:
            return baseline_action
        # Preserve an original attack unless the simulated line is decisively better.
        if best == baseline_action:
            return baseline_action
        if terminal_win or avoids_terminal_loss:
            return best
        if locked_lucario and combined[best] >= combined[baseline_action] + 6000:
            return best
        if self.phase == Phase.DEFEND and best_opt.type in {OptionType.RETREAT, OptionType.ATTACH, OptionType.PLAY}:
            if combined[best] >= combined[baseline_action] + 14000:
                return best
        return baseline_action

    def _rollout(self, state, max_steps: int = 34) -> float:
        """Roll out the original v63 policy through the rest of this turn and
        the opponent's reply. Only the root action is changed; all later own
        choices return to the original policy. This is deliberately conservative.
        """
        steps = 0
        while steps < max_steps and not self._timed_out():
            obs = state.observation
            if obs.current is None or obs.current.result != -1 or obs.select is None:
                return self._leaf(obs)
            if obs.current.turn >= self.root_turn + 2 and obs.current.yourIndex == self.root_index and obs.select.context == SelectContext.MAIN:
                return self._leaf(obs)
            if obs.current.yourIndex == self.root_index or self.archetype == "v63":
                scores = self._own_scores(obs)
            elif obs.select.context == SelectContext.MAIN:
                scores = self._opponent_scores(obs)
            else:
                scores = self._forced_opponent_scores(obs)
            if not scores:
                return self._leaf(obs)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            count = obs.select.maxCount
            action = ranked[:count]
            if len(action) < obs.select.minCount:
                action = list(range(obs.select.minCount))
            # Never let the rollout end a turn while a legal attack exists.
            if obs.select.context == SelectContext.MAIN and action:
                chosen = obs.select.option[action[0]]
                if chosen.type == OptionType.END:
                    attacks = [i for i, o in enumerate(obs.select.option) if o.type == OptionType.ATTACK]
                    if attacks:
                        action = [max(attacks, key=lambda i: scores[i])]
            try:
                self.nodes += 1
                state = search_step(state.searchId, action)
            except Exception:
                return self._leaf(obs)
            steps += 1
        return self._leaf(state.observation)

    def _own_scores(self, obs: Observation) -> list[float]:
        global plan
        saved=AttackPlan(plan.attacker,plan.target,plan.attack_index,plan.remain_hp,plan.needs_energy)
        try:
            policy=LucarioPolicy(obs)
            if obs.select.context == SelectContext.MAIN:
                policy._plan_attack()
            return [policy._score_option(o) for o in obs.select.option]
        except Exception:
            return [0.0]*len(obs.select.option)
        finally:
            plan=saved

    def _opponent_scores(self, obs: Observation) -> list[float]:
        if self.archetype == "v63":
            return self._own_scores(obs)
        scores=[]
        for option in obs.select.option:
            score=0.0
            if option.type == OptionType.ATTACK:
                score=self._op_attack_score(obs,option)
            elif option.type == OptionType.ATTACH:
                score=8000+self._op_target_energy_score(obs,option)
            elif option.type == OptionType.EVOLVE:
                score=9000
                try:
                    c=get_card(obs,option.area,option.index,obs.current.yourIndex)
                    if c is not None and card_table[c.id].megaEx: score+=1500
                except Exception: pass
            elif option.type == OptionType.ABILITY:
                score=9200
            elif option.type == OptionType.RETREAT:
                score=5000
            elif option.type == OptionType.PLAY:
                card=get_card(obs,AreaType.HAND,option.index,obs.current.yourIndex)
                if card is not None:
                    data=card_table[card.id]
                    score={CardType.POKEMON:6500,CardType.ITEM:7000,CardType.TOOL:6500,CardType.SUPPORTER:7600,CardType.STADIUM:6400}.get(data.cardType,4000)
                    if card.id==C.BOSS_ORDERS: score+=1800
                    if card.id in {C.PREMIUM_POWER_PRO,1120,1081}: score+=900
            elif option.type == OptionType.END:
                score=100
            scores.append(score)
        return scores

    def _op_attack_score(self, obs: Observation, option) -> float:
        current=obs.current
        attacker=current.players[current.yourIndex].active[0] if current.players[current.yourIndex].active else None
        target=current.players[1-current.yourIndex].active[0] if current.players[1-current.yourIndex].active else None
        if attacker is None or target is None:
            return 1000
        attack=attack_table.get(option.attackId)
        if attack is None:
            return 3000
        damage=attack.damage
        counter=False
        if attacker.id==C.ALAKAZAM and option.attackId==1072:
            damage=current.players[current.yourIndex].handCount*20; counter=True
        elif attacker.id==121 and option.attackId==154:
            damage=200
        if current.stadium and current.stadium[0].id==1247 and _is_rule_box(attacker) and not _is_rule_box(target):
            damage=0
        if not counter and damage>0:
            ad,td=card_table[attacker.id],card_table[target.id]
            if "isn’t affected by Weakness" not in attack.text and td.weakness==ad.energyType: damage*=2
            elif "isn’t affected by" not in attack.text and td.resistance==ad.energyType: damage=max(0,damage-30)
        ko=damage>=target.hp
        return 10000+damage*8+(prize_count(target)*9000 if ko else 0)

    def _op_target_energy_score(self, obs: Observation, option) -> float:
        try:
            p=get_card(obs,option.inPlayArea,option.inPlayIndex,obs.current.yourIndex)
            if not isinstance(p,Pokemon): return 0
            needed=min((len(attack_table[a].energies) for a in card_table[p.id].attacks),default=3)
            return max(0,needed-len(p.energies))*400+prize_count(p)*100
        except Exception:
            return 0

    def _forced_action(self, obs: Observation) -> list[int]:
        if obs.select.maxCount == 0:
            return []
        if obs.current.yourIndex == self.root_index or self.archetype == "v63":
            scores=self._own_scores(obs)
        else:
            scores=self._forced_opponent_scores(obs)
        ranked=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
        count=obs.select.maxCount
        action=ranked[:count]
        if len(action)<obs.select.minCount:
            action=list(range(obs.select.minCount))
        return action

    def _forced_opponent_scores(self, obs: Observation) -> list[float]:
        scores=[]
        ctx=obs.select.context
        for option in obs.select.option:
            score=0.0
            if option.type==OptionType.CARD:
                card=get_card(obs,option.area,option.index,option.playerIndex)
                if card is None:
                    score=0
                elif ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE,SelectContext.SETUP_ACTIVE_POKEMON} and isinstance(card,Pokemon):
                    score=len(card.energies)*500+card.hp+prize_count(card)*80
                    if card_table[card.id].basic and ctx==SelectContext.SETUP_ACTIVE_POKEMON: score-=prize_count(card)*50
                elif ctx==SelectContext.TO_HAND:
                    score=_card_utility(card.id,obs.current.players[obs.current.yourIndex],True)
                elif ctx in {SelectContext.DISCARD,SelectContext.TO_DECK,SelectContext.TO_DECK_BOTTOM}:
                    score=-_card_utility(card.id,obs.current.players[obs.current.yourIndex],True)
                elif ctx in {SelectContext.EFFECT_TARGET,SelectContext.DAMAGE,SelectContext.DAMAGE_COUNTER,SelectContext.DAMAGE_COUNTER_ANY} and isinstance(card,Pokemon):
                    score=prize_count(card)*1000+(card.maxHp-card.hp)*5-card.hp
                elif ctx==SelectContext.ATTACH_FROM and isinstance(card,Pokemon):
                    score=prize_count(card)*200+len(card.energies)*50
                else:
                    score=_card_utility(card.id,obs.current.players[obs.current.yourIndex],True)
            elif option.type==OptionType.NUMBER:
                score=option.number
            elif option.type==OptionType.YES:
                score=10
            elif option.type==OptionType.NO:
                score=0
            else:
                score=0
            scores.append(score)
        return scores

    def _leaf(self, obs: Observation) -> float:
        if obs.current is None:
            return -1e8
        state=obs.current
        if state.result != -1:
            if state.result==self.root_index: return 1e9
            if state.result==2: return 0
            return -1e9
        me=state.players[self.root_index]; op=state.players[self.op_index]
        value=(len(op.prize)-len(me.prize))*52000
        value+=(me.handCount-op.handCount)*180
        value+=(me.deckCount-op.deckCount)*25
        stadium_id=state.stadium[0].id if state.stadium else 0
        def unit_value(p: Pokemon, ours: bool) -> float:
            base=p.hp*8+len(p.energies)*850+len(p.tools)*500
            data=card_table[p.id]
            if data.stage1: base+=700
            if data.stage2: base+=1000
            if p.id==C.MEGA_LUCARIO_EX: base+=1700
            if p.id==C.HARIYAMA: base+=1100
            damage=_generic_attack_damage(p,(op.active[0] if ours and op.active else me.active[0] if (not ours and me.active) else None), me.handCount if ours else op.handCount,stadium_id)
            base+=damage*9
            return base
        for p in me.active+me.bench:
            if p is not None: value+=unit_value(p,True)
        for p in op.active+op.bench:
            if p is not None: value-=unit_value(p,False)*0.92
        if me.active:
            incoming=max((_generic_attack_damage(p,me.active[0],op.handCount,stadium_id) for p in op.active+op.bench if p is not None),default=0)
            if incoming>=me.active[0].hp:
                value-=prize_count(me.active[0])*14500
        if op.active:
            outgoing=max((_generic_attack_damage(p,op.active[0],me.handCount,stadium_id) for p in me.active+me.bench if p is not None),default=0)
            if outgoing>=op.active[0].hp:
                value+=prize_count(op.active[0])*12000
        if stadium_id==C.GRAVITY_MOUNTAIN:
            value+=sum(1000 for p in op.active+op.bench if p is not None and card_table[p.id].stage2)
        elif stadium_id==1255 and any(p is not None and p.id in {878,879,304} for p in op.active+op.bench):
            value-=3500
        if game_memory.searched_this_turn and state.supporterPlayed:
            # Search followed by hand-shuffle draw is usually the replayed inefficiency.
            value-=1200
        if self.phase==Phase.OPENING:
            value+=(sum(1 for p in me.active+me.bench if p is not None and p.id in {C.RIOLU,C.MEGA_LUCARIO_EX})*900)
        elif self.phase==Phase.ACCELERATE:
            value+=sum(min(len(p.energies),2)*500 for p in me.bench if p.id in {C.RIOLU,C.MEGA_LUCARIO_EX})
        elif self.phase==Phase.ENDGAME:
            value+=(6-len(me.prize))*1800
        return value

def agent(obs_dict: dict) -> list[int]:
    global pre_turn
    global ability_used
    global plan

    if obs_dict.get("select") is None and "current" not in obs_dict:
        pre_turn = -1
        ability_used = False
        plan = AttackPlan()
        game_memory.reset()
        return my_deck

    obs = to_observation_class(obs_dict)
    if obs.select is None:
        pre_turn = -1
        ability_used = False
        plan = AttackPlan()
        game_memory.reset()
        return my_deck

    game_memory.update(obs)
    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    return SmartLucarioPolicy(obs).choose()
