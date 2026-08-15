"""Counterfactual public-state residuals for Judge/Dusk Dragapult.

The residual never reads either hidden hand, Prize cards, or deck order.  It
keeps the v25 exact-deck Transformer as the parent policy and intervenes only
when Cursed Blast arithmetic is fully determined from the public board.
"""
from __future__ import annotations

from collections import Counter
import copy

DUSKULL = 131
DUSCLOPS = 132
DUSKNOIR = 133
JUDGE = 1213
CRISPIN = 1198
DRAGAPULT = 121
FIRE = 2
PSYCHIC = 5
PHANTOM_DIVE = 154

FULL_METAL_LAB = 1244
BATTLE_CAGE = 1264
ALAKAZAM_FAMILY = {741, 742, 743}
CRUSTLE_FAMILY = {344, 345}
METAL = 8

MAIN = 0
DAMAGE_COUNTER = 13
AREA_ACTIVE = 4
AREA_BENCH = 5


class ExactDeckResidual:
    """Regression-gated atomic policy repair.

    ``mode='hold'`` enables target repair and the fresh-Dusclops hold rule.
    ``mode='combo'`` additionally forces only mathematically exact same-turn
    Cursed Blast -> Phantom Dive wins.
    """

    def __init__(self, api, controller, base, mode="hold"):
        self.api = api
        self.controller = controller
        self.base = base
        self.mode = str(mode or "hold")
        self.stats = Counter()
        self.reset()

    def reset(self):
        self.pending_target_serial = -1
        self.pending_effect = 0
        self.pending_turn = -1
        self.pending_attack = False
        self.pending_phantom_serial = -1
        self.stats["games"] += 1

    def get_stats(self):
        return dict(self.stats)

    @staticmethod
    def _int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _board(self, player):
        return [
            p
            for p in list(player.active or []) + list(player.bench or [])
            if p is not None
        ]

    def _src(self, obs, option):
        try:
            return self.controller._source_card(obs, option)
        except Exception:
            return None

    def _cid(self, obs, option):
        card = self._src(obs, option)
        return self._int(getattr(card, "id", getattr(option, "cardId", 0)), 0)

    def _prizes(self, pokemon, is_attack_damage=False):
        try:
            return self._int(self.base.prize_count(pokemon, bool(is_attack_damage)), 1)
        except Exception:
            try:
                data = self.base.card_table[self._int(pokemon.id)]
                value = 3 if bool(getattr(data, "megaEx", False)) else 2 if bool(getattr(data, "ex", False)) else 1
                return max(0, value)
            except Exception:
                return 1

    def _stadium_id(self, obs):
        try:
            stadium = list(obs.current.stadium or [])
            return self._int(getattr(stadium[0], "id", 0), 0) if stadium else 0
        except Exception:
            return 0

    def _is_metal(self, pokemon):
        try:
            data = self.base.card_table.get(self._int(getattr(pokemon, "id", 0), 0))
            return self._int(getattr(data, "energyType", 0), 0) == METAL
        except Exception:
            return False

    def _card_skill_text(self, pokemon):
        try:
            data = self.base.card_table.get(self._int(getattr(pokemon, "id", 0), 0))
            return " ".join(
                str(getattr(skill, "text", "") or "")
                for skill in list(getattr(data, "skills", []) or [])
            ).lower()
        except Exception:
            return ""

    def _attack_wall(self, pokemon):
        """True when public card text/known semantics stop Dragapult ex damage."""
        try:
            if bool(self.base.no_damage_dex(self._int(getattr(pokemon, "id", 0), 0))):
                return True
        except Exception:
            pass
        text = self._card_skill_text(pokemon)
        ex_token = "pokémon ex" in text or "pokemon ex" in text or "{ex}" in text
        return "prevent all damage" in text and ex_token

    def _counter_legal(self, obs, pokemon, is_active):
        # Battle Cage blocks counters placed on the opponent's Bench by both
        # Cursed Blast and Phantom Dive.  It does not protect the Active.
        if not is_active and self._stadium_id(obs) == BATTLE_CAGE:
            self.stats["battle_cage_bench_block"] += 1
            return False
        try:
            return not bool(self.base.no_damage_counter(pokemon))
        except Exception:
            return True

    def _phantom_active_damage(self, obs, pokemon):
        damage = 200
        if self._stadium_id(obs) == FULL_METAL_LAB and self._is_metal(pokemon):
            damage -= 30
            self.stats["full_metal_active_adjust"] += 1
        return damage

    def _ready_phantom(self, mine):
        for pokemon in self._board(mine):
            if self._int(getattr(pokemon, "id", 0), 0) != DRAGAPULT:
                continue
            energies = {
                self._int(getattr(card, "id", 0), 0)
                for card in list(getattr(pokemon, "energyCards", []) or [])
            }
            if FIRE in energies and PSYCHIC in energies:
                return True
        return False

    def _public_opponent_ids(self, obs):
        out = set()
        try:
            state = obs.current
            me = self._int(state.yourIndex)
            opponent = state.players[1 - me]
            for pokemon in self._board(opponent):
                out.add(self._int(getattr(pokemon, "id", 0), 0))
                out.update(
                    self._int(getattr(card, "id", 0), 0)
                    for card in list(getattr(pokemon, "preEvolution", []) or [])
                )
            out.update(
                self._int(getattr(card, "id", 0), 0)
                for card in list(opponent.discard or [])
                if card is not None
            )
            for log in list(obs.logs or []):
                if self._int(getattr(log, "playerIndex", -1), -1) != 1 - me:
                    continue
                for name in (
                    "cardId", "cardIdAfter", "cardIdBefore", "cardIdActive",
                    "cardIdBench", "cardIdTarget",
                ):
                    cid = self._int(getattr(log, name, 0), 0)
                    if cid > 0:
                        out.add(cid)
        except Exception:
            pass
        out.discard(0)
        return out

    def _hold_blocked(self, obs):
        public_ids = self._public_opponent_ids(obs)
        if public_ids & ALAKAZAM_FAMILY:
            self.stats["hold_block_alakazam"] += 1
            return True
        if public_ids & CRUSTLE_FAMILY:
            self.stats["hold_block_crustle"] += 1
            return True
        # The frozen recognizer itself is also public-information only.  It can
        # identify a known portfolio from already revealed support cards before
        # the evolution family is physically in play, preventing an early
        # matchup-specific hold regression without peeking at hidden zones.
        try:
            name, confidence = self.controller.recognize(obs)
            if name == "alakazam" and float(confidence) >= 0.10:
                self.stats["hold_block_recognized_alakazam"] += 1
                return True
            if name == "crustle" and float(confidence) >= 0.10:
                self.stats["hold_block_recognized_crustle"] += 1
                return True
        except Exception:
            pass
        try:
            state = obs.current
            me = self._int(state.yourIndex)
            if any(self._attack_wall(pokemon) for pokemon in self._board(state.players[1 - me])):
                self.stats["hold_block_semantic_wall"] += 1
                return True
        except Exception:
            pass
        return False

    def _target_rows(self, obs, damage):
        state = obs.current
        me = self._int(state.yourIndex)
        mine = state.players[me]
        opponent = state.players[1 - me]
        my_left = len(mine.prize or [])
        opponent_left = len(opponent.prize or [])
        if opponent_left <= 1:
            return []

        ready = self._ready_phantom(mine)
        active = opponent.active[0] if opponent.active and opponent.active[0] is not None else None
        active_serial = self._int(getattr(active, "serial", -2), -2)
        rows = []
        for pokemon in self._board(opponent):
            hp = self._int(getattr(pokemon, "hp", 0), 0)
            serial = self._int(getattr(pokemon, "serial", -1), -1)
            is_active = active is not None and active_serial == serial
            if hp <= 0 or not self._counter_legal(obs, pokemon, is_active):
                continue

            direct = hp <= damage
            wall = self._attack_wall(pokemon)
            active_attack = self._phantom_active_damage(obs, pokemon) if is_active else 0
            active_bridge = bool(
                ready and is_active and not wall and active_attack < hp <= active_attack + damage
            )
            bench_bridge = bool(
                ready and not is_active and 60 < hp <= 60 + damage
            )
            bridge = active_bridge or bench_bridge
            if not (direct or bridge):
                continue

            direct_prize = self._prizes(pokemon, False)
            # Active Phantom Dive is attack damage.  Its Bench placement is an
            # effect of an attack, so Legacy Energy/Lillie's Pearl do not apply.
            bridge_prize = self._prizes(pokemon, is_active)
            prize = direct_prize if direct else bridge_prize
            exact = bool(direct and direct_prize >= my_left)
            energies = len(list(getattr(pokemon, "energyCards", []) or []))
            if direct:
                threshold = hp
            else:
                threshold = hp - (active_attack if is_active else 60)
            waste = max(0, damage - threshold)
            score = (
                1 if exact else 0,
                prize,
                1 if wall else 0,
                1 if direct else 0,
                energies,
                1 if is_active else 0,
                -waste,
                -hp,
                -serial,
            )
            rows.append(
                (
                    score,
                    pokemon,
                    {
                        "direct": direct,
                        "bridge": bridge,
                        "exact": exact,
                        "prize": prize,
                        "wall": wall,
                        "is_active": is_active,
                    },
                )
            )
        rows.sort(key=lambda item: item[0], reverse=True)
        return rows

    def _best_target(self, obs, damage):
        rows = self._target_rows(obs, damage)
        return (rows[0][1], rows[0][2]) if rows else None

    def _masked_parent(self, observation, blocked):
        """Ask the frozen v25 controller for its next-best legal action."""
        try:
            raw = copy.deepcopy(observation)
            select = raw.get("select") or {}
            old = list(select.get("option") or [])
            blocked_set = set(blocked)
            keep = [index for index in range(len(old)) if index not in blocked_set]
            if not keep:
                return None
            select["option"] = [old[index] for index in keep]
            output = self.controller.agent(raw)
            if not isinstance(output, list):
                return None
            mapped = [keep[self._int(index)] for index in output]
            raw_select = observation.get("select") or {}
            lo = self._int(raw_select.get("minCount", 0), 0)
            hi = self._int(raw_select.get("maxCount", 0), 0)
            if lo <= len(mapped) <= hi and len(set(mapped)) == len(mapped):
                return mapped
        except Exception:
            return None
        return None

    def _selected_main_dusk(self, obs, chosen):
        try:
            select = obs.select
            options = list(select.option or [])
            if self._int(select.context) != MAIN or not isinstance(chosen, list) or len(chosen) != 1:
                return None
            index = self._int(chosen[0], -1)
            if not (0 <= index < len(options)):
                return None
            option = options[index]
            if option.type != self.api.OptionType.ABILITY:
                return None
            card = self._src(obs, option)
            cid = self._int(getattr(card, "id", 0), 0)
            if cid not in (DUSCLOPS, DUSKNOIR):
                return None
            return index, cid, card
        except Exception:
            return None

    def _exact_bridge_plan(self, obs, damage):
        """Same-turn, public-state Cursed Blast -> Phantom Dive checkmate."""
        state = obs.current
        me = self._int(state.yourIndex)
        mine = state.players[me]
        opponent = state.players[1 - me]
        if len(opponent.prize or []) <= 1:
            return None
        my_left = len(mine.prize or [])
        active = opponent.active[0] if opponent.active and opponent.active[0] is not None else None
        active_serial = self._int(getattr(active, "serial", -2), -2)
        rows = []
        for pokemon in self._board(opponent):
            hp = self._int(getattr(pokemon, "hp", 0), 0)
            serial = self._int(getattr(pokemon, "serial", -1), -1)
            is_active = active is not None and active_serial == serial
            if hp <= 0 or not self._counter_legal(obs, pokemon, is_active):
                continue
            wall = self._attack_wall(pokemon)
            active_attack = self._phantom_active_damage(obs, pokemon) if is_active else 0
            bridge = bool(
                (is_active and not wall and active_attack < hp <= active_attack + damage)
                or (not is_active and 60 < hp <= 60 + damage)
            )
            if not bridge:
                continue
            prizes = self._prizes(pokemon, is_active)
            if prizes < my_left:
                continue
            threshold = hp - (active_attack if is_active else 60)
            waste = max(0, damage - threshold)
            rows.append(
                (
                    (
                        prizes,
                        len(list(getattr(pokemon, "energyCards", []) or [])),
                        1 if is_active else 0,
                        -waste,
                        -hp,
                        -serial,
                    ),
                    pokemon,
                )
            )
        rows.sort(key=lambda item: item[0], reverse=True)
        return rows[0][1] if rows else None

    def _clear_contract(self):
        self.pending_target_serial = -1
        self.pending_effect = 0
        self.pending_turn = -1
        self.pending_attack = False

    def _main_exact_combo(self, obs, chosen):
        if self.mode != "combo":
            return chosen
        select = obs.select
        options = list(select.option or [])
        turn = self._int(obs.current.turn, -1)
        if self.pending_turn >= 0 and self.pending_turn != turn:
            self.stats["exact_contract_expired"] += 1
            self._clear_contract()
        if self._int(select.context) != MAIN or self._int(select.minCount) != 1 or self._int(select.maxCount) != 1:
            return chosen

        phantom = next(
            (
                index
                for index, option in enumerate(options)
                if option.type == self.api.OptionType.ATTACK
                and self._int(getattr(option, "attackId", 0), 0) == PHANTOM_DIVE
            ),
            None,
        )
        # A legal Phantom Dive option proves the ready Dragapult is currently
        # Active, so a Benched Dusk self-KO cannot interrupt the final attack.
        if phantom is None:
            return chosen
        if self.pending_attack and self.pending_turn == turn:
            self.stats["exact_bridge_force_phantom"] += 1
            self._clear_contract()
            return [phantom]

        plans = {}
        for source_id, damage in ((DUSKNOIR, 130), (DUSCLOPS, 50)):
            target = self._exact_bridge_plan(obs, damage)
            if target is not None:
                plans[source_id] = target

        if DUSKNOIR in plans:
            evolve = next(
                (
                    index
                    for index, option in enumerate(options)
                    if option.type == self.api.OptionType.EVOLVE and self._cid(obs, option) == DUSKNOIR
                ),
                None,
            )
            if evolve is not None:
                self.stats["exact_bridge_evolve_dusknoir"] += 1
                return [evolve]
            ability = next(
                (
                    index
                    for index, option in enumerate(options)
                    if option.type == self.api.OptionType.ABILITY and self._cid(obs, option) == DUSKNOIR
                ),
                None,
            )
            if ability is not None:
                self.pending_target_serial = self._int(getattr(plans[DUSKNOIR], "serial", -1), -1)
                self.pending_effect = DUSKNOIR
                self.pending_turn = turn
                self.stats["exact_bridge_start_dusknoir"] += 1
                return [ability]

        if DUSCLOPS in plans:
            ability = next(
                (
                    index
                    for index, option in enumerate(options)
                    if option.type == self.api.OptionType.ABILITY and self._cid(obs, option) == DUSCLOPS
                ),
                None,
            )
            if ability is not None:
                self.pending_target_serial = self._int(getattr(plans[DUSCLOPS], "serial", -1), -1)
                self.pending_effect = DUSCLOPS
                self.pending_turn = turn
                self.stats["exact_bridge_start_dusclops"] += 1
                return [ability]
        return chosen


    def _option_pokemon_v28(self, obs, option):
        try:
            state=obs.current
            pi=self._int(getattr(option,"playerIndex",state.yourIndex),state.yourIndex)
            area=self._int(getattr(option,"area",-1),-1)
            index=self._int(getattr(option,"index",-1),-1)
            player=state.players[pi]
            cards=list(player.active or []) if area==AREA_ACTIVE else list(player.bench or []) if area==AREA_BENCH else []
            if 0<=index<len(cards): return cards[index]
        except Exception: pass
        return self._src(obs,option)

    def _marnie_visible_v28(self, obs):
        return bool(self._public_opponent_ids(obs) & {646,647,648,860,104})

    def _phantom_marnie_threshold_v28(self, obs, chosen):
        """Counterfactual lesson: concentrate Phantom's 60 so the next 50/130 Blast cashes a Prize.
        Only activates after the Marnie family is publicly identified; every other matchup is byte-for-byte v26 behavior.
        """
        select=obs.select
        if self._int(select.context)!=14: return chosen
        effect=self._int(getattr(getattr(select,"effect",None),"id",0),0)
        if effect!=DRAGAPULT or not self._marnie_visible_v28(obs): return chosen
        state=obs.current;me=self._int(state.yourIndex);mine=state.players[me]
        remain=max(1,self._int(getattr(select,"remainDamageCounter",0),0))*10
        board=self._board(mine)
        have50=any(self._int(getattr(p,"id",0),0)==DUSCLOPS for p in board)
        # Dusclops in play is also a credible next-turn 130 route because the deck carries Dusknoir.
        have130=any(self._int(getattr(p,"id",0),0) in (DUSCLOPS,DUSKNOIR) for p in board)
        rows=[]
        for idx,opt in enumerate(list(select.option or [])):
            p=self._option_pokemon_v28(obs,opt)
            if p is None: continue
            hp=self._int(getattr(p,"hp",0),0);serial=self._int(getattr(p,"serial",-1),-1)
            if hp<=0: continue
            try:
                if self.base.no_damage_counter(p): continue
            except Exception: pass
            prizes=self._prizes(p,False);after=max(0,hp-remain);cid=self._int(getattr(p,"id",0),0)
            ko=hp<=remain
            setup50=bool(not ko and have50 and 0<after<=50)
            setup130=bool(not ko and have130 and prizes>=2 and 0<after<=130)
            # One-prize Marnie evolution engines are worth setting to 50, but not spending 130 setup on spec.
            if not (ko or setup50 or setup130): continue
            energy=len(list(getattr(p,"energyCards",[]) or []));stage=len(list(getattr(p,"preEvolution",[]) or []))
            sticky=(serial==self.pending_phantom_serial)
            engine=(cid in {646,647,648,860,104})
            score=(500 if ko else 0)+prizes*100+(140 if setup50 else 0)+(90 if setup130 else 0)+(35 if engine else 0)+energy*8+stage*6+(35 if sticky else 0)-max(0,remain-hp)*0.1
            rows.append((score,idx,p,ko,setup50,setup130,prizes))
        if not rows: return chosen
        rows.sort(key=lambda x:x[0],reverse=True);best=rows[0]
        old=chosen[0] if isinstance(chosen,list) and len(chosen)==1 else -1
        # Do not replace an equal/higher-Prize immediate KO already found by the parent.
        if 0<=old<len(select.option or []):
            op=self._option_pokemon_v28(obs,list(select.option)[old])
            if op is not None and 0<self._int(getattr(op,"hp",0),0)<=remain and self._prizes(op,False)>=best[6]:
                self.pending_phantom_serial=self._int(getattr(op,"serial",-1),-1);return chosen
        self.pending_phantom_serial=self._int(getattr(best[2],"serial",-1),-1)
        if best[1]!=old:
            self.stats["v28_phantom_threshold_override"]+=1
            if best[3]: self.stats["v28_phantom_ko"]+=1
            if best[4]: self.stats["v28_phantom_to_50"]+=1
            if best[5]: self.stats["v28_phantom_to_130"]+=1
            return [best[1]]
        return chosen


    def _v28_exact_active_combo(self, obs, chosen):
        """Public same-turn Dusk -> Phantom conversion planner.

        v28 adds the 3-Prize Mega Lucario line that the one-Blast planner missed:
        130 + 50 + Phantom 200 = 380, enough for 340 HP while trading two
        one-Prize Dusk bodies for a three-Prize Mega KO.  The second Blast is
        only chained when two Dusk bodies are actually on board and the legal
        action list proves the pieces are available.
        """
        select=obs.select
        if self._int(select.context)!=MAIN or self._int(select.minCount)!=1 or self._int(select.maxCount)!=1:
            return chosen
        turn=self._int(obs.current.turn,-1)
        options=list(select.option or [])
        phantom=next((i for i,o in enumerate(options) if o.type==self.api.OptionType.ATTACK and self._int(getattr(o,"attackId",0),0)==PHANTOM_DIVE),None)
        if phantom is None: return chosen
        state=obs.current;me=self._int(state.yourIndex);mine,opp=state.players[me],state.players[1-me]
        if len(opp.prize or [])<=1:return chosen
        active=opp.active[0] if opp.active and opp.active[0] is not None else None
        if active is None:return chosen
        cid=self._int(getattr(active,"id",0),0)
        if cid not in (190,648,678): return chosen
        if self._attack_wall(active):return chosen
        hp=self._int(getattr(active,"hp",0),0);pd=self._phantom_active_damage(obs,active);need=hp-pd
        prizes=self._prizes(active,True)
        dusclops_abilities=[i for i,o in enumerate(options) if o.type==self.api.OptionType.ABILITY and self._cid(obs,o)==DUSCLOPS]
        dusknoir_abilities=[i for i,o in enumerate(options) if o.type==self.api.OptionType.ABILITY and self._cid(obs,o)==DUSKNOIR]
        dusknoir_evolves=[i for i,o in enumerate(options) if o.type==self.api.OptionType.EVOLVE and self._cid(obs,o)==DUSKNOIR]
        dusk_units=[p for p in self._board(mine) if self._int(getattr(p,"id",0),0) in (DUSCLOPS,DUSKNOIR)]

        # A previous Blast has resolved.  Before cashing Phantom, allow the
        # second 50 only for a 3-Prize Mega where it changes a 10-50 HP miss
        # into an immediate KO.
        if self.pending_attack and self.pending_turn==turn:
            if cid==678 and prizes>=3 and 0<need<=50 and dusclops_abilities:
                self.pending_attack=False
                self.pending_target_serial=self._int(getattr(active,"serial",-1),-1)
                self.pending_effect=DUSCLOPS;self.pending_turn=turn
                self.stats["v28_double_lucario_second_50"]+=1
                return [dusclops_abilities[0]]
            self.stats["v28_force_phantom_after_blast"]+=1
            self._clear_contract();return [phantom]

        if need<=0:return chosen

        # Existing one-Blast exact conversions for the two validated 2-Prize
        # actives, plus a cheaper one-Blast line if Mega Lucario is already
        # damaged into range.
        if cid in (190,648) and need>130:return chosen
        if cid==678 and (prizes<3 or need>180):return chosen

        if need<=50 and dusclops_abilities:
            self.pending_target_serial=self._int(getattr(active,"serial",-1),-1);self.pending_effect=DUSCLOPS;self.pending_turn=turn
            self.stats["v28_exact_start_dusclops"]+=1;return [dusclops_abilities[0]]

        if need<=130:
            if dusknoir_abilities:
                self.pending_target_serial=self._int(getattr(active,"serial",-1),-1);self.pending_effect=DUSKNOIR;self.pending_turn=turn
                self.stats["v28_exact_start_dusknoir"]+=1;return [dusknoir_abilities[0]]
            if dusknoir_evolves:
                self.stats["v28_exact_evolve_dusknoir"]+=1;return [dusknoir_evolves[0]]
            return chosen

        # 131-180 remaining HP: only a three-Prize Mega justifies sacrificing
        # two one-Prize Dusk pieces.  Require two evolved Dusk bodies in public
        # state so an evolve/ability option cannot accidentally consume the
        # only available body.
        if cid==678 and prizes>=3 and 130<need<=180 and len(dusk_units)>=2 and dusclops_abilities:
            if dusknoir_abilities:
                self.pending_target_serial=self._int(getattr(active,"serial",-1),-1);self.pending_effect=DUSKNOIR;self.pending_turn=turn
                self.stats["v28_double_lucario_start_130"]+=1;return [dusknoir_abilities[0]]
            if dusknoir_evolves:
                self.stats["v28_double_lucario_evolve_130"]+=1;return [dusknoir_evolves[0]]
        return chosen

    def _v29_own_target(self, obs, option):
        try:
            state = obs.current
            mine = state.players[self._int(state.yourIndex)]
            area = self._int(getattr(option, "inPlayArea", -1), -1)
            index = self._int(getattr(option, "inPlayIndex", -1), -1)
            cards = list(mine.active or []) if area == AREA_ACTIVE else list(mine.bench or []) if area == AREA_BENCH else []
            return cards[index] if 0 <= index < len(cards) else None
        except Exception:
            return None

    def _v29_energy_ids(self, pokemon):
        try:
            return {self._int(getattr(e, "id", 0), 0) for e in list(pokemon.energyCards or [])}
        except Exception:
            return set()

    def _v29_ready_drag(self, obs):
        try:
            mine = obs.current.players[self._int(obs.current.yourIndex)]
            return any(
                self._int(getattr(p, "id", 0), 0) == DRAGAPULT
                and {FIRE, PSYCHIC}.issubset(self._v29_energy_ids(p))
                for p in self._board(mine)
            )
        except Exception:
            return False

    def _v29_lucario_visible(self, obs):
        return bool(self._public_opponent_ids(obs) & {333, 675, 676, 677, 678})

    def _v29_energy_guard(self, obs, chosen):
        """Before the first F/P Dragapult is online, don't feed basic energy to Dusk
        if that same attachment can advance a Dragapult line.  Win/loss replay
        analysis showed this as a cross-matchup tempo discriminator.
        """
        select = obs.select
        if self._int(select.context) != MAIN or not isinstance(chosen, list) or len(chosen) != 1:
            return chosen
        options = list(select.option or [])
        old = self._int(chosen[0], -1)
        if not (0 <= old < len(options)) or options[old].type != self.api.OptionType.ATTACH:
            return chosen
        target = self._v29_own_target(obs, options[old])
        if target is None or self._int(getattr(target, "id", 0), 0) not in (DUSKULL, DUSCLOPS, DUSKNOIR):
            return chosen
        if self._v29_ready_drag(obs):
            return chosen
        eid = self._cid(obs, options[old])
        if eid not in (FIRE, PSYCHIC):
            return chosen
        rows = []
        for index, option in enumerate(options):
            if option.type != self.api.OptionType.ATTACH or self._cid(obs, option) != eid:
                continue
            pokemon = self._v29_own_target(obs, option)
            cid = self._int(getattr(pokemon, "id", 0), 0) if pokemon is not None else 0
            if cid not in (119, 120, DRAGAPULT):
                continue
            have = self._v29_energy_ids(pokemon)
            if eid in have:
                continue
            other = PSYCHIC if eid == FIRE else FIRE
            # Highest value is completing F/P; otherwise advance the most evolved line.
            complete = 1 if other in have else 0
            active = 1 if any(self._int(getattr(x, "serial", -2), -2) == self._int(getattr(pokemon, "serial", -3), -3) for x in list(obs.current.players[self._int(obs.current.yourIndex)].active or []) if x is not None) else 0
            stage = {119: 1, 120: 2, DRAGAPULT: 3}.get(cid, 0)
            rows.append(((complete, stage, active, -index), index))
        if rows:
            rows.sort(reverse=True)
            best = rows[0][1]
            if best != old:
                self.stats["v29_core_first_energy"] += 1
                return [best]
        return chosen

    def _v29_judge_gate(self, obs, chosen):
        """Lucario-only early-game gate: when the attacker is not ready, spend
        Crispin on guaranteed F/P development before voluntarily Judge-resetting.
        """
        select = obs.select
        if self._int(select.context) != MAIN or not isinstance(chosen, list) or len(chosen) != 1:
            return chosen
        if self._int(obs.current.turn, 99) > 8 or self._v29_ready_drag(obs) or not self._v29_lucario_visible(obs):
            return chosen
        options = list(select.option or [])
        old = self._int(chosen[0], -1)
        if not (0 <= old < len(options)) or options[old].type != self.api.OptionType.PLAY or self._cid(obs, options[old]) != JUDGE:
            return chosen
        crispin = next((i for i,o in enumerate(options) if o.type == self.api.OptionType.PLAY and self._cid(obs,o) == CRISPIN), None)
        if crispin is not None:
            self.stats["v29_lucario_crispin_over_judge"] += 1
            return [crispin]
        return chosen

    def _v29_blast_guard(self, observation, obs, chosen):
        """Do not cash a low-value Cursed Blast before the first attacker is
        online. Exact KO, wall removal, multi-Prize, and contracted combo lines
        remain untouched.
        """
        select = obs.select
        if self._int(select.context) != MAIN or not isinstance(chosen, list) or len(chosen) != 1 or self._v29_ready_drag(obs):
            return chosen
        options = list(select.option or [])
        old = self._int(chosen[0], -1)
        if not (0 <= old < len(options)) or options[old].type != self.api.OptionType.ABILITY:
            return chosen
        cid = self._cid(obs, options[old])
        if cid not in (DUSCLOPS, DUSKNOIR):
            return chosen
        turn = self._int(obs.current.turn, -1)
        if self.pending_effect == cid and self.pending_turn == turn and self.pending_target_serial >= 0:
            return chosen
        damage = 50 if cid == DUSCLOPS else 130
        if self._high_confidence_use(obs, damage):
            return chosen
        blocked = [i for i,o in enumerate(options) if o.type == self.api.OptionType.ABILITY and self._cid(obs,o) in (DUSCLOPS,DUSKNOIR)]
        alternative = self._masked_parent(observation, blocked)
        if alternative is not None:
            self.stats["v29_pre_ready_blast_hold"] += 1
            return alternative
        return chosen

    def _high_confidence_use(self, obs, damage):
        state = obs.current
        me = self._int(state.yourIndex)
        opponent = state.players[1 - me]
        if len(opponent.prize or []) <= 1:
            return False
        plan = self._best_target(obs, damage)
        if plan is None:
            return False
        _, detail = plan
        return bool(
            detail.get("exact")
            or detail.get("wall")
            or self._int(detail.get("prize", 1), 1) >= 2
        )

    def _target_option(self, obs, serial):
        for index, option in enumerate(list(obs.select.option or [])):
            card = self._src(obs, option)
            if card is not None and self._int(getattr(card, "serial", -2), -2) == serial:
                return index
            if self._int(getattr(option, "serial", -2), -2) == serial:
                return index
        return None

    def patch(self, observation, chosen):
        try:
            if not observation.get("select"):
                return chosen
            obs = self.api.to_observation_class(observation)
            select = obs.select
            if obs.current is None or select is None:
                return chosen

            chosen = self._main_exact_combo(obs, chosen)
            chosen = self._v28_exact_active_combo(obs, chosen)
            chosen = self._v29_judge_gate(obs, chosen)

            # The hold rule was positive against Archaludon/Lucario/Marnie but
            # negative when Alakazam can move counters or an ex-immunity wall
            # demands immediate non-attack pressure.  Resolve the parent pick
            # first so matchup recognition runs only on the rare Dusk Ability
            # decision, not on every callback.  Target repair remains on.
            main_pick = self._selected_main_dusk(obs, chosen)
            if main_pick is not None and not self._hold_blocked(obs):
                _, source_id, source = main_pick
                options = list(select.option or [])
                if source_id == DUSCLOPS:
                    dusknoir_evolves = [
                        index
                        for index, option in enumerate(options)
                        if option.type == self.api.OptionType.EVOLVE
                        and self._cid(obs, option) == DUSKNOIR
                    ]
                    if dusknoir_evolves:
                        self.stats["prefer_dusknoir_evolve"] += 1
                        return [dusknoir_evolves[0]]
                    fresh = bool(getattr(source, "appearThisTurn", False))
                    if fresh and not self._high_confidence_use(obs, 50):
                        blocked = [
                            index
                            for index, option in enumerate(options)
                            if option.type == self.api.OptionType.ABILITY
                            and self._cid(obs, option) == DUSCLOPS
                        ]
                        alternative = self._masked_parent(observation, blocked)
                        if alternative is not None:
                            self.stats["hold_fresh_dusclops"] += 1
                            return alternative

            effect = self._int(getattr(getattr(select, "effect", None), "id", 0), 0)
            if self._int(select.context) == 14 and effect == DRAGAPULT:
                return self._phantom_marnie_threshold_v28(obs, chosen)
            if self._int(select.context) != DAMAGE_COUNTER or effect not in (DUSCLOPS, DUSKNOIR):
                self.pending_phantom_serial = -1
                return chosen

            turn = self._int(obs.current.turn, -1)
            if (
                self.pending_effect == effect
                and self.pending_turn == turn
                and self.pending_target_serial >= 0
            ):
                target_index = self._target_option(obs, self.pending_target_serial)
                if target_index is not None:
                    self.stats["exact_bridge_target_contract"] += 1
                    self.pending_target_serial = -1
                    self.pending_effect = 0
                    self.pending_attack = True
                    return [target_index]
                self.stats["exact_bridge_target_missing"] += 1
                self._clear_contract()

            damage = 50 if effect == DUSCLOPS else 130
            plan = self._best_target(obs, damage)
            if plan is None:
                self.stats["target_no_high_confidence"] += 1
                return chosen
            target, detail = plan
            target_index = self._target_option(obs, self._int(getattr(target, "serial", -1), -1))
            if target_index is None:
                self.stats["target_not_legal"] += 1
                return chosen
            old = chosen[0] if isinstance(chosen, list) and len(chosen) == 1 else -1
            if old == target_index:
                self.stats["target_already_best"] += 1
                return chosen
            self.stats["target_override"] += 1
            self.stats["target_direct" if detail["direct"] else "target_bridge"] += 1
            if detail["wall"]:
                self.stats["target_wall"] += 1
            if self._int(detail["prize"], 1) >= 2:
                self.stats["target_multi_prize"] += 1
            return [target_index]
        except Exception:
            self.stats["exceptions"] += 1
            return chosen
