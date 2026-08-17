"""History-conditioned tactical planner for the Dragapult submission.

The large search model is deliberately distilled into conservative semantic
invariants.  The planner consumes only public state, public action history and
our exact selected-action intents.  It never reads hidden opponent cards.
"""
from __future__ import annotations

from collections import Counter


class TacticalBeliefPlanner:
    # Own deck cards.
    MUNK = 112
    DRAK = 120
    DRAG = 121
    PFOFFIN = 1086
    ULTRA = 1121
    POKEPAD = 1152
    BOSS = 1182
    FIRE = 2
    PSYCHIC = 5
    DARK = 7

    # Public matchup signatures.
    DWEBBLE = 344
    CRUSTLE = 345
    ARCH_FAMILY = frozenset((169, 190, 666))
    LUCARIO_FAMILY = frozenset((333, 677, 678, 675, 676))
    MARNIE_FAMILY = frozenset((646, 647, 648))

    ATTACK_MIND_BEND = 141
    ATTACK_DRAK = 152
    ATTACK_PHANTOM = 154
    DRAG_LINE = frozenset((119, 120, 121))

    def __init__(self, api, controller, base, history):
        self.api = api
        self.controller = controller
        self.base = base
        self.history = history
        self.stats = Counter()
        self._semantic_wall_cache = {}
        self.reset()

    @staticmethod
    def _int(value, default=0):
        try:
            return int(value) if value is not None else default
        except Exception:
            return default

    def reset(self):
        self.last_turn = -1
        self.wall_seen = False
        self.pending_wall_pivot_serial = None
        self.pending_wall_pivot_turn = None
        self.phantom_into_wall_streak = 0
        self.family_scores = Counter()
        self.stats.clear()

    def _bump(self, key):
        self.stats[str(key)] += 1

    def _board(self, player):
        return [p for p in list(player.active or []) + list(player.bench or []) if p is not None]

    def _visible(self, player):
        cards = [p for p in list(player.active or []) + list(player.bench or []) + list(player.discard or []) if p is not None]
        out = set()
        for p in cards:
            out.add(self._int(getattr(p, "id", 0), 0))
            for q in list(getattr(p, "preEvolution", None) or []):
                out.add(self._int(getattr(q, "id", 0), 0))
        return out

    def _is_ex_damage_wall_id(self, card_id):
        """Read bundled card text and recognize ex-only damage immunity.

        This intentionally uses only public card IDs and the competition's
        local card database.  It generalizes the Crustle route to a previously
        unseen card whose printed Ability uses the same damage-prevention
        semantics; no opponent hand/deck information is consulted.
        """
        card_id = self._int(card_id, 0)
        if card_id in self._semantic_wall_cache:
            return self._semantic_wall_cache[card_id]
        card = getattr(self.base, "card_table", {}).get(card_id)
        chunks = []
        if card is not None:
            for skill in list(getattr(card, "skills", None) or []):
                chunks.append(str(getattr(skill, "name", "") or ""))
                chunks.append(str(getattr(skill, "text", "") or ""))
        text = " ".join(chunks).lower().replace("pokémon", "pokemon").replace("’", "'")
        is_wall = (
            "prevent all damage" in text
            and "attacks" in text
            and ("pokemon {ex}" in text or "pokemon ex" in text)
        )
        self._semantic_wall_cache[card_id] = bool(is_wall)
        return bool(is_wall)

    def _update_belief(self, opponent):
        visible = self._visible(opponent)
        if visible & {self.DWEBBLE, self.CRUSTLE} or any(self._is_ex_damage_wall_id(cid) for cid in visible):
            self.family_scores["crustle"] += 3
            self.wall_seen = True
        if visible & self.ARCH_FAMILY:
            self.family_scores["archaludon"] += 2
        if visible & self.LUCARIO_FAMILY:
            self.family_scores["lucario"] += 2
        if visible & self.MARNIE_FAMILY:
            self.family_scores["marnie"] += 2
        return visible

    def _family(self):
        if not self.family_scores:
            return None
        return max(self.family_scores, key=lambda k: self.family_scores[k])

    def _wall_active(self, opponent):
        if not opponent.active or opponent.active[0] is None:
            return False
        card_id = self._int(opponent.active[0].id, 0)
        return bool(card_id == self.CRUSTLE or self._is_ex_damage_wall_id(card_id))

    def _eids(self, pokemon):
        return {self._int(getattr(e, "id", 0), 0) for e in list(getattr(pokemon, "energyCards", None) or [])}

    def _ready_munk(self, pokemon):
        return self._int(getattr(pokemon, "id", 0), 0) == self.MUNK and {self.DARK, self.PSYCHIC}.issubset(self._eids(pokemon))

    def _ready_drak(self, pokemon):
        return self._int(getattr(pokemon, "id", 0), 0) == self.DRAK and {self.FIRE, self.PSYCHIC}.issubset(self._eids(pokemon))

    def _is_breaker(self, pokemon):
        return self._ready_munk(pokemon) or self._ready_drak(pokemon)

    def _best_breaker(self, mine):
        board = self._board(mine)
        # Mind Bend is preferred because confusion can buy the second hit.
        for p in board:
            if self._ready_munk(p):
                return p
        for p in board:
            if self._ready_drak(p):
                return p
        return None

    def _source(self, obs, option):
        try:
            return self.controller._source_card(obs, option)
        except Exception:
            return None

    def _cid(self, obs, option):
        source = self._source(obs, option)
        return self._int(getattr(source, "id", getattr(option, "cardId", 0)), 0)

    def _target(self, obs, option):
        try:
            state = obs.current
            mine = state.players[state.yourIndex]
            area = self._int(getattr(option, "inPlayArea", -1), -1)
            index = self._int(getattr(option, "inPlayIndex", -1), -1)
            cards = list(mine.active or []) if area == 4 else list(mine.bench or []) if area == 5 else []
            return cards[index] if 0 <= index < len(cards) else None
        except Exception:
            return None

    def _find(self, obs, options, option_type=None, card_id=None, attack_id=None, target_serial=None):
        for index, option in enumerate(options):
            if option_type is not None and option.type != option_type:
                continue
            if card_id is not None and self._cid(obs, option) != card_id:
                continue
            if attack_id is not None and self._int(getattr(option, "attackId", 0), 0) != attack_id:
                continue
            if target_serial is not None:
                target = self._target(obs, option)
                if target is None or self._int(getattr(target, "serial", -1), -1) != self._int(target_serial, -2):
                    continue
            return index
        return None

    def _choose_card_serial(self, obs, options, serial):
        if serial is None:
            return None
        for index, option in enumerate(options):
            source = self._source(obs, option)
            if source is not None and self._int(getattr(source, "serial", -1), -1) == self._int(serial, -2):
                return index
        return None

    def _productive_damage_to_move(self, mine, opponent):
        # Adrena-Brain is useful only if at least one own counter exists and a
        # live opponent target can receive it.
        damaged = any(self._int(getattr(p, "hp", 0), 0) < self._int(getattr(p, "maxHp", 0), 0) for p in self._board(mine))
        live_target = any(self._int(getattr(p, "hp", 0), 0) > 0 for p in self._board(opponent))
        return damaged and live_target

    def _bench_counter_prizes_now(self, opponent):
        # Conservative lower bound: count only already-KO-range non-rule-box
        # bodies that Phantom's six counters can certainly remove.  This is
        # used solely to avoid suppressing an immediate prize finish.
        prizes = 0
        for p in list(opponent.bench or []):
            if p is None:
                continue
            hp = self._int(getattr(p, "hp", 0), 0)
            if hp <= 60:
                card = getattr(self.base, "card_table", {}).get(self._int(getattr(p, "id", 0), 0))
                prizes += 2 if bool(getattr(card, "ex", False)) else 1
        return prizes

    def _safe_fallback(self, obs, options, forbidden_index=None, exclude_evolve_serial=None):
        T = self.api.OptionType
        # Prefer actions that preserve information and board development.
        order = (T.ABILITY, T.PLAY, T.ATTACH, T.EVOLVE, T.END)
        for wanted_type in order:
            for index, option in enumerate(options):
                if index == forbidden_index or option.type != wanted_type:
                    continue
                if wanted_type == T.EVOLVE and exclude_evolve_serial is not None:
                    target = self._target(obs, option)
                    if target is not None and self._int(target.serial, -1) == self._int(exclude_evolve_serial, -2):
                        continue
                return index
        return None

    def _replace_or_append(self, chosen, index, select):
        out = [x for x in list(chosen or []) if isinstance(x, int)]
        if index not in out:
            if len(out) < self._int(select.maxCount, 1):
                out.append(index)
            elif out:
                out[-1] = index
            else:
                out = [index]
        dedup = []
        for x in out:
            if x not in dedup:
                dedup.append(x)
        return dedup[: self._int(select.maxCount, 1)]


    def patch(self, observation, chosen):
        try:
            if not observation.get("select") or not isinstance(chosen, list):
                return chosen
            obs = self.api.to_observation_class(observation)
            select = obs.select
            state = obs.current
            if state is None or select is None:
                return chosen
            turn = self._int(state.turn, 0)
            if turn < self.last_turn:
                self.reset()
            self.last_turn = turn
            me = state.yourIndex
            mine = state.players[me]
            opponent = state.players[1 - me]
            self._update_belief(opponent)
            wall_active = self._wall_active(opponent)
            options = list(select.option or [])
            board = self._board(mine)
            active = mine.active[0] if mine.active and mine.active[0] is not None else None
            munks = [p for p in board if self._int(p.id, 0) == self.MUNK]
            draks = [p for p in board if self._int(p.id, 0) == self.DRAK]
            ready_breaker = self._best_breaker(mine)
            T = self.api.OptionType
            C = self.api.SelectContext

            # Complete an announced retreat/promotion with the exact semantic
            # target; this runs after v21's stateless guards.
            if select.context in (C.SWITCH, C.TO_ACTIVE):
                wanted = self.pending_wall_pivot_serial
                if wanted is None and wall_active and ready_breaker is not None:
                    wanted = self._int(ready_breaker.serial, -1)
                index = self._choose_card_serial(obs, options, wanted)
                if index is not None:
                    self.pending_wall_pivot_serial = None
                    self.pending_wall_pivot_turn = None
                    self._bump("wall_pivot_target")
                    return [index]

            # Search resolution is history-conditioned: only divert to Munk
            # once a Dragapult line is already represented on board.
            if select.context in (C.TO_HAND, C.TO_BENCH):
                effect = self._int(getattr(getattr(select, "effect", None), "id", 0), 0)
                line_count = sum(self._int(p.id, 0) in self.DRAG_LINE for p in board)
                if self.wall_seen and effect in (self.ULTRA, self.POKEPAD) and line_count >= 1 and not munks:
                    index = self._find(obs, options, card_id=self.MUNK)
                    if index is not None:
                        self._bump("wall_search_munk")
                        if self._int(select.maxCount, 1) == 1:
                            return [index]
                        return self._replace_or_append(chosen, index, select)

            if select.context != C.MAIN or self._int(select.minCount, 0) != 1 or self._int(select.maxCount, 0) != 1:
                return chosen

            # A powered non-ex wall breaker must cash its attack.
            if wall_active and active is not None and self._ready_munk(active):
                ability = self._find(obs, options, option_type=T.ABILITY, card_id=self.MUNK)
                if ability is not None and self._productive_damage_to_move(mine, opponent):
                    self._bump("wall_munk_ability")
                    return [ability]
                attack = self._find(obs, options, option_type=T.ATTACK, attack_id=self.ATTACK_MIND_BEND)
                if attack is not None:
                    self._bump("wall_mind_bend")
                    return [attack]
            if wall_active and active is not None and self._ready_drak(active):
                ability = self._find(obs, options, option_type=T.ABILITY, card_id=self.DRAK)
                if ability is not None:
                    self._bump("wall_drak_draw")
                    return [ability]
                attack = self._find(obs, options, option_type=T.ATTACK, attack_id=self.ATTACK_DRAK)
                if attack is not None:
                    self._bump("wall_drak_attack")
                    return [attack]

            # Transition immediately once a breaker is ready.
            if wall_active and ready_breaker is not None and (active is None or self._int(active.serial, -1) != self._int(ready_breaker.serial, -2)):
                retreat = self._find(obs, options, option_type=T.RETREAT)
                if retreat is not None:
                    self.pending_wall_pivot_serial = self._int(ready_breaker.serial, -1)
                    self.pending_wall_pivot_turn = turn
                    try:
                        self.history.set_retreat_intent(self.pending_wall_pivot_serial)
                    except Exception:
                        pass
                    self._bump("wall_retreat_to_breaker")
                    return [retreat]

            # Do not evolve away the last attack-ready Drakloak until another
            # non-ex attacker is actually capable of damaging Crustle.
            if chosen and 0 <= chosen[0] < len(options):
                picked = options[chosen[0]]
                if picked.type == T.EVOLVE and self._cid(obs, picked) == self.DRAG:
                    target = self._target(obs, picked)
                    other_ready = any(self._is_breaker(p) and (target is None or self._int(p.serial, -1) != self._int(target.serial, -2)) for p in board)
                    if wall_active and target is not None and self._ready_drak(target) and not other_ready:
                        alt = self._safe_fallback(obs, options, forbidden_index=chosen[0], exclude_evolve_serial=self._int(target.serial, -1))
                        if alt is not None:
                            self._bump("preserve_last_drak")
                            return [alt]

            # Establish exactly one Munkidori and complete Darkness+Psychic;
            # Drakloak Fire+Psychic is the fallback breaker.
            occupied = len([p for p in list(mine.bench or []) if p is not None])
            if wall_active and not munks and occupied < 5:
                play_munk = self._find(obs, options, option_type=T.PLAY, card_id=self.MUNK)
                if play_munk is not None:
                    self._bump("wall_bench_munk")
                    return [play_munk]
            if wall_active:
                for pokemon in sorted(munks, key=lambda p: self._int(p.serial, 0)):
                    eids = self._eids(pokemon)
                    for energy_id in (self.DARK, self.PSYCHIC):
                        if energy_id in eids:
                            continue
                        attach = self._find(obs, options, option_type=T.ATTACH, card_id=energy_id, target_serial=self._int(pokemon.serial, -1))
                        if attach is not None:
                            self._bump("wall_attach_munk")
                            return [attach]
                for pokemon in sorted(draks, key=lambda p: self._int(p.serial, 0)):
                    eids = self._eids(pokemon)
                    for energy_id in (self.FIRE, self.PSYCHIC):
                        if energy_id in eids:
                            continue
                        attach = self._find(obs, options, option_type=T.ATTACH, card_id=energy_id, target_serial=self._int(pokemon.serial, -1))
                        if attach is not None:
                            self._bump("wall_attach_drak")
                            return [attach]

            # Prevent zero-active-damage Phantom loops unless its counters win
            # immediately or no concrete breaker-progress action exists.  A
            # public Dwebble on the Bench is an especially strong gust target:
            # Boss + Phantom converts a zero-damage attack into a guaranteed
            # one-Prize KO while still placing six counters.
            if chosen and 0 <= chosen[0] < len(options):
                picked = options[chosen[0]]
                if wall_active and picked.type == T.ATTACK and self._int(getattr(picked, "attackId", 0), 0) == self.ATTACK_PHANTOM:
                    immediate = self._bench_counter_prizes_now(opponent)
                    prizes_needed = len(list(mine.prize or []))
                    if immediate < prizes_needed:
                        boss = self._find(obs, options, option_type=T.PLAY, card_id=self.BOSS)
                        dwebble = next((p for p in list(opponent.bench or []) if p is not None and self._int(p.id, 0) == self.DWEBBLE and self._int(p.hp, 0) <= 200), None)
                        if boss is not None and dwebble is not None:
                            try:
                                self.history.set_gust_intent(self._int(dwebble.serial, -1))
                            except Exception:
                                pass
                            self._bump("wall_boss_dwebble")
                            return [boss]
                        alt = self._progress_option(obs, options, mine, ready_breaker)
                        if alt is not None:
                            self._bump("block_zero_damage_phantom")
                            return [alt]
            return chosen
        except Exception:
            self._bump("planner_exception")
            return chosen

    def _progress_option(self, obs, options, mine, ready_breaker):
        T = self.api.OptionType
        if ready_breaker is not None:
            index = self._find(obs, options, option_type=T.RETREAT)
            if index is not None:
                return index
        for index, option in enumerate(options):
            if option.type == T.ABILITY and self._cid(obs, option) in (self.MUNK, self.DRAK):
                return index
        for index, option in enumerate(options):
            if option.type == T.ATTACH:
                target = self._target(obs, option)
                if target is not None and self._int(target.id, 0) in (self.MUNK, self.DRAK):
                    return index
        for index, option in enumerate(options):
            if option.type == T.PLAY and self._cid(obs, option) in (self.MUNK, self.PFOFFIN, self.ULTRA, self.POKEPAD, 1198, 1227):
                return index
        for index, option in enumerate(options):
            if option.type == T.EVOLVE and self._cid(obs, option) == self.DRAK:
                return index
        return None

    def record(self, observation, action):
        try:
            if not observation.get("select") or not isinstance(action, list):
                return
            obs = self.api.to_observation_class(observation)
            select = obs.select
            if select is None or select.context != self.api.SelectContext.MAIN or len(action) != 1:
                return
            options = list(select.option or [])
            index = action[0]
            if not (0 <= index < len(options)):
                return
            option = options[index]
            state = obs.current
            if state is None:
                return
            opponent = state.players[1 - state.yourIndex]
            if option.type == self.api.OptionType.ATTACK:
                attack_id = self._int(getattr(option, "attackId", 0), 0)
                if self._wall_active(opponent) and attack_id == self.ATTACK_PHANTOM:
                    self.phantom_into_wall_streak += 1
                    self._bump("phantom_into_wall")
                elif attack_id in (self.ATTACK_MIND_BEND, self.ATTACK_DRAK):
                    self.phantom_into_wall_streak = 0
        except Exception:
            self._bump("record_exception")

    def get_stats(self):
        out = {str(k): int(v) for k, v in self.stats.items()}
        out.update({
            "wall_seen": int(self.wall_seen),
            "pending_wall_pivot": int(self.pending_wall_pivot_serial is not None),
            "phantom_into_wall_streak": int(self.phantom_into_wall_streak),
            "belief_crustle": int(self.family_scores.get("crustle", 0)),
            "belief_archaludon": int(self.family_scores.get("archaludon", 0)),
            "belief_lucario": int(self.family_scores.get("lucario", 0)),
            "belief_marnie": int(self.family_scores.get("marnie", 0)),
        })
        return out
