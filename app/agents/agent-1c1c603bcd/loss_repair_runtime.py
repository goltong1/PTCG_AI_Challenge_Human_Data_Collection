"""Deterministic repairs distilled from the 2026-08-15 human loss set.

This module is deliberately small and authoritative.  It does not replace the
history/temporal policy; it prevents a handful of objectively losing actions
that survived every learned layer:
  * fake Boss KOs caused by ignoring resistance,
  * feeding a 3-Prize Mega Lucario to a powered Alakazam,
  * abandoning a live Cornerstone ability wall,
  * spending the last search slots on Dudunsparce while no attacker survives,
  * taking the wrong Hilda evolution when Mega Lucario is the immediate bridge.
"""
from __future__ import annotations

from typing import Any, Optional

from cg.api import AreaType, OptionType, SelectContext, all_card_data, to_observation_class
from closeout_runtime import attack_damage, prize_value

CARD = {c.cardId: c for c in all_card_data()}

# Own deck / engine.
RIOLU70 = 333
RIOLU80 = 677
LUCARIO = 678
SOLROCK = 676
LUNATONE = 675
DUNSPARCE = 305
DUDUN = 66
DUDUN_EX = 306
OGERPON = 117
PPP = 1141
POKE_PAD = 1152
POFFIN = 1086
JUDGE = 1213
XEROSIC = 1197
HILDA = 1225
BASIC_F = 6
ROCK_F = 20

# Opponent signatures / priority pieces.
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
ALAKAZAM_ALT = 245
DURALUDON = 169
ARCHALUDON = 190
CINDERACE = 666
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL = 648
FROSLASS = 104
MUNKIDORI = 112

# Attacks.
COSMIC = 980
AURA_JAB = 982
MEGA_BRAVE = 983
TENACIOUS = 425
DRILL = 426
DEMOLISH = 148


def _field(player):
    return [q for q in list(player.active or []) + list(player.bench or []) if q is not None]


def _card(obs, area, index, player):
    try:
        area = AreaType(int(area))
        p = obs.current.players[player]
        if area == AreaType.DECK:
            return (obs.select.deck or [])[index]
        if area == AreaType.HAND:
            return (p.hand or [])[index]
        if area == AreaType.DISCARD:
            return (p.discard or [])[index]
        if area == AreaType.ACTIVE:
            return (p.active or [])[index]
        if area == AreaType.BENCH:
            return (p.bench or [])[index]
        if area == AreaType.PRIZE:
            return (p.prize or [])[index]
        if area == AreaType.LOOKING:
            return (obs.current.looking or [])[index]
        if area == AreaType.STADIUM:
            return (obs.current.stadium or [])[index]
    except Exception:
        return None
    return None


def _has_ability(pokemon) -> bool:
    c = CARD.get(getattr(pokemon, "id", -1))
    return bool(c and getattr(c, "skills", None))


def _has_rock(pokemon) -> bool:
    return bool(
        pokemon is not None
        and any(getattr(e, "id", None) == ROCK_F for e in (pokemon.energyCards or []))
    )


def _effect_id(obs) -> int:
    try:
        return int(getattr(getattr(obs.select, "effect", None), "id", -1))
    except Exception:
        return -1


def _is_single(base) -> bool:
    return bool(base and len(base) == 1 and isinstance(base[0], int))


class ReplayLossRepairGate:
    """Final, legal-action-only guard for the audited loss patterns."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = {"calls": 0, "overrides": {}}
        self.turn = -1
        self.ppp_used = 0
        self.ogerpon_unavailable = False

    def _note(self, key: str):
        d = self.stats["overrides"]
        d[key] = d.get(key, 0) + 1

    def _emit(self, action, reason: Optional[str] = None):
        if reason:
            self._note(reason)
        return action

    def _track_turn_resources(self, obs, me):
        turn = int(obs.current.turn or 0)
        if turn != self.turn:
            self.turn = turn
            self.ppp_used = 0
        # CABT observations expose only logs since the immediately preceding
        # selection, so counting a newly logged PPP here does not double count.
        try:
            for lg in (obs.logs or []):
                if getattr(lg, "playerIndex", None) == me and getattr(lg, "cardId", None) == PPP:
                    self.ppp_used += 1
        except Exception:
            pass

    def _candidate_attacks(self, attacker, mine, op):
        if attacker is None:
            return []
        en = len(attacker.energies or [])
        if attacker.id == LUCARIO:
            out = []
            if en >= 1:
                out.append(AURA_JAB)
            if en >= 2:
                out.append(MEGA_BRAVE)
            return out
        if attacker.id == SOLROCK and en >= 1 and any(q.id == LUNATONE for q in _field(mine)):
            return [COSMIC]
        if attacker.id == DUDUN_EX:
            out = [TENACIOUS] if en >= 1 else []
            if en >= 3:
                out.append(DRILL)
            return out
        if attacker.id == OGERPON and en >= 3:
            return [DEMOLISH]
        return []

    def _max_damage(self, attacker, mine, op, target) -> int:
        best = 0
        for aid in self._candidate_attacks(attacker, mine, op):
            try:
                best = max(best, int(attack_damage(attacker, aid, op, target, ppp=self.ppp_used)))
            except Exception:
                pass
        return best

    def _selected_direct_mate(self, obs, base, mine, op) -> bool:
        if not _is_single(base) or obs.select.context != SelectContext.MAIN or not mine.active or not op.active:
            return False
        try:
            o = list(obs.select.option or [])[base[0]]
            if o.type != OptionType.ATTACK:
                return False
            dmg = int(attack_damage(mine.active[0], o.attackId, op, op.active[0], ppp=self.ppp_used))
            return dmg >= int(op.active[0].hp or 0) and prize_value(op.active[0]) >= len(mine.prize or [])
        except Exception:
            return False

    def _next_turn_mate_from(self, candidate, mine, op) -> bool:
        if candidate is None or not op.active:
            return False
        target = op.active[0]
        if prize_value(target) < len(mine.prize or []):
            return False
        return self._max_damage(candidate, mine, op, target) >= int(target.hp or 0)

    def _pick_low_prize_shield(self, obs, opts, me, mine, op):
        candidates = []
        for i, o in enumerate(opts):
            q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
            if q is None:
                continue
            pr = prize_value(q)
            if pr >= len(op.prize or []):
                continue
            en = len(q.energies or [])
            attack_value = 0
            if q.id == SOLROCK and en >= 1 and any(x.id == LUNATONE for x in _field(mine)):
                attack_value = 500
            elif q.id in {RIOLU70, RIOLU80} and en >= 1:
                attack_value = 420
            elif q.id == DUNSPARCE and en >= 1:
                attack_value = 300
            elif q.id == LUNATONE:
                attack_value = 180
            # Lowest prize liability first, then useful tempo and survivability.
            candidates.append((-pr, attack_value, int(q.hp or 0), -i, i))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][-1]

    def _wall_mode(self, plan) -> bool:
        return getattr(plan, "strategy", "") in {
            "ABILITY_WALL",
            "MARNIE_ABILITY_WALL",
            "TEAL_ABILITY_WALL",
        }

    def choose(self, obs_dict: dict, base: list[int], plan: Any = None) -> list[int]:
        self.stats["calls"] += 1
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            return base
        if obs.current is None or obs.select is None:
            return base
        opts = list(obs.select.option or [])
        if not opts:
            return base
        me = obs.current.yourIndex
        mine = obs.current.players[me]
        op = obs.current.players[1 - me]
        ctx = obs.select.context
        archetype = getattr(plan, "archetype", "unknown")
        strategy = getattr(plan, "strategy", "")
        self._track_turn_resources(obs, me)

        # A proven immediate win outranks every defensive repair below.
        if self._selected_direct_mate(obs, base, mine, op):
            return base

        # 1) HILDA EVOLUTION BRIDGE -----------------------------------------
        # 93149894 searched Dudunsparce while an Energy-ready Riolu and Mega
        # Lucario were both public.  Only force Mega when no copy is already in
        # hand; this keeps the repair narrow and avoids redundant evolution cards.
        if ctx == SelectContext.TO_HAND and _effect_id(obs) == HILDA:
            riolu_live = [q for q in _field(mine) if q.id in {RIOLU70, RIOLU80}]
            # The audited mirror error had a mature, Energy-ready Active Riolu.
            # Do not turn every generic Hilda search into Mega Lucario; require a
            # line that can materially use the evolution on the next action/turn.
            ready_riolu = any(len(q.energies or []) >= 1 and not bool(getattr(q, "appearThisTurn", False)) for q in riolu_live)
            mega_in_hand = any(c.id == LUCARIO for c in (mine.hand or []))
            crustle_dudun_commit = strategy == "BYPASS_CRUSTLE" and getattr(plan, "primary", -1) == DUDUN_EX
            if ready_riolu and not mega_in_hand and not crustle_dudun_commit:
                mega = []
                for i, o in enumerate(opts):
                    q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                    if q is not None and q.id == LUCARIO:
                        mega.append(i)
                if mega:
                    return self._emit([mega[0]], "loss93149894:hilda_search_mega")

        # 2) ATTACKER CONTINUITY IN SEARCHES -------------------------------
        # Poké Pad is the only clean way to find the 80-HP Riolu.  In 93158333
        # both Pad searches became Dudunsparce while the board had only one
        # attacker; the active Riolu was KO'd and the game ended by bench-out.
        if ctx == SelectContext.TO_HAND and _effect_id(obs) == POKE_PAD and strategy != "BYPASS_CRUSTLE":
            lines = sum(1 for q in _field(mine) if q.id in {RIOLU70, RIOLU80, LUCARIO})
            hand_has_riolu = any(c.id in {RIOLU70, RIOLU80} for c in (mine.hand or []))
            bench_room = int(getattr(mine, "benchMax", 5) or 5) - len([q for q in (mine.bench or []) if q is not None])
            own_field = _field(mine)
            mirror_emergency = bool(
                archetype == "lucario"
                and int(obs.current.turn or 0) <= 4
                and len(own_field) == 2
                and mine.active
                and mine.active[0].id in {RIOLU70, RIOLU80}
                and len([q for q in (mine.bench or []) if q is not None]) == 1
                and mine.bench[0].id == DUNSPARCE
                and op.active
                and op.active[0].id in {RIOLU70, RIOLU80, LUCARIO}
                and len(op.active[0].energies or []) >= 1
            )
            if lines < 2 and not hand_has_riolu and bench_room > 0 and (archetype == "marnie" or mirror_emergency):
                riolu = []
                for i, o in enumerate(opts):
                    q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                    if q is not None and q.id in {RIOLU80, RIOLU70}:
                        riolu.append((1 if q.id == RIOLU80 else 0, -i, i))
                if riolu:
                    riolu.sort(reverse=True)
                    return self._emit([riolu[0][-1]], "loss93158333:pad_preserve_riolu")

        # Poffin's one remaining slot must be a Riolu when fewer than two lines
        # exist.  The old rescue required two duplicate Dunsparce and missed the
        # maxCount=1 state in 93154551.
        if ctx == SelectContext.TO_BENCH and _effect_id(obs) == POFFIN and strategy != "BYPASS_CRUSTLE":
            lines = sum(1 for q in _field(mine) if q.id in {RIOLU70, RIOLU80, LUCARIO})
            if lines < 2 and archetype != "crustle":
                riolu = []
                for i, o in enumerate(opts):
                    q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                    if q is not None and q.id == RIOLU70:
                        riolu.append(i)
                if riolu:
                    if int(obs.select.maxCount or 0) == 1:
                        return self._emit([riolu[0]], "loss93154551:poffin_one_slot_riolu")
                    chosen = list(base or [])
                    chosen_ids = []
                    for bi in chosen:
                        if 0 <= bi < len(opts):
                            q = _card(obs, opts[bi].area, opts[bi].index, getattr(opts[bi], "playerIndex", me))
                            chosen_ids.append((bi, getattr(q, "id", -1)))
                    if not any(cid == RIOLU70 for _, cid in chosen_ids):
                        replace = next((bi for bi, cid in reversed(chosen_ids) if cid == DUNSPARCE), None)
                        if replace is not None:
                            chosen.remove(replace)
                            chosen.append(riolu[0])
                            return self._emit(chosen, "loss_set:poffin_preserve_one_riolu")

        # 3) CORNERSTONE ABILITY-WALL COMMITMENT ---------------------------
        # Archaludon and Grimmsnarl both attack from Pokémon with Abilities.
        # Once Cornerstone is live, abandoning it is dominated unless another
        # action wins immediately (already checked above).
        if self._wall_mode(plan):
            oger = next((q for q in _field(mine) if q.id == OGERPON), None)
            hand_has_oger = any(c.id == OGERPON for c in (mine.hand or []))
            if oger is not None or hand_has_oger:
                self.ogerpon_unavailable = False
            opp_active = op.active[0] if op.active else None
            ability_threat = bool(opp_active is not None and _has_ability(opp_active))

            # Search and deploy the wall before generic engine cards consume Pad.
            if oger is None and not hand_has_oger and not self.ogerpon_unavailable:
                if ctx == SelectContext.MAIN:
                    for i, o in enumerate(opts):
                        if o.type != OptionType.PLAY:
                            continue
                        q = _card(obs, AreaType.HAND, o.index, me)
                        if q is not None and q.id == POKE_PAD:
                            return self._emit([i], "ability_wall:play_pad_for_ogerpon")
                if ctx == SelectContext.TO_HAND and _effect_id(obs) == POKE_PAD:
                    found = []
                    for i, o in enumerate(opts):
                        q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                        if q is not None and q.id == OGERPON:
                            found.append(i)
                    if found:
                        return self._emit([found[0]], "ability_wall:search_ogerpon")
                    # Poké Pad exposes every legal Pokémon in the deck.  If the
                    # one-copy Ogerpon is absent while it is neither in hand nor
                    # in play, do not burn the remaining Pads chasing a prized card.
                    self.ogerpon_unavailable = True

            if ctx == SelectContext.MAIN:
                # When the current Ability attacker needs multiple Demolish hits,
                # one Boss can remove the support that actually breaks the lock:
                # Froslass checkup damage, Munkidori counter movement, or Cinderace
                # acceleration.  Never divert when the current Active is already a KO.
                if mine.active and mine.active[0].id == OGERPON and ability_threat and not obs.current.supporterPlayed:
                    current_dmg = self._max_damage(mine.active[0], mine, op, opp_active)
                    support_ids = {FROSLASS, MUNKIDORI} if archetype == "marnie" else {CINDERACE} if archetype == "archaludon" else set()
                    support_ko = any(q.id in support_ids and self._max_damage(mine.active[0], mine, op, q) >= int(q.hp or 0) for q in (op.bench or []) if q is not None)
                    if current_dmg < int(opp_active.hp or 0) and support_ko:
                        for i, o in enumerate(opts):
                            if o.type != OptionType.PLAY:
                                continue
                            q = _card(obs, AreaType.HAND, o.index, me)
                            if q is not None and q.id == 1182:  # Boss's Orders
                                return self._emit([i], "ability_wall:boss_lock_breaker")

                # A legal Demolish into an Ability attacker is the anchor action.
                if mine.active and mine.active[0].id == OGERPON and ability_threat:
                    for i, o in enumerate(opts):
                        if o.type == OptionType.ATTACK and o.attackId == DEMOLISH:
                            return self._emit([i], "loss93155500:keep_wall_demolish")

                if oger is None and hand_has_oger:
                    for i, o in enumerate(opts):
                        if o.type != OptionType.PLAY:
                            continue
                        q = _card(obs, AreaType.HAND, o.index, me)
                        if q is not None and q.id == OGERPON:
                            return self._emit([i], "ability_wall:bench_ogerpon")

                if oger is not None and len(oger.energies or []) < 3 and not obs.current.energyAttached:
                    attach = []
                    for i, o in enumerate(opts):
                        if o.type != OptionType.ATTACH:
                            continue
                        target = _card(obs, o.inPlayArea, o.inPlayIndex, me)
                        energy = _card(obs, AreaType.HAND, o.index, me)
                        if target is not None and energy is not None and target.serial == oger.serial and energy.id in {BASIC_F, ROCK_F}:
                            attach.append((1 if energy.id == BASIC_F else 0, -i, i))
                    if attach:
                        attach.sort(reverse=True)
                        return self._emit([attach[0][-1]], "ability_wall:charge_ogerpon")

                # Promote a ready wall only when it actually locks the current
                # Ability attacker and the selected action is not a direct KO.
                if oger is not None and len(oger.energies or []) >= 3 and ability_threat and mine.active and mine.active[0].serial != oger.serial:
                    active_dmg = self._max_damage(mine.active[0], mine, op, opp_active)
                    if active_dmg < int(opp_active.hp or 0):
                        for i, o in enumerate(opts):
                            if o.type == OptionType.RETREAT:
                                return self._emit([i], "ability_wall:retreat_to_ready_ogerpon")

                # Never walk out of a live damage lock merely because the frozen
                # executor prefers a utility retreat.  If Demolish is temporarily
                # unavailable, hold the wall rather than feed a prize.
                if mine.active and mine.active[0].id == OGERPON and ability_threat and _is_single(base):
                    try:
                        bo = opts[base[0]]
                        if bo.type == OptionType.RETREAT:
                            end = next((i for i, o in enumerate(opts) if o.type == OptionType.END), None)
                            if end is not None:
                                return self._emit([end], "ability_wall:block_abandonment")
                    except Exception:
                        pass

            # Aura Jab target selection: finish the 3-Energy wall first, then
            # preserve the next Lucario line rather than overcharging Dudunsparce ex.
            if ctx == SelectContext.ATTACH_FROM:
                if oger is not None and len(oger.energies or []) < 3:
                    for i, o in enumerate(opts):
                        q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                        if q is not None and q.serial == oger.serial:
                            return self._emit([i], "ability_wall:aura_to_ogerpon")
                lucario_targets = []
                for i, o in enumerate(opts):
                    q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                    if q is not None and q.id in {RIOLU70, RIOLU80, LUCARIO} and len(q.energies or []) < 2:
                        lucario_targets.append((1 if q.id == LUCARIO else 0, -len(q.energies or []), -i, i))
                if lucario_targets:
                    lucario_targets.sort(reverse=True)
                    return self._emit([lucario_targets[0][-1]], "ability_wall:aura_to_next_lucario")

            if ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE} and oger is not None and len(oger.energies or []) >= 3 and ability_threat:
                for i, o in enumerate(opts):
                    q = _card(obs, o.area, o.index, getattr(o, "playerIndex", me))
                    if q is not None and q.serial == oger.serial:
                        return self._emit([i], "ability_wall:promote_ogerpon")

        # 4) ALAKAZAM PRIZE-LIABILITY SHIELD -------------------------------
        # Generalize the old exact-three rule to <=3 prizes and project modest
        # next-turn hand growth.  Rock Energy is not treated as unconditionally
        # safe in a 15+ card hand because the archetype publicly plays Enhanced
        # Hammer and can remove the protection before Powerful Hand.
        powered_alakazam = bool(
            archetype == "alakazam"
            and op.active
            and op.active[0].id in {ALAKAZAM, ALAKAZAM_ALT}
            and len(op.active[0].energies or []) >= 1
        )
        if powered_alakazam and 0 < len(op.prize or []) <= 3:
            projected = 20 * (int(op.handCount or 0) + 4)

            if ctx == SelectContext.TO_ACTIVE and _is_single(base):
                bo = opts[base[0]]
                chosen = _card(obs, bo.area, bo.index, getattr(bo, "playerIndex", me))
                if chosen is not None and not self._next_turn_mate_from(chosen, mine, op):
                    immediate_loss = prize_value(chosen) >= len(op.prize or [])
                    effect_lethal = projected >= int(chosen.hp or 0)
                    hammer_risk = _has_rock(chosen) and int(op.handCount or 0) >= 8
                    if immediate_loss and (effect_lethal or hammer_risk):
                        shield = self._pick_low_prize_shield(obs, opts, me, mine, op)
                        if shield is not None and shield != base[0]:
                            return self._emit([shield], "loss_set:alakazam_low_prize_promotion")

            # 93161917 correctly promoted a one-Prize shield, then immediately
            # retreated it into Mega Brave for only one Prize and lost all three.
            if ctx == SelectContext.MAIN and mine.active and prize_value(mine.active[0]) == 1 and _is_single(base):
                bo = opts[base[0]]
                if bo.type == OptionType.RETREAT:
                    mate_bench = any(self._next_turn_mate_from(q, mine, op) for q in (mine.bench or []) if q is not None)
                    if not mate_bench:
                        attacks = []
                        for i, o in enumerate(opts):
                            if o.type == OptionType.ATTACK:
                                try:
                                    dmg = int(attack_damage(mine.active[0], o.attackId, op, op.active[0], ppp=self.ppp_used))
                                except Exception:
                                    dmg = 0
                                attacks.append((dmg, -i, i))
                        if attacks:
                            attacks.sort(reverse=True)
                            return self._emit([attacks[0][-1]], "loss93161917:block_mega_feed_attack")
                        end = next((i for i, o in enumerate(opts) if o.type == OptionType.END), None)
                        if end is not None:
                            return self._emit([end], "loss93161917:block_mega_feed_end")

            # When Powerful Hand is online and the hand is already lethal-sized,
            # compress it before ordinary development unless a direct mate exists.
            if ctx == SelectContext.MAIN and not obs.current.supporterPlayed and int(op.handCount or 0) >= 7:
                denial = []
                for i, o in enumerate(opts):
                    if o.type != OptionType.PLAY:
                        continue
                    q = _card(obs, AreaType.HAND, o.index, me)
                    if q is None:
                        continue
                    if q.id == XEROSIC:
                        denial.append((2, -i, i))
                    elif q.id == JUDGE:
                        denial.append((1, -i, i))
                if denial:
                    denial.sort(reverse=True)
                    return self._emit([denial[0][-1]], "alakazam:lethal_hand_denial")

        # 5) RESISTANCE-AWARE BOSS TARGETING -------------------------------
        # Target menus contain opponent cards only.  Use the same authoritative
        # damage routine as CloseoutPlanner, so Fighting resistance and bypass
        # attacks are applied before a target is called a KO.
        target_menu = ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE} and any(
            getattr(o, "playerIndex", me) == 1 - me for o in opts
        )
        if target_menu and mine.active and archetype in {"alakazam", "marnie", "archaludon"}:
            attacker = mine.active[0]
            scored = []
            fallback = []
            for i, o in enumerate(opts):
                if getattr(o, "playerIndex", me) != 1 - me:
                    continue
                q = _card(obs, o.area, o.index, 1 - me)
                if q is None:
                    continue
                dmg = self._max_damage(attacker, mine, op, q)
                bonus = 0
                if archetype == "alakazam":
                    bonus = 90 if q.id in {ALAKAZAM, ALAKAZAM_ALT} else 70 if q.id == KADABRA else 40 if q.id == ABRA else 0
                elif archetype == "marnie":
                    bonus = 100 if q.id == FROSLASS else 90 if q.id == MUNKIDORI else 80 if q.id == GRIMMSNARL else 65 if q.id == MORGREM else 35 if q.id == IMPIDIMP else 0
                elif archetype == "archaludon":
                    bonus = 100 if q.id == CINDERACE else 80 if q.id == ARCHALUDON else 60 if q.id == DURALUDON else 0
                fallback.append((bonus, -int(q.hp or 0), -i, i))
                if dmg >= int(q.hp or 0):
                    scored.append((prize_value(q) * 100 + bonus, -max(0, dmg - int(q.hp or 0)), -i, i))
            if scored:
                scored.sort(reverse=True)
                chosen = scored[0][-1]
                if not _is_single(base) or chosen != base[0]:
                    return self._emit([chosen], f"loss_set:{archetype}_effective_ko_target")
            elif fallback and archetype in {"marnie", "archaludon"}:
                # When no KO exists, deny the most dangerous engine piece rather
                # than selecting a low-value target by raw card order.
                fallback.sort(reverse=True)
                chosen = fallback[0][-1]
                if not _is_single(base) or chosen != base[0]:
                    return self._emit([chosen], f"loss_set:{archetype}_engine_target")

        return base

    def get_stats(self):
        return {"calls": int(self.stats["calls"]), "overrides": dict(self.stats["overrides"])}
