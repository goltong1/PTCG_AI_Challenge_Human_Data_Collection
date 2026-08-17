"""Text-conditioned safety and adaptation layer for Lucario v155.

This gate reads CardData.skills and Attack.text directly from CABT's local
catalogue.  It is intentionally authoritative only when the rules text proves a
relationship (for example, an ex-damage wall and a legal effect-bypassing attack).
All outputs are indices from the current legal option list.
"""
from __future__ import annotations

from typing import Any

from cg.api import AreaType, CardType, OptionType, SelectContext, all_attack, all_card_data, to_observation_class
import card_text_semantics as sem

CARD = {int(c.cardId): c for c in all_card_data()}
ATTACK = {int(a.attackId): a for a in all_attack()}

DUNSPARCE = 305
DUDUN_EX = 306
CRUSTLE = 345
LUCARIO = 678
OGERPON = 117
HILDA = 1225
AURA_JAB = 982
DRILL = 426
DEMOLISH = 148
BASIC_F = 6
ROCK_F = 20
FROSLASS = 104
MUNKIDORI = 112


def _field(player):
    return [q for q in list(player.active or []) + list(player.bench or []) if q is not None]


def _card(obs, area, index, player):
    try:
        area = AreaType(int(area)); p = obs.current.players[player]
        if area == AreaType.DECK: return (obs.select.deck or [])[index]
        if area == AreaType.HAND: return (p.hand or [])[index]
        if area == AreaType.DISCARD: return (p.discard or [])[index]
        if area == AreaType.ACTIVE: return (p.active or [])[index]
        if area == AreaType.BENCH: return (p.bench or [])[index]
        if area == AreaType.PRIZE: return (p.prize or [])[index]
        if area == AreaType.LOOKING: return (obs.current.looking or [])[index]
        if area == AreaType.STADIUM: return (obs.current.stadium or [])[index]
    except Exception:
        return None
    return None


def _is_energy_card(card) -> bool:
    c = CARD.get(getattr(card, 'id', -1))
    return bool(c and c.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY})


def _find_serial(player, serial):
    return next((q for q in _field(player) if int(getattr(q, 'serial', -1)) == int(serial)), None)


def _attack_cost(aid: int) -> int:
    a = ATTACK.get(int(aid or 0)); return len(getattr(a, 'energies', None) or []) if a is not None else 99


def _attack_damage(aid: int, opponent) -> int:
    aid = int(aid or 0)
    if aid == 425:
        return 60 * sum(1 for q in _field(opponent) if sem.is_ex_like(q.id))
    return sem.attack_base_damage(aid)


class CardTextReasonerGate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = {
            'calls': 0, 'overrides': {}, 'cards_profiled': 0,
            'unknown_wall_hits': 0, 'bypass_hits': 0, 'crustle_lock_hits': 0,
            'ability_wall_hits': 0, 'crustle_ogerpon_setup_hits': 0, 'errors': 0,
        }
        self.last = {}

    def _note(self, key: str):
        d = self.stats['overrides']; d[key] = d.get(key, 0) + 1

    def _emit(self, action, key: str, **extra):
        self._note(key); self.last = {'reason': key, 'action': list(action), **extra}; return action

    def _base_option(self, opts, base):
        try:
            if isinstance(base, list) and len(base) == 1 and 0 <= int(base[0]) < len(opts):
                return opts[int(base[0])]
        except Exception:
            pass
        return None

    def _direct_attack_is_bypass(self, option) -> bool:
        return bool(option is not None and option.type == OptionType.ATTACK and sem.attack_bypasses_active_effects(option.attackId))

    def _ready_bypass_choices(self, obs, opts, me, mine, defender):
        choices = []
        for i, o in enumerate(opts):
            q = _card(obs, o.area, o.index, getattr(o, 'playerIndex', me))
            if q is None: continue
            for aid in (getattr(CARD.get(q.id), 'attacks', None) or []):
                if len(q.energies or []) < _attack_cost(aid) or not sem.attack_bypasses_active_effects(aid):
                    continue
                dmg = _attack_damage(aid, obs.current.players[1-me])
                choices.append((int(q.id == DUDUN_EX and defender.id == CRUSTLE), dmg, -i, i, q))
        choices.sort(reverse=True, key=lambda x: x[:4])
        return choices

    def choose(self, obs_dict: dict, base: list[int], plan: Any = None) -> list[int]:
        self.stats['calls'] += 1
        try:
            obs = to_observation_class(obs_dict)
            if obs.current is None or obs.select is None: return base
            opts = list(obs.select.option or [])
            if not opts: return base
            me = obs.current.yourIndex; mine = obs.current.players[me]; op = obs.current.players[1-me]
            ctx = obs.select.context; bo = self._base_option(opts, base)
            own_active = mine.active[0] if mine.active else None
            opp_active = op.active[0] if op.active else None
            archetype = str(getattr(plan, 'archetype', 'unknown') or 'unknown')
            marnie_chip = bool(archetype == 'marnie' and any(q.id in {FROSLASS, MUNKIDORI} for q in _field(op)))
            if own_active is not None: self.stats['cards_profiled'] += 1
            if opp_active is not None: self.stats['cards_profiled'] += 1

            # Forced promotions and switch targets: select a fully charged text-proven
            # bypass attacker.  Dudunsparce ex is the exact Crustle answer.
            own_switch_menu = (ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE} and opp_active is not None
                               and any(int(getattr(o, 'playerIndex', me)) == me for o in opts)
                               and not any(int(getattr(o, 'playerIndex', me)) == 1-me for o in opts))
            if own_switch_menu:
                preferred = []
                for i, o in enumerate(opts):
                    q = _card(obs, o.area, o.index, getattr(o, 'playerIndex', me))
                    if q is None: continue
                    bypass = [aid for aid in (getattr(CARD.get(q.id), 'attacks', None) or [])
                              if sem.attack_bypasses_active_effects(aid) and len(q.energies or []) >= _attack_cost(aid)]
                    if bypass:
                        dmg = max((_attack_damage(aid, op) for aid in bypass), default=0)
                        preferred.append((int(q.id == DUDUN_EX and opp_active.id == CRUSTLE), dmg, -i, i, q))
                if preferred:
                    preferred.sort(reverse=True, key=lambda x: x[:4]); pick = preferred[0]
                    if not (isinstance(base, list) and base == [pick[3]]):
                        return self._emit([pick[3]], 'text:promote_ready_bypass', card_id=pick[4].id)

                # Previously unseen Ability attacker: a ready Cornerstone is a text-
                # proven wall unless the opponent's card itself has an obvious bypass.
                if sem.has_ability(opp_active.id) and not marnie_chip:
                    for i, o in enumerate(opts):
                        q = _card(obs, o.area, o.index, getattr(o, 'playerIndex', me))
                        if q is not None and q.id == OGERPON and len(q.energies or []) >= 3:
                            self.stats['ability_wall_hits'] += 1
                            return self._emit([i], 'text:promote_cornerstone_vs_ability')

            # Aura Jab allocation from any matchup dynamically detected as an ex wall.
            if ctx == SelectContext.ATTACH_FROM and opp_active is not None:
                dud = next((q for q in _field(mine) if q.id == DUDUN_EX and len(q.energies or []) < 3), None)
                if dud is not None and (opp_active.id == CRUSTLE or sem.damage_prevention_applies(opp_active.id, DUDUN_EX)):
                    for i, o in enumerate(opts):
                        q = _card(obs, o.area, o.index, getattr(o, 'playerIndex', me))
                        if q is not None and q.serial == dud.serial:
                            return self._emit([i], 'text:aura_charge_bypass')
                if dud is None and opp_active.id == CRUSTLE:
                    oger = next((q for q in _field(mine) if q.id == OGERPON and len(q.energies or []) < 3), None)
                    if oger is not None:
                        for i,o in enumerate(opts):
                            q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                            if q is not None and q.serial==oger.serial:
                                self.stats['crustle_ogerpon_setup_hits'] += 1
                                return self._emit([i], 'crustle:aura_charge_cornerstone_fallback')

            if ctx != SelectContext.MAIN or own_active is None or opp_active is None:
                return base

            # Live replay 93753865: Fighting Gong found Cornerstone Mask Ogerpon ex
            # against Crustle, but the policy repeatedly ended the turn with Ogerpon
            # still in hand.  If the exact Dudunsparce-ex bypass is not already on
            # the field, committing the legal Cornerstone body is a text-proven
            # fallback because Demolish ignores effects on the opponent's Active.
            if opp_active.id == CRUSTLE:
                dud_field = next((q for q in _field(mine) if q.id == DUDUN_EX), None)
                oger_field = next((q for q in _field(mine) if q.id == OGERPON), None)
                if dud_field is None and oger_field is None:
                    for i,o in enumerate(opts):
                        if o.type != OptionType.PLAY: continue
                        c = _card(obs, AreaType.HAND, o.index, me)
                        if c is not None and c.id == OGERPON:
                            self.stats['crustle_ogerpon_setup_hits'] += 1
                            return self._emit([i], 'crustle:play_cornerstone_fallback')

                # Once the fallback is committed, finish it rather than spending
                # manual Energy on attackers whose damage Crustle prevents.
                if dud_field is None and oger_field is not None and len(oger_field.energies or []) < 3 and not obs.current.energyAttached:
                    attach=[]
                    for i,o in enumerate(opts):
                        if o.type != OptionType.ATTACH: continue
                        q=_card(obs,o.inPlayArea,o.inPlayIndex,me);e=_card(obs,AreaType.HAND,o.index,me)
                        if q is not None and e is not None and q.serial==oger_field.serial and _is_energy_card(e):
                            attach.append((int(e.id in {BASIC_F,ROCK_F}),-i,i))
                    if attach:
                        attach.sort(reverse=True);self.stats['crustle_ogerpon_setup_hits'] += 1
                        return self._emit([attach[0][2]], 'crustle:charge_cornerstone_fallback', energy_count=len(oger_field.energies or []))

                if dud_field is None and oger_field is not None and len(oger_field.energies or []) >= 3 and oger_field.serial != own_active.serial:
                    retreats=[i for i,o in enumerate(opts) if o.type==OptionType.RETREAT]
                    if retreats:
                        self.stats['crustle_ogerpon_setup_hits'] += 1
                        return self._emit([retreats[0]], 'crustle:pivot_ready_cornerstone_fallback')

            # Legal attacks are already energy-validated by CABT.
            attacks = [(i, o, _attack_damage(o.attackId, op)) for i, o in enumerate(opts) if o.type == OptionType.ATTACK]
            bypass_attacks = [(dmg, -i, i, o) for i, o, dmg in attacks if sem.attack_bypasses_active_effects(o.attackId)]

            # CRUSTLE / DUDUNSPARCE EX HARD LOCK ---------------------------
            if own_active.id == DUDUN_EX and opp_active.id == CRUSTLE:
                # Once Destructive Drill is legal, nothing—LLM, retreat heuristic,
                # support setup, or generic search—may pull Dudunsparce ex away.
                for i, o, dmg in attacks:
                    if o.attackId == DRILL:
                        self.stats['crustle_lock_hits'] += 1
                        return self._emit([i], 'crustle:dudun_destructive_drill_lock', damage=dmg)

                # At two Energy, the third manual attachment is the whole win line.
                if len(own_active.energies or []) < 3 and not obs.current.energyAttached:
                    attach = []
                    for i, o in enumerate(opts):
                        if o.type != OptionType.ATTACH: continue
                        q = _card(obs, o.inPlayArea, o.inPlayIndex, me); e = _card(obs, AreaType.HAND, o.index, me)
                        if q is not None and e is not None and q.serial == own_active.serial and _is_energy_card(e):
                            attach.append((int(e.id in {BASIC_F, ROCK_F}), -i, i))
                    if attach:
                        attach.sort(reverse=True)
                        return self._emit([attach[0][2]], 'crustle:charge_active_dudun', energy_count=len(own_active.energies or []))

                # Hilda can fetch an Evolution plus the missing Energy.  Use it only
                # when a manual attachment is still available this turn.
                if len(own_active.energies or []) < 3 and not obs.current.supporterPlayed and not obs.current.energyAttached:
                    for i, o in enumerate(opts):
                        if o.type == OptionType.PLAY:
                            c = _card(obs, AreaType.HAND, o.index, me)
                            if c is not None and c.id == HILDA:
                                return self._emit([i], 'crustle:hilda_for_dudun_energy')

                # Never cash in Dudunsparce ex's stored Energy merely to expose a
                # Lucario and try to put the same Energy back with Aura Jab.  That
                # loop caused the supplied loss: the route repeatedly fell from two
                # Energy to one and never reached Destructive Drill.  A ready
                # alternative bypass attacker is still unnecessary—the exact line is
                # to keep Dudunsparce Active, preserve all Energy, and attach next turn.
                if bo is not None and bo.type == OptionType.RETREAT:
                    non_retreat_attack = next((i for i, o, _ in attacks if o.attackId == DRILL), None)
                    if non_retreat_attack is not None:
                        return self._emit([non_retreat_attack], 'crustle:dudun_drill_over_retreat')
                    end = next((i for i, o in enumerate(opts) if o.type == OptionType.END), None)
                    if end is not None:
                        return self._emit([end], 'crustle:block_all_unready_dudun_retreat', energy_count=len(own_active.energies or []))

            # Bench route: charge Dudunsparce ex to three before exposing it.  This
            # is stronger than repeatedly promoting a one-Energy body into Crustle.
            if opp_active.id == CRUSTLE:
                dud = next((q for q in _field(mine) if q.id == DUDUN_EX), None)
                if dud is not None and dud.serial != own_active.serial:
                    if len(dud.energies or []) < 3 and not obs.current.energyAttached:
                        for i, o in enumerate(opts):
                            if o.type != OptionType.ATTACH: continue
                            q = _card(obs, o.inPlayArea, o.inPlayIndex, me); e = _card(obs, AreaType.HAND, o.index, me)
                            if q is not None and e is not None and q.serial == dud.serial and _is_energy_card(e):
                                return self._emit([i], 'crustle:bench_charge_dudun')
                    if len(dud.energies or []) >= 3:
                        retreats = [i for i, o in enumerate(opts) if o.type == OptionType.RETREAT]
                        if retreats:
                            return self._emit([retreats[0]], 'crustle:pivot_ready_dudun')
                    # Aura Jab deals zero through the wall but its Energy-attachment
                    # effect remains useful and completes the bypass attacker.
                    if own_active.id == LUCARIO and len(dud.energies or []) < 3:
                        discard_basic = sum(1 for c in (mine.discard or []) if c.id == BASIC_F)
                        if discard_basic:
                            for i, o, _ in attacks:
                                if o.attackId == AURA_JAB:
                                    return self._emit([i], 'crustle:aura_jab_charge_dudun')

            # GENERIC TEXT WALL / BYPASS ----------------------------------
            # Evaluate each legal attack against the defender's current text.  If a
            # normal attack is provably blanked and a legal bypass attack exists,
            # take the bypass even when the card ID has never appeared in training.
            base_attack_damage = _attack_damage(bo.attackId, op) if bo is not None and bo.type == OptionType.ATTACK else None
            stadium_ids = [int(getattr(c, 'id', 0) or 0) for c in (obs.current.stadium or []) if c is not None]
            prevention = sem.damage_prevention_applies(
                opp_active.id, own_active.id, raw_damage=base_attack_damage
            ) or sem.global_damage_prevention_applies(
                stadium_ids, opp_active.id, own_active.id, raw_damage=base_attack_damage
            )
            if prevention:
                self.stats['unknown_wall_hits'] += int(opp_active.id not in {CRUSTLE, OGERPON})
                if bypass_attacks:
                    bypass_attacks.sort(reverse=True); pick = bypass_attacks[0]
                    self.stats['bypass_hits'] += 1
                    if not (isinstance(base, list) and base == [pick[2]]):
                        return self._emit([pick[2]], 'text:choose_effect_bypass', attack_id=pick[3].attackId, defender_id=opp_active.id)
                # A ready bypass attacker on the Bench is a proven reason to retreat.
                ready = []
                for q in (mine.bench or []):
                    if q is None: continue
                    for aid in (getattr(CARD.get(q.id), 'attacks', None) or []):
                        if sem.attack_bypasses_active_effects(aid) and len(q.energies or []) >= _attack_cost(aid):
                            ready.append(q); break
                if ready and bo is not None and bo.type not in {OptionType.RETREAT, OptionType.ATTACK}:
                    retreats = [i for i, o in enumerate(opts) if o.type == OptionType.RETREAT]
                    if retreats:
                        return self._emit([retreats[0]], 'text:pivot_to_unknown_bypass')

            # Generic unseen Ability attacker wall preservation.  Do not abandon a
            # ready Cornerstone when its text condition is currently satisfied.
            if own_active.id == OGERPON and sem.has_ability(opp_active.id) and len(own_active.energies or []) >= 3:
                # Froslass and Munkidori operate through counters rather than attack
                # damage.  Against that public counter package, Cornerstone is a
                # useful pivot but not a hard lock.  Keep it only for an immediate
                # Demolish KO; otherwise allow the Lucario/support-denial plan to act.
                if marnie_chip and int(opp_active.hp or 0) > _attack_damage(DEMOLISH, op):
                    return base
                for i, o, _ in attacks:
                    if o.attackId == DEMOLISH:
                        self.stats['ability_wall_hits'] += 1
                        return self._emit([i], 'text:hold_cornerstone_demolish')
                if bo is not None and bo.type == OptionType.RETREAT:
                    end = next((i for i, o in enumerate(opts) if o.type == OptionType.END), None)
                    if end is not None:
                        return self._emit([end], 'text:block_cornerstone_abandonment')

            return base
        except Exception:
            self.stats['errors'] += 1
            return base

    def get_stats(self):
        out = dict(self.stats); out['overrides'] = dict(self.stats.get('overrides') or {}); out['last'] = dict(self.last); return out
