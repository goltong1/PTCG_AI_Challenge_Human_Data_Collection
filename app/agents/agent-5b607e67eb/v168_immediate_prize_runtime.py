from __future__ import annotations
from cg.api import AreaType, OptionType, SelectContext, all_card_data, to_observation_class

CARD={int(c.cardId):c for c in all_card_data()}
POKE_PAD=1152
LUNATONE=675
SOLROCK=676
BASIC_F=6
ROCK_F=20
COSMIC_BEAM=980
SNORUNT=860
MARNIE_IMPIDIMP=646
DUNSPARCE=305
DUDUN=66


def _card(obs, area, index, player):
    try:
        area=AreaType(int(area)); p=obs.current.players[player]
        if area==AreaType.HAND:return (p.hand or [])[index]
        if area==AreaType.ACTIVE:return (p.active or [])[index]
        if area==AreaType.BENCH:return (p.bench or [])[index]
        if area==AreaType.DISCARD:return (p.discard or [])[index]
        if area==AreaType.DECK:return (obs.select.deck or [])[index]
        if area==AreaType.LOOKING:return (obs.current.looking or [])[index]
    except Exception:return None
    return None


def _has_luna_bench(p):
    return any(q is not None and int(q.id)==LUNATONE for q in list(p.bench or []))


def _fighting_ready(active):
    # Engine exposes attached produced energy types in pokemon.energies.
    return active is not None and BASIC_F in list(active.energies or [])


def _hand_fighting(p):
    return any(int(getattr(c,'id',-1) or -1) in {BASIC_F,ROCK_F} for c in list(p.hand or []))


def _safe_simple_cosmic_target(q):
    if q is None or int(getattr(q,'hp',9999) or 9999)>70:return False
    # v168 is intentionally a Marnie/Froslass opening correction.  Broadly
    # forcing every 70-HP prize target regressed Dragapult in A/B, so only
    # evolution-engine basics whose immediate removal has proven matchup value
    # are eligible before archetype identification is complete.
    if int(getattr(q,'id',-1) or -1) not in {SNORUNT,MARNIE_IMPIDIMP}:return False
    cd=CARD.get(int(getattr(q,'id',-1) or -1))
    if cd is None:return False
    # Early-prize gate is deliberately limited to one-Prize Pokemon.  This avoids
    # overriding more valuable Boss / multi-Prize lines on damaged ex targets.
    if bool(getattr(cd,'ex',False)) or bool(getattr(cd,'megaEx',False)):return False
    # Solrock is a non-ex Pokemon with no Ability, so the common ex/Ability walls
    # do not stop Cosmic Beam.  Unknown unconditional prevention remains a veto.
    for s in list(getattr(cd,'skills',[]) or []):
        t=str(getattr(s,'text','') or '').lower()
        if 'prevent all damage' in t:
            if 'pokémon ex' in t or 'pokemon ex' in t:continue
            if 'have an ability' in t or 'has an ability' in t:continue
            return False
    return True


class V168ImmediatePrizeGate:
    """Proof-gated one-turn prize continuation.

    Distilled from replay 93491468: when an early Poké Pad can complete
    Solrock + Lunatone + Fighting Energy into a guaranteed 70-HP one-Prize KO,
    take that line instead of redundant draw-engine search.  The gate stores the
    reason across the sub-decisions so Lunar Cycle or unrelated utility cannot
    consume the only attack Energy before Cosmic Beam.
    """
    def __init__(self):self.reset()
    def reset(self):
        self.chain_turn=-1; self.chain_active_serial=-1; self.chain_target_serial=-1
        self.stats={'calls':0,'overrides':{},'errors':0}; self.last={}
    def _emit(self,a,r,**kw):
        self.stats['overrides'][r]=self.stats['overrides'].get(r,0)+1
        self.last={'reason':r,'action':list(a),**kw}; return list(a)
    def _clear_chain(self):
        self.chain_turn=-1;self.chain_active_serial=-1;self.chain_target_serial=-1
    def _chain_valid(self,obs,mine,op):
        if int(obs.current.turn or -1)!=self.chain_turn:return False
        if not mine.active or int(mine.active[0].id)!=SOLROCK:return False
        if self.chain_active_serial>=0 and int(mine.active[0].serial)!=self.chain_active_serial:return False
        if not op.active:return False
        if self.chain_target_serial>=0 and int(op.active[0].serial)!=self.chain_target_serial:return False
        return _safe_simple_cosmic_target(op.active[0])
    def choose(self,obs_dict,base):
        self.stats['calls']+=1
        try:
            if not isinstance(base,list):return base
            obs=to_observation_class(obs_dict)
            if obs.current is None or obs.select is None:return base
            me=int(obs.current.yourIndex); mine=obs.current.players[me]; op=obs.current.players[1-me]
            opts=list(obs.select.option or []); ctx=obs.select.context
            turn=int(obs.current.turn or 0)
            active=mine.active[0] if mine.active else None

            # Search decision: only intervene when Poké Pad itself is already being
            # resolved and the entire one-turn prize line is publicly provable.
            if ctx==SelectContext.TO_HAND and int(getattr(getattr(obs.select,'effect',None),'id',-1) or -1)==POKE_PAD:
                fp=int(getattr(obs.current,'firstPlayer',-1) or -1)
                first_own_turn=(turn==(1 if me==fp else 2))
                early=(first_own_turn and len(mine.prize or [])>=5)
                bench_room=int(getattr(mine,'benchMax',5) or 5)-len([q for q in list(mine.bench or []) if q is not None])
                energy_path=_fighting_ready(active) or (not bool(obs.current.energyAttached) and _hand_fighting(mine))
                target=op.active[0] if op.active else None
                # Exact replay proof: the bad search was a *second* Dudunsparce
                # while only one Dunsparce existed in play.  Do not generalize the
                # first-prize rule to healthy search decisions; intervene only when
                # the base policy is about to spend Pad on that immediately redundant
                # second evolution card.
                hand_dudun=sum(1 for c in list(mine.hand or []) if int(getattr(c,'id',-1) or -1)==DUDUN)
                field_duns=sum(1 for q in list(mine.active or [])+list(mine.bench or []) if q is not None and int(q.id)==DUNSPARCE)
                base_dudun=False
                if isinstance(base,list) and len(base)==1 and 0<=int(base[0])<len(opts):
                    bo=opts[int(base[0])]
                    bq=_card(obs,bo.area,bo.index,int(getattr(bo,'playerIndex',me)))
                    base_dudun=(bq is not None and int(bq.id)==DUDUN)
                redundant_dudun=(base_dudun and hand_dudun>=1 and field_duns<=1)
                if early and redundant_dudun and active is not None and int(active.id)==SOLROCK and not _has_luna_bench(mine) and bench_room>0 and energy_path and _safe_simple_cosmic_target(target):
                    luna=[]
                    for i,o in enumerate(opts):
                        q=_card(obs,o.area,o.index,int(getattr(o,'playerIndex',me)))
                        if q is not None and int(q.id)==LUNATONE:luna.append(i)
                    if luna:
                        self.chain_turn=turn;self.chain_active_serial=int(active.serial);self.chain_target_serial=int(target.serial)
                        return self._emit([luna[0]],'cf:early_pad_lunatone_for_prize',target=int(target.id),hp=int(target.hp))

            # Continue the exact chain.  This is intentionally stronger than the
            # ordinary ranking only while the original KO proof is still valid.
            if self.chain_turn>=0:
                if not self._chain_valid(obs,mine,op):
                    self._clear_chain();return base
                active=mine.active[0]
                if ctx==SelectContext.MAIN:
                    # 1) Bench the searched Lunatone.
                    if not _has_luna_bench(mine):
                        for i,o in enumerate(opts):
                            if o.type==OptionType.PLAY:
                                q=_card(obs,AreaType.HAND,o.index,me)
                                if q is not None and int(q.id)==LUNATONE:
                                    return self._emit([i],'cf:early_play_lunatone')
                        self._clear_chain();return base
                    # 2) Protect the only attack Energy: attach before Lunar Cycle.
                    if not _fighting_ready(active):
                        if bool(obs.current.energyAttached):
                            self._clear_chain();return base
                        cand=[]
                        for i,o in enumerate(opts):
                            if o.type!=OptionType.ATTACH:continue
                            src=_card(obs,o.area,o.index,me);dst=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                            if src is None or dst is None:continue
                            if int(src.id) in {BASIC_F,ROCK_F} and int(dst.serial)==int(active.serial):
                                # Preserve Rock when Basic is also available.
                                cand.append((0 if int(src.id)==BASIC_F else 1,i))
                        if cand:
                            cand.sort();return self._emit([cand[0][1]],'cf:early_attach_solrock_for_prize')
                        self._clear_chain();return base
                    # 3) Take the guaranteed KO before draw/utility actions.
                    for i,o in enumerate(opts):
                        if o.type==OptionType.ATTACK and int(getattr(o,'attackId',-1) or -1)==COSMIC_BEAM:
                            self._clear_chain();return self._emit([i],'cf:early_cosmic_guaranteed_prize')
                    self._clear_chain();return base
                # Lunar Cycle asks for a discard after the Ability is selected.  The
                # MAIN gate above should prevent entering it, but if an upstream layer
                # did so before this wrapper, never discard the last attack Energy.
                if ctx==SelectContext.DISCARD and not _fighting_ready(active):
                    self._clear_chain();return base
            return base
        except Exception:
            self.stats['errors']+=1;self._clear_chain();return base
    def get_stats(self):
        return {'calls':self.stats['calls'],'overrides':dict(self.stats['overrides']),'errors':self.stats['errors'],'last':dict(self.last)}
