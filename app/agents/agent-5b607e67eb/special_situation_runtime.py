"""Conservative special-situation gate for Lucario v156.

The learned policy remains responsible for ordinary sequencing.  This final gate
changes an action only when public state and local CABT rules text prove a narrow
dominance relation: a guaranteed Active KO is being abandoned, a repeated Energy
attachment is being wasted on an already retreat-ready engine Basic, or an
opponent target menu contains a clearly stronger text-defined threat/KO.  A small
set of post-audit promotion, stall-recovery, and successor-energy repairs is
restricted to recognized Dragapult games after independent control ablation.
"""
from __future__ import annotations
import re
from typing import Any
from cg.api import AreaType,CardType,OptionType,SelectContext,all_attack,all_card_data,to_observation_class
import card_text_semantics as sem

CARD={int(c.cardId):c for c in all_card_data()}
ATTACK={int(a.attackId):a for a in all_attack()}
RIOLU70=333;RIOLU80=677;LUCARIO=678;DUNSPARCE=305;DUDUN=66;DUDUN_EX=306
SOLROCK=676;LUNATONE=675;OGERPON=117;HERO_CAPE=1159;CRUSTLE=345
FROSLASS=104;MUNKIDORI=112;SNORUNT=860
HILDA=1225;LILLIE=1227;JUDGE=1213;FIGHTING_GONG=1142;POKE_PAD=1152;POFFIN=1086


def _field(p):return [q for q in list(p.active or [])+list(p.bench or []) if q is not None]

def _card(obs,area,index,player):
    try:
        area=AreaType(int(area));p=obs.current.players[player]
        if area==AreaType.DECK:return (obs.select.deck or [])[index]
        if area==AreaType.HAND:return (p.hand or [])[index]
        if area==AreaType.DISCARD:return (p.discard or [])[index]
        if area==AreaType.ACTIVE:return (p.active or [])[index]
        if area==AreaType.BENCH:return (p.bench or [])[index]
        if area==AreaType.PRIZE:return (p.prize or [])[index]
        if area==AreaType.LOOKING:return (obs.current.looking or [])[index]
        if area==AreaType.STADIUM:return (obs.current.stadium or [])[index]
    except Exception:return None
    return None

def _find_serial(p,serial):
    try:return next((q for q in _field(p) if int(q.serial)==int(serial)),None)
    except Exception:return None

def _prize_value(q):
    c=CARD.get(int(getattr(q,'id',0) or 0))
    if c is None:return 1
    if bool(getattr(c,'megaEx',False)):return 3
    if bool(getattr(c,'ex',False)):return 2
    return 1

def _attack_cost(aid):
    a=ATTACK.get(int(aid or 0));return len(getattr(a,'energies',None) or []) if a is not None else 99

def _has_named_bench_requirement(aid,mine):
    a=ATTACK.get(int(aid or 0))
    if a is None:return True
    text=sem.normalize(getattr(a,'text','') or '')
    m=re.search(r"if you don'?t have (?:a |an )?(.+?) on your bench, this attack does nothing",text)
    if not m:return True
    need=m.group(1).strip().replace('pokemon','').replace('pokémon','').strip()
    names=[sem.normalize(getattr(CARD.get(int(q.id)),'name','')) for q in (mine.bench or []) if q is not None]
    return any(need==n or need in n for n in names)

def _guaranteed_damage(aid,mine,op,attacker,defender,stadiums):
    a=ATTACK.get(int(aid or 0))
    if a is None or attacker is None or defender is None:return 0
    if not _has_named_bench_requirement(aid,mine):return 0
    text=sem.normalize(getattr(a,'text','') or '')
    dmg=int(getattr(a,'damage',0) or 0)
    if dmg<=0 and 'for each of your opponent' in text and (' ex' in text or '{ex}' in text):
        m=re.search(r'does (\d+) damage for each',text)
        if m:dmg=int(m.group(1))*sum(1 for q in _field(op) if sem.is_ex_like(q.id))
    if dmg<=0:return 0
    if sem.attack_bypasses_active_effects(int(aid)):return dmg
    special=False
    for e in list(getattr(attacker,'energyCards',None) or getattr(attacker,'energies',None) or []):
        eid=int(getattr(e,'id',e) or 0);cd=CARD.get(eid)
        if cd is not None and cd.cardType==CardType.SPECIAL_ENERGY:special=True;break
    if sem.damage_prevention_applies(defender.id,attacker.id,attacker_has_special_energy=special,raw_damage=dmg):return 0
    if sem.global_damage_prevention_applies(stadiums,defender.id,attacker.id,attacker_has_special_energy=special,raw_damage=dmg):return 0
    return dmg

def _threat_score(q):
    cid=int(getattr(q,'id',0) or 0);c=CARD.get(cid)
    if c is None:return 0
    tags=set(sem.card_skill_tags(cid));score=0
    weights={'ENERGY_ACCELERATION':45,'DRAW':34,'SEARCH_DECK':30,'DAMAGE_COUNTER':30,'MOVE_DAMAGE_COUNTER':34,
             'HEAL':16,'SWITCH':12,'PREVENT_DAMAGE':26,'PREVENT_EFFECTS':16,'NO_RETREAT':18,'ON_EVOLVE':12,
             'DISCARD':10,'PREVENT_BENCH':18,'PREVENT_BENCH_COUNTERS':20}
    score+=sum(weights.get(t,0) for t in tags)
    for aid in list(getattr(c,'attacks',None) or []):
        at=set(sem.attack_tags(aid));score+=sum({'BENCH_DAMAGE_COUNTERS':48,'BENCH_DAMAGE':38,'DAMAGE_COUNTER':30,
            'ENERGY_ACCELERATION':32,'DISCARD_ENERGY':30,'SCALE_OPPONENT_EX':26,'SCALE_ENERGY':22,
            'NEXT_TURN_LOCK':16,'PARALYZE':24,'CONFUSE':14,'POISON':12,'SLEEP':12,'SWITCH':10}.get(t,0) for t in at)
        if len(getattr(q,'energies',None) or [])>=_attack_cost(aid):score+=10
    if bool(getattr(c,'stage2',False)):score+=12
    if bool(getattr(c,'ex',False) or getattr(c,'megaEx',False)):score+=12
    score+=max(0,(int(getattr(q,'maxHp',0) or 0)-int(getattr(q,'hp',0) or 0))//20)
    return score

class SpecialSituationGate:
    def __init__(self):self.reset()
    def reset(self):
        self.stats={'calls':0,'overrides':{},'exact_ko_hits':0,'engine_redirects':0,'last_slot_rescues':0,'text_target_hits':0,'promotion_hits':0,'stall_recovery_hits':0,'successor_redirects':0,'errors':0};self.last={}
    def _emit(self,a,key,**kw):
        self.stats['overrides'][key]=self.stats['overrides'].get(key,0)+1;self.last={'reason':key,'action':list(a),**kw};return a
    def choose(self,obs_dict:dict,base:list[int],plan:Any=None)->list[int]:
        self.stats['calls']+=1
        try:
            obs=to_observation_class(obs_dict)
            if obs.current is None or obs.select is None:return base
            opts=list(obs.select.option or [])
            if not opts:return base
            me=obs.current.yourIndex;mine=obs.current.players[me];op=obs.current.players[1-me]
            ctx=obs.select.context;bi=int(base[0]) if isinstance(base,list) and len(base)==1 else -1
            bo=opts[bi] if 0<=bi<len(opts) else None
            archetype=str(getattr(plan,'archetype','unknown') or 'unknown')
            active=mine.active[0] if mine.active else None;defender=op.active[0] if op.active else None
            stadiums=[int(c.id) for c in (obs.current.stadium or []) if c is not None]
            marnie_chip=bool(archetype=='marnie' and any(q.id in {FROSLASS,MUNKIDORI,SNORUNT} for q in _field(op)))
            luc_lines=sum(q.id in {RIOLU70,RIOLU80,LUCARIO} for q in _field(mine))

            # Opponent target menus (Boss/switch effects): card text supplies a
            # generic target prior even when the card ID never appeared in training.
            if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE} and active is not None:
                opp_choices=[]
                for i,o in enumerate(opts):
                    if int(getattr(o,'playerIndex',me))!=1-me:continue
                    q=_card(obs,o.area,o.index,1-me)
                    if q is None:continue
                    best=0
                    for aid in (getattr(CARD.get(active.id),'attacks',None) or []):
                        if len(active.energies or [])>=_attack_cost(aid):
                            best=max(best,_guaranteed_damage(aid,mine,op,active,q,stadiums))
                    ko=int(best>=int(q.hp or 0)>0);pv=_prize_value(q);th=_threat_score(q)
                    opp_choices.append((ko,pv if ko else 0,th,-int(q.hp or 0),-i,i,q,best))
                if opp_choices:
                    opp_choices.sort(reverse=True,key=lambda x:x[:6]);pick=opp_choices[0]
                    base_row=next((x for x in opp_choices if x[5]==bi),None)
                    if archetype=='dragapult':
                        superior=(pick[0] and (base_row is None or not base_row[0] or pick[1]>base_row[1]))
                    else:
                        superior=(pick[0] and (base_row is None or not base_row[0] or pick[1]>base_row[1])) or (not pick[0] and base_row is not None and pick[2]>=base_row[2]+32)
                    if superior and pick[5]!=bi:
                        self.stats['text_target_hits']+=1
                        return self._emit([pick[5]],'special:text_threat_target',card_id=pick[6].id,ko=bool(pick[0]),damage=pick[7],threat=pick[2])

            # Dragapult-only forced promotion repair.  Override only when live
            # card text proves an immediate KO or a real damaging attacker while
            # the baseline selected a body that cannot damage the current Active.
            if archetype=='dragapult' and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE} and defender is not None:
                own=[]
                for i,o in enumerate(opts):
                    if int(getattr(o,'playerIndex',me))!=me:continue
                    q=_card(obs,o.area,o.index,me)
                    if q is None:continue
                    best=0
                    for aid in (getattr(CARD.get(int(q.id)),'attacks',None) or []):
                        if len(q.energies or [])>=_attack_cost(aid):best=max(best,_guaranteed_damage(aid,mine,op,q,defender,stadiums))
                    own.append((int(best>=int(defender.hp or 0)>0),best,-i,i,q))
                if own:
                    own.sort(reverse=True,key=lambda x:x[:3]);pick=own[0];base_row=next((x for x in own if x[3]==bi),None)
                    superior=(base_row is None or (pick[0] and not base_row[0]) or (pick[1]>=70 and base_row[1]<=0))
                    if superior and pick[3]!=bi:
                        self.stats['promotion_hits']+=1
                        return self._emit([pick[3]],'special:dragapult_promote_ready_attacker',card_id=pick[4].id,ko=bool(pick[0]),damage=pick[1])

            if ctx!=SelectContext.MAIN:return base

            # Guaranteed KO may replace only a retreat/end or a weaker attack.
            # Setup actions remain untouched so the agent can still develop before
            # taking the same legal attack later in the turn.
            if active is not None and defender is not None:
                attacks=[]
                for i,o in enumerate(opts):
                    if o.type!=OptionType.ATTACK:continue
                    dmg=_guaranteed_damage(o.attackId,mine,op,active,defender,stadiums)
                    if dmg>=int(defender.hp or 0)>0:
                        attacks.append((_prize_value(defender),dmg,int(sem.attack_bypasses_active_effects(o.attackId)),-i,i,o.attackId))
                if attacks:
                    attacks.sort(reverse=True);pick=attacks[0]
                    base_dmg=_guaranteed_damage(bo.attackId,mine,op,active,defender,stadiums) if bo is not None and bo.type==OptionType.ATTACK else 0
                    switch_play=False
                    if bo is not None and bo.type==OptionType.PLAY:
                        src=_card(obs,AreaType.HAND,bo.index,me);switch_play=bool(archetype=='dragapult' and src is not None and 'SWITCH' in set(sem.card_skill_tags(src.id)))
                    if bo is not None and (bo.type in {OptionType.RETREAT,OptionType.END} or switch_play or (bo.type==OptionType.ATTACK and base_dmg<int(defender.hp or 0))):
                        self.stats['exact_ko_hits']+=1
                        return self._emit([pick[4]],'special:guaranteed_active_ko',attack_id=pick[5],damage=pick[1],defender_id=defender.id,blocked_switch_play=switch_play)

                # Ending the turn while a deterministic, non-sacrificial damaging
                # attack is legal appeared repeatedly in the expanded loss league.
                # Only attacks without self-damage, own-Energy discard, conditional
                # zero damage, or a self next-turn lock qualify for this dominance
                # repair; setup actions are never interrupted.
                if bo is not None and bo.type==OptionType.END:
                    safe=[]
                    banned={'SELF_DAMAGE','DISCARD_OWN_ENERGY','DISCARD_ALL_OWN_ENERGY','SELF_NEXT_TURN_LOCK','CONDITIONAL_ZERO'}
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACK:continue
                        dmg=_guaranteed_damage(o.attackId,mine,op,active,defender,stadiums)
                        tags=set(sem.attack_tags(o.attackId))
                        if dmg>0 and not (tags&banned):safe.append((dmg,int(sem.attack_bypasses_active_effects(o.attackId)),-i,i,o.attackId))
                    if safe:
                        safe.sort(reverse=True);pick=safe[0]
                        return self._emit([pick[3]],'special:safe_damage_over_end',attack_id=pick[4],damage=pick[0])

            if bo is None:return base

            # Narrow Dragapult stall repair: with no text-valid attack and a small
            # energyless hand, use a legal draw/search card rather than END.
            if archetype=='dragapult' and bo.type==OptionType.END:
                positive=any(o.type==OptionType.ATTACK and active is not None and defender is not None and _guaranteed_damage(o.attackId,mine,op,active,defender,stadiums)>0 for o in opts)
                hand=list(mine.hand or []);hn=len(hand);has_energy=any(getattr(CARD.get(int(c.id)),'cardType',None) in {CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY} for c in hand)
                if not positive and any(q.id in {RIOLU70,RIOLU80,LUCARIO} for q in _field(mine)) and (hn<=4 or not has_energy):
                    recovery=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.PLAY:continue
                        src=_card(obs,AreaType.HAND,o.index,me)
                        if src is None:continue
                        score=-999
                        if src.id==HILDA:score=130
                        elif src.id==LILLIE and (len(mine.prize or [])==6 or hn<=5):score=120
                        elif src.id==JUDGE and hn<=4:score=100
                        elif src.id==FIGHTING_GONG:score=95
                        if score>-900:recovery.append((score,-i,i,src.id))
                    if recovery:
                        recovery.sort(reverse=True);self.stats['stall_recovery_hits']+=1
                        return self._emit([recovery[0][2]],'special:dragapult_stall_recovery',card_id=recovery[0][3],hand=hn,has_energy=has_energy)

            # A one-slot board with no Lucario line cannot spend its final slot on
            # generic utility when a legal Riolu is already in hand.  Text-proven
            # Crustle/Ability-wall routes remain exempt.
            if bo.type==OptionType.PLAY:
                source=_card(obs,AreaType.HAND,bo.index,me)
                bench=[q for q in (mine.bench or []) if q is not None];room=int(mine.benchMax or 5)-len(bench)
                lines=luc_lines
                need_line=(lines==0) or (marnie_chip and lines<2) or (archetype=='alakazam' and lines<2)
                if source is not None and room<=1 and need_line and source.id in {DUNSPARCE,SOLROCK,LUNATONE,OGERPON}:
                    preserve=(archetype=='crustle' and source.id==DUNSPARCE) or (archetype=='archaludon' and source.id==OGERPON) or (archetype=='marnie' and source.id==OGERPON and not marnie_chip)
                    ri=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.PLAY:continue
                        q=_card(obs,AreaType.HAND,o.index,me)
                        if q is not None and q.id in {RIOLU70,RIOLU80}:ri.append((int(q.id==RIOLU80),-i,i,q.id))
                    if ri and not preserve:
                        ri.sort(reverse=True);self.stats['last_slot_rescues']+=1
                        return self._emit([ri[0][2]],'special:last_slot_riolu',card_id=ri[0][3])

                # Counter-based Marnie support invalidates an Ogerpon-only lock.
                # When the baseline tries to deploy another wall before a second
                # Lucario line exists, a legal Riolu is the higher-continuity body.
                if source is not None and source.id==OGERPON and marnie_chip and lines<2:
                    ri=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.PLAY:continue
                        q=_card(obs,AreaType.HAND,o.index,me)
                        if q is not None and q.id in {RIOLU70,RIOLU80}:ri.append((int(q.id==RIOLU80),-i,i,q.id))
                    if ri:
                        ri.sort(reverse=True);self.stats['last_slot_rescues']+=1
                        return self._emit([ri[0][2]],'special:marnie_riolu_over_chip_wall',card_id=ri[0][3])

            # In the public Froslass/Munkidori state, a fourth wall Energy does not
            # answer the counter route.  Redirect only the exact same Energy card to
            # an undercharged Lucario line; this is the recurring paired repair in
            # the new loss audit and leaves ordinary Ogerpon setup untouched.
            if bo.type==OptionType.ATTACH and marnie_chip:
                source=_card(obs,AreaType.HAND,bo.index,me);target=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                sc=CARD.get(int(getattr(source,'id',0) or 0)) if source is not None else None
                redirectable=bool(sc is not None and (sc.cardType in {CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY} or int(source.id)==HERO_CAPE)) if source is not None else False
                if source is not None and target is not None and redirectable and target.id==OGERPON:
                    cand=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        e=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if e is None or q is None or int(e.serial)!=int(source.serial):continue
                        en=len(q.energies or []);score=-999
                        if q.id==LUCARIO and en<2:score=600-en*55
                        elif q.id in {RIOLU70,RIOLU80} and en<2:score=520-en*45+int(q.id==RIOLU80)*15
                        elif q.id==SOLROCK and en<1 and any(x.id==LUNATONE for x in _field(mine)):score=350
                        if score>-900:cand.append((score,-i,i,q.id))
                    if cand:
                        cand.sort(reverse=True);self.stats['engine_redirects']+=1
                        return self._emit([cand[0][2]],'special:marnie_redirect_wall_energy',target_id=cand[0][3])

            # Damaged, attack-ready Lucario needs a successor against Dragapult.
            # Redirect only the same manual Energy from a utility body to a legal
            # undercharged Benched Lucario/Riolu.
            if bo.type==OptionType.ATTACH and archetype=='dragapult' and active is not None and active.id==LUCARIO and len(active.energies or [])>=2 and int(active.hp or 0)<=140:
                source=_card(obs,AreaType.HAND,bo.index,me);target=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                sc=CARD.get(int(getattr(source,'id',0) or 0)) if source is not None else None
                if source is not None and target is not None and sc is not None and sc.cardType in {CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY} and target.id in {DUNSPARCE,SOLROCK,LUNATONE}:
                    cand=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        e=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if e is None or q is None or int(e.serial)!=int(source.serial) or q.serial==active.serial:continue
                        en=len(q.energies or []);score=-999
                        if q.id==LUCARIO and en<2:score=600-en*40
                        elif q.id in {RIOLU70,RIOLU80} and en<2:score=520-en*35+int(q.id==RIOLU80)*10
                        if score>-900:cand.append((score,-i,i,q.id))
                    if cand:
                        cand.sort(reverse=True);self.stats['successor_redirects']+=1
                        return self._emit([cand[0][2]],'special:dragapult_successor_energy',target_id=cand[0][3],active_hp=int(active.hp or 0))

            # Human-loss repair: against Alakazam, the newest real replay showed
            # an early manual Fighting Energy being spent on Solrock while an
            # energyless Riolu line was already available.  In turns 1-3, redirect
            # only that exact Energy card from a utility body to an undercharged
            # Riolu/Lucario.  This preserves card/phase semantics and cannot fire
            # when the baseline is already charging an attacker.
            if bo.type==OptionType.ATTACH and archetype=='alakazam' and int(obs.current.turn or 0)<=3:
                source=_card(obs,AreaType.HAND,bo.index,me);target=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                sc=CARD.get(int(getattr(source,'id',0) or 0)) if source is not None else None
                if source is not None and target is not None and sc is not None and sc.cardType in {CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY} and target.id in {DUNSPARCE,SOLROCK,LUNATONE}:
                    cand=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        e=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if e is None or q is None or int(e.serial)!=int(source.serial):continue
                        en=len(q.energies or []);score=-999
                        if q.id==LUCARIO and en<2:score=700-en*55
                        elif q.id in {RIOLU70,RIOLU80} and en<1:score=630+int(q.id==RIOLU80)*15
                        if score>-900:cand.append((score,-i,i,q.id))
                    if cand:
                        cand.sort(reverse=True);self.stats['engine_redirects']+=1
                        return self._emit([cand[0][2]],'special:alakazam_early_attacker_energy',target_id=cand[0][3],source_id=source.id)

            # Dunsparce needs one Energy to retreat; repeatedly attaching a second
            # or third Energy in non-Crustle games delayed every actual attacker in
            # the new cross-play losses. Redirect only the same source Energy to a
            # legal, undercharged win-condition.
            if bo.type==OptionType.ATTACH and archetype!='crustle':
                source=_card(obs,AreaType.HAND,bo.index,me);target=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                sc=CARD.get(int(getattr(source,'id',0) or 0)) if source is not None else None
                hand_ids={int(c.id) for c in (mine.hand or [])}
                if source is not None and target is not None and sc is not None and sc.cardType in {CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY} and target.id==DUNSPARCE and len(target.energies or [])>=1 and (not ({DUDUN,DUDUN_EX}&hand_ids) or luc_lines>=2):
                    cand=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        e=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if e is None or q is None or e.serial!=source.serial:continue
                        en=len(q.energies or []);score=-999
                        if archetype=='archaludon' and q.id==OGERPON and en<3:score=520-en*20
                        elif q.id==LUCARIO and en<2:score=460-en*35
                        elif q.id in {RIOLU70,RIOLU80} and en<2:score=410-en*30+int(q.id==RIOLU80)*15
                        elif q.id==SOLROCK and en<1:score=270
                        if score>-900:cand.append((score,-i,i,q.id))
                    if cand:
                        cand.sort(reverse=True);self.stats['engine_redirects']+=1
                        return self._emit([cand[0][2]],'special:dunsparce_overattach_redirect',target_id=cand[0][3])

            # Hero's Cape is long-lived prize protection.  Outside the Crustle
            # bypass route, placing it on a one-Prize Dunsparce while a Lucario
            # line is available wastes the ACE SPEC on a disposable pivot.
            if bo.type==OptionType.ATTACH and archetype!='crustle':
                source=_card(obs,AreaType.HAND,bo.index,me);target=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                if source is not None and target is not None and int(source.id)==HERO_CAPE and target.id==DUNSPARCE and luc_lines>0:
                    cand=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        e=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if e is None or q is None or int(e.serial)!=int(source.serial):continue
                        score=500 if q.id==LUCARIO else 430+int(q.id==RIOLU80)*15 if q.id in {RIOLU70,RIOLU80} else -1
                        if score>=0:cand.append((score,-i,i,q.id))
                    if cand:
                        cand.sort(reverse=True)
                        return self._emit([cand[0][2]],'special:hero_cape_win_condition',target_id=cand[0][3])
            return base
        except Exception:
            self.stats['errors']+=1;return base
    def get_stats(self):
        z=dict(self.stats);z['overrides']=dict(self.stats.get('overrides') or {});z['last']=dict(self.last);return z
