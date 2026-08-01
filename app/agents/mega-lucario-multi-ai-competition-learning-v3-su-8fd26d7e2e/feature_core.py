from __future__ import annotations
import math, hashlib
from collections import Counter, deque
from typing import Any
from cg.api import AreaType, OptionType, SelectContext, all_card_data, all_attack

CARD_DATA={c.cardId:c for c in all_card_data()}
ATTACK_DATA={a.attackId:a for a in all_attack()}
OWN_IDS=[333,677,678,676,675,305,66,306,1141,1142,1152,1086,1213,1197,1174,1159,1227,1225,1182,1252,1211,6,20]
OWN_POKEMON=[333,677,678,676,675,305,66,306]
ARCHES=['unknown','alakazam','dragapult','lucario']
PHASES=['open','mid','late','end']
HIST_LEN=8


def new_memory():
    return {'turn':-1,'current':deque(maxlen=HIST_LEN),'previous':deque(maxlen=HIST_LEN),'revealed':set(),'arch':'unknown'}


def phase(obs):
    me=obs.current.players[obs.current.yourIndex];op=obs.current.players[1-obs.current.yourIndex]
    if len(me.prize or [])<=2 or len(op.prize or [])<=2:return 'end'
    if obs.current.turn<=3:return 'open'
    if obs.current.turn<=8:return 'mid'
    return 'late'


def infer_arch(obs,memory=None):
    me=obs.current.yourIndex;op=obs.current.players[1-me];ids=set()
    for p in list(op.active or [])+list(op.bench or []):
        if p is None:continue
        ids.add(int(p.id));ids.update(int(c.id) for c in (p.preEvolution or []))
    ids.update(int(c.id) for c in (op.discard or []))
    if any(x in ids for x in (741,742,743,245)):a='alakazam'
    elif any(x in ids for x in (119,120,121,689)):a='dragapult'
    elif any(x in ids for x in (333,677,678)):a='lucario'
    else:a='unknown'
    if memory is not None:
        memory['revealed'].update(ids)
        if a!='unknown':memory['arch']=a
        elif memory.get('arch') in ARCHES:a=memory['arch']
    return a


def update_memory(obs,memory):
    t=int(obs.current.turn)
    if memory.get('turn',-1)!=t:
        if memory.get('turn',-1)>=0:
            memory['previous']=deque(memory.get('current',[]),maxlen=HIST_LEN)
        memory['current']=deque(maxlen=HIST_LEN);memory['turn']=t
    return infer_arch(obs,memory)


def _int(v,default=-1):
    try:return int(v) if v is not None else default
    except:return default


def card_at(obs,area,index,player_index):
    try:
        area=AreaType(area);p=obs.current.players[player_index];i=int(index)
        if area==AreaType.DECK:return obs.select.deck[i]
        if area==AreaType.HAND:return p.hand[i]
        if area==AreaType.DISCARD:return p.discard[i]
        if area==AreaType.ACTIVE:return p.active[i]
        if area==AreaType.BENCH:return p.bench[i]
        if area==AreaType.PRIZE:return p.prize[i]
        if area==AreaType.STADIUM:return obs.current.stadium[i]
        if area==AreaType.LOOKING:return obs.current.looking[i]
        if area==AreaType.ENERGY:
            # For attached-card selection, area/index identify the Pokemon and energyIndex the actual card.
            return None
    except Exception:pass
    return None


def source_id(obs):
    for c in (getattr(obs.select,'effect',None),getattr(obs.select,'contextCard',None)):
        if c is not None:return int(c.id)
    return 0


def extended_action_desc(obs,o):
    me=obs.current.yourIndex;typ=_int(o.type,0);sub=tar=0
    try:
        if typ==int(OptionType.PLAY):
            c=card_at(obs,AreaType.HAND,o.index,me);sub=int(c.id) if c else 0
        elif typ in (int(OptionType.ATTACH),int(OptionType.EVOLVE)):
            c=card_at(obs,o.area,o.index,me);p=card_at(obs,o.inPlayArea,o.inPlayIndex,me);sub=int(c.id) if c else 0;tar=int(p.id) if p else 0
        elif typ in (int(OptionType.ABILITY),int(OptionType.DISCARD)):
            c=card_at(obs,o.area,o.index,me);sub=int(c.id) if c else 0
        elif typ==int(OptionType.ATTACK):
            p=(obs.current.players[me].active or [None])[0];sub=int(p.id) if p else 0;tar=int(o.attackId or 0)
        elif typ==int(OptionType.RETREAT):
            p=(obs.current.players[me].active or [None])[0];sub=int(p.id) if p else 0
        elif typ in (int(OptionType.CARD),int(OptionType.TOOL_CARD),int(OptionType.ENERGY_CARD),int(OptionType.ENERGY)):
            owner=me if o.playerIndex is None else int(o.playerIndex);p=card_at(obs,o.area,o.index,owner)
            if typ==int(OptionType.TOOL_CARD) and p is not None:
                sub=int(p.tools[int(o.toolIndex)].id) if p.tools and int(o.toolIndex)<len(p.tools) else 0;tar=int(p.id)
            elif typ in (int(OptionType.ENERGY_CARD),int(OptionType.ENERGY)) and p is not None:
                sub=int(p.energyCards[int(o.energyIndex)].id) if p.energyCards and int(o.energyIndex)<len(p.energyCards) else 0;tar=int(p.id)
            else:sub=int(p.id) if p else 0
        elif typ==int(OptionType.SKILL):sub=int(o.cardId or 0)
        elif typ==int(OptionType.NUMBER):sub=int(o.number or 0)
    except Exception:pass
    return [int(obs.select.context),source_id(obs),typ,sub,tar,
            _int(o.area),_int(o.index),_int(o.playerIndex),_int(o.toolIndex),_int(o.energyIndex),_int(o.count,0),
            _int(o.inPlayArea),_int(o.inPlayIndex),_int(o.attackId,0),_int(o.cardId,0),_int(o.number,0),_int(o.specialConditionType)]


def record_action(memory,descs,selected):
    toks=[]
    for i in selected or []:
        if 0<=int(i)<len(descs):
            d=descs[int(i)];toks.append((d[0],d[2],d[3],d[4],d[13]))
    for t in toks:memory['current'].append(t)


def _bits(x,n=11):
    x=max(0,int(x or 0));return [float((x>>b)&1) for b in range(n)]


def _prize_value(cid):
    c=CARD_DATA.get(int(cid));return 3 if c and c.megaEx else 2 if c and c.ex else 1


def _min_cost(cid):
    c=CARD_DATA.get(int(cid));vals=[]
    if c:
        for aid in c.attacks:
            a=ATTACK_DATA.get(int(aid));
            if a:vals.append(len(a.energies))
    return min(vals) if vals else 0


def _max_damage(cid):
    c=CARD_DATA.get(int(cid));vals=[]
    if c:
        for aid in c.attacks:
            a=ATTACK_DATA.get(int(aid));
            if a:vals.append(int(a.damage))
    return max(vals) if vals else 0


def _slot(p,own):
    if p is None:
        return ([1.0]+[0.0]*(1+len(OWN_POKEMON))+[0.0]*11+[0.0]*31) if own else ([0.0]*11+[0.0]*31)
    cid=int(p.id);c=CARD_DATA.get(cid);mx=max(1,int(p.maxHp));hp=max(0,int(p.hp));ec=Counter(int(x) for x in (p.energies or []));total=len(p.energies or [])
    nums=[hp/400.,(mx-hp)/400.,mx/400.,float(p.appearThisTurn),total/6.,len(p.energyCards or [])/6.,len(p.tools or [])/3.,len(p.preEvolution or [])/3.,_prize_value(cid)/3.,_min_cost(cid)/5.,_max_damage(cid)/400.]
    nums.extend(ec[i]/6. for i in range(12))
    nums.extend([float(bool(c and c.basic)),float(bool(c and c.stage1)),float(bool(c and c.stage2)),float(bool(c and c.ex)),float(bool(c and c.megaEx)),float(bool(c and c.tera)),(int(c.retreatCost)/5. if c else 0.),((int(c.energyType)+1)/12. if c else 0.)])
    if own:
        one=[0.0,float(cid not in OWN_POKEMON)]+[float(cid==x) for x in OWN_POKEMON]
        return one+_bits(cid)+nums
    return _bits(cid)+nums


def _token(tok):
    if tok is None:return [0.0]*35
    ctx,typ,sub,tar,atk=tok
    out=[ctx/50.,typ/20.];out+=_bits(sub);out+=_bits(tar);out+=_bits(atk);return out


def rich_state_features(obs,memory):
    st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me];active=(mine.active or [None])[0];oa=(op.active or [None])[0]
    ownfield=[p for p in list(mine.active or [])+list(mine.bench or []) if p is not None];opfield=[p for p in list(op.active or [])+list(op.bench or []) if p is not None]
    hand=Counter(int(c.id) for c in (mine.hand or []));disc=Counter(int(c.id) for c in (mine.discard or []));field=Counter(int(p.id) for p in ownfield)
    ownenergy=sum(len(p.energies or []) for p in ownfield);openergy=sum(len(p.energies or []) for p in opfield);owndmg=sum(max(0,p.maxHp-p.hp) for p in ownfield);opdmg=sum(max(0,p.maxHp-p.hp) for p in opfield)
    legal=obs.select.option or [];typecount=Counter(int(o.type) for o in legal)
    base=[
      st.turn/12.,st.turnActionCount/12.,float(st.yourIndex==st.firstPlayer),float(st.supporterPlayed),float(st.stadiumPlayed),float(st.energyAttached),float(st.retreated),
      (6-len(mine.prize or []))/6.,(6-len(op.prize or []))/6.,(len(op.prize or [])-len(mine.prize or []))/6.,mine.handCount/12.,op.handCount/12.,(mine.handCount-op.handCount)/12.,mine.deckCount/60.,op.deckCount/60.,
      len(mine.bench or [])/5.,len(op.bench or [])/5.,max(0,mine.benchMax-len(mine.bench or []))/5.,ownenergy/15.,openergy/15.,owndmg/800.,opdmg/800.,
      float(bool(mine.poisoned)),float(bool(mine.burned)),float(bool(mine.asleep)),float(bool(mine.paralyzed)),float(bool(mine.confused)),
      float(bool(op.poisoned)),float(bool(op.burned)),float(bool(op.asleep)),float(bool(op.paralyzed)),float(bool(op.confused)),
      (active.hp/400. if active else 0.),(oa.hp/400. if oa else 0.),(_prize_value(active.id)/3. if active else 0.),(_prize_value(oa.id)/3. if oa else 0.),
      len(legal)/30.,float(any(int(o.type)==int(OptionType.ATTACK) for o in legal)),float(any(int(o.type)==int(OptionType.EVOLVE) for o in legal)),float(any(int(o.type)==int(OptionType.ATTACH) for o in legal)),
      float(any(int(o.type)==int(OptionType.END) for o in legal)),float(any(int(o.type)==int(OptionType.ABILITY) for o in legal)),float(any(int(o.type)==int(OptionType.RETREAT) for o in legal)),
      int(obs.select.context)/50.,int(obs.select.minCount)/6.,int(obs.select.maxCount)/6.,int(obs.select.remainDamageCounter)/60.,int(obs.select.remainEnergyCost)/10.,
      source_id(obs)/1400.,(int(st.stadium[0].id)/1400. if st.stadium else 0.),len(memory.get('revealed',set()))/30.,
    ]
    # Exact own-deck counts by zone, with hand and discard cardinality retained.
    for cid in OWN_IDS:base.extend([hand[cid]/4.,disc[cid]/4.,field[cid]/4.])
    slots=(list(mine.active or [])+list(mine.bench or [])+[None]*6)[:6]
    for p in slots:base.extend(_slot(p,True))
    oslots=(list(op.active or [])+list(op.bench or [])+[None]*6)[:6]
    for p in oslots:base.extend(_slot(p,False))
    # Opponent discard and revealed IDs as hashed frequency summaries.
    h=[0.0]*48
    for c in op.discard or []:h[int(c.id)%48]+=1/4.
    for cid in memory.get('revealed',set()):h[int(cid)%48]+=0.25
    base.extend(h)
    base.extend(typecount[i]/8. for i in range(17))
    sh=[0.0]*48
    for o in legal:
        d=extended_action_desc(obs,o);sh[int(d[3])%48]+=1/6.
    base.extend(sh)
    cur=list(memory.get('current',[]))[-4:];prev=list(memory.get('previous',[]))[-4:]
    toks=[None]*(4-len(cur))+cur+[None]*(4-len(prev))+prev
    for t in toks:base.extend(_token(t))
    return base


def board_value(obs):
    st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
    own=[p for p in list(mine.active or [])+list(mine.bench or []) if p is not None];opp=[p for p in list(op.active or [])+list(op.bench or []) if p is not None]
    def val(ps,sign):
        z=0.0
        for p in ps:
            cid=int(p.id);c=CARD_DATA.get(cid);hp=max(0,int(p.hp));mx=max(1,int(p.maxHp));en=len(p.energies or []);stage=2 if c and c.stage2 else 1 if c and c.stage1 else 0
            z+=sign*(1.5*_prize_value(cid)+.45*stage+.22*en+.35*hp/mx+.12*len(p.tools or []))
        return z
    return 3.2*(len(op.prize or [])-len(mine.prize or []))+val(own,1)+val(opp,-1)+.06*(mine.handCount-op.handCount)+.02*(mine.deckCount-op.deckCount)
