from __future__ import annotations
import re
from collections import Counter
from typing import Any
from cg.api import AreaType, OptionType, SelectContext, Pokemon, all_card_data, all_attack

CARD={c.cardId:c for c in all_card_data()}
ATTACK={a.attackId:a for a in all_attack()}
DARK_ENERGY=7
IMPIDIMP=646;MORGREM=647;GRIMMSNARL_EX=648;MUNKIDORI=112;SNORUNT=860;FROSLASS=104
RARE_CANDY=1079;UNFAIR_STAMP=1080;BUDDY_BUDDY_POFFIN=1086;NIGHT_STRETCHER=1097;POKEGEAR=1122;TOOL_SCRAPPER=1137;POKE_PAD=1152;BOSSS_ORDERS=1182;TEAM_ROCKETS_PETREL=1219;LILLIES_DETERMINATION=1227;DAWN=1231;SPIKEMUTH_GYM=1259;JUDGE=1213
MARNIE_LINE={IMPIDIMP,MORGREM,GRIMMSNARL_EX}
OWN_IDS=[7,104,112,646,647,648,860,1079,1080,1086,1097,1122,1137,1152,1182,1219,1227,1231,1259]
OWN_POKEMON=[104,112,646,647,648,860]
ARCHES=['unknown','lucario','dragapult','alakazam','archaludon','hydrapple','marnie','crustle','rocket','cynthia','festival']
PHASES=['open','mid','late','end']
TOKEN_TYPES=['NONE','P','E','A','B','K','R','Z']
SLOT_COUNT=6;HASH_OPP_DISCARD=32;HASH_REVEALED=32;HASH_LEGAL=32


def new_memory():
    return {'last_turn':-1,'last_prizes':6,'recent_ko':False,'revealed':set(),'archetype':'unknown','turn_tokens':[],'previous_turn_tokens':[],'setup_active_id':0}

def field(player):
    out=[]
    for i,p in enumerate(player.active or []):
        if p is not None:out.append((AreaType.ACTIVE,i,p))
    for i,p in enumerate(player.bench or []):
        if p is not None:out.append((AreaType.BENCH,i,p))
    return out

def get_card(obs,area,index,player_index):
    try:
        player=obs.current.players[player_index]
        if area==AreaType.DECK:return (obs.select.deck or [])[index]
        if area==AreaType.HAND:return (player.hand or [])[index]
        if area==AreaType.DISCARD:return (player.discard or [])[index]
        if area==AreaType.ACTIVE:return (player.active or [])[index]
        if area==AreaType.BENCH:return (player.bench or [])[index]
        if area==AreaType.PRIZE:return (player.prize or [])[index]
        if area==AreaType.STADIUM:return (obs.current.stadium or [])[index]
        if area==AreaType.LOOKING:return (obs.current.looking or [])[index]
    except Exception:return None
    return None

def source_id(select):
    c=getattr(select,'contextCard',None)
    if c is not None:return int(c.id)
    e=getattr(select,'effect',None)
    if e is not None:return int(e.id)
    return 0

def energy_count(p):return len(getattr(p,'energyCards',[]) or []) if p is not None else 0
def dark_count(p):return sum(1 for e in (getattr(p,'energyCards',[]) or []) if int(e.id)==DARK_ENERGY) if p is not None else 0
def damage_on(p):return max(0,int(getattr(p,'maxHp',0))-int(getattr(p,'hp',0))) if p is not None else 0

def prize_value(p):
    if p is None:return 0
    d=CARD.get(int(p.id));v=3 if bool(getattr(d,'megaEx',False)) else 2 if bool(getattr(d,'ex',False)) else 1
    if any(int(getattr(e,'id',-1))==12 for e in (getattr(p,'energyCards',[]) or [])):v-=1
    return max(0,v)
def has_ability(cid):return bool(getattr(CARD.get(int(cid)),'skills',[]) or [])
def min_attack_cost(cid):
    d=CARD.get(int(cid));vals=[]
    for aid in (getattr(d,'attacks',[]) or []):
        a=ATTACK.get(int(aid));vals.append(len(getattr(a,'energies',[]) or [])) if a is not None else None
    return min(vals) if vals else 1
def max_attack_damage(cid):
    d=CARD.get(int(cid));vals=[]
    for aid in (getattr(d,'attacks',[]) or []):
        a=ATTACK.get(int(aid));vals.append(int(getattr(a,'damage',0) or 0)) if a is not None else None
    return max(vals) if vals else 0
def ready(p,extra=0):return p is not None and energy_count(p)+extra>=min_attack_cost(int(p.id))
def stage_value(cid):
    d=CARD.get(int(cid));return 2 if bool(getattr(d,'stage2',False)) else 1 if bool(getattr(d,'stage1',False)) else 0

def simple_phase(obs):
    mine=obs.current.players[obs.current.yourIndex];pr=len(mine.prize or [])
    if pr<=2:return 'end'
    if obs.current.turn<=3:return 'open'
    if obs.current.turn<=7:return 'mid'
    return 'late'

def recognize(ids):
    ids=set(map(int,ids))
    if ids & {673,674,675,676,677,678}:return 'lucario'
    if ids & {119,120,121}:return 'dragapult'
    if ids & {741,742,743}:return 'alakazam'
    if ids & {169,190,666}:return 'archaludon'
    if ids & {92,93,96,708,709,710}:return 'hydrapple'
    if ids & {646,647,648}:return 'marnie'
    if ids & {344,345}:return 'crustle'
    return 'unknown'

def update_memory(obs,memory,forced_arch=None):
    st=obs.current;me=st.yourIndex;opp=st.players[1-me]
    opp_pr=len(opp.prize or [])
    if st.turn!=memory.get('last_turn',-1):
        memory['recent_ko']=memory.get('last_turn',-1)>=0 and opp_pr<memory.get('last_prizes',6)
        memory['previous_turn_tokens']=list(memory.get('turn_tokens',[]) or [])[-5:]
        memory['turn_tokens']=[];memory['last_turn']=st.turn;memory['last_prizes']=opp_pr
    rev=set(memory.get('revealed',set()) or set())
    for _,_,p in field(opp):
        rev.add(int(p.id))
        for pre in (p.preEvolution or []):rev.add(int(pre.id))
    for c in (opp.discard or []):rev.add(int(c.id))
    for log in (obs.logs or []):
        if getattr(log,'playerIndex',None)==1-me:
            for k in ('cardId','cardIdTarget','cardIdActive','cardIdBench','cardIdBefore','cardIdAfter'):
                z=getattr(log,k,None)
                if z is not None:rev.add(int(z))
    memory['revealed']=rev
    a=recognize(rev)
    if forced_arch:a=forced_arch
    elif a=='unknown':a=memory.get('archetype','unknown')
    memory['archetype']=a
    return a

def enum_int(v,default=0):
    if isinstance(v,int):return v
    if hasattr(v,'value'):
        try:return int(v.value)
        except Exception:pass
    try:return int(v)
    except Exception:return default

def option_subject(obs,o):
    typ=o.type
    if typ==OptionType.ATTACK:return int(getattr(o,'attackId',0) or 0),0
    if typ in {OptionType.NUMBER,OptionType.YES,OptionType.NO,OptionType.END,OptionType.SKILL,OptionType.SPECIAL_CONDITION}:
        return int(getattr(o,'number',0) or 0),0
    area=o.area
    if typ==OptionType.PLAY and area is None:area=AreaType.HAND
    card=get_card(obs,area,o.index,o.playerIndex if o.playerIndex is not None else obs.current.yourIndex)
    target=get_card(obs,o.inPlayArea,o.inPlayIndex,obs.current.yourIndex)
    return int(getattr(card,'id',0) or 0),int(getattr(target,'id',0) or 0)

def extended_action_desc(obs,o):
    c=enum_int(obs.select.context);s=source_id(obs.select);t=enum_int(o.type);u,v=option_subject(obs,o)
    def iv(name,default=-1):
        z=getattr(o,name,None)
        try:return int(z) if z is not None else default
        except Exception:return default
    return [c,s,t,u,v,iv('area'),iv('index'),iv('playerIndex'),iv('toolIndex'),iv('energyIndex'),iv('count',0),iv('inPlayArea'),iv('inPlayIndex'),iv('attackId',0),iv('cardId',0),iv('number',0),iv('specialConditionType')]

def token_from_desc(d):
    c,s,t,u,v=(list(d)+[0]*5)[:5]
    if t==enum_int(OptionType.PLAY):return f'P:{u}'
    if t==enum_int(OptionType.ATTACH):return f'E:{u}>{v}'
    if t==enum_int(OptionType.ATTACK):return f'A:{u}'
    if t==enum_int(OptionType.SKILL):return f'B:{s}'
    if t==enum_int(OptionType.END):return 'Z'
    return f'K:{c}>{u}'

def record_action(memory,descs,selected):
    toks=memory.setdefault('turn_tokens',[])
    for i in selected or []:
        if 0<=int(i)<len(descs):toks.append(token_from_desc(descs[int(i)]))
    if len(toks)>8:memory['turn_tokens']=toks[-8:]

def _bits(z,n=11):
    z=max(0,int(z or 0));return [float((z>>b)&1) for b in range(n)]
def _slot_features(p,own):
    if p is None:return ([1.0]+[0.0]*(1+len(OWN_POKEMON))+[0.0]*11+[0.0]*11) if own else ([0.0]*11+[0.0]*11)
    cid=int(p.id);mx=max(1,int(p.maxHp or 1));hp=max(0,int(p.hp or 0));dmg=max(0,mx-hp)
    num=[hp/400.,dmg/400.,mx/400.,dark_count(p)/5.,energy_count(p)/5.,float(ready(p)),prize_value(p)/3.,float(has_ability(cid)),len(p.tools or [])/3.,len(p.preEvolution or [])/3.,float(bool(p.appearThisTurn))]
    if own:return [0.0,float(cid not in OWN_POKEMON)]+[float(cid==x) for x in OWN_POKEMON]+_bits(cid)+num
    return _bits(cid)+num

def _parse_token(token):
    if not token:return 'NONE',0,0
    if token in ('R','Z'):return token,0,0
    m=re.match(r'([PEABK]):(\d+)(?:>(\d+))?',str(token))
    return (m.group(1),int(m.group(2)),int(m.group(3) or 0)) if m else ('NONE',0,0)
def _token_features(token):
    typ,a,b=_parse_token(token);return [float(typ==x) for x in TOKEN_TYPES]+_bits(a)+_bits(b)

def base_features(obs):
    st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me];mf=field(mine);of=field(op)
    fc=Counter(int(p.id) for _,_,p in mf);hand=Counter(int(c.id) for c in (mine.hand or []));disc=Counter(int(c.id) for c in (mine.discard or []))
    active=mine.active[0] if mine.active else None;oa=op.active[0] if op.active else None
    ready_grim=sum(1 for _,_,p in mf if int(p.id)==GRIMMSNARL_EX and ready(p));my_ready=sum(ready(p) for _,_,p in mf);opp_ready=sum(ready(p) for _,_,p in of)
    charged=sum(1 for _,_,p in mf if int(p.id)==MUNKIDORI and dark_count(p)>=1);bench_slots=max(0,int(mine.benchMax or 5)-len(mine.bench or []))
    grim_now=int((fc[IMPIDIMP]>0 and hand[RARE_CANDY]>0 and hand[GRIMMSNARL_EX]>0) or (fc[MORGREM]>0 and hand[GRIMMSNARL_EX]>0));grim_next=int(bench_slots>0 and hand[IMPIDIMP]>0 and (hand[MORGREM]>0 or (hand[RARE_CANDY]>0 and hand[GRIMMSNARL_EX]>0)))
    fros_now=int(fc[SNORUNT]>fc[FROSLASS] and hand[FROSLASS]>0);fros_next=int(bench_slots>0 and hand[SNORUNT]>0 and hand[FROSLASS]>0)
    recovery=int(hand[NIGHT_STRETCHER]>0 and any(disc[x]>0 for x in (IMPIDIMP,MORGREM,GRIMMSNARL_EX,SNORUNT,FROSLASS,MUNKIDORI,DARK_ENERGY)))
    my_energy=sum(dark_count(p) for _,_,p in mf);useful=sum(min(dark_count(p),2 if int(p.id) in MARNIE_LINE else 1 if int(p.id)==MUNKIDORI else max(1,min_attack_cost(int(p.id)))) for _,_,p in mf)
    damage_pool=sum(damage_on(p) for _,_,p in mf);opp_damage=sum(damage_on(p) for _,_,p in of)
    ko30=sum(1 for _,_,p in of if 0<int(p.hp)<=30);ko180=sum(1 for _,_,p in of if 0<int(p.hp)<=180);ko210=sum(1 for _,_,p in of if 0<int(p.hp)<=210);ko180pr=sum(prize_value(p) for _,_,p in of if 0<int(p.hp)<=180)
    oppabilities=sum(has_ability(int(p.id)) for _,_,p in of);opprule=sum(prize_value(p) for _,_,p in of if prize_value(p)>1)
    dead=0
    if hand[FROSLASS] and fc[SNORUNT]==0 and hand[SNORUNT]==0 and not (recovery and disc[SNORUNT]):dead+=hand[FROSLASS]
    if hand[GRIMMSNARL_EX] and fc[IMPIDIMP]+fc[MORGREM]==0 and hand[IMPIDIMP]+hand[MORGREM]==0:dead+=hand[GRIMMSNARL_EX]
    redundant=sum(max(0,n-(3 if cid==DARK_ENERGY else 2)) for cid,n in hand.items())
    search_ids={BUDDY_BUDDY_POFFIN,POKE_PAD,POKEGEAR,RARE_CANDY};draw_ids={LILLIES_DETERMINATION,TEAM_ROCKETS_PETREL,DAWN};disrupt_ids={JUDGE,UNFAIR_STAMP}
    attack_margin=(max_attack_damage(int(active.id))-int(oa.hp))/100. if active and oa else 0.;survival=(int(active.hp)-max((max_attack_damage(int(p.id)) for _,_,p in of if ready(p,1)),default=0))/100. if active else 0.
    stadium=int(st.stadium[0].id) if st.stadium else 0
    vals=[st.turn/10.,6-len(mine.prize or []),6-len(op.prize or []),len(op.prize or [])-len(mine.prize or []),mine.handCount,op.handCount,mine.handCount-op.handCount,mine.deckCount/10.,op.deckCount/10.,fc[IMPIDIMP],fc[MORGREM],fc[GRIMMSNARL_EX],fc[SNORUNT],fc[FROSLASS],fc[MUNKIDORI],charged,ready_grim,my_ready,max(0,my_ready-1),opp_ready,float(ready(oa)) if oa else 0,grim_now,grim_next,fros_now,fros_next,recovery,bench_slots,float(bench_slots==0),my_energy,useful,max(0,my_energy-useful),hand[DARK_ENERGY],damage_pool/30.,opp_damage/30.,min(damage_pool,charged*30)/30.,ko30,ko180,ko210,ko180pr,oppabilities,opprule,fc[FROSLASS]*oppabilities,dead,redundant,sum(hand[x] for x in search_ids),sum(hand[x] for x in draw_ids),sum(hand[x] for x in disrupt_ids),hand[BOSSS_ORDERS],float(st.supporterPlayed),float(st.energyAttached),float(stadium==SPIKEMUTH_GYM),float(ready(active)) if active else 0,0.0,int(active.hp)/100. if active else 0,int(oa.hp)/100. if oa else 0,attack_margin,survival,len(obs.select.option or []),len(mine.bench or []),len(op.bench or []),fc[IMPIDIMP]+fc[SNORUNT],fc[MORGREM]+fc[GRIMMSNARL_EX]+fc[FROSLASS],sum(energy_count(p) for _,_,p in of),sum(energy_count(p) for _,_,p in mf),float(mine.handCount<=3),float(len(mine.prize or [])<=2 or len(op.prize or [])<=2)]
    return [float(x) for x in vals]

def rich_state_features(obs,memory):
    st=obs.current;me=st.yourIndex;mine=st.players[me];opp=st.players[1-me];out=base_features(obs)
    hand=Counter(int(c.id) for c in (mine.hand or []));disc=Counter(int(c.id) for c in (mine.discard or []));out.extend(float(hand[i]) for i in OWN_IDS);out.extend(float(disc[i]) for i in OWN_IDS)
    oslots=(list(mine.active or [])+list(mine.bench or [])+[None]*SLOT_COUNT)[:SLOT_COUNT];pslots=(list(opp.active or [])+list(opp.bench or [])+[None]*SLOT_COUNT)[:SLOT_COUNT]
    for p in oslots:out.extend(_slot_features(p,True))
    for p in pslots:out.extend(_slot_features(p,False))
    h=[0.0]*HASH_OPP_DISCARD
    for c in (opp.discard or []):h[int(c.id)%HASH_OPP_DISCARD]+=1.;out.extend([])
    out.extend(h);rh=[0.0]*HASH_REVEALED
    for cid in memory.get('revealed',set()) or set():rh[int(cid)%HASH_REVEALED]=1.
    out.extend(rh);tc=[0.0]*24;lh=[0.0]*HASH_LEGAL
    for o in obs.select.option or []:
        d=extended_action_desc(obs,o);tc[max(0,min(23,int(d[2])))]+=1.;lh[int(d[3])%HASH_LEGAL]+=1.
    out.extend(tc);out.extend(lh)
    cur=list(memory.get('turn_tokens',[]) or [])[-5:];prev=list(memory.get('previous_turn_tokens',[]) or [])[-5:];toks=[None]*(5-len(cur))+cur+[None]*(5-len(prev))+prev
    for tok in toks:out.extend(_token_features(tok))
    out.extend([float(bool(memory.get('recent_ko',False))),float(memory.get('last_prizes',6))/6.,float(memory.get('setup_active_id',0) or 0)/1400.,float(len(memory.get('revealed',set()) or set()))/20.,float(len(cur))/5.,float(len(prev))/5.])
    return out

def board_value(obs):
    st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
    v=(len(op.prize or [])-len(mine.prize or []))*8.0+(mine.handCount-op.handCount)*0.15
    for _,_,p in field(mine):v+=prize_value(p)*0.6+stage_value(int(p.id))*0.4+energy_count(p)*0.35+float(ready(p))*1.2-damage_on(p)/250.
    for _,_,p in field(op):v-=prize_value(p)*0.55+stage_value(int(p.id))*0.35+energy_count(p)*0.3+float(ready(p))*1.0-damage_on(p)/280.
    return float(v)
