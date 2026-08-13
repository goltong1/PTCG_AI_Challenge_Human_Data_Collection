from __future__ import annotations
import os,sys,importlib.util,hashlib
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:sys.path.insert(0,R)

def _load(tag):
    fn=os.path.join(R,f'policy_{tag}.py')
    name='_tera_exact_'+tag+'_'+hashlib.sha1((R+tag).encode()).hexdigest()[:10]
    sp=importlib.util.spec_from_file_location(name,fn);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

NAMES=['generic','archaludon','crustle','dragapult','marnie','alakazam','spidops','grass_ogerpon','dusk','okidogi','cynthia','dipplin','lopunny','lucario','special_alakazam']
POL={n:_load(n) for n in NAMES}
SIG={
 'marnie':{104,646,647,648,860,1259},
 'archaludon':{57,169,190,666,1244},
 'crustle':{58,343,344,345,1264},
 'alakazam':{245,272,741,742,743},
 'spidops':{400,401,414,431,434},
 'grass_ogerpon':{96,10,11,25,1127},
 'okidogi':{116,135},
 'cynthia':{341,342,379,380,381,387},
 'dipplin':{89,90,92,93},
 'lopunny':{174,848,849},
 'lucario':{333,677,678},
}
DRAG={119,120,121,235};DUSK={130,131,132,133}
_seen=set();_route='generic'

def _reset():
    global _seen,_route
    _seen=set();_route='generic'
    init={'current':None,'logs':[],'select':None,'step':0}
    for m in POL.values():
        try:m.agent(init)
        except Exception:pass

def _observe(d):
    global _route
    cur=d.get('current') if isinstance(d,dict) else None
    if not cur:return
    me=int(cur.get('yourIndex',0));pls=cur.get('players') or []
    if len(pls)<2:return
    op=pls[1-me]
    for z in ('active','bench','discard','lostZone'):
        for c in op.get(z) or []:
            if c and c.get('id') is not None:_seen.add(int(c['id']))
    for c in cur.get('stadium') or []:
        if c and c.get('id') is not None:_seen.add(int(c['id']))
    if _seen & DUSK:_route='dusk';return
    for n,s in SIG.items():
        if _seen & s:_route=n;return
    if _seen & DRAG:_route='dragapult'

def agent(d):
    if d.get('select') is None and d.get('current') is None:
        _reset();return POL['generic'].agent(d)
    _observe(d)
    return _policy_mod().agent(d)

def _policy_mod():
    # Exact weak-deck specialists.  The public signature remains the routing
    # key; this mapping only selects the policy core after that signature has
    # appeared on the public board/discard.
    tag={
        'dragapult':'lucario',
        'lucario':'lucario',
        'alakazam':'special_alakazam',
        'crustle':'crustle',
    }.get(_route,_route)
    return POL.get(tag,POL['generic'])


# === v16 one-step exact-engine counterfactual selector ======================
import copy as _cf_copy, random as _cf_random
from dataclasses import asdict as _cf_asdict
from collections import Counter as _cf_Counter
from cg.api import search_begin as _cf_begin,search_step as _cf_step,search_end as _cf_end,search_release as _cf_release,to_observation_class as _cf_obs,SelectContext,AreaType
_CF_DECKS={
 'dragapult':[119,119,119,119,120,120,120,120,121,121,121,140,235,1079,1079,1080,1086,1086,1086,1086,1121,1121,1121,1121,1152,1152,1152,1182,1182,1182,1198,1198,1198,1227,1227,1227,1227,1152,5,5,5,7,112,7,7,112,1260,1097,1120,112,2,2,2,1097,1097,1120,2,121,1198,1079],
 'lucario':[333,333,333,677,678,678,678,676,676,675,675,305,305,305,66,66,306,1141,1141,1141,1141,1142,1142,1142,1142,1152,1152,1152,1152,1086,1086,1086,1213,1197,678,1174,1159,1227,1227,1227,1227,1225,1225,1225,1182,1182,1182,1252,1211,6,6,6,6,6,6,6,6,6,20,6],
 'alakazam':[741,741,741,741,742,742,742,742,743,743,743,305,305,305,66,66,140,272,1152,1152,1152,1152,1086,1086,1086,1086,1079,1079,1079,1079,1097,1129,1081,1081,1156,1182,1231,1231,1231,1231,1225,1225,1225,1225,1197,1264,1264,1264,1264,5,5,5,19,19,19,19,13,245,1120,1120],
 'crustle':[344,344,344,344,345,345,345,345,117,117,117,343,1086,1219,1219,1219,1219,1227,1227,1227,1227,1182,1182,1182,1225,1225,1186,1197,1147,1147,1147,1147,1122,1122,1122,1086,1086,1121,1121,1123,1159,1137,1264,1264,1264,11,11,11,11,18,18,18,14,20,1,6,14,1,6,117],
}
_CF_MATCHES={'dragapult','lucario','alakazam','crustle'}
_CF_MARGIN=50000.0
_CF_ROLE_SCALE=130000.0
_CF_MAX=5
_CF_SEEN={}
_CF_STATS={'calls':0,'overrides':0,'errors':0}

def _cf_snapshot(mod):
    z={}
    for k,v in mod.__dict__.items():
        if k.endswith('MEM') and isinstance(v,(dict,set,list)):z[k]=_cf_copy.deepcopy(v)
    return z

def _cf_restore(mod,z):
    for k,v in z.items():
        if k in mod.__dict__:
            cur=mod.__dict__[k]
            if isinstance(cur,dict):cur.clear();cur.update(_cf_copy.deepcopy(v))
            elif isinstance(cur,set):cur.clear();cur.update(_cf_copy.deepcopy(v))
            elif isinstance(cur,list):cur[:]=_cf_copy.deepcopy(v)

def _cf_add_pokemon(cnt,p):
    if p is None:return
    cnt[int(p.id)]+=1
    for q in (getattr(p,'energyCards',None) or []):cnt[int(q.id)]+=1
    for q in (getattr(p,'tools',None) or []):cnt[int(q.id)]+=1
    for q in (getattr(p,'preEvolution',None) or []):cnt[int(q.id)]+=1

def _cf_unknown(full,pl,state,own):
    cnt=_cf_Counter(map(int,full))
    if own:
        for q in (pl.hand or []):cnt[int(q.id)]-=1
    for q in list(pl.active or [])+list(pl.bench or []):_cf_add_pokemon(cnt,q)
    for q in (pl.discard or []):cnt[int(q.id)]-=1
    for q in (getattr(pl,'lostZone',None) or []):cnt[int(q.id)]-=1
    for q in (state.stadium or []):
        if getattr(q,'playerIndex',None)==state.yourIndex if own else getattr(q,'playerIndex',None)==1-state.yourIndex:cnt[int(q.id)]-=1
    rem=[]
    for cid,n in cnt.items():rem += [cid]*max(0,n)
    need=int(pl.deckCount or 0)+len(pl.prize or [])+(0 if own else int(pl.handCount or 0))
    if len(rem)<need:rem += list(map(int,full))*((need-len(rem))//max(1,len(full))+1)
    seed=int(state.turn or 0)*1009+int(state.turnActionCount or 0)*917+sum(rem[:20])
    _cf_random.Random(seed).shuffle(rem)
    if own:
        return rem[:int(pl.deckCount or 0)],rem[int(pl.deckCount or 0):int(pl.deckCount or 0)+len(pl.prize or [])],[],[]
    h=int(pl.handCount or 0);dc=int(pl.deckCount or 0);pc=len(pl.prize or [])
    return rem[h:h+dc],rem[h+dc:h+dc+pc],rem[:h],[]

def _cf_hidden(obs,mod,match):
    s=obs.current;me=s.yourIndex;a=s.players[me];b=s.players[1-me]
    yd,yp,_,_=_cf_unknown(mod.MY_DECK,a,s,True)
    od,op,oh,oa=_cf_unknown(_CF_DECKS[match],b,s,False)
    if b.active and b.active[0] is None:
        # Choose a legal known-basic prediction from the opponent deck.
        for cid in od+oh:
            cd=mod.CARDS.get(cid)
            if cd and cd.basic:oa=[cid];break
    return yd,yp,od,op,oh,oa

def _cf_role(match):
    return {
        'crustle':{31,117},
        'dragapult':{756,272},
        'lucario':{756,272},
        'alakazam':{96,117,31},
    }.get(match,{756})

def _cf_ready(mod,p):
    try:return mod._x_ready_damage(_CF_MOD_OBS,p)
    except Exception:
        try:return mod._ready_damage(p)
        except Exception:return 0

def _cf_value(mod,obs,me,match,first_type):
    global _CF_MOD_OBS
    _CF_MOD_OBS=obs
    s=obs.current
    if s is None:return -1e12
    if s.result>=0:
        if s.result==2:return 0
        return 1e9 if s.result==me else -1e9
    a=s.players[me];b=s.players[1-me];v=0.0
    v+=(len(b.prize or [])-len(a.prize or []))*180000
    own=[p for p in list(a.active or [])+list(a.bench or []) if p];opp=[p for p in list(b.active or [])+list(b.bench or []) if p]
    tech=_cf_role(match)
    for p in own:
        rd=_cf_ready(mod,p);active=bool(a.active and a.active[0] is p)
        v+=int(p.hp or 0)*35+len(p.energyCards or [])*5000+rd*170
        if rd>=10:v+=12000
        if rd>=60:v+=22000
        if active and rd>=10:v+=40000
        if p.id==184:v+=22000
        if p.id==756:v+=24000+(12000 if active else 0)+min(3,len(p.energyCards or []))*7000
        if p.id==96:v+=13000+len(p.energyCards or [])*3500
        if p.id in tech:
            v+=_CF_ROLE_SCALE+rd*220
            if rd>=10:v+=_CF_ROLE_SCALE*0.45
            if active:v+=_CF_ROLE_SCALE*0.35
    for p in opp:
        v-=int(p.hp or 0)*16;v+=max(0,int(p.maxHp or 0)-int(p.hp or 0))*100
    v+=int(a.handCount or 0)*1600-int(b.handCount or 0)*350
    # Reward actual role completion, not raw bench width.
    ids={p.id for p in own}
    v+=(18000 if 184 in ids else 0)+(16000 if 756 in ids else 0)+(12000 if 96 in ids else 0)
    if len(own)>6:v-=7000*(len(own)-6)
    if match=='okidogi' and len(own)>5:v-=18000*(len(own)-5)
    if first_type==13:v+=45000
    elif first_type==14:v-=30000
    return v

def _cf_candidates(mod,obs,base):
    vals=[];seen=set()
    for i,o in enumerate(obs.select.option):
        c=mod._source(obs,o);cid=int(getattr(c,'id',0) or 0);key=(int(o.type),cid,int(getattr(o,'attackId',0) or 0),int(getattr(o,'inPlayArea',-1) or -1),int(getattr(o,'inPlayIndex',-1) or -1))
        if key in seen:continue
        seen.add(key)
        pri={13:100000,10:70000,8:65000,12:60000,9:55000,7:40000,14:-10000}.get(int(o.type),0)
        try:pri+=float(mod._main_score(obs,o))
        except Exception:pass
        vals.append((pri,i))
    vals.sort(reverse=True)
    inds=[base[0]] if base and len(base)==1 else []
    for _,i in vals:
        if i not in inds:inds.append(i)
        if len(inds)>=_CF_MAX:break
    return inds

def _cf_branch(mod,obs,idx,match):
    snap=_cf_snapshot(mod);sid=None
    try:
        h=_cf_hidden(obs,mod,match);root=_cf_begin(obs,*h);sid=root.searchId
        st=_cf_step(sid,[idx]);_cf_release(sid);sid=st.searchId
        me=obs.current.yourIndex;steps=0
        while steps<12:
            o=st.observation
            if o.current is None or o.current.result>=0 or o.current.yourIndex!=me or o.select is None or o.select.context==SelectContext.MAIN:break
            d=_cf_asdict(o);act=mod.agent(d)
            if not act:break
            st2=_cf_step(sid,act);_cf_release(sid);st=st2;sid=st.searchId;steps+=1
        return _cf_value(mod,st.observation,me,match,int(obs.select.option[idx].type))
    except Exception:
        _CF_STATS['errors']+=1;return -1e15
    finally:
        if sid is not None:
            try:_cf_release(sid)
            except Exception:pass
        try:_cf_end()
        except Exception:pass
        _cf_restore(mod,snap)

def _cf_choose(d,mod,base,match):
    try:obs=_cf_obs(d)
    except Exception:return base
    if obs.current is None or obs.select is None or obs.select.context!=SelectContext.MAIN or not base or len(base)!=1:return base
    turn=int(obs.current.turn or 0);key=(turn,int(obs.current.turnActionCount or 0)//4)
    if _CF_SEEN.get(match)==key:return base
    _CF_SEEN[match]=key;inds=_cf_candidates(mod,obs,base)
    if len(inds)<2:return base
    _CF_STATS['calls']+=1;vals={i:_cf_branch(mod,obs,i,match) for i in inds}
    bi=base[0];best=max(vals,key=vals.get)
    if best!=bi and vals[best]>=vals.get(bi,-1e15)+_CF_MARGIN:
        _CF_STATS['overrides']+=1;return [best]
    return base

_CF_OLD_PUBLIC=agent
def agent(d):
    if d.get('select') is None and d.get('current') is None:
        _CF_SEEN.clear();return _CF_OLD_PUBLIC(d)
    _observe(d)
    mod=_policy_mod();base=mod.agent(d)
    return _cf_choose(d,mod,base,_route) if _route in _CF_MATCHES else base


# === v16 replay-contrast value residual for Dragapult =======================
_V16_CFR_OLD_ROLE=_cf_role
_V16_CFR_OLD_VALUE=_cf_value
_V16_CFR_ROLE='clef'
_V16_CFR_LOW=15000
_V16_CFR_INFRA=60000
_V16_CFR_FOLLOW=35000
_CF_MARGIN=float(130000)

def _cf_role(match):
    if match=='dragapult':
        if _V16_CFR_ROLE=='teal':return {756,96}
        if _V16_CFR_ROLE=='mix':return {756,96,272}
        return {756,272}
    return _V16_CFR_OLD_ROLE(match)

def _cf_value(mod,obs,me,match,first_type):
    v=_V16_CFR_OLD_VALUE(mod,obs,me,match,first_type)
    try:
        if match!='dragapult' or obs.current is None or obs.current.result>=0:return v
        s=obs.current;a=s.players[me];turn=int(s.turn or 0)
        own=[p for p in list(a.active or [])+list(a.bench or []) if p]
        ids={int(p.id) for p in own};bench=[p for p in (a.bench or []) if p]
        # New 200-replay contrast: Kangaskhan 1.83 vs 2.81 turns and Latias
        # 2.36 vs 4.26 turns were the largest infrastructure differences.
        if turn<=4:
            v += _V16_CFR_INFRA if 756 in ids else -_V16_CFR_INFRA
            v += _V16_CFR_INFRA*0.85 if 184 in ids else -_V16_CFR_INFRA*0.85
        # Phantom Dive makes every low-HP utility a future prize and damage sink.
        low=sum(int(p.hp or 0)<=120 for p in bench)
        v -= _V16_CFR_LOW*low
        if len(bench)>4:v-=_V16_CFR_LOW*0.75*(len(bench)-4)
        k=next((p for p in own if p.id==756),None)
        kready=bool(k and _cf_ready(mod,k)>=60)
        follow=[p for p in own if p.id in ({96,272} if _V16_CFR_ROLE=='mix' else {96} if _V16_CFR_ROLE=='teal' else {272})]
        fready=max([_cf_ready(mod,p) for p in follow] or [0])
        if kready:
            v += _V16_CFR_FOLLOW if fready>=60 else -_V16_CFR_FOLLOW*0.55
        # Prefer actual Active pressure over a large but inert board.
        active=a.active[0] if a.active else None
        if active and _cf_ready(mod,active)>=60:v+=_V16_CFR_FOLLOW*0.45
    except Exception:pass
    return v


# === v19 weak-deck replay specialists ======================================
_V19_CF_VALUE=_cf_value
_CF_MARGIN=float(50000)
_CF_MAX=7

def _cf_value(mod,obs,me,match,first_type):
    """One-step value residual learned from exact win/loss replay contrasts."""
    v=_V19_CF_VALUE(mod,obs,me,match,first_type)
    try:
        if obs.current is None or obs.current.result>=0:return v
        s=obs.current;a=s.players[me];b=s.players[1-me];turn=int(s.turn or 0)
        own=[p for p in list(a.active or [])+list(a.bench or []) if p]
        opp=[p for p in list(b.active or [])+list(b.bench or []) if p]
        ids={int(p.id) for p in own}
        active=a.active[0] if a.active else None
        ready=_cf_ready(mod,active) if active else 0

        # The exact replays separate mainly on first-ready timing, not raw
        # board size.  Front-load completed attackers and penalize inert slots.
        if ready>=60:v+=52000
        if ready>=140:v+=26000
        inert=sum(_cf_ready(mod,p)<=0 and int(p.id) not in {184,96,1071} for p in own)
        v-=inert*7000

        if match in {'dragapult','lucario'}:
            kang=next((p for p in own if int(p.id)==756),None)
            clef=next((p for p in own if int(p.id)==272),None)
            if turn<=6:
                v+=(62000 if kang else -31000)
                v+=(72000 if clef else -36000)
            if kang:
                kr=_cf_ready(mod,kang);v+=kr*260+(38000 if kr>=60 else 0)
            if clef:
                cr=_cf_ready(mod,clef);v+=cr*330+(52000 if cr>=60 else 0)
                if active is clef:v+=30000
                oa=b.active[0] if b.active else None
                # Fairy Zone doubles Clefairy's attack into both exact Dragon
                # lines, so reward the actual knockout threshold.
                if oa and int(oa.id) in ({119,120,121,235} if match=='dragapult' else {333,677,678}):
                    dealt=cr*2 if match=='dragapult' else cr
                    if dealt>=int(oa.hp or 0):v+=190000

        elif match=='crustle':
            corner=next((p for p in own if int(p.id)==117),None)
            chiyu=next((p for p in own if int(p.id)==31),None)
            if turn<=7:v+=(115000 if corner else -70000)
            if corner:
                rr=_cf_ready(mod,corner);v+=rr*360+(90000 if rr>=140 else 0)
                if active is corner:v+=42000
            if chiyu:
                rr=_cf_ready(mod,chiyu);v+=rr*230+(30000 if rr>0 else 0)
            # Low-HP utility benches were 36% more common in exact losses and
            # feed Crustle's spread/prize map.
            v-=22000*sum(int(p.hp or 0)<=120 for p in (a.bench or []) if p)

        elif match=='alakazam':
            teal=[p for p in own if int(p.id)==96]
            corner=next((p for p in own if int(p.id)==117),None)
            chiyu=next((p for p in own if int(p.id)==31),None)
            if turn<=6:v+=(52000 if teal else -30000)
            v+=sum(_cf_ready(mod,p)*220 for p in teal)
            if corner:v+=_cf_ready(mod,corner)*250+24000
            if chiyu:v+=_cf_ready(mod,chiyu)*210+16000
            # Powerful Hand damage is proportional to our hand.  Spending
            # setup resources before attacking reduces incoming counters.
            v-=max(0,int(a.handCount or 0)-4)*9000
            if int(b.handCount or 0)<=3:v+=18000

        # Reward a live prize conversion against evolution seeds/low-HP tech.
        if ready>0:
            for p in opp:
                if ready>=int(p.hp or 0):
                    cd=mod.CARDS.get(int(p.id));pr=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
                    v+=26000*pr
    except Exception:pass
    return v


# === v17 Clefairy threshold-aware counterfactual value ======================
_V17_CF_VALUE=_cf_value
_CF_MARGIN=float(70000)
_CF_MAX=7

def _cf_value(mod,obs,me,match,first_type):
    v=_V17_CF_VALUE(mod,obs,me,match,first_type)
    try:
        if match!='dragapult' or obs.current is None or obs.current.result>=0:return v
        s=obs.current;a=s.players[me];b=s.players[1-me]
        own=[p for p in list(a.active or [])+list(a.bench or []) if p]
        clef=next((p for p in own if int(p.id)==272),None)
        oa=b.active[0] if b.active else None
        if clef:
            damage=_cf_ready(mod,clef)
            active=bool(a.active and a.active[0] is clef)
            if damage>=60:
                v+=damage*260+(52000 if active else 26000)
                if oa and damage>=int(oa.hp or 0):
                    cd=mod.CARDS.get(int(oa.id));pr=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
                    v+=150000+pr*65000
                    # A real one-hit threshold justifies the fifth/sixth safe body.
                    v+=22000*max(0,len([p for p in (a.bench or []) if p])-4)
        # Bossing an evolution seed into a ready KO is worth its denied future
        # Dragapult, not merely the current 1-prize card.
        if oa and int(oa.id) in {119,120,235}:
            attacker=a.active[0] if a.active else None
            damage=_cf_ready(mod,attacker) if attacker else 0
            if damage>=int(oa.hp or 0):v+=135000
    except Exception:pass
    return v
