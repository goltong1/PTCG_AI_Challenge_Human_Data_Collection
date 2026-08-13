from __future__ import annotations
import os,sys,importlib.util,hashlib
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:sys.path.insert(0,R)

def _load(tag):
    fn=os.path.join(R,f'policy_{tag}.py')
    name='_tera_exact_'+tag+'_'+hashlib.sha1((R+tag).encode()).hexdigest()[:10]
    sp=importlib.util.spec_from_file_location(name,fn);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

NAMES=['generic','archaludon','crustle','dragapult','marnie','alakazam','spidops','grass_ogerpon','dusk','okidogi','cynthia','dipplin','lopunny','lucario']
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
    return POL.get(_route,POL['generic']).agent(d)


# === v16 one-step exact-engine counterfactual selector ======================
import copy as _cf_copy, random as _cf_random
from dataclasses import asdict as _cf_asdict
from collections import Counter as _cf_Counter
from cg.api import search_begin as _cf_begin,search_step as _cf_step,search_end as _cf_end,search_release as _cf_release,to_observation_class as _cf_obs,SelectContext,AreaType
_CF_DECKS={'alakazam': [741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 305, 305, 305, 66, 66, 140, 272, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1086, 1079, 1079, 1079, 1079, 1097, 1129, 1081, 1081, 1156, 1182, 1231, 1231, 1231, 1231, 1225, 1225, 1225, 1225, 1197, 1264, 1264, 1264, 1264, 5, 5, 5, 19, 19, 19, 19, 13, 245, 1120, 1120], 'crustle': [756, 756, 756, 756, 344, 344, 344, 345, 345, 345, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1182, 1219, 1219, 1219, 1219, 1225, 1225, 1186, 1186, 1197, 1212, 1190, 1204, 1147, 1147, 1147, 1147, 1122, 1122, 1122, 1086, 1086, 1121, 1123, 1087, 1159, 1161, 1257, 1242, 1245, 14, 14, 14, 14, 18, 18, 18, 18, 11, 11, 11, 11, 1], 'dragapult': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 235, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1182, 1182, 1182, 1198, 1198, 1198, 1227, 1227, 1227, 1227, 1152, 5, 5, 5, 7, 112, 7, 7, 112, 1260, 1097, 1120, 112, 2, 2, 2, 1097, 1097, 1120, 2, 121, 1198, 1079], 'dusk': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 131, 131, 132, 132, 133, 235, 140, 1071, 112, 1227, 1227, 1227, 1227, 1198, 1198, 1198, 1182, 1182, 1231, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1086, 1120, 1120, 1120, 1120, 1097, 1097, 1080, 1213, 1161, 1256, 1246, 5, 5, 5, 2, 2, 2, 7, 7], 'okidogi': [116, 116, 116, 116, 676, 676, 676, 675, 675, 112, 117, 135, 135, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 6, 1197, 1122, 1188, 1194, 1194, 1152, 1152, 1213, 1142, 1142, 1142, 1142, 1122, 1071, 1097, 1141, 1118, 1174, 1247, 1264, 1264, 1264, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 6]}
_CF_MATCHES={'alakazam','crustle','dragapult','dusk','okidogi'}
_CF_MARGIN=90000.0
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
    return {'crustle':{31,230,117},'okidogi':{117},'dragapult':{756,272},'dusk':{756,272},'alakazam':{756,117}}.get(match,{756})

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
        if p.id in tech:v+=16000+rd*100
    for p in opp:
        v-=int(p.hp or 0)*16;v+=max(0,int(p.maxHp or 0)-int(p.hp or 0))*100
    v+=int(a.handCount or 0)*1600-int(b.handCount or 0)*350
    # Reward actual role completion, not raw bench width.
    ids={p.id for p in own}
    v+=(18000 if 184 in ids else 0)+(16000 if 756 in ids else 0)+(12000 if 96 in ids else 0)
    if len(own)>6:v-=7000*(len(own)-6)
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
    mod=POL.get(_route,POL['generic']);base=mod.agent(d)
    return _cf_choose(d,mod,base,_route) if _route in _CF_MATCHES else base
