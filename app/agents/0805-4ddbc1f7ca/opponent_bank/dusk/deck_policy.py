from __future__ import annotations
import os,sys,types,inspect,json,math
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
_HERE=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if _HERE not in sys.path:sys.path.insert(0,_HERE)
from cg.api import (to_observation_class,SelectContext,OptionType,search_begin,search_step,
                    search_end,search_release)
import policy_v13,policy_v6
my_deck=list(policy_v13.my_deck)

DECKS={
'alakazam':[int(x) for x in '''741 741 741 741 742 742 742 742 743 743 743 305 305 305 66 66 140 142 272 343 1152 1152 1152 1152 1086 1086 1086 1086 1079 1079 1079 1097 1129 1081 1174 1156 1156 1182 1182 1231 1231 1231 1231 1225 1225 1225 1184 1197 1197 1264 1264 1264 1264 5 5 19 19 19 19 13'''.split()],
'archaludon':[int(x) for x in '''169 169 169 169 190 190 190 190 666 666 666 666 1244 57 1152 1152 1152 1152 1121 1121 1121 1121 1122 1122 1122 1122 1097 1097 1097 1147 1147 1147 1147 1159 1182 1182 1182 1182 1185 1185 1185 1185 1227 1227 1227 1227 1244 1244 1244 8 8 8 8 8 8 8 8 8 8 8'''.split()],
'crustle':[int(x) for x in '''756 756 756 756 344 344 344 345 345 345 1227 1227 1227 1227 1182 1182 1182 1182 1219 1219 1219 1219 1225 1225 1186 1186 1197 1212 1190 1204 1147 1147 1147 1147 1122 1122 1122 1086 1086 1121 1123 1087 1159 1161 1257 1242 1245 14 14 14 14 18 18 18 18 11 11 11 11 1'''.split()],
'dragapult':[int(x) for x in '''119 119 119 119 120 120 120 120 121 121 112 112 235 235 31 140 343 1071 689 1227 1227 1227 1227 1182 1182 1182 1198 1198 1213 1240 1121 1121 1121 1121 1086 1086 1086 1086 1152 1152 1152 1152 1097 1097 1097 1260 1080 1081 1137 1256 1256 2 2 2 2 5 5 5 7 7'''.split()],
'marnie':[int(x) for x in '''7 7 7 7 7 7 7 7 7 7 104 104 112 112 112 112 646 646 646 646 647 647 647 648 648 648 860 860 1079 1079 1079 1080 1086 1086 1086 1086 1097 1097 1097 1122 1137 1152 1152 1152 1152 1182 1182 1219 1219 1219 1219 1227 1227 1227 1227 1231 1259 1259 1259 1259'''.split()],
'zoroark':[int(x) for x in '''292 292 292 292 293 293 293 293 344 345 862 906 863 303 1227 1227 1227 1227 1182 1182 1182 1205 1213 1213 1086 1086 1086 1086 1152 1152 1152 1113 1113 1113 1113 1097 1097 1121 1121 1121 1121 1123 1123 1123 1123 1159 1253 1253 7 7 7 7 7 7 7 7 7 7 1152 1097'''.split()],
}
_seen=set();_marnie=False;_planned=[];_planned_turn=-1;_searched_turn=-1
_loop_sig=None;_loop_count=0

# MAIN card priority for candidate generation (not final scoring).
IMPORTANT={1198:100,1227:95,1231:90,1182:88,1121:85,1152:82,1086:80,1120:75,1097:70,1213:68,1080:66}
MATCHES=['alakazam','dragapult_old','archaludon','zoroark','crustle','marnie']
_VALUE_MODEL=json.load(open(os.path.join(_HERE,'value_model.json'),'r'))
_KEY=[119,120,121,131,132,133,235,140,1071,112,1227,1198,1182,1231,1121,1152,1086,1120,1097,1080,1213,1161,1256,1246,2,5,7]

def detect(obs):
 global _seen,_marnie
 s=obs.current
 if s is None:return None
 if s.turn==0:_seen.clear();_marnie=False
 me=s.yourIndex;op=s.players[1-me]
 for p in list(op.active)+list(op.bench):
  if p is not None:
   _seen.add(p.id);_seen.update(x.id for x in p.preEvolution)
 for l in obs.logs:
  if l.playerIndex!=me:
   for x in (l.cardId,l.cardIdAfter,l.cardIdActive,l.cardIdBench):
    if x and x>0:_seen.add(x)
 if _seen & {646,647,648,860}:_marnie=True;return 'marnie'
 if _seen & {741,742,743}:return 'alakazam'
 if _seen & {169,190,666}:return 'archaludon'
 if _seen & {292,293,862,906,863}:return 'zoroark'
 if 756 in _seen or (345 in _seen and not (_seen&{292,293})):return 'crustle'
 if _seen & {119,120,121}:return 'dragapult'
 return None

def _active_policy(name):return policy_v13

def _known(pl):
 a=[]
 if pl.hand:a += [c.id for c in pl.hand if c]
 a += [c.id for c in pl.discard if c]
 for p in list(pl.active)+list(pl.bench):
  if not p:continue
  a.append(p.id);a += [x.id for x in p.preEvolution];a += [x.id for x in p.energyCards];a += [x.id for x in p.tools]
 return a

def _remain(full,known,n):
 c=Counter(full)
 for x in known:
  if c[x]>0:c[x]-=1
 a=[]
 for k in sorted(c):a += [k]*c[k]
 if len(a)<n:a += [full[-1]]*(n-len(a))
 return a[:n]

def _hidden(obs,name):
 s=obs.current;me=s.yourIndex;a=s.players[me];b=s.players[1-me]
 r=_remain(my_deck,_known(a),a.deckCount+len(a.prize));rot=(s.turn*5+me)%max(1,len(r));r=r[rot:]+r[:rot]
 yp=r[:len(a.prize)];yd=r[len(a.prize):]
 q=_remain(DECKS[name],_known(b),b.deckCount+len(b.prize)+b.handCount);rot=(s.turn*7+me)%max(1,len(q));q=q[rot:]+q[:rot]
 oh=q[:b.handCount];op=q[b.handCount:b.handCount+len(b.prize)];od=q[b.handCount+len(b.prize):]
 return yd,yp,od,op,oh,[]

def _field_pokemon(obs,area,index):
 s=obs.current;me=s.yourIndex;pl=s.players[me]
 if area is None or index is None:return None
 if int(area)==4:return pl.active[index] if index<len(pl.active) else None
 if int(area)==5:return pl.bench[index] if index<len(pl.bench) else None
 return None

def _source_card(obs,opt):
 s=obs.current;me=s.yourIndex;pl=s.players[me]
 ar=int(opt.area) if opt.area is not None else None;idx=opt.index
 try:
  # MAIN PLAY options omit area but index the hand.
  if opt.type==OptionType.PLAY or ar==2:return pl.hand[idx] if pl.hand and idx is not None else None
  if ar==3:return pl.discard[idx]
  if ar==4:return pl.active[idx]
  if ar==5:return pl.bench[idx]
  if ar==7:return s.stadium[idx]
  if ar==12 and s.looking:return s.looking[idx]
  if ar==1 and obs.select.deck:return obs.select.deck[idx]
 except Exception:return None
 return None

def _sig(obs,opt):
 src=_source_card(obs,opt);tgt=_field_pokemon(obs,opt.inPlayArea,opt.inPlayIndex)
 return (int(opt.type),getattr(src,'id',opt.cardId),getattr(src,'serial',opt.serial),
         int(opt.area) if opt.area is not None else None,
         int(opt.inPlayArea) if opt.inPlayArea is not None else None,
         getattr(tgt,'id',None),getattr(tgt,'serial',None),opt.attackId,opt.number,
         opt.energyIndex,opt.toolIndex)

def _find(obs,sig):
 for i,o in enumerate(obs.select.option):
  if _sig(obs,o)==sig:return [i]
 return None

def _snapshot(mod):
 out={}
 skip={'my_deck','all_card','card_table'}
 for n,v in mod.__dict__.items():
  if n in skip or n.startswith('__') or n.isupper():continue
  if isinstance(v,(types.ModuleType,type)) or inspect.isfunction(v):continue
  if isinstance(v,(int,bool,str,float,list,dict,set,defaultdict,tuple)) or v.__class__.__name__=='AttackPlan':
   try:out[n]=deepcopy(v)
   except Exception:pass
 return out

def _restore(mod,s):
 for n,v in s.items():setattr(mod,n,v)

def _ready(p):
 if not p:return 0
 ids=[e.id for e in p.energyCards]
 return int(len(ids)>=2 or (p.id==121 and 2 in ids and 5 in ids))

def _field_counts(pl):
 c=Counter()
 for p in list(pl.active)+list(pl.bench):
  if not p:continue
  c[p.id]+=1
  for q in p.preEvolution:c[q.id]+=1
 return c

def _features(o,me,name):
 s=o.current;a=s.players[me];b=s.players[1-me]
 af=_field_counts(a);bf=_field_counts(b);hand=Counter(c.id for c in (a.hand or []));disc=Counter(c.id for c in a.discard)
 aa=a.active[0] if a.active else None;ba=b.active[0] if b.active else None
 f=[1.0,s.turn/20,s.turnActionCount/20,int(s.firstPlayer==me),int(s.supporterPlayed),int(s.energyAttached),int(s.retreated)]
 f += [len(a.prize)/6,len(b.prize)/6,(len(b.prize)-len(a.prize))/6,a.handCount/12,b.handCount/12,a.deckCount/60,b.deckCount/60]
 f += [len([p for p in a.bench if p])/5,len([p for p in b.bench if p])/5]
 f += [int(aa.id==121 if aa else 0),int(aa.id==235 if aa else 0),_ready(aa),int(ba.id in {121,190,293,648,743,345,756,860} if ba else 0),_ready(ba)]
 f += [sum(p.hp for p in list(a.active)+list(a.bench) if p)/1500,sum(p.hp for p in list(b.active)+list(b.bench) if p)/1500]
 f += [sum((p.maxHp-p.hp) for p in list(a.active)+list(a.bench) if p)/1000,sum((p.maxHp-p.hp) for p in list(b.active)+list(b.bench) if p)/1000]
 f += [sum(1 for p in list(a.active)+list(a.bench) if p and p.id==121 and _ready(p))/2,sum(1 for p in list(b.active)+list(b.bench) if p and _ready(p))/3]
 for k in _KEY:f.append(min(4,af[k])/4)
 for k in _KEY:f.append(min(4,hand[k])/4)
 for k in _KEY:f.append(min(4,disc[k])/4)
 for ids in ({741,742,743},{119,120,121},{169,190,666},{292,293},{344,345,756},{646,647,648,860}):f.append(min(3,sum(bf[x] for x in ids))/3)
 lab='dragapult_old' if name=='dragapult' else name
 f += [1.0 if lab==m else 0.0 for m in MATCHES]
 return f

def _tree(t,x):
 n=0
 while t['l'][n]!=-1:
  n=t['l'][n] if x[t['f'][n]]<=t['th'][n] else t['r'][n]
 return t['v'][n]

def _predict(o,me,name):
 x=_features(o,me,name);vals=[_tree(t,x) for t in _VALUE_MODEL['trees']]
 return sum(vals)/len(vals), (sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5

def _eval(o,root_turn,me,name):
 s=o.current
 if s is None:return -10**12
 if s.result>=0:return (10**12 if s.result==me else -10**12)
 p,sd=_predict(o,me,name)
 a=s.players[me];b=s.players[1-me]
 # learned value is primary; explicit prize/attack terms preserve tactical certainty off-distribution
 return p*1000000 + (len(b.prize)-len(a.prize))*160000 + sum(_ready(q) for q in a.active)*50000 - sd*60000

def _candidate_indices(obs,base):
 opts=obs.select.option;rank=[]
 for i,o in enumerate(opts):
  if o.type==OptionType.ATTACK:pri=130
  elif o.type==OptionType.EVOLVE:pri=120
  elif o.type==OptionType.ABILITY:pri=115
  elif o.type==OptionType.ATTACH:pri=105
  elif o.type==OptionType.RETREAT:pri=92
  elif o.type==OptionType.PLAY:pri=IMPORTANT.get(getattr(_source_card(obs,o),'id',o.cardId),40)
  elif o.type==OptionType.END:pri=10
  else:continue
  rank.append((pri,i))
 rank.sort(reverse=True)
 ids=[]
 if base:ids.append(base[0])
 for _,i in rank:
  if i not in ids:ids.append(i)
  if len(ids)>=5:break
 return ids

def _simulate(root,first_idx,pol,root_turn,me,name,maxn=12):
 st=search_step(root.searchId,[first_idx]);sid=st.searchId;seq=[]
 # first signature is recorded by caller; subsequent exact signatures form executable plan
 try:
  for _ in range(maxn):
   o=st.observation
   if o.current is None or o.current.result>=0 or o.current.turn!=root_turn or o.select is None or not o.select.option:break
   d=asdict(o);act=pol.agent(d)
   if not act:break
   # record every selected option by identity, including multi-select
   sigs=[_sig(o,o.select.option[i]) for i in act]
   seq.append(sigs)
   st2=search_step(sid,act);search_release(sid);st=st2;sid=st.searchId
  return _eval(st.observation,root_turn,me),seq
 finally:
  try:search_release(sid)
  except Exception:pass

def _plan(obs,name,pol,base):
 global _searched_turn
 s=obs.current
 if name not in {'crustle','marnie','archaludon'} or s.turn==_searched_turn or s.turn>8:return None
 # only plan at MAIN and only once per own turn
 if obs.select.context!=SelectContext.MAIN:return None
 inds=_candidate_indices(obs,base)
 if len(inds)<2:return None
 snap=_snapshot(pol)
 try:
  root=search_begin(obs,*_hidden(obs,name));vals=[]
  for i in inds:
   _restore(pol,deepcopy(snap))
   try:v,seq=_simulate(root,i,pol,s.turn,s.yourIndex,name)
   except Exception:continue
   vals.append((v,i,seq))
  search_end();_restore(pol,snap);_searched_turn=s.turn
  if not vals:return None
  vals.sort(reverse=True,key=lambda x:x[0]);best=vals[0]
  bval=next((v for v,i,q in vals if i==(base[0] if base else -1)),-10**15)
  # material threshold avoids replacing stable FSM for tiny heuristic differences
  if best[1]!=(base[0] if base else -1) and best[0]>=bval+100000:
   first=[_sig(obs,obs.select.option[best[1]])]
   return [first]+best[2]
 except Exception:
  try:search_end()
  except Exception:pass
  _restore(pol,snap)
 return None

def agent(d):
 global _planned,_planned_turn,_loop_sig,_loop_count
 if d.get('select') is None:
  policy_v13.agent(d);policy_v6.agent(d);return my_deck
 obs=to_observation_class(d);name=detect(obs)
 if obs.current is not None and obs.current.turn==0:
  _planned=[];_planned_turn=-1;_loop_sig=None;_loop_count=0
  # preserve v15/v14 stable opening
  policy_v13.agent(d);return policy_v6.agent(d)
 # keep both policies synchronized with the actual trajectory
 a13=policy_v13.agent(d);a6=policy_v6.agent(d)
 pol=_active_policy(name);base=a13
 # replay a previously validated plan if the next exact option identities exist
 if _planned and obs.current.turn==_planned_turn:
  sigs=_planned[0];idxs=[]
  for sg in sigs:
   z=_find(obs,sg)
   if z is None:idxs=[];break
   idxs += z
  if idxs and len(set(idxs))==len(idxs) and obs.select.minCount<=len(idxs)<=obs.select.maxCount:
   _planned.pop(0);return idxs
  _planned=[]
 # create a complete turn plan only after opponent deck is identified
 p=_plan(obs,name,pol,base)
 if p:
  _planned=p[1:];_planned_turn=obs.current.turn
  z=[]
  for sg in p[0]:
   q=_find(obs,sg)
   if q:z+=q
  if z:return z
 # exact-state loop breaker retained from v15
 if obs.current is not None and obs.select.context==SelectContext.MAIN:
  me=obs.current.players[obs.current.yourIndex]
  opts=tuple(_sig(obs,x) for x in obs.select.option)
  field=tuple((p.id,p.hp,tuple(e.id for e in p.energyCards)) for p in list(me.active)+list(me.bench) if p)
  sig=(obs.current.turn,len(obs.logs),me.handCount,me.deckCount,len(me.prize),field,opts,tuple(base))
  if sig==_loop_sig:_loop_count+=1
  else:_loop_sig=sig;_loop_count=1
  if _loop_count>=3:
   for i,o in enumerate(obs.select.option):
    if o.type==OptionType.END:_loop_count=0;return [i]
 else:_loop_sig=None;_loop_count=0
 return base
