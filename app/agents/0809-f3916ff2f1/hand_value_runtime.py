from __future__ import annotations
import json, math, collections, hashlib, random
from hand_value_common import feature_from_current, IDS, DECK, SEARCH, LILLIE
class MultiTurnHandPolicy:
 def __init__(self,model,profile):
  self.m=model;self.p=profile;self.active=profile;self.turn=None;self.searched=[];self.progress=0;self.last_search_progress=0
 def _reset(self,t):
  if t!=self.turn:self.turn=t;self.searched=[];self.progress=0;self.last_search_progress=0
 def _predict(self,c,seat,opp,hand=None):
  x,_,_=feature_from_current(c,seat,opp,hand);mu=self.m['mean'];sd=self.m['std'];W=self.m['weights'];b=self.m['bias'];z=[(x[i]-mu[i])/sd[i] for i in range(len(x))];y=[b[j]+sum(z[i]*W[i][j] for i in range(len(z))) for j in range(4)];h=int(self.active.get('horizon',3));w=[.0,.52,.31,.17] if h==3 else [.0,.68,.32,0];return sum(w[i+1]*y[i] for i in range(h))+.18*y[3]
 def _source(self,o,opt):
  c=o.get('current') or {};seat=int(c.get('yourIndex') or 0);ps=c.get('players') or [];p=ps[seat] if len(ps)>seat else {};t=int(opt.get('type',-1));ar=opt.get('area');ix=opt.get('index')
  try:
   if t==7 and ar is None:return int((p.get('hand') or [])[ix].get('id') or 0)
   arr={1:(o.get('select') or {}).get('deck') or [],2:p.get('hand') or [],3:p.get('discard') or [],4:p.get('active') or [],5:p.get('bench') or [],7:c.get('stadium') or [],12:c.get('looking') or []}.get(ar,[])
   return int(arr[ix].get('id') or 0) if ix is not None and 0<=ix<len(arr) else int(opt.get('cardId') or 0)
  except:return int(opt.get('cardId') or 0)
 def _opp(self,controller,api,o):
  try:
   obs=api.to_observation_class(o);name,_=controller.recognize(obs);n=(name or '').lower()
   if 'marnie' in n or 'grimmsnarl' in n:return 'marnie'
   if 'dragapult' in n:return 'dragapult'
   if 'lucario' in n:return 'lucario'
   if 'alakazam' in n:return 'alakazam'
   if 'archaludon' in n:return 'archaludon'
   if 'crustle' in n:return 'crustle'
  except:pass
  return 'other'
 def _expected_hand(self,o,seat):
  c=o['current'];p=c['players'][seat];hand=collections.Counter(int(x.get('id') or 0) for x in p.get('hand') or []);known=hand.copy()
  for zone in ['discard']:
   known.update(int(x.get('id') or 0) for x in p.get(zone) or [])
  for z in list(p.get('active') or [])+list(p.get('bench') or []):
   if z:
    known[int(z.get('id') or 0)]+=1;known.update(int(e.get('id') or 0) for e in z.get('energyCards') or []);known.update(int(e.get('id') or 0) for e in z.get('preEvolution') or [])
  rem=DECK.copy()
  for k,v in known.items():rem[k]=max(0,rem[k]-v)
  n=max(1,sum(rem.values()));return collections.Counter({cid:6*cnt/n for cid,cnt in rem.items() if cnt>0})
 def _hand(self,o,seat):return [int(x.get('id') or 0) for x in o['current']['players'][seat].get('hand') or []]
 def decide(self,o,base,controller,api):
  if o.get('select') is None:return base
  sel=o.get('select') or {};opts=sel.get('option') or [];ctx=int(sel.get('context',-1));c=o.get('current') or {};seat=int(c.get('yourIndex') or 0);self._reset(int(c.get('turn') or 0));opp=self._opp(controller,api,o)
  prof=dict(self.p);prof.update((self.p.get('per_opponent') or {}).get(opp,{}));self.active=prof
  if not prof.get('enabled',True):return base
  if not isinstance(base,list) or len(base)!=1 or not isinstance(base[0],int) or not 0<=base[0]<len(opts):return base
  bi=base[0]
  # Search target: only high-margin changes, based on 3-turn hand value.
  if ctx==7 and prof.get('allow_target',True):
   hand=self._hand(o,seat);vals=[]
   for i,x in enumerate(opts):
    cid=self._source(o,x);vals.append((self._predict(c,seat,opp,hand+[cid]),i,cid))
   best=max(vals);bv=vals[bi][0]
   if best[0]>=bv+float(prof.get('target_margin',.45)):
    self.searched.append(best[2]);return [best[1]]
   cid=vals[bi][2]
   if cid:self.searched.append(cid)
   return base
  if ctx!=0:return base
  bopt=opts[bi];bcid=self._source(o,bopt);btyp=int(bopt.get('type',-1));hand=self._hand(o,seat);cur=self._predict(c,seat,opp,hand)
  # Only challenge Lillie/search/cycle/end; preserve normal attack/evolve/attach choices.
  mode=prof.get('mode','all');questionable=(bcid==LILLIE) if mode=='lillie_only' else ((bcid==LILLIE or bcid in SEARCH) if mode=='cycle_lillie' else (bcid==LILLIE or bcid in SEARCH or btyp==14))
  if not questionable:
   if bcid in self.searched:
    try:self.searched.remove(bcid)
    except ValueError:pass
   if btyp in {8,9,10,13}:self.progress+=1
   return base
  vals=[]
  for i,x in enumerate(opts):
   typ=int(x.get('type',-1));cid=self._source(o,x);score=cur
   if typ==13:score=cur+1.02
   elif typ==9:
    hh=list(hand)
    try:hh.remove(cid)
    except ValueError:pass
    score=self._predict(c,seat,opp,hh)+.72
   elif typ==8:
    hh=list(hand)
    try:hh.remove(cid)
    except ValueError:pass
    score=self._predict(c,seat,opp,hh)+.42
   elif typ==10:score=cur+.50
   elif typ==14:score=cur-.06
   elif typ==7 and cid==LILLIE:
    score=self._predict(c,seat,opp,self._expected_hand(o,seat))-.20-.30*sum(1 for z in self.searched if z in hand)-.36*int(self.progress>self.last_search_progress)
   elif typ==7 and cid in SEARCH:
    repeat=int(bool(self.searched) and self.progress==self.last_search_progress);score=cur+.18-.40*repeat-.08*len(self.searched)
   else:score=cur-.04
   if i==bi:score+=float(prof.get('base_bonus',.35))
   vals.append((score,i,cid,typ))
   
  best=max(vals);bv=vals[bi][0]
  if best[0]>=bv+float(prof.get('override_margin',.28)):
   _,i,cid,typ=best
   if typ==7 and cid in SEARCH:self.last_search_progress=self.progress
   if typ in {8,9,10,13}:self.progress+=1
   return [i]
  if bcid in SEARCH:self.last_search_progress=self.progress
  return base
