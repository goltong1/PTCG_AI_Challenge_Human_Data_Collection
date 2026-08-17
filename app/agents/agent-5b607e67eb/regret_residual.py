from __future__ import annotations
import json,math
from pathlib import Path
# This module is bundled beside main.py. It uses only public observations.
from regret_features_runtime import infer_family,candidates,option_desc,pair_features,support_key

class ResidualPolicy:
 def __init__(self,root):
  root=Path(root);self.model=json.loads((root/'regret_model.json').read_text(encoding='utf-8'));self.names=self.model['feature_names'];self.index={k:i for i,k in enumerate(self.names)};self.models=self.model['models'];self.seen=set();self.last_override_turn=-1;self.stats={'calls':0,'overrides':0,'unknown_family':0,'unsupported':0,'low_margin':0}
  chosen=((self.model.get('report') or {}).get('chosen') or {});self.conservative=not bool(chosen.get('feasible',False))
 def reset(self):self.seen.clear();self.last_override_turn=-1
 def _score(self,features):
  vals=[]
  sparse=[(self.index[k],float(v)) for k,v in features.items() if k in self.index and v]
  for m in self.models:
   weights=dict(m.get('weights') or []);z=float(m.get('intercept',0.0))
   for i,v in sparse:z+=float(weights.get(i,0.0))*v
   vals.append(z)
  mean=sum(vals)/len(vals) if vals else 0.0
  sd=math.sqrt(sum((x-mean)**2 for x in vals)/len(vals)) if vals else 0.0
  return mean,sd
 def choose(self,obs,base,history_features=None):
  if not isinstance(obs,dict) or obs.get('select') is None:
   self.reset();return base
  if not isinstance(base,list) or not all(isinstance(x,int) for x in base):return base
  self.stats['calls']+=1;family=infer_family(obs,self.seen)
  if family=='unknown':self.stats['unknown_family']+=1;return base
  if family in set(self.model.get('blocked_families') or []):return base
  turn=int((obs.get('current') or {}).get('turn',0))
  # A single learned commitment per turn prevents independent residual choices
  # from fighting over one action sequence.
  if turn==self.last_override_turn:return base
  cs=candidates(obs,base,int(self.model.get('max_candidates',6)))
  if len(cs)<2:return base
  bi=next((i for i,c in enumerate(cs) if c==base),0);bdesc=[option_desc(obs,i) for i in cs[bi]];ctx=int((obs.get('select') or {}).get('context',-1));best=None
  threshold=float(self.model.get('threshold',0.03));uz=float(self.model.get('uncertainty_z',.5));minsup=int(self.model.get('min_support',2));support=self.model.get('support') or {};strong=self.model.get('strong_keys') or {}
  for j,c in enumerate(cs):
   if j==bi:continue
   desc=[option_desc(obs,i) for i in c];key=support_key(family,ctx,bdesc,desc);sup=int(support.get(key,0));proto=strong.get(key)
   if not proto and sup<minsup:
    self.stats['unsupported']+=1;continue
   feat=pair_features(obs,bdesc,desc,family,history_features);mu,sd=self._score(feat);lower=mu-uz*sd
   if proto:lower=max(lower,float(proto.get('ci95_low',-1)))
   elif self.conservative:
    if sup<max(3,minsup) or lower<threshold+.02:continue
   if lower>threshold and (best is None or lower>best[0]):best=(lower,c,key,mu,sd,sup,bool(proto))
  if best is None:self.stats['low_margin']+=1;return base
  self.last_override_turn=turn;self.stats['overrides']+=1;return list(best[1])
 def get_stats(self):return dict(self.stats)
