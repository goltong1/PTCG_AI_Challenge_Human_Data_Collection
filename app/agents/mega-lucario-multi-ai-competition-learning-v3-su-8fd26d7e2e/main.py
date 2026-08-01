from __future__ import annotations
import os,sys
from pathlib import Path
from cg.api import to_observation_class
import feature_core as fc
from model_runtime import TreeDumpModel,CountModel,zscore

BASE=Path(os.getcwd())
if not (BASE/'deck.csv').exists(): BASE=Path('/kaggle_simulations/agent')
my_deck=[int(x) for x in (BASE/'deck.csv').read_text().split()]
_MEMORY=fc.new_memory()
_GENERIC=TreeDumpModel(BASE/'generic_v2.json.gz')
_SPECIAL={'alakazam':TreeDumpModel(BASE/'alakazam_v2.json.gz'),'dragapult':TreeDumpModel(BASE/'dragapult_v3.json.gz')}
_ALAK_FULL=TreeDumpModel(BASE/'alakazam_v3.json.gz')
_ALAK_ON=TreeDumpModel(BASE/'alakazam_onpolicy_v5.json.gz')
_ALAK_CORR=TreeDumpModel(BASE/'alakazam_correction_v5.json.gz')
_COUNT=CountModel(BASE/'count_v2.json.gz')
_ALAK_COMP=TreeDumpModel(BASE/'alakazam_competition_base_v4.json.gz')
_ALAK_GAP=3.0

def _reset():
 global _MEMORY
 _MEMORY=fc.new_memory()

def _validate(res,sel):
 n=len(sel.option or []);lo=max(0,int(sel.minCount or 0));hi=max(lo,min(n,int(sel.maxCount or lo)));out=[]
 for x in res or []:
  try:i=int(x)
  except:continue
  if 0<=i<n and i not in out:out.append(i)
 if len(out)<lo:
  for i in range(n):
   if i not in out:out.append(i)
   if len(out)>=lo:break
 return out[:hi]

def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  _reset();return my_deck
 try:
  obs=to_observation_class(observation);arch=fc.update_memory(obs,_MEMORY);ph=fc.phase(obs);rx=fc.rich_state_features(obs,_MEMORY);descs=[fc.extended_action_desc(obs,o) for o in obs.select.option]
  if not descs:return []
  g=zscore(_GENERIC.scores(rx,arch,ph,descs));scores=g
  if arch in _SPECIAL:
   s=zscore(_SPECIAL[arch].scores(rx,arch,ph,descs))
   if arch=='alakazam':
    f=zscore(_ALAK_FULL.scores(rx,arch,ph,descs));n=zscore(_ALAK_ON.scores(rx,arch,ph,descs));c=zscore(_ALAK_CORR.scores(rx,arch,ph,descs));scores=[x+0.5*y+0.35*z+1.0*u+0.25*v for x,y,z,u,v in zip(g,s,f,n,c)]
   else:scores=[x+0.5*y for x,y in zip(g,s)]
  lo=int(obs.select.minCount or 0);hi=int(obs.select.maxCount or lo)
  if arch=='alakazam' and lo==hi==1 and len(descs)>1:
   bs=_ALAK_COMP.scores(rx,arch,ph,descs);bo=sorted(range(len(bs)),key=lambda i:(bs[i],-i),reverse=True)
   if bs[bo[0]]-bs[bo[1]]>=_ALAK_GAP:
    res=_validate([bo[0]],obs.select);fc.record_action(_MEMORY,descs,res);return res
  if lo==hi:k=lo
  else:k=_COUNT.pick(rx,arch,ph,int(obs.select.context),int(descs[0][1]),len(descs),lo,hi)
  k=max(lo,min(hi,k,len(descs)));rank=sorted(range(len(scores)),key=lambda i:(scores[i],-i),reverse=True);res=_validate(rank[:k],obs.select);fc.record_action(_MEMORY,descs,res);return res
 except Exception:
  try:
   sel=observation.get('select') or {};n=len(sel.get('option') or []);lo=int(sel.get('minCount',0) or 0);return list(range(min(lo,n)))
  except:return []
