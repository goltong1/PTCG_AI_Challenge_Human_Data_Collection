from __future__ import annotations
from collections import Counter
class LeaguePairwiseResidual:
 DREEPY=119;DRAK=120;DRAG=121;MUNK=112
 CRUST={344,345};MARNIE={646,647,648};LUCARIO={333,677,678}
 def __init__(self,api,controller):self.api=api;self.c=controller;self.stats=Counter()
 def reset(self):self.stats.clear()
 def get_stats(self):return dict(self.stats)
 def _b(self,k):self.stats[str(k)]+=1
 def _i(self,x,d=0):
  try:return int(x) if x is not None else d
  except:return d
 def _src(self,obs,o):
  try:return self.c._source_card(obs,o)
  except:return None
 def _cid(self,obs,o):
  c=self._src(obs,o);return self._i(getattr(c,'id',getattr(o,'cardId',0)))
 def _public(self,obs):
  try:
   st=obs.current;me=st.yourIndex;op=st.players[1-me];ids=set()
   for p in list(op.active or [])+list(op.bench or [])+list(op.discard or []):
    if p is not None:ids.add(self._i(getattr(p,'id',0)))
   for l in list(obs.logs or []):
    if self._i(getattr(l,'playerIndex',-1),-1)!=1-me:continue
    for k in ('cardId','cardIdAfter','cardIdBefore','cardIdActive','cardIdBench','cardIdTarget'):
     v=self._i(getattr(l,k,0));
     if v:ids.add(v)
   return ids
  except:return set()
 def patch(self,observation,chosen):
  try:
   if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'):return chosen
   obs=self.api.to_observation_class(observation);sel=obs.select
   if obs.current is None or sel is None or sel.context!=self.api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1:return chosen
   opts=list(sel.option or []);ci=self._i(chosen[0],-1)
   if not(0<=ci<len(opts)):return chosen
   old=opts[ci];pub=self._public(obs);st=obs.current;mine=st.players[st.yourIndex]
   # 2020-game pairwise signal, Crustle family: when a Drakloak free draw is
   # still legal, using it before a Munkidori wall attack / retreat was much
   # stronger than committing first. This is a pure sequencing correction.
   if pub & self.CRUST:
    drak_ability=next((i for i,o in enumerate(opts) if o.type==self.api.OptionType.ABILITY and self._cid(obs,o)==self.DRAK),None)
    if drak_ability is not None:
     is_munk_attack=(old.type==self.api.OptionType.ATTACK and self._i(getattr(old,'attackId',0))==141)
     if is_munk_attack or old.type==self.api.OptionType.RETREAT:
      self._b('crust_drak_draw_before_commit');return [drak_ability]
   return chosen
  except Exception:self._b('exceptions');return chosen
