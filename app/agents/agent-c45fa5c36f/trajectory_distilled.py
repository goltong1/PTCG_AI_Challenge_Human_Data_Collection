from collections import Counter
MAIN=0; ULTRA=1121
class TrajectoryDistilled:
 def __init__(self,api,controller): self.api=api; self.controller=controller; self.stats=Counter(); self.turn=-1; self.ultras=0
 def reset(self): self.turn=-1; self.ultras=0; self.stats['games']+=1
 def get_stats(self): return dict(self.stats)
 def _i(self,x,d=0):
  try:return int(x)
  except:return d
 def _src(self,obs,o):
  try:return self.controller._source_card(obs,o)
  except:return None
 def _cid(self,obs,o):
  c=self._src(obs,o);return self._i(getattr(c,'id',getattr(o,'cardId',0)))
 def _arch(self,obs):
  try:
   st=obs.current; me=self._i(st.yourIndex); op=st.players[1-me]; ids=set()
   for p in list(op.active or [])+list(op.bench or []):
    if p is not None: ids.add(self._i(getattr(p,'id',0)))
   for c in list(op.discard or []):
    if c is not None: ids.add(self._i(getattr(c,'id',0)))
   return bool(ids & {169,190,666})
  except:return False
 def patch(self,observation,chosen):
  try:
   if not observation.get('select') or not isinstance(chosen,list) or len(chosen)!=1:return chosen
   obs=self.api.to_observation_class(observation); sel=obs.select; st=obs.current
   if st is None or sel is None or self._i(sel.context,-1)!=MAIN:return chosen
   turn=self._i(st.turn,-1)
   if turn!=self.turn:self.turn=turn;self.ultras=0
   opts=list(sel.option or []); old=self._i(chosen[0],-1)
   oldcid=self._cid(obs,opts[old]) if 0<=old<len(opts) else 0
   # Learn from exact branch: on our first turn vs Archaludon, if one Ultra Ball
   # has already been committed and another remains legal, spend the second
   # before other setup actions. Never touches target/discard sub-selections.
   if turn<=2 and self._arch(obs) and self.ultras>=1 and oldcid!=ULTRA:
    u=next((i for i,o in enumerate(opts) if self._cid(obs,o)==ULTRA and self._i(getattr(o,'type',-1),-1)==7),None)
    if u is not None:
     self.stats['arch_second_ultra']+=1
     self.ultras+=1
     return [u]
   if oldcid==ULTRA:self.ultras+=1
   return chosen
  except Exception:
   self.stats['exceptions']+=1;return chosen
