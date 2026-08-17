from __future__ import annotations
import json,os
from collections import Counter
class LeagueSetupValueResidual:
 DREEPY=119;DRAK=120;DRAG=121;MUNK=112;DUNS=305;DUDUN=66;BUDEW=235
 FIRE=2;PSY=5
 ALLOWED={'PLAY_POFFIN','EVOLVE_DRAK','EVOLVE_DRAG','PLAY_LILLIE','PLAY_CRISPIN'}
 LOW={'END','PLAY_HAMMER','PLAY_RISKY','PLAY_JUDGE','PLAY_PAD','PLAY_STRETCHER'}
 def __init__(self,api,controller,path):
  self.api=api;self.c=controller;self.stats=Counter()
  try:self.m=json.load(open(path,encoding='utf-8')).get('recommendations',{})
  except Exception:self.m={}
 def reset(self):self.stats.clear()
 def get_stats(self):return dict(self.stats)
 def _b(self,k):self.stats[str(k)]+=1
 def _i(self,x,d=0):
  try:return int(x) if x is not None else d
  except:return d
 def _board(self,p):return [x for x in list(p.active or [])+list(p.bench or []) if x is not None]
 def _e(self,p):return {self._i(getattr(x,'id',0)) for x in list(getattr(p,'energyCards',[]) or [])}
 def _role(self,cid):return 'dreepy' if cid==119 else 'drak' if cid==120 else 'drag' if cid==121 else 'pivot' if cid in (305,66,235,112) else 'other'
 def _bucket(self,obs):
  st=obs.current;mine=st.players[st.yourIndex];ms=self._board(mine);ids=[self._i(getattr(p,'id',0)) for p in ms];turn=self._i(st.turn)
  if turn>10:return None
  tb=0 if turn<=2 else 1 if turn<=4 else 2 if turn<=6 else 3
  hasd=119 in ids;hask=120 in ids;hasg=121 in ids
  line=0 if not(hasd or hask or hasg) else 1 if hasd and not(hask or hasg) else 2 if hask and not hasg else 3
  prog=0
  for p in ms:
   cid=self._i(getattr(p,'id',0))
   if cid in (119,120,121):
    es=self._e(p);q=int(2 in es)+int(5 in es);prog=max(prog,q)
    if cid==121 and q==2:return None
  a=mine.active[0] if mine.active and mine.active[0] is not None else None;ar=self._role(self._i(getattr(a,'id',0))) if a is not None else 'none'
  hc=len(list(mine.hand or []));hb=0 if hc<=3 else 1 if hc<=6 else 2
  bc=len(list(mine.bench or []));bb=0 if bc<=1 else 1 if bc<=3 else 2
  return f'{tb}|{line}|{prog}|{ar}|{hb}|{bb}'
 def _src(self,obs,o):
  try:return self.c._source_card(obs,o)
  except:return None
 def _cid(self,obs,o):
  c=self._src(obs,o);return self._i(getattr(c,'id',getattr(o,'cardId',0)))
 def _target_role(self,obs,o):
  try:
   mine=obs.current.players[obs.current.yourIndex];ar=self._i(getattr(o,'inPlayArea',-1),-1);ix=self._i(getattr(o,'inPlayIndex',-1),-1)
   arr=list(mine.active or []) if ar==4 else list(mine.bench or []) if ar==5 else []
   p=arr[ix] if 0<=ix<len(arr) else None
   return self._role(self._i(getattr(p,'id',0))) if p is not None else 'x'
  except:return 'x'
 def _cat(self,obs,o):
  t=o.type;cid=self._cid(obs,o)
  if t==self.api.OptionType.PLAY:
   return {1086:'PLAY_POFFIN',1121:'PLAY_ULTRA',1227:'PLAY_LILLIE',1198:'PLAY_CRISPIN',1120:'PLAY_HAMMER',1260:'PLAY_RISKY',1213:'PLAY_JUDGE',1182:'PLAY_BOSS',1152:'PLAY_PAD',1097:'PLAY_STRETCHER',1080:'PLAY_STAMP',119:'PLAY_DREEPY',305:'PLAY_DUNS',112:'PLAY_MUNK',235:'PLAY_BUDEW',1071:'PLAY_MEOWTH',140:'PLAY_FEZ'}.get(cid,f'PLAY_{cid}')
  if t==self.api.OptionType.ATTACH:
   en={2:'F',5:'P',7:'D'}.get(cid,str(cid));return f'ATTACH_{en}_{self._target_role(obs,o)}'
  if t==self.api.OptionType.EVOLVE:return 'EVOLVE_DRAK' if cid==120 else 'EVOLVE_DRAG' if cid==121 else f'EVOLVE_{cid}'
  if t==self.api.OptionType.ABILITY:return 'ABILITY_DRAK' if cid==120 else 'ABILITY_DUDUN' if cid==66 else 'ABILITY_MUNK' if cid==112 else f'ABILITY_{cid}'
  if t==self.api.OptionType.RETREAT:return 'RETREAT'
  if t==self.api.OptionType.ATTACK:
   aid=self._i(getattr(o,'attackId',0));return 'ATTACK_PHANTOM' if aid==154 else 'ATTACK_JET' if aid==153 else 'ATTACK_BUDEW' if aid==323 else 'ATTACK_MUNK' if aid==141 else f'ATTACK_{aid}'
  if t==self.api.OptionType.END:return 'END'
  return 'OTHER'
 def patch(self,observation,chosen):
  try:
   if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'):return chosen
   obs=self.api.to_observation_class(observation);sel=obs.select
   if obs.current is None or sel is None or sel.context!=self.api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1:return chosen
   b=self._bucket(obs);r=self.m.get(b) if b else None
   if not r:return chosen
   best=r.get('best')
   if best not in self.ALLOWED:return chosen
   opts=list(sel.option or []);ci=self._i(chosen[0],-1)
   if not(0<=ci<len(opts)):return chosen
   parent=self._cat(obs,opts[ci])
   if parent==best:return chosen
   # Learned residual may only replace optional/low-tempo actions. Never steal an
   # F/P attachment, attack, retreat, Ultra Ball, evolution, or free ability.
   if parent not in self.LOW:return chosen
   cand=[i for i,o in enumerate(opts) if self._cat(obs,o)==best]
   if not cand:return chosen
   self._b('applied_'+best);self._b('from_'+parent);return [cand[0]]
  except Exception:self._b('exceptions');return chosen
