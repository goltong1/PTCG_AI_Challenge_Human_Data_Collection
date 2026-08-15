"""Tiny action-conditioned GRU residual for full public game histories."""
from __future__ import annotations

from collections import Counter
import hashlib,json,math,os


MAIN=0;PLAY=7;ATTACH=8;EVOLVE=9;ABILITY=10;ATTACK=13;END=14


def _int(x,d=0):
 try:return int(x if x is not None else d)
 except Exception:return d


def _add(out,name,value,dim):
 if not value:return
 raw=hashlib.blake2b(str(name).encode('utf-8'),digest_size=8).digest();num=int.from_bytes(raw,'little')
 i=num%dim;out[i]=out.get(i,0.0)+(-1.0 if (num>>63)&1 else 1.0)*float(value)


def event_vector(event,me,dim):
 p=_int(event.get('playerIndex'),-1);side='own' if p==me else 'opp' if p==1-me else 'global'
 typ=_int(event.get('type'),-1);cid=_int(event.get('cardId'));target=_int(event.get('cardIdTarget'));aid=_int(event.get('attackId'))
 out={};_add(out,'event:bias',1,dim);_add(out,f'event:side={side}',1,dim);_add(out,f'event:type={typ}',1,dim)
 _add(out,f'event:side_type={side}:{typ}',1,dim)
 if cid:_add(out,f'event:card={cid}',1,dim);_add(out,f'event:side_type_card={side}:{typ}:{cid}',1,dim)
 if target:_add(out,f'event:target={target}',1,dim)
 if aid:_add(out,f'event:attack={aid}',1,dim)
 fr=_int(event.get('fromArea'),-1);to=_int(event.get('toArea'),-1)
 if fr>=0 or to>=0:_add(out,f'event:move={side}:{fr}>{to}',1,dim)
 if event.get('head') is not None:_add(out,f'event:head={bool(event.get("head"))}',1,dim)
 value=_int(event.get('value'))
 if value:_add(out,f'event:value_bin={max(-6,min(6,value//30))}',1,dim)
 _add(out,f'event:turn_bin={min(15,_int(event.get("turn"))//2)}',1,dim)
 return out


def decision_vector(entry,me,dim):
 out={};_add(out,'decision:bias',1,dim);ctx=_int(entry.get('context'),-1);_add(out,f'decision:ctx={ctx}',1,dim)
 for pos,d in enumerate(entry.get('actions') or []):
  typ=_int(d.get('type'),-1);cid=_int(d.get('cardId'));target=_int(d.get('targetId'));aid=_int(d.get('attackId'))
  _add(out,f'decision:type={typ}',1,dim);_add(out,f'decision:pos_type={pos}:{typ}',1,dim)
  if cid:_add(out,f'decision:card={cid}',1,dim);_add(out,f'decision:type_card={typ}:{cid}',1,dim)
  if target:_add(out,f'decision:target={target}',1,dim)
  if aid:_add(out,f'decision:attack={aid}',1,dim)
 return out


def dense_action_features(replay_mod,history,obs,descs,family,dim):
 return replay_mod.hash_features(replay_mod.feature_map(history,obs,descs,family),dim)


def _sigmoid(x):return 1.0/(1.0+math.exp(-max(-30.0,min(30.0,x))))


class TemporalGRUPolicy:
 def __init__(self,root,history,replay_mod,model_file='temporal_gru_model.json'):
  self.history=history;self.replay_mod=replay_mod
  try:self.model=json.load(open(os.path.join(root,model_file),encoding='utf-8'))
  except Exception:self.model={'enabled':False}
  self.input_dim=max(1,_int(self.model.get('input_dim'),1));self.action_dim=max(1,_int(self.model.get('action_dim'),1));self.hidden=max(1,_int(self.model.get('hidden'),1))
  self.Wz=self.model.get('Wz') or [];self.Wr=self.model.get('Wr') or [];self.Wn=self.model.get('Wn') or []
  self.Uz=self.model.get('Uz') or [];self.Ur=self.model.get('Ur') or [];self.Un=self.model.get('Un') or []
  self.bz=self.model.get('bz') or [0.0]*self.hidden;self.br=self.model.get('br') or [0.0]*self.hidden;self.bn=self.model.get('bn') or [0.0]*self.hidden
  self.wh=self.model.get('wh') or [0.0]*self.hidden;self.ws=self.model.get('ws') or [0.0]*self.action_dim;self.V=self.model.get('V') or [[0.0]*self.action_dim for _ in range(self.hidden)];self.bias=float(self.model.get('bias',0.0))
  self.support=self.model.get('support') or {};self.gate=self.model.get('gate') or {};self.reset()

 def reset(self):
  self.h=[0.0]*self.hidden;self.event_cursor=0;self.decision_cursor=0;self.last_turn=-1;self.game_overrides=0
  self.stats={'calls':0,'overrides':0,'unsupported':0,'low_margin':0,'risk_gate':0,'synced_events':0,'synced_decisions':0,'keys':{}}

 def _step(self,x):
  h0=self.h;z=[0.0]*self.hidden;r=[0.0]*self.hidden
  for j in range(self.hidden):
   az=self.bz[j]+sum(float(self.Wz[i][j])*v for i,v in x.items())+sum(float(self.Uz[k][j])*h0[k] for k in range(self.hidden))
   ar=self.br[j]+sum(float(self.Wr[i][j])*v for i,v in x.items())+sum(float(self.Ur[k][j])*h0[k] for k in range(self.hidden))
   z[j]=_sigmoid(az);r[j]=_sigmoid(ar)
  n=[0.0]*self.hidden
  for j in range(self.hidden):
   an=self.bn[j]+sum(float(self.Wn[i][j])*v for i,v in x.items())+sum(float(self.Un[k][j])*(r[k]*h0[k]) for k in range(self.hidden))
   n[j]=math.tanh(an)
  self.h=[(1.0-z[j])*n[j]+z[j]*h0[j] for j in range(self.hidden)]

 def _sync(self):
  me=self.history.me_index if self.history.me_index in (0,1) else 0
  # A decision was emitted before the logs observed on the following call.
  while self.decision_cursor<len(self.history.decisions):
   self._step(decision_vector(self.history.decisions[self.decision_cursor],me,self.input_dim));self.decision_cursor+=1;self.stats['synced_decisions']+=1
  while self.event_cursor<len(self.history.events):
   self._step(event_vector(self.history.events[self.event_cursor],me,self.input_dim));self.event_cursor+=1;self.stats['synced_events']+=1

 def _score(self,sparse):
  interaction=0.0
  for j,hj in enumerate(self.h):
   if hj:interaction+=hj*sum(float(self.V[j][i])*v for i,v in sparse.items())
  return self.bias+sum(self.wh[j]*self.h[j] for j in range(self.hidden))+sum(self.ws[i]*v for i,v in sparse.items())+interaction

 def _support(self,obs,family,sig):
  for key in self.replay_mod.support_keys(self.history,obs,family,sig):
   row=self.support.get(key)
   if row:return key,row
  return None,None

 def choose(self,obs,base):
  self._sync()
  if not self.model.get('enabled',False) or not isinstance(obs,dict) or not isinstance(base,list):return base
  sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
  if _int(sel.get('context'),-1)!=MAIN or _int(sel.get('minCount'))!=1 or _int(sel.get('maxCount'))!=1 or len(base)!=1 or not opts:return base
  self.stats['calls']+=1;turn=_int(cur.get('turn'))
  if turn==self.last_turn or self.game_overrides>=_int(self.gate.get('max_game_overrides'),1):return base
  bi=_int(base[0],-1)
  if not 0<=bi<len(opts):return base
  bdesc=self.replay_mod.action_desc(self.history,obs,base);btype=_int(bdesc[0].get('type'),-1) if bdesc else -1
  if btype not in {_int(x) for x in (self.gate.get('allowed_types') or [PLAY])}:return base
  family=self.replay_mod.recognize(self.history,obs);allowed=set(self.gate.get('allowed_families') or [])
  if 'allowed_families' in self.gate and not allowed:return base
  if allowed and family not in allowed:return base
  if family=='unknown' and not bool(self.gate.get('allow_unknown',False)):return base
  bsig=self.replay_mod.desc_signature(bdesc);bx=dense_action_features(self.replay_mod,self.history,obs,bdesc,family,self.action_dim);bscore=self._score(bx);_,bsup=self._support(obs,family,bsig)
  if bool(self.gate.get('require_base_loss_support',True)):
   if not bsup:self.stats['risk_gate']+=1;return base
   bw=_int(bsup.get('win_games'));bl=_int(bsup.get('loss_games'));brate=(bw+1.0)/(bw+bl+2.0)
   if bl<_int(self.gate.get('min_base_loss_games'),4) or brate>float(self.gate.get('max_base_support_rate',.42)):
    self.stats['risk_gate']+=1;return base
  if bscore>float(self.gate.get('max_base_logit',0.0)):self.stats['risk_gate']+=1;return base
  cand=[]
  for i in range(len(opts)):
   if i==bi:continue
   desc=self.replay_mod.action_desc(self.history,obs,[i]);typ=_int(desc[0].get('type'),-1) if desc else -1
   if typ!=btype:continue
   sig=self.replay_mod.desc_signature(desc)
   if sig==bsig:continue
   key,sup=self._support(obs,family,sig)
   if not sup:continue
   wg=_int(sup.get('win_games'));lg=_int(sup.get('loss_games'));rate=(wg+1.0)/(wg+lg+2.0)
   if wg<_int(self.gate.get('min_win_games'),8) or rate<float(self.gate.get('min_candidate_rate',.68)):continue
   x=dense_action_features(self.replay_mod,self.history,obs,desc,family,self.action_dim);score=self._score(x)
   cand.append((score-bscore,rate,wg,-lg,-i,i,key))
  if not cand:self.stats['unsupported']+=1;return base
  cand.sort(reverse=True);best=cand[0]
  if best[0]<float(self.gate.get('min_logit_margin',.8)):self.stats['low_margin']+=1;return base
  if bsup:
   bw=_int(bsup.get('win_games'));bl=_int(bsup.get('loss_games'));brate=(bw+1.0)/(bw+bl+2.0)
   if brate>=best[1]-float(self.gate.get('min_support_rate_gap',.15)):return base
  self.last_turn=turn;self.game_overrides+=1;self.stats['overrides']+=1;self.stats['keys'][best[6]]=self.stats['keys'].get(best[6],0)+1
  return [best[5]]

 def get_stats(self):return dict(self.stats)
