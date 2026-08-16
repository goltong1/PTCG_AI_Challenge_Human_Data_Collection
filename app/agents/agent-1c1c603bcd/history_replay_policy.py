"""Replay-trained history-conditioned risk residual.

The model is deliberately conservative: it scores legal alternatives with a
linear hashed sequence model, but an override is allowed only when the
alternative also has positive game-level replay support in a comparable
matchup/context.  Same-turn closeout and matchup rules remain downstream.
"""
from __future__ import annotations

from collections import Counter
import hashlib,json,math,os


PLAY=7;ATTACH=8;EVOLVE=9;ABILITY=10;ATTACK=13;END=14;MAIN=0

SIG={
 'crustle':{344,345},'dragapult':{119,120,121},
 'alakazam':{741,742,743,245},'starmie':{1030,1031},
 'archaludon':{169,190,666},'marnie':{646,647,648},
 'lucario':{333,677,678},'hydrapple':{150,710,917,709,149,93},
 'cynthia':{379,380,381,341,342},'slowking':{162,163,144},
}
TERA_STRONG={96,108,756};TERA_WEAK={272,184,1071,230,140,31}


def _int(x,d=0):
 try:return int(x if x is not None else d)
 except Exception:return d


def turn_bin(turn):return 'early' if int(turn)<=3 else 'mid' if int(turn)<=7 else 'late'


def recognize(history,obs):
 cur=obs.get('current') or {};me=history.me_index if history.me_index in (0,1) else _int(cur.get('yourIndex'),0)
 ids=history.revealed_ids(1-me)
 if 345 in ids:return 'crustle'
 if ids&{169,190}:return 'archaludon'
 for name in ('crustle','dragapult','alakazam','starmie','archaludon','marnie','lucario','hydrapple','cynthia','slowking'):
  if ids&SIG[name]:return name
 if ids&TERA_STRONG or len(ids&TERA_WEAK)>=2:return 'terabox'
 return 'unknown'


def desc_signature(descs):
 def one(d):return f"{_int(d.get('type'),-1)}:{_int(d.get('cardId'))}:{_int(d.get('targetId'))}:{_int(d.get('attackId'))}"
 return '+'.join(sorted(one(d) for d in descs)) or 'empty'


def action_desc(history,obs,action):
 try:return [history._option_desc(obs,_int(i,-1)) for i in action]
 except Exception:return []


def _event_token(event,me):
 p=_int(event.get('playerIndex'),-1);side='own' if p==me else 'opp' if p==1-me else 'global'
 return f"{side}:{_int(event.get('type'),-1)}:{_int(event.get('cardId'))}:{_int(event.get('attackId'))}:{_int(event.get('cardIdTarget'))}"


def _current_ids(obs,me):
 cur=obs.get('current') or {};ps=cur.get('players') or [{},{}]
 mine=ps[me] if me<len(ps) else {};opp=ps[1-me] if 1-me<len(ps) else {}
 def active(p):
  xs=p.get('active') or [];return _int((xs[0] or {}).get('id')) if xs and xs[0] else 0
 return active(mine),active(opp),mine,opp


def support_keys(history,obs,family,action_sig):
 cur=obs.get('current') or {};me=history.me_index if history.me_index in (0,1) else _int(cur.get('yourIndex'),0)
 ctx=_int((obs.get('select') or {}).get('context'),-1);tb=turn_bin(_int(cur.get('turn')))
 recent=[_event_token(e,me) for e in history.events[-12:]]
 last_opp=next((x for x in reversed(recent) if x.startswith('opp:')),'none')
 prev='none'
 if history.decisions:prev=desc_signature(history.decisions[-1].get('actions') or [])
 return [
  f'FCTLP|{family}|{ctx}|{tb}|{last_opp}|{prev}|{action_sig}',
  f'FCTL|{family}|{ctx}|{tb}|{last_opp}|{action_sig}',
  f'FCT|{family}|{ctx}|{tb}|{action_sig}',
  f'FC|{family}|{ctx}|{action_sig}',
  f'C|{ctx}|{action_sig}',
 ]


def feature_map(history,obs,descs,family=None):
 family=family or recognize(history,obs);cur=obs.get('current') or {};me=history.me_index if history.me_index in (0,1) else _int(cur.get('yourIndex'),0)
 ctx=_int((obs.get('select') or {}).get('context'),-1);tb=turn_bin(_int(cur.get('turn')));sig=desc_signature(descs)
 own_active,opp_active,mine,opp=_current_ids(obs,me)
 f={f'state:{k}':float(v) for k,v in history.features(obs).items() if isinstance(v,(int,float)) and v}
 f.update({
  'bias':1.0,f'family={family}':1.0,f'context={ctx}':1.0,f'turn={tb}':1.0,
  f'action={sig}':1.0,f'family_action={family}|{sig}':1.0,
  f'context_action={ctx}|{sig}':1.0,f'turn_action={tb}|{sig}':1.0,
  f'family_context_action={family}|{ctx}|{sig}':1.0,
  f'own_active_action={own_active}|{sig}':1.0,f'opp_active_action={opp_active}|{sig}':1.0,
 })
 my_pr=len(mine.get('prize') or []);op_pr=len(opp.get('prize') or [])
 f[f'prize_action={my_pr}:{op_pr}|{sig}']=1.0
 f[f'hand_action={min(_int(mine.get("handCount")),9)}:{min(_int(opp.get("handCount")),9)}|{sig}']=1.0
 recent=[_event_token(e,me) for e in history.events[-8:]]
 for pos,tok in enumerate(reversed(recent)):
  f[f'action_recent{pos}={sig}|{tok}']=1.0/(1.0+pos)
 if history.decisions:
  prev=desc_signature(history.decisions[-1].get('actions') or [])
  f[f'action_prev={prev}>{sig}']=1.0;f[f'family_action_prev={family}|{prev}>{sig}']=1.0
 for d in descs:
  typ=_int(d.get('type'),-1);cid=_int(d.get('cardId'));tid=_int(d.get('targetId'));aid=_int(d.get('attackId'))
  f[f'action_type={typ}']=f.get(f'action_type={typ}',0.0)+1.0
  if cid:f[f'action_card={cid}']=f.get(f'action_card={cid}',0.0)+1.0
  if tid:f[f'action_target={tid}']=f.get(f'action_target={tid}',0.0)+1.0
  if aid:f[f'action_attack={aid}']=f.get(f'action_attack={aid}',0.0)+1.0
 return f


def hash_features(features,dim):
 out=Counter()
 for name,value in features.items():
  if not value:continue
  raw=hashlib.blake2b(str(name).encode('utf-8'),digest_size=8).digest();num=int.from_bytes(raw,'little')
  idx=num%int(dim);sign=-1.0 if (num>>63)&1 else 1.0;out[idx]+=sign*float(value)
 return out


class HistoryReplayPolicy:
 def __init__(self,root,history,model_file='history_replay_model.json'):
  self.history=history;self.path=os.path.join(root,model_file)
  try:self.model=json.load(open(self.path,encoding='utf-8'))
  except Exception:self.model={'enabled':False,'dim':1,'weights':[],'intercept':0.0,'support':{},'gate':{}}
  self.dim=max(1,_int(self.model.get('dim'),1));self.weights={_int(i):float(w) for i,w in (self.model.get('weights') or [])};self.intercept=float(self.model.get('intercept',0.0));self.support=self.model.get('support') or {};self.gate=self.model.get('gate') or {};self.reset()
 def reset(self):self.last_turn=-1;self.game_overrides=0;self.stats={'calls':0,'overrides':0,'unsupported':0,'low_margin':0,'risk_gate':0,'keys':{}}
 def _logit(self,obs,descs,family):
  x=hash_features(feature_map(self.history,obs,descs,family),self.dim);return self.intercept+sum(self.weights.get(i,0.0)*v for i,v in x.items())
 def _support(self,obs,family,sig):
  for key in support_keys(self.history,obs,family,sig):
   row=self.support.get(key)
   if row:return key,row
  return None,None
 def choose(self,obs,base):
  if not self.model.get('enabled',False) or not isinstance(obs,dict) or not isinstance(base,list):return base
  sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
  if _int(sel.get('context'),-1)!=MAIN or _int(sel.get('minCount'))!=1 or _int(sel.get('maxCount'))!=1 or len(base)!=1 or not opts:return base
  self.stats['calls']+=1;turn=_int(cur.get('turn'))
  if turn==self.last_turn or self.game_overrides>=_int(self.gate.get('max_game_overrides'),2):return base
  bi=_int(base[0],-1)
  if not 0<=bi<len(opts):return base
  bdesc=action_desc(self.history,obs,base);bsig=desc_signature(bdesc);btype=_int(bdesc[0].get('type'),-1) if bdesc else -1
  allowed={_int(x) for x in (self.gate.get('allowed_types') or [PLAY])}
  if btype not in allowed:return base
  family=recognize(self.history,obs)
  allowed_families=set(self.gate.get('allowed_families') or [])
  if allowed_families and family not in allowed_families:return base
  if family=='unknown' and not bool(self.gate.get('allow_unknown',False)):return base
  bscore=self._logit(obs,bdesc,family);_,bsup=self._support(obs,family,bsig)
  if bool(self.gate.get('require_base_loss_support',True)):
   if not bsup:self.stats['risk_gate']+=1;return base
   bw=_int(bsup.get('win_games'));bl=_int(bsup.get('loss_games'));brate=(bw+1.0)/(bw+bl+2.0)
   if bl<_int(self.gate.get('min_base_loss_games'),2) or brate>float(self.gate.get('max_base_support_rate',.4)):
    self.stats['risk_gate']+=1;return base
  max_base=float(self.gate.get('max_base_logit',0.0))
  if bscore>max_base:self.stats['risk_gate']+=1;return base
  candidates=[]
  for i,o in enumerate(opts):
   if i==bi:continue
   desc=action_desc(self.history,obs,[i]);typ=_int(desc[0].get('type'),-1) if desc else -1
   if typ!=btype:continue
   sig=desc_signature(desc)
   if sig==bsig:continue
   # Black Belt is a damage-threshold Supporter.  Never promote it from replay
   # correlation when the current menu has no legal attack at all.
   if _int(desc[0].get('cardId'))==1211 and not any(_int(x.get('type'),-1)==ATTACK for x in opts):continue
   key,sup=self._support(obs,family,sig)
   if not sup:continue
   wg=_int(sup.get('win_games'));lg=_int(sup.get('loss_games'));rate=(wg+1.0)/(wg+lg+2.0)
   if wg<_int(self.gate.get('min_win_games'),2) or rate<float(self.gate.get('min_candidate_rate',.62)):continue
   score=self._logit(obs,desc,family);delta=score-bscore
   candidates.append((delta,score,rate,wg,-lg,-i,i,key,sig))
  if not candidates:self.stats['unsupported']+=1;return base
  candidates.sort(reverse=True);best=candidates[0]
  if best[0]<float(self.gate.get('min_logit_margin',.35)):
   self.stats['low_margin']+=1;return base
  # If baseline support exists, require it not to be more successful than the
  # alternative.  Missing baseline support is treated conservatively by margin.
  if bsup:
   bw=_int(bsup.get('win_games'));bl=_int(bsup.get('loss_games'));brate=(bw+1.0)/(bw+bl+2.0)
   if brate>=best[2]-float(self.gate.get('min_support_rate_gap',.08)):return base
  self.last_turn=turn;self.game_overrides+=1;self.stats['overrides']+=1
  self.stats['keys'][best[7]]=self.stats['keys'].get(best[7],0)+1
  return [best[6]]
 def get_stats(self):return dict(self.stats)
