"""Conservative card/situation LLM residual for Dragapult v36.

The legacy deck core already contains detailed Dunsparce/Dudunsparce/Risky
Ruins heuristics.  This layer therefore does NOT replace the core.  It uses
public state/history to intervene only on high-confidence role conflicts that
repeatedly lose tempo: exposing a charged unfinished Dragapult line, wasting
Darkness away from Munkidori, skipping free draw/damage abilities before an
irreversible attack/end, or playing Risky Ruins before our own setup when no
Munkidori can monetize the self-damage.
"""
from __future__ import annotations
from collections import Counter
class CardSituationLLMRouter:
 DREEPY=119;DRAK=120;DRAG=121;MUNK=112;DUNS=305;DUDUN=66;MEOWTH=1071;FEZ=140;BUDEW=235
 RISKY=1260;POFFIN=1086;ULTRA=1121;CRISPIN=1198;HAMMER=1120;JUDGE=1213;BOSS=1182
 FIRE=2;PSY=5;DARK=7;PHANTOM=154;JET=153
 WALL={344,345}
 def __init__(self,api,controller,base,history):self.api=api;self.controller=controller;self.base=base;self.history=history;self.stats=Counter()
 def reset(self):self.stats.clear()
 def get_stats(self):return dict(self.stats)
 def _b(self,k):self.stats[str(k)]+=1
 def _i(self,x,d=0):
  try:return int(x) if x is not None else d
  except:return d
 def _board(self,p):return [x for x in list(p.active or [])+list(p.bench or []) if x is not None]
 def _e(self,p):return {self._i(getattr(x,'id',0)) for x in list(getattr(p,'energyCards',[]) or [])} if p is not None else set()
 def _src(self,obs,o):
  try:return self.controller._source_card(obs,o)
  except:return None
 def _cid(self,obs,o):
  c=self._src(obs,o);return self._i(getattr(c,'id',getattr(o,'cardId',0)))
 def _target(self,obs,o):
  try:
   mine=obs.current.players[obs.current.yourIndex];ar=self._i(getattr(o,'inPlayArea',-1),-1);ix=self._i(getattr(o,'inPlayIndex',-1),-1)
   arr=list(mine.active or []) if ar==4 else list(mine.bench or []) if ar==5 else []
   return arr[ix] if 0<=ix<len(arr) else None
  except:return None
 def _find(self,obs,opts,typ=None,cid=None,attack=None,area=None,target_serial=None):
  for i,o in enumerate(opts):
   if typ is not None and o.type!=typ:continue
   if cid is not None and self._cid(obs,o)!=cid:continue
   if attack is not None and self._i(getattr(o,'attackId',0))!=attack:continue
   if area is not None and self._i(getattr(o,'inPlayArea',-1),-1)!=area:continue
   if target_serial is not None:
    p=self._target(obs,o)
    if p is None or self._i(getattr(p,'serial',-1),-1)!=self._i(target_serial,-2):continue
   return i
  return None
 def _ready_drag(self,p):return self._i(getattr(p,'id',0))==self.DRAG and {self.FIRE,self.PSY}.issubset(self._e(p))
 def _threat(self,obs):
  try:
   st=obs.current;op=st.players[1-st.yourIndex];a=op.active[0] if op.active and op.active[0] is not None else None
   if a is None or not list(getattr(a,'energyCards',[]) or []):return False
   cd=self.base.card_table.get(self._i(getattr(a,'id',0)))
   return bool(cd and (getattr(cd,'stage1',False) or getattr(cd,'stage2',False) or getattr(cd,'ex',False) or getattr(cd,'megaEx',False)))
  except:return False
 def _public_wall(self,obs):
  try:
   st=obs.current;op=st.players[1-st.yourIndex];return any(self._i(getattr(p,'id',0)) in self.WALL for p in self._board(op))
  except:return False
 def _promotion(self,obs,chosen):
  opts=list(obs.select.option or []);ci=self._i(chosen[0],-1) if chosen else -1;cards=[self._src(obs,o) for o in opts]
  for i,p in enumerate(cards):
   if p is not None and self._ready_drag(p):self._b('promote_ready_drag');return [i]
  p=cards[ci] if 0<=ci<len(cards) else None
  if p is None:return chosen
  if self._i(getattr(p,'id',0)) in (self.DREEPY,self.DRAK) and self._e(p):
   # Dunsparce is the preferred true pivot; Dudunsparce can draw/shuffle out;
   # Budew is a one-prize emergency shield; unarmed Munk last.
   for want in (self.DUNS,self.DUDUN,self.BUDEW):
    for i,q in enumerate(cards):
     if q is not None and self._i(getattr(q,'id',0))==want and not self._e(q):self._b('pivot_over_charger');return [i]
   for i,q in enumerate(cards):
    if q is not None and self._i(getattr(q,'id',0))==self.MUNK and self.DARK not in self._e(q):self._b('unarmed_munk_over_charger');return [i]
  return chosen
 def _main(self,obs,chosen):
  opts=list(obs.select.option or []);ci=self._i(chosen[0],-1) if chosen else -1
  if not(0<=ci<len(opts)):return chosen
  st=obs.current;mine=st.players[st.yourIndex];op=st.players[1-st.yourIndex];board=self._board(mine);old=opts[ci];oldcid=self._cid(obs,old)
  ready=any(self._ready_drag(p) for p in board)
  # Drakloak Recon Directive is free information: use it before evolving that line,
  # attacking, or ending, unless the deck is effectively empty.
  if self._i(getattr(mine,'deckCount',0))>2:
   q=self._find(obs,opts,typ=self.api.OptionType.ABILITY,cid=self.DRAK)
   if q is not None and old.type in (self.api.OptionType.EVOLVE,self.api.OptionType.ATTACK,self.api.OptionType.END):self._b('drak_draw_before_commit');return [q]
  # Dudunsparce is a draw/pivot engine, not a 3-energy attacker.  Only force the
  # ability when the hand is genuinely low or Dudunsparce is stranded Active.
  q=self._find(obs,opts,typ=self.api.OptionType.ABILITY,cid=self.DUDUN)
  active=mine.active[0] if mine.active and mine.active[0] is not None else None
  if q is not None and (self._i(getattr(mine,'handCount',len(mine.hand or [])))<=3 or (active is not None and self._i(getattr(active,'id',0))==self.DUDUN)) and old.type in (self.api.OptionType.END,self.api.OptionType.ATTACK):
   self._b('runaway_draw');return [q]
  # Adrena-Brain is free healing + damage.  Use before a Phantom/END whenever
  # there is actual own damage and a live opponent target.
  q=self._find(obs,opts,typ=self.api.OptionType.ABILITY,cid=self.MUNK)
  if q is not None and old.type in (self.api.OptionType.ATTACK,self.api.OptionType.END):
   damaged=any(self._i(getattr(p,'hp',0))<self._i(getattr(p,'maxHp',0)) for p in board)
   live=any(self._i(getattr(p,'hp',0))>0 for p in self._board(op))
   if damaged and live:self._b('adrena_before_commit');return [q]
  # Phantom Dive strictly dominates Jet whenever damage is not blocked.
  phantom=self._find(obs,opts,typ=self.api.OptionType.ATTACK,attack=self.PHANTOM)
  if phantom is not None and old.type==self.api.OptionType.ATTACK and self._i(getattr(old,'attackId',0))==self.JET and not self._public_wall(obs):self._b('phantom_over_jet');return [phantom]
  # Darkness is Munkidori's job.  Redirect only when an actual legal Munk target
  # exists; otherwise leave the parent untouched rather than ENDing the turn.
  if old.type==self.api.OptionType.ATTACH and oldcid==self.DARK:
   t=self._target(obs,old)
   if t is not None and self._i(getattr(t,'id',0))!=self.MUNK:
    q=self._find(obs,opts,typ=self.api.OptionType.ATTACH,cid=self.DARK,area=5)
    # find() alone may hit another target, so scan exact Munk candidates.
    for i,o in enumerate(opts):
     if o.type!=self.api.OptionType.ATTACH or self._cid(obs,o)!=self.DARK:continue
     p=self._target(obs,o)
     if p is not None and self._i(getattr(p,'id',0))==self.MUNK and self.DARK not in self._e(p):self._b('dark_to_munk');return [i]
  # User-requested back-line charging: only redirect Active F/P when the opponent
  # is already a public threat AND a legal equivalent Bench charger exists.
  if old.type==self.api.OptionType.ATTACH and oldcid in (self.FIRE,self.PSY) and self._i(getattr(old,'inPlayArea',-1),-1)==4 and self._threat(obs) and not ready:
   t=self._target(obs,old)
   if t is not None and self._i(getattr(t,'id',0)) in (self.DREEPY,self.DRAK):
    other=self.PSY if oldcid==self.FIRE else self.FIRE
    can_finish=(self._i(getattr(t,'id',0))==self.DRAK and other in self._e(t) and self._find(obs,opts,typ=self.api.OptionType.EVOLVE,cid=self.DRAG,target_serial=getattr(t,'serial',-1)) is not None)
    if not can_finish:
     cand=[]
     for i,o in enumerate(opts):
      if o.type!=self.api.OptionType.ATTACH or self._cid(obs,o)!=oldcid or self._i(getattr(o,'inPlayArea',-1),-1)!=5:continue
      p=self._target(obs,o)
      if p is None or self._i(getattr(p,'id',0)) not in (self.DREEPY,self.DRAK,self.DRAG) or oldcid in self._e(p):continue
      cand.append((1 if other in self._e(p) else 0,{119:1,120:2,121:3}.get(self._i(getattr(p,'id',0)),0),len(self._e(p)),-self._i(getattr(p,'serial',0)),i))
     if cand:self._b('fp_to_backline');return [max(cand)[-1]]
  # Risky Ruins is powerful with Dark Munkidori (its 20 self-damage becomes
  # Adrena-Brain fuel), but before that it should not precede our own Poffin.
  if old.type==self.api.OptionType.PLAY and oldcid==self.RISKY:
   dark_munk=any(self._i(getattr(p,'id',0))==self.MUNK and self.DARK in self._e(p) for p in board)
   if not dark_munk:
    pof=self._find(obs,opts,typ=self.api.OptionType.PLAY,cid=self.POFFIN)
    if pof is not None:self._b('poffin_before_risky');return [pof]
  return chosen
 def patch(self,observation,chosen):
  try:
   if not observation.get('select') or not isinstance(chosen,list) or len(chosen)!=1:return chosen
   obs=self.api.to_observation_class(observation);sel=obs.select
   if obs.current is None or sel is None:return chosen
   if sel.context in (self.api.SelectContext.TO_ACTIVE,self.api.SelectContext.SWITCH):return self._promotion(obs,chosen)
   if sel.context==self.api.SelectContext.MAIN and self._i(sel.minCount)==1 and self._i(sel.maxCount)==1:return self._main(obs,chosen)
   return chosen
  except Exception:self._b('exceptions');return chosen
