"""History-aware invariant corrections for known temporal resource failures.

These guards cover effects whose outcome is deterministic from the public state.
They intentionally do not replace the learned policy for ordinary strategy.
"""
from __future__ import annotations


MAIN=0;SWITCH=3;TO_ACTIVE=4;TO_HAND=7;ATTACH_FROM=21;ATTACH_TO=22
PLAY=7;ATTACH=8;EVOLVE=9;ABILITY=10;RETREAT=12;ATTACK=13;END=14
HAND=2;ACTIVE=4;BENCH=5

RIOLU70=333;RIOLU80=677;LUCARIO=678;SOLROCK=676;LUNATONE=675
DUNSPARCE=305;DUDUN=66;DUDUN_EX=306;OGERPON=117
HERO_CAPE=1159;BASIC_F=6;ROCK_F=20
POKE_PAD=1152;POFFIN=1086;FIGHTING_GONG=1142
DRAGAPULT=121;DRAKLOAK=120
AURA_JAB=982;MEGA_BRAVE=983;COSMIC_BEAM=980


def _int(x,d=0):
 try:return int(x if x is not None else d)
 except Exception:return d


def _players(obs):return ((obs.get('current') or {}).get('players') or [{},{}])


def _card_id(card):return _int((card or {}).get('id')) if isinstance(card,dict) else _int(getattr(card,'id',0))


def _zone(obs,area,index,player):
 cur=obs.get('current') or {};sel=obs.get('select') or {};ps=_players(obs)
 try:
  if _int(area)==HAND:items=(ps[player].get('hand') or [])
  elif _int(area)==ACTIVE:items=(ps[player].get('active') or [])
  elif _int(area)==BENCH:items=(ps[player].get('bench') or [])
  elif _int(area)==3:items=(ps[player].get('discard') or [])
  elif _int(area)==1:items=(sel.get('deck') or [])
  else:return None
  return items[_int(index,-1)] if 0<=_int(index,-1)<len(items) else None
 except Exception:return None


def _energies(p):return len((p or {}).get('energyCards') or (p or {}).get('energies') or [])


def _tools(p):return [_card_id(x) for x in ((p or {}).get('tools') or [])]


class TemporalSafetyGate:
 def __init__(self,history,card_table=None):
  self.history=history;self.card_table=card_table or {};self.reset()

 def reset(self):
  self.pending_pivot_turn=-1;self.pending_pivot_serial=0
  self.pending_successor_turn=-1;self.successor_attempted_turn=-1;self.successor_developed_turn=-1
  self.pending_relay_turn=-1;self.pending_relay_serial=0
  self.stats={'calls':0,'overrides':{},'cosmic_fail_blocked':0,'aura_energy_trimmed':0,'aura_target_redirected':0,'cape_redirected':0,'ready_attacker_pivots':0,'successor_search_attempted':0,'successor_search_empty':0,'successor_relay_armed':0,'aura_successor_relayed':0,'successor_before_safe_ko':0}

 def _note(self,key):
  self.stats['overrides'][key]=self.stats['overrides'].get(key,0)+1
  if key in self.stats:self.stats[key]+=1

 def _option_card(self,obs,o,me):
  typ=_int(o.get('type'),-1)
  if typ in {PLAY,ATTACH}:return _zone(obs,HAND,o.get('index'),me)
  return _zone(obs,o.get('area'),o.get('index'),_int(o.get('playerIndex'),me))

 def _target(self,obs,o,me):return _zone(obs,o.get('inPlayArea'),o.get('inPlayIndex'),me)

 def _last_own_attack(self,me):
  for e in reversed(self.history.events):
   if _int(e.get('playerIndex'),-1)==me and _int(e.get('type'),-1)==15:return _int(e.get('attackId')), _int(e.get('turn'),-1)
  return 0,-1

 def _goal(self,p,bench,hand):
  cid=_card_id(p)
  if cid in {RIOLU70,RIOLU80,LUCARIO}:return 2
  if cid==SOLROCK:return 1
  if cid in {DUDUN_EX,OGERPON}:return 3
  if cid in {DUNSPARCE,DUDUN}:
   # Only pre-charge the draw pivot when its attack evolution is already public
   # to us; otherwise reserve acceleration for an actual Lucario line.
   has_ex=any(_card_id(x)==DUDUN_EX for x in hand)
   return 3 if has_ex else 0
  return 0

 def _family(self,obs,me):
  op=1-me;ids=set(self.history.revealed_ids(op))
  ps=_players(obs);enemy=ps[op] if 0<=op<len(ps) else {}
  for zone in ('active','bench','discard'):
   ids.update(_card_id(x) for x in (enemy.get(zone) or []) if x)
  if ids&{119,120,121}:return 'dragapult'
  if ids&{741,742,743,245}:return 'alakazam'
  # This league list can open on its support attackers (Munkidori/Snorunt/Yveltal)
  # several turns before the Impidimp line is exposed.
  if ids&{104,112,646,647,648,860}:return 'marnie'
  return 'unknown'

 def _deficit(self,p,bench,hand):return max(0,self._goal(p,bench,hand)-_energies(p))

 def _attack_damage(self,p,last_attack,last_turn,turn):
  cid=_card_id(p);en=_energies(p)
  if cid==LUCARIO:
   locked=last_attack==MEGA_BRAVE and last_turn==turn-2
   if en>=2 and not locked:return 270
   if en>=1:return 130
  if cid==SOLROCK and en>=1 and any(_card_id(x)==LUNATONE for x in []):return 70
  if cid==DUDUN_EX and en>=3:return 150
  if cid==OGERPON and en>=3:return 140
  if cid in {RIOLU70,RIOLU80} and en>=1:return 30
  return 0

 def _ready_pivot(self,obs,me):
  cur=obs.get('current') or {};ps=_players(obs);mine=ps[me];opp=ps[1-me]
  active=(mine.get('active') or [None])[0];target=(opp.get('active') or [None])[0]
  if not active or not target:return None
  bench=[x for x in (mine.get('bench') or []) if x];turn=_int(cur.get('turn'))
  last_attack,last_turn=self._last_own_attack(me)
  luna=any(_card_id(x)==LUNATONE for x in bench)
  current=70 if _card_id(active)==SOLROCK and luna and _energies(active)>=1 else 0
  target_meta=self.card_table.get(_card_id(target));prizes=3 if bool(getattr(target_meta,'megaEx',False)) else 2 if bool(getattr(target_meta,'ex',False)) else 1
  best=None
  for q in bench:
   dmg=self._attack_damage(q,last_attack,last_turn,turn)
   if dmg<=0:continue
   meta=self.card_table.get(_card_id(target))
   try:
    if _int(getattr(meta,'weakness',-1),-1)==6:dmg*=2
    if _int(getattr(meta,'resistance',-1),-1)==6:dmg=max(0,dmg-30)
   except Exception:pass
   if dmg<_int(target.get('hp')):continue
   if current>=_int(target.get('hp')):continue
   # For a one-Prize target, preserve the pivot unless the current Active has no
   # functioning attack.  Ex/Mega ex immediate KOs are always tempo-critical.
   if prizes<2 and current>0:continue
   row=(prizes,dmg,-_energies(q),-_int(q.get('serial')),q)
   if best is None or row[:-1]>best[:-1]:best=row
  return best[-1] if best else None

 def _redirect_cape(self,obs,base,me,opts):
  if len(base)!=1:return None
  bi=_int(base[0],-1)
  if not 0<=bi<len(opts):return None
  bo=opts[bi]
  if _int(bo.get('type'),-1)!=ATTACH:return None
  tool=self._option_card(obs,bo,me);target=self._target(obs,bo,me)
  if _card_id(tool)!=HERO_CAPE or not target:return None
  if _card_id(target) not in {SOLROCK,LUNATONE,DUNSPARCE,DUDUN}:return None
  maxhp=_int(target.get('maxHp'),_int(target.get('hp')))
  if _int(target.get('hp'))<maxhp:return None
  choices=[]
  for i,o in enumerate(opts):
   if _int(o.get('type'),-1)!=ATTACH:continue
   c=self._option_card(obs,o,me);q=self._target(obs,o,me)
   if _card_id(c)!=HERO_CAPE or not q or _card_id(q) not in {RIOLU70,RIOLU80,LUCARIO}:continue
   cid=_card_id(q);score=(3 if cid==LUCARIO else 2, _energies(q), _int(q.get('hp')), -i)
   choices.append((score,i))
  if not choices:return None
  choices.sort(reverse=True);return [choices[0][1]]

 def choose(self,obs,base):
  if not isinstance(obs,dict) or not isinstance(base,list):return base
  cur=obs.get('current') or {};sel=obs.get('select') or {};opts=sel.get('option') or []
  if not cur or not opts:return base
  self.stats['calls']+=1;ctx=_int(sel.get('context'),-1);turn=_int(cur.get('turn'));me=_int(cur.get('yourIndex'),0)
  ps=_players(obs);mine=ps[me];bench=[x for x in (mine.get('bench') or []) if x];hand=[x for x in (mine.get('hand') or []) if x]
  family=self._family(obs,me)

  # The stored Marnie league is already a retained non-regression family in
  # v148.  Even individually sensible redirects changed its long counterfactual
  # lines and lost ground in the 300-game gate, so preserve v148 bit-for-bit as
  # soon as that public family is identified.
  if family=='marnie':
   self.pending_pivot_turn=-1;self.pending_pivot_serial=0
   self.pending_successor_turn=-1;self.successor_attempted_turn=-1;self.successor_developed_turn=-1
   self.pending_relay_turn=-1;self.pending_relay_serial=0
   return base

  # A search started by the relay guard must actually resolve to a Riolu and
  # put it onto the Bench before ordinary priorities resume.
  if self.pending_successor_turn==turn:
   if ctx==TO_HAND:
    for i,o in enumerate(opts):
     q=self._option_card(obs,o,me)
     if q and _card_id(q) in {RIOLU70,RIOLU80}:
      return [i]
    # A public search can miss because all Riolu are prized, discarded or
    # already consumed.  Do not carry a stale relay obligation into later
    # choices, and do not burn a second search Item in the same turn.
    self.pending_successor_turn=-1;self._note('successor_search_empty')
    return base
   if ctx==MAIN:
    for i,o in enumerate(opts):
     if _int(o.get('type'),-1)!=PLAY:continue
     q=self._option_card(obs,o,me)
     if q and _card_id(q) in {RIOLU70,RIOLU80}:
      self.pending_successor_turn=-1;self.successor_developed_turn=turn
      self._note('successor_before_safe_ko');return [i]

  # Complete a pivot chosen at the preceding Main selection.
  if ctx in {SWITCH,TO_ACTIVE} and self.pending_pivot_turn==turn and self.pending_pivot_serial:
   for i,o in enumerate(opts):
    q=self._option_card(obs,o,me)
    if q and _int(q.get('serial'))==self.pending_pivot_serial:
     self.pending_pivot_turn=-1;self.pending_pivot_serial=0;self._note('ready_attacker_pivots');return [i]

  # Aura Jab selects the discard energies first.  Never select more cards than
  # the current bench can use at its real attack thresholds.
  if ctx==ATTACH_TO and family in {'dragapult','unknown'}:
   capacity=sum(self._deficit(q,bench,hand) for q in bench)
   want=min(3,capacity,_int(sel.get('maxCount'),3))
   want=max(_int(sel.get('minCount'),0),want)
   if len(base)>want:
    trimmed=[]
    if want<=0:
     self._note('aura_energy_trimmed');return []
    for i in list(base)+list(range(len(opts))):
     if i not in trimmed:trimmed.append(i)
     if len(trimmed)>=want:break
    self._note('aura_energy_trimmed');return trimmed

  # Each selected Aura Jab Energy then asks for a destination.  Respect the
  # threshold after every attachment so a 2-Energy Dudunsparce ex cannot become
  # a 5-Energy sink and a Riolu cannot receive a third Energy.
  if ctx==ATTACH_FROM and len(base)==1:
   if self.pending_relay_turn==turn and self.pending_relay_serial:
    for i,o in enumerate(opts):
     q=self._option_card(obs,o,me)
     if q and _int(q.get('serial'))==self.pending_relay_serial and self._deficit(q,bench,hand)>0:
      if self._deficit(q,bench,hand)==1:
       self.pending_relay_turn=-1;self.pending_relay_serial=0;self.successor_developed_turn=turn
       self._note('successor_before_safe_ko')
      self._note('aura_successor_relayed');return [i]
    self.pending_relay_turn=-1;self.pending_relay_serial=0
   bi=_int(base[0],-1);chosen=self._option_card(obs,opts[bi],me) if 0<=bi<len(opts) else None
   chosen_goal=self._goal(chosen,bench,hand) if chosen is not None else 0
   # Only redirect a proven over-cap attachment.  Do not reinterpret ordinary
   # pre-charging of Dunsparce or another speculative body from sparse data.
   if chosen is not None and (chosen_goal<=0 or _energies(chosen)<chosen_goal):return base
   cand=[]
   for i,o in enumerate(opts):
    q=self._option_card(obs,o,me)
    if not q:continue
    need=self._deficit(q,bench,hand)
    if need<=0:continue
    cid=_card_id(q);role=5 if cid==LUCARIO else 4 if cid in {RIOLU70,RIOLU80} else 3 if cid==DUDUN_EX else 2 if cid==SOLROCK else 1
    cand.append((role,need,-_energies(q),-i,i))
   if cand:
    cand.sort(reverse=True);self._note('aura_target_redirected');return [cand[0][-1]]

  if ctx==MAIN:
   cape=self._redirect_cape(obs,base,me,opts) if family!='marnie' else None
   if cape is not None:
    self._note('cape_redirected');return cape

   chosen=[opts[_int(i)] for i in base if 0<=_int(i)<len(opts)]
   attack_ids={_int(o.get('attackId')) for o in chosen if _int(o.get('type'),-1)==ATTACK}
   active=(mine.get('active') or [None])[0]

   # Human replay 20260815_010149: do not spend an otherwise safe 130-damage
   # Aura Jab KO before establishing the next Riolu when the damaged Active is
   # about to be answered by another Dragapult.  Search/play is Item/basic-only,
   # so it preserves the attack and the Aura Jab acceleration window.
   enemy=ps[1-me];enemy_active=(enemy.get('active') or [None])[0]
   enemy_bench=[x for x in (enemy.get('bench') or []) if x]
   no_successor=not any(_card_id(q) in {RIOLU70,RIOLU80,LUCARIO} for q in bench)
   return_threat=any(_card_id(q) in {DRAKLOAK,DRAGAPULT} and _energies(q)>=1 for q in enemy_bench)
   safe_ko=(AURA_JAB in attack_ids and _card_id(active)==LUCARIO and _card_id(enemy_active)==DRAGAPULT and _int(enemy_active.get('hp'))<=130)
   my_prizes=len(mine.get('prize') or [])
   if family=='dragapult' and safe_ko and no_successor and return_threat and _int(active.get('hp'))<=220 and my_prizes>2 and len(bench)<5 and self.successor_developed_turn!=turn and self.successor_attempted_turn!=turn:
    # A public, partially charged secondary attacker is safer than spending a
    # search Item that may reveal no Riolu.  Aura Jab can advance it without
    # giving up the guaranteed KO; prefer the line nearest its real threshold.
    alternate=[]
    for q in bench:
     if _card_id(q) not in {DUDUN_EX,OGERPON}:continue
     need=self._deficit(q,bench,hand)
     if need>0:alternate.append((need,0 if _card_id(q)==DUDUN_EX else 1,_int(q.get('serial'))))
    if alternate:
     alternate.sort();self.pending_relay_turn=turn;self.pending_relay_serial=alternate[0][-1]
     self.successor_attempted_turn=turn;self._note('successor_relay_armed')
     return base
    direct=[];search=[]
    for i,o in enumerate(opts):
     if _int(o.get('type'),-1)!=PLAY:continue
     q=self._option_card(obs,o,me);cid=_card_id(q)
     if cid in {RIOLU70,RIOLU80}:direct.append(i)
     elif cid in {POKE_PAD,POFFIN,FIGHTING_GONG}:search.append((2 if cid==POKE_PAD else 1,i))
    if direct:
     self.successor_developed_turn=turn;self._note('successor_before_safe_ko');return [direct[0]]
    if search:
     search.sort(reverse=True);self.pending_successor_turn=turn;self.successor_attempted_turn=turn
     self._note('successor_search_attempted')
     return [search[0][1]]

   # Cosmic Beam explicitly does nothing without a Benched Lunatone.  A legal
   # End action is state-equivalent and avoids consuming attack-only resources.
   if family!='marnie' and COSMIC_BEAM in attack_ids and _card_id(active)==SOLROCK and not any(_card_id(q)==LUNATONE for q in bench):
    ready=self._ready_pivot(obs,me)
    if ready is not None:
     for i,o in enumerate(opts):
      if _int(o.get('type'),-1)==RETREAT:
       self.pending_pivot_turn=turn;self.pending_pivot_serial=_int(ready.get('serial'));self._note('ready_attacker_pivots');return [i]
    for i,o in enumerate(opts):
     if _int(o.get('type'),-1)==END:
      self._note('cosmic_fail_blocked');return [i]

   # If a utility Active cannot take the opposing ex while a Benched attacker
   # can do so immediately, take the public guaranteed Prize line.
   if family=='dragapult' and _card_id(active) in {SOLROCK,LUNATONE,DUNSPARCE,DUDUN,DUDUN_EX} and not any(_int(o.get('type'),-1)==RETREAT for o in chosen):
    ready=self._ready_pivot(obs,me)
    if ready is not None:
     for i,o in enumerate(opts):
      if _int(o.get('type'),-1)==RETREAT:
       self.pending_pivot_turn=turn;self.pending_pivot_serial=_int(ready.get('serial'));self._note('ready_attacker_pivots');return [i]
  return base

 def get_stats(self):return dict(self.stats)
