"""Dense public-state/action evaluator distilled from v23 battle replays.

The evaluator scores every legal MAIN action.  Its replay target combines
field, hand, prize map, attacker readiness and a discounted three-decision
future.  Overrides are deliberately conservative: an exact supported replay
bucket or a guaranteed tactical improvement is required.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from cg.api import AreaType, CardType, OptionType, SelectContext, all_attack, all_card_data, to_observation_class

ROOT=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
try:MODEL=json.load(open(os.path.join(ROOT,'dense_value_model_v23.json'),encoding='utf-8'))
except Exception:MODEL={'minimum_runtime_count':999999,'action_table':{}}
TABLE=MODEL.get('action_table') or {};MIN_N=int(MODEL.get('minimum_runtime_count',8))
CARDS={int(x.cardId):x for x in all_card_data()};ATTACKS={int(x.attackId):x for x in all_attack()}
READY={96:3,108:3,117:3,184:3,272:2,31:2,756:3,230:2,112:2,140:3,1071:3}
ENGINE={96,184,756,1071};TECH={'dragapult':{272,108},'lucario':{272},'alakazam':{117,96},'crustle':{117,31,230}}
DRAW={1094,1188,1227};SEARCH={1121,1127};COMPLEX={1116,1182,1221}
STATS={'decisions':0,'options':0,'supported':0,'overrides':0,'ko_overrides':0,'completion_overrides':0}
PROMOTED_OVERRIDE_MATCHES=set()

def reset():
 for key in STATS:STATS[key]=0

def _phase(turn):return 'early' if turn<=3 else 'mid' if turn<=8 else 'late'
def _pl(obs,own=True):
 s=obs.current;return s.players[s.yourIndex if own else 1-s.yourIndex]
def _board(obs,own=True):
 p=_pl(obs,own);return [x for x in list(p.active or [])+list(p.bench or []) if x]

def _source(obs,o):
 s=obs.current;me=int(s.yourIndex);pi=int(o.playerIndex) if o.playerIndex is not None else me;pl=s.players[pi]
 try:
  area=int(o.area) if o.area is not None else (int(AreaType.HAND) if o.type==OptionType.PLAY else -1)
  zones={int(AreaType.DECK):obs.select.deck or [],int(AreaType.HAND):pl.hand or [],int(AreaType.DISCARD):pl.discard or [],int(AreaType.ACTIVE):pl.active or [],int(AreaType.BENCH):pl.bench or [],int(AreaType.STADIUM):s.stadium or [],int(AreaType.LOOKING):s.looking or []}
  return zones.get(area,[])[int(o.index)]
 except Exception:return None

def _target(obs,o):
 try:
  pl=_pl(obs,True);area=int(o.inPlayArea);index=int(o.inPlayIndex)
  return (pl.active if area==int(AreaType.ACTIVE) else pl.bench if area==int(AreaType.BENCH) else [])[index]
 except Exception:return None

def _energy(p):return len(getattr(p,'energyCards',None) or [])
def _progress(p,plus=0):
 if p is None:return 0.0
 need=READY.get(int(p.id),3);return min(1.0,(_energy(p)+plus)/max(1,need))

def _can_pay(p,attack):
 if p is None or attack is None:return False
 pool=list(getattr(p,'energies',None) or [])
 for req in attack.energies:
  req=int(req)
  if req==0:
   if not pool:return False
   pool.pop(0);continue
  j=next((i for i,e in enumerate(pool) if int(e) in {req,10}),None)
  if j is None:return False
  pool.pop(j)
 return True

def _damage(obs,p,aid):
 if p is None:return 0
 aid=int(aid or 0);opp=_pl(obs,False);oa=opp.active[0] if opp.active else None
 if aid==120:return 30+30*(_energy(p)+(_energy(oa) if oa else 0))
 if aid==371:
  value=20+20*(len([x for x in _pl(obs,True).bench if x])+len([x for x in opp.bench if x]))
  return value*2 if oa and int(oa.id) in {119,120,121,235} else value
 if aid==1092:return 250
 if aid==148:return 140
 if aid==136:return 100
 if aid==243:return 200
 if aid==183:return 100
 if aid==20:return 120 if obs.current.stadium else 60
 if aid==315:return 85
 if aid==141:return 85
 attack=ATTACKS.get(aid);return int(attack.damage or 0) if attack else 0

def _ready_damage(obs,p):
 card=CARDS.get(int(getattr(p,'id',0) or 0));best=0
 if not card:return 0
 for aid in card.attacks:
  attack=ATTACKS.get(int(aid))
  if _can_pay(p,attack):best=max(best,_damage(obs,p,aid))
 return best

def _ids(obs):return {int(p.id) for p in _board(obs,True)}

def _key(obs,o,match):
 mine=_pl(obs,True);theirs=_pl(obs,False);bucket='ahead' if len(mine.prize or [])<len(theirs.prize or []) else 'behind' if len(mine.prize or [])>len(theirs.prize or []) else 'even'
 active=mine.active[0] if mine.active else None;c=_source(obs,o);t=_target(obs,o)
 fields=(match,_phase(int(obs.current.turn or 0)),bucket,int(obs.select.context),int(o.type),int(getattr(c,'id',0) or 0),int(getattr(t,'id',0) or 0),int(getattr(o,'attackId',0) or 0),int(getattr(active,'id',0) or 0))
 return '|'.join(map(str,fields))

def _learned(obs,o,match):
 row=TABLE.get(_key(obs,o,match))
 if not row or int(row.get('n',0))<MIN_N:return 0.0,0
 STATS['supported']+=1
 return float(row.get('residual',0.0)),int(row.get('n',0))

def _tactical(obs,o,match):
 pl=_pl(obs,True);active=pl.active[0] if pl.active else None;opp=_pl(obs,False);oa=opp.active[0] if opp.active else None
 c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);t=_target(obs,o);typ=o.type;score=0.0;certain=False
 active_damage=_ready_damage(obs,active) if active else 0
 if typ==OptionType.ATTACK:
  damage=_damage(obs,active,o.attackId);score+=damage*0.28
  if oa and damage>=int(oa.hp or 0):score+=115.0;certain=True
  if active and int(active.id) in TECH.get(match,set()):score+=16.0
 elif typ==OptionType.ATTACH and t is not None:
  before=_progress(t);after=_progress(t,1);score+=(after-before)*105.0
  if before<1.0<=after:score+=82.0;certain=True
  if int(t.id) in TECH.get(match,set()):score+=24.0
  if before>=1.0:score-=72.0
 elif typ==OptionType.ABILITY:
  if cid==96:score+=62.0
  elif cid==756:score+=48.0
  elif cid==140:score+=30.0
  else:score+=18.0
 elif typ==OptionType.PLAY:
  ids=_ids(obs);turn=int(obs.current.turn or 0)
  if cid==1094:score+=52.0 if turn<=6 else 18.0
  elif cid in DRAW:score+=max(0,7-int(pl.handCount or 0))*8.0
  elif cid in SEARCH:score+=38.0
  elif cid==1250:score+=35.0 if 96 in ids else -25.0
  elif c is not None and CARDS.get(cid) and CARDS[cid].cardType==CardType.POKEMON:
   score+=35.0 if cid in ENGINE and cid not in ids else 20.0 if cid in TECH.get(match,set()) and cid not in ids else -15.0
 elif typ==OptionType.RETREAT:
  bench=max([_ready_damage(obs,p) for p in pl.bench if p] or [0]);score+=(bench-active_damage)*0.5
  if bench>active_damage:score+=35.0
 elif typ==OptionType.END:
  if active_damage>0:score-=130.0;certain=True
  else:score-=12.0
 return score,certain

def _base_guard(obs,base_index,best_index):
 opts=obs.select.option;base=opts[base_index];best=opts[best_index];bc=_source(obs,base);bid=int(getattr(bc,'id',0) or 0)
 # Preserve proven engine actions and multi-stage trainer intents unless the
 # replacement is a guaranteed knockout or repairs a literal END leak.
 if base.type==OptionType.ABILITY and bid in {96,756,140}:return True
 if base.type==OptionType.PLAY and bid in {1094,1116,1182,1221}:return True
 if base.type==OptionType.ATTACK and best.type!=OptionType.ATTACK:return True
 return False

def choose(observation,base,match):
 if os.environ.get('TERA_DENSE_DISABLE')=='1':return base
 try:obs=to_observation_class(observation)
 except Exception:return base
 if match not in {'dragapult','lucario','alakazam','crustle','marnie','archaludon'} or obs.select is None or obs.select.context!=SelectContext.MAIN or not base or len(base)!=1:return base
 if not (0<=int(base[0])<len(obs.select.option)):return base
 STATS['decisions']+=1;STATS['options']+=len(obs.select.option)
 scored=[]
 for i,o in enumerate(obs.select.option):
  learned,n=_learned(obs,o,match);tactical,certain=_tactical(obs,o,match)
  value=learned+tactical+(46.0 if i==int(base[0]) else 0.0)
  scored.append((value,n,certain,-i,i))
 # Every matchup is evaluated.  No replay-only override survived its 100-game
 # promotion gate, so the dense scores remain advisory and the verified direct
 # specialist retains final authority.  This avoids converting correlation in
 # loss replays into a causal action rule.
 if match not in PROMOTED_OVERRIDE_MATCHES:return base
 best=max(scored);bi=int(base[0]);base_score=scored[bi][0]
 if best[4]==bi or best[0]<base_score+12.0:return base
 candidate=obs.select.option[best[4]];base_opt=obs.select.option[bi]
 # An unsupported correlation never overrides.  Tactical certainties are
 # restricted to KO, attack-completing attachment, or END-with-attack repair.
 if best[1]<MIN_N and not best[2]:return base
 if _base_guard(obs,bi,best[4]):
  active=_pl(obs,True).active[0] if _pl(obs,True).active else None;oa=_pl(obs,False).active[0] if _pl(obs,False).active else None
  ko=candidate.type==OptionType.ATTACK and oa and _damage(obs,active,candidate.attackId)>=int(oa.hp or 0)
  if not ko:return base
 if candidate.type==OptionType.ATTACK:
  active=_pl(obs,True).active[0] if _pl(obs,True).active else None;oa=_pl(obs,False).active[0] if _pl(obs,False).active else None
  if oa and _damage(obs,active,candidate.attackId)>=int(oa.hp or 0):STATS['ko_overrides']+=1
 if candidate.type==OptionType.ATTACH and best[2]:STATS['completion_overrides']+=1
 STATS['overrides']+=1
 return [best[4]]
