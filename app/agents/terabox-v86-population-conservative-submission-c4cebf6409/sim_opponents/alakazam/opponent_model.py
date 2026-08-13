from __future__ import annotations
import os,sys,hashlib,importlib.util
HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
from cg.api import all_attack

def _load(tag,fn):
 n='_portfolio_'+tag+'_'+hashlib.sha1((HERE+fn).encode()).hexdigest()[:10]
 s=importlib.util.spec_from_file_location(n,os.path.join(HERE,fn));m=importlib.util.module_from_spec(s);sys.modules[n]=m
 assert s.loader is not None;s.loader.exec_module(m);return m
ENGINES={'current':_load('current','engine_current.py'), 'core':_load('core','engine_core.py'), 'legacy':_load('legacy','engine_legacy.py')}
MY_DECK=[int(x) for x in open(os.path.join(HERE,'deck.csv'),encoding='utf-8').read().replace(',',' ').split()]
ROUTES={'dragapult': 'current', 'dusk': 'current', 'marnie': 'current', 'lucario': 'current', 'alakazam': 'current', 'default': 'current'}
SEEN=set();TURN=0;LOCKED=False
ATTACK_DAMAGE={int(a.attackId):int(a.damage or 0) for a in all_attack()}
SIG_DRAG={119,120,121};SIG_DUSK={131,132,133};SIG_ALA={741,742,743};SIG_MARNIE={646,647,648,860,104};SIG_LUC={333,675,676,677,678}
DECK_NAME='alakazam'
STATS={'routes':{},'safety_ko':0,'safety_future':0}

def _reset():
 global TURN,LOCKED
 SEEN.clear();TURN=0;LOCKED=False

def _observe(o):
 global TURN,LOCKED
 c=o.get('current') or {};TURN=int(c.get('turn') or 0);me=int(c.get('yourIndex') or 0);ps=c.get('players') or []
 if TURN==0:_reset()
 if len(ps)>=2:
  op=ps[1-me]
  for z in (op.get('active') or [])+(op.get('bench') or [])+(op.get('discard') or []):
   if not z:continue
   if z.get('id'):SEEN.add(int(z['id']))
   for q in z.get('preEvolution') or []:
    if q and q.get('id'):SEEN.add(int(q['id']))
 for l in o.get('logs') or []:
  if int(l.get('playerIndex',me))!=1-me:continue
  for k in ('cardId','cardIdAfter','cardIdBefore','cardIdActive','cardIdBench','cardIdTarget'):
   if l.get(k):SEEN.add(int(l[k]))
 if TURN>=3:LOCKED=True

def _family():
 if SEEN & SIG_DUSK:return 'dusk'
 if SEEN & SIG_ALA:return 'alakazam'
 if SEEN & SIG_MARNIE:return 'marnie'
 if SEEN & SIG_LUC:return 'lucario'
 if SEEN & SIG_DRAG:return 'dragapult'
 return 'default'

def _get_target(c,opt):
 try:
  me=int(c.get('yourIndex') or 0);pl=c['players'][me];ar=int(opt.get('inPlayArea',-1));ix=int(opt.get('inPlayIndex',0))
  arr=pl.get('active') or [] if ar==4 else pl.get('bench') or [] if ar==5 else []
  return arr[ix] if 0<=ix<len(arr) else None
 except Exception:return None

def _guaranteed_ko_attack(o):
 s=o.get('select') or {};c=o.get('current') or {};opts=s.get('option') or []
 try:
  me=int(c.get('yourIndex') or 0);op=c['players'][1-me];active=(op.get('active') or [None])[0];hp=int(active.get('hp') or 0) if active else 0
  my=c['players'][me];hand=int(my.get('handCount') or len(my.get('hand') or []))
 except Exception:return None
 best=None;bestd=-1
 for i,x in enumerate(opts):
  if int(x.get('type',-1))!=13:continue
  aid=int(x.get('attackId') or -1);d=ATTACK_DAMAGE.get(aid,0)
  if aid==1072:d=20*hand
  if d>=hp and d>bestd:best=i;bestd=d
 return best

def _safety(o,chosen):
 if not isinstance(chosen,list) or len(chosen)!=1:return chosen
 s=o.get('select') or {};c=o.get('current') or {}
 if int(s.get('context',-1))!=0:return chosen
 opts=s.get('option') or [];ix=chosen[0]
 if not (0<=ix<len(opts)):return chosen
 typ=int(opts[ix].get('type',-1));ko=_guaranteed_ko_attack(o)
 # A guaranteed prize is never discarded by simply ending. Lucario also avoids
 # wasting another attachment on an already attack-ready active attacker.
 luc_overattach=False
 if DECK_NAME=='lucario' and typ==8 and int(opts[ix].get('inPlayArea',-1))==4 and (SEEN & SIG_ALA):
  try:
   me=int(c.get('yourIndex') or 0);active=(c['players'][me].get('active') or [None])[0]
   luc_overattach=bool(active and int(active.get('id',-1))==678 and len(active.get('energyCards') or [])>=1)
  except Exception:luc_overattach=False
 if ko is not None and (typ==14 or luc_overattach):
  STATS['safety_ko']+=1;return [ko]
 # Alakazam's official loss data contained a clean no-attack leak while no
 # corresponding winning example benefited from passing. Preserve hand-building
 # actions, but never simply end when a legal attack is already available.
 if DECK_NAME=='alakazam' and typ==14:
  attacks=[(ATTACK_DAMAGE.get(int(x.get('attackId') or -1),0),j) for j,x in enumerate(opts) if int(x.get('type',-1))==13]
  if attacks:STATS['safety_ko']+=1;return [max(attacks)[1]]
 # Dragapult families: when badly behind, do not end while a concrete next-turn
 # evolution or a useful bench energy attachment remains available.
 if DECK_NAME in {'dragapult','dusk'} and typ==14:
  try:
   me=int(c.get('yourIndex') or 0);my=c['players'][me];op=c['players'][1-me]
   behind=len(my.get('prize') or [])-len(op.get('prize') or [])
  except Exception:behind=0
  if behind>=3:
   # evolution first: it preserves a threatened line and unlocks future attacks.
   for j,x in enumerate(opts):
    if int(x.get('type',-1))==9:
     t=_get_target(c,x)
     if t and int(t.get('id',-1)) in {119,120,131,132}:STATS['safety_future']+=1;return [j]
   # then attach only to a benched Dragapult line, preferring evolved/older targets.
   cand=[]
   for j,x in enumerate(opts):
    if int(x.get('type',-1))!=8 or int(x.get('inPlayArea',-1))!=5:continue
    t=_get_target(c,x)
    if t and int(t.get('id',-1)) in {119,120,121}:
     rank={121:3,120:2,119:1}.get(int(t.get('id')),0)+(0 if t.get('appearThisTurn') else 1)
     cand.append((rank,j))
   if cand:STATS['safety_future']+=1;return [max(cand)[1]]
 return chosen

def agent(o:dict)->list[int]:
 if o.get('select') is None:
  _reset()
  for m in ENGINES.values():
   try:m.agent(o)
   except Exception:pass
  return list(MY_DECK)
 _observe(o);fam=_family();ename=ROUTES.get(fam,ROUTES['default'])
 STATS['routes'][fam]=STATS['routes'].get(fam,0)+1
 # Engines are intentionally isolated. Calling several policies on the same
 # decision was found to contaminate search state and recreate the stale shared
 # controller problem. Only the selected engine is executed.
 try:chosen=ENGINES[ename].agent(o)
 except Exception:
  fallback=ROUTES['default'];chosen=ENGINES[fallback].agent(o)
 return _safety(o,chosen)
