"""Public-signature router for promoted weak-match specialists."""
from __future__ import annotations
import hashlib,importlib.util,os,sys
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:sys.path.insert(0,R)
def _load(tag):
 p=os.path.join(R,f'policy_{tag}.py');name='_tera_v19_weak_'+tag+'_'+hashlib.sha1((R+tag).encode()).hexdigest()[:10]
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
POL={
 'generic':_load('generic'),
 'dragapult':_load('memory_dragapult'),
 'lucario':_load('memory_lucario'),
 'alakazam':_load('rollout_alakazam'),
 'crustle':_load('memory_crustle'),
}
SIG={
 'dragapult':{119,120,121,235},
 'lucario':{333,677,678},
 'alakazam':{245,741,742,743},
 'crustle':{343,344,345,1264},
}
_seen=set();_route='generic'
def setup_agent(d):
 votes={};generic=None
 for name,module in POL.items():
  try:
   action=tuple(int(x) for x in module.agent(d))
   if name=='generic':generic=action
   else:votes[action]=votes.get(action,0)+1
  except Exception:pass
 if not votes:return list(generic or ())
 best=max(votes,key=lambda action:(votes[action],action==generic,tuple(-x for x in action)))
 return list(best)
def _reset(d):
 global _seen,_route
 _seen=set();_route='generic';deck=None
 for name,module in POL.items():
  try:
   value=module.agent(d)
   if name=='generic':deck=value
  except Exception:pass
 return deck or POL['generic'].agent(d)
def _observe(d):
 global _route
 current=d.get('current') if isinstance(d,dict) else None
 if not current:return
 me=int(current.get('yourIndex',0));players=current.get('players') or []
 if len(players)<2:return
 opponent=players[1-me]
 for zone in ('active','bench','discard','lostZone'):
  for card in opponent.get(zone) or []:
   if card and card.get('id') is not None:_seen.add(int(card['id']))
   for prior in (card or {}).get('preEvolution') or []:
    if prior and prior.get('id') is not None:_seen.add(int(prior['id']))
 for name in ('lucario','dragapult','crustle','alakazam'):
  if _seen&SIG[name]:_route=name;return
def agent(d):
 if d.get('select') is None and d.get('current') is None:return _reset(d)
 _observe(d)
 try:return POL.get(_route,POL['generic']).agent(d)
 except Exception:return POL['generic'].agent(d)
