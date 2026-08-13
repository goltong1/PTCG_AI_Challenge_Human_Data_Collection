"""Promoted direct specialists selected by the v23 entry-policy league."""
from __future__ import annotations
import hashlib,importlib.util,os,sys
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:sys.path.insert(0,R)
def _load(tag):
 p=os.path.join(R,'policy_'+tag+'.py');name='_tera_v23_direct_'+tag+'_'+hashlib.sha1((R+tag).encode()).hexdigest()[:10]
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
POL={'generic':_load('generic'),'lucario':_load('memory_lucario'),'alakazam':_load('special_alakazam')}
SEEN=set();ROUTE='generic';LUC={333,675,676,677,678};ALA={245,272,741,742,743}
def _reset(d):
 global ROUTE
 SEEN.clear();ROUTE='generic';deck=None
 for name,module in POL.items():
  try:value=module.agent(d);deck=value if name=='generic' else deck
  except Exception:pass
 return deck or POL['generic'].agent(d)
def _observe(d):
 global ROUTE
 cur=d.get('current') if isinstance(d,dict) else None
 if not cur:return
 me=int(cur.get('yourIndex',0));players=cur.get('players') or []
 if len(players)<2:return
 for zone in ('active','bench','discard','lostZone'):
  for card in players[1-me].get(zone) or []:
   if card and card.get('id') is not None:SEEN.add(int(card['id']))
   for prior in (card or {}).get('preEvolution') or []:
    if prior and prior.get('id') is not None:SEEN.add(int(prior['id']))
 if SEEN&LUC:ROUTE='lucario'
 elif SEEN&ALA:ROUTE='alakazam'
def agent(d):
 if d.get('select') is None and d.get('current') is None:return _reset(d)
 _observe(d)
 try:return POL[ROUTE].agent(d)
 except Exception:return POL['generic'].agent(d)
