from __future__ import annotations
import os,sys,importlib.util,hashlib
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:sys.path.insert(0,R)
def _load(tag,fn):
 p=os.path.join(R,fn);key='_tera_v24_'+tag+'_'+hashlib.sha1((R+tag+fn).encode()).hexdigest()[:10]
 sp=importlib.util.spec_from_file_location(key,p);m=importlib.util.module_from_spec(sp);sys.modules[key]=m;sp.loader.exec_module(m);return m
BASE=_load('base','engine_default.py');STRONG=_load('strong','policy_ok_strong.py')
MODE='prism';_seen=set();_route='base';LUC={333,677,678};SOL={675,676};OK={116,135}
def _reset(d):
 global _seen,_route
 _seen=set();_route='base';deck=None
 for n,m in [('base',BASE),('strong',STRONG)]:
  try:z=m.agent(d);deck=z if n=='base' else deck
  except Exception:pass
 return deck or BASE.agent(d)
def _observe(d):
 global _route
 cur=d.get('current') if isinstance(d,dict) else None
 if not cur:return
 me=int(cur.get('yourIndex',0));ps=cur.get('players') or []
 if len(ps)<2:return
 op=ps[1-me];ids=set();universal=False
 for z in ('active','bench','discard','lostZone'):
  for c in op.get(z) or []:
   if not c:continue
   if c.get('id') is not None:ids.add(int(c['id']))
   if z in ('active','bench') and any(int(e)==10 for e in (c.get('energies') or [])):universal=True
 _seen.update(ids)
 if _seen&LUC:_route='base';return
 if _seen&OK:_route='strong';return
 turn=int(cur.get('turn') or 0)
 if MODE in {'prism','prism_turn3'} and (_seen&SOL) and universal:_route='strong';return
 if MODE=='prism_turn3' and (_seen&SOL) and turn>=3:_route='strong';return
def agent(d):
 if d.get('select') is None and d.get('current') is None:return _reset(d)
 _observe(d);return (STRONG if _route=='strong' else BASE).agent(d)
