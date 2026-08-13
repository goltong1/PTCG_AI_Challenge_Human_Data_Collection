from __future__ import annotations
import hashlib,importlib.util,os,sys
HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
def load(tag,file):
 n=tag+'_'+hashlib.sha1((HERE+file).encode()).hexdigest()[:10];s=importlib.util.spec_from_file_location(n,os.path.join(HERE,file));m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
base=load('base','legacy_main.py');core=load('core','deck_policy.py')
MY_DECK=[int(x) for x in open(os.path.join(HERE,'deck.csv')).read().split()]
seen=set();drag=False;SIG={119,120,121}
def reset():
 global drag;seen.clear();drag=False
def observe(o):
 global drag
 c=o.get('current') or {};me=int(c.get('yourIndex',0));ps=c.get('players') or []
 if len(ps)<2:return
 op=ps[1-me]
 for z in (op.get('active') or [])+(op.get('bench') or [])+(op.get('discard') or []):
  if z and z.get('id'):seen.add(int(z['id']))
 for l in o.get('logs') or []:
  if l.get('playerIndex')==1-me:
   for k in ('cardId','cardIdAfter','cardIdBefore','cardIdActive','cardIdBench','cardIdTarget'):
    if l.get(k):seen.add(int(l[k]))
 if seen & SIG:drag=True
def agent(o:dict)->list[int]:
 if o.get('select') is None:
  reset()
  for m in (base,core):
   try:m.agent(o)
   except:pass
  return list(MY_DECK)
 observe(o)
 if drag:
  try:return core.agent(o)
  except:pass
 try:b=base.agent(o)
 except:b=[]
 # Keep core synchronized only until the opponent family is known.
 try:c=core.agent(o)
 except:c=[]
 return b if b else c
