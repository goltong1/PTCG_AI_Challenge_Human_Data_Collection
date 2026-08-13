from __future__ import annotations
import os,sys,importlib.util,hashlib
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:sys.path.insert(0,R)

def _load(tag):
    fn=os.path.join(R,f'policy_{tag}.py')
    name='_tera_exact_'+tag+'_'+hashlib.sha1((R+tag).encode()).hexdigest()[:10]
    sp=importlib.util.spec_from_file_location(name,fn);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

NAMES=['generic','archaludon','crustle','dragapult','marnie','alakazam','spidops','grass_ogerpon','dusk','okidogi','cynthia','dipplin','lopunny','lucario']
POL={n:_load(n) for n in NAMES}
SIG={
 'marnie':{104,646,647,648,860,1259},
 'archaludon':{57,169,190,666,1244},
 'crustle':{58,343,344,345,1264},
 'alakazam':{245,272,741,742,743},
 'spidops':{400,401,414,431,434},
 'grass_ogerpon':{96,10,11,25,1127},
 'okidogi':{116,135},
 'cynthia':{341,342,379,380,381,387},
 'dipplin':{89,90,92,93},
 'lopunny':{174,848,849},
 'lucario':{333,677,678},
}
DRAG={119,120,121,235};DUSK={130,131,132,133}
_seen=set();_route='generic'

def _reset():
    global _seen,_route
    _seen=set();_route='generic'
    init={'current':None,'logs':[],'select':None,'step':0}
    for m in POL.values():
        try:m.agent(init)
        except Exception:pass

def _observe(d):
    global _route
    cur=d.get('current') if isinstance(d,dict) else None
    if not cur:return
    me=int(cur.get('yourIndex',0));pls=cur.get('players') or []
    if len(pls)<2:return
    op=pls[1-me]
    for z in ('active','bench','discard','lostZone'):
        for c in op.get(z) or []:
            if c and c.get('id') is not None:_seen.add(int(c['id']))
    for c in cur.get('stadium') or []:
        if c and c.get('id') is not None:_seen.add(int(c['id']))
    if _seen & DUSK:_route='dusk';return
    for n,s in SIG.items():
        if _seen & s:_route=n;return
    if _seen & DRAG:_route='dragapult'

def agent(d):
    if d.get('select') is None and d.get('current') is None:
        _reset();return POL['generic'].agent(d)
    _observe(d)
    return POL.get(_route,POL['generic']).agent(d)
