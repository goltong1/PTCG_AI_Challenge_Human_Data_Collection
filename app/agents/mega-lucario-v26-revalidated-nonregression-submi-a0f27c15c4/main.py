from __future__ import annotations
import os,sys,importlib.util,hashlib

def _root():
    c=[];f=globals().get('__file__')
    if not f:
        try:f=sys._getframe().f_code.co_filename
        except Exception:f=None
    if f and not str(f).startswith('<'):c.append(os.path.dirname(os.path.abspath(f)))
    c += ['/kaggle_simulations/agent',os.getcwd()]+[p for p in sys.path if isinstance(p,str)]
    seen=set()
    for p in c:
        try:p=os.path.abspath(p)
        except Exception:continue
        if p in seen:continue
        seen.add(p)
        if os.path.isfile(os.path.join(p,'deck_policy.py')) and os.path.isfile(os.path.join(p,'deck.csv')) and os.path.isdir(os.path.join(p,'cg')):return p
    return os.path.abspath(os.getcwd())
HERE=_root()
if HERE not in sys.path:sys.path.insert(0,HERE)
n='lucario_v15_'+hashlib.sha1(HERE.encode()).hexdigest()[:12]
s=importlib.util.spec_from_file_location(n,os.path.join(HERE,'deck_policy.py'));m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m)
def agent(observation:dict)->list[int]:return m.agent(observation)
