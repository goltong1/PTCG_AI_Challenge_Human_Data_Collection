from __future__ import annotations
import os,sys,importlib.util,hashlib
HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
def _load():
 n="_portfolio_"+hashlib.sha1(HERE.encode()).hexdigest()[:10]
 s=importlib.util.spec_from_file_location(n,os.path.join(HERE,"deck_policy.py"));m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
M=_load()
def agent(o:dict)->list[int]:return M.agent(o)
