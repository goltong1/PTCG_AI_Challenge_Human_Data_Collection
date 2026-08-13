from __future__ import annotations
import os,sys,importlib.util,hashlib
HERE=os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:sys.path.insert(0,HERE)
n="_legacy_"+hashlib.sha1(HERE.encode()).hexdigest()[:10]
s=importlib.util.spec_from_file_location(n,os.path.join(HERE,"legacy_main.py"));m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m)
def agent(o:dict)->list[int]:return m.agent(o)
