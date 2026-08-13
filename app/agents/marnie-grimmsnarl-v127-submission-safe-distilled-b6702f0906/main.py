from __future__ import annotations
import os,sys,importlib.util,hashlib
_HERE=os.path.dirname(os.path.abspath(globals().get('__file__','/kaggle_simulations/agent/main.py')))
if _HERE not in sys.path:sys.path.insert(0,_HERE)
def _load(name,file):
 s=importlib.util.spec_from_file_location(name,os.path.join(_HERE,file));m=importlib.util.module_from_spec(s);sys.modules[name]=m
 old=os.getcwd()
 try:os.chdir(_HERE);s.loader.exec_module(m)
 finally:os.chdir(old)
 return m
_base=_load('_v127_deck_'+hashlib.sha1(_HERE.encode()).hexdigest()[:10],'deck_policy.py')
_csi=_load('_v127_csi_'+hashlib.sha1((_HERE+'csi').encode()).hexdigest()[:10],'offline_csiql_runtime.py')
_policy=_csi.OfflineCSIQLPolicy(_HERE)
def agent(observation:dict)->list[int]:
 base=_base.agent(observation)
 return _policy.choose(observation,base)
def get_csiql_policy_stats():return _policy.get_stats()
