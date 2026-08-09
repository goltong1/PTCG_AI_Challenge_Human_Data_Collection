from __future__ import annotations
import os,sys,json,importlib.util,hashlib
def _resolve_submission_root():
 _c=[]
 _f=globals().get("__file__")
 if _f:_c.append(os.path.dirname(os.path.abspath(_f)))
 _c.extend(["/kaggle_simulations/agent",os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else "",os.getcwd()])
 _c.extend([p for p in sys.path if isinstance(p,str)])
 for _p in _c:
  if _p and os.path.isfile(os.path.join(_p,"unified_core.py")) and os.path.isfile(os.path.join(_p,"deck_policy.py")):
   return _p
 raise FileNotFoundError("submission companion files not found beside main.py")
_HERE=_resolve_submission_root()
if _HERE not in sys.path:sys.path.insert(0,_HERE)
from cg import api as _api
_uc_name="_unified_core_"+hashlib.sha1(_HERE.encode()).hexdigest()[:12]
_uc_spec=importlib.util.spec_from_file_location(_uc_name,os.path.join(_HERE,"unified_core.py"))
_uc=importlib.util.module_from_spec(_uc_spec);sys.modules[_uc_name]=_uc;_uc_spec.loader.exec_module(_uc)
UnifiedController=_uc.UnifiedController
_ob_name="_double_agent_bank_"+hashlib.sha1((_HERE+"bank").encode()).hexdigest()[:12]
_ob_spec=importlib.util.spec_from_file_location(_ob_name,os.path.join(_HERE,"double_agent_bank.py"))
_ob=importlib.util.module_from_spec(_ob_spec);sys.modules[_ob_name]=_ob;_ob_spec.loader.exec_module(_ob)
OpponentPolicyBank=_ob.OpponentPolicyBank
_name="_deck_policy_"+hashlib.sha1(_HERE.encode()).hexdigest()[:12]
_spec=importlib.util.spec_from_file_location(_name,os.path.join(_HERE,"deck_policy.py"))
_base=importlib.util.module_from_spec(_spec);sys.modules[_name]=_base
_old_cwd=os.getcwd()
try:
 os.chdir(_HERE);_spec.loader.exec_module(_base)
finally:os.chdir(_old_cwd)
_cfg=json.load(open(os.path.join(_HERE,"unified_config.json"),encoding="utf-8"))
_catalog=json.load(open(os.path.join(_HERE,"deck_catalog.json"),encoding="utf-8"))
_prior=json.load(open(os.path.join(_HERE,"replay_prior.json"),encoding="utf-8")) if os.path.exists(os.path.join(_HERE,"replay_prior.json")) else {}
_opp_bank=OpponentPolicyBank(os.path.join(_HERE,'opponent_bank'),_cfg.get('opponent_policy_map'))
_controller=UnifiedController(_base,_api,_cfg,_catalog,_prior,_opp_bank)
def agent(observation:dict)->list[int]:
 return _controller.agent(observation)
