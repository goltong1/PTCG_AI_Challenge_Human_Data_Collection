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
_base=importlib.util.module_from_spec(_spec);sys.modules[_name]=_base;_spec.loader.exec_module(_base)
_cfg=json.load(open(os.path.join(_HERE,"unified_config.json"),encoding="utf-8"))
_catalog=json.load(open(os.path.join(_HERE,"deck_catalog.json"),encoding="utf-8"))
_prior=json.load(open(os.path.join(_HERE,"replay_prior.json"),encoding="utf-8")) if os.path.exists(os.path.join(_HERE,"replay_prior.json")) else {}
_opp_bank=OpponentPolicyBank(os.path.join(_HERE,'opponent_bank'),_cfg.get('opponent_policy_map'))
_controller=UnifiedController(_base,_api,_cfg,_catalog,_prior,_opp_bank)
def _human_order_patch(observation, chosen):
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or observation.get("select") is None:return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if sel is None or sel.context!=_api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1:return chosen
  opts=list(sel.option);turn=int(obs.current.turn if obs.current else 0);ta=int(obs.current.turnActionCount if obs.current else 0)
  name,_conf=_controller.recognize(obs)
  def src(i):
   try:return _controller._source_card(obs,opts[i])
   except Exception:return None
  def cid(i):
   c=src(i);return int(getattr(c,'id',getattr(opts[i],'cardId',0)) or 0)
  # Human replay: use Drakloak's draw ability before evolving that Drakloak.
  drak_abilities=[i for i,o in enumerate(opts) if o.type==_api.OptionType.ABILITY and cid(i)==120]
  drag_evolves=[i for i,o in enumerate(opts) if o.type==_api.OptionType.EVOLVE and cid(i)==121]
  if drak_abilities and drag_evolves:return [drak_abilities[0]]
  # Establish Drakloak lines early before optional trainer/stadium actions.
  drak_evolves=[i for i,o in enumerate(opts) if o.type==_api.OptionType.EVOLVE and cid(i)==120]
  if turn<=4 and drak_evolves and opts[chosen[0]].type in (_api.OptionType.PLAY,_api.OptionType.ABILITY,_api.OptionType.ATTACH):return [drak_evolves[0]]
  # Against Marnie, once setup actions have been taken, do not postpone a legal Phantom Dive
  # for another optional Stadium ability. This is deliberately matchup- and timing-gated.
  phantom=[i for i,o in enumerate(opts) if o.type==_api.OptionType.ATTACK and int(getattr(o,'attackId',0) or 0)==154]
  if name=='marnie' and ta>=4 and phantom:
   ct=opts[chosen[0]].type
   if ct in (_api.OptionType.ABILITY,_api.OptionType.PLAY,_api.OptionType.END):return [phantom[0]]
  # Avoid spending the opening on Lucky Helmet to Latias when setup/retreat is still available.
  ci=chosen[0];o=opts[ci]
  if turn<=2 and o.type==_api.OptionType.ATTACH and cid(ci)==1156:
   for i,x in enumerate(opts):
    if x.type==_api.OptionType.PLAY:return [i]
   for i,x in enumerate(opts):
    if x.type==_api.OptionType.RETREAT:return [i]
  return chosen
 except Exception:return chosen

def _visible_ids(obs,player):
 try:return {int(getattr(x,'id',0) or 0) for x in list(player.active)+list(player.bench)+list(player.discard) if x is not None}
 except Exception:return set()
def _ultrasafe_drag_patch(observation,chosen):
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if str(sel.context).split('.')[-1].lower()!='main' or sel.minCount!=1 or sel.maxCount!=1:return chosen
  op=obs.current.players[1-obs.current.yourIndex]
  if not (_visible_ids(obs,op)&{646,647,648}):return chosen
  opts=list(sel.option);ci=chosen[0]
  def typ(o,n,e):return o.type==e or str(o.type).split('.')[-1].lower()==n
  def cid(i):
   try:c=_controller._source_card(obs,opts[i]);return int(getattr(c,'id',getattr(opts[i],'cardId',0)) or 0)
   except:return 0
  phantom=[i for i,o in enumerate(opts) if typ(o,'attack',_api.OptionType.ATTACK) and int(getattr(o,'attackId',0) or 0)==154]
  jet=[i for i,o in enumerate(opts) if typ(o,'attack',_api.OptionType.ATTACK) and int(getattr(o,'attackId',0) or 0)==153]
  if phantom and jet:return [phantom[0]]
  drak=[i for i,o in enumerate(opts) if typ(o,'ability',_api.OptionType.ABILITY) and cid(i)==120]
  drag=[i for i,o in enumerate(opts) if typ(o,'evolve',_api.OptionType.EVOLVE) and cid(i)==121]
  if drak and drag:return [drak[0]]
  return chosen
 except Exception:return chosen

def agent(observation:dict)->list[int]:
 return _ultrasafe_drag_patch(observation,_human_order_patch(observation,_controller.agent(observation)))
