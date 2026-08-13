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
_name="_deck_policy_"+hashlib.sha1(_HERE.encode()).hexdigest()[:12]
_spec=importlib.util.spec_from_file_location(_name,os.path.join(_HERE,"deck_policy.py"))
_base=importlib.util.module_from_spec(_spec);sys.modules[_name]=_base;_spec.loader.exec_module(_base)
_cfg=json.load(open(os.path.join(_HERE,"unified_config.json"),encoding="utf-8"))
_catalog=json.load(open(os.path.join(_HERE,"deck_catalog.json"),encoding="utf-8"))
_prior=json.load(open(os.path.join(_HERE,"replay_prior.json"),encoding="utf-8")) if os.path.exists(os.path.join(_HERE,"replay_prior.json")) else {}
_controller=UnifiedController(_base,_api,_cfg,_catalog,_prior,None)
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


_wall_stats={}
def _flg_crust_patch(observation,chosen):
 try:
  if not isinstance(chosen,list) or not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if obs.current is None or sel is None:return chosen
  st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
  vis=[x for x in list(op.active)+list(op.bench)+list(op.discard) if x is not None]
  if not any(int(x.id) in (344,345) for x in vis):return chosen
  wall=bool(op.active and op.active[0] is not None and int(op.active[0].id)==345)
  board=[x for x in list(mine.active)+list(mine.bench) if x is not None]
  munks=[x for x in board if int(x.id)==112]
  draks=[x for x in board if int(x.id)==120]
  def es(p):return {int(e.id) for e in list(p.energyCards)}
  ready_m=next((x for x in munks if 7 in es(x) and 5 in es(x)),None)
  ready_d=next((x for x in draks if 2 in es(x) and 5 in es(x)),None)
  opts=list(sel.option)
  def src(i):
   try:return _controller._source_card(obs,opts[i])
   except:return None
  def cid(i):
   c=src(i);return int(getattr(c,'id',getattr(opts[i],'cardId',0)) or 0)
  def target(i):
   try:
    o=opts[i];ar=int(getattr(o,'inPlayArea',-1));ix=int(getattr(o,'inPlayIndex',-1));arr=mine.active if ar==4 else mine.bench if ar==5 else []
    return list(arr)[ix] if 0<=ix<len(arr) else None
   except:return None
  # Search result: once Crustle is public, secure one/two Munkidori after a Dreepy line exists.
  if sel.context in (_api.SelectContext.TO_HAND,_api.SelectContext.TO_BENCH):
   eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
   lines=sum(1 for x in board if int(x.id) in (119,120,121))
   if eff in (1121,1152) and lines>=1 and len(munks)<2:
    q=next((i for i in range(len(opts)) if cid(i)==112),None)
    if q is not None and q not in chosen:
     if int(sel.maxCount)==1:return [q]
     out=list(chosen)
     if len(out)<int(sel.maxCount):out.append(q)
     elif out:out[-1]=q
     if len(set(out))==len(out):return out
  # Promotion/switch: ready non-ex attackers first.
  if wall and sel.context in (_api.SelectContext.TO_ACTIVE,_api.SelectContext.SWITCH):
   for want in (ready_m,ready_d):
    if want is None:continue
    q=next((i for i in range(len(opts)) if (src(i) is not None and int(getattr(src(i),'serial',-1))==int(want.serial))),None)
    if q is not None:return [q]
  if sel.context!=_api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1:return chosen
  active=mine.active[0] if mine.active and mine.active[0] is not None else None
  # A ready Munkidori uses Adrena-Brain then Mind Bend. Confusion may buy the second hit.
  if wall and active is not None and int(active.id)==112 and 7 in es(active) and 5 in es(active):
   ab=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.ABILITY and cid(i)==112),None)
   if ab is not None:return [ab]
   at=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.ATTACK and int(getattr(o,'attackId',0) or 0)==141),None)
   if at is not None:return [at]
  if wall and active is not None and int(active.id)==120 and 2 in es(active) and 5 in es(active):
   ab=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.ABILITY and cid(i)==120),None)
   if ab is not None:return [ab]
   at=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.ATTACK and int(getattr(o,'attackId',0) or 0)==152),None)
   if at is not None:return [at]
  if wall:
   # Put a Munkidori into play if naturally held.
   if len(munks)<2 and len(mine.bench)<5:
    q=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.PLAY and cid(i)==112),None)
    if q is not None:return [q]
   # Build D+P on Munkidori; then Fire+Psychic on a Drakloak fallback.
   for m in munks:
    ee=es(m)
    for eid in ((7,5) if 7 not in ee else (5,7)):
     if eid in ee:continue
     q=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.ATTACH and cid(i)==eid and target(i) is not None and int(target(i).serial)==int(m.serial)),None)
     if q is not None:return [q]
   for d in draks:
    ee=es(d)
    for eid in (2,5):
     if eid in ee:continue
     q=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.ATTACH and cid(i)==eid and target(i) is not None and int(target(i).serial)==int(d.serial)),None)
     if q is not None:return [q]
   # Move an already ready wall attacker active if retreat is legal.
   if active is not None and int(active.id) not in (112,120) and (ready_m is not None or ready_d is not None):
    q=next((i for i,o in enumerate(opts) if o.type==_api.OptionType.RETREAT),None)
    if q is not None:return [q]
  return chosen
 except Exception:return chosen
def get_plan_stats():return dict(_wall_stats)
def agent(observation:dict)->list[int]:
 if not observation.get('select'):
  for k in _wall_stats:_wall_stats[k]=0
 chosen=_ultrasafe_drag_patch(observation,_human_order_patch(observation,_controller.agent(observation)))
 return _flg_crust_patch(observation,chosen)
