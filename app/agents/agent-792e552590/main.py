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
from causal_history import CausalHistory
from belief_tactical import TacticalBeliefPlanner
from exact_deck_residual import ExactDeckResidual
_tr_name="_transformer_intent_"+hashlib.sha1(_HERE.encode()).hexdigest()[:12]
_tr_spec=importlib.util.spec_from_file_location(_tr_name,os.path.join(_HERE,"transformer_intent_policy.py"))
_tr=importlib.util.module_from_spec(_tr_spec);sys.modules[_tr_name]=_tr;_tr_spec.loader.exec_module(_tr)
TransformerIntentPolicy=_tr.TransformerIntentPolicy
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
_history=CausalHistory()
_belief_tactical=TacticalBeliefPlanner(_api,_controller,_base,_history)
_transformer=TransformerIntentPolicy(os.path.join(_HERE,"transformer_intent_model.npz"))
_exact_residual=ExactDeckResidual(_api,_controller,_base,mode="hold")

try:
 from trajectory_distilled import TrajectoryDistilled
except Exception:
 import importlib.util as _itd
 _sp=os.path.join(_HERE,'trajectory_distilled.py')
 _ss=_itd.spec_from_file_location('trajectory_distilled',_sp);_mm=_itd.module_from_spec(_ss);_ss.loader.exec_module(_mm);TrajectoryDistilled=_mm.TrajectoryDistilled
_distilled=TrajectoryDistilled(_api,_controller)
_history_guard_stats={}
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


# v16 core safety layer -----------------------------------------------------
# These guards are deliberately state based. They only override the legacy
# macro for a literal anti-donk setup failure or an attack known to do no
# useful damage into the Crustle wall. The rejected broad search/energy forcing
# layers remain disabled.
_DRAG_LINE_IDS={119,120,121}

def _opt_type(o,t):
 try:return o.type==t
 except Exception:return False

def _src(obs,opts,i):
 try:return _controller._source_card(obs,opts[i])
 except Exception:return None

def _cid(obs,opts,i):
 c=_src(obs,opts,i)
 try:return int(getattr(c,'id',getattr(opts[i],'cardId',0)) or 0)
 except Exception:return 0

def _own_target(obs,o):
 try:
  me=obs.current.yourIndex; pl=obs.current.players[me]
  ar=int(getattr(o,'inPlayArea',-1)); ix=int(getattr(o,'inPlayIndex',-1))
  arr=pl.active if ar==4 else pl.bench if ar==5 else []
  return list(arr)[ix] if 0<=ix<len(arr) else None
 except Exception:return None

def _board(obs,pl):
 return [x for x in list(pl.active)+list(pl.bench) if x is not None]

def _setup_survival_patch(observation,chosen):
 """Never voluntarily end a turn with a lone fragile Active when a legal
 setup action can create/search a second body or the first Dreepy line.

 Replay 92591450 exposed the old local-score failure: lone Budew + Meowth ex +
 Ultra Ball chose END because both productive options had negative local EV.
 """
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation); sel=obs.select
  if obs.current is None or sel is None or sel.context!=_api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1:return chosen
  opts=list(sel.option); me=obs.current.yourIndex; mine=obs.current.players[me]
  board=_board(obs,mine); line=[p for p in board if int(p.id) in _DRAG_LINE_IDS]
  ci=chosen[0]
  chosen_end=_opt_type(opts[ci],_api.OptionType.END)
  # Immediate anti-donk invariant.  A lone body must not pass while a legal
  # bench/search action exists, especially when no Dreepy line exists yet.
  # Keep this guard extremely narrow: it exists only to prevent a literal
  # one-Pokemon donk when no Dragapult line is in play.  Broader 'opening'
  # forcing regressed mirror play in ablation.
  # Cross-play losses also contained lone-Dreepy boards.  A Dreepy line is not
  # donk protection by itself: if it is the only body, the same one-KO loss is
  # still available.  Keep the override gated by a literal lone board and a
  # legal direct-bench/search play below.
  lone_fragile=(len(board)==1)
  if not (chosen_end and lone_fragile):
   return chosen
  plays=[i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.PLAY)]
  # Direct Dreepy is safest; then deterministic search access; only then a
  # support Basic as a survival body.  Ultra Ball is allowed even when its
  # discard heuristic dislikes the hand because losing on board is worse.
  for want in (119,1086,1121,1152,1071,112,140,235):
   for i in plays:
    if _cid(obs,opts,i)==want:
     # Ultra Ball needs two cards besides itself; legal availability in MAIN
     # already encodes engine legality, so no extra hand-size guess is needed.
     return [i]
  # Evolution/attach/ability can also represent real progress; do not force an
  # inert END if the engine offers one of them and no search/basic is present.
  for t in (_api.OptionType.EVOLVE,_api.OptionType.ATTACH,_api.OptionType.ABILITY):
   for i,o in enumerate(opts):
    if _opt_type(o,t):return [i]
  return chosen
 except Exception:return chosen

def _crustle_zero_damage_guard(observation,chosen):
 """Do not repeat Jet Headbutt into an active Crustle wall for zero damage.
 Prefer Phantom Dive only when available for bench-counter utility; otherwise
 continue setup/retreat or simply end rather than burn the attack action.
 """
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if obs.current is None or sel is None or sel.context!=_api.SelectContext.MAIN:return chosen
  st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
  active_op=op.active[0] if op.active and op.active[0] is not None else None
  active_me=mine.active[0] if mine.active and mine.active[0] is not None else None
  if active_op is None or int(active_op.id)!=345 or active_me is None:return chosen
  opts=list(sel.option);ci=chosen[0]
  if not _opt_type(opts[ci],_api.OptionType.ATTACK):return chosen
  aid=int(getattr(opts[ci],'attackId',0) or 0)
  if int(active_me.id)!=121 or aid not in (153,154):return chosen
  # Phantom Dive may still convert bench counters; prefer it over Jet if legal.
  phantom=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.ATTACK) and int(getattr(o,'attackId',0) or 0)==154),None)
  if phantom is not None and aid==153:return [phantom]
  # If there is no useful Phantom option, spend the turn advancing the wall
  # breaker rather than knowingly taking a zero-damage Jet action.
  for t in (_api.OptionType.ABILITY,_api.OptionType.PLAY,_api.OptionType.ATTACH,_api.OptionType.RETREAT,_api.OptionType.EVOLVE):
   for i,o in enumerate(opts):
    if _opt_type(o,t):return [i]
  end=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.END)),None)
  return [end] if end is not None else chosen
 except Exception:return chosen


# v17 Lucario tactical conversion --------------------------------------------
# 500-loss audit: the legacy Lucario layer misses Riolu card id 333 (3/4 of
# v143's Riolu count) and rarely converts Munkidori damage into a same-turn
# Phantom-Dive bench KO.  This patch is intentionally matchup-local and only
# fires when the KO arithmetic is explicit.
_LUCARIO_IDS={333,673,674,675,676,677,678}
_LUCARIO_EVOLUTION_SEED_IDS={333,677,673}

def _lucario_munk_combo_patch(observation,chosen):
 try:
  if not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if obs.current is None or sel is None:return chosen
  st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
  opp_board=_board(obs,op)
  if not any(int(p.id) in _LUCARIO_IDS for p in opp_board+list(op.discard)):
   return chosen
  opts=list(sel.option); own=_board(obs,mine)
  active=mine.active[0] if mine.active and mine.active[0] is not None else None
  ready_active=False
  if active is not None and int(active.id)==121:
   eids={int(e.id) for e in list(active.energyCards)}
   ready_active=(2 in eids and 5 in eids)
  own_damage=sum(max(0,int(_base.card_table[int(p.id)].hp)-int(p.hp)) for p in own)
  transferable=min(30,own_damage)
  riolu=[p for p in list(op.bench) if p is not None and int(p.id) in _LUCARIO_EVOLUTION_SEED_IDS]
  # Exact same-turn conversion window: Munk can move enough damage that the
  # following 60 Phantom counters can finish a benched Riolu.
  combo_targets=[p for p in riolu if int(p.hp)>0 and int(p.hp)<=60+transferable]
  phantom=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.ATTACK) and int(getattr(o,'attackId',0) or 0)==154),None)
  board_munks=[p for p in own if int(p.id)==112]
  dark_munks=[p for p in board_munks if 7 in {int(e.id) for e in list(p.energyCards)}]

  if sel.context==_api.SelectContext.MAIN and sel.minCount==1 and sel.maxCount==1 and ready_active and phantom is not None and combo_targets:
   # If the Munk is already powered, use Adrena-Brain before attacking.
   if dark_munks:
    for i,o in enumerate(opts):
     if _opt_type(o,_api.OptionType.ABILITY) and _cid(obs,opts,i)==112:
      return [i]
   # Power an existing Munk if a Darkness attachment is legal.  Manual attach
   # does not consume the attack and creates a guaranteed extra-prize line.
   for m in board_munks:
    if 7 in {int(e.id) for e in list(m.energyCards)}:continue
    for i,o in enumerate(opts):
     if not _opt_type(o,_api.OptionType.ATTACH) or _cid(obs,opts,i)!=7:continue
     t=_own_target(obs,o)
     if t is not None and int(t.serial)==int(m.serial):return [i]
   # If Munk is in hand, bench it first; subsequent MAIN decisions can power
   # and activate it before Phantom Dive.
   if not board_munks and len(mine.bench)<int(mine.benchMax):
    for i,o in enumerate(opts):
     if _opt_type(o,_api.OptionType.PLAY) and _cid(obs,opts,i)==112:return [i]

  # Adrena-Brain target selection.  Put the moved damage on the Riolu that is
  # cheapest to bring into the <=60 Phantom finish window; prefer id333 because
  # it is three copies in Lucario v143 and was absent from the legacy matcher.
  eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
  if eff==112 and sel.context in (_api.SelectContext.DAMAGE_COUNTER,_api.SelectContext.DAMAGE_COUNTER_ANY) and ready_active and riolu:
   cand=[]
   for i,o in enumerate(opts):
    try:
     pi=int(getattr(o,'playerIndex',1-me)); ar=int(getattr(o,'area',-1)); ix=int(getattr(o,'index',-1))
     pl=st.players[pi]; arr=pl.active if ar==4 else pl.bench if ar==5 else []
     p=list(arr)[ix] if 0<=ix<len(arr) else None
    except Exception:p=None
    if p is not None and int(p.id) in _LUCARIO_EVOLUTION_SEED_IDS and int(p.hp)<=90:
     # lower resulting HP is better; id333 gets a small tie-break bonus
     cand.append((int(p.hp)-(3 if int(p.id)==333 else 0),i))
   if cand:return [min(cand)[1]]

  # Phantom counters: if a Riolu can be KO'd with the counters still available,
  # cash the prize immediately instead of starting another Lunatone damage plan.
  if eff==121 and sel.context in (_api.SelectContext.DAMAGE_COUNTER,_api.SelectContext.DAMAGE_COUNTER_ANY):
   remain=max(0,int(getattr(sel,'remainDamageCounter',0) or 0))*10
   cand=[]
   for i,o in enumerate(opts):
    try:
     pi=int(getattr(o,'playerIndex',1-me)); ar=int(getattr(o,'area',-1)); ix=int(getattr(o,'index',-1))
     pl=st.players[pi]; arr=pl.active if ar==4 else pl.bench if ar==5 else []
     p=list(arr)[ix] if 0<=ix<len(arr) else None
    except Exception:p=None
    if p is not None and int(p.id) in _LUCARIO_EVOLUTION_SEED_IDS and 0<int(p.hp)<=max(10,remain):
     cand.append((int(p.hp),0 if int(p.id)==333 else 1,i))
   if cand:return [min(cand)[2]]
  return chosen
 except Exception:return chosen


# v18 replay-causal action completion layer ---------------------------------
# 1,000 Lucario losses exposed turns where the validated v17 policy attacked
# while a legal Drakloak ability or a complementary F/P attachment was still
# available.  These resources disappear at turn end.  The gate requires an
# actual Riolu/Mega-Lucario line, not generic Solrock/Lunatone IDs.
try:
 _V18C_CFG=json.load(open(os.path.join(_HERE,'v18_completion_config.json'),encoding='utf-8'))
except Exception:
 _V18C_CFG={}
_V18C_LUCARIO_LINE_IDS={333,677,678}

def _v18c_lucario(obs):
 try:
  st=obs.current;me=st.yourIndex;op=st.players[1-me]
  ids={int(getattr(x,'id',0) or 0) for x in list(op.active)+list(op.bench)+list(op.discard) if x is not None}
  for l in obs.logs:
   if int(getattr(l,'playerIndex',-1))==1-me:
    for k in ('cardId','cardIdAfter','cardIdBefore','cardIdActive','cardIdBench','cardIdTarget'):
     v=int(getattr(l,k,0) or 0)
     if v:ids.add(v)
  return bool(ids&_V18C_LUCARIO_LINE_IDS)
 except Exception:return False

def _v18c_target(obs,o):
 try:
  st=obs.current;mine=st.players[st.yourIndex]
  ar=getattr(o,'inPlayArea',None);ix=getattr(o,'inPlayIndex',None)
  ar=int(ar) if ar is not None else -1;ix=int(ix) if ix is not None else -1
  arr=list(mine.active) if ar==4 else list(mine.bench) if ar==5 else []
  return arr[ix] if 0<=ix<len(arr) else None
 except Exception:return None

def _v18c_energy_ids(p):
 try:return {int(getattr(e,'id',0) or 0) for e in list(p.energyCards or [])}
 except Exception:return set()

def _v18c_patch(observation,chosen):
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if obs.current is None or sel is None or sel.context!=_api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1:return chosen
  if not _v18c_lucario(obs):return chosen
  opts=list(sel.option);ci=chosen[0]
  if not (0<=ci<len(opts) and _opt_type(opts[ci],_api.OptionType.ATTACK)):return chosen
  mine=obs.current.players[obs.current.yourIndex]
  if bool(_V18C_CFG.get('complete_fp_before_attack',False)) and not bool(obs.current.energyAttached):
   cand=[]
   for i,o in enumerate(opts):
    if not _opt_type(o,_api.OptionType.ATTACH):continue
    eid=_cid(obs,opts,i)
    if eid not in (2,5):continue
    p=_v18c_target(obs,o)
    if p is None or int(p.id) not in (119,120,121):continue
    have=_v18c_energy_ids(p);other=5 if eid==2 else 2
    if eid in have or other not in have or {2,5}.issubset(have):continue
    ar=getattr(o,'inPlayArea',None);ar=int(ar) if ar is not None else -1
    if bool(_V18C_CFG.get('bench_only',False)) and ar!=5:continue
    mode=str(_V18C_CFG.get('target_priority','bench'))
    stage={119:1,120:2,121:3}.get(int(p.id),0)
    if mode=='active':loc=2 if ar==4 else 1
    elif mode=='stage':loc=1
    else:loc=2 if ar==5 else 1
    # deterministic highest-priority target; lower tuple wins
    cand.append((-(loc*100+stage*10),i))
   if cand:return [min(cand)[1]]
  if bool(_V18C_CFG.get('drak_ability_before_attack',False)):
   own_prizes=len(mine.prize);opp_prizes=len(obs.current.players[1-obs.current.yourIndex].prize)
   closeout=(own_prizes<=int(_V18C_CFG.get('own_prize_gate',2)) or opp_prizes<=int(_V18C_CFG.get('opp_prize_gate',3)))
   attack_id=int(getattr(opts[ci],'attackId',0) or 0)
   board=[p for p in list(mine.active)+list(mine.bench) if p is not None]
   allow_attack=(attack_id==154 or (attack_id==153 and own_prizes<=1) or (attack_id==152 and not any(int(p.id)==121 for p in board) and opp_prizes<=2))
   if closeout and allow_attack:
    qs=[i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.ABILITY) and _cid(obs,opts,i)==120]
    if qs:
     ready_drag=sum(int(p.id)==121 and {2,5}.issubset(_v18c_energy_ids(p)) for p in board)
     active_op=obs.current.players[1-obs.current.yourIndex].active[0] if obs.current.players[1-obs.current.yourIndex].active else None
     immediate_phantom_ko=bool(attack_id==154 and active_op is not None and 0<int(active_op.hp)<=200)
     if ready_drag<2 and not (len(qs)>=2 and immediate_phantom_ko):return [qs[0]]
  return chosen
 except Exception:return chosen


# v20 causal-history safety layer ------------------------------------------
_H_DRAG_IDS={119,120,121}
_H_RESET_IDS={1080,1213,1227}
_H_ALAKAZAM_IDS={245,743}
_H_DRAW_ENGINE_IDS={140}
_H_SINGLE_PRIZE_SETUP_IDS={119,112,235}

def _h_bump(name):
 _history_guard_stats[name]=int(_history_guard_stats.get(name,0))+1

def _h_eids(p):
 try:return {int(getattr(e,'id',0) or 0) for e in list(p.energyCards or [])}
 except Exception:return set()

def _h_ready_drag(p):
 return bool(p is not None and int(getattr(p,'id',0) or 0)==121 and {2,5}.issubset(_h_eids(p)))

def _h_live_lucky(obs):
 try:
  st=obs.current;op=st.players[1-st.yourIndex]
  active=op.active[0] if op.active and op.active[0] is not None else None
  stadium=int(st.stadium[0].id) if st.stadium else 0
  return bool(active is not None and stadium!=1246 and any(int(getattr(t,'id',0) or 0)==1156 for t in list(active.tools or [])))
 except Exception:return False

def _h_card_option(obs,opts,i):
 try:return _src(obs,opts,i)
 except Exception:return None

def _h_reset_setup_candidate(obs,opts):
 """Choose only resource-evacuation actions learned from reset sequences.

 The gate is intentionally small: evolutions, deterministic setup searches,
 single-Prize Basics, Crispin, and one basic-Energy attachment.  It excludes
 draw supporters, two-Prize bench liabilities, attacks, retreat, stadiums,
 and coin-flip effects.  All candidates are already legal engine options.
 """
 try:
  mine=obs.current.players[obs.current.yourIndex]
  ranked=[]
  for i,o in enumerate(opts):
   typ=o.type;cid=_cid(obs,opts,i)
   if typ==_api.OptionType.EVOLVE and cid in (120,121):
    ranked.append((0 if cid==120 else 1,0,i));continue
   if typ==_api.OptionType.PLAY and cid in (1152,1121,1086):
    # Ultra Ball is only an evacuation action when two cards other than the
    # pending Stamp and the played Ball can pay its discard cost.
    if cid==1121:
     safe_discards=sum(1 for c in list(mine.hand or []) if int(getattr(c,'id',0) or 0) not in (1080,1121))
     if safe_discards<2:continue
    pri={1152:10,1121:11,1086:12}[cid]
    ranked.append((pri,0,i));continue
   if typ==_api.OptionType.PLAY and cid in _H_SINGLE_PRIZE_SETUP_IDS:
    ranked.append(({119:20,112:21,235:22}[cid],0,i));continue
   if typ==_api.OptionType.PLAY and cid==1198:
    ranked.append((30,0,i));continue
   if typ==_api.OptionType.ATTACH and cid in (2,5,7):
    p=_own_target(obs,o)
    if p is None:continue
    pid=int(getattr(p,'id',0) or 0);energies=_h_eids(p)
    if cid==7:
     if pid!=112 or 7 in energies:continue
     # New top-rank sequences establish Adrena-Brain before refreshing the
     # hand; this also keeps Darkness off the Dragapult line.
     quality=400
    else:
     if pid not in _H_DRAG_IDS or cid in energies:continue
     other=5 if cid==2 else 2
     quality=40*int(other in energies)+int(pid)*2+int(p in list(mine.active or []))
    ranked.append((40,-quality,i))
  return min(ranked)[2] if ranked else None
 except Exception:return None

def _h_protect_pending_reset_discard(obs,opts,chosen):
 """Keep an armed Stamp out of an Ultra Ball discard when alternatives exist."""
 try:
  if _history.reset_sequence_phase!=1 or _history.reset_sequence_reset_id!=1080:return chosen
  if int(getattr(getattr(obs.select,'effect',None),'id',0) or 0)!=1121:return chosen
  picked=list(chosen or []);bad=[i for i in picked if _cid(obs,opts,i)==1080]
  if not bad:return chosen
  replacements=[i for i in range(len(opts)) if i not in picked and _cid(obs,opts,i)!=1080]
  if len(replacements)<len(bad):return chosen
  for old,new in zip(bad,replacements):picked[picked.index(old)]=new
  if len(set(picked))==len(picked):
   _h_bump('reset_sequence_protect_reset');return picked
  return chosen
 except Exception:return chosen

def _h_nonreset_fallback(obs,opts):
 """A deterministic productive action after rejecting a second hand reset."""
 def indices(kind):return [i for i,o in enumerate(opts) if _opt_type(o,kind)]
 for i in indices(_api.OptionType.ABILITY):return i
 for i in indices(_api.OptionType.EVOLVE):return i
 for i in indices(_api.OptionType.ATTACH):
  if _cid(obs,opts,i) in (2,5):return i
 for want in (1152,1121,1086,1097,1231,1246):
  for i in indices(_api.OptionType.PLAY):
   if _cid(obs,opts,i)==want:return i
 for i in indices(_api.OptionType.PLAY):
  if _cid(obs,opts,i) not in _H_RESET_IDS and _cid(obs,opts,i) not in (140,1071):return i
 for kind in (_api.OptionType.ATTACK,_api.OptionType.RETREAT,_api.OptionType.END):
  q=indices(kind)
  if q:return q[0]
 return None

def _h_choose_intent(obs,opts,serial,player_index=None):
 if serial is None or int(serial)<0:return None
 for i in range(len(opts)):
  c=_h_card_option(obs,opts,i)
  if c is None or int(getattr(c,'serial',-1))!=int(serial):continue
  # Ownership lives on SelectOption, not on the resolved Pokemon dataclass.
  if player_index is not None and int(getattr(opts[i],'playerIndex',-1))!=int(player_index):continue
  return i
 return None

def _h_safe_retreat_target(mine):
 bench=[p for p in list(mine.bench or []) if p is not None]
 if not bench:return None
 ready=[p for p in bench if _h_ready_drag(p)]
 if ready:return max(ready,key=lambda p:(int(p.hp),-int(p.serial)))
 # Avoid making a two-Prize draw engine the default pivot.  A healthy,
 # non-liability body buys the most time against a known damage mover.
 safe=[p for p in bench if int(p.id) not in (140,1071)] or bench
 return max(safe,key=lambda p:(int(p.hp),len(getattr(p,'energyCards',[]) or []),-int(p.serial)))

def _h_tempo_gust_target(obs):
 try:
  st=obs.current;op=st.players[1-st.yourIndex]
  candidates=[p for p in list(op.bench or []) if p is not None and len(list(p.energyCards or []))==0]
  if not candidates:return None
  def key(p):
   data=_base.card_table.get(int(p.id));cost=int(getattr(data,'retreatCost',0) or 0)
   attacks=list(getattr(data,'attacks',[]) or [])
   min_cost=min((len(_base.attack_table[a].energies) for a in attacks if a in _base.attack_table),default=9)
   return (cost,min_cost,int(p.hp),-int(p.serial))
  return max(candidates,key=key)
 except Exception:return None

def _h_energyless_slowking_trap(obs,mine,op):
 """Return the Fez target for the audited two-turn Slowking trap, if exact.

 Phantom would KO the zero-Energy, three-retreat Slowking and unlock both an
 opposing promotion and Flip the Script.  With Boss already in hand and no
 bench-counter prize, preserving the trap for one turn is the stronger line.
 """
 try:
  active=op.active[0] if op.active and op.active[0] is not None else None
  if active is None or int(active.id)!=163 or len(list(active.energyCards or []))!=0:return None
  data=_base.card_table.get(163)
  if data is None or int(getattr(data,'retreatCost',0) or 0)<3:return None
  if not (0<int(active.hp)<=200) or len(mine.prize)<=2:return None
  if not any(int(getattr(c,'id',0) or 0)==1182 for c in list(mine.hand or [])):return None
  # Do not pass when Phantom can also cash a bench prize immediately.
  if any(0<int(p.hp)<=60 for p in list(op.bench or []) if p is not None):return None
  return next((p for p in list(op.bench or []) if p is not None and int(p.id)==140),None)
 except Exception:return None

def _causal_history_guards(observation,chosen):
 """Narrow dominance and sequencing guards backed by public causal history."""
 try:
  if not isinstance(chosen,list) or not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select
  if obs.current is None or sel is None:return chosen
  opts=list(sel.option);st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
  opponent_active=op.active[0] if op.active and op.active[0] is not None else None
  ci=chosen[0] if len(chosen)==1 and isinstance(chosen[0],int) and 0<=chosen[0]<len(opts) else None
  context=sel.context

  if context==_api.SelectContext.DISCARD:
   return _h_protect_pending_reset_discard(obs,opts,chosen)

  # Complete a previously selected retreat or Boss plan with the intended
  # serial, rather than allowing the next stateless argmax to change targets.
  if context in (_api.SelectContext.SWITCH,_api.SelectContext.TO_ACTIVE):
   # Only the audited Slowking trap uses serial-only matching: opponent
   # switch-option payloads omit playerIndex.  Other legacy gust intents keep
   # v20's stricter behavior to avoid widening this change unintentionally.
   trap_gust=bool(
    _history.pending_trap_boss_serial is not None
    and _history.pending_gust_serial is not None
    and int(_history.pending_trap_boss_serial)==int(_history.pending_gust_serial)
   )
   q=_h_choose_intent(obs,opts,_history.pending_gust_serial,None if trap_gust else 1-me)
   if q is not None:
    _h_bump('tempo_gust_target');return [q]
   retreat_serial=_history.pending_retreat_serial
   # The board may legally change between announcing Retreat and its target
   # prompt (or a replay may follow a different branch).  Re-evaluate the
   # safest destination, while still requiring an armed temporal intent.
   if retreat_serial is not None:
    safe_now=_h_safe_retreat_target(mine)
    if safe_now is not None:retreat_serial=safe_now.serial
   q=_h_choose_intent(obs,opts,retreat_serial,me)
   if q is not None:
    _h_bump('retreat_intent_target');return [q]
   ready=[];budew=[]
   for i in range(len(opts)):
    c=_h_card_option(obs,opts,i)
    if c is None or int(getattr(opts[i],'playerIndex',-1))!=me:continue
    if _h_ready_drag(c):ready.append(i)
    if int(getattr(c,'id',0) or 0)==235:budew.append(i)
   chosen_card=_h_card_option(obs,opts,ci) if ci is not None else None
   # Forced promotion always cashes the ready attack; an ordinary switch only
   # overrides the audited bad branch that explicitly selected Budew.
   if ready and (context==_api.SelectContext.TO_ACTIVE or (chosen_card is not None and int(chosen_card.id)==235)):
    _h_bump('ready_drag_over_budew');return [ready[0]]

  # Poke Pad cannot fetch a Stage 1 with no Dreepy in play.  Prefer a live
  # Dragapult evolution, then Dreepy, then Munkidori, before any other Basic.
  effect=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
  if effect==1152 and context in (_api.SelectContext.TO_HAND,_api.SelectContext.TO_BENCH) and ci is not None and _cid(obs,opts,ci)==120:
   board=[p for p in list(mine.active)+list(mine.bench) if p is not None]
   if not any(int(p.id)==119 for p in board):
    drak_live=any(int(p.id)==120 for p in board)
    order=(121,119,112,235) if drak_live else (119,112,235,121)
    for want in order:
     q=next((i for i in range(len(opts)) if _cid(obs,opts,i)==want),None)
     if q is not None:
      _h_bump('poke_pad_live_target');return [q]
    q=next((i for i in range(len(opts)) if _cid(obs,opts,i)!=120),None)
    if q is not None:
     _h_bump('poke_pad_live_target');return [q]

  # Forced attachment target selection: never put Darkness on the Dragapult
  # line, and never duplicate F/P when another line accepts that colour.
  if context==_api.SelectContext.ATTACH_FROM and ci is not None:
   energy_id=int(getattr(getattr(sel,'contextCard',None),'id',0) or 0)
   chosen_target=_h_card_option(obs,opts,ci)
   chosen_on_active=bool(chosen_target is not None and any(int(getattr(x,'serial',-1))==int(getattr(chosen_target,'serial',-2)) for x in list(mine.active or []) if x is not None))
   if energy_id==7 and chosen_target is not None and int(chosen_target.id) in _H_DRAG_IDS and (not chosen_on_active or 7 in _h_eids(chosen_target)):
    alts=[]
    for i in range(len(opts)):
     p=_h_card_option(obs,opts,i)
     if p is None or int(getattr(p,'playerIndex',me))!=me or int(p.id) in _H_DRAG_IDS:continue
     pri=3 if int(p.id)==112 and 7 not in _h_eids(p) else 1
     alts.append((-pri,i))
    if alts:
     _h_bump('no_darkness_on_drag');return [min(alts)[1]]
   if energy_id in (2,5) and chosen_target is not None and int(chosen_target.id) in _H_DRAG_IDS and energy_id in _h_eids(chosen_target):
    other=5 if energy_id==2 else 2;cand=[]
    for i in range(len(opts)):
     p=_h_card_option(obs,opts,i)
     if p is not None and int(p.id) in _H_DRAG_IDS and energy_id not in _h_eids(p):
      cand.append((-(int(other in _h_eids(p))*10+int(p.id)),i))
    if cand:
     _h_bump('complementary_energy');return [min(cand)[1]]

  if context!=_api.SelectContext.MAIN or sel.minCount!=1 or sel.maxCount!=1 or ci is None:return chosen
  picked=opts[ci]
  phantom=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.ATTACK) and int(getattr(o,'attackId',0) or 0)==154),None)
  # Complete the Boss follow-up remembered from the previous own turn before
  # any stateless setup proposal can abandon the two-turn trap plan.
  if _history.pending_trap_boss_serial is not None:
   target=next((p for p in list(op.bench or []) if p is not None and int(p.serial)==int(_history.pending_trap_boss_serial)),None)
   boss=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.PLAY) and _cid(obs,opts,i)==1182),None)
   if target is not None and boss is not None and phantom is not None:
    _history.set_gust_intent(target.serial)
    _h_bump('slowking_trap_boss_followup');return [boss]
   _history.clear_trap_boss_intent();_h_bump('slowking_trap_expired')
  if phantom is not None:
   trap_target=_h_energyless_slowking_trap(obs,mine,op)
   trap_end=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.END)),None)
   trap_commit=bool(
    _opt_type(picked,_api.OptionType.END) or _opt_type(picked,_api.OptionType.RETREAT)
    or _opt_type(picked,_api.OptionType.ATTACK)
   )
   if trap_target is not None and trap_end is not None and trap_commit:
    _history.set_trap_boss_intent(trap_target.serial)
    _h_bump('preserve_energyless_slowking_trap');return [trap_end]
   if _opt_type(picked,_api.OptionType.END):
    _h_bump('phantom_dominance');return [phantom]
   if _opt_type(picked,_api.OptionType.RETREAT) or (_opt_type(picked,_api.OptionType.ATTACK) and int(getattr(picked,'attackId',0) or 0)==153):
    _h_bump('phantom_dominance');return [phantom]

  stamp=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.PLAY) and _cid(obs,opts,i)==1080),None)
  picked_is_draw=bool(_opt_type(picked,_api.OptionType.ABILITY) and _cid(obs,opts,ci) in _H_DRAW_ENGINE_IDS)
  # A causal within-turn plan replaces the old immediate Stamp override:
  # evacuate deterministic setup resources, commit Stamp, then draw from the
  # refreshed hand.  It arms only in the audited early lone-Dreepy window.
  board_ids=[int(getattr(p,'id',0) or 0) for p in list(mine.active or [])+list(mine.bench or []) if p is not None]
  open_dreepy_window=int(getattr(st,'turn',99) or 99)<=6 and board_ids.count(119)==1 and not ({120,121}&set(board_ids))
  if stamp is not None and ((_history.reset_sequence_phase==1) or (picked_is_draw and open_dreepy_window)):
   if _history.arm_reset_sequence(1080):
    setup=_h_reset_setup_candidate(obs,opts)
    if setup is not None:
     _history.mark_reset_sequence_setup()
     _h_bump('reset_sequence_setup_before_stamp');return [setup]
    _history.mark_reset_sequence_forced_reset()
    _h_bump('stamp_before_draw');return [stamp]
  if stamp is not None and picked_is_draw:
   # Preserve v20's human-loss correction outside the narrow setup window.
   _h_bump('stamp_before_fez');return [stamp]
  elif _history.reset_sequence_phase==1:
   _history.abort_reset_sequence();_h_bump('reset_sequence_abort')
  # Disable a live Lucky Helmet before passing or feeding it a low-value hit.
  jamming=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.PLAY) and _cid(obs,opts,i)==1246),None)
  low_attack=False
  if _opt_type(picked,_api.OptionType.ATTACK):
   aid=int(getattr(picked,'attackId',0) or 0);atk=_base.attack_table.get(aid)
   low_attack=int(getattr(atk,'damage',999) or 0)<=70
  if _h_live_lucky(obs) and jamming is not None and (_opt_type(picked,_api.OptionType.END) or low_attack):
   _h_bump('jamming_before_helmet');return [jamming]
  active=mine.active[0] if mine.active and mine.active[0] is not None else None
  if _h_live_lucky(obs) and active is not None and int(active.id)==119 and _opt_type(picked,_api.OptionType.ATTACK) and int(getattr(picked,'attackId',0) or 0)==150:
   end=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.END)),None)
   if end is not None:
    _h_bump('no_dreepy_ten_into_helmet');return [end]

  # A publicly revealed Alakazam in hand can move the existing counters off a
  # uniquely damaged Active Drakloak.  Retreat that exact exposed body first.
  known_alakazam=_history.known_opponent_has(_H_ALAKAZAM_IDS)
  visible_kadabra=bool(opponent_active is not None and int(opponent_active.id)==742)
  if _opt_type(picked,_api.OptionType.END) and (known_alakazam or visible_kadabra) and active is not None and int(active.id)==120:
   board=[p for p in list(mine.active)+list(mine.bench) if p is not None]
   damaged=[p for p in board if int(p.id)==120 and int(p.hp)<int(getattr(p,'maxHp',_base.card_table[120].hp) or _base.card_table[120].hp)]
   retreat=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.RETREAT)),None)
   if len(damaged)==1 and int(damaged[0].serial)==int(active.serial) and retreat is not None:
    target=_h_safe_retreat_target(mine)
    if target is not None:_history.set_retreat_intent(target.serial)
    _h_bump('known_alakazam_retreat' if known_alakazam else 'visible_kadabra_retreat');return [retreat]

  # Do not bench a two-Prize draw liability into an exactly known Boss when
  # the opponent is already within three Prizes of winning.
  known_boss=_history.known_opponent_has((1182,))
  visible_active_alakazam=bool(opponent_active is not None and int(opponent_active.id)==743)
  if _opt_type(picked,_api.OptionType.PLAY) and _cid(obs,opts,ci)==140 and len(op.prize)<=3 and (known_boss or visible_active_alakazam):
   q=_h_nonreset_fallback(obs,opts)
   if q is not None and q!=ci:
    _h_bump('avoid_known_boss_fez' if known_boss else 'avoid_visible_alakazam_fez');return [q]

  # Prefer an attachment that completes F/P, and reject a Darkness attachment
  # to any Dragapult-line target.
  completions=[]
  for i,o in enumerate(opts):
   if not _opt_type(o,_api.OptionType.ATTACH):continue
   eid=_cid(obs,opts,i);p=_own_target(obs,o)
   if p is None or int(p.id) not in _H_DRAG_IDS or eid not in (2,5):continue
   other=5 if eid==2 else 2
   if eid not in _h_eids(p) and other in _h_eids(p):
    completions.append((-(int(p.id)*10+int(p in mine.active)),i))
  if completions:
   best=min(completions)[1]
   chosen_is_completion=any(i==ci for _,i in completions)
   chosen_target=_own_target(obs,picked) if _opt_type(picked,_api.OptionType.ATTACH) else None
   own_damage=sum(max(0,int(_base.card_table[int(p.id)].hp)-int(p.hp)) for p in list(mine.active)+list(mine.bench) if p is not None)
   preserve_munk_combo=bool(
    phantom is not None and own_damage>0 and _opt_type(picked,_api.OptionType.ATTACH)
    and _cid(obs,opts,ci)==7 and chosen_target is not None and int(chosen_target.id)==112
    and 7 not in _h_eids(chosen_target)
   )
   chosen_low=_opt_type(picked,_api.OptionType.END) or _opt_type(picked,_api.OptionType.RETREAT) or (_opt_type(picked,_api.OptionType.ATTACK) and int(getattr(picked,'attackId',0) or 0)!=154)
   if (not chosen_is_completion) and (not preserve_munk_combo) and (_opt_type(picked,_api.OptionType.ATTACH) or chosen_low):
    _h_bump('complementary_energy');return [best]
   if preserve_munk_combo:_h_bump('preserve_munk_phantom_dark')
  if _opt_type(picked,_api.OptionType.ATTACH) and _cid(obs,opts,ci)==7:
   p=_own_target(obs,picked)
   on_active=bool(p is not None and any(int(getattr(x,'serial',-1))==int(getattr(p,'serial',-2)) for x in list(mine.active or []) if x is not None))
   if p is not None and int(p.id) in _H_DRAG_IDS and (not on_active or 7 in _h_eids(p)):
    q=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.ATTACH) and _cid(obs,opts,i)==7 and (lambda x:x is not None and int(x.id)==112 and 7 not in _h_eids(x))(_own_target(obs,o))),None)
    if q is None:q=_h_nonreset_fallback(obs,opts)
    if q is not None and q!=ci:
     _h_bump('no_darkness_on_drag');return [q]

  # With only a two-Prize support Active plus one developing line, passing lets
  # a single response create a board-out clock.  Boss an energyless Bench body
  # to buy a turn, and retain the exact target serial for the switch prompt.
  if _opt_type(picked,_api.OptionType.END):
   board=[p for p in list(mine.active)+list(mine.bench) if p is not None]
   no_ready=not any(_h_ready_drag(p) for p in board)
   exposed_support=bool(active is not None and (
    (int(active.id) in (112,140,1071) and len(board)<=2) or
    (int(active.id)==119 and len(board)==1)
   ))
   boss=next((i for i,o in enumerate(opts) if _opt_type(o,_api.OptionType.PLAY) and _cid(obs,opts,i)==1182),None)
   target=_h_tempo_gust_target(obs) if exposed_support and no_ready else None
   if boss is not None and target is not None:
    _history.set_gust_intent(target.serial)
    _h_bump('tempo_boss_board_out');return [boss]

  if _opt_type(picked,_api.OptionType.RETREAT):
   ready=next((p for p in list(mine.bench) if _h_ready_drag(p)),None)
   if ready is not None:_history.set_retreat_intent(ready.serial)
  return chosen
 except Exception:return chosen


# Public card-text semantic guard: Full Metal Lab protects the opponent's
# Metal board and Hero's Cape can extend the same two-hit race.  When the
# bundled card text and public board prove that the opponent benefits, replace
# it with Jamming Tower before committing the attack.  No hidden-zone access.
def _text_semantic_stadium_patch(observation,chosen):
 try:
  if not observation.get('select'):return chosen
  obs=_api.to_observation_class(observation);st=obs.current;sel=obs.select
  if st is None or sel is None or sel.context!=_api.SelectContext.MAIN:return chosen
  stadium=int(st.stadium[0].id) if st.stadium else 0
  stadium_data=_base.card_table.get(stadium)
  stadium_text=' '.join(str(getattr(x,'text','') or '') for x in list(getattr(stadium_data,'skills',[]) or [])).lower()
  if 'less damage from attacks' not in stadium_text:return chosen
  me=int(st.yourIndex);op=st.players[1-me]
  metal_visible=any(int(getattr(_base.card_table.get(int(getattr(p,'id',0) or 0)),'energyType',0) or 0)==8 for p in list(op.active or [])+list(op.bench or []) if p is not None)
  if not metal_visible:return chosen
  opts=list(sel.option)
  q=None
  for i,o in enumerate(opts):
   if not _opt_type(o,_api.OptionType.PLAY):continue
   cid=_cid(obs,opts,i);data=_base.card_table.get(cid)
   text=' '.join(str(getattr(x,'text','') or '') for x in list(getattr(data,'skills',[]) or [])).lower()
   if 'pokemon tools' in text and ('have no effect' in text or "don't have any effect" in text):q=i;break
  if q is None:return chosen
  _h_bump('text_replace_full_metal_lab');return [q]
 except Exception:return chosen

def _validated_action(observation,candidate,fallback):
 try:
  sel=observation.get('select')
  if not sel:return candidate
  n=len(sel.get('option') or []);lo=int(sel.get('minCount',0));hi=int(sel.get('maxCount',0))
  def valid(x):return isinstance(x,list) and lo<=len(x)<=hi and len(set(x))==len(x) and all(isinstance(i,int) and 0<=i<n for i in x)
  if valid(candidate):return candidate
  if valid(fallback):
   _h_bump('invalid_guard_fallback');return fallback
  out=list(range(min(lo,n)))
  _h_bump('invalid_controller_fallback');return out
 except Exception:return fallback


_recog_basic_stats={}
def _rbump(k): _recog_basic_stats[k]=int(_recog_basic_stats.get(k,0))+1
def _recognized_mirror_basic_patch(observation,chosen):
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get('select'): return chosen
  obs=_api.to_observation_class(observation); sel=obs.select; st=obs.current
  if st is None or sel is None or int(sel.context)!=int(_api.SelectContext.MAIN) or int(sel.minCount)!=1 or int(sel.maxCount)!=1:return chosen
  opts=list(sel.option or []); old=int(chosen[0]); turn=int(st.turn)
  if turn>3:return chosen
  me=int(st.yourIndex); mine=st.players[me]
  board=[p for p in list(mine.active or [])+list(mine.bench or []) if p is not None]
  if any(int(getattr(p,'id',0) or 0)==121 for p in board):return chosen
  def cid(i):
   try:
    c=_controller._source_card(obs,opts[i]);return int(getattr(c,'id',getattr(opts[i],'cardId',0)) or 0)
   except:return 0
  if not (0<=old<len(opts)) or cid(old)!=1121:return chosen
  q=next((i for i,o in enumerate(opts) if int(getattr(o,'type',-1))==int(_api.OptionType.PLAY) and cid(i)==1086),None)
  if q is None or len(list(mine.bench or []))>=int(getattr(mine,'benchMax',5) or 5):return chosen
  # Hard public-information veto for known non-mirror families.  This prevents
  # generic support-card overlap from causing an early recognizer false positive.
  op=st.players[1-me];pub=set()
  for p in list(op.active or [])+list(op.bench or []):
   if p is not None:
    pub.add(int(getattr(p,'id',0) or 0));pub.update(int(getattr(x,'id',0) or 0) for x in list(getattr(p,'preEvolution',[]) or []))
  pub.update(int(getattr(x,'id',0) or 0) for x in list(op.discard or []) if x is not None)
  blockers={333,675,676,677,678,305,306,66,646,647,648,104,860,169,190,666,57,343,344,345,741,742,743,43,272,330,791,292,293,257,258,906,303,141,689}
  if pub & blockers:
   _rbump('public_veto');return chosen
  name,conf=_controller.recognize(obs)
  _rbump('conflict_seen');_rbump('name_'+str(name));
  if name in ('dusk','dragapult') and float(conf)>=0.1:
   _rbump('poffin_over_ultra');_rbump('gate_'+str(name));return [q]
  _rbump('gate_reject');return chosen
 except Exception:
  _rbump('exceptions');return chosen

_basic_setup_stats={}
def _bsb(k):_basic_setup_stats[k]=int(_basic_setup_stats.get(k,0))+1
def _basic_turn1_dreepy_patch(observation,chosen):
 try:
  if not isinstance(chosen,list) or len(chosen)!=1 or not observation.get("select"):return chosen
  obs=_api.to_observation_class(observation);sel=obs.select;st=obs.current
  if st is None or sel is None or int(sel.context)!=int(_api.SelectContext.MAIN) or int(st.turn)!=1:return chosen
  opts=list(sel.option or []);old=int(chosen[0])
  if not(0<=old<len(opts)) or int(getattr(opts[old],"type",-1))!=int(_api.OptionType.END):return chosen
  meaningful=[i for i,o in enumerate(opts) if int(getattr(o,"type",-1))!=int(_api.OptionType.END)]
  if len(meaningful)!=1:return chosen
  i=meaningful[0];o=opts[i]
  if int(getattr(o,"type",-1))!=int(_api.OptionType.PLAY):return chosen
  try:cid=int(getattr(_controller._source_card(obs,o),"id",0) or 0)
  except:cid=0
  if cid==119:
   _bsb("turn1_end_to_only_dreepy");return [i]
  return chosen
 except Exception:
  _bsb("exceptions");return chosen

def get_plan_stats():
 out=dict(_wall_stats)
 out.update({"recog_basic_"+str(k):int(v) for k,v in _recog_basic_stats.items()})
 out.update({"basic_setup_"+str(k):int(v) for k,v in _basic_setup_stats.items()})
 out.update(_history.stats())
 out.update({str(k):int(v) for k,v in _history_guard_stats.items()})
 out.update(_transformer.get_stats())
 out.update({"exact_"+str(k):int(v) for k,v in _exact_residual.get_stats().items()})
 out.update({"belief_"+str(k):int(v) for k,v in _belief_tactical.get_stats().items()})
 out.update({"distill_"+str(k):int(v) for k,v in _distilled.get_stats().items()})
 return out
def agent(observation:dict)->list[int]:
 try:_history.prepare(_api.to_observation_class(observation))
 except Exception:pass
 if not observation.get('select') and observation.get('current') is None:
  for k in _wall_stats:_wall_stats[k]=0
  _history_guard_stats.clear()
  _belief_tactical.reset()
  _transformer.reset()
  _exact_residual.reset()
  _distilled.reset()
 chosen=_controller.agent(observation)
 chosen=_human_order_patch(observation,chosen)
 chosen=_ultrasafe_drag_patch(observation,chosen)
 chosen=_setup_survival_patch(observation,chosen)
 chosen=_lucario_munk_combo_patch(observation,chosen)
 chosen=_v18c_patch(observation,chosen)
 chosen=_flg_crust_patch(observation,chosen)
 chosen=_crustle_zero_damage_guard(observation,chosen)
 chosen=_text_semantic_stadium_patch(observation,chosen)
 try:
  _tobs=_api.to_observation_class(observation)
  _tname,_tconf=_controller.recognize(_tobs)
  chosen=_transformer.rerank(observation,chosen,_history,_tname,_tconf)
 except Exception:pass
 original_chosen=list(chosen) if isinstance(chosen,list) else chosen
 chosen=_causal_history_guards(observation,chosen)
 chosen=_belief_tactical.patch(observation,chosen)
 chosen=_exact_residual.patch(observation,chosen)
 chosen=_distilled.patch(observation,chosen)
 chosen=_recognized_mirror_basic_patch(observation,chosen)
 chosen=_basic_turn1_dreepy_patch(observation,chosen)
 chosen=_validated_action(observation,chosen,original_chosen)
 if observation.get('select'):
  try:
   obs=_api.to_observation_class(observation)
   _history.record_action(obs,chosen,_src)
   _belief_tactical.record(observation,chosen)
  except Exception:pass
 return chosen
