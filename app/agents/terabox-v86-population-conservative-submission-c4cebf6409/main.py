from __future__ import annotations
import os,sys,importlib.util,hashlib

def _resolve_root():
    """Locate bundled modules when Kaggle executes main.py as raw source.

    Kaggle's loader compiles ``/kaggle_simulations/agent/main.py`` without
    defining ``__file__``.  The code object's filename still points at the
    extracted submission, so prefer it over the unrelated process cwd.
    """
    code_file=globals().get('__file__') or sys._getframe().f_code.co_filename
    candidates=[]
    if code_file and not str(code_file).startswith('<'):
        candidates.append(os.path.dirname(os.path.abspath(code_file)))
    candidates.extend(('/kaggle_simulations/agent',os.getcwd()))
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate,'engine_base.py')):
            return candidate
    return candidates[0] if candidates else os.getcwd()

R=_resolve_root()
if R not in sys.path:sys.path.insert(0,R)

def _load(name):
    p=os.path.join(R,'engine_'+name+'.py') if name!='weak_v19' else os.path.join(R,'engine_weak_v19.py')
    key='_tera_v18_engine_'+name+'_'+hashlib.sha1((R+name).encode()).hexdigest()[:12]
    sp=importlib.util.spec_from_file_location(key,p);m=importlib.util.module_from_spec(sp);sys.modules[key]=m;sp.loader.exec_module(m);return m

ENG={n:_load(n) for n in ('base','alakazam','crustle','dragapult','dusk','okidogi','direct_v23','drag_moe_v31')}

def _load_policy(name):
    p=os.path.join(R,'policy_'+name+'.py')
    key='_tera_v36_policy_'+name+'_'+hashlib.sha1((R+name+'v36').encode()).hexdigest()[:12]
    sp=importlib.util.spec_from_file_location(key,p);m=importlib.util.module_from_spec(sp);sys.modules[key]=m;sp.loader.exec_module(m);return m
LUCARIO_V17=_load_policy('lucario')
DENSE=_load('dense_v23')
PAIRWISE=_load('pairwise_v24')
TERMINAL=_load('terminal_v25')
EFFICIENCY=_load('efficiency_v27')
REPLAY_VALUE=_load('replay_value_v29')
MATCHUP_V30=_load('matchup_v32')
GUARD_V34=_load('guard_v34')
FLOW_V38=_load('flow_v38')
_WEAK=None
def _weak():
    global _WEAK
    if _WEAK is None:
        _WEAK=_load('weak_v19')
        try:_WEAK.agent({'current':None,'logs':[],'select':None,'step':0})
        except Exception:pass
    return _WEAK
SIG={
 # Matchups are identified with Pokemon-line ids only.  The v28 signatures
 # included generic trainers (notably 1264), which caused Alakazam to be routed
 # through the Crustle policy as soon as that shared trainer was discarded.
 'marnie':{104,646,647,648,860},
 'archaludon':{57,169,190,666},
 'crustle':{58,343,344,345},
 'alakazam':{245,272,741,742,743},
 'spidops':{400,401,414,431,434},
 'grass_ogerpon':{96,10,11,25},
 'cynthia':{341,342,379,380,381,387},
 'dipplin':{89,90,92,93},
 'lopunny':{174,848,849},
 'zoroark':{292,293,303,906},
 'hydrapple':{149,150,709,710,917,918},
 'cinderace':{1030,1031},
}
DRAG={119,120,121,235};DUSK={130,131,132,133};OK={116,135};SOL={675,676};LUC={333,677,678}
_seen=set();_route='base'

def _choose_second(d):
    """Use the supporter/attack-enabled second turn when the coin winner asks.

    SelectContext.IS_FIRST is 41 and OptionType.NO is 2.  The guard deliberately
    returns only a legal NO option, so forced-order and unrelated yes/no prompts
    still fall through to the normal policy.
    """
    select=d.get('select') if isinstance(d,dict) else None
    if not isinstance(select,dict) or int(select.get('context',-1))!=41:return None
    for index,option in enumerate(select.get('option') or []):
        if isinstance(option,dict) and int(option.get('type',-1))==2:return [index]
    return None

def _reset(d):
    global _seen,_route
    _seen=set();_route='base';deck=None
    for n,m in ENG.items():
        try:z=m.agent(d);deck=z if n=='base' else deck
        except Exception:pass
    try:LUCARIO_V17.agent(d)
    except Exception:pass
    if _WEAK is not None:
        try:_WEAK.agent(d)
        except Exception:pass
    try:DENSE.reset()
    except Exception:pass
    try:PAIRWISE.reset()
    except Exception:pass
    try:TERMINAL.reset()
    except Exception:pass
    try:EFFICIENCY.reset()
    except Exception:pass
    try:REPLAY_VALUE.reset()
    except Exception:pass
    try:MATCHUP_V30.reset()
    except Exception:pass
    try:GUARD_V34.reset()
    except Exception:pass
    try:FLOW_V38.reset()
    except Exception:pass
    return deck or ENG['base'].agent(d)

def _dense_matchup():
    if _seen&DRAG:return 'dragapult'
    if _seen&LUC:return 'lucario'
    if _seen&SIG['crustle']:return 'crustle'
    if _seen&SIG['alakazam']:return 'alakazam'
    if _seen&SIG['marnie']:return 'marnie'
    if _seen&SIG['archaludon']:return 'archaludon'
    return 'generic'

def _replay_matchup():
    # The three locally gated opponents have unambiguous evolution-line ids;
    # check them before broader replay families and never let a shared trainer
    # silently change the value-policy context.
    if _seen&DRAG:return 'dragapult'
    if _seen&LUC:return 'lucario'
    for name in ('alakazam','crustle','cynthia','spidops','cinderace','marnie','archaludon',
                 'lopunny','zoroark','hydrapple','dipplin'):
        if _seen&SIG.get(name,set()):return name
    return 'generic'

def _observe(d):
    global _route
    cur=d.get('current') if isinstance(d,dict) else None
    if not cur:return
    me=int(cur.get('yourIndex',0));ps=cur.get('players') or []
    if len(ps)<2:return
    op=ps[1-me];universal=False
    for z in ('active','bench','discard','lostZone'):
        for c in op.get(z) or []:
            if not c:continue
            if c.get('id') is not None:_seen.add(int(c['id']))
            # Public evolution ancestry keeps matchup recognition stable when
            # only the evolved card is currently visible.
            for prior in c.get('preEvolution') or []:
                if prior and prior.get('id') is not None:_seen.add(int(prior['id']))
            if z in ('active','bench') and any(int(e)==10 for e in (c.get('energies') or [])):universal=True
    for c in cur.get('stadium') or []:
        if c and c.get('id') is not None:_seen.add(int(c['id']))
    # Hard promotions/reclassifications are ordered before broad signatures.
    if _seen&DUSK:_route='dusk';return
    if _seen&LUC:_route='lucario_v17';return
    if _seen&OK or ((_seen&SOL) and universal):_route='okidogi';return
    # The previous counterfactual engine was unstable (41/200).  The compact
    # direct policy retained 47/200 over two independent gates and is faster.
    if _seen&DRAG:_route='drag_moe_v31';return
    if _seen&SIG['crustle']:_route='crustle';return
    if _seen&SIG['alakazam']:_route='direct_v23';return
    _route='base'

def agent(d):
    if d.get('select') is None and d.get('current') is None:return _reset(d)
    second=_choose_second(d)
    if second is not None:return second
    _observe(d)
    try:
        if _route=='weak_v19' and int((d.get('select') or {}).get('context',-1)) in {1,2}:
            return _weak().setup_agent(d)
    except Exception:pass
    if _route=='weak_v19':
        try:base=_weak().agent(d)
        except Exception:base=ENG['base'].agent(d)
        try:base=DENSE.choose(d,base,_dense_matchup())
        except Exception:pass
        try:base=PAIRWISE.choose(d,base,_dense_matchup())
        except Exception:pass
        try:base=EFFICIENCY.choose(d,base,_dense_matchup())
        except Exception:pass
        try:base=REPLAY_VALUE.choose(d,base,_replay_matchup())
        except Exception:pass
        try:base=MATCHUP_V30.choose(d,base,_replay_matchup())
        except Exception:pass
        try:base=TERMINAL.choose(d,base,_dense_matchup())
        except Exception:pass
        try:base=GUARD_V34.choose(d,base,_replay_matchup())
        except Exception:pass
        try:return FLOW_V38.choose(d,base,_replay_matchup())
        except Exception:return base
    key=_route if _route in ENG else 'base'
    try:
        base=LUCARIO_V17.agent(d) if _route=='lucario_v17' else ENG[key].agent(d)
    except Exception:
        base=ENG['base'].agent(d)
    try:base=DENSE.choose(d,base,_dense_matchup())
    except Exception:pass
    try:base=PAIRWISE.choose(d,base,_dense_matchup())
    except Exception:pass
    try:base=EFFICIENCY.choose(d,base,_dense_matchup())
    except Exception:pass
    try:base=REPLAY_VALUE.choose(d,base,_replay_matchup())
    except Exception:pass
    try:base=MATCHUP_V30.choose(d,base,_replay_matchup())
    except Exception:pass
    try:base=TERMINAL.choose(d,base,_dense_matchup())
    except Exception:pass
    try:base=GUARD_V34.choose(d,base,_replay_matchup())
    except Exception:pass
    try:return FLOW_V38.choose(d,base,_replay_matchup())
    except Exception:return base


# === Counterfactual regret residual policy (public-state only) ===
_base_agent_before_regret = agent
_rr_name = "_regret_residual_" + hashlib.sha1((R+"terabox").encode()).hexdigest()[:12]
_rr_spec = importlib.util.spec_from_file_location(_rr_name, os.path.join(R, "regret_residual.py"))
_rr_mod = importlib.util.module_from_spec(_rr_spec); sys.modules[_rr_name] = _rr_mod; _rr_spec.loader.exec_module(_rr_mod)
_rr_policy = _rr_mod.ResidualPolicy(R)
def agent(observation:dict)->list[int]:
    base = _base_agent_before_regret(observation)
    return _rr_policy.choose(observation, base)
def get_regret_policy_stats():
    return _rr_policy.get_stats()


# === v48 search-distilled terminal closeout guard ===
_co_name = "_tera_closeout_v50_" + hashlib.sha1((R+"closeout").encode()).hexdigest()[:12]
_co_spec = importlib.util.spec_from_file_location(_co_name, os.path.join(R, "engine_closeout_v50.py"))
_co_mod = importlib.util.module_from_spec(_co_spec); sys.modules[_co_name] = _co_mod; _co_spec.loader.exec_module(_co_mod)
_base_agent_before_closeout = agent
def agent(observation:dict)->list[int]:
    base = _base_agent_before_closeout(observation)
    try:
        return _co_mod.choose(observation, base, _replay_matchup())
    except Exception:
        return base
def get_closeout_stats():
    return _co_mod.get_stats()

# === v86 conservative population specialist ensemble ===
# Only independently gated matchup-specific modules are layered over v50.
def _v86_load(tag, filename):
    name="_tera_v86_"+tag+"_"+hashlib.sha1((R+tag).encode()).hexdigest()[:12]
    sp=importlib.util.spec_from_file_location(name,os.path.join(R,filename))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
_v86_marn=_v86_load("marn72","engine_marn_pivot_v72.py")
_v86_luc=_v86_load("luc80","engine_luc_mega_exact_v80.py")
_v86_base_agent=agent

def agent(observation:dict)->list[int]:
    reset = observation.get('select') is None and observation.get('current') is None
    if reset:
        for m in (_v86_marn,_v86_luc):
            try:m.reset()
            except Exception:pass
    base=_v86_base_agent(observation)
    try: matchup=_replay_matchup()
    except Exception: matchup='generic'
    # Strict matchup gating inside each module prevents cross-matchup regression.
    try: base=_v86_marn.choose(observation,base,matchup)
    except Exception: pass
    try: base=_v86_luc.choose(observation,base,matchup)
    except Exception: pass
    return base

def get_marn_v72_stats(): return _v86_marn.get_stats()
def get_luc_v80_stats(): return _v86_luc.get_stats()
