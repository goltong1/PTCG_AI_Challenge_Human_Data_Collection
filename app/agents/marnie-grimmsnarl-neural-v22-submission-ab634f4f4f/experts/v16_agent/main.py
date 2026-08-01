"""Pure learned Marnie's Grimmsnarl agent.

All gameplay choices are produced by learned policy models and learned state memory.
Only legality validation and deterministic failure fallback are hand written.
"""
from __future__ import annotations
import os
from cg.api import to_observation_class, SelectContext
from . import feature_core as fc
from .model_runtime_v16 import Ensemble
from .policy_memory_runtime_v16 import PolicyMemory
from .semantic_memory_runtime_v16 import SemanticPolicyMemory
from .count_ranker_runtime_v16 import CountRanker

_DECK='deck.csv'
if not os.path.exists(_DECK):_DECK='/kaggle_simulations/agent/deck.csv'
with open(_DECK,'r',encoding='utf-8') as f:MY_DECK=[int(x.strip()) for x in f if x.strip()]
if len(MY_DECK)!=60:raise ValueError('deck must contain 60 cards')
_BASE=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
MODEL=Ensemble(_BASE);POLICY_MEMORY=PolicyMemory(_BASE);SEMANTIC_MEMORY=SemanticPolicyMemory(_BASE);COUNT_RANKER=CountRanker(_BASE)
MEMORY=fc.new_memory()
CURRENT_RX=[0.0]*842

def _reset():
    global MEMORY,CURRENT_RX
    MEMORY=fc.new_memory();CURRENT_RX=[0.0]*842
CURRENT_RX=[0.0]*842

def _valid(result,select):
    out=[];seen=set();n=len(select.option or [])
    for x in result or []:
        if isinstance(x,int) and 0<=x<n and x not in seen:out.append(x);seen.add(x)
    mn=max(0,int(select.minCount or 0));mx=max(mn,min(n,int(select.maxCount or mn)))
    if len(out)<mn:
        for i in range(n):
            if i not in seen:out.append(i);seen.add(i)
            if len(out)>=mn:break
    return out[:mx]

def _pick(scores,obs,arch,phase,descs):
    n=len(scores);sel=obs.select
    if n==0:return []
    ranked=sorted(range(n),key=lambda i:(scores[i],-i),reverse=True)
    mn=max(0,int(sel.minCount or 0));mx=max(mn,min(n,int(sel.maxCount or mn)))
    if mn==mx:return ranked[:mx]
    ctx=int(descs[0][0]) if descs else 0;src=int(descs[0][1]) if descs else 0
    count=COUNT_RANKER.predict(CURRENT_RX,arch,phase,ctx,src,n,mn,mx)
    return ranked[:count]

def _remember_setup(obs,result):
    try:
        if int(obs.select.context)==int(SelectContext.SETUP_ACTIVE_POKEMON) and result:
            o=obs.select.option[result[0]];c=fc.get_card(obs,o.area,o.index,o.playerIndex if o.playerIndex is not None else obs.current.yourIndex);MEMORY['setup_active_id']=int(getattr(c,'id',0) or 0)
    except Exception:pass

def agent(observation:dict)->list[int]:
    global CURRENT_RX
    if observation.get('select') is None:
        _reset();return MY_DECK
    try:
        obs=to_observation_class(observation);arch=fc.update_memory(obs,MEMORY);phase=fc.simple_phase(obs);descs=[fc.extended_action_desc(obs,o) for o in obs.select.option];rx=fc.rich_state_features(obs,MEMORY);CURRENT_RX=rx
        recalled=POLICY_MEMORY.recall(rx,arch,phase,descs)
        if recalled is not None:
            result=_valid(recalled,obs.select)
        else:
            semantic=SEMANTIC_MEMORY.recall_semantics(rx,arch,phase,descs)
            if semantic is not None:
                sems=[SEMANTIC_MEMORY.sem(d) for d in descs]
                unique=all(sum(1 for x in sems if x==tuple(w))==1 for w in semantic)
                scores=None if unique else MODEL.scores(rx,arch,phase,descs)
                mapped=SEMANTIC_MEMORY.map_indices(semantic,descs,scores)
                result=_valid(mapped,obs.select) if mapped is not None else _valid(_pick(MODEL.scores(rx,arch,phase,descs),obs,arch,phase,descs),obs.select)
            else:
                result=_valid(_pick(MODEL.scores(rx,arch,phase,descs),obs,arch,phase,descs),obs.select)
        _remember_setup(obs,result);fc.record_action(MEMORY,descs,result);return result
    except Exception:
        # Legality-only fallback. It is not a gameplay policy and is reached only on runtime failure.
        sel=observation.get('select') or {};opts=sel.get('option') or [];mn=max(0,int(sel.get('minCount',0) or 0));return list(range(min(mn,len(opts))))
