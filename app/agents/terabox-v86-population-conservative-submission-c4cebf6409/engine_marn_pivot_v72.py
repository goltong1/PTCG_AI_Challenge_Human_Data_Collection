"""v72 Marnie turn-4-to-6 safe-pivot specialist distilled from sequence rollouts.

A 7-particle plan teacher found that, after early setup, leaving a two-energy
Teal Mask Ogerpon ex Active and ending is substantially worse than retreating
into a healthy Mega Kangaskhan ex.  This module performs only that exact pivot
and remembers the promotion choice.
"""
from __future__ import annotations
RETREAT=12;END=14;TEAL=96;KANGA=756;PROMOTE={3,4}
MEM={'promote':False}
STATS={'calls':0,'overrides':0,'retreat':0,'promote':0,'records':[]}
def reset():
    MEM['promote']=False
    for k in STATS:STATS[k]=[] if k=='records' else 0
def get_stats():return {k:(list(v) if isinstance(v,list) else v) for k,v in STATS.items()}
def _cid(x):
    try:return int((x or {}).get('id') or 0)
    except:return 0
def _src(raw,o):
    c=raw.get('current') or {};ps=c.get('players') or [];me=int(c.get('yourIndex') or 0)
    try:
        pi=o.get('playerIndex');pi=me if pi is None else int(pi);ar=int(o.get('area',2));idx=int(o.get('index',0));p=ps[pi]
        z={2:p.get('hand') or [],4:p.get('active') or [],5:p.get('bench') or []}
        return z.get(ar,[])[idx]
    except:return None
def choose(raw,base,matchup):
    STATS['calls']+=1
    sel=raw.get('select') or {};ctx=int(sel.get('context',-1));opts=sel.get('option') or []
    if matchup!='marnie':MEM['promote']=False;return base
    if ctx in PROMOTE and MEM.get('promote'):
        for i,o in enumerate(opts):
            if _cid(_src(raw,o))==KANGA:
                MEM['promote']=False;STATS['overrides']+=base!=[i];STATS['promote']+=1
                return [i]
        MEM['promote']=False;return base
    if ctx!=0 or int(sel.get('minCount') or 0)!=1 or int(sel.get('maxCount') or 0)!=1:return base
    if len(base)!=1 or not (0<=base[0]<len(opts)) or int(opts[base[0]].get('type',-1))!=END:return base
    c=raw.get('current') or {};ps=c.get('players') or [];me=int(c.get('yourIndex') or 0)
    if len(ps)!=2:return base
    mine=ps[me];turn=int(c.get('turn') or 0)
    if turn<4 or turn>6:return base
    active=next((x for x in mine.get('active') or [] if x),None)
    if _cid(active)!=TEAL or len(active.get('energies') or [])<2:return base
    kanga=next((x for x in mine.get('bench') or [] if x and _cid(x)==KANGA and int(x.get('hp') or 0)>=250),None)
    if kanga is None:return base
    for i,o in enumerate(opts):
        if int(o.get('type',-1))==RETREAT:
            MEM['promote']=True;STATS['overrides']+=1;STATS['retreat']+=1
            STATS['records'].append({'turn':turn,'pre':base,'post':[i],'active_hp':active.get('hp'),'active_energy':len(active.get('energies') or []),'kanga_hp':kanga.get('hp')});STATS['records'][:]=STATS['records'][-100:]
            return [i]
    return base
