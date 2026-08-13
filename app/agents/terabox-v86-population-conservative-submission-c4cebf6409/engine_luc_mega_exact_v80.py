"""v80 Lucario Mega-only exact prize-conversion guard.

Only public-state, counterfactual-supported conversions are allowed:
1) Full Moon Rondo immediately KOs Active Mega Lucario ex.
2) Boss a benched Mega Lucario ex when a ready Clefairy can exactly KO it and
   the same turn can reach Clefairy.
3) Retreat into a ready Clefairy when Active Mega Lucario is an exact Rondo KO.
No generic Clefairy attachments, bench filling, Riolu chasing, or conservation
heuristics are included because those showed harm under higher-particle gates.
"""
from __future__ import annotations
import engine_matchup_v32 as H
CLEF=272; MEGA=678; BOSS=1182
PLAY=7; RETREAT=12; ATTACK=13; MAIN=0; RONDO=371
MEM={'boss':False,'promote':False}
STATS={'calls':0,'overrides':0,'boss_exact':0,'boss_target':0,'exact_retreat':0,'exact_promote':0,'exact_attack':0,'records':[]}

def reset():
    MEM.update({'boss':False,'promote':False})
    for k in list(STATS): STATS[k]=[] if k=='records' else 0

def get_stats(): return {k:(list(v) if isinstance(v,list) else v) for k,v in STATS.items()}
def _opts(r): return (r.get('select') or {}).get('option') or []
def _typ(o):
    try:return int(o.get('type',-1))
    except:return -1
def _cid(x): return H._cid(x)
def _src(r,o): return H._source(r,o)
def _board(r): _m,b,_h=H._board(r); return b
def _opp(r): a,b=H._opponent_board(r); return (a[0] if a else None),b
def _own(r,o):
    c=r.get('current') or {}; me=int(c.get('yourIndex') or 0)
    try:return int(o.get('playerIndex',me))==me
    except:return False
def _ready(c): return c is not None and _cid(c)==CLEF and H._missing(c)==0
def _lethal(r,c,t):
    if not _ready(c) or t is None or _cid(t)!=MEGA:return False
    try:return H._clefairy_damage(r,t)>=int(t.get('hp') or 0)>0
    except:return False
def _find(r,typ,cid=None,attack=None):
    for i,o in enumerate(_opts(r)):
        if _typ(o)!=typ:continue
        if cid is not None and _cid(_src(r,o))!=cid:continue
        if attack is not None and int(o.get('attackId') or 0)!=attack:continue
        return i
    return None
def _rec(r,kind,base,out):
    c=r.get('current') or {}; ps=c.get('players') or []; me=int(c.get('yourIndex') or 0)
    x={'turn':int(c.get('turn') or 0),'kind':kind,'pre':base,'post':out}
    if len(ps)==2:x['prizes']=[len(ps[me].get('prize') or []),len(ps[1-me].get('prize') or [])]
    STATS['records'].append(x); STATS['records'][:]=STATS['records'][-100:]
def _ret(r,base,i,kind):
    if i is None:return base
    out=[int(i)]
    if out!=base:
        STATS['overrides']+=1;STATS[kind]+=1;_rec(r,kind,base,out)
    return out

def choose(r,base,matchup='generic'):
    STATS['calls']+=1
    if r.get('select') is None and r.get('current') is None:reset();return base
    if matchup!='lucario' or not isinstance(base,list) or len(base)!=1:return base
    sel=r.get('select') or {};ctx=int(sel.get('context',-1));opts=_opts(r)
    # Boss target selector: choose only Mega Lucario.
    if ctx==3 and MEM['boss']:
        MEM['boss']=False
        for i,o in enumerate(opts):
            if _cid(_src(r,o))==MEGA:return _ret(r,base,i,'boss_target')
        return base
    # Retreat promotion selector: choose only an already attack-ready Clefairy.
    if ctx in {3,4} and MEM['promote']:
        MEM['promote']=False; t,_=_opp(r)
        for i,o in enumerate(opts):
            p=_src(r,o)
            if _own(r,o) and _lethal(r,p,t):return _ret(r,base,i,'exact_promote')
        return base
    if ctx!=MAIN:return base
    board=_board(r); a=board[0] if board else None; bench=board[1:]
    t,obench=_opp(r)
    # Immediate exact 3-prize KO.
    if _lethal(r,a,t):
        return _ret(r,base,_find(r,ATTACK,attack=RONDO),'exact_attack')
    ready=next((x for x in board if _ready(x)),None)
    # Exact Boss -> Mega conversion. This is the strongest 15-particle signal.
    if ready is not None:
        mega=next((x for x in obench if _lethal(r,ready,x)),None)
        if mega is not None:
            bi=_find(r,PLAY,cid=BOSS)
            can_reach=(_cid(a)==CLEF) or (_find(r,RETREAT) is not None)
            if bi is not None and can_reach:
                MEM['boss']=True
                return _ret(r,base,bi,'boss_exact')
    # Active Mega exact KO -> move ready Clefairy in.
    bc=next((x for x in bench if _lethal(r,x,t)),None)
    if bc is not None:
        ri=_find(r,RETREAT)
        if ri is not None:
            MEM['promote']=True
            return _ret(r,base,ri,'exact_retreat')
    return base
