"""Search-distilled public-state terminal closeout guard.

Derived from paired belief-rollout states, but execution uses only public state.
It intervenes only when an attachment/attack gives a conservative, immediate
game-ending KO. Unsupported interactions retain the base policy.
"""
from __future__ import annotations
from cg.api import all_card_data, all_attack

MAIN=0; ATTACH=8; ATTACK=13
ACTIVE=4; COLORLESS=0; RAINBOW=10
TEAL=96; CLEFAIRY=272; CORNERSTONE=117; CRUSTLE=345; DEMOLISH=148
MYRIAD=120; FULL_MOON=371; PRISM=16
CARDS={int(c.cardId):c for c in all_card_data()}; ATTACKS={int(a.attackId):a for a in all_attack()}
STATS={'decisions':0,'attack_finishes':0,'attach_finishes':0,'overrides':0}

def reset():
    for k in STATS: STATS[k]=0

def _cid(c):
    try:return int((c or {}).get('id') or 0)
    except:return 0

def _players(raw):
    cur=raw.get('current') or {}; ps=cur.get('players') or []
    try:me=int(cur.get('yourIndex') or 0)
    except:me=0
    if len(ps)!=2:return cur,{}, {},me
    return cur,ps[me],ps[1-me],me

def _source(raw,o):
    cur,mine,opp,me=_players(raw); ps=cur.get('players') or []
    try:
        pi=me if o.get('playerIndex') is None else int(o.get('playerIndex'))
        area=int(o.get('area',2) if o.get('area') is not None else 2)
        zones={1:(raw.get('select') or {}).get('deck') or [],2:ps[pi].get('hand') or [],3:ps[pi].get('discard') or [],4:ps[pi].get('active') or [],5:ps[pi].get('bench') or [],7:cur.get('stadium') or [],12:cur.get('looking') or []}
        return zones.get(area,[])[int(o.get('index'))]
    except:return None

def _energy_type(card):
    cid=_cid(card)
    if cid==PRISM:return RAINBOW
    d=CARDS.get(cid)
    try:return int(d.energyType)
    except:return 1 if cid==1 else None

def _pool(p, added=None):
    vals=(p or {}).get('energies')
    if vals is None: vals=[RAINBOW if _cid(x)==PRISM else 1 for x in (p or {}).get('energyCards') or []]
    out=[int(x) for x in vals]
    if added is not None:out.append(int(added))
    return out

def _can_pay(p,attack,added=None):
    if not p or attack is None:return False
    pool=_pool(p,added)
    for req0 in attack.energies:
        req=int(req0)
        if req==COLORLESS:
            if not pool:return False
            pool.pop(0);continue
        j=next((i for i,v in enumerate(pool) if int(v) in {req,RAINBOW}),None)
        if j is None:return False
        pool.pop(j)
    return True

def _prizes_for(card):
    d=CARDS.get(_cid(card))
    if d is None:return 1
    if bool(getattr(d,'megaEx',False)):return 3
    if bool(getattr(d,'ex',False)):return 2
    return 1

def _blocked(attacker,target,aid):
    ac=CARDS.get(_cid(attacker)); tid=_cid(target)
    if tid==CORNERSTONE and ac is not None and bool(getattr(ac,'skills',None)) and int(aid)!=DEMOLISH:return True
    if tid==CRUSTLE and ac is not None and (bool(getattr(ac,'ex',False)) or bool(getattr(ac,'megaEx',False))) and int(aid)!=DEMOLISH:return True
    return False

def _damage(raw,attacker,target,aid,added=None):
    a=ATTACKS.get(int(aid or 0))
    if a is None or _blocked(attacker,target,aid):return 0
    cid=_cid(attacker)
    if int(aid)==MYRIAD and cid==TEAL:
        return 30+30*(len(_pool(attacker,added))+len((target or {}).get('energyCards') or (target or {}).get('energies') or []))
    if int(aid)==FULL_MOON and cid==CLEFAIRY:
        cur,mine,opp,_=_players(raw)
        dmg=20*(1+len([x for x in mine.get('bench') or [] if x])+len([x for x in opp.get('bench') or [] if x]))
        td=CARDS.get(_cid(target)); weak=bool(td is not None and int(getattr(td,'weakness',-1) or -1)==5)
        if _cid(target) in {119,120,121} or weak:dmg*=2
        return dmg
    # Only use printed damage as a conservative floor when text cannot reduce it.
    text=(getattr(a,'text','') or '').lower()
    if any(s in text for s in ('instead','less damage','is reduced','does no damage')):return 0
    return int(getattr(a,'damage',0) or 0)

def _winning(raw,attacker,target,aid,added=None):
    cur,mine,opp,_=_players(raw)
    if not attacker or not target:return False
    a=ATTACKS.get(int(aid or 0))
    if a is None or not _can_pay(attacker,a,added):return False
    prizes_left=len(mine.get('prize') or [])
    if prizes_left>_prizes_for(target):return False
    hp=int((target or {}).get('hp') or 0)
    return hp>0 and _damage(raw,attacker,target,aid,added)>=hp

def choose(raw,base,matchup='generic'):
    if not isinstance(raw,dict) or raw.get('current') is None or raw.get('select') is None:
        reset();return base
    STATS['decisions']+=1
    # Search teacher was robustly confirmed only for Dragapult.  Keep every
    # other matchup bit-for-bit on the retained v41 path.
    if matchup != 'dragapult':
        return base
    sel=raw.get('select') or {}
    if int(sel.get('context',-1))!=MAIN:return base
    cur,mine,opp,_=_players(raw)
    active=(mine.get('active') or [None])[0]; target=(opp.get('active') or [None])[0]
    if not active or not target:return base
    opts=sel.get('option') or []
    # 1) Never spend more setup after a conservative public-state game-winning attack is legal.
    finish=[]
    for i,o in enumerate(opts):
        if int(o.get('type',-1))!=ATTACK:continue
        aid=int(o.get('attackId') or 0)
        if _winning(raw,active,target,aid):finish.append((int(_damage(raw,active,target,aid)), -i, i))
    if finish:
        i=max(finish)[2]
        if base!=[i]:STATS['overrides']+=1
        STATS['attack_finishes']+=1
        return [i]
    # 2) If the once-per-turn attachment itself creates such a finish, lock it in.
    if bool(cur.get('energyAttached')):return base
    finish_attach=[]
    data=CARDS.get(_cid(active))
    attacks=list(getattr(data,'attacks',[]) or []) if data else []
    for i,o in enumerate(opts):
        if int(o.get('type',-1))!=ATTACH:continue
        if int(o.get('inPlayArea',-1) if o.get('inPlayArea') is not None else -1)!=ACTIVE or int(o.get('inPlayIndex',-1) if o.get('inPlayIndex') is not None else -1)!=0:continue
        et=_energy_type(_source(raw,o))
        if et is None:continue
        winning_aids=[int(aid) for aid in attacks if _winning(raw,active,target,int(aid),et)]
        if winning_aids:
            bestd=max(_damage(raw,active,target,aid,et) for aid in winning_aids)
            finish_attach.append((bestd,-i,i))
    if finish_attach:
        i=max(finish_attach)[2]
        if base!=[i]:STATS['overrides']+=1
        STATS['attach_finishes']+=1
        return [i]
    return base

def get_stats():return dict(STATS)
