"""Shared pseudo-language serialization for the Lucario micro decoder.

The runtime never sees hidden opponent cards.  Tokens are built from the same
public history and current private-to-self state already available to the agent.
"""
from __future__ import annotations
from collections import Counter
import hashlib, re
import card_text_semantics as _sem

SPECIAL = {
    '<PAD>':0,'<BOS>':1,'<STATE>':2,'<EVENT>':3,'<DECISION>':4,
    '<ACTION>':5,'<SCORE>':6,'<UNK>':7,
}


def _int(x, d=0):
    try:return int(x if x is not None else d)
    except Exception:return d


def _cid(card):
    if not card:return 0
    return _int(card.get('id',0) if isinstance(card,dict) else getattr(card,'id',0),0)


def _serial(card):
    if not card:return 0
    return _int(card.get('serial',0) if isinstance(card,dict) else getattr(card,'serial',0),0)


def _player(cur,index):
    ps=cur.get('players') or []
    return ps[index] if 0<=index<len(ps) and isinstance(ps[index],dict) else {}


def _hp_bin(card):
    if not card:return 0
    hp=max(0,_int(card.get('hp') if isinstance(card,dict) else getattr(card,'hp',0)))
    mx=max(1,_int(card.get('maxHp') if isinstance(card,dict) else getattr(card,'maxHp',hp),hp or 1))
    return max(0,min(10,int(round(10.0*hp/mx))))


def _energies(card):
    if not card:return []
    if isinstance(card,dict):xs=card.get('energyCards') or card.get('energies') or []
    else:xs=getattr(card,'energyCards',None) or getattr(card,'energies',None) or []
    return [_cid(x) or _int(x) for x in xs]


def _tools(card):
    if not card:return []
    xs=(card.get('tools') or []) if isinstance(card,dict) else (getattr(card,'tools',None) or [])
    return [_cid(x) for x in xs if _cid(x)]


def phase_token(turn,prizes):
    if turn<=0:return 'PHASE=SETUP'
    if prizes<=2:return 'PHASE=CLOSEOUT'
    if turn<=4:return 'PHASE=OPENING'
    if turn>=11:return 'PHASE=LATE'
    return 'PHASE=MID'


def enrich_desc(obs,index,desc):
    out=dict(desc or {});opts=((obs.get('select') or {}).get('option') or []) if isinstance(obs,dict) else []
    if 0<=_int(index,-1)<len(opts):
        o=opts[_int(index)] or {}
        for key in ('area','inPlayArea','inPlayIndex','playerIndex','attackId','cardId'):
            if o.get(key) is not None and key not in out:out[key]=_int(o.get(key))
        # Source index is deliberately omitted for hand cards; hand order is not semantic.
        if _int(o.get('area'),-1) in (4,5,12) and o.get('index') is not None:out['sourceSlot']=_int(o.get('index'))
    return out


def action_signature(desc):
    return f"{_int(desc.get('type'),-1)}:{_int(desc.get('cardId'))}:{_int(desc.get('targetId'))}:{_int(desc.get('attackId'))}:{_int(desc.get('playerIndex'),-1)}:{_int(desc.get('inPlayArea'),-1)}:{_int(desc.get('inPlayIndex'),-1)}:{_int(desc.get('area'),-1)}:{_int(desc.get('sourceSlot'),-1)}"


def action_tokens(desc):
    typ=_int(desc.get('type'),-1);cid=_int(desc.get('cardId'));tid=_int(desc.get('targetId'));aid=_int(desc.get('attackId'));pi=_int(desc.get('playerIndex'),-1)
    out=['<ACTION>',f'ACT_TYPE={typ}']
    if cid:
        out.append(f'ACT_CARD={cid}')
        out.extend(list(_sem.card_semantic_tokens(cid,'ACTCARD',False))[:7])
    if tid:
        out.append(f'ACT_TARGET={tid}')
        out.extend(list(_sem.card_semantic_tokens(tid,'ACTTARGET',False))[:5])
    if aid:
        out.append(f'ACT_ATTACK={aid}')
        # Keep action text less compressed than board text.  Shared lexical tokens
        # let a never-before-seen Shred/Demolish-style attack inherit the bypass rule.
        out.extend(list(_sem.attack_semantic_tokens(aid,'ACTATTACK',True))[:12])
    if pi>=0:out.append(f'ACT_PLAYER={pi}')
    ia=_int(desc.get('inPlayArea'),-1);ii=_int(desc.get('inPlayIndex'),-1);ar=_int(desc.get('area'),-1);ss=_int(desc.get('sourceSlot'),-1)
    if ia>=0:out.append(f'ACT_TARGET_LOC={ia}:{ii}')
    if ar>=0:out.append(f'ACT_SOURCE_AREA={ar}')
    if ss>=0:out.append(f'ACT_SOURCE_SLOT={ss}')
    return out+['<SCORE>']


def _pokemon_tokens(prefix,card,slot):
    cid=_cid(card)
    if not cid:return []
    es=_energies(card);ts=_tools(card)
    out=[f'{prefix}_{slot}_CARD={cid}',f'{prefix}_{slot}_HP={_hp_bin(card)}',f'{prefix}_{slot}_EN={min(6,len(es))}']
    for e,count in Counter(es).items():out.append(f'{prefix}_{slot}_ENERGY={e}x{min(4,count)}')
    for t in ts[:2]:out.append(f'{prefix}_{slot}_TOOL={t}')
    # Active text is retained at higher resolution; Bench cards receive compact
    # rule tags so the context budget remains focused on causal history.
    full=('ACTIVE' in prefix)
    out.extend(list(_sem.card_semantic_tokens(cid,f'{prefix}_{slot}',full))[:(16 if full else 9)])
    return out


def context_tokens(history,obs,family='unknown'):
    cur=obs.get('current') or {};sel=obs.get('select') or {}
    me=history.me_index if getattr(history,'me_index',None) in (0,1) else _int(cur.get('yourIndex'),0);op=1-me
    mine=_player(cur,me);enemy=_player(cur,op)
    turn=_int(cur.get('turn'));my_pr=len(mine.get('prize') or []);op_pr=len(enemy.get('prize') or [])
    toks=['<BOS>',f'FAMILY={family}',f'CTX={_int(sel.get("context"),-1)}',f'TURN_BIN={min(15,turn//2)}',phase_token(turn,my_pr),
          f'MY_PRIZE={min(6,my_pr)}',f'OPP_PRIZE={min(6,op_pr)}',f'PRIZE_DIFF={max(-6,min(6,op_pr-my_pr))}',
          f'MY_HAND={min(12,_int(mine.get("handCount")))}',f'OPP_HAND={min(12,_int(enemy.get("handCount")))}',
          f'MY_DECK={min(12,_int(mine.get("deckCount"))//5)}',f'OPP_DECK={min(12,_int(enemy.get("deckCount"))//5)}',
          f'FLAG_SUPPORTER={int(bool(cur.get("supporterPlayed")))}',f'FLAG_ENERGY={int(bool(cur.get("energyAttached")))}',
          f'FLAG_RETREAT={int(bool(cur.get("retreated")))}',f'FIRST_REL={0 if _int(cur.get("firstPlayer"),-1)==me else 1}']
    # Long-horizon summaries retain information beyond the recent token window.
    try:
        toks += [f'HIST_EVENTS={min(15,len(history.events)//16)}',f'HIST_DECISIONS={min(15,len(history.decisions)//8)}',
                 f'HIST_OWN_ATTACKS={min(10,history.attack_count(me))}',f'HIST_OPP_ATTACKS={min(10,history.attack_count(op))}',
                 f'HIST_OPP_KNOWN={min(8,len(history.known_hand_ids(op)))}']
    except Exception:pass
    # Recent public event language.  Keep event order exactly causal.
    for e in list(getattr(history,'events',[]) or [])[-18:]:
        p=_int(e.get('playerIndex'),-1);side='OWN' if p==me else 'OPP' if p==op else 'GLOBAL';typ=_int(e.get('type'),-1)
        toks += ['<EVENT>',f'EV_SIDE={side}',f'EV_TYPE={typ}']
        cid=_int(e.get('cardId'));target=_int(e.get('cardIdTarget'));aid=_int(e.get('attackId'))
        if cid:toks.append(f'EV_CARD={cid}')
        if target:toks.append(f'EV_TARGET={target}')
        if aid:toks.append(f'EV_ATTACK={aid}')
        fr=_int(e.get('fromArea'),-1);to=_int(e.get('toArea'),-1)
        if fr>=0 or to>=0:toks.append(f'EV_MOVE={fr}>{to}')
        val=_int(e.get('value'))
        if val:toks.append(f'EV_VALUE={max(-8,min(8,val//30))}')
        if e.get('head') is not None:toks.append(f'EV_HEAD={int(bool(e.get("head")))}')
    # Own emitted decisions are separate from public logs and matter for intent continuity.
    for d in list(getattr(history,'decisions',[]) or [])[-5:]:
        toks += ['<DECISION>',f'DEC_CTX={_int(d.get("context"),-1)}']
        for a in (d.get('actions') or [])[:3]:
            toks.append(f'DEC_TYPE={_int(a.get("type"),-1)}')
            if _int(a.get('cardId')):toks.append(f'DEC_CARD={_int(a.get("cardId"))}')
            if _int(a.get('targetId')):toks.append(f'DEC_TARGET={_int(a.get("targetId"))}')
            if _int(a.get('attackId')):toks.append(f'DEC_ATTACK={_int(a.get("attackId"))}')
    toks.append('<STATE>')
    # Exact self hand is legal private information.
    for cid,count in sorted(Counter(_cid(x) for x in (mine.get('hand') or []) if _cid(x)).items()):
        toks.append(f'OWN_HAND_CARD={cid}x{min(4,count)}')
    try:
        for cid,count in sorted(Counter(history.known_hand_ids(op)).items()):toks.append(f'OPP_KNOWN_CARD={cid}x{min(4,count)}')
    except Exception:pass
    for zone,prefix,p in [('active','OWN_ACTIVE',mine),('bench','OWN_BENCH',mine),('active','OPP_ACTIVE',enemy),('bench','OPP_BENCH',enemy)]:
        for j,c in enumerate((p.get(zone) or [])[:8]):toks.extend(_pokemon_tokens(prefix,c,j))
    for c in (cur.get('stadium') or []):
        if _cid(c):toks.append(f'STADIUM={_cid(c)}')
    return enrich_context_tokens(toks)


def enrich_context_tokens(tokens):
    """Add idempotent text semantics to replay rows serialized by older agents."""
    src=list(tokens or [])
    if any(str(t).startswith('SEM_VERSION=') for t in src):return src
    out=[];seen_cards=set();seen_attacks=set()
    for pos,tok in enumerate(src):
        out.append(tok)
        st=str(tok)
        # Preserve the side/zone token itself, then append general text semantics.
        m=re.search(r'(?:CARD|TARGET|STADIUM)=([0-9]+)',st)
        if m:
            cid=int(m.group(1));full=(cid not in seen_cards and ('ACTIVE' in st or 'EV_CARD' in st or 'OWN_HAND' in st))
            seen_cards.add(cid)
            out.extend(list(_sem.card_semantic_tokens(cid,'SEM_CARD',full))[:(15 if full else 7)])
        ma=re.search(r'ATTACK=([0-9]+)',st)
        if ma:
            aid=int(ma.group(1));full=aid not in seen_attacks;seen_attacks.add(aid)
            out.extend(list(_sem.attack_semantic_tokens(aid,'SEM_ATTACK',full))[:(13 if full else 5)])
    if out and out[0]=='<BOS>':out.insert(1,'SEM_VERSION=155_TEXT4')
    else:out.insert(0,'SEM_VERSION=155_TEXT4')
    return out


def token_id(token,vocab_size):
    if token in SPECIAL:return SPECIAL[token]
    if vocab_size<=len(SPECIAL):return SPECIAL['<UNK>']
    raw=hashlib.blake2b(str(token).encode('utf-8'),digest_size=8).digest();num=int.from_bytes(raw,'little')
    return len(SPECIAL)+(num%(vocab_size-len(SPECIAL)))


def encode(tokens,vocab_size,max_len,keep_bos=True):
    ids=[token_id(t,vocab_size) for t in tokens]
    if len(ids)>max_len:
        if keep_bos and ids and ids[0]==SPECIAL['<BOS>']:ids=[ids[0]]+ids[-(max_len-1):]
        else:ids=ids[-max_len:]
    return ids
