from __future__ import annotations

"""Shared feature representation for replay-trained offline action models.

The representation is deliberately index/order invariant where possible.  It
scores the semantics of a legal option (source card, target, attack/effect) in
its public board context instead of memorising the option index or a full-state
fingerprint.
"""

from collections import Counter
from typing import Any

DIM = 32768
MASK64 = (1 << 64) - 1

# Public signature groups are only input features.  They never directly choose
# an action and an unknown group remains fully supported.
SIG = {
    "marnie": {104, 646, 647, 648, 860},
    "dragapult": {119, 120, 121, 235},
    "lucario": {333, 673, 675, 676, 677, 678},
    "archaludon": {57, 169, 190, 666},
    "alakazam": {66, 245, 272, 741, 742, 743},
    "crustle": {58, 343, 344, 345, 756},
    "cynthia": {341, 342, 379, 380, 381, 387},
    "spidops": {400, 401, 414, 431, 434},
    "grass": {10, 11, 25, 96, 149, 150, 709, 710},
    "okidogi": {116, 135},
    "terabox": {31, 96, 108, 117, 184, 230, 272, 756},
}
MATCHUP_ID = {"unknown": 0, **{k: i + 1 for i, k in enumerate(SIG)}}

CARD_META: dict[int, dict[str, Any]] = {}
ATTACK_META: dict[int, dict[str, Any]] = {}


def configure_metadata(cards, attacks) -> None:
    global CARD_META, ATTACK_META
    CARD_META = {}
    for c in cards:
        if isinstance(c, dict):
            get = c.get
        else:
            get = lambda k, d=None, _c=c: getattr(_c, k, d)
        cid = int(get("cardId", 0) or 0)
        CARD_META[cid] = {
            "type": int(get("cardType", -1) or 0),
            "retreat": int(get("retreatCost", 0) or 0),
            "hp": int(get("hp", 0) or 0),
            "weakness": int(get("weakness", -1) if get("weakness", None) is not None else -1),
            "resistance": int(get("resistance", -1) if get("resistance", None) is not None else -1),
            "energy": int(get("energyType", -1) if get("energyType", None) is not None else -1),
            "basic": int(bool(get("basic", False))),
            "stage1": int(bool(get("stage1", False))),
            "stage2": int(bool(get("stage2", False))),
            "ex": int(bool(get("ex", False))),
            "mega": int(bool(get("megaEx", False))),
            "tera": int(bool(get("tera", False))),
            "attacks": tuple(int(x) for x in (get("attacks", ()) or ())),
        }
    ATTACK_META = {}
    for a in attacks:
        if isinstance(a, dict):
            get = a.get
        else:
            get = lambda k, d=None, _a=a: getattr(_a, k, d)
        aid = int(get("attackId", 0) or 0)
        ATTACK_META[aid] = {
            "damage": int(get("damage", 0) or 0),
            "energies": tuple(int(x) for x in (get("energies", ()) or ())),
        }


def _mix64(x: int) -> int:
    x &= MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return x & MASK64


def hashed(ns: int, *values: int) -> tuple[int, int]:
    h = _mix64(0x9E3779B97F4A7C15 ^ int(ns))
    for i, value in enumerate(values):
        h = _mix64(h ^ _mix64((int(value) + 0x9E37 * (i + 1)) & MASK64))
    return int(h & (DIM - 1)), (-1 if (h >> 63) else 1)


def _int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _cid(card) -> int:
    if not card:
        return 0
    if isinstance(card, dict):
        return _int(card.get("id", card.get("cardId", 0)), 0)
    return _int(getattr(card, "id", getattr(card, "cardId", 0)), 0)


def _cards(value):
    return [x for x in (value or []) if x]


def _players(raw):
    cur = (raw or {}).get("current") or {}
    ps = cur.get("players") or []
    me = _int(cur.get("yourIndex"), 0)
    if me not in (0, 1):
        me = 0
    return cur, ps, me


def _zone(raw, area: int, owner: int):
    cur, ps, me = _players(raw)
    if owner < 0 or owner >= len(ps):
        return []
    p = ps[owner]
    return {
        1: ((raw or {}).get("select") or {}).get("deck") or [],
        2: p.get("hand") or [],
        3: p.get("discard") or [],
        4: p.get("active") or [],
        5: p.get("bench") or [],
        6: p.get("prize") or [],
        7: cur.get("stadium") or [],
        12: cur.get("looking") or [],
    }.get(area, [])


def option_source(raw, option):
    if option is None:
        return None
    typ = _int(option.get("type"), -1)
    cur, ps, me = _players(raw)
    # Attack and retreat semantics originate from the Active Pokémon.
    if typ in (12, 13):
        if len(ps) > me and _cards(ps[me].get("active")):
            return _cards(ps[me].get("active"))[0]
        return None
    if option.get("index") is None:
        return None
    owner = option.get("playerIndex")
    owner = me if owner is None else _int(owner, me)
    area = option.get("area")
    if area is None:
        # PLAY, ATTACH and EVOLVE options normally point into the hand.
        if typ in (7, 8, 9):
            area = 2
        # ABILITY options without an area sometimes index the in-play source.
        elif typ == 10 and option.get("inPlayArea") is not None:
            area = option.get("inPlayArea")
        else:
            return None
    idx = _int(option.get("index"), -1)
    z = _zone(raw, _int(area, -1), owner)
    return z[idx] if 0 <= idx < len(z) else None


def option_target(raw, option):
    if option is None:
        return None
    cur, ps, me = _players(raw)
    owner = option.get("playerIndex")
    # inPlay targets are normally ours; explicit playerIndex can point at the opponent.
    owner = me if owner is None else _int(owner, me)
    area = _int(option.get("inPlayArea"), -1)
    idx = _int(option.get("inPlayIndex"), -1)
    if area in (4, 5):
        z = _zone(raw, area, owner)
        if 0 <= idx < len(z):
            return z[idx]
    # Some gust/target choices use area/index directly.
    area2 = _int(option.get("area"), -1)
    idx2 = _int(option.get("index"), -1)
    if area2 in (4, 5):
        z = _zone(raw, area2, owner)
        if 0 <= idx2 < len(z):
            return z[idx2]
    return None


def context_source(raw) -> int:
    sel = (raw or {}).get("select") or {}
    return _cid(sel.get("contextCard")) or _cid(sel.get("effect"))


def semantic_option(raw, option) -> tuple[int, int, int, int, int, int, int]:
    if option is None:
        return (-2, 0, 0, 0, 0, -1, -1)
    return (
        _int(option.get("type"), -1),
        _cid(option_source(raw, option)),
        _cid(option_target(raw, option)),
        _int(option.get("attackId"), 0),
        _int(option.get("number"), 0),
        _int(option.get("specialConditionType"), -1),
        _int(option.get("inPlayArea"), -1),
    )


def semantic_signature(raw, option) -> str:
    return "|".join(map(str, semantic_option(raw, option)))


def _hp_bin(card) -> int:
    if not card:
        return 0
    hp = max(0, _int(card.get("hp"), 0))
    mx = max(1, _int(card.get("maxHp"), hp or 1))
    ratio = hp / mx
    return 1 if ratio <= 0.25 else 2 if ratio <= 0.5 else 3 if ratio <= 0.75 else 4


def _energy_count(card) -> int:
    if not card:
        return 0
    return min(6, len(card.get("energyCards") or card.get("energies") or []))


def _phase(turn: int) -> int:
    return 0 if turn <= 2 else 1 if turn <= 5 else 2 if turn <= 9 else 3 if turn <= 13 else 4


def public_matchup(raw) -> int:
    cur, ps, me = _players(raw)
    if len(ps) < 2:
        return 0
    ids = set()
    opp = ps[1 - me]
    for zone in ("active", "bench", "discard"):
        for c in _cards(opp.get(zone)):
            ids.add(_cid(c))
            ids.update(_cid(x) for x in _cards(c.get("preEvolution")))
    # Prefer highly distinctive lines before broad mixed lists.
    order = ("dragapult", "lucario", "crustle", "marnie", "archaludon", "cynthia", "spidops", "grass", "okidogi", "alakazam", "terabox")
    for name in order:
        if ids & SIG[name]:
            return MATCHUP_ID[name]
    return 0


def _add(out, ns, *vals):
    out.append(hashed(ns, *vals))


def state_features(raw) -> list[tuple[int, int]]:
    cur, ps, me = _players(raw)
    sel = (raw or {}).get("select") or {}
    if len(ps) < 2:
        return [hashed(1, 0)]
    mine, opp = ps[me], ps[1 - me]
    turn = _int(cur.get("turn"), 0)
    ta = _int(cur.get("turnActionCount"), 0)
    phase = _phase(turn)
    flags = (
        int(bool(cur.get("energyAttached")))
        | (int(bool(cur.get("supporterPlayed"))) << 1)
        | (int(bool(cur.get("stadiumPlayed"))) << 2)
        | (int(bool(cur.get("retreated"))) << 3)
    )
    my_pr = len(mine.get("prize") or [])
    op_pr = len(opp.get("prize") or [])
    matchup = public_matchup(raw)
    out: list[tuple[int, int]] = []
    _add(out, 1, 1)  # bias
    _add(out, 2, min(turn, 20)); _add(out, 3, phase)
    _add(out, 4, min(ta, 20)); _add(out, 5, flags)
    _add(out, 6, my_pr, op_pr); _add(out, 7, max(-6, min(6, op_pr - my_pr)) + 6)
    _add(out, 8, min(_int(mine.get("deckCount"), 0) // 5, 12), min(_int(opp.get("deckCount"), 0) // 5, 12))
    _add(out, 9, min(_int(mine.get("handCount"), len(mine.get("hand") or [])), 20), min(_int(opp.get("handCount"), 0), 20))
    _add(out, 10, min(len(mine.get("discard") or []), 30) // 3, min(len(opp.get("discard") or []), 30) // 3)
    _add(out, 11, min(len(_cards(mine.get("bench"))), 5), min(len(_cards(opp.get("bench"))), 5))
    _add(out, 12, _int(sel.get("type"), -1) + 3, _int(sel.get("context"), -1) + 3)
    _add(out, 13, _int(sel.get("minCount"), 0), _int(sel.get("maxCount"), 0))
    _add(out, 14, context_source(raw)); _add(out, 15, matchup)
    stadium = _cid((cur.get("stadium") or [None])[0] if cur.get("stadium") else None)
    _add(out, 16, stadium)

    # Zone bags: counts are binned rather than repeated to keep inference bounded.
    zone_specs = (
        (20, mine.get("hand") or []),
        (21, mine.get("discard") or []),
        (22, opp.get("discard") or []),
    )
    for ns, zone in zone_specs:
        cc = Counter(_cid(c) for c in _cards(zone))
        for cid, count in cc.items():
            if cid:
                _add(out, ns, cid, min(count, 4))

    own_ids=[]; opp_ids=[]
    for side, player, base_ns, collect in ((0, mine, 30, own_ids), (1, opp, 40, opp_ids)):
        for area_code, zone_name in ((4, "active"), (5, "bench")):
            for pos, card in enumerate(_cards(player.get(zone_name))):
                cid=_cid(card); collect.append(cid)
                _add(out, base_ns, area_code, cid)
                _add(out, base_ns+1, area_code, cid, _hp_bin(card), _energy_count(card), min(3, len(card.get("preEvolution") or [])))
                if area_code == 4:
                    _add(out, base_ns+2, cid, _hp_bin(card), _energy_count(card))
                ec=Counter(_int(e, -1) for e in (card.get("energies") or []))
                for eid,count in ec.items():
                    _add(out, base_ns+3, cid, eid+2, min(count,4))
                for tool in _cards(card.get("tools")):
                    _add(out, base_ns+4, cid, _cid(tool))
                for pre in _cards(card.get("preEvolution")):
                    _add(out, base_ns+5, cid, _cid(pre))
    # Board co-occurrence summaries.
    for cid in sorted(set(x for x in own_ids if x)):
        _add(out, 50, cid, matchup, phase)
    for cid in sorted(set(x for x in opp_ids if x)):
        _add(out, 51, cid, matchup, phase)
    return out


def action_features(raw, option, include_state: bool = False) -> list[tuple[int, int]]:
    cur, ps, me = _players(raw)
    sel = (raw or {}).get("select") or {}
    out = state_features(raw) if include_state else []
    if len(ps) < 2:
        return out + [hashed(100, -2)]
    mine, opp = ps[me], ps[1 - me]
    turn = _int(cur.get("turn"), 0); phase=_phase(turn)
    matchup=public_matchup(raw)
    flags=(int(bool(cur.get("energyAttached"))) | (int(bool(cur.get("supporterPlayed")))<<1) | (int(bool(cur.get("stadiumPlayed")))<<2) | (int(bool(cur.get("retreated")))<<3))
    ctx=_int(sel.get("context"),-1); stype=_int(sel.get("type"),-1); eff=context_source(raw)
    typ, src, tgt, aid, number, special, target_area = semantic_option(raw, option)
    source=option_source(raw,option) if option is not None else None
    target=option_target(raw,option) if option is not None else None
    own_active=_cid(_cards(mine.get("active"))[0]) if _cards(mine.get("active")) else 0
    opp_active=_cid(_cards(opp.get("active"))[0]) if _cards(opp.get("active")) else 0
    my_pr=len(mine.get("prize") or []);op_pr=len(opp.get("prize") or [])

    _add(out,100,typ+3);_add(out,101,src);_add(out,102,tgt);_add(out,103,aid)
    _add(out,104,number+2,special+2,target_area+2)
    if option is not None:
        _add(out,105,_int(option.get("area"),-1)+2,_int(option.get("inPlayArea"),-1)+2)
        _add(out,106,_int(option.get("playerIndex"),me)-me)
    _add(out,107,ctx+3,stype+3,typ+3)
    _add(out,108,eff,typ+3);_add(out,109,phase,typ+3);_add(out,110,matchup,typ+3)
    _add(out,111,flags,typ+3);_add(out,112,my_pr,op_pr,typ+3)
    _add(out,113,src,ctx+3);_add(out,114,src,phase);_add(out,115,src,matchup)
    _add(out,116,src,own_active);_add(out,117,src,opp_active)
    _add(out,118,src,tgt);_add(out,119,tgt,ctx+3);_add(out,120,tgt,matchup)
    _add(out,121,aid,opp_active);_add(out,122,aid,_hp_bin(_cards(opp.get("active"))[0]) if _cards(opp.get("active")) else 0)
    _add(out,123,aid,my_pr,op_pr)

    sm=CARD_META.get(src) or {}
    if sm:
        _add(out,130,typ+3,sm.get("type",-1)+2)
        _add(out,131,src,sm.get("basic",0),sm.get("stage1",0),sm.get("stage2",0),sm.get("ex",0),sm.get("mega",0),sm.get("tera",0))
        _add(out,132,src,min(sm.get("hp",0)//50,8),min(sm.get("retreat",0),5),sm.get("energy",-1)+2)
    tm=CARD_META.get(tgt) or {}
    if tm:
        _add(out,133,tgt,tm.get("type",-1)+2,tm.get("ex",0),tm.get("mega",0),tm.get("tera",0))
        _add(out,134,tgt,_hp_bin(target),_energy_count(target),min(tm.get("retreat",0),5))
    am=ATTACK_META.get(aid) or {}
    if am:
        energies=am.get("energies") or ()
        _add(out,135,aid,min(am.get("damage",0)//30,15),min(len(energies),5))
        for e in Counter(energies):
            _add(out,136,aid,e+2)

    # Interaction with currently available resources and board pieces.  These
    # crosses give a linear model non-linear conditional behaviour without an
    # exact state hash.
    hand_ids=sorted(set(_cid(c) for c in _cards(mine.get("hand")) if _cid(c)))
    own_field=sorted(set(_cid(c) for c in _cards(mine.get("active"))+_cards(mine.get("bench")) if _cid(c)))
    opp_field=sorted(set(_cid(c) for c in _cards(opp.get("active"))+_cards(opp.get("bench")) if _cid(c)))
    for cid in hand_ids:
        _add(out,140,src,typ+3,cid)
        _add(out,141,aid,typ+3,cid)
    for cid in own_field:
        _add(out,142,src,typ+3,cid)
        _add(out,143,tgt,typ+3,cid)
    for cid in opp_field:
        _add(out,144,src,typ+3,cid)
        _add(out,145,aid,typ+3,cid)
        _add(out,146,tgt,typ+3,cid)
    return out


def candidates_for(raw):
    sel=(raw or {}).get("select") or {}
    n=len(sel.get("option") or [])
    mn=_int(sel.get("minCount"),0);mx=_int(sel.get("maxCount"),0)
    if mx != 1 or n == 0:
        return None
    out=[]
    if mn == 0:
        out.append([])
    out.extend([[i] for i in range(n)])
    return out


def option_for_action(raw, action):
    if not action:
        return None
    opts=((raw or {}).get("select") or {}).get("option") or []
    i=action[0]
    return opts[i] if isinstance(i,int) and 0<=i<len(opts) else None


def feature_indices(features):
    return [x[0] for x in features], [float(x[1]) for x in features]
