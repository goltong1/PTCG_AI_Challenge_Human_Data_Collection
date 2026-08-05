"""Crustle v140: no-Munkidori consistency and Future-matchup policy.
The deck removes the low-frequency Munkidori/Dark Energy package, adds a third
Buddy-Buddy Poffin and a second Spiky Energy, and recognizes Miraidon / Iron
Thorns Future boards separately from the existing matchup branches.
"""
from __future__ import annotations
import os
from collections import Counter, defaultdict
import cg.api as cg_api
from cg.api import (AreaType, CardType, Observation, OptionType, SelectContext,
                    all_card_data, all_attack, to_observation_class)

my_deck = [344, 344, 344, 344, 345, 345, 345, 345, 117, 117, 117, 343, 1086, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1225, 1225, 1186, 1197, 1147, 1147, 1147, 1147, 1122, 1122, 1122, 1086, 1086, 1121, 1121, 1123, 1159, 1137, 1264, 1264, 1264, 11, 11, 11, 11, 18, 18, 18, 14, 20, 1, 6, 14, 1, 6, 117]
assert len(my_deck) == 60

cards = {c.cardId: c for c in all_card_data()}
attacks = {a.attackId: a for a in all_attack()}

GRASS=1; FIGHTING=6; DARK=7; MIST=11; LEGACY=12; ENRICHING=13; PRISM=16; IGNITION=17; GROW_GRASS=18; ROCK_FIGHTING=20
DWEBBLE=344; CRUSTLE=345; CORNERSTONE=117; SHAYMIN=343; BUDEW=235; MUNKIDORI=112; FUECOCO=935; CROCALOR=936; SKELEDIRGE=203
CRISPIN=1198; LILLIE=1227; BOSS=1182; COOK=1212; DAWN=1231
POFFIN=1086; ULTRA=1121; SWITCH=1123; CAGE=1264; VITAL=1096; PRIME=1088
GALETTE=1153; ICECREAM=1147; STRETCHER=1097; CAPE=1159; SCRAPPER=1137; UNFAIR=1080; HAND_TRIMMER=1087; XEROSIC=1197; NIGHT_MINE=1266
ASCENSION=478; SCISSORS=479; DEMOLISH=148; ITCHY_POLLEN=323; TORCHERTO=274

OPPONENT_IDS = {
    'archaludon': {169,190,666,1244},
    'marnie': {646,647,648,104,112,1259},
    'hydrapple': {96,149,150,709,710,917,921,1261},
    'dragapult': {119,120,121,235,1256},
    'mirror': {344,345,117,343,1137},
    'alakazam': {741,742,743,305,66,19,1081,1266},
}
_seen = set()
_last_step = None

def get_card(obs: Observation, area, index, player):
    if index is None or area is None: return None
    ps=obs.current.players[player]
    try:
        if area==AreaType.DECK: return obs.select.deck[index]
        if area==AreaType.HAND: return ps.hand[index]
        if area==AreaType.DISCARD: return ps.discard[index]
        if area==AreaType.ACTIVE: return ps.active[index]
        if area==AreaType.BENCH: return ps.bench[index]
        if area==AreaType.PRIZE: return ps.prize[index]
        if area==AreaType.STADIUM: return obs.current.stadium[index]
        if area==AreaType.LOOKING: return obs.current.looking[index]
    except Exception:
        return None
    return None

def field(ps):
    return [p for p in list(ps.active or [])+list(ps.bench or []) if p is not None]

def maxhp(p):
    base=cards[p.id].hp if p.id in cards else getattr(p,'hp',0)
    cape = 100 if any(t.id==CAPE for t in (getattr(p,'tools',[]) or [])) else 0
    grow = 0
    cd=cards.get(p.id)
    if cd and cd.energyType==GRASS:
        grow = 20*sum(e.id==GROW_GRASS for e in (getattr(p,'energyCards',[]) or []))
    return base + cape + grow

def damage(p): return max(0,maxhp(p)-getattr(p,'hp',cards.get(p.id).hp if cards.get(p.id) else 0))
def nrg(p): return len(getattr(p,'energyCards',[]) or [])
def eids(p): return [e.id for e in (getattr(p,'energyCards',[]) or [])]
def energy_units(p):
    total=0
    evo=bool(cards.get(p.id) and not cards[p.id].basic)
    for eid in eids(p):
        total += 3 if eid==IGNITION and evo else 1
    return total
def provides_grass(p,eid):
    if eid in (GRASS,GROW_GRASS,LEGACY): return True
    if eid==PRISM and cards.get(p.id) and cards[p.id].basic: return True
    return False
def provides_fighting(p,eid):
    if eid in (FIGHTING,ROCK_FIGHTING,LEGACY): return True
    if eid==PRISM and cards.get(p.id) and cards[p.id].basic: return True
    return False

def provides_dark(p,eid):
    if eid in (DARK,LEGACY): return True
    if eid==PRISM and cards.get(p.id) and cards[p.id].basic: return True
    return False
def effect_guard(p):
    if p.id==SKELEDIRGE: return True
    ids=eids(p)
    if MIST in ids: return True
    if ROCK_FIGHTING in ids and cards.get(p.id) and cards[p.id].energyType==FIGHTING: return True
    return False
def guard_count(p):
    return sum(eid==MIST or (eid==ROCK_FIGHTING and cards.get(p.id) and cards[p.id].energyType==FIGHTING) for eid in eids(p))
def prizes(p):
    c=cards.get(p.id)
    if not c: return 1
    return 2 if c.ex else 1

def archetype(obs):
    st=obs.current; me=st.yourIndex
    for p in field(st.players[1-me]): _seen.add(p.id)
    for c in (st.players[1-me].discard or []): _seen.add(c.id)
    best=None; score=0
    for name,ids in OPPONENT_IDS.items():
        s=len(_seen & ids)
        if s>score: best,score=name,s
    return best

def has_ability(p): return bool(cards.get(p.id) and cards[p.id].skills)
def is_ex(p): return bool(cards.get(p.id) and cards[p.id].ex)
def wall_blocks(wall, attacker):
    if wall is None or attacker is None: return False
    if wall.id==CRUSTLE: return is_ex(attacker)
    if wall.id==CORNERSTONE: return has_ability(attacker)
    return False

def ready(p):
    if p is None or not hasattr(p,'energyCards'): return False
    ids=eids(p); units=energy_units(p)
    if p.id==CRUSTLE: return any(provides_grass(p,eid) for eid in ids) and units>=3
    if p.id==CORNERSTONE: return any(provides_fighting(p,eid) for eid in ids) and units>=3
    if p.id==DWEBBLE: return units>=1
    if p.id==SKELEDIRGE: return any(eid==2 or eid==LEGACY or (eid==PRISM and cards.get(p.id) and cards[p.id].basic) for eid in ids) and units>=2
    return False

def attack_damage(p):
    if p is None: return 0
    if p.id==CRUSTLE and ready(p): return 120
    if p.id==CORNERSTONE and ready(p): return 140
    if p.id==SKELEDIRGE and ready(p): return 180
    return 0

def wall_fit(p, arch):
    if p is None: return -999
    if arch=='dragapult':
        return 100 if p.id==CRUSTLE else 62 if p.id==CORNERSTONE else 0
    if arch=='marnie':
        return 115 if p.id==CRUSTLE else 50 if p.id==CORNERSTONE else 0
    if arch=='mirror':
        return 135 if p.id==CRUSTLE else 85 if p.id==CORNERSTONE else 0
    if arch=='alakazam':
        base=230 if p.id==CRUSTLE else 150 if p.id==CORNERSTONE else 0
        return base + (130 if effect_guard(p) else 0)
    if arch in ('archaludon','hydrapple'):
        return 90 if p.id==CORNERSTONE else 65 if p.id==CRUSTLE else 0
    return 55 if p.id==CRUSTLE else 50 if p.id==CORNERSTONE else 0

def target_value(p, reach, arch):
    c=cards.get(p.id); v=prizes(p)*12000 + nrg(p)*1300 + p.hp*4
    if c and c.skills: v+=2500
    if c and (c.stage1 or c.stage2): v+=1300
    if p.hp<=reach:
        v+=45000+prizes(p)*22000
    # wall bypassers/non-ex engines must die first
    if arch=='dragapult' and p.id in (112,31,120,119,235):
        v += {112:70000,31:62000,120:52000,119:40000,235:38000}.get(p.id,30000)
    if arch=='archaludon' and p.id in (169,666,57,190):
        v += {666:115000,169:105000,57:32000,190:26000}.get(p.id,28000)
        if p.id in (169,666) and p.hp<=reach: v += 90000
    if arch=='marnie' and p.id in (104,112,646,647):
        v += {104:130000,112:122000,647:62000,646:50000}.get(p.id,26000)
        if p.id in (104,112) and p.hp<=reach: v += 100000
    if arch=='hydrapple' and p.id in (917,709,710,149,921): v+=22000
    if arch=='alakazam':
        if p.id==245: v += 155000
        elif p.id==743: v += 120000
        elif p.id==742: v += 62000
        elif p.id==741: v += 45000
        elif p.id==66: v += 36000
        if p.hp<=reach: v += 50000
    if arch=='mirror':
        if p.id==CORNERSTONE: v += 52000  # two prizes in the same two-hit clock
        elif p.id==CRUSTLE: v += 18000
        elif p.id==DWEBBLE: v += 26000
        if any(t.id==CAPE for t in (getattr(p,'tools',[]) or [])): v += 24000
    return v

def desired_wall(arch): return CRUSTLE if arch in ('alakazam','dragapult','marnie','mirror') else CORNERSTONE if arch else CRUSTLE

def missing_cost(p):
    if p is None or not hasattr(p,'energyCards'): return 3
    ids=eids(p); units=energy_units(p)
    if p.id==CRUSTLE:
        return (0 if any(provides_grass(p,eid) for eid in ids) else 1) + max(0,3-units-(0 if any(provides_grass(p,eid) for eid in ids) else 1))
    if p.id==CORNERSTONE:
        return (0 if any(provides_fighting(p,eid) for eid in ids) else 1) + max(0,3-units-(0 if any(provides_fighting(p,eid) for eid in ids) else 1))
    if p.id==DWEBBLE: return max(0,1-units)
    if p.id==SKELEDIRGE:
        fire=any(eid==2 or eid==LEGACY for eid in ids)
        return (0 if fire else 1)+max(0,2-units-(0 if fire else 1))
    return 99

def discard_score(cid, own, arch, hand_counts):
    # Higher = safer to discard.
    count=hand_counts[cid]
    if cid==BOSS: return 9500 if count>=2 else 2500
    if cid==CAGE: return 10000 if arch=='mirror' else (9500 if arch not in ('dragapult','marnie') or count>=2 else 500)
    if cid==COOK: return 9000 if count>=2 else 4500
    if cid==LILLIE: return 6500 if count>=2 else (-5000 if not any(ready(p) for p in own if p.id in (CRUSTLE,CORNERSTONE)) else 600)
    if cid==DAWN: return 8500 if count>=2 else 3000
    if cid==SWITCH: return 7500 if count>=2 else 2200
    if cid in (GALETTE,ICECREAM): return 7000 if count>=2 else 3000
    if cid==POFFIN: return 10500 if sum(p.id in (DWEBBLE,CRUSTLE) for p in own)>=1 else 1800
    if cid==ULTRA: return 8000 if count>=2 else 2000
    if cid==STRETCHER: return 7000 if count>=2 else 1200
    if cid==CAPE: return -5000
    if cid==SCRAPPER: return 8500 if count>=2 else 1500
    if cid==SHAYMIN: return 10000 if any(p.id==SHAYMIN for p in own) or arch!='marnie' else 6500
    if cid==CRUSTLE: return 9000 if sum(p.id==CRUSTLE for p in own)>=2 else -5000
    if cid==DWEBBLE: return 8000 if sum(p.id in (DWEBBLE,CRUSTLE) for p in own)>=2 else -5000
    if cid==CORNERSTONE: return 12000 if arch=='mirror' else (8500 if sum(p.id==CORNERSTONE for p in own)>=2 else -5000)
    if cid==MUNKIDORI: return 9000 if arch!='marnie' or any(p.id==MUNKIDORI for p in own) else -7000
    if cid==FUECOCO: return (9000 if any(p.id in (FUECOCO,CROCALOR,SKELEDIRGE) for p in own) else -9000) if arch=='alakazam' else 14000
    if cid==CROCALOR: return (7500 if any(p.id in (CROCALOR,SKELEDIRGE) for p in own) else -4500) if arch=='alakazam' else 14000
    if cid==SKELEDIRGE: return (8500 if any(p.id==SKELEDIRGE for p in own) else -9000) if arch=='alakazam' else 15000
    if cid==1079: return (8000 if count>=2 else (-7000 if any(p.id==FUECOCO for p in own) else 1000)) if arch=='alakazam' else 15000
    if cid==XEROSIC: return 8000 if count>=2 else (-6500 if arch=='alakazam' else 2000)
    if cid in (GRASS,FIGHTING,DARK): return 5000 if count>=2 else (-5500 if arch=='marnie' and cid==DARK else -2500)
    if cid in (MIST,GROW_GRASS,ROCK_FIGHTING,ENRICHING,PRISM,IGNITION,LEGACY):
        if arch=='alakazam' and cid in (MIST,ROCK_FIGHTING): return -8000
        return 4500 if count>=2 else -1800
    return 4000

def base_scores(obs: Observation):
    st=obs.current; sel=obs.select; me=st.yourIndex; opi=1-me
    mine=st.players[me]; opp=st.players[opi]
    own=field(mine); enemy=field(opp)
    active=mine.active[0] if mine.active and mine.active[0] is not None else None
    oa=opp.active[0] if opp.active and opp.active[0] is not None else None
    arch=archetype(obs)
    effect=sel.effect.id if sel.effect is not None else 0
    stadium=st.stadium[0].id if st.stadium and st.stadium[0] is not None else 0
    hand_counts=Counter(c.id for c in (mine.hand or []))
    own_counts=Counter(p.id for p in own)
    reach=attack_damage(active)
    if active and active.id==SKELEDIRGE and ready(active): reach=60+20*(len(mine.bench or [])+len(opp.bench or []))
    attack_available=any(o.type==OptionType.ATTACK for o in sel.option)
    desired=desired_wall(arch)
    core_ids=(CRUSTLE,CORNERSTONE)
    core_ready=any(ready(p) for p in own if p.id in core_ids)
    active_progress=(energy_units(active) if active and (active.id in (DWEBBLE,CRUSTLE,CORNERSTONE) or (arch=='alakazam' and active.id==SKELEDIRGE)) else 0)
    dark_access=bool(hand_counts.get(DARK,0) or hand_counts.get(PRISM,0) or hand_counts.get(CRISPIN,0))
    pending_ice = bool(arch=='marnie' and active and nrg(active)>=3 and hand_counts.get(ICECREAM,0)>0 and damage(active)>=30)
    pending_cook = bool(arch=='marnie' and active and not st.supporterPlayed and hand_counts.get(COOK,0)>0 and damage(active)>=40)
    pending_cage = bool(True and arch=='marnie' and stadium!=CAGE and hand_counts.get(CAGE,0)>0 and any(p.id in (CRUSTLE,CORNERSTONE,SHAYMIN,MUNKIDORI) for p in (mine.bench or []) if p is not None))
    pending_munk = bool(True and arch=='marnie' and any(p.id==MUNKIDORI and any(provides_dark(p,eid) for eid in eids(p)) for p in own) and any(damage(p)>=10 for p in own))
    pending_vital = bool(arch in ('marnie','alakazam') and active and hand_counts.get(VITAL,0)>0 and damage(active)>=50)
    pending_guard = bool(arch=='alakazam' and active and not st.energyAttached and guard_count(active)<2 and any((c.id==MIST) or (c.id==ROCK_FIGHTING and active.id==CORNERSTONE) for c in (mine.hand or [])))
    pending_backup = bool(arch!='dragapult' and len(mine.bench or [])==0 and any(c.id in (DWEBBLE,CORNERSTONE,MUNKIDORI,SHAYMIN) for c in (mine.hand or [])))
    pending_utility = pending_ice or pending_cook or pending_cage or pending_munk or pending_vital or pending_guard or pending_backup
    setup_basic_ids=[]
    if sel.context in (SelectContext.SETUP_ACTIVE_POKEMON,SelectContext.SETUP_BENCH_POKEMON):
        for _o in sel.option:
            if _o.type==OptionType.CARD:
                _c=get_card(obs,_o.area,_o.index,_o.playerIndex if _o.playerIndex is not None else me)
                if _c is not None: setup_basic_ids.append(_c.id)
    scores=[]
    for o in sel.option:
        s=0.0
        card=None
        if o.type in (OptionType.CARD,OptionType.TOOL_CARD,OptionType.ENERGY_CARD,OptionType.ENERGY):
            card=get_card(obs,o.area,o.index,o.playerIndex if o.playerIndex is not None else me)
        if o.type==OptionType.NUMBER:
            s=float(o.number or 0)
        elif o.type==OptionType.YES:
            if sel.context==SelectContext.IS_FIRST: s=-1000 # choose second
            elif sel.context==SelectContext.MULLIGAN: s=-100
            else: s=100
        elif o.type==OptionType.NO:
            if sel.context==SelectContext.IS_FIRST: s=1000
            elif sel.context==SelectContext.MULLIGAN: s=100
            else: s=0
        elif o.type==OptionType.CARD and card is not None:
            # Setup / promotion / switch.
            if sel.context in (SelectContext.SETUP_ACTIVE_POKEMON,SelectContext.TO_ACTIVE,SelectContext.SWITCH):
                if (o.playerIndex is not None and o.playerIndex != me):
                    s=target_value(card,reach,arch) + (60000 if active and not wall_blocks(active,card) else -15000)
                else:
                    s=wall_fit(card,arch)*1000 + (22000 if ready(card) else -2000*missing_cost(card))
                    if arch=='alakazam' and effect_guard(card): s+=260000
                if (o.playerIndex is None or o.playerIndex == me):
                    if sel.context==SelectContext.SETUP_ACTIVE_POKEMON:
                        if arch in ('archaludon','hydrapple'):
                            s=150000 if card.id==CORNERSTONE else (90000 if card.id==DWEBBLE else (-30000 if card.id==BUDEW else 0))
                        elif arch=='alakazam':
                            if card.id==BUDEW and FUECOCO in setup_basic_ids: s=360000
                            elif card.id==FUECOCO: s=320000
                            elif card.id==DWEBBLE: s=120000
                            elif card.id==CORNERSTONE: s=90000
                            else: s=0
                        elif arch in ('marnie','dragapult','mirror'):
                            s=155000 if card.id==DWEBBLE else (72000 if card.id==CORNERSTONE else -30000)
                        else:
                            s=125000 if card.id==DWEBBLE else (90000 if card.id==CORNERSTONE else -25000)
                    if card.id==DWEBBLE and sel.context!=SelectContext.SETUP_ACTIVE_POKEMON: s-=12000
            elif sel.context in (SelectContext.SETUP_BENCH_POKEMON,SelectContext.TO_BENCH,SelectContext.TO_FIELD):
                if sel.context==SelectContext.SETUP_BENCH_POKEMON:
                    if arch in ('archaludon','hydrapple'):
                        s=180000 if card.id==CORNERSTONE else (65000 if card.id==DWEBBLE else -25000)
                    elif arch=='alakazam':
                        s=380000 if card.id==FUECOCO else (170000 if card.id==DWEBBLE else (110000 if card.id==CORNERSTONE else (120000 if card.id==BUDEW else -1000)))
                    elif arch in ('marnie','dragapult','mirror'):
                        s=165000 if card.id==DWEBBLE else (65000 if card.id==CORNERSTONE else -25000)
                    else:
                        s=120000 if card.id==DWEBBLE else (85000 if card.id==CORNERSTONE else -20000)
                elif card.id==DWEBBLE:
                    if arch=='alakazam' and len(mine.bench or [])==0: s=600000
                    else: s=(420000 if arch=='marnie' and own_counts[DWEBBLE]+own_counts[CRUSTLE]==0 else (135000 if arch=='marnie' and own_counts[DWEBBLE]+own_counts[CRUSTLE]<2 else (52000 if own_counts[DWEBBLE]+own_counts[CRUSTLE]<2 else 8000)))
                    if arch=='dragapult' and stadium!=CAGE: s=-4000
                elif card.id==CORNERSTONE:
                    no_crustle_line=own_counts[DWEBBLE]+own_counts[CRUSTLE]==0
                    if arch=='alakazam' and len(mine.bench or [])==0: s=580000
                    if arch=='mirror': s=80000 if no_crustle_line and own_counts[CORNERSTONE]==0 else -30000
                    elif arch=='alakazam': s=580000 if len(mine.bench or [])==0 else (26000 if no_crustle_line and own_counts[CORNERSTONE]==0 else -16000)
                    else: s=62000 if arch=='dragapult' and no_crustle_line and own_counts[CORNERSTONE]<1 else (50000 if own_counts[CORNERSTONE]<1 and arch not in ('dragapult','marnie') else (18000 if own_counts[CORNERSTONE]<2 and arch not in ('dragapult','marnie') else -10000))
                elif card.id==SHAYMIN:
                    s=540000 if arch=='alakazam' and len(mine.bench or [])==0 else (90000 if arch=='marnie' and stadium==CAGE and own_counts[SHAYMIN]==0 and core_ready else (-60000 if arch=='marnie' else -5000))
                elif card.id==MUNKIDORI:
                    s=560000 if arch=='alakazam' and len(mine.bench or [])==0 else (315000 if arch=='marnie' and own_counts[MUNKIDORI]==0 and len(mine.bench or [])<4 and core_ready and dark_access else (65000 if arch=='marnie' and own_counts[MUNKIDORI]==0 and active_progress>=2 and dark_access else -25000))
                elif card.id==BUDEW:
                    s=105000 if arch=='alakazam' and st.turn<=2 and own_counts[BUDEW]==0 and any(p.id in (DWEBBLE,CORNERSTONE,FUECOCO) for p in own) else -25000
                elif card.id==FUECOCO:
                    s=(480000 if own_counts[FUECOCO]+own_counts[CROCALOR]+own_counts[SKELEDIRGE]==0 else (115000 if own_counts[FUECOCO]+own_counts[CROCALOR]+own_counts[SKELEDIRGE]<2 else -10000)) if arch=='alakazam' else -35000
                else: s=-1000
            elif sel.context in (SelectContext.TO_HAND,SelectContext.LOOK,SelectContext.EVOLVES_TO,SelectContext.EVOLVES_FROM):
                cid=card.id
                if effect==ULTRA or effect==DAWN or effect==ASCENSION or sel.context in (SelectContext.EVOLVES_TO,SelectContext.EVOLVES_FROM):
                    if cid==CRUSTLE:
                        dweb_ready=any(p.id==DWEBBLE and not p.appearThisTurn for p in own)
                        s=80000 if dweb_ready or effect==ASCENSION else 38000
                    elif cid==DWEBBLE:
                        s=540000 if arch=='marnie' and own_counts[DWEBBLE]+own_counts[CRUSTLE]==0 else (110000 if arch=='marnie' and own_counts[DWEBBLE]+own_counts[CRUSTLE]<2 else (42000 if own_counts[DWEBBLE]+own_counts[CRUSTLE]<2 else 5000))
                    elif cid==CORNERSTONE:
                        no_crustle_line=own_counts[DWEBBLE]+own_counts[CRUSTLE]==0
                        if arch=='mirror': s=-50000
                        elif arch=='alakazam': s=22000 if no_crustle_line and own_counts[CORNERSTONE]==0 else -25000
                        else: s=72000 if arch=='dragapult' and no_crustle_line and own_counts[CORNERSTONE]<1 else (50000 if arch not in ('dragapult','marnie') and own_counts[CORNERSTONE]<1 else 18000)
                    elif cid==SHAYMIN: s=62000 if arch=='marnie' and stadium==CAGE and own_counts[SHAYMIN]==0 and core_ready else (-20000 if arch=='marnie' else 3000)
                    elif cid==MUNKIDORI: s=(300000 if arch=='marnie' and own_counts[MUNKIDORI]==0 and core_ready and dark_access else (55000 if arch=='marnie' and own_counts[MUNKIDORI]==0 and active_progress>=2 and dark_access else -12000))
                    elif cid==BUDEW: s=70000 if arch=='alakazam' and st.turn<=2 and own_counts[BUDEW]==0 else -20000
                    elif cid==FUECOCO: s=(430000 if own_counts[FUECOCO]+own_counts[CROCALOR]+own_counts[SKELEDIRGE]==0 else 85000) if arch=='alakazam' else -30000
                    elif cid==CROCALOR: s=(210000 if any(p.id==FUECOCO for p in own) else 45000) if arch=='alakazam' else -30000
                    elif cid==SKELEDIRGE: s=(520000 if any(p.id in (FUECOCO,CROCALOR) for p in own) else 120000) if arch=='alakazam' else -30000
                    else: s=-1000
                elif effect==CRISPIN:
                    # Prefer the type the desired attacker still lacks.
                    target=next((p for p in own if p.id==desired),active)
                    munk=next((p for p in own if p.id==MUNKIDORI and not any(provides_dark(p,eid) for eid in eids(p))),None)
                    if cid==2: s=270000 if target and target.id==SKELEDIRGE and not any(eid==2 for eid in eids(target)) else 45000
                    elif cid==DARK: s=150000 if target and target.id==SKELEDIRGE and energy_units(target)<2 else (165000 if arch=='marnie' and munk else 25000)
                    elif cid==FIGHTING:
                        s=65000 if target and target.id==CORNERSTONE and not any(provides_fighting(target,eid) for eid in eids(target)) else 43000
                    elif cid==GRASS:
                        s=65000 if target and target.id==CRUSTLE and not any(provides_grass(target,eid) for eid in eids(target)) else 43000
                    else: s=-1000
                elif effect==STRETCHER:
                    if cid==CRUSTLE: s=70000
                    elif cid==CORNERSTONE: s=68000 if arch not in ('dragapult','marnie') else 35000
                    elif cid==DWEBBLE: s=54000
                    elif cid in (GRASS,FIGHTING): s=40000
                    elif cid==DARK: s=52000 if any(p.id==MUNKIDORI for p in own) else 5000
                    else: s=0
                else:
                    s=1000
            elif sel.context==SelectContext.DISCARD:
                s=discard_score(card.id,own,arch,hand_counts)
            elif sel.context in (SelectContext.HEAL,SelectContext.REMOVE_DAMAGE_COUNTER,SelectContext.EFFECT_TARGET,SelectContext.ATTACH_FROM):
                if hasattr(card,'hp'):
                    if arch=='marnie' and any(p.id==MUNKIDORI for p in own):
                        s=damage(card)*1200 + (90000 if card is active else 0) + wall_fit(card,arch)*100
                    else:
                        s=damage(card)*500 + (20000 if card is active else 0) + wall_fit(card,arch)*100
                else: s=0
            elif sel.context==SelectContext.ATTACH_TO:
                # This can be either choosing an Energy from deck or choosing target.
                if card.id in (GRASS,2,FIGHTING,DARK,MIST,GROW_GRASS,ROCK_FIGHTING,ENRICHING,PRISM,IGNITION,LEGACY):
                    target=next((p for p in own if p.id==desired),active)
                    if arch=='alakazam' and target and not effect_guard(target) and (card.id==MIST or (card.id==ROCK_FIGHTING and target.id==CORNERSTONE)):
                        s=130000
                    elif card.id in (FIGHTING,ROCK_FIGHTING) and target and target.id==CORNERSTONE: s=65000
                    elif card.id in (GRASS,GROW_GRASS) and target and target.id==CRUSTLE: s=65000
                    elif card.id==ENRICHING: s=56000
                    else: s=42000
                elif hasattr(card,'hp'):
                    s=wall_fit(card,arch)*1000 - missing_cost(card)*4000 + (15000 if card is active else 0)
            elif sel.context in (SelectContext.DAMAGE,SelectContext.DAMAGE_COUNTER,SelectContext.DAMAGE_COUNTER_ANY):
                if hasattr(card,'hp'): s=target_value(card,sel.remainDamageCounter*10 if sel.remainDamageCounter else 50,arch)
            elif sel.context==SelectContext.NOT_MOVE:
                s=-discard_score(card.id,own,arch,hand_counts)
            else:
                s=1000
        elif o.type==OptionType.PLAY:
            card=get_card(obs,AreaType.HAND,o.index,me)
            if card is None: s=-10000
            elif card.id==DWEBBLE:
                lines=own_counts[DWEBBLE]+own_counts[CRUSTLE]
                if arch in ('marnie','dragapult','mirror'): s=125000 if lines<2 and len(mine.bench or [])<3 else -8000
                elif arch=='alakazam': s=105000 if lines<1 else (25000 if lines<2 else -10000)
                elif arch in ('archaludon','hydrapple'): s=55000 if lines<1 else -12000
                else: s=85000 if lines<2 else -5000
                if arch=='dragapult' and stadium!=CAGE and lines>=1: s=-5000
            elif card.id==CORNERSTONE:
                no_crustle_line=own_counts[DWEBBLE]+own_counts[CRUSTLE]==0
                if arch=='mirror': s=95000 if no_crustle_line and own_counts[CORNERSTONE]==0 else -25000
                elif arch=='alakazam':
                    protected=any(effect_guard(p) for p in own)
                    s=220000 if own_counts[CORNERSTONE]==0 and not protected else (70000 if own_counts[CORNERSTONE]==0 else -18000)
                elif arch in ('archaludon','hydrapple'):
                    s=205000 if own_counts[CORNERSTONE]==0 else (65000 if own_counts[CORNERSTONE]<2 and len(mine.bench or [])<3 else -12000)
                elif arch=='dragapult': s=65000 if no_crustle_line and own_counts[CORNERSTONE]<1 else -12000
                elif arch=='marnie': s=25000 if no_crustle_line and own_counts[CORNERSTONE]<1 else -15000
                else: s=75000 if own_counts[CORNERSTONE]<1 else -5000
            elif card.id==SHAYMIN:
                s=95000 if arch=='marnie' and stadium==CAGE and own_counts[SHAYMIN]==0 else -5000
            elif card.id==MUNKIDORI:
                s=(350000 if arch=='marnie' and own_counts[MUNKIDORI]==0 and len(mine.bench or [])<4 and core_ready and dark_access else (65000 if arch=='marnie' and own_counts[MUNKIDORI]==0 and active_progress>=2 and dark_access else -30000))
            elif card.id==BUDEW:
                backup=any(p.id in (DWEBBLE,CRUSTLE,CORNERSTONE) for p in own)
                s=165000 if arch=='alakazam' and st.turn<=2 and backup and own_counts[BUDEW]==0 and len(mine.bench or [])<3 else -30000
            elif card.id==FUECOCO:
                s=(520000 if own_counts[FUECOCO]+own_counts[CROCALOR]+own_counts[SKELEDIRGE]==0 else (120000 if own_counts[FUECOCO]+own_counts[CROCALOR]+own_counts[SKELEDIRGE]<2 else -10000)) if arch=='alakazam' else -40000
            elif card.id==1079:
                can_candy=any(p.id==FUECOCO and not p.appearThisTurn for p in own) and hand_counts[SKELEDIRGE]>0
                s=(760000 if can_candy else -5000) if arch=='alakazam' else -40000
            elif card.id==CRISPIN:
                viable=[p for p in own if p.id in ((DWEBBLE,CRUSTLE,CORNERSTONE,SKELEDIRGE) if arch=='alakazam' else (DWEBBLE,CRUSTLE,CORNERSTONE))]
                need=min([missing_cost(p) for p in viable] or [99])
                target=next((p for p in own if p.id==desired and not ready(p)),None)
                if not viable: s=-12000
                elif arch=='archaludon' and any(p.id==CORNERSTONE and not ready(p) for p in own): s=255000
                elif arch in ('marnie','dragapult','mirror') and any(p.id in (DWEBBLE,CRUSTLE) and not ready(p) for p in own): s=210000
                elif arch=='alakazam' and target is not None: s=420000 if target.id==SKELEDIRGE else 145000
                else: s=105000 if need>=1 else -5000
            elif card.id==LILLIE:
                no_core=not any(p.id in ((DWEBBLE,CRUSTLE,CORNERSTONE,FUECOCO,CROCALOR,SKELEDIRGE) if arch=='alakazam' else (DWEBBLE,CRUSTLE,CORNERSTONE)) for p in own)
                not_ready=not core_ready
                wrong_arch=(arch in ('archaludon','hydrapple') and not any(p.id==CORNERSTONE for p in own)) or (arch in ('marnie','dragapult','mirror') and not any(p.id in (DWEBBLE,CRUSTLE) for p in own))
                dead_hand=(not_ready and not any(c.id in (CRISPIN,ULTRA) for c in (mine.hand or [])))
                if no_core or wrong_arch:
                    s=320000
                elif dead_hand:
                    s=285000
                elif not_ready and mine.handCount<=6:
                    s=235000
                elif core_ready and hand_counts[BOSS]==0 and mine.handCount>=5 and arch in ('marnie','dragapult','archaludon'):
                    s=135000
                else:
                    s=90000 if mine.handCount<=4 else (80000 if st.turn<=2 and len(mine.prize or [])==6 and mine.handCount<=6 else -5000)
            elif card.id==BOSS:
                if reach<=0 or not opp.bench: s=-10000
                else:
                    best=max(target_value(p,reach,arch) + (50000 if active and not wall_blocks(active,p) else 0) for p in opp.bench if p is not None)
                    cur=(target_value(oa,reach,arch) if oa else 0)
                    if active and oa and wall_blocks(active,oa): cur -= 65000
                    killable=[p for p in (opp.bench or []) if p is not None and p.hp<=reach]
                    if arch=='marnie':
                        engines=[p for p in (opp.bench or []) if p is not None and p.id in (104,112) and p.hp<=reach]
                        s=430000 if engines else (330000 if killable else 205000 if best>cur+5000 else -3000)
                    elif arch=='archaludon':
                        bypass=[p for p in (opp.bench or []) if p is not None and p.id in (169,666) and p.hp<=reach]
                        s=440000 if active and active.id==CORNERSTONE and bypass else (220000 if killable else 110000 if best>cur+5000 else -3000)
                    elif arch=='dragapult':
                        engines=[p for p in (opp.bench or []) if p is not None and p.id in (120,119,235,112,31) and p.hp<=reach]
                        s=380000 if engines else (250000 if killable else 125000 if best>cur+5000 else -3000)
                    elif arch=='alakazam':
                        engines=[p for p in (opp.bench or []) if p is not None and p.id in (743,742,741) and p.hp<=reach]
                        s=410000 if engines else (230000 if killable else 115000 if best>cur+5000 else -3000)
                    else: s=170000 if killable else (100000 if best>cur+5000 else -3000)
            elif card.id==XEROSIC:
                if arch=='alakazam': s=620000 if core_ready and opp.handCount>=4 else (450000 if opp.handCount>=5 else (290000 if opp.handCount>=4 else -5000))
                else: s=130000 if opp.handCount>=7 else -5000
            elif card.id==HAND_TRIMMER:
                s=260000 if arch=='alakazam' and opp.handCount>=7 else (150000 if opp.handCount>=6 else -5000)
            elif card.id==VITAL:
                s=900000 if arch in ('marnie','alakazam') and active and damage(active)>=50 else (260000 if active and damage(active)>=100 else -5000)
            elif card.id==UNFAIR:
                s=420000 if arch=='alakazam' else 180000
            elif card.id==COOK:
                if arch=='mirror' and active: s=300000 if damage(active)>=120 else (220000 if damage(active)>=70 else -5000)
                else: s=(250000 if damage(active)>=70 else 175000 if damage(active)>=40 else -5000) if arch=='marnie' and active else (76000 if active and damage(active)>=60 else -5000)
            elif card.id==DAWN:
                s=150000 if arch=='mirror' and any(p.id==DWEBBLE for p in own) and hand_counts[CRUSTLE]==0 else (64000 if own_counts[DWEBBLE]+own_counts[CRUSTLE]<2 or any(p.id==DWEBBLE for p in own) else 18000)
            elif card.id==POFFIN:
                lines=own_counts[DWEBBLE]+own_counts[CRUSTLE]
                target_lines=2 if (arch in ('marnie','dragapult','mirror') or len(mine.bench or [])==0 or st.turn<=2) else 1
                if len(mine.bench or [])==0: s=650000
                elif arch=='marnie' and lines==0 and len(mine.bench or [])<4: s=480000
                elif lines==0 and len(mine.bench or [])<4: s=190000
                elif lines<target_lines and len(mine.bench or [])<3: s=85000
                else: s=-12000
                if arch=='dragapult' and stadium!=CAGE and lines>=1: s=-8000
                if arch=='marnie' and stadium!=CAGE and lines>=1: s=-18000
            elif card.id==ULTRA:
                dweb_ready=any(p.id==DWEBBLE and not p.appearThisTurn for p in own)
                if arch=='mirror': s=190000 if any(p.id==DWEBBLE for p in own) and hand_counts[CRUSTLE]==0 and mine.handCount>=3 else (30000 if mine.handCount>=5 else -5000)
                else:
                    missing_arch_wall=(arch in ('archaludon','hydrapple') and own_counts[CORNERSTONE]==0) or (arch in ('marnie','dragapult','alakazam') and own_counts[DWEBBLE]+own_counts[CRUSTLE]==0)
                    s=520000 if arch=='marnie' and own_counts[DWEBBLE]+own_counts[CRUSTLE]==0 and mine.handCount>=3 else (245000 if missing_arch_wall and mine.handCount>=3 else (115000 if dweb_ready and hand_counts[CRUSTLE]==0 and mine.handCount>=3 else (30000 if mine.handCount>=6 else -5000)))
            elif card.id==SWITCH:
                best=max([wall_fit(p,arch)*1000+(25000 if ready(p) else 0) for p in (mine.bench or []) if p is not None] or [-99999])
                cur=wall_fit(active,arch)*1000+(25000 if ready(active) else 0)
                arch_ogerpon=any(p.id==CORNERSTONE and ready(p) for p in (mine.bench or []) if p is not None)
                mirror_ready_bench=any(p is not None and ready(p) for p in (mine.bench or []))
                if arch=='mirror': s=180000 if mirror_ready_bench and (not ready(active) or (active and damage(active)>=120)) else -5000
                elif arch=='alakazam':
                    guarded=any(p is not None and effect_guard(p) for p in (mine.bench or []))
                    s=280000 if guarded and (not active or not effect_guard(active)) else (90000 if best>cur+10000 else -5000)
                elif arch=='archaludon':
                    s=380000 if arch_ogerpon and (not active or active.id!=CORNERSTONE) and oa and oa.id in (169,666) else (170000 if arch_ogerpon and best>cur else (85000 if best>cur+10000 else -5000))
                elif arch=='marnie':
                    ready_crustle=any(p.id==CRUSTLE and ready(p) for p in (mine.bench or []) if p is not None)
                    s=270000 if ready_crustle and (not active or active.id!=CRUSTLE or damage(active)>=60) else (90000 if best>cur+10000 else -5000)
                elif arch=='dragapult':
                    ready_crustle=any(p.id==CRUSTLE and ready(p) for p in (mine.bench or []) if p is not None)
                    s=235000 if ready_crustle and (not active or active.id!=CRUSTLE) else (85000 if best>cur+10000 else -5000)
                else: s=80000 if best>cur+10000 else -5000
            elif card.id==CAGE:
                s=-15000 if arch in ('mirror','archaludon','hydrapple') else (290000 if arch=='alakazam' and stadium==NIGHT_MINE else (470000 if arch=='marnie' and stadium!=CAGE else (300000 if arch=='dragapult' and stadium!=CAGE else (18000 if stadium!=CAGE else -10000))))
            elif card.id==GALETTE:
                status=mine.poisoned or mine.burned or mine.asleep or mine.paralyzed or mine.confused
                s=(210000 if damage(active)>=40 or status else 145000 if damage(active)>=20 else -5000) if arch=='marnie' and active else (65000 if active and (damage(active)>=20 or status) else -5000)
            elif card.id==ICECREAM:
                if arch=='mirror' and active: s=330000 if nrg(active)>=3 and damage(active)>=120 else (240000 if nrg(active)>=3 and damage(active)>=70 else -5000)
                else: s=(260000 if nrg(active)>=3 and damage(active)>=70 else -5000) if arch=='marnie' and active else (79000 if active and nrg(active)>=3 and damage(active)>=70 else -5000)
            elif card.id==STRETCHER:
                key=any(c.id in (DWEBBLE,CRUSTLE,CORNERSTONE,GRASS,FIGHTING) for c in (mine.discard or []))
                s=50000 if key else -5000
            elif card.id==CAPE:
                no_cape=all(not p.tools for p in own)
                if arch=='mirror': s=260000 if active and active.id==CRUSTLE and no_cape else (90000 if no_cape else -5000)
                else: s=360000 if active and active.id==SKELEDIRGE and no_cape else (85000 if active and active.id in (CRUSTLE,CORNERSTONE) and no_cape else 30000 if no_cape else -5000)
            elif card.id==SCRAPPER:
                enemy_tools=sum(len(getattr(p,'tools',[]) or []) for p in enemy)
                dangerous=any(p.id==169 and any(t.id==CAPE for t in (p.tools or [])) for p in enemy)
                mirror_cape=arch=='mirror' and any(any(t.id==CAPE for t in (getattr(p,'tools',[]) or [])) for p in enemy)
                s=360000 if mirror_cape else (240000 if dangerous else (90000 if enemy_tools else -5000))
            else: s=0
        elif o.type==OptionType.ATTACH:
            en=get_card(obs,o.area,o.index,me); p=get_card(obs,o.inPlayArea,o.inPlayIndex,me)
            if en is None or p is None: s=-10000
            else:
                s=wall_fit(p,arch)*1000 + (22000 if p is active else 0) - missing_cost(p)*5000
                if arch in ('archaludon','hydrapple'):
                    if p.id==CORNERSTONE:
                        typed=any(provides_fighting(p,eid) for eid in eids(p))
                        s += 360000 if (not typed and provides_fighting(p,en.id)) else (190000 if not ready(p) else -140000)
                        if en.id==ROCK_FIGHTING: s+=45000
                    elif p.id==DWEBBLE:
                        no_og=not any(q.id==CORNERSTONE for q in own)
                        if no_og and p is active and provides_grass(p,en.id): s+=285000
                        elif no_og and p is active: s+=25000
                        else: s-=90000
                    elif p.id==CRUSTLE:
                        no_og=not any(q.id==CORNERSTONE for q in own)
                        typed=any(provides_grass(p,eid) for eid in eids(p))
                        if no_og and (typed or provides_grass(p,en.id)): s+=205000 if not ready(p) else -90000
                        else: s-=90000
                elif arch in ('marnie','dragapult','mirror'):
                    if p.id in (DWEBBLE,CRUSTLE):
                        typed=any(provides_grass(p,eid) for eid in eids(p))
                        s += 320000 if (not typed and provides_grass(p,en.id)) else (170000 if not ready(p) else -150000)
                        if en.id==GROW_GRASS: s+=50000
                    elif p.id==CORNERSTONE: s-=75000
                if arch=='alakazam' and p is active and ready(p) and en.id not in (MIST,ROCK_FIGHTING):
                    s-=500000
                if arch=='alakazam' and p.id==CORNERSTONE and en.id==ROCK_FIGHTING and not effect_guard(p):
                    # A protected Ogerpon is the emergency pivot when Mist is unavailable.
                    s += 430000 + (90000 if p is not active else 0)
                elif arch=='alakazam' and p is active and not effect_guard(p) and en.id==MIST:
                    s += 460000
                elif arch=='alakazam' and p is active and guard_count(p)<2 and (en.id==MIST or (en.id==ROCK_FIGHTING and p.id==CORNERSTONE)):
                    s += 150000
                if p.id==CRUSTLE:
                    has_typed=any(provides_grass(p,eid) for eid in eids(p))
                    if not has_typed and not provides_grass(p,en.id): s-=520000
                    else:
                        s += 90000 if provides_grass(p,en.id) and not has_typed else 26000
                        if en.id==GROW_GRASS: s+=26000
                elif p.id==CORNERSTONE:
                    has_typed=any(provides_fighting(p,eid) for eid in eids(p))
                    if not has_typed and not provides_fighting(p,en.id): s-=520000
                    else:
                        s += 90000 if provides_fighting(p,en.id) and not has_typed else 26000
                        if en.id==ROCK_FIGHTING: s+=30000
                elif p.id==FUECOCO:
                    if arch=='alakazam' and en.id==MIST and not effect_guard(p):
                        # Protect the Stage-2 seed from Powerful Hand counters;
                        # Mist remains attached after Rare Candy evolution and
                        # also supplies the colourless part of Torcherto.
                        s+=860000 + (120000 if p is not active else 0)
                    elif arch=='alakazam' and en.id==2 and not any(eid==2 for eid in eids(p)):
                        s+=180000
                    else:
                        s-=420000
                elif p.id==SKELEDIRGE:
                    hasfire=any(eid==2 or eid==LEGACY for eid in eids(p))
                    if arch!='alakazam': s-=500000
                    elif not hasfire and en.id==2: s+=620000
                    elif not ready(p): s+=420000
                    else: s-=180000
                elif p.id==MUNKIDORI:
                    hasdark=any(provides_dark(p,eid) for eid in eids(p))
                    if arch=='marnie' and not hasdark and provides_dark(p,en.id):
                        s += 345000 if core_ready else (70000 if active_progress>=2 else -130000)
                    else: s-=190000
                elif p.id==DWEBBLE:
                    if arch=='marnie' and p is not active and not provides_grass(p,en.id): s-=420000
                    elif arch=='marnie' and provides_grass(p,en.id): s+=190000 if nrg(p)==0 else 30000
                    elif arch=='alakazam' and p is active and en.id==MIST: s+=420000
                    elif arch=='alakazam' and en.id==ROCK_FIGHTING: s-=260000
                    elif arch=='alakazam' and en.id==GROW_GRASS: s+=125000
                    else: s += 210000 if arch=='mirror' and p is active and nrg(p)==0 else (70000 if p is active and nrg(p)==0 else -15000)
                else: s-=25000
                if en.id==ENRICHING: s+=45000
                if arch in ('marnie','dragapult','mirror') and ready(p): s-=240000
                elif arch=='alakazam' and ready(p) and effect_guard(p) and guard_count(p)>=2: s-=100000
                elif energy_units(p)>=3: s-=40000
        elif o.type==OptionType.EVOLVE:
            p=get_card(obs,o.inPlayArea,o.inPlayIndex,me)
            s=520000 if p and p.id in (FUECOCO,CROCALOR) and arch=='alakazam' else (-30000 if p and p.id in (FUECOCO,CROCALOR) else (90000 if p and p.id==DWEBBLE else 10000))
        elif o.type==OptionType.ABILITY:
            if arch=='marnie' and any(p.id==MUNKIDORI and any(provides_dark(p,eid) for eid in eids(p)) for p in own) and any(damage(p)>=10 for p in own): s=520000
            else: s=30000
        elif o.type==OptionType.RETREAT:
            best=max([wall_fit(p,arch)*1000+(25000 if ready(p) else 0) for p in (mine.bench or []) if p is not None] or [-99999])
            cur=wall_fit(active,arch)*1000+(25000 if ready(active) else 0)
            mirror_ready_bench=any(p is not None and ready(p) for p in (mine.bench or []))
            if arch=='mirror': s=140000 if mirror_ready_bench and (not ready(active) or (active and damage(active)>=120)) else -5000
            elif arch=='alakazam':
                guarded=any(p is not None and effect_guard(p) for p in (mine.bench or []))
                s=260000 if guarded and (not active or not effect_guard(active)) else (70000 if best>cur+10000 else -5000)
            elif arch=='archaludon':
                ready_og=any(p is not None and p.id==CORNERSTONE and ready(p) for p in (mine.bench or []))
                s=330000 if ready_og and active and active.id!=CORNERSTONE and oa and oa.id in (169,666) else (80000 if best>cur+10000 else -5000)
            elif arch in ('marnie','dragapult'):
                ready_cr=any(p is not None and p.id==CRUSTLE and ready(p) for p in (mine.bench or []))
                s=230000 if ready_cr and active and active.id!=CRUSTLE else (80000 if best>cur+10000 else -5000)
            else: s=70000 if best>cur+10000 else -5000
        elif o.type==OptionType.ATTACK:
            if o.attackId==ASCENSION:
                if arch in ('archaludon','hydrapple') and active and active.id==DWEBBLE and not any(p.id==CORNERSTONE for p in own): s=390000
                else: s=220000 if active and active.id==DWEBBLE and not any(p.id==CRUSTLE for p in own) else (135000 if active and active.id==DWEBBLE else 30000)
            elif o.attackId==ITCHY_POLLEN:
                s=580000 if arch=='alakazam' and not any(ready(p) and p.id==SKELEDIRGE for p in own) else (220000 if arch=='alakazam' else (145000 if arch in ('marnie','dragapult') and st.turn<=2 else 35000))
            elif o.attackId==TORCHERTO:
                dmg=60+20*(len(mine.bench or [])+len(opp.bench or []))
                s=760000+(target_value(oa,dmg,arch) if oa else 0) if arch=='alakazam' else 145000+(target_value(oa,dmg,arch) if oa else 0)
            elif o.attackId in (SCISSORS,DEMOLISH):
                dmg=120 if o.attackId==SCISSORS else 140
                cape_duraludon=(arch=='archaludon' and oa and oa.id==169 and any(t.id==CAPE for t in (oa.tools or [])))
                if arch=='mirror':
                    s=260000 + (target_value(oa,dmg,arch) if oa else 0)
                elif arch=='alakazam':
                    s=340000 + (target_value(oa,dmg,arch) if oa else 0) + (70000 if active and effect_guard(active) else 0)
                elif cape_duraludon:
                    s=-30000
                elif arch=='archaludon' and oa and oa.id==169 and active and active.id==CRUSTLE:
                    s=-20000
                elif arch=='archaludon' and active and active.id==CORNERSTONE and oa and oa.id in (169,666):
                    s=520000 + target_value(oa,dmg,arch)
                elif arch=='marnie' and oa and oa.id in (104,112):
                    s=480000 + target_value(oa,dmg,arch)
                elif active and oa and wall_blocks(active,oa):
                    s=12000 + (5000 if oa.hp<=dmg else 0)
                else:
                    s=100000 + (target_value(oa,dmg,arch) if oa else 0)
            else: s=50000
            if pending_utility: s-=900000
        elif o.type==OptionType.END:
            if pending_utility:
                s=-950000
                scores.append(s)
                continue
            cape_duraludon=(arch=='archaludon' and oa and oa.id==169 and any(t.id==CAPE for t in (oa.tools or [])))
            if arch in ('mirror','alakazam') and attack_available:
                s=-400000
            elif cape_duraludon or (arch=='archaludon' and active and active.id==CRUSTLE and oa and oa.id==169):
                s=50000
            elif arch=='marnie':
                s=-180000 if attack_available else -1000
            elif active and oa and wall_blocks(active,oa):
                s=26000
            else:
                s=-100000 if attack_available else -1000
        elif o.type==OptionType.TOOL_CARD:
            tool=get_card(obs,o.area,o.index,o.playerIndex if o.playerIndex is not None else me)
            s=(420000 if arch=='alakazam' and tool and tool.id==CAPE and active and active.id==SKELEDIRGE else 120000 if tool and tool.id==CAPE else 30000)
        elif o.type in (OptionType.ENERGY,OptionType.ENERGY_CARD):
            s=1000
        else:
            s=0
        scores.append(s)
    return scores

def choose(obs: Observation):
    scores=base_scores(obs)
    sel=obs.select
    cap=sel.maxCount
    effect=sel.effect.id if sel.effect is not None else 0
    if effect==POFFIN:
        st=obs.current; me=st.yourIndex; arch=archetype(obs)
        lines=sum(p.id in (DWEBBLE,CRUSTLE) for p in field(st.players[me]))
        target_lines=2 if (arch in ('marnie','dragapult','mirror') or len(st.players[me].bench or [])==0 or st.turn<=2) else 1
        cap=max(sel.minCount,min(cap,max(0,target_lines-lines)))
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    out=[]
    for i in order:
        if len(out)>=cap: break
        if len(out)<sel.minCount or scores[i]>=0:
            out.append(i)
    if len(out)<sel.minCount:
        out=order[:sel.minCount]
    return out

def agent(observation: dict):
    global _seen, _last_step
    if not observation or observation.get('select') is None:
        return list(my_deck)
    # Kaggle may reuse an imported module for multiple episodes. Reset opponent
    # knowledge whenever the environment step counter goes backwards at a new game.
    raw_step = observation.get('step')
    if raw_step is not None:
        try:
            step = int(raw_step)
            if _last_step is None or step < _last_step:
                _seen=set()
            _last_step=step
        except Exception:
            pass
    obs=to_observation_class(observation)
    if raw_step is None and obs.current.turn==0 and not obs.logs:
        _seen=set()
    return choose(obs)


# Tournament-ratio Crustle core retained from v130.  Keep the empirically strong 4-4 Crustle and
# three Cornerstone walls, but move the list toward the Limitless trainer and
# energy engine. Kangaskhan is a one-of non-ex bypass answer, not the default.
KANGASKHAN=756;PETREL=1219;HILDA=1225;ERI=1186;POKEGEAR=1122;SPIKY=14;RAPID_COMBO=1092

def is_ex(p):
 c=cards.get(p.id);return bool(c and (getattr(c,'ex',False) or getattr(c,'megaEx',False)))
def prizes(p):
 c=cards.get(p.id);return 3 if c and getattr(c,'megaEx',False) else 2 if c and getattr(c,'ex',False) else 1
_oldready=ready
def ready(p):
 if p and p.id==KANGASKHAN:return energy_units(p)>=3
 return _oldready(p)
_oldad=attack_damage
def attack_damage(p):
 if p and p.id==KANGASKHAN and ready(p):return 250
 return _oldad(p)
_oldmc=missing_cost
def missing_cost(p):
 if p and p.id==KANGASKHAN:return max(0,3-energy_units(p))
 return _oldmc(p)
_oldwf=wall_fit
def wall_fit(p,arch):
 if p and p.id==KANGASKHAN:return 145 if arch in ('mirror','alakazam') else 45
 return _oldwf(p,arch)

_legacy=base_scores

def base_scores(obs:Observation):
 scores=_legacy(obs)
 st=obs.current;sel=obs.select;me=st.yourIndex;mine=st.players[me];opp=st.players[1-me]
 own=field(mine);enemy=field(opp);active=mine.active[0] if mine.active and mine.active[0] is not None else None;oa=opp.active[0] if opp.active and opp.active[0] is not None else None
 arch=archetype(obs);effect=sel.effect.id if sel.effect is not None else 0;ownc=Counter(p.id for p in own);reach=attack_damage(active)
 wall_id=CORNERSTONE if arch in ('archaludon','hydrapple') else CRUSTLE
 wall=next((p for p in own if p.id==wall_id),None);seed=next((p for p in own if p.id==DWEBBLE),None);kang=next((p for p in own if p.id==KANGASKHAN),None)
 need_kang=arch in ('alakazam','mirror') or bool(oa and not is_ex(oa) and wall and not wall_blocks(wall,oa))
 for i,o in enumerate(sel.option):
  c=None
  if o.type in (OptionType.CARD,OptionType.TOOL_CARD,OptionType.ENERGY_CARD,OptionType.ENERGY):c=get_card(obs,o.area,o.index,o.playerIndex if o.playerIndex is not None else me)
  elif o.type==OptionType.PLAY:c=get_card(obs,AreaType.HAND,o.index,me)
  if o.type==OptionType.CARD and c is not None:
   if sel.context==SelectContext.SETUP_ACTIVE_POKEMON and c.id==KANGASKHAN:
    others=[]
    for oo in sel.option:
     if oo.type==OptionType.CARD:
      cc=get_card(obs,oo.area,oo.index,oo.playerIndex if oo.playerIndex is not None else me)
      if cc:others.append(cc.id)
    scores[i]=3800000 if not any(x in others for x in (DWEBBLE,CORNERSTONE)) else 500000
   elif sel.context==SelectContext.SETUP_BENCH_POKEMON and c.id==KANGASKHAN:scores[i]=2500000 if need_kang else -1500000
   elif sel.context in (SelectContext.TO_BENCH,SelectContext.TO_FIELD) and c.id==KANGASKHAN:scores[i]=3600000 if need_kang and ownc[KANGASKHAN]==0 else -1200000
   elif sel.context in (SelectContext.TO_HAND,SelectContext.LOOK,SelectContext.EVOLVES_TO,SelectContext.EVOLVES_FROM):
    if effect in (ULTRA,HILDA,ASCENSION) or sel.context in (SelectContext.EVOLVES_TO,SelectContext.EVOLVES_FROM):
     if c.id==KANGASKHAN:scores[i]=3500000 if need_kang else -1000000
     elif c.id in (MIST,GROW_GRASS,SPIKY,ROCK_FIGHTING,GRASS,FIGHTING,DARK):
      if wall and not ready(wall):
       if wall.id==CRUSTLE and c.id==GROW_GRASS:scores[i]=4800000
       elif wall.id==CORNERSTONE and c.id==ROCK_FIGHTING:scores[i]=4800000
       elif c.id==MIST and arch=='alakazam':scores[i]=4700000
       else:scores[i]=4200000
      elif kang and not ready(kang):scores[i]=4300000
    if effect==POKEGEAR and cards.get(c.id) and cards[c.id].cardType==CardType.SUPPORTER:
     v={HILDA:4800000,LILLIE:4600000,BOSS:4400000,ERI:4200000,XEROSIC:4100000,PETREL:3900000}.get(c.id,3000000)
     if mine.handCount<=4 and c.id==LILLIE:v=5000000
     if seed and not any(p.id==CRUSTLE for p in own) and c.id==HILDA:v=5100000
     scores[i]=v
    if effect==PETREL and cards.get(c.id) and cards[c.id].cardType in (CardType.ITEM,CardType.TOOL,CardType.STADIUM,CardType.SUPPORTER):
     dmg=damage(active) if active else 0;v=0
     if c.id==POFFIN:v=1300000 if len(mine.bench or [])==0 or ownc[DWEBBLE]+ownc[CRUSTLE]==0 else 200000
     elif c.id==ULTRA:v=920000 if seed and not any(p.id==CRUSTLE for p in own) else 180000
     elif c.id==ICECREAM:v=1050000 if active and nrg(active)>=3 and dmg>=50 else 120000
     elif c.id==SWITCH:v=900000 if wall and ready(wall) and active is not wall else 100000
     elif c.id==CAGE:v=920000 if arch in ('dragapult','marnie','alakazam') and (not st.stadium or st.stadium[0].id!=CAGE) else 100000
     elif c.id==CAPE:v=760000 if active and not active.tools else 100000
     elif c.id==SCRAPPER:v=800000 if any(getattr(p,'tools',[]) for p in enemy) else 100000
     elif c.id==POKEGEAR:v=620000
     else:v=250000
     scores[i]=5000000+v
  if o.type==OptionType.PLAY and c is not None:
   if c.id==KANGASKHAN:scores[i]=3500000 if need_kang and ownc[KANGASKHAN]==0 else -1500000
   elif c.id==PETREL:scores[i]=6400000 if len(mine.bench or [])==0 else 4550000
   elif c.id==HILDA:
    need=(seed and not any(p.id==CRUSTLE for p in own)) or (wall and not ready(wall)) or (kang and need_kang and not ready(kang))
    scores[i]=5000000 if need else 1600000
   elif c.id==POKEGEAR:scores[i]=4300000 if not st.supporterPlayed else 1200000
   elif c.id==ERI:scores[i]=4500000 if arch in ('alakazam','dragapult','marnie') and opp.handCount>=5 else 1000000
  elif o.type==OptionType.ATTACH:
   en=get_card(obs,o.area,o.index,me);p=get_card(obs,o.inPlayArea,o.inPlayIndex,me)
   if en and p and p.id==KANGASKHAN:
    scores[i]=4700000 if need_kang and not ready(p) and (not wall or ready(wall)) else -4000000
  elif o.type==OptionType.ABILITY:
   if active and active.id==KANGASKHAN:scores[i]=5600000
  elif o.type==OptionType.RETREAT:
   if active and active.id==KANGASKHAN and wall and ready(wall):scores[i]=5700000
   elif need_kang and kang and ready(kang) and active is not kang:scores[i]=5500000
  elif o.type==OptionType.ATTACK and o.attackId==RAPID_COMBO:scores[i]=5500000+(target_value(oa,250,arch) if oa else 0)
  elif o.type==OptionType.END and any(x.type==OptionType.ATTACK for x in sel.option):scores[i]=-5500000
 # Never finish the action chain with an empty Bench when a backup can be deployed.
 if arch!='dragapult' and len(mine.bench or [])==0:
  for j,oo in enumerate(sel.option):
   cc=None
   if oo.type==OptionType.CARD and sel.context in (SelectContext.TO_BENCH,SelectContext.TO_FIELD):
    cc=get_card(obs,oo.area,oo.index,oo.playerIndex if oo.playerIndex is not None else me)
    if cc and cc.id in (DWEBBLE,CORNERSTONE): scores[j]=max(scores[j],6800000 if cc.id==DWEBBLE else 6600000)
   elif oo.type==OptionType.PLAY:
    cc=get_card(obs,AreaType.HAND,oo.index,me)
    if cc and cc.id==POFFIN: scores[j]=max(scores[j],6700000)
    elif cc and cc.id==ULTRA and mine.handCount>=4: scores[j]=max(scores[j],6250000)

 return scores


# Kaggle raw-source loader: keep the submission callable as the final new symbol.
def submission_agent(observation):
    return agent(observation)


# v131+ replay-derived Future/Iron Thorns matchup branch.
STONJOURNER=682
BOUNDLESS_POWER=988
STONY_KICK=987
OPPONENT_IDS['future']={37,80,87,75,140,184}

_prev_target_value=target_value
def target_value(p,reach,arch):
    v=_prev_target_value(p,reach,arch)
    if arch=='future':
        if p.id==87: v+=150000  # Miraidon is the non-ex bypass and energy engine.
        elif p.id==80: v+=85000 # Iron Crown damage amplifier.
        elif p.id==37: v+=70000
        elif p.id==75: v+=45000
        if p.hp<=reach: v+=90000
    return v

_prev_ready=ready
def ready(p):
    if p and p.id==STONJOURNER:
        ids=eids(p); return sum(1 for x in ids if x in (FIGHTING,ROCK_FIGHTING,LEGACY) or (x==PRISM and cards.get(p.id) and cards[p.id].basic))>=2 and energy_units(p)>=3
    return _prev_ready(p)

_prev_attack_damage=attack_damage
def attack_damage(p):
    if p and p.id==STONJOURNER and ready(p): return 140
    return _prev_attack_damage(p)

_prev_missing_cost=missing_cost
def missing_cost(p):
    if p and p.id==STONJOURNER:
        ids=eids(p); f=sum(1 for x in ids if x in (FIGHTING,ROCK_FIGHTING,LEGACY) or (x==PRISM and cards.get(p.id) and cards[p.id].basic))
        return max(0,2-f)+max(0,3-energy_units(p)-max(0,2-f))
    return _prev_missing_cost(p)

_prev_wall_fit=wall_fit
def wall_fit(p,arch):
    if arch=='future' and p:
        if p.id==STONJOURNER:return 230
        if p.id==CRUSTLE:return 180
        if p.id==KANGASKHAN:return 170
        if p.id==DWEBBLE:return 115
        if p.id==CORNERSTONE:return 5
        if p.id in (MUNKIDORI,SHAYMIN):return -80
    return _prev_wall_fit(p,arch)

_prev_base_scores=base_scores
def base_scores(obs:Observation):
    scores=_prev_base_scores(obs)
    st=obs.current;sel=obs.select;me=st.yourIndex;mine=st.players[me];opp=st.players[1-me]
    own=field(mine);enemy=field(opp);active=mine.active[0] if mine.active and mine.active[0] is not None else None;oa=opp.active[0] if opp.active and opp.active[0] is not None else None
    arch=archetype(obs)
    if arch!='future': return scores
    effect=sel.effect.id if sel.effect is not None else 0
    ownc=Counter(p.id for p in own); handc=Counter(c.id for c in (mine.hand or []))
    has_ston=any(p.id==STONJOURNER for p in own); has_kang=any(p.id==KANGASKHAN for p in own)
    miraidon_active=bool(oa and oa.id==87)
    ex_active=bool(oa and is_ex(oa))
    preferred=STONJOURNER if STONJOURNER in my_deck else (KANGASKHAN if KANGASKHAN in my_deck else CRUSTLE)
    for i,o in enumerate(sel.option):
        c=None
        if o.type in (OptionType.CARD,OptionType.TOOL_CARD,OptionType.ENERGY_CARD,OptionType.ENERGY):
            c=get_card(obs,o.area,o.index,o.playerIndex if o.playerIndex is not None else me)
        elif o.type==OptionType.PLAY:
            c=get_card(obs,AreaType.HAND,o.index,me)
        if o.type==OptionType.CARD and c is not None:
            if sel.context==SelectContext.SETUP_ACTIVE_POKEMON:
                pri={DWEBBLE:9000000,STONJOURNER:7600000,KANGASKHAN:6900000,CRUSTLE:1000000,CORNERSTONE:300000,SHAYMIN:-4000000,MUNKIDORI:-5000000}
                scores[i]=pri.get(c.id,0)
            elif sel.context==SelectContext.SETUP_BENCH_POKEMON:
                pri={DWEBBLE:8500000,STONJOURNER:7800000,KANGASKHAN:7000000,CORNERSTONE:-1000000,SHAYMIN:-3500000,MUNKIDORI:-4500000}
                scores[i]=pri.get(c.id,-500000)
            elif sel.context in (SelectContext.TO_BENCH,SelectContext.TO_FIELD):
                if c.id==DWEBBLE:scores[i]=9200000 if ownc[DWEBBLE]+ownc[CRUSTLE]==0 else 5000000
                elif c.id==STONJOURNER:scores[i]=8800000 if not has_ston else -2000000
                elif c.id==KANGASKHAN:scores[i]=8000000 if not has_kang else -2500000
                elif c.id==CORNERSTONE:scores[i]=-3500000
                elif c.id in (MUNKIDORI,SHAYMIN):scores[i]=-5000000
            elif sel.context in (SelectContext.TO_ACTIVE,SelectContext.SWITCH):
                if o.playerIndex is not None and o.playerIndex!=me:
                    scores[i]=target_value(c,attack_damage(active),arch)
                else:
                    if miraidon_active:
                        scores[i]=(10000000 if c.id==STONJOURNER and ready(c) else 9000000 if c.id==KANGASKHAN and ready(c) else 7000000 if c.id==CRUSTLE and ready(c) else -1000000)
                    else:
                        scores[i]=(9200000 if c.id==CRUSTLE and ready(c) else 8800000 if c.id==STONJOURNER and ready(c) else 7800000 if c.id==KANGASKHAN and ready(c) else -1000000)
            elif sel.context in (SelectContext.TO_HAND,SelectContext.LOOK,SelectContext.EVOLVES_TO,SelectContext.EVOLVES_FROM):
                if effect in (ULTRA,HILDA,ASCENSION) or sel.context in (SelectContext.EVOLVES_TO,SelectContext.EVOLVES_FROM):
                    if c.id==CRUSTLE and any(p.id==DWEBBLE for p in own):scores[i]=10500000
                    elif c.id==preferred and not any(p.id==preferred for p in own):scores[i]=9800000
                    elif c.id==DWEBBLE and ownc[DWEBBLE]+ownc[CRUSTLE]==0:scores[i]=9000000
                    elif c.id==CORNERSTONE:scores[i]=-3500000
                    elif c.id in (MUNKIDORI,SHAYMIN):scores[i]=-4500000
                    elif c.id in (GROW_GRASS,GRASS):
                        scores[i]=9400000 if any(p.id in (DWEBBLE,CRUSTLE) and not ready(p) for p in own) else 5500000
                    elif c.id in (FIGHTING,ROCK_FIGHTING):
                        scores[i]=9300000 if any(p.id==STONJOURNER and not ready(p) for p in own) else 5200000
                    elif c.id in (SPIKY,MIST):scores[i]=4800000
            elif sel.context==SelectContext.DISCARD:
                if c.id in (MUNKIDORI,SHAYMIN,CORNERSTONE):scores[i]=9500000
                elif c.id==CAGE:scores[i]=8500000
        elif o.type==OptionType.PLAY and c is not None:
            if c.id==POFFIN:scores[i]=9800000 if ownc[DWEBBLE]+ownc[CRUSTLE]==0 else 6000000
            elif c.id==ULTRA:scores[i]=9400000 if mine.handCount>=3 and ((not has_ston and preferred==STONJOURNER) or ownc[DWEBBLE]+ownc[CRUSTLE]==0) else 4000000
            elif c.id==HILDA:scores[i]=9700000 if any(p.id==DWEBBLE for p in own) and not any(p.id==CRUSTLE for p in own) else 6500000
            elif c.id==PETREL:scores[i]=7600000
            elif c.id==CAGE:scores[i]=-3000000
            elif c.id==BOSS:
                scores[i]=8500000 if any(p.id in (87,80) for p in (opp.bench or []) if p) and active and ready(active) else 2500000
            elif c.id==ICECREAM:
                scores[i]=9000000 if active and nrg(active)>=3 and damage(active)>=70 else -1000000
        elif o.type==OptionType.ATTACH:
            en=get_card(obs,o.area,o.index,me);p=get_card(obs,o.inPlayArea,o.inPlayIndex,me)
            if en and p:
                if p.id==DWEBBLE:
                    scores[i]=11000000 if nrg(p)==0 and provides_grass(p,en.id) else -3000000
                elif p.id==CRUSTLE:
                    typed=any(provides_grass(p,x) for x in eids(p))
                    scores[i]=10200000 if not typed and provides_grass(p,en.id) else 8500000 if not ready(p) else -2500000
                elif p.id==STONJOURNER:
                    fs=sum(1 for x in eids(p) if x in (FIGHTING,ROCK_FIGHTING,LEGACY))
                    scores[i]=10100000 if fs<2 and en.id in (FIGHTING,ROCK_FIGHTING,LEGACY) else 8200000 if not ready(p) else -2500000
                elif p.id==KANGASKHAN:
                    scores[i]=9000000 if not ready(p) else -2500000
                elif p.id==CORNERSTONE:scores[i]=-4000000
        elif o.type==OptionType.EVOLVE:
            p=get_card(obs,o.inPlayArea,o.inPlayIndex,me)
            if p and p.id==DWEBBLE:scores[i]=11000000
        elif o.type==OptionType.RETREAT:
            ready_ston=any(p.id==STONJOURNER and ready(p) for p in (mine.bench or []) if p)
            ready_cr=any(p.id==CRUSTLE and ready(p) for p in (mine.bench or []) if p)
            ready_k=any(p.id==KANGASKHAN and ready(p) for p in (mine.bench or []) if p)
            if miraidon_active and (ready_ston or ready_k):scores[i]=10800000
            elif ex_active and (ready_cr or ready_ston):scores[i]=10200000
        elif o.type==OptionType.ATTACK:
            if o.attackId==ASCENSION:scores[i]=12000000
            elif o.attackId==BOUNDLESS_POWER:scores[i]=11500000+(target_value(oa,140,arch) if oa else 0)
            elif o.attackId==STONY_KICK:scores[i]=6500000+(target_value(oa,20,arch) if oa else 0)
            elif o.attackId==RAPID_COMBO:scores[i]=10800000+(target_value(oa,200,arch) if oa else 0)
            elif o.attackId==SCISSORS:scores[i]=10000000+(target_value(oa,120,arch) if oa else 0)
            elif o.attackId==DEMOLISH:scores[i]=7000000+(target_value(oa,140,arch) if oa else 0)
        elif o.type==OptionType.END:
            if any(x.type==OptionType.ATTACK for x in sel.option):scores[i]=-12000000
    return scores

# Keep the Kaggle raw-source loader's final callable unambiguous.
def submission_agent(observation):
    return agent(observation)
