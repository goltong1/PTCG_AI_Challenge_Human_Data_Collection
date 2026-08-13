"""Conservative paired-counterfactual policy overlay for Tera Box v24.

The table is trained by rolling the baseline and legal alternatives from the
same sampled hidden cards.  Runtime lookup backs off Exact -> Coarse -> Loose;
missing or unsafe states always retain the verified v23 baseline (SPIBB-style
baseline bootstrapping).
"""
from __future__ import annotations

import json
import os

from cg.api import AreaType, OptionType, SelectContext, to_observation_class

ROOT=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
try:
    MODEL=json.load(open(os.path.join(ROOT,"pairwise_model_v24.json"),encoding="utf-8"))
except Exception:
    MODEL={"entries":{}}
TABLE=MODEL.get("entries") or {}

READY={96:3,108:3,117:3,184:3,272:2,31:2,756:3,230:2,112:2,140:3,1071:3}
KEY_ROLES={96,108,117,184,272,31,756,230,112,140,1071}
PROTECTED_ABILITIES={96,140,756}
PROTECTED_PLAYS={1094,1116,1182,1221}
# Runtime promotion is narrower than the offline table.  Dragapult had no
# repeat-supported general action; Lucario's live audit was negative.  Their
# entries remain research evidence only and therefore fall back to v23.
ALLOWED_MATCHUPS=set()
STATS={"decisions":0,"hits":0,"exact_hits":0,"strategic_hits":0,"general_hits":0,"coarse_hits":0,"loose_hits":0,
       "unsafe_rejects":0,"missing_actions":0,"overrides":0}
TOTAL_STATS={key:0 for key in STATS}


def reset():
    for key in STATS:STATS[key]=0


def _bump(key,amount=1):
    STATS[key]+=amount;TOTAL_STATS[key]+=amount


def _card_id(card):
    return int(getattr(card,"id",0) or 0) if card is not None else 0


def _source(obs,option):
    state=obs.current;me=int(state.yourIndex)
    pi=int(option.playerIndex) if option.playerIndex is not None else me
    player=state.players[pi]
    try:
        area=int(option.area) if option.area is not None else (int(AreaType.HAND) if option.type==OptionType.PLAY else -1)
        zones={
            int(AreaType.DECK):obs.select.deck or [],int(AreaType.HAND):player.hand or [],
            int(AreaType.DISCARD):player.discard or [],int(AreaType.ACTIVE):player.active or [],
            int(AreaType.BENCH):player.bench or [],int(AreaType.STADIUM):state.stadium or [],
            int(AreaType.LOOKING):state.looking or [],
        }
        return zones.get(area,[])[int(option.index)]
    except Exception:
        return None


def _target(obs,option):
    try:
        player=obs.current.players[obs.current.yourIndex]
        area=int(option.inPlayArea);index=int(option.inPlayIndex)
        zone=player.active if area==int(AreaType.ACTIVE) else player.bench if area==int(AreaType.BENCH) else []
        return zone[index]
    except Exception:
        return None


def _signature(obs,index):
    option=obs.select.option[index]
    return ":".join(map(str,(int(option.type),_card_id(_source(obs,option)),
                              _card_id(_target(obs,option)),int(option.attackId or 0))))


def _phase(turn):
    return "e" if turn<=3 else "m" if turn<=8 else "l"


def _keys(obs,matchup,base_signature):
    state=obs.current;me=int(state.yourIndex);mine=state.players[me];opp=state.players[1-me]
    own=[x for x in list(mine.active or [])+list(mine.bench or []) if x]
    active=_card_id(mine.active[0]) if mine.active else 0
    oppactive=_card_id(opp.active[0]) if opp.active else 0
    prize=len(opp.prize or [])-len(mine.prize or [])
    pb="A" if prize>=2 else "a" if prize==1 else "0" if prize==0 else "b" if prize==-1 else "B"
    hand=int(mine.handCount or 0)-int(opp.handCount or 0)
    hb="H" if hand>=3 else "h" if hand>0 else "0" if hand==0 else "l" if hand>=-2 else "L"
    ready=0
    if mine.active and mine.active[0]:
        pokemon=mine.active[0];energy=len(pokemon.energyCards or []);need=READY.get(int(pokemon.id),3)
        ready=2 if energy>=need else 1 if energy>=max(1,need-1) else 0
    ownbench=",".join(map(str,sorted(_card_id(x) for x in mine.bench or [] if x)))
    oppbench=",".join(map(str,sorted(_card_id(x) for x in opp.bench or [] if x)))
    energies=",".join(f"{_card_id(x)}={min(4,len(x.energyCards or []))}" for x in own if _card_id(x) in KEY_ROLES)
    types=",".join(map(str,sorted({int(x.type) for x in obs.select.option})))
    ph=_phase(int(state.turn or 0))
    return [
        ("E",f"{matchup}|{ph}|{pb}|{hb}|{active}|{oppactive}|{ownbench}|{oppbench}|{energies}|{types}"),
        ("B",f"{matchup}|{ph}|{pb}|{active}|{oppactive}|{ready}|{base_signature}|{types}"),
        ("G",f"{matchup}|{ph}|{base_signature}"),
        ("C",f"{matchup}|{ph}|{pb}|{active}|{oppactive}|{ready}|{types}"),
        ("L",f"{matchup}|{ph}|{pb}|{active}|{oppactive}|{types}"),
    ]


def _quality(level,row):
    labels=int(row.get("labels",0));loss_labels=int(row.get("loss_labels",0))
    share=float(row.get("share",0));lcb=float(row.get("mean_lcb",0))
    if level=="E":return labels>=1 and loss_labels>=1 and share>=0.75 and lcb>=80
    if level=="B":return labels>=2 and loss_labels>=1 and share>=0.67 and lcb>=85
    if level=="G":return labels>=2 and loss_labels>=2 and share>=0.67 and lcb>=100
    if level=="C":return labels>=2 and loss_labels>=1 and share>=0.67 and lcb>=75
    return labels>=4 and loss_labels>=2 and share>=0.75 and lcb>=100


def _safe(obs,base_index,new_index):
    base=obs.select.option[base_index];new=obs.select.option[new_index]
    base_card=_card_id(_source(obs,base));new_card=_card_id(_source(obs,new))
    # Never break the verified draw/energy-distribution sequence.  A learned
    # ATTACK may replace it only when the baseline itself is already ATTACK.
    if base.type==OptionType.ABILITY and base_card in PROTECTED_ABILITIES:
        # Reordering one manual attachment onto the same engine is reversible:
        # the once-per-turn ability remains available immediately afterward.
        if not (new.type==OptionType.ATTACH and _card_id(_target(obs,new))==base_card):return False
    if base.type==OptionType.PLAY and base_card in PROTECTED_PLAYS:
        # Bug Catching Set also remains playable after a hand Energy is attached.
        if not (base_card==1094 and new.type==OptionType.ATTACH):return False
    if base.type==OptionType.ATTACK and new.type!=OptionType.ATTACK:return False
    if new.type==OptionType.END and base.type!=OptionType.END:return False
    # Complex cards have follow-up selections whose intent is owned by the
    # baseline specialist; do not enter them from a one-step table.
    if new.type==OptionType.PLAY and new_card in {1116,1182,1221}:return False
    return True


def choose(observation,base,matchup):
    if os.environ.get("TERA_PAIRWISE_DISABLE")=="1":return base
    if matchup not in ALLOWED_MATCHUPS or not base or len(base)!=1:return base
    try:obs=to_observation_class(observation)
    except Exception:return base
    if obs.select is None or obs.select.context!=SelectContext.MAIN:return base
    base_index=int(base[0])
    if not 0<=base_index<len(obs.select.option):return base
    _bump("decisions")
    row=level=None
    for candidate_level,key in _keys(obs,matchup,_signature(obs,base_index)):
        candidate=TABLE.get(candidate_level+"|"+key)
        if candidate and _quality(candidate_level,candidate):
            row=candidate;level=candidate_level;break
    if row is None:return base
    _bump("hits");_bump({"E":"exact_hits","B":"strategic_hits","G":"general_hits","C":"coarse_hits","L":"loose_hits"}[level])
    # Only the audited Alakazam sequencing rule survived live counterfactuals:
    # Active Teal Mask, opposing Alakazam completion/basic pressure, then place
    # Prism on that same Active before using Teal Dance.  Other table levels
    # remain offline diagnostics and cannot change a submitted action.
    mine=obs.current.players[obs.current.yourIndex];opp=obs.current.players[1-obs.current.yourIndex]
    own_active=_card_id(mine.active[0]) if mine.active else 0
    opp_active=_card_id(opp.active[0]) if opp.active else 0
    own_active_energy=len(mine.active[0].energyCards or []) if mine.active and mine.active[0] else 0
    if not (matchup=="alakazam" and level=="G" and own_active==96 and own_active_energy==2 and opp_active in {305,743}):
        _bump("unsafe_rejects");return base
    wanted=str(row.get("action",""));new_index=next((i for i in range(len(obs.select.option)) if _signature(obs,i)==wanted),None)
    if new_index is None:
        _bump("missing_actions");return base
    if new_index==base_index:return base
    if not _safe(obs,base_index,new_index):
        _bump("unsafe_rejects");return base
    _bump("overrides")
    return [new_index]
