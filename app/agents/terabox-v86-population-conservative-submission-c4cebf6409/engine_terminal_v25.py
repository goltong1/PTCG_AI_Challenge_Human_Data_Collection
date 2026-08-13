"""Replay-audited belief-particle Lucario terminal re-solver for v28.

The verified direct policy proposes the baseline.  At one high-leverage MAIN
window per turn, this overlay rolls the baseline and at most two alternatives
to terminal/cutoff from the same sampled hidden cards.  It changes the action
only when every paired particle is non-regressive and the lower-confidence
advantage is positive.  A repeated 0/3 replay regression is vetoed.
Unsupported/error states retain the baseline.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict

from cg.api import (AreaType, LogType, OptionType, SelectContext, all_attack,
                    all_card_data, search_begin, search_end, search_release,
                    search_step, to_observation_class)

ROOT=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
INIT={"current":None,"logs":[],"select":None,"step":0}
CARDS={int(x.cardId):x for x in all_card_data()};ATTACKS={int(x.attackId):x for x in all_attack()}
OWN_ENTRY={"lucario":"policy_memory_lucario.py"}
OPP_ENTRY={"lucario":os.path.join("opponent_models_v25","lucario","opponent_model.py")}
READY={96:3,108:3,117:3,184:3,272:2,31:2,756:3,230:2,112:2,140:3,1071:3}
ENGINE_IDS={96,184,756,1071};TECH={"dragapult":{272,108},"lucario":{272},"alakazam":{117,96}}
COMPLEX={1116,1182,1221}
PLANNERS={}
STATS={"calls":0,"branches":0,"branch_steps":0,"terminal_branches":0,"cutoff_branches":0,
       "errors":0,"robust_rejects":0,"overrides":0,"seconds":0.0,"max_call_seconds":0.0}
AUDIT=[]


def _audit(record):
    AUDIT.append(record)
    if len(AUDIT)>200:del AUDIT[:-200]


def _field_summary(player):
    result=[]
    for area,pokemon in (("a",x) for x in player.active or []):
        if pokemon:result.append([area,int(pokemon.id),int(pokemon.hp or 0),int(pokemon.maxHp or 0),len(pokemon.energyCards or [])])
    for pokemon in player.bench or []:
        if pokemon:result.append(["b",int(pokemon.id),int(pokemon.hp or 0),int(pokemon.maxHp or 0),len(pokemon.energyCards or [])])
    return result


def _state_summary(obs,known_hand):
    state=obs.current;seat=int(state.yourIndex);mine=state.players[seat];opp=state.players[1-seat]
    return {"turn":int(state.turn or 0),"turn_action":int(state.turnActionCount or 0),
            "prizes":[len(mine.prize or []),len(opp.prize or [])],
            "hands":[int(mine.handCount or 0),int(opp.handCount or 0)],
            "decks":[int(mine.deckCount or 0),int(opp.deckCount or 0)],
            "flags":[bool(state.supporterPlayed),bool(state.stadiumPlayed),bool(state.energyAttached),bool(state.retreated)],
            "own_field":_field_summary(mine),"opp_field":_field_summary(opp),
            "known_opp_hand":sorted(map(int,known_hand)),
            "options":[list(_signature(obs,i)) for i in range(len(obs.select.option))]}


def _load(path,tag):
    path=os.path.abspath(path);root=os.path.dirname(path)
    name="_tera_v25_terminal_"+tag+"_"+hashlib.sha1(path.encode()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name]=module;old=os.getcwd();old_path=list(sys.path)
    try:
        sys.path.insert(0,root);os.chdir(root);spec.loader.exec_module(module)
    finally:
        os.chdir(old);sys.path[:]=old_path
    return module


def _init(module):return list(module.agent(INIT))
def _cid(card):return int(getattr(card,"id",0) or 0) if card is not None else 0


def _source(obs,option):
    state=obs.current;me=int(state.yourIndex);pi=int(option.playerIndex) if option.playerIndex is not None else me;pl=state.players[pi]
    try:
        area=int(option.area) if option.area is not None else (int(AreaType.HAND) if option.type==OptionType.PLAY else -1)
        zones={int(AreaType.DECK):obs.select.deck or [],int(AreaType.HAND):pl.hand or [],int(AreaType.DISCARD):pl.discard or [],
               int(AreaType.ACTIVE):pl.active or [],int(AreaType.BENCH):pl.bench or [],int(AreaType.STADIUM):state.stadium or [],
               int(AreaType.LOOKING):state.looking or []}
        return zones.get(area,[])[int(option.index)]
    except Exception:return None


def _target(obs,option):
    try:
        pl=obs.current.players[obs.current.yourIndex];area=int(option.inPlayArea);index=int(option.inPlayIndex)
        return (pl.active if area==int(AreaType.ACTIVE) else pl.bench if area==int(AreaType.BENCH) else [])[index]
    except Exception:return None


def _signature(obs,index):
    o=obs.select.option[index]
    return (int(o.type),_cid(_source(obs,o)),_cid(_target(obs,o)),int(o.attackId or 0),int(o.inPlayArea or -1),int(o.inPlayIndex or -1))


def _add_pokemon(counter,pokemon):
    if pokemon is None:return
    counter[int(pokemon.id)]+=1
    for card in pokemon.energyCards or []:counter[int(card.id)]+=1
    for card in pokemon.tools or []:counter[int(card.id)]+=1
    for card in pokemon.preEvolution or []:counter[int(card.id)]+=1


def _unknown(full_deck,player,own,seed,known_hand=None):
    counter=Counter(map(int,full_deck))
    if own:
        for card in player.hand or []:counter[int(card.id)]-=1
    for pokemon in list(player.active or [])+list(player.bench or []):_add_pokemon(counter,pokemon)
    for card in player.discard or []:counter[int(card.id)]-=1
    for card in getattr(player,"lostZone",None) or []:counter[int(card.id)]-=1
    remaining=[]
    for cid,count in counter.items():remaining.extend([cid]*max(0,count))
    need=int(player.deckCount or 0)+len(player.prize or [])+(0 if own else int(player.handCount or 0))
    if len(remaining)<need:remaining.extend(list(map(int,full_deck))*((need-len(remaining))//max(1,len(full_deck))+1))
    forced=[]
    if not own:
        for cid in known_hand or []:
            try:index=remaining.index(int(cid))
            except ValueError:continue
            forced.append(remaining.pop(index))
    random.Random(seed).shuffle(remaining);dc=int(player.deckCount or 0);pc=len(player.prize or [])
    if own:return remaining[:dc],remaining[dc:dc+pc],[],[]
    hc=int(player.handCount or 0);forced=forced[:hc];unknown_count=max(0,hc-len(forced));hand=forced+remaining[:unknown_count]
    offset=unknown_count
    return remaining[offset:offset+dc],remaining[offset+dc:offset+dc+pc],hand,[]


def _hidden(obs,own_deck,opp_deck,seed,known_hand=None):
    state=obs.current;me=int(state.yourIndex);mine=state.players[me];opp=state.players[1-me]
    yd,yp,_,_=_unknown(own_deck,mine,True,seed*2+1);od,op,oh,oa=_unknown(opp_deck,opp,False,seed*2+2,known_hand)
    return yd,yp,od,op,oh,oa


def _can_pay(pokemon,attack):
    if pokemon is None or attack is None:return False
    pool=list(pokemon.energies or [])
    for requirement in attack.energies:
        req=int(requirement)
        if req==0:
            if not pool:return False
            pool.pop(0);continue
        index=next((i for i,value in enumerate(pool) if int(value) in {req,10}),None)
        if index is None:return False
        pool.pop(index)
    return True


def _ready_damage(pokemon):
    if pokemon is None:return 0
    card=CARDS.get(int(pokemon.id));best=0
    if card is None:return 0
    for aid in card.attacks:
        attack=ATTACKS.get(int(aid))
        if _can_pay(pokemon,attack):best=max(best,int(attack.damage or 1))
    return best


def _cutoff_value(obs,seat,matchup):
    state=obs.current
    if state is None:return -50000.0
    mine=state.players[seat];opp=state.players[1-seat]
    own=[x for x in list(mine.active or [])+list(mine.bench or []) if x];other=[x for x in list(opp.active or [])+list(opp.bench or []) if x]
    value=(len(opp.prize or [])-len(mine.prize or []))*6500.0
    ma=mine.active[0] if mine.active else None;oa=opp.active[0] if opp.active else None
    value+=_ready_damage(ma)*16-_ready_damage(oa)*13
    value+=max([_ready_damage(x) for x in mine.bench or [] if x] or [0])*7
    value-=max([_ready_damage(x) for x in opp.bench or [] if x] or [0])*5
    value+=(sum(int(x.hp or 0) for x in own)-sum(int(x.hp or 0) for x in other))*2.2
    value+=(sum(max(0,int(x.maxHp or 0)-int(x.hp or 0)) for x in other)-sum(max(0,int(x.maxHp or 0)-int(x.hp or 0)) for x in own))*4.0
    ids={int(x.id) for x in own};value+=(1800 if 184 in ids else 0)+(1500 if 756 in ids else 0)+(1200 if 96 in ids else 0)
    value+=1000*len(ids&TECH.get(matchup,set()))
    value+=sum(min(1.0,len(x.energyCards or [])/max(1,READY.get(int(x.id),3)))*1100 for x in own if int(x.id) in READY)
    value-=800*sum(int(x.maxHp or 0)<=120 for x in mine.bench or [] if x)
    value+=(min(8,int(mine.handCount or 0))-min(8,int(opp.handCount or 0)))*160
    return max(-50000.0,min(50000.0,value))


class Planner:
    def __init__(self,matchup):
        self.matchup=matchup
        self.own=_load(os.path.join(ROOT,OWN_ENTRY[matchup]),"own_"+matchup)
        self.opp=_load(os.path.join(ROOT,OPP_ENTRY[matchup]),"opp_"+matchup)
        self.own_deck=_init(self.own);self.opp_deck=_init(self.opp)
        if len(self.own_deck)!=60 or len(self.opp_deck)!=60:raise ValueError("terminal planner requires 60-card models")
        self.last_turn=-1
        self.particles=max(2,int(os.environ.get("TERA_TERMINAL_PARTICLES","3")))
        self.max_candidates=max(2,int(os.environ.get("TERA_TERMINAL_CANDIDATES","3")))
        self.max_steps=max(120,int(os.environ.get("TERA_TERMINAL_STEPS","360")))
        self.margin=float(os.environ.get("TERA_TERMINAL_MARGIN","5000"))
        self.min_flips=max(1,int(os.environ.get("TERA_TERMINAL_MIN_FLIPS",str(max(2,math.ceil(self.particles*.4))))))
        self.known_hand={}

    def reset(self):
        self.last_turn=-1;self.known_hand.clear();_init(self.own);_init(self.opp)

    def observe_logs(self,obs):
        if obs.current is None:return
        seat=int(obs.current.yourIndex);opponent=1-seat
        for log in obs.logs or []:
            if log.playerIndex is None or int(log.playerIndex)!=opponent:continue
            typ=int(log.type);serial=int(log.serial or 0);cid=int(log.cardId or 0)
            from_area=int(log.fromArea) if log.fromArea is not None else -1
            to_area=int(log.toArea) if log.toArea is not None else -1
            if typ==int(LogType.MOVE_CARD):
                if from_area==int(AreaType.HAND) and serial:self.known_hand.pop(serial,None)
                if to_area==int(AreaType.HAND) and serial and cid:self.known_hand[serial]=cid
            elif typ==int(LogType.MOVE_CARD_REVERSE) and from_area==int(AreaType.HAND):
                # A face-down hand return makes the identity of every retained
                # known card ambiguous; clear rather than over-condition.
                self.known_hand.clear()
            elif typ in {int(LogType.PLAY),int(LogType.ATTACH),int(LogType.EVOLVE)} and serial:
                self.known_hand.pop(serial,None)
        hand_count=int(obs.current.players[opponent].handCount or 0)
        if len(self.known_hand)>hand_count:
            self.known_hand=dict(list(self.known_hand.items())[-hand_count:]) if hand_count else {}

    def opponent_action(self,raw):
        controller=getattr(self.opp,"_controller",None);owner=self.opp
        if controller is None:
            for engine in (getattr(self.opp,"ENGINES",{}) or {}).values():
                for candidate in (engine,getattr(engine,"base",None)):
                    if candidate is not None and getattr(candidate,"_controller",None) is not None:
                        owner=candidate;controller=candidate._controller;break
                if controller is not None:break
        if controller is None:return self.opp.agent(raw)
        obs=to_observation_class(raw);turn=int(obs.current.turn or 0)
        try:controller.recognize(obs)
        except Exception:pass
        action=controller.validate(controller.call_base(raw,turn,disable_search=True),obs.select)
        for name in ("_human_order_patch","_ultrasafe_drag_patch","_fast_backup","_alak_line"):
            patch=getattr(owner,name,None) or getattr(self.opp,name,None)
            if patch is not None:
                try:action=patch(raw,action)
                except Exception:pass
        return action

    def candidates(self,obs,base):
        base_index=int(base[0]);ranked=[];seen={_signature(obs,base_index)}
        for index,option in enumerate(obs.select.option):
            sig=_signature(obs,index)
            if index==base_index or sig in seen:continue
            seen.add(sig);card=_source(obs,option);cid=_cid(card);target=_target(obs,option);typ=option.type
            priority={int(OptionType.ATTACK):1100,int(OptionType.ABILITY):900,int(OptionType.EVOLVE):850,
                      int(OptionType.ATTACH):800,int(OptionType.RETREAT):720,int(OptionType.PLAY):600,
                      int(OptionType.END):0}.get(int(typ),100)
            if typ==OptionType.PLAY:
                if cid in COMPLEX:priority-=250
                if cid in {1094,1121,1127,1188,1227}:priority+=180
                if cid==1182:priority+=260
            elif typ==OptionType.ATTACH and target is not None:
                need=READY.get(int(target.id),3);energy=len(target.energyCards or [])
                if energy+1>=need:priority+=350
                if int(target.id) in TECH.get(self.matchup,set()):priority+=180
            elif typ==OptionType.ATTACK:
                attack=ATTACKS.get(int(option.attackId or 0));priority+=int(getattr(attack,"damage",0) or 0)
            elif typ==OptionType.RETREAT:
                priority+=120
            ranked.append((priority,-index,index))
        ranked.sort(reverse=True);result=[base_index]
        for _,__,index in ranked:
            result.append(index)
            if len(result)>=self.max_candidates:break
        return result

    def branch(self,obs,index,hidden_state,seat):
        search_id=None;steps=0;rng_state=random.getstate()
        try:
            _init(self.own);_init(self.opp)
            node=search_begin(obs,*hidden_state,manual_coin=True);search_id=node.searchId
            nxt=search_step(search_id,[index]);search_release(search_id);node=nxt;search_id=nxt.searchId
            while steps<self.max_steps:
                current=node.observation
                if current.current is None:raise ValueError("missing current")
                result=int(current.current.result)
                if result>=0:
                    STATS["terminal_branches"]+=1
                    score=0.0 if result==2 else 100000.0 if result==seat else -100000.0
                    return score,result,steps,None
                actor=int(current.current.yourIndex);raw=asdict(current)
                action=self.own.agent(raw) if actor==seat else self.opponent_action(raw)
                # [] is a legal decline at optional zero-minimum selection
                # windows.  Let the simulator validate it instead of treating
                # it as a policy failure.
                if action is None:raise ValueError("missing rollout action")
                nxt=search_step(search_id,action);search_release(search_id);node=nxt;search_id=nxt.searchId;steps+=1
            STATS["cutoff_branches"]+=1
            return _cutoff_value(node.observation,seat,self.matchup),-1,steps,None
        except Exception as exc:
            STATS["errors"]+=1
            return None,None,steps,f"{type(exc).__name__}:{exc}"
        finally:
            random.setstate(rng_state)
            if search_id is not None:
                try:search_release(search_id)
                except Exception:pass
            try:search_end()
            except Exception:pass

    def choose(self,raw,base):
        if os.environ.get("TERA_TERMINAL_DISABLE")=="1":return base
        try:obs=to_observation_class(raw)
        except Exception:return base
        self.observe_logs(obs)
        if obs.select is None or obs.select.context!=SelectContext.MAIN or not base or len(base)!=1:return base
        turn=int(obs.current.turn or 0)
        if turn==self.last_turn or len(obs.select.option)<2:return base
        choices=self.candidates(obs,base)
        if len(choices)<2:return base
        # Search only at decisions that can change attack timing or a committed
        # board resource; pure duplicate play ordering remains with baseline.
        types={obs.select.option[i].type for i in choices}
        if not (OptionType.ATTACK in types or OptionType.ATTACH in types or OptionType.RETREAT in types or OptionType.EVOLVE in types):return base
        self.last_turn=turn;STATS["calls"]+=1;started=time.perf_counter();seat=int(obs.current.yourIndex)
        values={index:[] for index in choices};outcomes={index:[] for index in choices};failed=False;branch_errors=[]
        public_seed=turn*1000003+int(obs.current.turnActionCount or 0)*9176+sum(_cid(x) for x in obs.current.players[seat].discard or [])*37
        for particle in range(self.particles):
            hs=_hidden(obs,self.own_deck,self.opp_deck,public_seed+particle*104729,list(self.known_hand.values()))
            for index in choices:
                score,outcome,steps,error=self.branch(obs,index,hs,seat);STATS["branches"]+=1;STATS["branch_steps"]+=steps
                if error or score is None:
                    failed=True;branch_errors.append({"particle":particle,"choice":index,"error":error})
                values[index].append(score);outcomes[index].append(outcome)
        elapsed=time.perf_counter()-started;STATS["seconds"]+=elapsed;STATS["max_call_seconds"]=max(STATS["max_call_seconds"],elapsed)
        if failed:
            STATS["robust_rejects"]+=1
            _audit({"matchup":self.matchup,"turn":turn,"base":_signature(obs,int(base[0])),"decision":"reject_error",
                    "seconds":round(elapsed,4),"errors":branch_errors,"values":values,"outcomes":outcomes})
            return base
        bi=int(base[0]);best=bi;best_key=(-1e18,-1e18)
        comparisons={}
        for index in choices:
            if index==bi:continue
            diffs=[a-b for a,b in zip(values[index],values[bi])]
            mean=statistics.fmean(diffs);sd=statistics.pstdev(diffs) if len(diffs)>1 else 0.0;lcb=mean-.85*sd
            base_wins=sum(x==seat for x in outcomes[bi]);alt_wins=sum(x==seat for x in outcomes[index])
            catastrophe=any(b==seat and a not in {seat,2} for a,b in zip(outcomes[index],outcomes[bi]))
            completed=all(x in {0,1,2} for x in outcomes[bi]+outcomes[index])
            flips=alt_wins-base_wins
            base_sig=_signature(obs,bi)[:4];alt_sig=_signature(obs,index)[:4]
            replay_veto=((base_sig==(int(OptionType.ABILITY),96,0,0) and
                          alt_sig==(int(OptionType.RETREAT),0,0,0)) or
                         # Fresh 100-game loss audit: abandoning a legal Full
                         # Moon Rondo (attack 371) for retreat was 0-3.  Keep
                         # the baseline attack; the separate Teal attack-120
                         # retreat line remains legal because it scored 3-1.
                         (base_sig==(int(OptionType.ATTACK),0,0,371) and
                          alt_sig==(int(OptionType.RETREAT),0,0,0)))
            robust=(completed and not catastrophe and alt_wins>base_wins and min(diffs)>=0 and
                    mean>=self.margin and lcb>=self.margin*.35 and not replay_veto)
            comparisons[str(_signature(obs,index))]={"mean":round(mean,2),"lcb":round(lcb,2),"minimum":round(min(diffs),2),
                                                      "base_wins":base_wins,"alt_wins":alt_wins,"flips":flips,
                                                      "completed":completed,"catastrophe":catastrophe,
                                                      "replay_veto":replay_veto,"robust":robust}
            if robust and (lcb,mean)>best_key:best=index;best_key=(lcb,mean)
        if best==bi:
            STATS["robust_rejects"]+=1
            _audit({"matchup":self.matchup,"turn":turn,"base":_signature(obs,bi),"decision":"retain",
                    "seconds":round(elapsed,4),"state":_state_summary(obs,self.known_hand.values()),
                    "comparisons":comparisons,"outcomes":outcomes})
            return base
        STATS["overrides"]+=1
        _audit({"matchup":self.matchup,"turn":turn,"base":_signature(obs,bi),"chosen":_signature(obs,best),
                "decision":"override","seconds":round(elapsed,4),"state":_state_summary(obs,self.known_hand.values()),
                "comparisons":comparisons,"outcomes":outcomes})
        return [best]


def reset():
    for planner in PLANNERS.values():planner.reset()


def choose(observation,base,matchup):
    if matchup not in OWN_ENTRY:return base
    try:
        planner=PLANNERS.get(matchup)
        if planner is None:planner=PLANNERS.setdefault(matchup,Planner(matchup))
        return planner.choose(observation,base)
    except Exception:
        STATS["errors"]+=1;return base
