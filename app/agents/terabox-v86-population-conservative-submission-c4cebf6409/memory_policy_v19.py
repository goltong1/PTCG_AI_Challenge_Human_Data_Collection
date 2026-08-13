"""Conservative public-state replay memory specialists."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys


TYPE_NAME = {
    0:"Number",1:"Yes",2:"No",3:"Card",4:"ToolCard",5:"EnergyCard",6:"Energy",
    7:"Play",8:"Attach",9:"Evolve",10:"Ability",11:"Discard",12:"Retreat",
    13:"Attack",14:"End",15:"Skill",16:"SpecialCondition",
}


def load_module(root, filename, tag):
    path = os.path.join(root, filename)
    name = "_tera_v19_memory_" + tag + "_" + hashlib.sha1(path.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(root, profile):
    base_file = {
        "dragapult":"policy_lucario.py",
        "lucario":"policy_lucario.py",
        "alakazam":"policy_special_alakazam.py",
        "crustle":"policy_crustle.py",
    }[profile]
    module = load_module(root, base_file, profile)
    payload = json.load(open(os.path.join(root, "action_memory_v19.json"), encoding="utf-8"))
    memory = payload["profiles"][profile]["memory"]
    stats = {"hits":0,"overrides":0,"missing_action":0}

    if profile == "dragapult":
        old_damage = module._x_damage
        def damage(obs, pokemon, attack_id):
            value = old_damage(obs, pokemon, attack_id)
            try:
                active = module._x_pl(obs, False).active[0]
                if pokemon and int(pokemon.id)==module.X["CLEF"] and int(attack_id or 0)==371 and int(active.id) in {119,120,121,235}:
                    return value*2
            except Exception:pass
            return value
        module._x_damage = damage

    def cid(card):
        try:return int((card or {}).get("id") or 0)
        except Exception:return 0

    def source(raw, option):
        current=raw.get("current") or {};me=int(current.get("yourIndex") or 0);players=current.get("players") or []
        if me>=len(players):return None
        player=players[me];typ=int(option.get("type",-1));area=option.get("area");index=option.get("index")
        try:
            if typ==7 and area is None:return (player.get("hand") or [])[int(index)]
            arrays={1:(raw.get("select") or {}).get("deck") or [],2:player.get("hand") or [],3:player.get("discard") or [],4:player.get("active") or [],5:player.get("bench") or [],7:current.get("stadium") or [],12:current.get("looking") or []}
            return arrays.get(int(area),[])[int(index)]
        except Exception:return None

    def target(raw, option):
        current=raw.get("current") or {};me=int(current.get("yourIndex") or 0);players=current.get("players") or []
        if me>=len(players):return None
        try:
            area=int(option.get("inPlayArea",-1));index=int(option.get("inPlayIndex",-1));array=players[me].get("active") or [] if area==4 else players[me].get("bench") or [] if area==5 else []
            return array[index]
        except Exception:return None

    def signature(raw, option):
        return ":".join(map(str,(TYPE_NAME.get(int(option.get("type",-1)),""),cid(source(raw,option)),cid(target(raw,option)),int(option.get("attackId") or 0))))

    def bins(raw):
        current=raw.get("current") or {};me=int(current.get("yourIndex") or 0);players=current.get("players") or [{},{}]
        if len(players)<2:return "0","0"
        mine,theirs=players[me],players[1-me]
        prize=len(theirs.get("prize") or [])-len(mine.get("prize") or [])
        hand=int(mine.get("handCount") or len(mine.get("hand") or []))-int(theirs.get("handCount") or len(theirs.get("hand") or []))
        return ("A" if prize>=2 else "a" if prize==1 else "0" if prize==0 else "b" if prize==-1 else "B", "H" if hand>=3 else "h" if hand>0 else "0" if hand==0 else "l" if hand>=-2 else "L")

    def keys(raw):
        current=raw.get("current") or {};me=int(current.get("yourIndex") or 0);players=current.get("players") or [{},{}]
        if len(players)<2:return []
        mine,theirs=players[me],players[1-me];turn=int(current.get("turn") or 0);phase="e" if turn<=3 else "m" if turn<=8 else "l"
        ma=cid((mine.get("active") or [None])[0]);oa=cid((theirs.get("active") or [None])[0])
        mb=",".join(map(str,sorted(cid(x) for x in mine.get("bench") or [] if cid(x))));ob=",".join(map(str,sorted(cid(x) for x in theirs.get("bench") or [] if cid(x))))
        types=",".join(sorted({TYPE_NAME.get(int(x.get("type",-1)),"") for x in (raw.get("select") or {}).get("option") or []}));pb,hb=bins(raw)
        return [("E",f"{phase}|{pb}|{hb}|{ma}|{oa}|{mb}|{ob}|{types}"),("C",f"{phase}|{pb}|{ma}|{oa}|{types}"),("L",f"{phase}|{pb}|{ma}|{oa}")]

    def snapshot():
        saved={}
        for key,value in module.__dict__.items():
            if key.endswith("MEM") and isinstance(value,(dict,list,set)):saved[key]=copy.deepcopy(value)
        return saved

    def restore(saved):
        for key,value in saved.items():
            current=module.__dict__.get(key)
            if isinstance(current,dict):current.clear();current.update(copy.deepcopy(value))
            elif isinstance(current,list):current[:]=copy.deepcopy(value)
            elif isinstance(current,set):current.clear();current.update(copy.deepcopy(value))

    def record(raw, output):
        if not output or len(output)!=1:return
        try:
            option=(raw.get("select") or {}).get("option")[output[0]];card=source(raw,option);memory_state=module.__dict__.get("_XMEM",{});card_id=cid(card);typ=int(option.get("type",-1))
            if typ==10:
                if card_id==module.X["TEAL"]:memory_state["teal_used"]=int(memory_state.get("teal_used",0))+1
                elif card_id==module.X["KANGA"]:memory_state["kanga_used"]=True
                elif card_id==module.X["FEZ"]:memory_state["fez_used"]=True
            if typ==7 and card_id==module.X["MEOWTH"]:memory_state["supporter_fetch"]=True
        except Exception:pass

    old_agent=module.agent
    def agent(raw):
        if raw.get("select") is None:return old_agent(raw)
        before=snapshot();base=old_agent(raw);after=snapshot();select=raw.get("select") or {};options=select.get("option") or []
        if int(select.get("context",-1))!=0 or not isinstance(base,list) or len(base)!=1 or not 0<=int(base[0])<len(options):return base
        hit=None;level=None
        for current_level,key in keys(raw):
            candidate=memory.get(current_level+"|"+key)
            if candidate:hit=candidate;level=current_level;break
        if not hit:return base
        stats["hits"]+=1;wanted=next((index for index,option in enumerate(options) if signature(raw,option)==hit["action"]),None)
        if wanted is None:stats["missing_action"]+=1;return base
        if wanted==base[0]:return base
        base_type=int(options[base[0]].get("type",-1));wanted_type=int(options[wanted].get("type",-1))
        # Loose states intervene only on clear tempo leaks.  Exact/coarse states
        # may also replace one productive action with another replay-proven one.
        allow=(level in {"E","C"} and wanted_type in {7,8,9,10,12,13}) or (level=="L" and base_type==14 and wanted_type in {9,10,13})
        if not allow:return base
        stats["overrides"]+=1;restore(before);record(raw,[wanted]);return [wanted]

    module.agent=agent;module.MEMORY_PROFILE=profile;module.MEMORY_STATS=stats
    return module
