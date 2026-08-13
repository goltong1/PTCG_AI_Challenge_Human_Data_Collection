"""Opponent-aware multi-step rollout specialists for exact weak matchups."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict

from cg.api import (
    AreaType,
    OptionType,
    SelectContext,
    search_begin,
    search_end,
    search_release,
    search_step,
    to_observation_class,
)


def load_module(path, tag):
    path = os.path.abspath(path)
    root = os.path.dirname(path)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    name = "_tera_v19_rollout_" + tag + "_" + hashlib.sha1(path.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = os.getcwd()
    try:
        os.chdir(root)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


def build(root, profile):
    base_file = {
        "dragapult": "policy_lucario.py",
        "lucario": "policy_lucario.py",
        "alakazam": "policy_special_alakazam.py",
        "crustle": "policy_crustle.py",
    }[profile]
    own = load_module(os.path.join(root, base_file), "own_" + profile)
    opponent = load_module(os.path.join(root, "sim_opponents", profile, "opponent_model.py"), "opp_" + profile)
    init = {"current": None, "logs": [], "select": None, "step": 0}
    own_deck = list(own.agent(init))
    opponent_deck = list(opponent.agent(init))
    if len(own_deck) != 60 or len(opponent_deck) != 60:
        raise ValueError("rollout specialist requires two valid 60-card decks")

    if profile == "dragapult":
        old_damage = own._x_damage

        def dragapult_damage(obs, pokemon, attack_id):
            value = old_damage(obs, pokemon, attack_id)
            try:
                active = own._x_pl(obs, False).active[0]
                if pokemon and int(pokemon.id) == own.X["CLEF"] and int(attack_id or 0) == 371 and int(active.id) in {119, 120, 121, 235}:
                    return value * 2
            except Exception:
                pass
            return value

        own._x_damage = dragapult_damage

    last_turn = {"value": -1}
    stats = {"calls": 0, "overrides": 0, "errors": 0, "robust_rejects": 0}
    horizon = int(os.environ.get("TERA_ROLLOUT_HORIZON", "40"))
    max_candidates = int(os.environ.get("TERA_ROLLOUT_CANDIDATES", "4"))
    margin = float(os.environ.get("TERA_ROLLOUT_MARGIN", "50000"))
    particles = int(os.environ.get("TERA_ROLLOUT_PARTICLES", "3"))

    def snapshot(module):
        saved = {}
        for key, value in module.__dict__.items():
            if key.endswith("MEM") and isinstance(value, (dict, set, list)):
                saved[key] = copy.deepcopy(value)
        return saved

    def restore(module, saved):
        for key, value in saved.items():
            current = module.__dict__.get(key)
            if isinstance(current, dict):
                current.clear(); current.update(copy.deepcopy(value))
            elif isinstance(current, set):
                current.clear(); current.update(copy.deepcopy(value))
            elif isinstance(current, list):
                current[:] = copy.deepcopy(value)

    def record(obs, output):
        if not output or len(output) != 1:
            return
        try:
            option = obs.select.option[output[0]]
            card = own._source(obs, option)
            memory = own.__dict__.get("_XMEM", {})
            if option.type == OptionType.ABILITY and card:
                if int(card.id) == own.X["TEAL"]:
                    memory["teal_used"] = int(memory.get("teal_used", 0)) + 1
                elif int(card.id) == own.X["KANGA"]:
                    memory["kanga_used"] = True
                elif int(card.id) == own.X["FEZ"]:
                    memory["fez_used"] = True
            if option.type == OptionType.PLAY and card:
                if int(card.id) == own.X["MEOWTH"]:
                    memory["supporter_fetch"] = True
                elif int(card.id) == own.X["ESWITCH"]:
                    memory["move"] = own._x_transfer_plan(obs)
        except Exception:
            pass

    def add_pokemon(counter, pokemon):
        if pokemon is None:
            return
        counter[int(pokemon.id)] += 1
        for card in pokemon.energyCards or []:
            counter[int(card.id)] += 1
        for card in pokemon.tools or []:
            counter[int(card.id)] += 1
        for card in pokemon.preEvolution or []:
            counter[int(card.id)] += 1

    def unknown(full_deck, player, state, own_side, sample):
        counter = Counter(map(int, full_deck))
        if own_side:
            for card in player.hand or []:
                counter[int(card.id)] -= 1
        for pokemon in list(player.active or []) + list(player.bench or []):
            add_pokemon(counter, pokemon)
        for card in player.discard or []:
            counter[int(card.id)] -= 1
        for card in getattr(player, "lostZone", None) or []:
            counter[int(card.id)] -= 1
        remaining = []
        for card_id, count in counter.items():
            remaining.extend([card_id] * max(0, count))
        need = int(player.deckCount or 0) + len(player.prize or []) + (0 if own_side else int(player.handCount or 0))
        if len(remaining) < need:
            remaining.extend(list(map(int, full_deck)) * ((need - len(remaining)) // 60 + 1))
        seed = int(state.turn or 0) * 1009 + int(state.turnActionCount or 0) * 917 + sum(remaining[:20]) + sample * 104729
        random.Random(seed).shuffle(remaining)
        deck_count = int(player.deckCount or 0)
        prize_count = len(player.prize or [])
        if own_side:
            return remaining[:deck_count], remaining[deck_count:deck_count + prize_count], [], []
        hand_count = int(player.handCount or 0)
        return (
            remaining[hand_count:hand_count + deck_count],
            remaining[hand_count + deck_count:hand_count + deck_count + prize_count],
            remaining[:hand_count],
            [],
        )

    def hidden(obs, sample):
        state = obs.current
        me = int(state.yourIndex)
        mine = state.players[me]
        theirs = state.players[1 - me]
        own_hidden = unknown(own_deck, mine, state, True, sample)
        opp_hidden = unknown(opponent_deck, theirs, state, False, sample)
        return own_hidden[0], own_hidden[1], opp_hidden[0], opp_hidden[1], opp_hidden[2], opp_hidden[3]

    def ready(pokemon):
        try:
            return own._x_ready_damage(_rollout_obs["value"], pokemon)
        except Exception:
            return 0

    _rollout_obs = {"value": None}

    def value(obs, me):
        state = obs.current
        if state is None:
            return -1e12
        if int(state.result) >= 0:
            if int(state.result) == 2:
                return 0.0
            return 1e9 if int(state.result) == me else -1e9
        _rollout_obs["value"] = obs
        mine = state.players[me]
        theirs = state.players[1 - me]
        own_board = [p for p in list(mine.active or []) + list(mine.bench or []) if p]
        opp_board = [p for p in list(theirs.active or []) + list(theirs.bench or []) if p]
        result = (len(theirs.prize or []) - len(mine.prize or [])) * 210000.0
        for pokemon in own_board:
            damage = ready(pokemon)
            result += int(pokemon.hp or 0) * 34 + len(pokemon.energyCards or []) * 6200 + damage * 190
            if mine.active and mine.active[0] is pokemon and damage > 0:
                result += 42000
        for pokemon in opp_board:
            result -= int(pokemon.hp or 0) * 20
            result += max(0, int(pokemon.maxHp or 0) - int(pokemon.hp or 0)) * 105
        ids = {int(p.id) for p in own_board}
        result += (26000 if 184 in ids else 0) + (22000 if 96 in ids else 0)
        if profile in {"dragapult", "lucario"}:
            result += (70000 if 272 in ids else 0) + (45000 if 756 in ids else 0)
        elif profile == "crustle":
            result += (90000 if 117 in ids else 0) + (26000 if 31 in ids else 0)
            result -= 18000 * sum(int(p.hp or 0) <= 120 for p in mine.bench or [] if p)
        elif profile == "alakazam":
            result += (38000 if 117 in ids else 0) + (18000 if 31 in ids else 0)
            result -= max(0, int(mine.handCount or 0) - 4) * 7000
        return result

    def candidates(obs, base_output):
        if profile == "alakazam":
            base_index=int(base_output[0]);base_option=obs.select.option[base_index];base_card=own._source(obs,base_option)
            if int(obs.current.turn or 0)>3 or base_option.type!=OptionType.ABILITY or int(getattr(base_card,"id",0) or 0)!=96:return [base_index]
            preferred=[]
            for index,option in enumerate(obs.select.option):
                energy=own._source(obs,option);target=own._target(obs,option)
                if option.type==OptionType.ATTACH and int(option.inPlayArea or -1)==int(AreaType.BENCH) and int(getattr(energy,"id",0) or 0)==16 and int(getattr(target,"id",0) or 0)==96:
                    preferred.append(index)
            return [base_index]+preferred[:1]
        ranked = []
        seen = set()
        for index, option in enumerate(obs.select.option):
            card = own._source(obs, option)
            key = (int(option.type), int(getattr(card, "id", 0) or 0), int(option.attackId or 0), int(option.inPlayArea or -1), int(option.inPlayIndex or -1))
            if key in seen:
                continue
            seen.add(key)
            priority = {13: 90000, 10: 70000, 9: 65000, 8: 56000, 12: 52000, 7: 42000, 14: -10000}.get(int(option.type), 0)
            try:
                priority += float(own._main_score(obs, option))
            except Exception:
                pass
            ranked.append((priority, index))
        ranked.sort(reverse=True)
        output = [int(base_output[0])]
        for _, index in ranked:
            if index not in output:
                output.append(index)
            if len(output) >= max_candidates:
                break
        return output

    def opponent_action(raw):
        controller=getattr(opponent,"_controller",None);owner=opponent
        if controller is None:
            for engine in (getattr(opponent,"ENGINES",{}) or {}).values():
                for candidate in (engine,getattr(engine,"base",None)):
                    if candidate is not None and getattr(candidate,"_controller",None) is not None:
                        owner=candidate;controller=candidate._controller;break
                if controller is not None:break
        if controller is None:return opponent.agent(raw)
        o=to_observation_class(raw);turn=int(o.current.turn or 0)
        try:controller.recognize(o)
        except Exception:pass
        action=controller.validate(controller.call_base(raw,turn,disable_search=True),o.select)
        for name in ("_human_order_patch","_ultrasafe_drag_patch","_fast_backup","_alak_line"):
            patch=getattr(owner,name,None) or getattr(opponent,name,None)
            if patch is not None:
                try:action=patch(raw,action)
                except Exception:pass
        return action

    def branch(obs, root_index, me, pre_memory, sample):
        search_id = None
        restore(own, pre_memory)
        record(obs, [root_index])
        try:
            opponent.agent(init)
            root = search_begin(obs, *hidden(obs, sample))
            search_id = root.searchId
            stepped = search_step(search_id, [root_index])
            search_release(search_id)
            search_id = stepped.searchId
            for _ in range(horizon):
                current = stepped.observation
                if current.current is None or int(current.current.result) >= 0:
                    break
                actor = int(current.current.yourIndex)
                raw = asdict(current)
                action = old_agent(raw) if actor == me else opponent_action(raw)
                if not action:
                    break
                next_step = search_step(search_id, action)
                search_release(search_id)
                stepped = next_step
                search_id = stepped.searchId
            return value(stepped.observation, me)
        except Exception:
            stats["errors"] += 1
            return None
        finally:
            if search_id is not None:
                try:
                    search_release(search_id)
                except Exception:
                    pass
            try:
                search_end()
            except Exception:
                pass
            restore(own, pre_memory)

    old_agent = own.agent

    def agent(observation):
        if observation.get("select") is None:
            last_turn["value"] = -1
            opponent.agent(init)
            return old_agent(observation)
        pre_memory = snapshot(own)
        base_output = old_agent(observation)
        post_memory = snapshot(own)
        try:
            obs = to_observation_class(observation)
            if obs.select is None or obs.select.context != SelectContext.MAIN or len(base_output) != 1:
                return base_output
            turn = int(obs.current.turn or 0)
            if last_turn["value"] == turn or len(obs.select.option) < 2:
                return base_output
            choices = candidates(obs, base_output)
            if len(choices) < 2:
                return base_output
            if os.environ.get("TERA_ONLINE_DISABLE")=="1":return base_output
            last_turn["value"] = turn
            stats["calls"] += 1
            values = {index: [branch(obs,index,int(obs.current.yourIndex),pre_memory,sample) for sample in range(particles)] for index in choices}
            if any(any(value is None for value in bucket) for bucket in values.values()):
                stats["robust_rejects"]+=1;restore(own,post_memory);return base_output
            base_index = int(base_output[0])
            best_index=max(values,key=lambda index:statistics.fmean(values[index]))
            diffs=[a-b for a,b in zip(values[best_index],values[base_index])]
            mean=statistics.fmean(diffs);sd=statistics.pstdev(diffs) if len(diffs)>1 else 0.0;lcb=mean-.85*sd
            if best_index == base_index or mean < margin or lcb < margin or min(diffs)<0:
                stats["robust_rejects"]+=1
                restore(own, post_memory)
                return base_output
            stats["overrides"] += 1
            restore(own, pre_memory)
            record(obs, [best_index])
            return [best_index]
        except Exception:
            restore(own, post_memory)
            return base_output

    own.agent = agent
    own.ROLLOUT_PROFILE = profile
    own.ROLLOUT_STATS = stats
    return own
