"""Dedicated weak-match specialists distilled from exact cross-play replays."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys


def load_base(root, tag):
    path = os.path.join(root, f"policy_{tag}.py")
    name = "_tera_v19_special_base_" + tag + "_" + hashlib.sha1(path.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(root, base_tag, profile):
    module = load_base(root, base_tag)
    install(module.__dict__, profile)
    return module


def install(ns, profile):
    X = ns["X"]
    OptionType = ns["OptionType"]
    SelectContext = ns["SelectContext"]
    CardType = ns["CardType"]
    source = ns["_source"]
    target_of = ns["_target"]
    old_role = ns["_x_role"]
    old_search = ns["_x_search_score"]
    old_attach = ns["_x_attach_score"]
    old_main = ns["_x_main"]
    old_context = ns["_x_context"]
    old_damage = ns["_x_damage"]
    cards = ns["CARDS"]
    state = {"boss": None}

    if profile in {"dragapult", "lucario"}:
        primary = X["CLEF"]
        secondary = X["TEAL"]
        opponent_ids = {119, 120, 121, 235} if profile == "dragapult" else {333, 677, 678}
        denial = {119, 120, 235} if profile == "dragapult" else {333, 677}
    elif profile == "alakazam":
        primary = X["TEAL"]
        secondary = X["CHIYU"]
        opponent_ids = {245, 741, 742, 743}
        denial = {741, 742}
    else:
        primary = X["CORNER"]
        secondary = X["CHIYU"]
        opponent_ids = {343, 344, 345, 117}
        denial = {344}

    def board(obs, own=True):
        return ns["_x_board"](obs, own)

    def player(obs, own=True):
        return ns["_x_pl"](obs, own)

    def has(obs, cid):
        return any(int(p.id) == int(cid) for p in board(obs, True))

    def find_pokemon(obs, cid):
        return next((p for p in board(obs, True) if int(p.id) == int(cid)), None)

    def count(obs, cid):
        return sum(int(p.id) == int(cid) for p in board(obs, True))

    def matchup(_obs):
        return profile

    def primary_fn(_obs):
        return primary

    def tech_fn(_obs):
        return primary

    def secondary_fn(_obs):
        return secondary

    ns["_x_matchup"] = matchup
    ns["_x_primary"] = primary_fn
    ns["_x_tech_primary"] = tech_fn
    ns["_x_secondary"] = secondary_fn

    def damage(obs, pokemon, attack_id):
        value = old_damage(obs, pokemon, attack_id)
        if profile == "dragapult" and pokemon is not None and int(pokemon.id) == X["CLEF"] and int(attack_id or 0) == 371:
            opponent = player(obs, False).active[0] if player(obs, False).active else None
            if opponent is not None and int(opponent.id) in opponent_ids:
                return value * 2
        return value

    ns["_x_damage"] = damage

    def role(obs, card, instance=False):
        value = old_role(obs, card, instance)
        cid = int(getattr(card, "id", 0) or 0)
        turn = int(obs.current.turn or 0)
        if cid == primary:
            value += 18000 if not has(obs, cid) else 2200
        if cid == X["LATIAS"] and not has(obs, cid):
            value += 8500
        if cid == X["TEAL"]:
            cap = 2 if profile in {"dragapult", "lucario", "alakazam"} else 1
            value += 6500 if count(obs, cid) < cap else -500
        if cid == X["AREA"] and profile in {"dragapult", "lucario"}:
            value += 6000 if ns["_x_stadium"](obs) != cid and has(obs, X["TEAL"]) else -700
        if cid == int(X.get("PRISM_CARD", X.get("PRISM", 16))) and has(obs, primary):
            value += 10000
        if profile == "alakazam" and cid in {X["XERO"], X["STAMP"]}:
            value += 5000 if player(obs, False).handCount >= 5 else 500
        if profile == "crustle" and cid in {X["KANGA"], X["MEOWTH"], X["FEZ"], X["MUNKI"], X["PECH"]}:
            value -= 6500
        if profile in {"dragapult", "lucario"} and cid in {X["PECH"], X["MUNKI"], X["CHIYU"]} and turn <= 7:
            value -= 5500
        return value

    def search_score(obs, card, selected=None):
        value = old_search(obs, card, selected)
        cid = int(getattr(card, "id", 0) or 0)
        if cid == primary and not has(obs, cid):
            value += 24000
        if cid == X["LATIAS"] and not has(obs, cid):
            value += 11500
        if cid == X["TEAL"]:
            cap = 2 if profile in {"dragapult", "lucario", "alakazam"} else 1
            value += 8500 if count(obs, cid) < cap else -1200
        if cid == int(X.get("PRISM_CARD", X.get("PRISM", 16))) and has(obs, primary):
            value += 12500
        if profile == "alakazam" and cid in {X["XERO"], X["STAMP"], X["BOSS"]}:
            value += 6000
        return value

    def attach_score(obs, option):
        value = old_attach(obs, option)
        energy = source(obs, option)
        target = target_of(obs, option)
        if energy is None or target is None:
            return value
        eid = int(energy.id)
        if int(target.id) == primary:
            value += 16000
            if eid == int(X.get("PRISM_CARD", X.get("PRISM", 16))) and primary in {X["CLEF"], X["CORNER"]}:
                value += 10000
            if eid == X["G"] and primary == X["TEAL"]:
                value += 8000
        elif eid == int(X.get("PRISM_CARD", X.get("PRISM", 16))) and primary in {X["CLEF"], X["CORNER"]}:
            value -= 14000
        return value

    ns["_x_role"] = role
    ns["_x_search_score"] = search_score
    ns["_x_attach_score"] = attach_score

    def find_option(obs, option_type, cid=None):
        for index, option in enumerate(obs.select.option):
            if option.type != option_type:
                continue
            card = source(obs, option)
            if cid is None or (card is not None and int(card.id) == int(cid)):
                return index
        return None

    def best_attach(obs):
        values = [(attach_score(obs, option), index) for index, option in enumerate(obs.select.option) if option.type == OptionType.ATTACH]
        return max(values) if values else None

    def best_attack(obs, active):
        attacks = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.ATTACK]
        if not attacks:
            return None
        return max(attacks, key=lambda i: damage(obs, active, obs.select.option[i].attackId))

    def free_action(obs):
        # Free acceleration/draw cannot reduce the current attack and improves
        # both the present hand and the follow-up attacker.
        for cid in (X["TEAL"], X["KANGA"], X["FEZ"]):
            index = find_option(obs, OptionType.ABILITY, cid)
            if index is None:
                continue
            memory = ns.get("_XMEM", {})
            if cid == X["TEAL"] and ns["_x_hcount"](obs, X["G"]) <= 0:
                continue
            if cid == X["KANGA"] and memory.get("kanga_used"):
                continue
            if cid == X["FEZ"] and memory.get("fez_used"):
                continue
            return index
        return None

    def play_primary_or_search(obs):
        if has(obs, primary):
            return None
        direct = find_option(obs, OptionType.PLAY, primary)
        if direct is not None:
            return direct
        order = [X["TERA_ORB"], X["ULTRA"]] if primary in {X["TEAL"], X["CORNER"]} else [X["ULTRA"]]
        for cid in order:
            index = find_option(obs, OptionType.PLAY, cid)
            if index is not None:
                return index
        return None

    def safe_body_play(obs):
        values = []
        for index, option in enumerate(obs.select.option):
            if option.type != OptionType.PLAY:
                continue
            card = source(obs, option)
            data = cards.get(int(getattr(card, "id", 0) or 0))
            if not data or data.cardType != CardType.POKEMON or not data.basic or int(data.hp or 0) < 170:
                continue
            if int(card.id) in {X["PECH"], X["MUNKI"], X["CHIYU"]}:
                continue
            values.append((role(obs, card), index))
        return max(values)[1] if values else None

    def denial_boss(obs, ready):
        for opponent in player(obs, False).bench or []:
            if opponent and int(opponent.id) in denial and ready >= int(opponent.hp or 0):
                boss = find_option(obs, OptionType.PLAY, X["BOSS"])
                if boss is not None:
                    state["boss"] = int(opponent.id)
                    return boss
        return None

    def main_clefairy(obs):
        active = player(obs, True).active[0] if player(obs, True).active else None
        ready = ns["_x_ready_damage"](obs, active)
        free = free_action(obs)
        if free is not None:
            return [free]
        if ready > 0:
            attach = best_attach(obs)
            if attach and attach[0] >= 4500:
                return [attach[1]]
            boss = denial_boss(obs, ready)
            if boss is not None:
                return [boss]
            attack = best_attack(obs, active)
            if attack is not None:
                return [attack]
        setup = play_primary_or_search(obs)
        if setup is not None:
            return [setup]
        clefairy = find_pokemon(obs, X["CLEF"])
        if clefairy is not None and not has(obs, X["LATIAS"]):
            latias = find_option(obs, OptionType.PLAY, X["LATIAS"])
            if latias is not None:
                return [latias]
        if count(obs, X["TEAL"]) < 1:
            teal = find_option(obs, OptionType.PLAY, X["TEAL"])
            if teal is not None:
                return [teal]
        if ns["_x_stadium"](obs) != X["AREA"] and has(obs, X["TEAL"]):
            area = find_option(obs, OptionType.PLAY, X["AREA"])
            if area is not None:
                return [area]
        attach = best_attach(obs)
        if attach and attach[0] >= 2200:
            return [attach[1]]
        plan = ns["_x_transfer_plan"](obs)
        if plan and int(plan[7]) == X["CLEF"]:
            switch = find_option(obs, OptionType.PLAY, X["ESWITCH"])
            if switch is not None:
                ns.get("_XMEM", {})["move"] = plan
                return [switch]
        if clefairy is not None and ns["_x_ready_damage"](obs, clefairy) > 0:
            opponent = player(obs, False).active[0] if player(obs, False).active else None
            damage_now = damage(obs, clefairy, 371)
            if opponent is not None and damage_now < int(opponent.hp or 0):
                body = safe_body_play(obs)
                if body is not None:
                    return [body]
            if active is not clefairy and has(obs, X["LATIAS"]):
                retreat = find_option(obs, OptionType.RETREAT)
                if retreat is not None:
                    return [retreat]
        if ready > 0:
            attack = best_attack(obs, active)
            if attack is not None:
                return [attack]
        return old_main(obs)

    def main_alakazam(obs):
        active = player(obs, True).active[0] if player(obs, True).active else None
        ready = ns["_x_ready_damage"](obs, active)
        free = free_action(obs)
        if free is not None:
            return [free]
        if ready > 0:
            # Powerful Hand scales directly with hand size; reducing it to 3
            # before attacking removes 40-140 incoming damage counters.
            if player(obs, False).handCount >= 5:
                for cid in (X["XERO"], X["STAMP"]):
                    disrupt = find_option(obs, OptionType.PLAY, cid)
                    if disrupt is not None:
                        return [disrupt]
            boss = denial_boss(obs, ready)
            if boss is not None:
                return [boss]
            attach = best_attach(obs)
            if attach and attach[0] >= 5000:
                return [attach[1]]
            attack = best_attack(obs, active)
            if attack is not None:
                return [attack]
        setup = play_primary_or_search(obs)
        if setup is not None:
            return [setup]
        if count(obs, X["TEAL"]) < 2:
            teal = find_option(obs, OptionType.PLAY, X["TEAL"])
            if teal is not None:
                return [teal]
        attach = best_attach(obs)
        if attach and attach[0] >= 2200:
            return [attach[1]]
        if ready > 0:
            attack = best_attack(obs, active)
            if attack is not None:
                return [attack]
        return old_main(obs)

    def main_crustle(obs):
        active = player(obs, True).active[0] if player(obs, True).active else None
        corner = find_pokemon(obs, X["CORNER"])
        free = free_action(obs)
        if free is not None:
            return [free]
        if active is not None and int(active.id) == X["CORNER"] and ns["_x_ready_damage"](obs, active) > 0:
            attack = best_attack(obs, active)
            if attack is not None:
                return [attack]
        setup = play_primary_or_search(obs)
        if setup is not None:
            return [setup]
        corner = find_pokemon(obs, X["CORNER"])
        if corner is not None and not has(obs, X["LATIAS"]):
            latias = find_option(obs, OptionType.PLAY, X["LATIAS"])
            if latias is not None:
                return [latias]
        if count(obs, X["TEAL"]) < 1:
            teal = find_option(obs, OptionType.PLAY, X["TEAL"])
            if teal is not None:
                return [teal]
        attach = best_attach(obs)
        if attach and attach[0] >= 1800:
            return [attach[1]]
        plan = ns["_x_transfer_plan"](obs)
        if plan and int(plan[7]) == X["CORNER"]:
            switch = find_option(obs, OptionType.PLAY, X["ESWITCH"])
            if switch is not None:
                ns.get("_XMEM", {})["move"] = plan
                return [switch]
        if corner is not None and ns["_x_ready_damage"](obs, corner) > 0 and active is not corner and has(obs, X["LATIAS"]):
            retreat = find_option(obs, OptionType.RETREAT)
            if retreat is not None:
                return [retreat]
        ready = ns["_x_ready_damage"](obs, active)
        if ready > 0:
            attack = best_attack(obs, active)
            if attack is not None:
                return [attack]
        return old_main(obs)

    def main(obs):
        if profile in {"dragapult", "lucario"}:
            return main_clefairy(obs)
        if profile == "alakazam":
            return main_alakazam(obs)
        return main_crustle(obs)

    def context(obs):
        if state["boss"] is not None and obs.select.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            for index, option in enumerate(obs.select.option):
                card = source(obs, option)
                if card is not None and int(card.id) == state["boss"]:
                    state["boss"] = None
                    return [index]
        return old_context(obs)

    ns["_x_main"] = main
    ns["_x_context"] = context

