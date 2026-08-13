"""Loss-replay residuals distilled from exact 200-game opponent cross-play.

The installer is intentionally small and deterministic.  It leaves the mature
base policy intact, but makes the replay-separated decisions explicit:
opening safety, one matchup attacker, typed-energy conservation, immediate
attacks, and evolution-denial gusts.
"""

from __future__ import annotations


PROFILES = {
    "dragapult": {
        "mode": "kangaskhan_then_clefairy",
        "target": "KANGA",
        "tech": "CLEF",
        "secondary": "TEAL",
        "denial": {119, 120, 235},
        "target_bonus": 16000,
        "tech_bonus": 10500,
        "latias_bonus": 7000,
        "teal_cap": 1,
    },
    "alakazam": {
        "mode": "single_rush",
        "target": "TEAL",
        "tech": "TEAL",
        "secondary": "KANGA",
        "denial": {741, 742},
        "target_bonus": 15000,
        "latias_bonus": 6500,
        "teal_cap": 2,
    },
    "marnie": {
        "mode": "single_rush",
        "target": "TEAL",
        "tech": "TEAL",
        "secondary": "CORNER",
        "denial": {646, 647},
        "target_bonus": 15000,
        "latias_bonus": 6000,
        "teal_cap": 2,
    },
    "archaludon": {
        "mode": "single_rush",
        "target": "CORNER",
        "tech": "CORNER",
        "secondary": "TEAL",
        "denial": {169, 190},
        "target_bonus": 17500,
        "latias_bonus": 6500,
        "teal_cap": 1,
    },
    "crustle": {
        "mode": "single_rush",
        "target": "CORNER",
        "tech": "CORNER",
        "secondary": "CHIYU",
        "denial": {344},
        "target_bonus": 19000,
        "latias_bonus": 6500,
        "teal_cap": 1,
        "bench_cap": 5,
    },
    "lucario": {
        "mode": "single_rush",
        "target": "CLEF",
        "tech": "CLEF",
        "secondary": "TEAL",
        "denial": {677},
        "target_bonus": 18000,
        "latias_bonus": 7000,
        "kanga_bonus": 4500,
        "teal_cap": 1,
    },
}


def install(ns, profile):
    """Install a profile into one dynamically loaded policy module."""
    X = ns["X"]
    OptionType = ns["OptionType"]
    SelectContext = ns["SelectContext"]
    AreaType = ns["AreaType"]
    CardType = ns["CardType"]
    source = ns["_source"]

    old_context = ns["_x_context"]
    state = {"boss_target": None, "turn": -1}

    def setup_context(obs):
        sel = obs.select
        if sel.context not in {SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON}:
            return None
        active = sel.context == SelectContext.SETUP_ACTIVE_POKEMON
        minimum = max(0, int(sel.minCount))
        maximum = min(len(sel.option), int(sel.maxCount))
        rank = {
            X["KANGA"]: 100,
            X["TEAL"]: 94,
            X["LATIAS"]: 90,
            X["CORNER"]: 84,
            X["WELL"]: 82,
            X["CLEF"]: 80,
            X["FEZ"]: 76,
            X["MEOWTH"]: 60,
            X["MUNKI"]: 45,
            X["CHIYU"]: 44,
            X["PECH"]: 35,
        }
        values = []
        for index, option in enumerate(sel.option):
            card = source(obs, option)
            cid = int(getattr(card, "id", 0) or 0)
            hp = int(getattr(card, "hp", 0) or 0)
            values.append((rank.get(cid, hp / 10), hp, -index, index, cid))
        values.sort(reverse=True)
        count = minimum if active else maximum
        if not active and minimum == 0:
            # Bench only robust engines at blind setup.  Small utility Pokémon
            # caused extra Phantom Dive and single-prize liabilities in losses.
            values = [z for z in values if z[4] in {X["KANGA"], X["TEAL"], X["LATIAS"], X["CORNER"], X["WELL"], X["CLEF"], X["FEZ"]}]
            count = min(maximum, len(values))
        return [z[3] for z in values[:count]]

    if profile == "generic_setup":
        def context(obs):
            result = setup_context(obs)
            return old_context(obs) if result is None else result
        ns["_x_context"] = context
        return

    cfg = PROFILES[profile]
    old_role = ns["_x_role"]
    old_search = ns["_x_search_score"]
    old_attach = ns["_x_attach_score"]
    old_main = ns["_x_main"]
    old_transfer = ns["_x_transfer_plan"]
    attacks = ns["ATTACKS"]
    cards = ns["CARDS"]

    target_id = X[cfg["target"]]
    tech_id = X[cfg["tech"]]
    secondary_id = X[cfg["secondary"]]
    any_energy = 10
    prism_card = int(X.get("PRISM_CARD", X.get("PRISM", 16)))

    def board(obs, own=True):
        return ns["_x_board"](obs, own)

    def has(obs, cid):
        return any(int(p.id) == int(cid) for p in board(obs, True))

    def pokemon(obs, cid):
        return next((p for p in board(obs, True) if int(p.id) == int(cid)), None)

    def energy_ids(p):
        return list(getattr(p, "energies", []) or [])

    def can_pay(p, attack_id, energies=None):
        attack = attacks.get(int(attack_id or 0))
        pool = list(energy_ids(p) if energies is None else energies)
        if attack is None:
            return False
        for requirement in attack.energies:
            requirement = int(requirement)
            if requirement == 0:
                if not pool:
                    return False
                pool.pop(0)
                continue
            index = next((i for i, energy in enumerate(pool) if int(energy) in {requirement, any_energy}), None)
            if index is None:
                return False
            pool.pop(index)
        return True

    def energy_progress(p, energies=None):
        energy = list(energy_ids(p) if energies is None else energies)
        needs = {
            X["KANGA"]: (0, 3), X["TEAL"]: (0, 3), X["WELL"]: (1, 3),
            X["CORNER"]: (1, 3), X["CLEF"]: (1, 2), X["CHIYU"]: (1, 2),
            X["PECH"]: (1, 2), X["MUNKI"]: (1, 2), X["FEZ"]: (0, 3),
            X["LATIAS"]: (2, 3), X["MEOWTH"]: (0, 3),
        }
        typed, total = needs.get(int(p.id), (0, 99))
        universal = sum(int(e) == any_energy for e in energy)
        return 5 * max(0, typed - universal) + max(0, total - len(energy))

    ns["_x_can_pay"] = can_pay
    ns["_x_energy_progress"] = energy_progress

    def dynamic_target(obs):
        if cfg["mode"] == "kangaskhan_then_clefairy":
            kanga = pokemon(obs, X["KANGA"])
            turn = int(obs.current.turn or 0)
            if kanga is None or ns["_x_ready_damage"](obs, kanga) <= 0 or turn <= 4:
                return X["KANGA"]
            return X["CLEF"]
        return target_id

    def primary(obs):
        return dynamic_target(obs)

    def tech_primary(obs):
        return tech_id

    def secondary(obs):
        return secondary_id

    ns["_x_primary"] = primary
    ns["_x_tech_primary"] = tech_primary
    ns["_x_secondary"] = secondary

    def role(obs, card, instance=False):
        value = old_role(obs, card, instance)
        cid = int(getattr(card, "id", 0) or 0)
        wanted = dynamic_target(obs)
        if cid == wanted and not has(obs, cid):
            value += cfg["target_bonus"]
        if cid == tech_id and not has(obs, cid) and wanted != tech_id:
            value += cfg.get("tech_bonus", 0)
        if cid == X["LATIAS"] and not has(obs, cid):
            value += cfg["latias_bonus"]
        if cid == X["KANGA"] and not has(obs, cid):
            value += cfg.get("kanga_bonus", 0)
        if cid == X["TEAL"] and sum(p.id == X["TEAL"] for p in board(obs, True)) < cfg["teal_cap"]:
            value += 4200
        if cid == prism_card and has(obs, wanted):
            value += 7000
        if cid in {X["PECH"], X["MUNKI"], X["CHIYU"], X["MEOWTH"]} and cid not in {wanted, secondary_id}:
            value -= 5500
        cap = cfg.get("bench_cap")
        if cap is not None and len(board(obs, True)) >= cap and cid not in {wanted, X["LATIAS"], X["TEAL"]}:
            value -= 12000
        return value

    def search_score(obs, card, selected=None):
        value = old_search(obs, card, selected)
        cid = int(getattr(card, "id", 0) or 0)
        wanted = dynamic_target(obs)
        if cid == wanted and not has(obs, cid):
            value += cfg["target_bonus"] * 1.4
        if cid == tech_id and not has(obs, cid) and wanted != tech_id:
            value += cfg.get("tech_bonus", 0) * 1.25
        if cid == X["LATIAS"] and not has(obs, cid):
            value += cfg["latias_bonus"] * 1.25
        if cid == X["KANGA"] and not has(obs, cid):
            value += cfg.get("kanga_bonus", 0) * 1.25
        if cid == X["TEAL"] and sum(p.id == X["TEAL"] for p in board(obs, True)) < cfg["teal_cap"]:
            value += 6000
        if cid == prism_card and has(obs, wanted):
            value += 9000
        if cid in {X["PECH"], X["MUNKI"], X["CHIYU"], X["MEOWTH"]} and cid not in {wanted, secondary_id}:
            value -= 7500
        return value

    ns["_x_role"] = role
    ns["_x_search_score"] = search_score

    def attach_score(obs, option):
        value = old_attach(obs, option)
        energy = source(obs, option)
        target = ns["_target"](obs, option)
        if energy is None or target is None:
            return value
        eid = int(energy.id)
        wanted = dynamic_target(obs)
        if int(target.id) == wanted:
            value += 11500
            if eid == prism_card and wanted in {X["CLEF"], X["CORNER"]}:
                value += 9000
            if eid == X["G"] and wanted in {X["KANGA"], X["TEAL"]}:
                value += 5500
        elif cfg["mode"] == "kangaskhan_then_clefairy" and int(target.id) == X["CLEF"]:
            kanga = pokemon(obs, X["KANGA"])
            if kanga is None or ns["_x_ready_damage"](obs, kanga) <= 0:
                value -= 10000
        elif eid == prism_card and wanted in {X["CLEF"], X["CORNER"]}:
            value -= 10000
        return value

    ns["_x_attach_score"] = attach_score

    def transfer_plan(obs):
        # Start with the mature planner, but reject moves that do not improve the
        # replay-designated attacker's payment or immediate damage.
        plan = old_transfer(obs)
        wanted = dynamic_target(obs)
        target = pokemon(obs, wanted)
        if plan and int(plan[7]) == int(wanted) and target is not None:
            before = ns["_x_ready_damage"](obs, target)
            after = ns["_x_ready_damage"](obs, target, energy_ids(target) + [int(plan[4])])
            if after > before or energy_progress(target, energy_ids(target) + [int(plan[4])]) < energy_progress(target):
                return plan
        # Deterministic fallback: move an energy only when it advances the target.
        if target is None:
            return None
        player = ns["_x_pl"](obs, True)
        entries = []
        for area, pokemon_list in ((AreaType.ACTIVE, player.active), (AreaType.BENCH, player.bench)):
            for index, item in enumerate(pokemon_list or []):
                if item:
                    entries.append((int(area), index, item))
        target_entry = next((x for x in entries if x[2] is target), None)
        if target_entry is None:
            return None
        best = None
        for source_area, source_index, donor in entries:
            if donor is target:
                continue
            donor_energy = energy_ids(donor)
            for energy_index, energy in enumerate(donor_energy):
                if wanted in {X["CLEF"], X["CORNER"]} and not any_energy == int(energy) and not any_energy in energy_ids(target):
                    continue
                before = ns["_x_ready_damage"](obs, target)
                after_energy = energy_ids(target) + [int(energy)]
                after = ns["_x_ready_damage"](obs, target, after_energy)
                progress = energy_progress(target) - energy_progress(target, after_energy)
                donor_after = donor_energy[:energy_index] + donor_energy[energy_index + 1:]
                loss = max(0, ns["_x_ready_damage"](obs, donor) - ns["_x_ready_damage"](obs, donor, donor_after))
                score = 2500 * progress + 12 * (after - before) - 8 * loss
                if before < 60 <= after:
                    score += 7000
                item = (score, source_area, source_index, energy_index, int(energy), target_entry[0], target_entry[1], int(target.id))
                if best is None or item > best:
                    best = item
        return best if best and best[0] >= 2500 else None

    ns["_x_transfer_plan"] = transfer_plan

    def find_main(obs, option_type, card_id=None):
        for index, option in enumerate(obs.select.option):
            if option.type != option_type:
                continue
            card = source(obs, option)
            if card_id is None or (card is not None and int(card.id) == int(card_id)):
                return index
        return None

    def best_manual_attach(obs):
        choices = [(attach_score(obs, option), index) for index, option in enumerate(obs.select.option) if option.type == OptionType.ATTACH]
        return max(choices) if choices else None

    def main(obs):
        player = ns["_x_pl"](obs, True)
        active = player.active[0] if player.active else None
        turn = int(obs.current.turn or 0)
        if turn < state["turn"]:
            state["boss_target"] = None
        state["turn"] = turn

        # Resolve free acceleration/draw before every attack.
        if turn <= 4 or ns["_x_ready_damage"](obs, active) <= 0:
            bug = find_main(obs, OptionType.PLAY, X["BUG"])
            if bug is not None:
                return [bug]
        for cid in (X["TEAL"], X["KANGA"], X["FEZ"]):
            index = find_main(obs, OptionType.ABILITY, cid)
            if index is None:
                continue
            if cid == X["TEAL"] and ns["_x_hcount"](obs, X["G"]) <= 0:
                continue
            memory = ns.get("_XMEM", {})
            if cid == X["KANGA"] and memory.get("kanga_used"):
                continue
            if cid == X["FEZ"] and memory.get("fez_used"):
                continue
            return [index]

        wanted = dynamic_target(obs)
        target = pokemon(obs, wanted)
        active_ready = ns["_x_ready_damage"](obs, active)

        # When an attack is already available, attach toward the next attacker,
        # then take the prize before optional search/cycling.
        if active_ready > 0:
            attach = best_manual_attach(obs)
            if attach and attach[0] >= 3500:
                return [attach[1]]
            denial_target = None
            for opponent in ns["_x_pl"](obs, False).bench or []:
                if opponent and int(opponent.id) in cfg["denial"] and active_ready >= int(opponent.hp or 0):
                    denial_target = int(opponent.id)
                    break
            if denial_target is not None:
                boss = find_main(obs, OptionType.PLAY, X["BOSS"])
                if boss is not None:
                    state["boss_target"] = denial_target
                    return [boss]
            attacks_index = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.ATTACK]
            if attacks_index:
                return [max(attacks_index, key=lambda i: ns["_x_damage"](obs, active, obs.select.option[i].attackId))]

        # Put the replay-designated attacker in play before general hand cycling.
        if target is None:
            direct = find_main(obs, OptionType.PLAY, wanted)
            if direct is not None:
                return [direct]
            search_order = [X["ULTRA"]]
            if wanted in {X["TEAL"], X["CORNER"], X["WELL"]}:
                search_order.insert(0, X["TERA_ORB"])
            for search in search_order:
                index = find_main(obs, OptionType.PLAY, search)
                if index is not None:
                    return [index]

        # Latias converts a prepared Bench attacker into an immediate attack.
        if target is not None and not has(obs, X["LATIAS"]):
            latias = find_main(obs, OptionType.PLAY, X["LATIAS"])
            if latias is not None:
                return [latias]

        plan = transfer_plan(obs)
        if plan:
            energy_switch = find_main(obs, OptionType.PLAY, X["ESWITCH"])
            if energy_switch is not None:
                ns.get("_XMEM", {})["move"] = plan
                return [energy_switch]
        attach = best_manual_attach(obs)
        if attach and attach[0] >= 1800:
            return [attach[1]]

        # Promote a ready matchup attacker.  Against Crustle this explicitly
        # replaces a blocked rule-box attacker with Cornerstone's Demolish.
        target = pokemon(obs, wanted)
        if target is not None and target is not active and ns["_x_ready_damage"](obs, target) > 0 and has(obs, X["LATIAS"]):
            retreat = find_main(obs, OptionType.RETREAT)
            if retreat is not None:
                return [retreat]

        active_ready = ns["_x_ready_damage"](obs, active)
        if active_ready > 0:
            attacks_index = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.ATTACK]
            if attacks_index:
                return [max(attacks_index, key=lambda i: ns["_x_damage"](obs, active, obs.select.option[i].attackId))]

        result = old_main(obs)
        # Final guard: an executable attack must not be replaced by an optional
        # shuffle/cycle action in a later base-policy layer.
        if result and active is not None and ns["_x_ready_damage"](obs, active) > 0:
            attacks_index = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.ATTACK]
            if attacks_index:
                return [max(attacks_index, key=lambda i: ns["_x_damage"](obs, active, obs.select.option[i].attackId))]
        return result

    def context(obs):
        setup = setup_context(obs)
        if setup is not None:
            return setup
        if state["boss_target"] is not None and obs.select.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            for index, option in enumerate(obs.select.option):
                card = source(obs, option)
                if card is not None and int(card.id) == int(state["boss_target"]):
                    state["boss_target"] = None
                    return [index]
        return old_context(obs)

    ns["_x_main"] = main
    ns["_x_context"] = context

