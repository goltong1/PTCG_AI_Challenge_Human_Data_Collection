"""Okidogi Cosmic Adaptive v16 direct-opponent-trained agent for CABT.

The policy combines explicit card mechanics, full public-zone accounting,
opponent archetype recognition, tactical KO planning, and a shallow simulator
lookahead over the rest of the current turn.  Learned weights are intentionally
small residuals around general game rules so loss-replay tuning does not turn
into brittle opponent scripting.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Iterable, Optional


_HERE = os.path.dirname(os.path.abspath(globals().get(
    "__file__", "/kaggle_simulations/agent/main.py"
)))
for _candidate in (_HERE, "/kaggle_simulations/agent", os.getcwd()):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from cg.api import (  # noqa: E402
    AreaType,
    CardType,
    EnergyType,
    LogType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_release,
    search_step,
    to_observation_class,
)
import cg.api as _cg_api  # noqa: E402


# Submission-safe deck loading.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_cg_api.__file__)))
_DECK_CANDIDATES = (
    os.path.join(_HERE, "deck.csv"),
    os.path.join(_ROOT, "deck.csv"),
    "/kaggle_simulations/agent/deck.csv",
    os.path.join(os.getcwd(), "deck.csv"),
)
_DECK_PATH = next((p for p in _DECK_CANDIDATES if os.path.isfile(p)), None)
if _DECK_PATH is None:
    raise FileNotFoundError("deck.csv not found")
with open(_DECK_PATH, "r", encoding="utf-8") as _handle:
    MY_DECK = [int(line.strip()) for line in _handle if line.strip()]
if len(MY_DECK) != 60:
    raise ValueError(f"deck.csv must contain 60 cards, got {len(MY_DECK)}")


CARD = {card.cardId: card for card in all_card_data()}
ATTACK = {attack.attackId: attack for attack in all_attack()}
NEG = -(10**9)

# Pokémon.
OKIDOGI = 116
SOLROCK = 676
LUNATONE = 675
BINACLE = 1051
BARBARACLE = 1052
MUNKIDORI = 112
MOLTRES = 791
CORNERSTONE_OGERPON_EX = 117
BLOODMOON_URSALUNA = 135
MEOWTH_EX = 1071
FEZANDIPITI_EX = 140
DUNSPARCE = 305
DUDUNSPARCE = 66

# Trainers.
LILLIE = 1227
BOSS = 1182
MORTY = 1187
TARRAGON = 1238
CIPHERMANIAC = 1188
JUDGE = 1213
XEROSIC = 1197
COLRESS_TENACITY = 1194
POKE_PAD = 1152
FIGHTING_GONG = 1142
PREMIUM_POWER_PRO = 1141
POKEGEAR = 1122
NIGHT_STRETCHER = 1097
ENERGY_RETRIEVAL = 1118
AIR_BALLOON = 1174
ENHANCED_HAMMER = 1081
BATTLE_CAGE = 1264
NEUTRALIZATION_ZONE = 1247

# Energy.
FIGHTING_ENERGY = 6
PRISM_ENERGY = 16
LEGACY_ENERGY = 12

# Attacks.
GOOD_PUNCH = 147
COSMIC_BEAM = 980
POWER_GEM = 979
DOUBLE_DRAW = 1519
SCRATCH = 1520
HAMMER_IN = 1521
MIND_BEND = 141
FIGHTING_WINGS = 1143
DEMOLISH = 148
MAD_BITE = 175
POWERFUL_HAND = 1072


OUR_POKEMON = {
    OKIDOGI, SOLROCK, LUNATONE, BINACLE, BARBARACLE, MUNKIDORI,
    MOLTRES, CORNERSTONE_OGERPON_EX, BLOODMOON_URSALUNA,
    MEOWTH_EX, FEZANDIPITI_EX,
    DUNSPARCE, DUDUNSPARCE,
}
FIGHTING_POKEMON = {
    OKIDOGI, SOLROCK, LUNATONE, BINACLE, BARBARACLE,
    CORNERSTONE_OGERPON_EX, BLOODMOON_URSALUNA,
}
SUPPORTERS = {
    LILLIE, BOSS, MORTY, TARRAGON, CIPHERMANIAC, JUDGE, XEROSIC,
    COLRESS_TENACITY,
}
SPECIAL_ENERGY = {PRISM_ENERGY, LEGACY_ENERGY}
RULE_BOX = {CORNERSTONE_OGERPON_EX}

# Replay-trained switches remain explicit so every structural correction can
# be ablated during cross-play instead of becoming an untestable hard script.
ENABLE_SPECIFIC_ARCHETYPE_RECOGNITION = True
ENABLE_ALAKAZAM_ENGINE_POLICY = True
ENABLE_ROCKET_PROFILE = True
ENABLE_ABILITY_ATTACKER_BENCH_GUARD = True
ENABLE_MUNKIDORI_ENERGY_GUARD = True
ENABLE_RAINBOW_ENERGY_RULE = True
# v9 candidate switches.  These stay independently mutable so the evaluator
# can attribute gains instead of accepting a bundled change on one score.
ENABLE_MUNKIDORI_PRISM_RESERVE = True
ENABLE_OGERPON_SAFE_ATTACKER_GATE = True


# Small residuals are tuned from loss outcomes.  Hard mechanics and KO gates
# remain much larger than these values.
WEIGHTS = {
    "bench_okidogi": 800,
    "bench_cosmic": 900,
    "bench_binacle": 700,
    "barbaracle": 1600,
    "special_okidogi": 2200,
    "stone_arms": 1300,
    "lunar_cycle": 700,
    "battle_cage": 1400,
    "attack_ko": 1800,
    "boss_ko": 9000,
    "resource": 450,
    "lookahead": 1.0,
}
# Replay-learned resource rules.  The first league exposed a repeated failure:
# Stone Arms was charging Barbaracle itself while the real attackers stayed
# one attachment short.  These values remain explicit for later loss sweeps.
BARBARACLE_SELF_ATTACH_PENALTY = 30000
TARRAGON_MIN_RECOVERY = 3
KEEP_FIGHTING_FOR_URSALUNA = 2
PREFER_FIRST = False
USE_TURN_LOOKAHEAD = True
USE_OPPONENT_ROLLOUT = False
OPPONENT_ROLLOUT_MAX_ACTIONS = 12
USE_SECOND_OWN_ROLLOUT = False
SECOND_OWN_ROLLOUT_MAX_ACTIONS = 10
SUCCESSOR_READY_BONUS = 0
LOOKAHEAD_MAX_CANDIDATES = 4
LOOKAHEAD_MAX_ACTIONS = 10
LOOKAHEAD_MIN_TURN = 0
LOOKAHEAD_TACTICAL_ONLY = True
FUTURE_VALUE_SCALE = 0.5
FUTURE_BASE_MARGIN = 1000.0
FUTURE_RULE_MARGIN = 8000
FUTURE_CRUSTLE_MULTIPLIER = 1.0
FUTURE_DRAGAPULT_MULTIPLIER = 0.5
FUTURE_MARNIE_MULTIPLIER = 0.0
FUTURE_LUCARIO_MULTIPLIER = 0.0
SETUP_BENCH_OGERPON_SCORE = -2000
SETUP_BENCH_URSALUNA_SCORE = -6000
EMERGENCY_URSALUNA_BENCH_LIMIT = -1
ENABLE_HYDRAPPLE_PROFILE = True
HYDRAPPLE_CAGE_SCORE = 9000
HYDRAPPLE_TARGET_BONUS = 0
MARNIE_OGERPON_PLAY_BONUS = 16000
MARNIE_OGERPON_ATTACH_BONUS = 16000
MARNIE_OGERPON_SEARCH_BONUS = 16000


ARCHETYPE_IDS = {
    "dragapult": {119, 120, 121},
    "lucario": {673, 674, 675, 676, 677, 678, 305, 306},
    "marnie": {646, 647, 648},
    "crustle": {344, 345},
    "alakazam": {741, 742, 743, 245},
    "hydrapple": {96, 149, 917, 709, 710, 93, 921, 150},
    "rocket": {400, 401, 431, 432},
    "archaludon": {169, 190, 666},
}
ARCHETYPE_CORE_IDS = {
    "hydrapple": {96, 149, 917, 709, 710, 93, 921, 150},
    "crustle": {344, 345},
    "dragapult": {119, 120, 121},
    "marnie": {646, 647, 648},
    "alakazam": {741, 742, 743, 245},
    "lucario": {333, 677, 678},
    "rocket": {400, 401, 431, 432},
    "archaludon": {169, 190, 666},
}
SETUP_DENIAL = {
    104, 119, 120, 305, 333, 344, 400, 401, 646, 647, 677, 741, 742, 860,
}


_MEMORY = {
    "last_turn": -1,
    "archetype": "unknown",
    "revealed": set(),
    "searched_this_turn": False,
    "lookahead_turn": -1,
    "premium_turn": -1,
}
_LOOKAHEAD_ACTIVE = False
INTERNAL_EXCEPTIONS = 0
LAST_INTERNAL_EXCEPTION = ""


def _reset_memory() -> None:
    _MEMORY["last_turn"] = -1
    _MEMORY["archetype"] = "unknown"
    _MEMORY["revealed"] = set()
    _MEMORY["searched_this_turn"] = False
    _MEMORY["lookahead_turn"] = -1
    _MEMORY["premium_turn"] = -1


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _card_data(cid: int):
    return CARD.get(int(cid or 0))


def _field(player) -> list[tuple[AreaType, int, Pokemon]]:
    result = []
    for i, pokemon in enumerate(player.active or []):
        if pokemon is not None:
            result.append((AreaType.ACTIVE, i, pokemon))
    for i, pokemon in enumerate(player.bench or []):
        if pokemon is not None:
            result.append((AreaType.BENCH, i, pokemon))
    return result


def _pokemon_only(player) -> list[Pokemon]:
    return [pokemon for _, _, pokemon in _field(player)]


def _hand_has(player, cid: int) -> bool:
    return any(card.id == cid for card in (player.hand or []))


def _active(player) -> Optional[Pokemon]:
    return player.active[0] if player.active and player.active[0] else None


def _get_card(obs: Observation, area, index, player_index: int):
    if area is None or index is None:
        return None
    try:
        player = obs.current.players[player_index]
        zones = {
            AreaType.DECK: obs.select.deck or [],
            AreaType.HAND: player.hand or [],
            AreaType.DISCARD: player.discard or [],
            AreaType.ACTIVE: player.active or [],
            AreaType.BENCH: player.bench or [],
            AreaType.PRIZE: player.prize or [],
            AreaType.STADIUM: obs.current.stadium or [],
            AreaType.LOOKING: obs.current.looking or [],
        }
        return zones.get(area, [])[index]
    except (IndexError, TypeError, AttributeError):
        return None


def _option_card(obs: Observation, option):
    me = obs.current.yourIndex
    player = option.playerIndex if option.playerIndex is not None else me
    area = option.area
    if option.type == OptionType.PLAY and area is None:
        area = AreaType.HAND
    return _get_card(obs, area, option.index, player)


def _option_target(obs: Observation, option):
    return _get_card(
        obs, option.inPlayArea, option.inPlayIndex, obs.current.yourIndex
    )


def _source_id(select) -> int:
    source = getattr(select, "contextCard", None)
    if source is None:
        source = getattr(select, "effect", None)
    return int(getattr(source, "id", 0) or 0)


def _attached(pokemon: Optional[Pokemon]) -> int:
    return len(pokemon.energyCards or []) if pokemon is not None else 0


def _damage_on(pokemon: Optional[Pokemon]) -> int:
    return max(0, pokemon.maxHp - pokemon.hp) if pokemon is not None else 0


def _has_energy_type(pokemon: Optional[Pokemon], energy_type) -> bool:
    if pokemon is None:
        return False
    energies = pokemon.energies or []
    return bool(
        energy_type in energies
        or (
            ENABLE_RAINBOW_ENERGY_RULE
            and energy_type != EnergyType.COLORLESS
            and EnergyType.RAINBOW in energies
        )
    )


def _has_special(pokemon: Optional[Pokemon]) -> bool:
    return bool(
        pokemon and any(c.id in SPECIAL_ENERGY for c in (pokemon.energyCards or []))
    )


def _prize_value(pokemon: Optional[Pokemon]) -> int:
    if pokemon is None:
        return 0
    data = _card_data(pokemon.id)
    value = 3 if data and data.megaEx else 2 if data and data.ex else 1
    if any(c.id == LEGACY_ENERGY for c in (pokemon.energyCards or [])):
        value -= 1
    return max(0, value)


def _has_ability(cid: int) -> bool:
    data = _card_data(cid)
    return bool(data and data.skills)


def _retreat_cost(pokemon: Optional[Pokemon]) -> int:
    data = _card_data(pokemon.id) if pokemon is not None else None
    return int(getattr(data, "retreatCost", 0) or 0)


def _stadium_id(state) -> int:
    return state.stadium[0].id if state.stadium else 0


def _count_cards(player):
    field = Counter(p.id for p in _pokemon_only(player))
    hand = Counter(c.id for c in (player.hand or []))
    discard = Counter(c.id for c in (player.discard or []))
    return field, hand, discard


def _known_zone_ids(player) -> list[int]:
    result = [c.id for c in (player.hand or [])]
    result.extend(c.id for c in (player.discard or []))
    for pokemon in _pokemon_only(player):
        result.append(pokemon.id)
        result.extend(c.id for c in (pokemon.preEvolution or []))
        result.extend(c.id for c in (pokemon.energyCards or []))
        result.extend(c.id for c in (pokemon.tools or []))
    return result


def _remaining_counts(player) -> Counter:
    counts = Counter(MY_DECK)
    for cid in _known_zone_ids(player):
        if counts[cid] > 0:
            counts[cid] -= 1
    return counts


def _recognize(ids: Iterable[int]) -> str:
    ids = set(ids)
    if ENABLE_SPECIFIC_ARCHETYPE_RECOGNITION:
        order = [
            "hydrapple", "crustle", "dragapult", "marnie", "alakazam",
            "archaludon", "lucario",
        ]
        if ENABLE_ROCKET_PROFILE:
            order.append("rocket")
        for name in order:
            if name == "hydrapple" and not ENABLE_HYDRAPPLE_PROFILE:
                continue
            if ids & ARCHETYPE_CORE_IDS[name]:
                return name
    profiles = [
        "crustle", "dragapult", "lucario", "marnie", "alakazam",
        "archaludon",
    ]
    if ENABLE_HYDRAPPLE_PROFILE:
        profiles.insert(0, "hydrapple")
    for name in profiles:
        if ids & ARCHETYPE_IDS[name]:
            return name
    names = " ".join(_norm(getattr(_card_data(cid), "name", "")) for cid in ids)
    for profile, needles in {
        "dragapult": ("dragapult", "dreepy", "drakloak"),
        "lucario": ("lucario", "riolu"),
        "marnie": ("grimmsnarl", "impidimp", "morgrem"),
        "crustle": ("crustle", "dwebble"),
        "alakazam": ("alakazam", "abra", "kadabra"),
        "hydrapple": ("hydrapple", "teal mask ogerpon", "meganium", "chikorita"),
        "rocket": ("team rocket s mewtwo", "team rocket s spidops"),
        "archaludon": ("archaludon", "duraludon", "cinderace"),
    }.items():
        if profile == "hydrapple" and not ENABLE_HYDRAPPLE_PROFILE:
            continue
        if profile == "rocket" and not ENABLE_ROCKET_PROFILE:
            continue
        if any(needle in names for needle in needles):
            return profile
    return "unknown"


def _update_memory(obs: Observation) -> str:
    state = obs.current
    if state.turn < int(_MEMORY["last_turn"]):
        _reset_memory()
    if state.turn != int(_MEMORY["last_turn"]):
        _MEMORY["searched_this_turn"] = False
    searched_ids = {
        POKE_PAD, FIGHTING_GONG, POKEGEAR, CIPHERMANIAC,
        NIGHT_STRETCHER, ENERGY_RETRIEVAL, TARRAGON,
    }
    if any(
        getattr(log, "type", None) == LogType.PLAY
        and getattr(log, "playerIndex", None) == state.yourIndex
        and getattr(log, "cardId", None) in searched_ids
        for log in (obs.logs or [])
    ):
        _MEMORY["searched_this_turn"] = True
    if any(
        getattr(log, "type", None) == LogType.PLAY
        and getattr(log, "cardId", None) == PREMIUM_POWER_PRO
        for log in (obs.logs or [])
    ):
        _MEMORY["premium_turn"] = state.turn
    _MEMORY["last_turn"] = state.turn
    me = state.yourIndex
    opponent = state.players[1 - me]
    revealed = set(_MEMORY["revealed"])
    for pokemon in _pokemon_only(opponent):
        revealed.add(pokemon.id)
        revealed.update(c.id for c in (pokemon.preEvolution or []))
    revealed.update(c.id for c in (opponent.discard or []))
    for log in obs.logs or []:
        if getattr(log, "playerIndex", None) == 1 - me:
            for key in (
                "cardId", "cardIdTarget", "cardIdActive", "cardIdBench",
                "cardIdBefore", "cardIdAfter",
            ):
                value = getattr(log, key, None)
                if value is not None:
                    revealed.add(value)
    _MEMORY["revealed"] = revealed
    profile = _recognize(revealed)
    if profile != "unknown":
        _MEMORY["archetype"] = profile
    return str(_MEMORY["archetype"])


def _energy_need(pokemon: Optional[Pokemon]) -> int:
    if pokemon is None:
        return 99
    return {
        OKIDOGI: 2,
        SOLROCK: 1,
        LUNATONE: 2,
        BINACLE: 1,
        BARBARACLE: 3,
        MUNKIDORI: 2,
        MOLTRES: 1,
        CORNERSTONE_OGERPON_EX: 3,
        BLOODMOON_URSALUNA: 3,
        DUNSPARCE: 1,
        DUDUNSPARCE: 1,
        MEOWTH_EX: 3,
        FEZANDIPITI_EX: 3,
    }.get(pokemon.id, 99)


def _attack_damage_for(
    attack_id: int,
    attacker: Optional[Pokemon],
    target: Optional[Pokemon],
    mine,
) -> int:
    if attacker is None:
        return 0
    attack = ATTACK.get(attack_id)
    damage = int(getattr(attack, "damage", 0) or 0)
    if attack_id == GOOD_PUNCH:
        damage = 170 if _has_energy_type(attacker, EnergyType.DARKNESS) else 70
    elif attack_id == COSMIC_BEAM:
        damage = 70 if any(p.id == LUNATONE for p in _pokemon_only(mine)) else 0
    elif attack_id == POWER_GEM:
        damage = 50
    elif attack_id == DOUBLE_DRAW:
        damage = 0
    elif attack_id == SCRATCH:
        damage = 30
    elif attack_id == HAMMER_IN:
        damage = 80
    elif attack_id == MIND_BEND:
        damage = 60
    elif attack_id == FIGHTING_WINGS:
        data = _card_data(target.id) if target is not None else None
        damage = 110 if data and (data.ex or data.megaEx) else 20
    elif attack_id == DEMOLISH:
        damage = 140
    elif attack_id == MAD_BITE:
        damage = 100 + 3 * _damage_on(target)
    elif attack_id == POWERFUL_HAND:
        # Alakazam places two counters for each card in its controller's hand.
        # Model those counters as effective damage for KO/threat planning, but
        # do not apply weakness or ordinary damage-prevention effects.
        damage = 20 * int(mine.handCount)
    if target is None or damage <= 0:
        return damage
    attacker_data = _card_data(attacker.id)
    target_data = _card_data(target.id)
    ignores_weakness = attack_id in {COSMIC_BEAM, DEMOLISH, POWERFUL_HAND}
    if (
        not ignores_weakness and attacker_data and target_data
        and target_data.weakness == attacker_data.energyType
    ):
        damage *= 2
    if attack_id not in {DEMOLISH, POWERFUL_HAND} and target_data:
        text = " ".join(_norm(skill.text) for skill in target_data.skills)
        if "prevent all damage" in text:
            if "pokemon ex" in text and attacker_data and attacker_data.ex:
                damage = 0
            if "pokemon that have an ability" in text and _has_ability(attacker.id):
                damage = 0
    return max(0, damage)


def _attack_damage(attack_id: int, mine, opponent) -> int:
    attacker = _active(mine)
    target = _active(opponent)
    damage = _attack_damage_for(attack_id, attacker, target, mine)
    if (
        damage > 0 and int(_MEMORY.get("premium_turn", -1)) == int(_MEMORY.get("last_turn", -2))
        and attacker is not None
        and getattr(_card_data(attacker.id), "energyType", None) == EnergyType.FIGHTING
    ):
        bonus = 30
        target_data = _card_data(target.id) if target is not None else None
        if (
            attack_id not in {COSMIC_BEAM, DEMOLISH}
            and target_data is not None
            and target_data.weakness == EnergyType.FIGHTING
        ):
            bonus *= 2
        damage += bonus
    return damage


def _target_score(pokemon: Optional[Pokemon], damage: int, archetype: str) -> int:
    if pokemon is None:
        return NEG
    score = _prize_value(pokemon) * 7000
    score += _attached(pokemon) * 900
    score += _damage_on(pokemon) * 12
    data = _card_data(pokemon.id)
    if data and data.stage2:
        score += 3200
    elif data and data.stage1:
        score += 1800
    if pokemon.id in SETUP_DENIAL:
        score += 5000
    if archetype == "marnie" and pokemon.id in {MUNKIDORI, 104}:
        # Removing Adrena-Brain first stops the Grimmsnarl player from healing
        # Demolish damage; removing Froslass stops ability counters that bypass
        # both Cornerstone Stance and Neutralization Zone.
        score += 12000
    if archetype == "dragapult" and pokemon.id == MUNKIDORI:
        score += 10000
    if archetype == "lucario" and pokemon.id in {305, 306, 333, 677}:
        # These are either a future Mega Lucario or a Fighting-weak prize path.
        score += 8500
    if archetype == "alakazam" and ENABLE_ALAKAZAM_ENGINE_POLICY:
        score += {
            743: 20000,
            245: 20000,
            742: 14500,
            66: 11000,
            741: 7500,
            65: 3500,
            305: 3500,
        }.get(pokemon.id, 0)
    if archetype == "archaludon":
        # Removing the Basic before Assemble Alloy denies both the 300 HP ex
        # and two accelerated Metal Energy.  Cinderace is the backup engine.
        score += {169: 17500, 666: 12500, 190: 9000}.get(pokemon.id, 0)
    if archetype == "rocket" and ENABLE_ROCKET_PROFILE:
        score += {431: 18000, 401: 15000, 400: 7000, 432: 5000}.get(
            pokemon.id, 0
        )
    if archetype == "hydrapple":
        # Wild Growth doubles every Basic Grass Energy, so a one-prize
        # Meganium/pre-evolution KO can prevent much more damage than its
        # immediate prize count suggests.  Ogerpon/Hydrapple also retain a
        # smaller bonus because removing attached Energy shrinks Syrup Storm.
        if pokemon.id in {710, 709, 917, 149, 93, 921}:
            score += HYDRAPPLE_TARGET_BONUS
        elif pokemon.id in {96, 150}:
            score += HYDRAPPLE_TARGET_BONUS // 3
    if damage >= pokemon.hp and damage > 0:
        score += 32000 + _prize_value(pokemon) * 9000
    else:
        score += min(pokemon.hp, damage) * 28
    if archetype == "crustle" and pokemon.id == 344:
        score += 8000
    return score


def _best_attack_damage(pokemon: Optional[Pokemon], target, mine) -> int:
    if pokemon is None:
        return 0
    data = _card_data(pokemon.id)
    return max(
        (_attack_damage_for(aid, pokemon, target, mine) for aid in (data.attacks or [])),
        default=0,
    )


def _boss_has_ko(obs: Observation, mine, opponent, archetype: str) -> bool:
    active = _active(mine)
    if active is None:
        return False
    legal = [
        int(o.attackId or 0) for o in obs.select.option
        if o.type == OptionType.ATTACK
    ]
    for target in opponent.bench or []:
        if any(
            (
                _attack_damage_for(aid, active, target, mine)
                + (
                    60 if (
                        int(_MEMORY.get("premium_turn", -1)) == int(_MEMORY.get("last_turn", -2))
                        and aid not in {COSMIC_BEAM, DEMOLISH}
                        and getattr(_card_data(active.id), "energyType", None) == EnergyType.FIGHTING
                        and getattr(_card_data(target.id), "weakness", None) == EnergyType.FIGHTING
                    ) else 30 if (
                        int(_MEMORY.get("premium_turn", -1)) == int(_MEMORY.get("last_turn", -2))
                        and getattr(_card_data(active.id), "energyType", None) == EnergyType.FIGHTING
                    ) else 0
                )
            ) >= target.hp
            for aid in legal
        ):
            return True
    return False


def _boss_has_ko_from_state(mine, opponent, archetype: str) -> bool:
    """Conservative Boss estimate for Pokegear prompts outside MAIN options."""
    active = _active(mine)
    if active is None or _attached(active) < _energy_need(active):
        return False
    return any(
        _best_attack_damage(active, target, mine) >= target.hp
        for target in (opponent.bench or [])
    )


def _bench_slots(player) -> int:
    return max(0, 5 - len(player.bench or []))


def _opponent_has_ex(opponent) -> bool:
    return any(
        bool(
            getattr(_card_data(p.id), "ex", False)
            or getattr(_card_data(p.id), "megaEx", False)
        )
        for p in _pokemon_only(opponent)
    )


def _opponent_has_special_energy(opponent) -> bool:
    return any(
        getattr(_card_data(energy.id), "cardType", None) == CardType.SPECIAL_ENERGY
        for pokemon in _pokemon_only(opponent)
        for energy in (pokemon.energyCards or [])
    )


def _ogerpon_is_safe(opponent, archetype: str) -> bool:
    """Only expose a two-prize Ogerpon when it blocks the visible attackers.

    The three learned positive matchups have an explicit strategic reason to
    use Cornerstone Ogerpon.  Elsewhere, seeing *an* Ability is insufficient:
    a non-Ability attacker on the same board can simply take two prizes.
    """
    if archetype in {"crustle", "marnie", "rocket", "archaludon", "hydrapple"}:
        return True
    if archetype in {"dragapult", "lucario", "alakazam"}:
        return False
    attackers = []
    for pokemon in _pokemon_only(opponent):
        data = _card_data(pokemon.id)
        if data is not None and (getattr(data, "attacks", None) or []):
            attackers.append(pokemon)
    return bool(attackers) and all(_has_ability(p.id) for p in attackers)


def _play_pokemon_score(card, mine, opponent, field, hand, archetype, state) -> int:
    if card is None or _bench_slots(mine) <= 0:
        return NEG
    cid = card.id
    if cid == OKIDOGI:
        if field[OKIDOGI] >= 2:
            return -4000
        return 38500 + WEIGHTS["bench_okidogi"]
    if cid == SOLROCK:
        if field[SOLROCK] >= 2:
            return -5000
        value = 36500 + WEIGHTS["bench_cosmic"]
        if field[LUNATONE] == 0:
            value += 3500
        return value
    if cid == LUNATONE:
        if field[LUNATONE] >= 1:
            return -5000
        return 39500 + WEIGHTS["bench_cosmic"]
    if cid == BINACLE:
        if field[BINACLE] + field[BARBARACLE] >= 1:
            return -2500
        return 40500 + WEIGHTS["bench_binacle"]
    if cid == MUNKIDORI:
        if field[MUNKIDORI] > 0:
            return NEG
        damaged = sum(_damage_on(p) for p in _pokemon_only(mine))
        special_live = hand[PRISM_ENERGY] + hand[LEGACY_ENERGY] > 0
        return 34000 if damaged >= 20 and special_live else 8500
    if cid == MOLTRES:
        if field[MOLTRES] > 0:
            return NEG
        return 33000 if _opponent_has_ex(opponent) and hand[PRISM_ENERGY] else 7000
    if cid == CORNERSTONE_OGERPON_EX:
        if field[cid] > 0:
            return NEG
        if archetype in {"crustle", "marnie", "rocket", "archaludon", "hydrapple"}:
            return 47000 + (
                MARNIE_OGERPON_PLAY_BONUS if archetype == "marnie" else 0
            )
        if (
            ENABLE_ABILITY_ATTACKER_BENCH_GUARD
            and archetype in {"dragapult", "lucario", "alakazam"}
        ):
            return NEG
        if ENABLE_OGERPON_SAFE_ATTACKER_GATE:
            return 36000 if _ogerpon_is_safe(opponent, archetype) else NEG
        ability_attacker = any(_has_ability(p.id) for p in _pokemon_only(opponent))
        return 36000 if ability_attacker else 4500
    if cid == BLOODMOON_URSALUNA:
        if field[cid] > 0:
            return NEG
        target = _active(opponent)
        basics = hand[FIGHTING_ENERGY]
        finisher = target is not None and _damage_on(target) >= 30
        if basics >= 2 and (finisher or len(mine.prize or []) <= 3):
            return 78000 if finisher else 45500
        # Loss replays sometimes ended with Ursaluna stranded in hand while
        # the last one or two Pokémon were removed.  The configurable gate is
        # evaluated separately so the insurance play can be A/B tested.
        if len(mine.bench or []) <= EMERGENCY_URSALUNA_BENCH_LIMIT:
            return 62000 if not mine.bench else 50000
        return 10000 if basics >= 2 else NEG
    if cid == MEOWTH_EX:
        if field[cid] > 0:
            return NEG
        has_supporter = sum(hand[s] for s in SUPPORTERS) > 0
        value = 44000 if not has_supporter or archetype == "alakazam" else 25500
        if len(mine.bench or []) >= 4:
            value -= 24000
        return value
    if cid == FEZANDIPITI_EX:
        if field[cid] > 0:
            return NEG
        opponent_has_taken_prize = len(opponent.prize or []) < 6
        if opponent_has_taken_prize and mine.handCount <= 5:
            return 41500
        return 9000 if mine.handCount <= 3 else NEG
    if cid == DUNSPARCE:
        if field[DUNSPARCE] + field[DUDUNSPARCE] >= 2:
            return -3000
        return 39000
    return -5000


def _evolve_score(evolution, target, field) -> int:
    if evolution is None or target is None:
        return NEG
    if evolution.id == BARBARACLE and target.id == BINACLE:
        value = 54000 + WEIGHTS["barbaracle"]
        value += min(3, _attached(target)) * 900
        if field[BARBARACLE] == 0:
            value += 5000
        return value
    if evolution.id == DUDUNSPARCE and target.id == DUNSPARCE:
        return 55500
    return 10000


def _manual_attach_score(card, target, mine, opponent, field, hand, archetype, source) -> int:
    if card is None or target is None:
        return NEG
    cid = card.id
    if cid == AIR_BALLOON:
        score = 17000
        if target == _active(mine):
            score += 9000 + _retreat_cost(target) * 1500
        if target.id in {BLOODMOON_URSALUNA, OKIDOGI, BARBARACLE}:
            score += 3500
        if target.tools:
            score -= 20000
        return score
    if cid not in {FIGHTING_ENERGY, PRISM_ENERGY, LEGACY_ENERGY}:
        return NEG
    current = _attached(target)
    need = _energy_need(target)
    score = 27000 + max(0, need - current) * 3300
    active = _active(mine)
    if active is not None and target.serial == active.serial:
        score += 5500
        if current + 1 >= need:
            score += 11500
    elif current + 1 >= need and SUCCESSOR_READY_BONUS > 0:
        ready_bench = any(
            p.serial != target.serial
            and _attached(p) >= _energy_need(p)
            and _best_attack_damage(p, _active(opponent), mine) > 0
            for p in (mine.bench or [])
        )
        if not ready_bench:
            score += SUCCESSOR_READY_BONUS
    target_data = _card_data(target.id)
    if cid == LEGACY_ENERGY:
        score += 10500 - _prize_value(target) * 1200
        if target.id == OKIDOGI and not _has_energy_type(target, EnergyType.DARKNESS):
            score += 16000 + WEIGHTS["special_okidogi"]
        elif _has_energy_type(target, EnergyType.DARKNESS):
            score -= 26000
    elif cid == PRISM_ENERGY:
        if target_data and not target_data.basic:
            score -= 22000
        if target.id == OKIDOGI and not _has_energy_type(target, EnergyType.DARKNESS):
            score += 17000 + WEIGHTS["special_okidogi"]
        elif target.id == OKIDOGI:
            score -= 36000
        elif target.id == MOLTRES and _opponent_has_ex(opponent):
            score += 13000
        elif target.id == MUNKIDORI and sum(_damage_on(p) for p in _pokemon_only(mine)):
            score += 10500
        elif target.id == CORNERSTONE_OGERPON_EX and archetype in {
            "crustle", "marnie", "rocket", "archaludon", "hydrapple"
        }:
            score += 19000 if archetype in {"archaludon", "hydrapple"} else 7500
        elif target.id == BINACLE:
            score -= 17000
    else:
        if target.id == SOLROCK and current == 0:
            score += 11500
        elif target.id == BLOODMOON_URSALUNA:
            score += 7500
        elif target.id == CORNERSTONE_OGERPON_EX and archetype in {
            "crustle", "marnie", "rocket", "archaludon", "hydrapple"
        }:
            score += 17000 if archetype in {"archaludon", "hydrapple"} else 6000
        elif target.id == OKIDOGI and current == 0 and not _has_special(target):
            score -= 2500
    if target.id == CORNERSTONE_OGERPON_EX and archetype == "marnie" and current < need:
        score += MARNIE_OGERPON_ATTACH_BONUS
    if target.id == CORNERSTONE_OGERPON_EX and archetype == "rocket" and current < need:
        score += 14000
    if ENABLE_MUNKIDORI_ENERGY_GUARD and target.id == MUNKIDORI:
        has_darkness = _has_energy_type(target, EnergyType.DARKNESS)
        if cid == FIGHTING_ENERGY and not has_darkness:
            # A lone Fighting attachment neither enables Adrena-Brain nor an
            # attack.  The new loss replay showed it being lost immediately
            # while a real attacker was still searchable.
            score -= 70000 if current >= 1 else 33000
        elif cid in SPECIAL_ENERGY and not has_darkness:
            if ENABLE_MUNKIDORI_PRISM_RESERVE:
                own_damage = sum(_damage_on(p) for p in _pokemon_only(mine))
                primary_ready = any(
                    p.id in {OKIDOGI, SOLROCK, BLOODMOON_URSALUNA}
                    and _attached(p) >= _energy_need(p)
                    for p in _pokemon_only(mine)
                )
                starving_okidogi = any(
                    p.id == OKIDOGI
                    and not _has_energy_type(p, EnergyType.DARKNESS)
                    and _attached(p) < _energy_need(p)
                    for p in _pokemon_only(mine)
                )
                if own_damage <= 0:
                    score -= 26000
                elif starving_okidogi and not primary_ready:
                    score -= 14000
                else:
                    score += 12000
            else:
                score += 18000
    if target.id == BARBARACLE:
        # Stone Arms is an engine, not the default energy sink. Charge it only
        # after a primary attacker is ready or when it is the emergency Active.
        primary_ready = any(
            p.id in {OKIDOGI, SOLROCK, BLOODMOON_URSALUNA}
            and _attached(p) >= _energy_need(p)
            for p in _pokemon_only(mine)
        )
        if not primary_ready or target != _active(mine):
            score -= BARBARACLE_SELF_ATTACH_PENALTY
    if source == BARBARACLE:
        score += 10000 + WEIGHTS["stone_arms"]
    if current >= need:
        # Replays showed scarce Prism Energy being stacked as a third Energy
        # on an already-ready Okidogi.  Only Legacy's prize effect can justify
        # an over-attachment; ordinary Energy must start the next attacker.
        score -= 22000 if cid == LEGACY_ENERGY else 52000
    return score


def _ability_score(card, mine, opponent, field, hand, discard, archetype) -> int:
    if card is None:
        return NEG
    if card.id == BARBARACLE:
        if hand[FIGHTING_ENERGY] <= 0:
            return NEG
        return 61000 + WEIGHTS["stone_arms"]
    if card.id == LUNATONE:
        if hand[FIGHTING_ENERGY] <= 0 or mine.deckCount <= 7:
            return NEG
        value = 40000 + WEIGHTS["lunar_cycle"]
        if mine.handCount <= 4:
            value += 19000
        elif mine.handCount <= 6:
            value += 7000
        ready_energy = any(
            _attached(p) < _energy_need(p) for p in _pokemon_only(mine)
            if p.id in {OKIDOGI, SOLROCK, BLOODMOON_URSALUNA}
        )
        if hand[FIGHTING_ENERGY] == 1 and ready_energy:
            value -= 13000
        target = _active(opponent)
        if (
            target is not None and _damage_on(target) >= 30
            and field[BLOODMOON_URSALUNA] == 0
            and hand[FIGHTING_ENERGY] <= KEEP_FIGHTING_FOR_URSALUNA
        ):
            return NEG
        return value
    if card.id == MUNKIDORI:
        own_damage = sum(_damage_on(p) for p in _pokemon_only(mine))
        if own_damage <= 0 or not _has_energy_type(card, EnergyType.DARKNESS):
            return NEG
        target = max((_damage_on(p) for p in _pokemon_only(opponent)), default=0)
        return 59000 + min(30, own_damage) * 80 + target * 20
    if card.id == BLOODMOON_URSALUNA:
        return 62000 if hand[FIGHTING_ENERGY] >= 2 else NEG
    if card.id == MEOWTH_EX:
        return 61000
    if card.id == FEZANDIPITI_EX:
        return 64000 if mine.handCount <= 5 else 32000
    if card.id == DUDUNSPARCE:
        return 64000 if mine.deckCount > 5 else NEG
    return 18000


def _recoverable(discard: Counter) -> int:
    return discard[FIGHTING_ENERGY] + sum(discard[cid] for cid in FIGHTING_POKEMON)


def _play_trainer_score(
    cid, obs, mine, opponent, field, hand, discard, remaining, archetype
) -> int:
    state = obs.current
    if cid == BATTLE_CAGE:
        if _stadium_id(state) == NEUTRALIZATION_ZONE:
            return NEG
        if _stadium_id(state) == BATTLE_CAGE:
            return NEG
        if archetype == "hydrapple":
            return HYDRAPPLE_CAGE_SCORE
        threat = archetype == "dragapult" or any(p.id == MUNKIDORI for p in _pokemon_only(opponent))
        return (56000 + WEIGHTS["battle_cage"]) if threat else 9000
    if cid == NEUTRALIZATION_ZONE:
        if _stadium_id(state) == NEUTRALIZATION_ZONE:
            return NEG
        protected = any(
            not bool(getattr(_card_data(p.id), "ex", False))
            and not bool(getattr(_card_data(p.id), "megaEx", False))
            for p in _pokemon_only(mine)
        )
        threat = archetype in {
            "dragapult", "marnie", "lucario", "hydrapple", "rocket",
            "archaludon",
        }
        return 72000 if protected and threat else 18000
    if cid == PREMIUM_POWER_PRO:
        active = _active(mine)
        target = _active(opponent)
        data = _card_data(active.id) if active is not None else None
        if active is None or target is None or data is None or data.energyType != EnergyType.FIGHTING:
            return NEG
        base = _best_attack_damage(active, target, mine)
        if base <= 0:
            return 12000
        if base < target.hp <= base + 30:
            return 69000
        if archetype == "marnie" and active.id == CORNERSTONE_OGERPON_EX:
            return 52000
        if target.hp <= base * 2 + 30:
            return 45500
        return 33000
    if cid == FIGHTING_GONG:
        if mine.deckCount <= 5:
            return NEG
        missing = int(field[OKIDOGI] == 0) + int(field[LUNATONE] == 0)
        missing += int(field[SOLROCK] == 0) + int(field[BINACLE] + field[BARBARACLE] == 0)
        energy_need = int(hand[FIGHTING_ENERGY] == 0)
        return 41500 + missing * 1800 + energy_need * 3200
    if cid == POKE_PAD:
        if mine.deckCount <= 5:
            return NEG
        missing = int(field[BARBARACLE] == 0 and field[BINACLE] > 0)
        missing += int(field[LUNATONE] == 0) + int(field[OKIDOGI] == 0)
        return 40500 + missing * 2300 if missing else 17000
    if cid == POKEGEAR:
        if mine.deckCount <= 7:
            return NEG
        has_supporter = sum(hand[s] for s in SUPPORTERS) > 0
        return 39000 if not has_supporter else 14500
    if cid == NIGHT_STRETCHER:
        useful = discard[FIGHTING_ENERGY] + sum(discard[p] for p in OUR_POKEMON)
        return 39000 if useful > 0 else NEG
    if cid == ENHANCED_HAMMER:
        return 63500 if _opponent_has_special_energy(opponent) else NEG
    if cid == ENERGY_RETRIEVAL:
        count = discard[FIGHTING_ENERGY]
        if count <= 0:
            return NEG
        value = 40500 + min(2, count) * 3000
        if field[LUNATONE] or field[BARBARACLE]:
            value += 5000
        return value
    if cid == AIR_BALLOON:
        active = _active(mine)
        if active is None or active.tools:
            return 15000
        return 34000 + _retreat_cost(active) * 1300
    if cid == TARRAGON:
        count = _recoverable(discard)
        if count < TARRAGON_MIN_RECOVERY and len(mine.prize or []) > 2:
            return NEG
        return 43000 + min(4, count) * 3100
    if cid == MORTY:
        bench = len(opponent.bench or [])
        if bench < 3 or mine.handCount <= 1 or mine.deckCount <= bench + 4:
            return NEG
        value = 33500 + bench * 3300
        if mine.handCount <= 4:
            value += 7000
        return value
    if cid == JUDGE:
        if mine.deckCount <= 8:
            return NEG
        if archetype == "alakazam" and ENABLE_ALAKAZAM_ENGINE_POLICY:
            engine_active = (
                _active(opponent) is not None
                and _active(opponent).id in {245, 742, 743}
            )
            if opponent.handCount >= 7 or (engine_active and opponent.handCount >= 6):
                return 72000 + max(0, opponent.handCount - 4) * 5200
        if mine.handCount > 6:
            return NEG
        # Direct opponents refill efficiently.  Shuffling six cards to four
        # on turn two cost our only supporter and did not slow their engine.
        if opponent.handCount < 8:
            return 18500 if mine.handCount <= 2 and opponent.handCount >= 7 else NEG
        disruption = max(0, opponent.handCount - 4) * 4200
        refill = max(0, 4 - mine.handCount) * 3200
        matchup = 7000 if archetype in {
            "dragapult", "marnie", "lucario", "hydrapple", "alakazam", "rocket"
        } else 0
        return 30000 + disruption + refill + matchup if opponent.handCount >= 6 else 18500
    if cid == XEROSIC:
        if opponent.handCount < 7:
            return NEG
        value = 61000 + max(0, opponent.handCount - 7) * 5200
        if archetype == "alakazam":
            value += 22000
        return value
    if cid == CIPHERMANIAC:
        if mine.deckCount <= 5:
            return NEG
        lunar_live = field[LUNATONE] and field[SOLROCK] and hand[FIGHTING_ENERGY] > 0
        missing_special = field[OKIDOGI] and not any(
            _has_special(p) for p in _pokemon_only(mine) if p.id == OKIDOGI
        )
        return 53500 if lunar_live else 35000 + 4500 * int(bool(missing_special))
    if cid == COLRESS_TENACITY:
        zone_live = _stadium_id(state) == NEUTRALIZATION_ZONE
        zone_available = remaining[NEUTRALIZATION_ZONE] > 0
        needs_prism = any(
            p.id == OKIDOGI and not _has_special(p)
            for p in _pokemon_only(mine)
        )
        if not zone_live and zone_available:
            return 73500
        if needs_prism and remaining[PRISM_ENERGY] > 0:
            return 54500
        return 25000 if mine.deckCount > 8 else NEG
    if cid == LILLIE:
        draw = 8 if len(mine.prize or []) == 6 else 6
        if _MEMORY["searched_this_turn"] or mine.deckCount <= draw + 3:
            return NEG
        prepared = hand[PRISM_ENERGY] + hand[LEGACY_ENERGY] + hand[BARBARACLE]
        value = 31500 + draw * 350 - min(3, prepared) * 2500
        if mine.handCount <= 3:
            value += 18500
        elif mine.handCount <= 5:
            value += 9500
        elif mine.handCount >= 7:
            return NEG
        return value
    if cid == BOSS:
        if archetype == "alakazam" and ENABLE_ALAKAZAM_ENGINE_POLICY:
            current = _active(opponent)
            attacker = _active(mine)
            legal = [
                int(option.attackId or 0) for option in obs.select.option
                if option.type == OptionType.ATTACK
            ]
            if (
                current is not None
                and attacker is not None
                and current.id in {245, 66, 742, 743}
                and any(
                    _attack_damage_for(aid, attacker, current, mine) >= current.hp
                    for aid in legal
                )
            ):
                return NEG
        if _boss_has_ko(obs, mine, opponent, archetype):
            return 57000 + WEIGHTS["boss_ko"]
        active = _active(opponent)
        if active and archetype == "crustle" and active.id == 345:
            return 42000
        return NEG
    return 0


def _attack_score(attack_id, mine, opponent, field, archetype, state) -> int:
    active = _active(mine)
    target = _active(opponent)
    if active is None:
        return NEG
    damage = _attack_damage(attack_id, mine, opponent)
    score = 26000 + min(12000, damage * 38)
    if attack_id == DOUBLE_DRAW:
        return 39000 if mine.handCount <= 4 and mine.deckCount > 5 else 22000
    if attack_id == GOOD_PUNCH and damage >= 170:
        score += 6500
    if attack_id == COSMIC_BEAM:
        score += 3800
    if attack_id == MIND_BEND:
        score += 2500
    if attack_id == FIGHTING_WINGS and target is not None:
        data = _card_data(target.id)
        score += 6000 if data and data.ex else -3500
    if attack_id == DEMOLISH:
        score += 6000
        if archetype == "crustle":
            score += 25000
    if attack_id == MAD_BITE:
        score += min(16000, _damage_on(target) * 65)
    if target is not None and damage >= target.hp and damage > 0:
        score += 36000 + _prize_value(target) * 10500 + WEIGHTS["attack_ko"]
        if _prize_value(target) >= len(mine.prize or []):
            score += 900000
    if damage <= 0:
        score -= 30000
    return score


def _ready(pokemon: Pokemon, target, mine) -> bool:
    return _attached(pokemon) >= _energy_need(pokemon) and _best_attack_damage(pokemon, target, mine) > 0


def _switch_score(pokemon, mine, opponent, archetype) -> int:
    if pokemon is None:
        return NEG
    target = _active(opponent)
    damage = _best_attack_damage(pokemon, target, mine)
    score = damage * 45 + pokemon.hp * 10 - _prize_value(pokemon) * 2000
    if _ready(pokemon, target, mine):
        score += 27000
    if target is not None and damage >= target.hp:
        score += 35000 + _prize_value(target) * 8000
    if pokemon.id == CORNERSTONE_OGERPON_EX and archetype in {"crustle", "marnie", "rocket"}:
        score += 18000
    if pokemon.id in {LUNATONE, MUNKIDORI, BINACLE, DUNSPARCE, DUDUNSPARCE} and not _ready(pokemon, target, mine):
        score -= 12000
    return score


def _retreat_score(mine, opponent, archetype) -> int:
    active = _active(mine)
    target = _active(opponent)
    if active is None:
        return NEG
    active_damage = _best_attack_damage(active, target, mine)
    bench = [p for p in (mine.bench or []) if _ready(p, target, mine)]
    if not bench:
        return NEG
    best = max(_best_attack_damage(p, target, mine) for p in bench)
    if active_damage <= 0 or best >= active_damage + 60:
        return 46500
    if active.id in {LUNATONE, MUNKIDORI, BINACLE}:
        return 41000
    return NEG


def _main_score(obs, option, mine, opponent, field, hand, discard, remaining, archetype):
    typ = option.type
    source = _source_id(obs.select)
    if typ == OptionType.ABILITY:
        return _ability_score(_option_card(obs, option), mine, opponent, field, hand, discard, archetype)
    if typ == OptionType.EVOLVE:
        return _evolve_score(_option_card(obs, option), _option_target(obs, option), field)
    if typ == OptionType.ATTACH:
        return _manual_attach_score(
            _option_card(obs, option), _option_target(obs, option), mine,
            opponent, field, hand, archetype, source,
        ) + 32000
    if typ == OptionType.PLAY:
        card = _option_card(obs, option)
        if card is None:
            return NEG
        data = _card_data(card.id)
        if data and data.cardType == CardType.POKEMON:
            return _play_pokemon_score(card, mine, opponent, field, hand, archetype, obs.current)
        return _play_trainer_score(
            card.id, obs, mine, opponent, field, hand, discard, remaining, archetype
        )
    if typ == OptionType.RETREAT:
        return _retreat_score(mine, opponent, archetype)
    if typ == OptionType.ATTACK:
        return _attack_score(int(option.attackId or 0), mine, opponent, field, archetype, obs.current)
    if typ == OptionType.END:
        return -50000
    if typ == OptionType.DISCARD:
        return -5000
    return 0


def _search_score(cid, source, mine, opponent, field, hand, discard, remaining, archetype) -> int:
    score = 1000
    active = _active(mine)
    emergency_attacker = (
        len(mine.bench or []) == 0
        and active is not None
        and active.id in {MUNKIDORI, LUNATONE, CORNERSTONE_OGERPON_EX, BLOODMOON_URSALUNA}
    )
    damaged_target = (
        _active(opponent) is not None and _damage_on(_active(opponent)) >= 30
    )
    if cid == OKIDOGI:
        score = 28500 if field[OKIDOGI] < 2 else 5000
        if emergency_attacker:
            score += 22000
    elif cid == LUNATONE:
        score = 30000 if field[LUNATONE] == 0 else 3000
    elif cid == SOLROCK:
        score = 27500 if field[SOLROCK] == 0 else 6000
        if emergency_attacker and field[LUNATONE] > 0:
            score += 18000
    elif cid == BINACLE:
        score = 29000 if field[BINACLE] + field[BARBARACLE] == 0 else 4000
    elif cid == BARBARACLE:
        score = 32500 if field[BINACLE] > 0 and field[BARBARACLE] == 0 else 6000
    elif cid == BLOODMOON_URSALUNA:
        target = _active(opponent)
        score = 47000 if target and _damage_on(target) >= 30 else 8000
    elif cid == DUNSPARCE:
        score = 27500 if field[DUNSPARCE] + field[DUDUNSPARCE] == 0 else 5000
    elif cid == DUDUNSPARCE:
        score = 35000 if field[DUNSPARCE] > 0 else 4500
    elif cid == MUNKIDORI:
        score = 21000 if sum(_damage_on(p) for p in _pokemon_only(mine)) else 5000
    elif cid == MOLTRES:
        score = 23000 if _opponent_has_ex(opponent) else 3500
    elif cid == MEOWTH_EX:
        score = 26000 if sum(hand[s] for s in SUPPORTERS) == 0 else 9000
    elif cid == FEZANDIPITI_EX:
        score = 22000 if len(opponent.prize or []) < 6 and mine.handCount <= 5 else 4000
    elif cid == CORNERSTONE_OGERPON_EX:
        if ENABLE_OGERPON_SAFE_ATTACKER_GATE:
            score = 36000 if _ogerpon_is_safe(opponent, archetype) else 1500
        else:
            score = 36000 if archetype in {"crustle", "marnie", "rocket"} else 7000
        if archetype == "marnie":
            score += MARNIE_OGERPON_SEARCH_BONUS
        elif archetype == "rocket":
            score += 16000
        elif archetype == "archaludon":
            score += 22000
        elif archetype == "hydrapple":
            score += 22000
    elif cid == FIGHTING_ENERGY:
        need = sum(_attached(p) < _energy_need(p) for p in _pokemon_only(mine))
        score = 25000 + min(3, need) * 1800
        if damaged_target and hand[BLOODMOON_URSALUNA] and hand[FIGHTING_ENERGY] < 2:
            score += 21000
    elif cid == PRISM_ENERGY:
        okidogi_needs = any(p.id == OKIDOGI and not _has_special(p) for p in _pokemon_only(mine))
        score = 31500 if okidogi_needs else 22000
    elif cid == LEGACY_ENERGY:
        score = 34000
    elif cid == BATTLE_CAGE:
        if archetype == "hydrapple":
            score = HYDRAPPLE_CAGE_SCORE
        else:
            score = 28000 if archetype == "dragapult" else 9000
    elif cid == NEUTRALIZATION_ZONE:
        score = 51000 if archetype in {
            "dragapult", "marnie", "lucario", "hydrapple", "rocket",
            "archaludon",
        } else 7000
    elif cid == PREMIUM_POWER_PRO:
        score = 17000
    elif cid == LILLIE:
        score = 24500 if mine.handCount <= 4 else 10000
    elif cid == TARRAGON:
        score = 25000 if _recoverable(discard) >= 2 else 6000
    elif cid == BOSS:
        score = 14000
    elif cid == CIPHERMANIAC:
        score = 17500
    elif cid == COLRESS_TENACITY:
        score = 36000 if remaining[NEUTRALIZATION_ZONE] > 0 else 21000
    elif cid == MORTY:
        score = 15000 if len(opponent.bench or []) >= 3 else 4000
    elif cid == JUDGE:
        if archetype == "alakazam" and opponent.handCount >= 7:
            score = 44000
        else:
            score = 28000 if opponent.handCount >= 6 and mine.handCount <= 5 else 6500
    elif cid == XEROSIC:
        score = 50000 if archetype == "alakazam" and opponent.handCount >= 7 else 9000
    elif cid == ENHANCED_HAMMER:
        score = 42000 if _opponent_has_special_energy(opponent) else 1000
    if source == FIGHTING_GONG:
        if (
            damaged_target and field[BLOODMOON_URSALUNA] == 0
            and hand[BLOODMOON_URSALUNA] == 0
            and cid == BLOODMOON_URSALUNA
        ):
            score += 24000
        if cid == FIGHTING_ENERGY and hand[FIGHTING_ENERGY] == 0:
            score += 8000
        if cid == OKIDOGI and field[OKIDOGI] == 0:
            score += 20000 if emergency_attacker else 6000
        if cid == LUNATONE and field[LUNATONE] == 0:
            score += 6500
    elif source == POKE_PAD:
        if cid == BARBARACLE and field[BINACLE] > 0:
            score += 11000
        if cid == BLOODMOON_URSALUNA:
            if damaged_target:
                score += 26000
            elif hand[FIGHTING_ENERGY] >= 2:
                score += 6500
        if cid == DUDUNSPARCE and field[DUNSPARCE] > 0:
            score += 13000
    elif source == POKEGEAR:
        if cid == LILLIE:
            score += 22000 if mine.handCount <= 5 else 7000
        elif cid == CIPHERMANIAC:
            lunar_live = field[LUNATONE] and field[SOLROCK] and hand[FIGHTING_ENERGY]
            score += 14000 if lunar_live else 3000
        elif cid == BOSS:
            score += 16000 if _boss_has_ko_from_state(mine, opponent, archetype) else -12000
        elif cid == JUDGE:
            if archetype == "alakazam" and opponent.handCount >= 7:
                score += 30000
            else:
                score += 18000 if opponent.handCount >= 7 and mine.handCount <= 5 else -5000
        elif cid == COLRESS_TENACITY:
            score += 26000 if remaining[NEUTRALIZATION_ZONE] > 0 else 7000
        elif cid == XEROSIC:
            score += 39000 if archetype == "alakazam" and opponent.handCount >= 7 else -7000
    elif source == MEOWTH_EX:
        if cid == JUDGE:
            score += 52000 if archetype == "alakazam" and opponent.handCount >= 7 else -8000
        elif cid == COLRESS_TENACITY:
            needs_prism = any(
                p.id == OKIDOGI and not _has_special(p)
                for p in _pokemon_only(mine)
            )
            score += 34000 if needs_prism else 13000
        elif cid == LILLIE:
            score += 30000 if mine.handCount <= 5 else 5000
        elif cid == BOSS:
            score += 30000 if _boss_has_ko_from_state(mine, opponent, archetype) else -9000
        elif cid == CIPHERMANIAC:
            score += 15000 if field[LUNATONE] and field[SOLROCK] else 3000
        elif cid == XEROSIC:
            score += 42000 if archetype == "alakazam" and opponent.handCount >= 7 else -7000
    elif source == NIGHT_STRETCHER:
        if cid == FIGHTING_ENERGY:
            score += 6000
        elif cid in {OKIDOGI, BARBARACLE, BLOODMOON_URSALUNA}:
            score += 7000
    elif source == TARRAGON:
        if cid == FIGHTING_ENERGY:
            score += 7000
        elif cid in {OKIDOGI, BARBARACLE, BLOODMOON_URSALUNA}:
            score += 6500
    elif source == CIPHERMANIAC:
        if cid in {
            PRISM_ENERGY, LEGACY_ENERGY, BARBARACLE, BATTLE_CAGE,
            NEUTRALIZATION_ZONE,
        }:
            score += 9000
        elif cid == FIGHTING_ENERGY and field[LUNATONE]:
            score += 6000
    elif source == COLRESS_TENACITY:
        if cid == NEUTRALIZATION_ZONE:
            score += 30000
        elif cid == PRISM_ENERGY:
            score += 26000
        elif cid == FIGHTING_ENERGY:
            score += 9000
    return score


def _discard_score(cid, field, hand, discard, remaining, state, source) -> int:
    # Higher is safer to discard.
    if cid in SPECIAL_ENERGY:
        return -18000
    if cid == FIGHTING_ENERGY:
        recovery = hand[ENERGY_RETRIEVAL] + hand[TARRAGON]
        return 5000 if hand[cid] >= 3 or recovery else -6500
    if cid == BATTLE_CAGE:
        return 8000 if _stadium_id(state) == BATTLE_CAGE or hand[cid] >= 2 else -4000
    if cid == NEUTRALIZATION_ZONE:
        return -20000 if _stadium_id(state) != NEUTRALIZATION_ZONE else 10000
    if cid == PREMIUM_POWER_PRO:
        return 5500 if hand[cid] >= 2 else -1000
    if cid == AIR_BALLOON:
        return 6500 if hand[cid] >= 2 else 500
    if cid == LILLIE:
        return 5500 if hand[cid] >= 2 or state.supporterPlayed else -2500
    if cid == BOSS:
        return 4500 if hand[cid] >= 2 else -5500
    if cid in {MORTY, CIPHERMANIAC, JUDGE, XEROSIC, COLRESS_TENACITY}:
        return 3500
    if cid == TARRAGON:
        return 4500 if _recoverable(discard) == 0 else -5500
    if cid in OUR_POKEMON:
        if hand[cid] >= 2 or field[cid] > 0:
            return 5000
        if remaining[cid] <= 1:
            return -9000
        return -3500
    if cid in {POKE_PAD, FIGHTING_GONG, POKEGEAR}:
        return 5000 if hand[cid] >= 2 else 1000
    return 2000


def _card_choice_score(
    obs, option, mine, opponent, field, hand, discard, remaining, source, archetype
) -> int:
    me = obs.current.yourIndex
    player = option.playerIndex if option.playerIndex is not None else me
    card = _option_card(obs, option)
    if card is None:
        return NEG
    cid = card.id
    context = obs.select.context
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        priorities = {
            OKIDOGI: 22000,
            SOLROCK: 20500 if hand[LUNATONE] > 0 else 12500,
            MOLTRES: 14000,
            LUNATONE: 10500,
            BINACLE: 9000,
            MUNKIDORI: 7000,
            MEOWTH_EX: 4500,
            FEZANDIPITI_EX: 3500,
            CORNERSTONE_OGERPON_EX: 6000,
            BLOODMOON_URSALUNA: 2000,
            DUNSPARCE: 14500,
        }
        return priorities.get(cid, 0)
    if context == SelectContext.SETUP_BENCH_POKEMON:
        priorities = {
            LUNATONE: 18000,
            BINACLE: 17000,
            OKIDOGI: 16500,
            SOLROCK: 12500,
            MUNKIDORI: 2500,
            MEOWTH_EX: 8000,
            FEZANDIPITI_EX: 6500,
            MOLTRES: 1500,
            CORNERSTONE_OGERPON_EX: SETUP_BENCH_OGERPON_SCORE,
            BLOODMOON_URSALUNA: SETUP_BENCH_URSALUNA_SCORE,
            DUNSPARCE: 15000,
        }
        value = priorities.get(cid, NEG)
        if cid == LUNATONE and field[LUNATONE] > 0:
            return NEG
        if cid == BINACLE and field[BINACLE] + field[BARBARACLE] > 0:
            return NEG
        if cid == OKIDOGI and field[OKIDOGI] >= 2:
            return NEG
        if cid == SOLROCK and field[SOLROCK] >= 2:
            return NEG
        return value
    if context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        if player != me:
            active = _active(mine)
            damage = _best_attack_damage(active, card, mine)
            return _target_score(card, damage, archetype)
        return _switch_score(card, mine, opponent, archetype)
    if context in {SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
        return _play_pokemon_score(card, mine, opponent, field, hand, archetype, obs.current)
    if context == SelectContext.TO_HAND:
        return _search_score(cid, source, mine, opponent, field, hand, discard, remaining, archetype)
    if context in {SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO}:
        return 40000 if cid in {BARBARACLE, DUDUNSPARCE} else 5000
    if context == SelectContext.DISCARD:
        if source == LUNATONE:
            return 50000 if cid == FIGHTING_ENERGY else NEG
        return _discard_score(cid, field, hand, discard, remaining, obs.current, source)
    if context in {SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM}:
        if isinstance(card, Pokemon):
            energy_card = next(
                (c for c in (mine.hand or []) if c.id == FIGHTING_ENERGY),
                None,
            )
            return _manual_attach_score(
                energy_card, card, mine, opponent, field, hand, archetype, source
            )
        if cid == FIGHTING_ENERGY:
            return 50000
        return NEG
    if context in {
        SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
        SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.EFFECT_TARGET,
    } and isinstance(card, Pokemon):
        if player == me:
            return _damage_on(card) * 100 + int(card.id == OKIDOGI) * 1200
        ko = int(card.hp <= 30) * 25000
        return ko + _prize_value(card) * 5000 + _damage_on(card) * 35
    if context in {
        SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DETACH_FROM,
    } and isinstance(card, Pokemon):
        return _attached(card) * 5000 + _prize_value(card) * 1200
    return _search_score(cid, source, mine, opponent, field, hand, discard, remaining, archetype)


def _choose(scores: list[int], select, positive_only: bool = False) -> list[int]:
    if not scores or select.maxCount == 0:
        return []
    ranked = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)
    minimum = int(select.minCount or 0)
    maximum = min(int(select.maxCount or 0), len(ranked))
    if positive_only:
        chosen = [i for i in ranked if scores[i] > 0][:maximum]
        if len(chosen) < minimum:
            chosen = ranked[:minimum]
        return chosen
    count = max(minimum, maximum)
    return ranked[:count]


def _validate(result: list[int], select) -> list[int]:
    result = [int(i) for i in result if isinstance(i, int)]
    result = list(dict.fromkeys(result))
    result = [i for i in result if 0 <= i < len(select.option)]
    minimum = int(select.minCount or 0)
    maximum = int(select.maxCount or minimum)
    if len(result) > maximum:
        result = result[:maximum]
    if len(result) < minimum:
        for i in range(len(select.option)):
            if i not in result:
                result.append(i)
                if len(result) >= minimum:
                    break
    return result


# Public deck templates of the local sparring agents, used only for simulator
# hidden-zone determinizations. Visible cards always override these priors.
_OPPONENT_TEMPLATES = {
    "dragapult": [
        119,119,119,119,120,120,120,120,121,121,121,305,305,66,66,112,112,235,235,140,1071,306,
        1227,1227,1227,1227,1182,1182,1182,1198,1198,1198,1213,1086,1086,1086,1086,1152,1152,1152,1152,
        1121,1121,1121,1121,1097,1097,1080,1260,1260,1260,2,2,2,5,5,5,7,7,7,
    ],
    "lucario": [
        333,333,333,677,678,678,678,678,676,676,675,675,305,305,305,66,66,306,
        1141,1141,1141,1141,1142,1142,1142,1142,1152,1152,1152,1152,1086,1086,1086,1213,1197,1174,1159,
        1227,1227,1227,1227,1225,1225,1225,1182,1182,1182,1252,1211,6,6,6,6,6,6,6,6,6,6,20,
    ],
    "marnie": [
        7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,646,646,646,646,647,647,647,648,648,648,860,860,
        1079,1079,1079,1080,1086,1086,1086,1086,1097,1097,1097,1122,1137,1152,1152,1152,1152,1182,1182,
        1219,1219,1219,1219,1227,1227,1227,1227,1231,1259,1259,1259,1259,
    ],
    "crustle": [
        *([1] * 11), *([11] * 4), *([344] * 4), *([345] * 4), *([1086] * 4), *([1117] * 4),
        *([1120] * 4), *([1121] * 4), *([1097] * 3), *([1213] * 3), *([1227] * 3), *([1225] * 3),
        *([1123] * 2), *([1122] * 2), *([1081] * 2), *([1182] * 2), 1159,
    ],
}


def _remove_known(deck: Iterable[int], known: Iterable[int]) -> list[int]:
    counts = Counter(deck)
    for cid in known:
        if counts[cid] > 0:
            counts[cid] -= 1
    result = []
    for cid in sorted(counts):
        result.extend([cid] * counts[cid])
    return result


def _rotated(values: list[int], offset: int) -> list[int]:
    if not values:
        return values
    offset %= len(values)
    return values[offset:] + values[:offset]


def _hidden_prediction(obs: Observation, archetype: str):
    state = obs.current
    me = state.yourIndex
    mine = state.players[me]
    opponent = state.players[1 - me]
    own = _remove_known(MY_DECK, _known_zone_ids(mine))
    own = _rotated(own, state.turn * 7 + mine.deckCount)
    own_need = len(mine.prize or []) + mine.deckCount
    if len(own) < own_need:
        own.extend([FIGHTING_ENERGY] * (own_need - len(own)))
    your_prize = own[:len(mine.prize or [])]
    your_deck = own[len(mine.prize or []):own_need]
    template = _OPPONENT_TEMPLATES.get(
        archetype, [SOLROCK] * 4 + [FIGHTING_ENERGY] * 56
    )
    hidden = _remove_known(template, _known_zone_ids(opponent))
    hidden_need = opponent.handCount + len(opponent.prize or []) + opponent.deckCount
    if len(hidden) < hidden_need:
        hidden.extend([FIGHTING_ENERGY] * (hidden_need - len(hidden)))
    hidden = _rotated(hidden, state.turn * 11 + opponent.deckCount)
    opponent_hand = hidden[:opponent.handCount]
    start = opponent.handCount
    opponent_prize = hidden[start:start + len(opponent.prize or [])]
    start += len(opponent.prize or [])
    opponent_deck = hidden[start:start + opponent.deckCount]
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, []


def _snapshot(obs: Observation, root_index: int) -> dict[str, int]:
    state = obs.current
    mine = state.players[root_index]
    opponent = state.players[1 - root_index]
    return {
        "my_prize": len(mine.prize or []),
        "op_prize": len(opponent.prize or []),
        "my_hp": sum(p.hp for p in _pokemon_only(mine)),
        "op_hp": sum(p.hp for p in _pokemon_only(opponent)),
    }


def _future_board_value(obs: Observation, root_index: int, archetype: str) -> float:
    """Portable two-turn value estimate using only the simulated public state.

    It rewards a prepared successor, one-attachment attack lines and reachable
    two-hit prize maps, while charging for the opponent's analogous threats.
    The scalar is replay-trained and defaults to zero until an A/B test wins.
    """
    if FUTURE_VALUE_SCALE <= 0:
        return 0.0
    state = obs.current
    mine = state.players[root_index]
    opponent = state.players[1 - root_index]
    my_field = _pokemon_only(mine)
    op_field = _pokemon_only(opponent)
    my_active = _active(mine)
    op_active = _active(opponent)

    def projected_damage(pokemon, target, player) -> tuple[float, bool, bool]:
        if pokemon is None or target is None:
            return 0.0, False, False
        damage = float(_best_attack_damage(pokemon, target, player))
        need = _energy_need(pokemon)
        attached = _attached(pokemon)
        ready = attached >= need and damage > 0
        one_short = attached + 1 >= need and damage > 0 and not ready
        if ready:
            return damage, True, False
        if one_short:
            return damage * 0.72, False, True
        return 0.0, False, False

    value = 0.0
    my_ready = my_one_short = op_ready = op_one_short = 0
    for pokemon in my_field:
        damage, ready, one_short = projected_damage(pokemon, op_active, mine)
        my_ready += int(ready)
        my_one_short += int(one_short)
        if ready:
            value += 3300 if pokemon != my_active else 1250
        elif one_short:
            value += 1750
        value += min(220.0, damage) * 5.0
        if pokemon.id == OKIDOGI and _has_special(pokemon):
            value += 1900
    for pokemon in op_field:
        damage, ready, one_short = projected_damage(pokemon, my_active, opponent)
        op_ready += int(ready)
        op_one_short += int(one_short)
        if ready:
            value -= 3150 if pokemon != op_active else 1100
        elif one_short:
            value -= 1550
        value -= min(220.0, damage) * 4.5

    # A current attacker plus a successor is much harder to strand after a KO.
    value += 2100 * int(my_ready >= 2) + 800 * min(2, my_one_short)
    value -= 1950 * int(op_ready >= 2) + 700 * min(2, op_one_short)
    if len(mine.bench or []) == 0:
        value -= 3800
    elif len(mine.bench or []) == 1:
        value -= 900
    if len(opponent.bench or []) == 0:
        value += 3000

    # Estimate the best one- or two-hit prize route from every visible target.
    for target in op_field:
        best = max(
            (projected_damage(attacker, target, mine)[0] for attacker in my_field),
            default=0.0,
        )
        prize = max(1, _prize_value(target))
        if best >= target.hp and best > 0:
            value += 2800 * prize
        elif best * 2 >= target.hp and best > 0:
            value += 1350 * prize
        if target.id in SETUP_DENIAL and target in (opponent.bench or []):
            value += 650
    for target in my_field:
        best = max(
            (projected_damage(attacker, target, opponent)[0] for attacker in op_field),
            default=0.0,
        )
        prize = max(1, _prize_value(target))
        if best >= target.hp and best > 0:
            value -= 2650 * prize
        elif best * 2 >= target.hp and best > 0:
            value -= 1200 * prize

    hand = Counter(card.id for card in (mine.hand or []))
    remaining = _remaining_counts(mine)
    energy_now = hand[FIGHTING_ENERGY] + hand[PRISM_ENERGY]
    search_now = hand[FIGHTING_GONG] + hand[COLRESS_TENACITY]
    value += 320 * min(3, energy_now) + 240 * min(2, search_now)
    value += 90 * min(6, remaining[FIGHTING_ENERGY])
    if any(p.id == OKIDOGI and not _has_special(p) for p in my_field):
        value += 420 * min(2, hand[PRISM_ENERGY] + remaining[PRISM_ENERGY])

    stadium = _stadium_id(state)
    if stadium == NEUTRALIZATION_ZONE:
        protected = sum(
            not bool(getattr(_card_data(p.id), "ex", False))
            and not bool(getattr(_card_data(p.id), "megaEx", False))
            for p in my_field
        )
        rule_box_attackers = sum(
            bool(getattr(_card_data(p.id), "ex", False))
            or bool(getattr(_card_data(p.id), "megaEx", False))
            for p in op_field
        )
        value += 1050 * min(protected, 3) * int(rule_box_attackers > 0)
    elif stadium == BATTLE_CAGE and archetype in {"dragapult", "marnie"}:
        value += 2600
    elif archetype in {"dragapult", "marnie"} and len(mine.bench or []) >= 3:
        value -= 900
    matchup_multiplier = {
        "crustle": FUTURE_CRUSTLE_MULTIPLIER,
        "dragapult": FUTURE_DRAGAPULT_MULTIPLIER,
        "marnie": FUTURE_MARNIE_MULTIPLIER,
        "lucario": FUTURE_LUCARIO_MULTIPLIER,
    }.get(archetype, 1.0)
    return value * FUTURE_VALUE_SCALE * matchup_multiplier


def _board_value(
    obs: Observation, root_index: int, root: dict[str, int], archetype: str
) -> float:
    state = obs.current
    mine = state.players[root_index]
    opponent = state.players[1 - root_index]
    if state.result == root_index:
        return 1_000_000.0
    if state.result == 1 - root_index:
        return -1_000_000.0
    end = _snapshot(obs, root_index)
    value = 29000.0 * max(0, root["my_prize"] - end["my_prize"])
    value -= 31500.0 * max(0, root["op_prize"] - end["op_prize"])
    value += 8.0 * max(0, root["op_hp"] - end["op_hp"])
    value -= 5.0 * max(0, root["my_hp"] - end["my_hp"])
    value += 75.0 * mine.handCount + 15.0 * min(15, mine.deckCount)
    card_values = {
        OKIDOGI: 2600, SOLROCK: 1300, LUNATONE: 1800, BINACLE: 700,
        BARBARACLE: 3100, MUNKIDORI: 900, MOLTRES: 650,
        CORNERSTONE_OGERPON_EX: 1100, BLOODMOON_URSALUNA: 1700,
        DUNSPARCE: 900, DUDUNSPARCE: 2300,
    }
    target = _active(opponent)
    for pokemon in _pokemon_only(mine):
        value += card_values.get(pokemon.id, 0)
        value += _attached(pokemon) * 720
        value += _best_attack_damage(pokemon, target, mine) * 10
        if pokemon.id == OKIDOGI and _has_energy_type(pokemon, EnergyType.DARKNESS):
            value += 2600
    if _stadium_id(state) == BATTLE_CAGE:
        value += 1900
    if _stadium_id(state) == NEUTRALIZATION_ZONE:
        value += 12500
    # Anticipate the opponent's immediately legal attack response when the
    # simulated node has passed the turn.
    if state.yourIndex == 1 - root_index and obs.select and obs.select.context == SelectContext.MAIN:
        op_active = _active(opponent)
        my_active = _active(mine)
        threats = [
            _attack_damage_for(int(o.attackId or 0), op_active, my_active, opponent)
            for o in obs.select.option if o.type == OptionType.ATTACK
        ]
        threat = max(threats, default=0)
        op_data = _card_data(op_active.id) if op_active is not None else None
        my_data = _card_data(my_active.id) if my_active is not None else None
        if (
            _stadium_id(state) == NEUTRALIZATION_ZONE
            and op_data is not None and (op_data.ex or op_data.megaEx)
            and my_data is not None and not (my_data.ex or my_data.megaEx)
        ):
            threat = 0
        if my_active is not None:
            value -= min(threat, my_active.hp) * 18
            if threat >= my_active.hp:
                value -= 10500 * max(1, _prize_value(my_active))
    return value


def _copy_memory() -> dict:
    return {k: set(v) if isinstance(v, set) else v for k, v in _MEMORY.items()}


def _restore_memory(saved: dict) -> None:
    _MEMORY.clear()
    _MEMORY.update({k: set(v) if isinstance(v, set) else v for k, v in saved.items()})


def _opponent_rollout_action(obs: Observation, root_player: int) -> list[int]:
    """Generic, submission-safe adversary used for one simulated reply turn."""
    state = obs.current
    select = obs.select
    actor = state.yourIndex
    mine = state.players[actor]
    target_player = state.players[1 - actor]
    active = _active(mine)
    target = _active(target_player)
    if select is None:
        return []
    if select.type == SelectType.COUNT:
        return _choose(
            [int(getattr(option, "number", 0) or 0) for option in select.option],
            select,
        )
    if select.type == SelectType.ENERGY:
        return _choose(
            [int(getattr(option, "count", 1) or 1) for option in select.option],
            select,
        )
    if select.type == SelectType.SKILL:
        return _choose([1000 - i for i, _ in enumerate(select.option)], select)
    if select.type == SelectType.YES_NO:
        yes = select.context not in {SelectContext.MULLIGAN}
        return _choose([
            1000 if option.type == (OptionType.YES if yes else OptionType.NO) else 0
            for option in select.option
        ], select)
    if select.type == SelectType.ATTACK or select.context == SelectContext.ATTACK:
        scores = []
        for option in select.option:
            if option.type != OptionType.ATTACK:
                scores.append(NEG)
                continue
            damage = _attack_damage_for(
                int(option.attackId or 0), active, target, mine
            )
            ko = int(target is not None and damage >= target.hp and damage > 0)
            scores.append(damage * 100 + ko * 50000)
        return _choose(scores, select)
    if select.context == SelectContext.MAIN:
        scores = []
        legal_attacks = [
            _attack_damage_for(int(o.attackId or 0), active, target, mine)
            for o in select.option if o.type == OptionType.ATTACK
        ]
        has_attack = bool(legal_attacks)
        for option in select.option:
            typ = option.type
            if typ == OptionType.EVOLVE:
                score = 88000
            elif typ == OptionType.ATTACH:
                score = 83500
            elif typ == OptionType.ABILITY:
                score = 79000
            elif typ == OptionType.PLAY:
                card = _option_card(obs, option)
                data = _card_data(card.id) if card is not None else None
                card_type = getattr(data, "cardType", None)
                score = {
                    CardType.SUPPORTER: 75000,
                    CardType.ITEM: 70000,
                    CardType.POKEMON: 66500,
                    CardType.TOOL: 62500,
                    CardType.STADIUM: 59000,
                }.get(card_type, 56000)
            elif typ == OptionType.ATTACK:
                damage = _attack_damage_for(
                    int(option.attackId or 0), active, target, mine
                )
                ko = int(target is not None and damage >= target.hp and damage > 0)
                score = 52000 + damage * 80 + ko * 90000
            elif typ == OptionType.RETREAT:
                score = 57000 if not has_attack else 22000
            elif typ == OptionType.END:
                score = -50000
            else:
                score = 10000
            scores.append(score)
        return _choose(scores, select)

    scores = []
    for option in select.option:
        if option.type not in {
            OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD,
        }:
            scores.append(0)
            continue
        card = _option_card(obs, option)
        if card is None:
            scores.append(NEG)
            continue
        player = option.playerIndex if option.playerIndex is not None else actor
        data = _card_data(card.id)
        if select.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            if player == actor and isinstance(card, Pokemon):
                damage = _best_attack_damage(card, target, mine)
                score = damage * 70 + card.hp * 15 - _prize_value(card) * 900
                if _attached(card) >= _energy_need(card):
                    score += 18000
            else:
                damage = _best_attack_damage(active, card, mine)
                score = _target_score(card, damage, "unknown")
        elif select.context in {SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
            score = int(getattr(data, "hp", 0) or 0) * 20
            score += 7000 * int(bool(getattr(data, "skills", [])))
        elif select.context == SelectContext.TO_HAND:
            card_type = getattr(data, "cardType", None)
            score = {
                CardType.POKEMON: 18000,
                CardType.BASIC_ENERGY: 17000,
                CardType.SPECIAL_ENERGY: 17500,
                CardType.SUPPORTER: 16000,
                CardType.ITEM: 14500,
                CardType.TOOL: 12000,
                CardType.STADIUM: 10500,
            }.get(card_type, 5000)
            score += int(getattr(data, "stage2", False)) * 3000
            score += int(getattr(data, "stage1", False)) * 1500
        elif select.context == SelectContext.DISCARD:
            card_type = getattr(data, "cardType", None)
            score = {
                CardType.STADIUM: 7000,
                CardType.TOOL: 6200,
                CardType.ITEM: 5400,
                CardType.BASIC_ENERGY: 2500,
                CardType.SPECIAL_ENERGY: 800,
                CardType.SUPPORTER: 1500,
                CardType.POKEMON: 500,
            }.get(card_type, 3000)
        elif select.context in {
            SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO,
        } and isinstance(card, Pokemon):
            missing = max(0, _energy_need(card) - _attached(card))
            score = missing * 8500 + int(card == active) * 5000
        elif select.context in {
            SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
            SelectContext.DAMAGE, SelectContext.EFFECT_TARGET,
        } and isinstance(card, Pokemon):
            if player == root_player:
                score = (card.maxHp - card.hp) * 20
                score += max(0, 200 - card.hp) * 35
                score += _prize_value(card) * 5500
            else:
                score = max(0, card.maxHp - card.hp) * 30
        elif select.context in {
            SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER,
        } and isinstance(card, Pokemon):
            score = max(0, card.maxHp - card.hp) * 60
        elif select.context in {SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO}:
            score = 20000 + int(getattr(data, "stage2", False)) * 5000
        else:
            score = 5000 + int(getattr(data, "hp", 0) or 0)
        scores.append(score)
    positive = select.minCount == 0 or select.context in {
        SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD,
    }
    return _choose(scores, select, positive_only=positive)


def _lookahead_scores(obs: Observation, scores: list[int], archetype: str) -> list[int]:
    global _LOOKAHEAD_ACTIVE
    mine = obs.current.players[obs.current.yourIndex]
    opponent = obs.current.players[1 - obs.current.yourIndex]
    tactical = any(o.type == OptionType.ATTACK for o in obs.select.option)
    tactical = tactical or (
        _hand_has(mine, BLOODMOON_URSALUNA)
        and _active(opponent) is not None
        and _damage_on(_active(opponent)) >= 30
    )
    tactical = tactical or (
        _hand_has(mine, BOSS) and _boss_has_ko(obs, mine, opponent, archetype)
    )
    if (
        _LOOKAHEAD_ACTIVE or not USE_TURN_LOOKAHEAD
        or obs.current.turn < LOOKAHEAD_MIN_TURN
        or (LOOKAHEAD_TACTICAL_ONLY and not tactical)
        or obs.search_begin_input is None or len(scores) < 2
    ):
        return scores
    ranked = sorted(
        ((scores[i], i) for i, o in enumerate(obs.select.option)
         if o.type != OptionType.END and scores[i] > NEG // 2),
        reverse=True,
    )
    candidates = [i for _, i in ranked[:LOOKAHEAD_MAX_CANDIDATES]]
    if len(candidates) < 2:
        return scores
    saved = _copy_memory()
    created = set()
    root_node = None
    root_player = obs.current.yourIndex
    root_turn = obs.current.turn
    root_snapshot = _snapshot(obs, root_player)
    values = {}
    future_values = {}
    try:
        root_node = search_begin(obs, *_hidden_prediction(obs, archetype), manual_coin=False)
        created.add(root_node.searchId)
        _LOOKAHEAD_ACTIVE = True
        for candidate in candidates:
            _restore_memory(saved)
            node = search_step(root_node.searchId, [candidate])
            created.add(node.searchId)
            actions = 1
            while (
                node.observation.select is not None
                and node.observation.current.result < 0
                and node.observation.current.turn == root_turn
                and actions < LOOKAHEAD_MAX_ACTIONS
            ):
                action = _validate(_strategy(node.observation), node.observation.select)
                node = search_step(node.searchId, action)
                created.add(node.searchId)
                actions += 1
            opponent_actions = 0
            while (
                USE_OPPONENT_ROLLOUT
                and node.observation.select is not None
                and node.observation.current.result < 0
                and node.observation.current.turn == root_turn + 1
                and node.observation.current.yourIndex != root_player
                and opponent_actions < OPPONENT_ROLLOUT_MAX_ACTIONS
            ):
                action = _validate(
                    _opponent_rollout_action(node.observation, root_player),
                    node.observation.select,
                )
                node = search_step(node.searchId, action)
                created.add(node.searchId)
                opponent_actions += 1
            second_actions = 0
            while (
                USE_SECOND_OWN_ROLLOUT
                and node.observation.select is not None
                and node.observation.current.result < 0
                and node.observation.current.turn == root_turn + 2
                and node.observation.current.yourIndex == root_player
                and second_actions < SECOND_OWN_ROLLOUT_MAX_ACTIONS
            ):
                action = _validate(
                    _strategy(node.observation), node.observation.select
                )
                node = search_step(node.searchId, action)
                created.add(node.searchId)
                second_actions += 1
            values[candidate] = _board_value(
                node.observation, root_player, root_snapshot, archetype
            ) + 0.012 * scores[candidate] * WEIGHTS["lookahead"]
            future_values[candidate] = _future_board_value(
                node.observation, root_player, archetype
            )
    except Exception:
        return scores
    finally:
        _restore_memory(saved)
        for search_id in sorted(created, reverse=True):
            try:
                search_release(search_id)
            except Exception:
                pass
        try:
            search_end()
        except Exception:
            pass
        _LOOKAHEAD_ACTIVE = False
    if len(values) < 2:
        return scores
    base_best = max(values, key=values.get)
    best = base_best
    if FUTURE_VALUE_SCALE > 0 and FUTURE_BASE_MARGIN > 0:
        eligible = [
            candidate for candidate in values
            if values[candidate] >= values[base_best] - FUTURE_BASE_MARGIN
            and scores[candidate] >= scores[base_best] - FUTURE_RULE_MARGIN
        ]
        if eligible:
            best = max(
                eligible,
                key=lambda candidate: values[candidate] + future_values[candidate],
            )
    result = list(scores)
    result[best] = max(result) + 50000
    return result


def _strategy(obs: Observation) -> list[int]:
    state = obs.current
    select = obs.select
    me = state.yourIndex
    mine = state.players[me]
    opponent = state.players[1 - me]
    archetype = _update_memory(obs)
    field, hand, discard = _count_cards(mine)
    remaining = _remaining_counts(mine)
    source = _source_id(select)
    if select.context == SelectContext.IS_FIRST:
        desired = OptionType.YES if PREFER_FIRST else OptionType.NO
        return _choose([1000 if o.type == desired else 0 for o in select.option], select)
    if select.context == SelectContext.MULLIGAN:
        return _choose([1000 if o.type == OptionType.YES else 0 for o in select.option], select)
    if select.context in {SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT, SelectContext.MORE_DEVOLVE}:
        yes = 10000 if source in {LUNATONE, BARBARACLE, MUNKIDORI, BLOODMOON_URSALUNA} else 1000
        return _choose([yes if o.type == OptionType.YES else 0 for o in select.option], select)
    if select.context == SelectContext.COIN_HEAD:
        return _choose([100 if o.type == OptionType.YES else 0 for o in select.option], select)
    if select.context == SelectContext.MAIN:
        scores = [
            _main_score(obs, o, mine, opponent, field, hand, discard, remaining, archetype)
            for o in select.option
        ]
        attack_indices = [i for i, o in enumerate(select.option) if o.type == OptionType.ATTACK]
        if attack_indices:
            winning = [i for i in attack_indices if scores[i] >= 500000]
            if not winning:
                useful = [
                    scores[i] for i, o in enumerate(select.option)
                    if o.type in {OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY, OptionType.ATTACH}
                    and scores[i] >= 39000
                ]
                if useful:
                    ceiling = max(useful) - 1
                    for i in attack_indices:
                        scores[i] = min(scores[i], ceiling)
        if (
            not _LOOKAHEAD_ACTIVE
            and int(_MEMORY.get("lookahead_turn", -1)) != state.turn
        ):
            _MEMORY["lookahead_turn"] = state.turn
            scores = _lookahead_scores(obs, scores, archetype)
        return _choose(scores, select)
    if select.type == SelectType.ATTACK or select.context == SelectContext.ATTACK:
        scores = [
            _attack_score(int(o.attackId or 0), mine, opponent, field, archetype, state)
            if o.type == OptionType.ATTACK else NEG for o in select.option
        ]
        return _choose(scores, select)
    if select.type == SelectType.COUNT:
        scores = []
        for o in select.option:
            number = int(o.number or 0)
            value = number * 100
            if select.context == SelectContext.DRAW_COUNT and mine.deckCount <= number + 2:
                value -= 20000
            scores.append(value)
        return _choose(scores, select)
    if select.type == SelectType.ENERGY:
        remain = int(getattr(select, "remainEnergyCost", 0) or 0)
        scores = [1000 - abs(remain - int(getattr(o, "count", 1) or 1)) * 100 for o in select.option]
        return _choose(scores, select)
    if select.type == SelectType.SKILL:
        priority = {BARBARACLE: 5000, LUNATONE: 4500, MUNKIDORI: 4000, BLOODMOON_URSALUNA: 3500}
        return _choose([priority.get(int(o.cardId or 0), 0) for o in select.option], select)
    scores = [
        _card_choice_score(
            obs, o, mine, opponent, field, hand, discard, remaining, source, archetype
        ) if o.type in {OptionType.CARD, OptionType.ENERGY_CARD, OptionType.TOOL_CARD} else 0
        for o in select.option
    ]
    positive = select.minCount == 0 or select.context in {
        SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_HAND,
        SelectContext.TO_BENCH, SelectContext.TO_FIELD,
    }
    return _choose(scores, select, positive_only=positive)


def _fallback(observation: dict) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", minimum) or minimum)
    if not options or maximum == 0:
        return []
    count = minimum if minimum > 0 else 1
    return list(range(min(len(options), maximum, count)))


def agent(observation: dict) -> list[int]:
    if observation.get("select") is None:
        _reset_memory()
        return list(MY_DECK)
    try:
        obs = to_observation_class(observation)
        return _validate(_strategy(obs), obs.select)
    except Exception as exc:
        global INTERNAL_EXCEPTIONS, LAST_INTERNAL_EXCEPTION
        INTERNAL_EXCEPTIONS += 1
        LAST_INTERNAL_EXCEPTION = f"{type(exc).__name__}: {exc}"
        return _fallback(observation)
