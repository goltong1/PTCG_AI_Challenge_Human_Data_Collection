"""Runtime card-text semantics for previously unseen cards.

The CABT engine exposes the complete local card and attack database.  This module
reads those texts at import time and converts recurring rule language into stable
semantic tags.  No card ID whitelist is required for the generic reasoning path.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from cg.api import all_attack, all_card_data

CARD = {int(c.cardId): c for c in all_card_data()}
ATTACK = {int(a.attackId): a for a in all_attack()}

_APOS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "})
_STOP = {
    'a','an','and','any','are','as','at','be','both','by','can','card','cards','do','does','done','during','each','for','from',
    'has','have','if','in','into','is','it','its','may','more','of','on','once','one','or','other','put','that','the','their',
    'them','then','this','those','to','up','was','way','when','with','your','you','opponent','opponents','pokemon','pokémon',
    'attack','attacks','turn','active','bench','benched','play','playing','player','players'
}


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '').translate(_APOS).lower()).strip()


def lexical_words(text: str, limit: int = 14) -> list[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text).replace('{', ' ').replace('}', ' '))
    out: list[str] = []
    for w in words:
        if len(w) < 2 or w in _STOP:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _contains(t: str, *phrases: str) -> bool:
    return any(p in t for p in phrases)


@lru_cache(maxsize=None)
def skill_tags_from_text(text: str) -> tuple[str, ...]:
    t = normalize(text)
    tags: list[str] = []
    if 'prevent all damage' in t:
        tags.append('PREVENT_DAMAGE')
        if 'and effects of attacks' in t or 'damage from and effects' in t:
            tags.append('PREVENT_EFFECTS')
        if 'bench' in t:
            tags.append('PREVENT_BENCH')
        if _contains(t, "pokemon {ex}", "pokémon {ex}", "pokemon ex", "pokémon ex"):
            tags.append('PREVENT_FROM_EX')
        if 'basic' in t and ('{ex}' in t or ' ex' in t):
            tags.append('PREVENT_FROM_BASIC_EX')
        if 'tera' in t:
            tags.append('PREVENT_FROM_TERA')
        if 'have an ability' in t or 'that have an ability' in t:
            tags.append('PREVENT_FROM_ABILITY')
        if 'special energy attached' in t:
            tags.append('PREVENT_FROM_SPECIAL_ENERGY')
        if '200 or more' in t:
            tags.append('PREVENT_DAMAGE_GE_200')
        if "don't have a rule box" in t or 'do not have a rule box' in t:
            tags.append('PROTECT_NO_RULEBOX')
    if 'prevent all damage counters' in t and 'bench' in t:
        tags.append('PREVENT_BENCH_COUNTERS')
    if 'once during your turn' in t:
        tags.append('ONCE_PER_TURN')
    if 'when you play this pokemon' in t or 'when you play this pokémon' in t:
        tags.append('ON_PLAY')
        if 'evolve' in t:
            tags.append('ON_EVOLVE')
    if 'search your deck' in t:
        tags.append('SEARCH_DECK')
    if 'attach' in t and 'energy' in t:
        tags.append('ENERGY_ACCELERATION')
    if 'draw' in t:
        tags.append('DRAW')
    if 'discard' in t:
        tags.append('DISCARD')
    if 'damage counter' in t:
        tags.append('DAMAGE_COUNTER')
        if 'move' in t:
            tags.append('MOVE_DAMAGE_COUNTER')
    if 'heal' in t or 'remove all damage' in t:
        tags.append('HEAL')
    if 'switch' in t:
        tags.append('SWITCH')
    if 'cannot retreat' in t or "can't retreat" in t:
        tags.append('NO_RETREAT')
    return tuple(dict.fromkeys(tags))


@lru_cache(maxsize=None)
def attack_tags_from_text(text: str) -> tuple[str, ...]:
    t = normalize(text)
    tags: list[str] = []
    if ((_contains(t, "isn't affected", 'is not affected')) and 'any effects' in t and ('active pokemon' in t or 'active pokémon' in t)):
        tags.append('BYPASS_ACTIVE_EFFECTS')
    if ('weakness' in t and 'resistance' in t and
            _contains(t, "isn't affected", 'is not affected', "don't apply", 'do not apply')):
        tags.append('IGNORE_WEAKNESS_RESISTANCE')
    if 'benched pokemon' in t or 'benched pokémon' in t:
        if 'damage counter' in t:
            tags.append('BENCH_DAMAGE_COUNTERS')
        elif 'damage' in t:
            tags.append('BENCH_DAMAGE')
    if 'for each of your opponent' in t and ('{ex}' in t or ' ex' in t):
        tags.append('SCALE_OPPONENT_EX')
    if 'for each' in t and 'energy' in t:
        tags.append('SCALE_ENERGY')
    if 'attach' in t and 'energy' in t:
        tags.append('ENERGY_ACCELERATION')
    if 'discard' in t and 'energy' in t:
        tags.append('DISCARD_ENERGY')
    if 'discard' in t and 'stadium' in t:
        tags.append('DISCARD_STADIUM')
    if 'draw' in t:
        tags.append('DRAW')
    if 'switch' in t:
        tags.append('SWITCH')
    if 'heal' in t or 'remove damage' in t:
        tags.append('HEAL')
    if 'damage counter' in t:
        tags.append('DAMAGE_COUNTER')
    if 'confused' in t:
        tags.append('CONFUSE')
    if 'paralyzed' in t:
        tags.append('PARALYZE')
    if 'poisoned' in t:
        tags.append('POISON')
    if 'asleep' in t:
        tags.append('SLEEP')
    if "can't use" in t or 'cannot use' in t:
        tags.append('NEXT_TURN_LOCK')
    return tuple(dict.fromkeys(tags))


@lru_cache(maxsize=None)
def card_skill_tags(card_id: int) -> tuple[str, ...]:
    c = CARD.get(int(card_id or 0))
    if c is None:
        return ()
    out: list[str] = []
    for s in list(getattr(c, 'skills', None) or []):
        out.extend(skill_tags_from_text(getattr(s, 'text', '') or ''))
    return tuple(dict.fromkeys(out))


@lru_cache(maxsize=None)
def attack_tags(attack_id: int) -> tuple[str, ...]:
    a = ATTACK.get(int(attack_id or 0))
    return attack_tags_from_text(getattr(a, 'text', '') or '') if a is not None else ()


@lru_cache(maxsize=None)
def card_semantic_tokens(card_id: int, prefix: str = 'CARD', full_text: bool = False) -> tuple[str, ...]:
    cid = int(card_id or 0)
    c = CARD.get(cid)
    if c is None:
        return (f'{prefix}_UNKNOWN=1',)
    stage = 'BASIC' if bool(getattr(c, 'basic', False)) else 'STAGE1' if bool(getattr(c, 'stage1', False)) else 'STAGE2' if bool(getattr(c, 'stage2', False)) else 'OTHER'
    toks = [
        f'{prefix}_STAGE={stage}', f'{prefix}_EX={int(bool(getattr(c,"ex",False) or getattr(c,"megaEx",False)))}',
        f'{prefix}_MEGA={int(bool(getattr(c,"megaEx",False)))}', f'{prefix}_TERA={int(bool(getattr(c,"tera",False)))}',
        f'{prefix}_ABILITY={int(bool(getattr(c,"skills",None)))}', f'{prefix}_HPBIN={min(8,max(0,int(getattr(c,"hp",0) or 0)//50))}',
        f'{prefix}_RETREAT={min(4,max(0,int(getattr(c,"retreatCost",0) or 0)))}'
    ]
    for tag in card_skill_tags(cid):
        toks.append(f'{prefix}_SKILLTAG={tag}')
    if full_text:
        words: list[str] = []
        words.extend(lexical_words(getattr(c, 'name', ''), 4))
        for s in list(getattr(c, 'skills', None) or []):
            words.extend(lexical_words((getattr(s, 'name', '') or '') + ' ' + (getattr(s, 'text', '') or ''), 12))
        for w in list(dict.fromkeys(words))[:16]:
            toks.append(f'{prefix}_TEXT={w}')
    return tuple(toks)


@lru_cache(maxsize=None)
def attack_semantic_tokens(attack_id: int, prefix: str = 'ATTACK', full_text: bool = False) -> tuple[str, ...]:
    aid = int(attack_id or 0)
    a = ATTACK.get(aid)
    if a is None:
        return (f'{prefix}_UNKNOWN=1',)
    dmg = int(getattr(a, 'damage', 0) or 0)
    energies = list(getattr(a, 'energies', None) or [])
    toks = [f'{prefix}_DMGBIN={min(10,max(0,dmg//30))}', f'{prefix}_COST={min(5,len(energies))}']
    for tag in attack_tags(aid):
        toks.append(f'{prefix}_TAG={tag}')
    if full_text:
        words = lexical_words((getattr(a, 'name', '') or '') + ' ' + (getattr(a, 'text', '') or ''), 18)
        for w in words:
            toks.append(f'{prefix}_TEXT={w}')
    return tuple(toks)


def is_ex_like(card_id: int) -> bool:
    c = CARD.get(int(card_id or 0))
    return bool(c and (getattr(c, 'ex', False) or getattr(c, 'megaEx', False)))


def has_rule_box(card_id: int) -> bool:
    # CABT's current card schema exposes ex/Mega ex explicitly.  Tera is a
    # subtype of ex in this pool and is therefore covered by is_ex_like.
    return is_ex_like(card_id)


def has_ability(card_id: int) -> bool:
    c = CARD.get(int(card_id or 0))
    return bool(c and getattr(c, 'skills', None))


def attack_bypasses_active_effects(attack_id: int) -> bool:
    return 'BYPASS_ACTIVE_EFFECTS' in attack_tags(int(attack_id or 0))


def attack_base_damage(attack_id: int) -> int:
    a = ATTACK.get(int(attack_id or 0))
    return int(getattr(a, 'damage', 0) or 0) if a is not None else 0


def attack_has_useful_effect(attack_id: int) -> bool:
    tags = set(attack_tags(int(attack_id or 0)))
    return bool(tags - {'BYPASS_ACTIVE_EFFECTS', 'IGNORE_WEAKNESS_RESISTANCE', 'NEXT_TURN_LOCK'})


def damage_prevention_applies(defender_id: int, attacker_id: int, *, target_is_bench: bool = False,
                              attacker_has_special_energy: bool = False, raw_damage: int | None = None) -> bool:
    """Conservatively evaluate text-based damage prevention for a legal attack.

    Only effects whose condition can be proved from public state are returned as
    applicable.  Unknown conditions return False rather than inventing immunity.
    """
    d = CARD.get(int(defender_id or 0)); a = CARD.get(int(attacker_id or 0))
    if d is None or a is None:
        return False
    for sk in list(getattr(d, 'skills', None) or []):
        tags = set(skill_tags_from_text(getattr(sk, 'text', '') or ''))
        if 'PREVENT_DAMAGE' not in tags:
            continue
        if 'PREVENT_BENCH' in tags and not target_is_bench:
            continue
        if 'PREVENT_FROM_BASIC_EX' in tags and not (bool(getattr(a, 'basic', False)) and is_ex_like(attacker_id)):
            continue
        if 'PREVENT_FROM_EX' in tags and not is_ex_like(attacker_id):
            continue
        if 'PREVENT_FROM_TERA' in tags and not bool(getattr(a, 'tera', False)):
            continue
        if 'PREVENT_FROM_ABILITY' in tags and not has_ability(attacker_id):
            continue
        if 'PREVENT_FROM_SPECIAL_ENERGY' in tags and not attacker_has_special_energy:
            continue
        if 'PREVENT_DAMAGE_GE_200' in tags and (raw_damage is None or int(raw_damage) < 200):
            continue
        if 'PROTECT_NO_RULEBOX' in tags and has_rule_box(defender_id):
            continue
        return True
    return False


def global_damage_prevention_applies(effect_card_ids: Iterable[int], defender_id: int, attacker_id: int, *,
                                      target_is_bench: bool = False,
                                      attacker_has_special_energy: bool = False,
                                      raw_damage: int | None = None) -> bool:
    """Evaluate Stadium/Trainer/Pokémon global text against the current attack.

    This uses exactly the same conservative conditions as a defender's own
    Ability, but reads the skills printed on every currently active public
    effect card.  It lets a never-seen Stadium such as Neutralization Zone be
    understood from text rather than from a hard-coded card ID.
    """
    d = CARD.get(int(defender_id or 0)); a = CARD.get(int(attacker_id or 0))
    if d is None or a is None:
        return False
    for source_id in effect_card_ids or ():
        source = CARD.get(int(source_id or 0))
        if source is None:
            continue
        for sk in list(getattr(source, 'skills', None) or []):
            tags = set(skill_tags_from_text(getattr(sk, 'text', '') or ''))
            if 'PREVENT_DAMAGE' not in tags:
                continue
            if 'PREVENT_BENCH' in tags and not target_is_bench:
                continue
            if 'PREVENT_FROM_BASIC_EX' in tags and not (bool(getattr(a, 'basic', False)) and is_ex_like(attacker_id)):
                continue
            if 'PREVENT_FROM_EX' in tags and not is_ex_like(attacker_id):
                continue
            if 'PREVENT_FROM_TERA' in tags and not bool(getattr(a, 'tera', False)):
                continue
            if 'PREVENT_FROM_ABILITY' in tags and not has_ability(attacker_id):
                continue
            if 'PREVENT_FROM_SPECIAL_ENERGY' in tags and not attacker_has_special_energy:
                continue
            if 'PREVENT_DAMAGE_GE_200' in tags and (raw_damage is None or int(raw_damage) < 200):
                continue
            if 'PROTECT_NO_RULEBOX' in tags and has_rule_box(defender_id):
                continue
            return True
    return False


def describe_card(card_id: int) -> dict:
    c = CARD.get(int(card_id or 0))
    if c is None:
        return {'card_id': int(card_id or 0), 'unknown': True}
    return {
        'card_id': int(c.cardId), 'name': c.name, 'skill_tags': list(card_skill_tags(c.cardId)),
        'attacks': [{'attack_id': int(aid), 'name': getattr(ATTACK.get(int(aid)), 'name', ''), 'tags': list(attack_tags(int(aid)))} for aid in (getattr(c, 'attacks', None) or [])]
    }
