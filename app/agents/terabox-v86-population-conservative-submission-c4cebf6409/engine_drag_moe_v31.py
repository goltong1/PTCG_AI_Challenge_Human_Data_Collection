"""Guarded generic/specialist mixture for the Dragapult matchup."""
from __future__ import annotations
import hashlib, importlib.util, os, sys

R = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if R not in sys.path:
    sys.path.insert(0, R)

def _load(tag):
    path = os.path.join(R, f'policy_{tag}.py')
    name = '_tera_v31_moe_' + tag + '_' + hashlib.sha1((R + tag).encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

GENERIC = _load('generic')
SPECIALIST = _load('dragapult')

import engine_matchup_v31 as H

PLAY, ATTACH, ATTACK, RETREAT = 7, 8, 13, 12
CLEF, KANGA, TEAL, LATIAS, PRISM = 272, 756, 96, 184, 16
DRAG = {119, 120, 121}
STATS = {'decisions': 0, 'advisor_overrides': 0, 'search': 0, 'attach': 0}

def _board(raw, own=True):
    current = raw.get('current') or {}; players = current.get('players') or []
    if len(players) != 2:
        return []
    me = int(current.get('yourIndex') or 0); player = players[me if own else 1-me]
    return [x for x in list(player.get('active') or []) + list(player.get('bench') or []) if x]

def _chosen(raw, output):
    try:
        return (raw.get('select') or {}).get('option')[int(output[0])]
    except Exception:
        return None

def _safe_advice(raw, generic, specialist):
    if not isinstance(specialist, list) or len(specialist) != 1 or specialist == generic:
        return generic
    so = _chosen(raw, specialist); go = _chosen(raw, generic)
    if so is None:
        return generic
    context = int(((raw.get('select') or {}).get('context') or 0))
    own = _board(raw, True); opponent = _board(raw, False)
    own_ids = {H._cid(x) for x in own}; opponent_ids = {H._cid(x) for x in opponent}
    if not (opponent_ids & DRAG):
        return generic

    # Search advice is accepted only after the opening infrastructure exists.
    if context != 0:
        source_id = H._cid(H._source(raw, so))
        safe = False
        if source_id == CLEF:
            safe = KANGA in own_ids and TEAL in own_ids and LATIAS in own_ids
        elif source_id == LATIAS:
            clef = next((x for x in own if H._cid(x) == CLEF), None)
            safe = clef is not None and H._missing(clef) <= 1
        elif source_id == PRISM:
            safe = CLEF in own_ids
        if safe:
            STATS['advisor_overrides'] += 1; STATS['search'] += 1
            return specialist
        return generic

    # Main-action advice is limited to an attachment that completes Clefairy,
    # or establishes its first Psychic source without breaking another attack.
    if int(so.get('type', -1)) == ATTACH:
        target = H._target(raw, so); energy = H._energy_type(H._source(raw, so))
        if H._cid(target) == CLEF and energy is not None:
            before, after = H._missing(target), H._missing(target, energy)
            base_target = H._target(raw, go) if go is not None else None
            base_energy = H._energy_type(H._source(raw, go)) if go is not None else None
            base_completes = bool(base_target is not None and base_energy is not None
                                  and H._missing(base_target) > 0 and H._missing(base_target, base_energy) == 0)
            first_psychic = energy == 10 and 10 not in H._pool(target)
            if after < before and (after == 0 or first_psychic) and not base_completes:
                STATS['advisor_overrides'] += 1; STATS['attach'] += 1
                return specialist
    return generic

def agent(observation):
    if observation.get('select') is None and observation.get('current') is None:
        STATS.update({'decisions': 0, 'advisor_overrides': 0, 'search': 0, 'attach': 0})
        deck = GENERIC.agent(observation)
        try: SPECIALIST.agent(observation)
        except Exception: pass
        return deck
    generic = GENERIC.agent(observation)
    try: specialist = SPECIALIST.agent(observation)
    except Exception: specialist = generic
    STATS['decisions'] += 1
    return _safe_advice(observation, generic, specialist)
