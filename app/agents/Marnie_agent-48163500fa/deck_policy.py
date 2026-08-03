"""v30 reasoned self-play efficiency Marnie Grimmsnarl ex agent for CABT.

The public entry point is agent(observation: dict) -> list[int].
The policy combines a persistent finite-state machine, public-board forecasting,
general tactical safeguards, replay regularization, and a conservative policy
delta distilled through multi-generation self-play and cross-play evaluation.
"""
from __future__ import annotations

import os
import json
import re
import unicodedata
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from cg.api import (
    AreaType,
    CardType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

# ---------------------------------------------------------------------------
# Submission-safe deck loading. Kaggle may exec() main.py without __file__.
# ---------------------------------------------------------------------------
_DECK_PATH = "deck.csv"
if not os.path.exists(_DECK_PATH):
    _DECK_PATH = "/kaggle_simulations/agent/deck.csv"
if not os.path.exists(_DECK_PATH):
    raise FileNotFoundError("deck.csv not found")
with open(_DECK_PATH, "r", encoding="utf-8") as _f:
    MY_DECK = [int(line.strip()) for line in _f if line.strip()]
if len(MY_DECK) != 60:
    raise ValueError(f"deck.csv must contain exactly 60 cards, got {len(MY_DECK)}")

# Replay and self-play correction memory. The final policy uses a conservative
# 30% blend of the accepted self-play delta to reduce overfitting.
_CORRECTION_PATH = "replay_corrections_v19.json"
if not os.path.exists(_CORRECTION_PATH):
    _CORRECTION_PATH = "/kaggle_simulations/agent/replay_corrections_v19.json"
try:
    with open(_CORRECTION_PATH, "r", encoding="utf-8") as _cf:
        _REPLAY_CORRECTION = json.load(_cf).get("entries", {})
except Exception:
    _REPLAY_CORRECTION = {}
REPLAY_CORRECTION_SCALE = 1

# v24 rule-first residual memory.  The v19 deterministic policy remains the
# primary controller; learned data is isolated by opponent archetype and only
# adds small bonuses to safe single-choice decisions.
_RESIDUAL_PATH = "residual_policy_v24.json"
if not os.path.exists(_RESIDUAL_PATH):
    _RESIDUAL_PATH = "/kaggle_simulations/agent/residual_policy_v24.json"
try:
    with open(_RESIDUAL_PATH, "r", encoding="utf-8") as _rf:
        _RESIDUAL_DATA = json.load(_rf)
        _RESIDUAL_POLICIES = _RESIDUAL_DATA.get("policies", {})
        _RESIDUAL_ENABLED = set(_RESIDUAL_DATA.get("enabled_archetypes", []))
        RESIDUAL_SCALE = float(_RESIDUAL_DATA.get("scale", 1.0))
except Exception:
    _RESIDUAL_POLICIES = {}
    _RESIDUAL_ENABLED = set()
    RESIDUAL_SCALE = 0.0

# Correctly aligned v25 replay residuals are independent from the proven v24
# memory. They are enabled one opponent archetype at a time only after local
# cross-play confirms a gain.
_CF_PATH = "counterfactual_policy_v25.json"
if not os.path.exists(_CF_PATH):
    _CF_PATH = "/kaggle_simulations/agent/counterfactual_policy_v25.json"
try:
    with open(_CF_PATH, "r", encoding="utf-8") as _cf:
        _CF_DATA = json.load(_cf)
        _CF_POLICIES = _CF_DATA.get("policies", {})
        _CF_ENABLED = set(_CF_DATA.get("enabled_archetypes", []))
        CF_SCALE = float(_CF_DATA.get("scale", 1.0))
except Exception:
    _CF_POLICIES = {}
    _CF_ENABLED = set()
    CF_SCALE = 0.0
_CF_HIT_COUNT = 0



# v37 exact loss memory: only high-margin states from aligned challenge replays.
_LOSS_MEMORY_V37_PATH = "loss_memory_v37.json"
if not os.path.exists(_LOSS_MEMORY_V37_PATH):
    _LOSS_MEMORY_V37_PATH = "/kaggle_simulations/agent/loss_memory_v37.json"
try:
    with open(_LOSS_MEMORY_V37_PATH, "r", encoding="utf-8") as _lmf:
        _LOSS_MEMORY_V37 = json.load(_lmf).get("entries", {})
except Exception:
    _LOSS_MEMORY_V37 = {}
_LOSS_MEMORY_V37_HITS = 0

CARD = {c.cardId: c for c in all_card_data()}
ATTACK = {a.attackId: a for a in all_attack()}

# Pokémon
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
MUNKIDORI = 112
SNORUNT = 860
FROSLASS = 104
YVELTAL = 689
CHI_YU = 31
BUDEW = 235  # retained as compatibility constant; not in the optimized list
ABRA = 741
KADABRA = 742
ALAKAZAM = 743
ALAKAZAM_ALT = 245

# Trainers / Energy
LILLIES_DETERMINATION = 1227
TEAM_ROCKETS_PETREL = 1219
BOSSS_ORDERS = 1182
JUDGE = 1213
POKE_PAD = 1152
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
NIGHT_STRETCHER = 1097
BUDDY_BUDDY_POFFIN = 1086
RARE_CANDY = 1079
SECRET_BOX = 1092
AIR_BALLOON = 1174
HANDHELD_FAN = 1161
UNFAIR_STAMP = 1080
SPIKEMUTH_GYM = 1259
BATTLE_CAGE = 1264
CRUSHING_HAMMER = 1120
DAWN = 1231
FIRE_ENERGY = 2
DARK_ENERGY = 7
PRISM_ENERGY = 16
PSYCHIC_ENERGY = 5

# Known attacks for this list
FILCH = 934
IMP_PUNCH = 935
MORGREM_PUNCH = 936
SHADOW_BULLET = 937
MIND_BEND = 141
CHILLY = 1239
FROST_SMASH = 131
CLUTCH = 997
DARK_FEATHER = 998
GROUND_MELTER = 20
ITCHY_POLLEN = 323

MARNIE_LINE = {IMPIDIMP, MORGREM, GRIMMSNARL_EX}
OUR_POKEMON = {IMPIDIMP, MORGREM, GRIMMSNARL_EX, MUNKIDORI, SNORUNT, FROSLASS, YVELTAL, CHI_YU}
DRAW_SUPPORTERS = {LILLIES_DETERMINATION, TEAM_ROCKETS_PETREL, JUDGE}
NEG = -10**9

# BEGIN REPLAY-TRAINED PRIORS
LEARNED_PRIOR = {
    ('arch', 0, 0, 10, 112, 0, 'alakazam'): 188,
    ('arch', 0, 0, 10, 112, 0, 'crustle'): 457,
    ('arch', 0, 0, 10, 112, 0, 'cynthia'): -647,
    ('arch', 0, 0, 10, 112, 0, 'dragapult'): 201,
    ('arch', 0, 0, 10, 112, 0, 'lucario'): 1299,
    ('arch', 0, 0, 10, 112, 0, 'marnie'): 72,
    ('arch', 0, 0, 10, 112, 0, 'rocket'): 173,
    ('arch', 0, 0, 10, 112, 0, 'unknown'): 323,
    ('arch', 0, 0, 10, 1257, 0, 'rocket'): 588,
    ('arch', 0, 0, 10, 1259, 0, 'alakazam'): -824,
    ('arch', 0, 0, 10, 1259, 0, 'crustle'): -1271,
    ('arch', 0, 0, 10, 1259, 0, 'cynthia'): -938,
    ('arch', 0, 0, 10, 1259, 0, 'dragapult'): -887,
    ('arch', 0, 0, 10, 1259, 0, 'lucario'): -817,
    ('arch', 0, 0, 10, 1259, 0, 'marnie'): -1318,
    ('arch', 0, 0, 10, 1259, 0, 'rocket'): -494,
    ('arch', 0, 0, 10, 1259, 0, 'unknown'): -476,
    ('arch', 0, 0, 12, 0, 0, 'alakazam'): -3514,
    ('arch', 0, 0, 12, 0, 0, 'crustle'): -3228,
    ('arch', 0, 0, 12, 0, 0, 'cynthia'): -3892,
    ('arch', 0, 0, 12, 0, 0, 'dragapult'): -2512,
    ('arch', 0, 0, 12, 0, 0, 'lucario'): -3792,
    ('arch', 0, 0, 12, 0, 0, 'marnie'): -3006,
    ('arch', 0, 0, 12, 0, 0, 'rocket'): -4394,
    ('arch', 0, 0, 12, 0, 0, 'unknown'): -2278,
    ('arch', 0, 0, 13, 934, 0, 'alakazam'): -3219,
    ('arch', 0, 0, 13, 934, 0, 'cynthia'): -1466,
    ('arch', 0, 0, 13, 934, 0, 'marnie'): -2663,
    ('arch', 0, 0, 13, 934, 0, 'rocket'): -2625,
    ('arch', 0, 0, 13, 934, 0, 'unknown'): -201,
    ('arch', 0, 0, 13, 935, 0, 'alakazam'): -3219,
    ('arch', 0, 0, 13, 935, 0, 'cynthia'): -2708,
    ('arch', 0, 0, 13, 935, 0, 'marnie'): -3807,
    ('arch', 0, 0, 13, 935, 0, 'rocket'): -4290,
    ('arch', 0, 0, 13, 935, 0, 'unknown'): -2944,
    ('arch', 0, 0, 13, 936, 0, 'alakazam'): -1946,
    ('arch', 0, 0, 13, 936, 0, 'cynthia'): -2944,
    ('arch', 0, 0, 13, 936, 0, 'marnie'): -2197,
    ('arch', 0, 0, 13, 936, 0, 'rocket'): -2197,
    ('arch', 0, 0, 13, 937, 0, 'alakazam'): -1463,
    ('arch', 0, 0, 13, 937, 0, 'crustle'): -1471,
    ('arch', 0, 0, 13, 937, 0, 'cynthia'): -452,
    ('arch', 0, 0, 13, 937, 0, 'dragapult'): -1213,
    ('arch', 0, 0, 13, 937, 0, 'lucario'): -1867,
    ('arch', 0, 0, 13, 937, 0, 'marnie'): -1702,
    ('arch', 0, 0, 13, 937, 0, 'rocket'): -1591,
    ('arch', 0, 0, 13, 937, 0, 'unknown'): -1249,
    ('arch', 0, 0, 14, 0, 0, 'alakazam'): -5680,
    ('arch', 0, 0, 14, 0, 0, 'crustle'): -2872,
    ('arch', 0, 0, 14, 0, 0, 'cynthia'): -4078,
    ('arch', 0, 0, 14, 0, 0, 'dragapult'): -3970,
    ('arch', 0, 0, 14, 0, 0, 'lucario'): -3160,
    ('arch', 0, 0, 14, 0, 0, 'marnie'): -4206,
    ('arch', 0, 0, 14, 0, 0, 'rocket'): -3363,
    ('arch', 0, 0, 14, 0, 0, 'unknown'): -2050,
    ('arch', 0, 0, 7, 1079, 0, 'alakazam'): 0,
    ('arch', 0, 0, 7, 1079, 0, 'crustle'): -847,
    ('arch', 0, 0, 7, 1079, 0, 'lucario'): -511,
    ('arch', 0, 0, 7, 1079, 0, 'marnie'): -943,
    ('arch', 0, 0, 7, 1079, 0, 'rocket'): 310,
    ('arch', 0, 0, 7, 1079, 0, 'unknown'): -427,
    ('arch', 0, 0, 7, 1080, 0, 'alakazam'): -435,
    ('arch', 0, 0, 7, 1080, 0, 'crustle'): -999,
    ('arch', 0, 0, 7, 1080, 0, 'cynthia'): -511,
    ('arch', 0, 0, 7, 1080, 0, 'lucario'): -1466,
    ('arch', 0, 0, 7, 1080, 0, 'marnie'): -596,
    ('arch', 0, 0, 7, 1080, 0, 'rocket'): -1149,
    ('arch', 0, 0, 7, 1080, 0, 'unknown'): -588,
    ('arch', 0, 0, 7, 1086, 0, 'alakazam'): -2833,
    ('arch', 0, 0, 7, 1086, 0, 'crustle'): -2565,
    ('arch', 0, 0, 7, 1086, 0, 'lucario'): -999,
    ('arch', 0, 0, 7, 1086, 0, 'marnie'): -2358,
    ('arch', 0, 0, 7, 1086, 0, 'rocket'): -1726,
    ('arch', 0, 0, 7, 1086, 0, 'unknown'): -263,
    ('arch', 0, 0, 7, 1097, 0, 'alakazam'): -1290,
    ('arch', 0, 0, 7, 1097, 0, 'crustle'): -1327,
    ('arch', 0, 0, 7, 1097, 0, 'cynthia'): -1946,
    ('arch', 0, 0, 7, 1097, 0, 'lucario'): 2398,
    ('arch', 0, 0, 7, 1097, 0, 'marnie'): -562,
    ('arch', 0, 0, 7, 1097, 0, 'rocket'): -879,
    ('arch', 0, 0, 7, 1097, 0, 'unknown'): 201,
    ('arch', 0, 0, 7, 112, 0, 'alakazam'): 887,
    ('arch', 0, 0, 7, 112, 0, 'crustle'): 0,
    ('arch', 0, 0, 7, 112, 0, 'lucario'): 788,
    ('arch', 0, 0, 7, 112, 0, 'marnie'): 261,
    ('arch', 0, 0, 7, 112, 0, 'rocket'): 302,
    ('arch', 0, 0, 7, 112, 0, 'unknown'): 854,
    ('arch', 0, 0, 7, 1122, 0, 'crustle'): -1946,
    ('arch', 0, 0, 7, 1122, 0, 'lucario'): 511,
    ('arch', 0, 0, 7, 1122, 0, 'marnie'): -480,
    ('arch', 0, 0, 7, 1122, 0, 'rocket'): 588,
    ('arch', 0, 0, 7, 1122, 0, 'unknown'): -995,
    ('arch', 0, 0, 7, 1137, 0, 'marnie'): -1466,
    ('arch', 0, 0, 7, 1137, 0, 'unknown'): -2398,
    ('arch', 0, 0, 7, 1152, 0, 'alakazam'): 379,
    ('arch', 0, 0, 7, 1152, 0, 'crustle'): 191,
    ('arch', 0, 0, 7, 1152, 0, 'cynthia'): -1435,
    ('arch', 0, 0, 7, 1152, 0, 'dragapult'): 0,
    ('arch', 0, 0, 7, 1152, 0, 'lucario'): -647,
    ('arch', 0, 0, 7, 1152, 0, 'marnie'): -913,
    ('arch', 0, 0, 7, 1152, 0, 'rocket'): -395,
    ('arch', 0, 0, 7, 1152, 0, 'unknown'): -542,
    ('arch', 0, 0, 7, 1182, 0, 'alakazam'): -3045,
    ('arch', 0, 0, 7, 1182, 0, 'crustle'): -2708,
    ('arch', 0, 0, 7, 1182, 0, 'cynthia'): -336,
    ('arch', 0, 0, 7, 1182, 0, 'dragapult'): -1299,
    ('arch', 0, 0, 7, 1182, 0, 'lucario'): -3367,
    ('arch', 0, 0, 7, 1182, 0, 'marnie'): -2156,
    ('arch', 0, 0, 7, 1182, 0, 'rocket'): -3864,
    ('arch', 0, 0, 7, 1182, 0, 'unknown'): -2172,
    ('arch', 0, 0, 7, 1219, 0, 'alakazam'): -1421,
    ('arch', 0, 0, 7, 1219, 0, 'crustle'): -2001,
    ('arch', 0, 0, 7, 1219, 0, 'cynthia'): -1435,
    ('arch', 0, 0, 7, 1219, 0, 'dragapult'): -1846,
    ('arch', 0, 0, 7, 1219, 0, 'lucario'): -1776,
    ('arch', 0, 0, 7, 1219, 0, 'marnie'): -2166,
    ('arch', 0, 0, 7, 1219, 0, 'rocket'): -2044,
    ('arch', 0, 0, 7, 1219, 0, 'unknown'): -2109,
    ('arch', 0, 0, 7, 1227, 0, 'alakazam'): -2663,
    ('arch', 0, 0, 7, 1227, 0, 'crustle'): -1474,
    ('arch', 0, 0, 7, 1227, 0, 'dragapult'): -1099,
    ('arch', 0, 0, 7, 1227, 0, 'lucario'): -1452,
    ('arch', 0, 0, 7, 1227, 0, 'marnie'): -1495,
    ('arch', 0, 0, 7, 1227, 0, 'rocket'): -1593,
    ('arch', 0, 0, 7, 1227, 0, 'unknown'): -956,
    ('arch', 0, 0, 7, 1231, 0, 'alakazam'): -2054,
    ('arch', 0, 0, 7, 1231, 0, 'crustle'): -1099,
    ('arch', 0, 0, 7, 1231, 0, 'lucario'): -3135,
    ('arch', 0, 0, 7, 1231, 0, 'marnie'): -1677,
    ('arch', 0, 0, 7, 1231, 0, 'rocket'): -1099,
    ('arch', 0, 0, 7, 1231, 0, 'unknown'): -1099,
    ('arch', 0, 0, 7, 1259, 0, 'alakazam'): 847,
    ('arch', 0, 0, 7, 1259, 0, 'crustle'): -747,
    ('arch', 0, 0, 7, 1259, 0, 'marnie'): 511,
    ('arch', 0, 0, 7, 1259, 0, 'rocket'): -552,
    ('arch', 0, 0, 7, 1259, 0, 'unknown'): 542,
    ('arch', 0, 0, 7, 646, 0, 'alakazam'): 1946,
    ('arch', 0, 0, 7, 646, 0, 'crustle'): -1099,
    ('arch', 0, 0, 7, 646, 0, 'lucario'): 1099,
    ('arch', 0, 0, 7, 646, 0, 'marnie'): -1157,
    ('arch', 0, 0, 7, 646, 0, 'rocket'): -91,
    ('arch', 0, 0, 7, 646, 0, 'unknown'): 807,
    ('arch', 0, 0, 7, 860, 0, 'alakazam'): -511,
    ('arch', 0, 0, 7, 860, 0, 'crustle'): 1946,
    ('arch', 0, 0, 7, 860, 0, 'lucario'): -2565,
    ('arch', 0, 0, 7, 860, 0, 'marnie'): -5849,
    ('arch', 0, 0, 7, 860, 0, 'rocket'): -452,
    ('arch', 0, 0, 7, 860, 0, 'unknown'): -1758,
    ('arch', 0, 0, 8, 7, 104, 'alakazam'): -2833,
    ('arch', 0, 0, 8, 7, 104, 'crustle'): -4466,
    ('arch', 0, 0, 8, 7, 104, 'dragapult'): -2979,
    ('arch', 0, 0, 8, 7, 104, 'lucario'): -3807,
    ('arch', 0, 0, 8, 7, 104, 'marnie'): -3455,
    ('arch', 0, 0, 8, 7, 104, 'rocket'): -4122,
    ('arch', 0, 0, 8, 7, 104, 'unknown'): -3045,
    ('arch', 0, 0, 8, 7, 112, 'alakazam'): -2349,
    ('arch', 0, 0, 8, 7, 112, 'crustle'): -3061,
    ('arch', 0, 0, 8, 7, 112, 'cynthia'): -511,
    ('arch', 0, 0, 8, 7, 112, 'dragapult'): -3730,
    ('arch', 0, 0, 8, 7, 112, 'lucario'): -2180,
    ('arch', 0, 0, 8, 7, 112, 'marnie'): -2968,
    ('arch', 0, 0, 8, 7, 112, 'rocket'): -2431,
    ('arch', 0, 0, 8, 7, 112, 'unknown'): -2438,
    ('arch', 0, 0, 8, 7, 646, 'alakazam'): -3850,
    ('arch', 0, 0, 8, 7, 646, 'crustle'): -4043,
    ('arch', 0, 0, 8, 7, 646, 'dragapult'): -4443,
    ('arch', 0, 0, 8, 7, 646, 'lucario'): -3219,
    ('arch', 0, 0, 8, 7, 646, 'marnie'): -6509,
    ('arch', 0, 0, 8, 7, 646, 'rocket'): -3649,
    ('arch', 0, 0, 8, 7, 646, 'unknown'): -4548,
    ('arch', 0, 0, 8, 7, 647, 'alakazam'): -3730,
    ('arch', 0, 0, 8, 7, 647, 'crustle'): -5525,
    ('arch', 0, 0, 8, 7, 647, 'dragapult'): -3761,
    ('arch', 0, 0, 8, 7, 647, 'lucario'): -3664,
    ('arch', 0, 0, 8, 7, 647, 'marnie'): -6544,
    ('arch', 0, 0, 8, 7, 647, 'rocket'): -3932,
    ('arch', 0, 0, 8, 7, 647, 'unknown'): -5004,
    ('arch', 0, 0, 8, 7, 648, 'alakazam'): -5451,
    ('arch', 0, 0, 8, 7, 648, 'crustle'): -3164,
    ('arch', 0, 0, 8, 7, 648, 'dragapult'): -4369,
    ('arch', 0, 0, 8, 7, 648, 'lucario'): -2689,
    ('arch', 0, 0, 8, 7, 648, 'marnie'): -3832,
    ('arch', 0, 0, 8, 7, 648, 'rocket'): -3836,
    ('arch', 0, 0, 8, 7, 648, 'unknown'): -3824,
    ('arch', 0, 0, 8, 7, 860, 'alakazam'): -3434,
    ('arch', 0, 0, 8, 7, 860, 'crustle'): -5333,
    ('arch', 0, 0, 8, 7, 860, 'dragapult'): -1946,
    ('arch', 0, 0, 8, 7, 860, 'lucario'): -2708,
    ('arch', 0, 0, 8, 7, 860, 'marnie'): -3761,
    ('arch', 0, 0, 8, 7, 860, 'rocket'): -5069,
    ('arch', 0, 0, 8, 7, 860, 'unknown'): -4498,
    ('arch', 0, 0, 9, 104, 860, 'alakazam'): 1099,
    ('arch', 0, 0, 9, 104, 860, 'crustle'): 1735,
    ('arch', 0, 0, 9, 104, 860, 'cynthia'): 511,
    ('arch', 0, 0, 9, 104, 860, 'lucario'): 511,
    ('arch', 0, 0, 9, 104, 860, 'marnie'): 511,
    ('arch', 0, 0, 9, 104, 860, 'rocket'): -111,
    ('arch', 0, 0, 9, 104, 860, 'unknown'): -236,
    ('arch', 0, 0, 9, 647, 646, 'alakazam'): -601,
    ('arch', 0, 0, 9, 647, 646, 'crustle'): -1006,
    ('arch', 0, 0, 9, 647, 646, 'dragapult'): -511,
    ('arch', 0, 0, 9, 647, 646, 'lucario'): -143,
    ('arch', 0, 0, 9, 647, 646, 'marnie'): -1471,
    ('arch', 0, 0, 9, 647, 646, 'rocket'): -631,
    ('arch', 0, 0, 9, 647, 646, 'unknown'): -504,
    ('arch', 0, 0, 9, 648, 647, 'alakazam'): -1920,
    ('arch', 0, 0, 9, 648, 647, 'crustle'): -1609,
    ('arch', 0, 0, 9, 648, 647, 'dragapult'): -2398,
    ('arch', 0, 0, 9, 648, 647, 'lucario'): -647,
    ('arch', 0, 0, 9, 648, 647, 'marnie'): -1664,
    ('arch', 0, 0, 9, 648, 647, 'rocket'): -111,
    ('arch', 0, 0, 9, 648, 647, 'unknown'): -2600,
    ('arch', 1, 0, 3, 112, 0, 'unknown'): -108,
    ('arch', 1, 0, 3, 646, 0, 'unknown'): 1242,
    ('arch', 1, 0, 3, 860, 0, 'unknown'): 1846,
    ('arch', 13, 112, 3, 104, 0, 'marnie'): -2989,
    ('arch', 13, 112, 3, 1071, 0, 'dragapult'): -1946,
    ('arch', 13, 112, 3, 112, 0, 'dragapult'): -1099,
    ('arch', 13, 112, 3, 112, 0, 'marnie'): -514,
    ('arch', 13, 112, 3, 119, 0, 'dragapult'): -2197,
    ('arch', 13, 112, 3, 119, 0, 'unknown'): -2833,
    ('arch', 13, 112, 3, 120, 0, 'dragapult'): 1946,
    ('arch', 13, 112, 3, 140, 0, 'alakazam'): -2565,
    ('arch', 13, 112, 3, 140, 0, 'dragapult'): -2197,
    ('arch', 13, 112, 3, 175, 0, 'unknown'): -2197,
    ('arch', 13, 112, 3, 272, 0, 'unknown'): 2197,
    ('arch', 13, 112, 3, 305, 0, 'alakazam'): -2565,
    ('arch', 13, 112, 3, 341, 0, 'cynthia'): 511,
    ('arch', 13, 112, 3, 342, 0, 'cynthia'): -956,
    ('arch', 13, 112, 3, 343, 0, 'alakazam'): 368,
    ('arch', 13, 112, 3, 344, 0, 'crustle'): -1099,
    ('arch', 13, 112, 3, 345, 0, 'crustle'): 752,
    ('arch', 13, 112, 3, 380, 0, 'cynthia'): -1099,
    ('arch', 13, 112, 3, 381, 0, 'cynthia'): -2833,
    ('arch', 13, 112, 3, 400, 0, 'rocket'): 636,
    ('arch', 13, 112, 3, 401, 0, 'rocket'): -111,
    ('arch', 13, 112, 3, 414, 0, 'rocket'): -1006,
    ('arch', 13, 112, 3, 431, 0, 'rocket'): -3135,
    ('arch', 13, 112, 3, 434, 0, 'rocket'): -1582,
    ('arch', 13, 112, 3, 646, 0, 'marnie'): -1046,
    ('arch', 13, 112, 3, 647, 0, 'marnie'): -2903,
    ('arch', 13, 112, 3, 648, 0, 'marnie'): -2278,
    ('arch', 13, 112, 3, 673, 0, 'lucario'): 336,
    ('arch', 13, 112, 3, 675, 0, 'lucario'): -2398,
    ('arch', 13, 112, 3, 676, 0, 'lucario'): -2398,
    ('arch', 13, 112, 3, 678, 0, 'lucario'): -2565,
    ('arch', 13, 112, 3, 689, 0, 'marnie'): -2565,
    ('arch', 13, 112, 3, 741, 0, 'alakazam'): 310,
    ('arch', 13, 112, 3, 742, 0, 'alakazam'): -1099,
    ('arch', 13, 112, 3, 743, 0, 'alakazam'): -3664,
    ('arch', 13, 112, 3, 756, 0, 'crustle'): -1184,
    ('arch', 13, 112, 3, 756, 0, 'unknown'): -336,
    ('arch', 13, 112, 3, 860, 0, 'marnie'): -563,
    ('arch', 15, 648, 3, 104, 0, 'marnie'): -2361,
    ('arch', 15, 648, 3, 112, 0, 'dragapult'): -511,
    ('arch', 15, 648, 3, 112, 0, 'marnie'): -231,
    ('arch', 15, 648, 3, 119, 0, 'dragapult'): -1946,
    ('arch', 15, 648, 3, 119, 0, 'unknown'): 0,
    ('arch', 15, 648, 3, 140, 0, 'alakazam'): -2565,
    ('arch', 15, 648, 3, 175, 0, 'unknown'): -1846,
    ('arch', 15, 648, 3, 184, 0, 'unknown'): 1099,
    ('arch', 15, 648, 3, 24, 0, 'unknown'): -2398,
    ('arch', 15, 648, 3, 272, 0, 'unknown'): 1946,
    ('arch', 15, 648, 3, 305, 0, 'alakazam'): -2152,
    ('arch', 15, 648, 3, 341, 0, 'cynthia'): -847,
    ('arch', 15, 648, 3, 343, 0, 'alakazam'): 788,
    ('arch', 15, 648, 3, 345, 0, 'crustle'): 762,
    ('arch', 15, 648, 3, 379, 0, 'cynthia'): -847,
    ('arch', 15, 648, 3, 380, 0, 'cynthia'): -511,
    ('arch', 15, 648, 3, 400, 0, 'rocket'): -143,
    ('arch', 15, 648, 3, 401, 0, 'rocket'): -511,
    ('arch', 15, 648, 3, 414, 0, 'rocket'): -1609,
    ('arch', 15, 648, 3, 431, 0, 'rocket'): -2565,
    ('arch', 15, 648, 3, 434, 0, 'rocket'): -788,
    ('arch', 15, 648, 3, 646, 0, 'marnie'): -906,
    ('arch', 15, 648, 3, 646, 0, 'unknown'): -1946,
    ('arch', 15, 648, 3, 647, 0, 'marnie'): -1989,
    ('arch', 15, 648, 3, 648, 0, 'marnie'): -1665,
    ('arch', 15, 648, 3, 673, 0, 'lucario'): 1299,
    ('arch', 15, 648, 3, 675, 0, 'lucario'): -2708,
    ('arch', 15, 648, 3, 676, 0, 'lucario'): -2565,
    ('arch', 15, 648, 3, 677, 0, 'lucario'): -1946,
    ('arch', 15, 648, 3, 678, 0, 'lucario'): -1946,
    ('arch', 15, 648, 3, 741, 0, 'alakazam'): 268,
    ('arch', 15, 648, 3, 741, 0, 'unknown'): -511,
    ('arch', 15, 648, 3, 742, 0, 'alakazam'): -1170,
    ('arch', 15, 648, 3, 743, 0, 'alakazam'): -2708,
    ('arch', 15, 648, 3, 756, 0, 'crustle'): 1946,
    ('arch', 15, 648, 3, 860, 0, 'marnie'): -731,
    ('arch', 15, 648, 3, 860, 0, 'unknown'): 1946,
    ('arch', 16, 112, 3, 104, 0, 'alakazam'): -1299,
    ('arch', 16, 112, 3, 112, 0, 'alakazam'): -903,
    ('arch', 16, 112, 3, 112, 0, 'crustle'): -1130,
    ('arch', 16, 112, 3, 112, 0, 'cynthia'): 251,
    ('arch', 16, 112, 3, 112, 0, 'dragapult'): -588,
    ('arch', 16, 112, 3, 112, 0, 'lucario'): -1735,
    ('arch', 16, 112, 3, 112, 0, 'marnie'): -109,
    ('arch', 16, 112, 3, 112, 0, 'rocket'): -236,
    ('arch', 16, 112, 3, 112, 0, 'unknown'): 368,
    ('arch', 16, 112, 3, 646, 0, 'marnie'): 1609,
    ('arch', 16, 112, 3, 647, 0, 'marnie'): -1099,
    ('arch', 16, 112, 3, 648, 0, 'alakazam'): -274,
    ('arch', 16, 112, 3, 648, 0, 'crustle'): 554,
    ('arch', 16, 112, 3, 648, 0, 'dragapult'): -452,
    ('arch', 16, 112, 3, 648, 0, 'lucario'): -636,
    ('arch', 16, 112, 3, 648, 0, 'marnie'): -994,
    ('arch', 16, 112, 3, 648, 0, 'rocket'): 0,
    ('arch', 16, 112, 3, 648, 0, 'unknown'): 368,
    ('arch', 2, 0, 3, 112, 0, 'unknown'): 3219,
    ('arch', 2, 0, 3, 646, 0, 'unknown'): 2398,
    ('arch', 21, 7, 3, 646, 0, 'alakazam'): -1358,
    ('arch', 21, 7, 3, 646, 0, 'cynthia'): -368,
    ('arch', 21, 7, 3, 646, 0, 'lucario'): -452,
    ('arch', 21, 7, 3, 646, 0, 'marnie'): -1134,
    ('arch', 21, 7, 3, 646, 0, 'rocket'): -847,
    ('arch', 21, 7, 3, 646, 0, 'unknown'): -1015,
    ('arch', 21, 7, 3, 647, 0, 'alakazam'): -236,
    ('arch', 21, 7, 3, 647, 0, 'crustle'): -619,
    ('arch', 21, 7, 3, 647, 0, 'lucario'): -336,
    ('arch', 21, 7, 3, 647, 0, 'marnie'): -1248,
    ('arch', 21, 7, 3, 647, 0, 'rocket'): -619,
    ('arch', 21, 7, 3, 647, 0, 'unknown'): -1466,
    ('arch', 21, 7, 3, 648, 0, 'alakazam'): -346,
    ('arch', 21, 7, 3, 648, 0, 'crustle'): 566,
    ('arch', 21, 7, 3, 648, 0, 'cynthia'): -588,
    ('arch', 21, 7, 3, 648, 0, 'lucario'): -310,
    ('arch', 21, 7, 3, 648, 0, 'marnie'): -38,
    ('arch', 21, 7, 3, 648, 0, 'rocket'): 394,
    ('arch', 21, 7, 3, 648, 0, 'unknown'): 45,
    ('arch', 22, 648, 3, 7, 0, 'alakazam'): 40,
    ('arch', 22, 648, 3, 7, 0, 'crustle'): 588,
    ('arch', 22, 648, 3, 7, 0, 'cynthia'): 619,
    ('arch', 22, 648, 3, 7, 0, 'lucario'): 336,
    ('arch', 22, 648, 3, 7, 0, 'marnie'): 341,
    ('arch', 22, 648, 3, 7, 0, 'rocket'): -251,
    ('arch', 22, 648, 3, 7, 0, 'unknown'): 371,
    ('arch', 3, 0, 3, 104, 0, 'crustle'): -2197,
    ('arch', 3, 0, 3, 112, 0, 'alakazam'): -2398,
    ('arch', 3, 0, 3, 112, 0, 'crustle'): -2944,
    ('arch', 3, 0, 3, 112, 0, 'marnie'): -4949,
    ('arch', 3, 0, 3, 112, 0, 'rocket'): -2398,
    ('arch', 3, 0, 3, 112, 0, 'unknown'): -2197,
    ('arch', 3, 0, 3, 646, 0, 'marnie'): -2398,
    ('arch', 3, 0, 3, 646, 0, 'unknown'): -3045,
    ('arch', 3, 0, 3, 647, 0, 'alakazam'): -2398,
    ('arch', 3, 0, 3, 647, 0, 'marnie'): -3367,
    ('arch', 3, 0, 3, 647, 0, 'unknown'): -2197,
    ('arch', 3, 0, 3, 648, 0, 'alakazam'): 847,
    ('arch', 3, 0, 3, 648, 0, 'crustle'): 847,
    ('arch', 3, 0, 3, 648, 0, 'dragapult'): -511,
    ('arch', 3, 0, 3, 648, 0, 'marnie'): 1810,
    ('arch', 3, 0, 3, 648, 0, 'unknown'): 1846,
    ('arch', 3, 1182, 3, 104, 0, 'marnie'): -1299,
    ('arch', 3, 1182, 3, 112, 0, 'marnie'): -762,
    ('arch', 3, 1182, 3, 342, 0, 'cynthia'): 511,
    ('arch', 3, 1182, 3, 646, 0, 'unknown'): 0,
    ('arch', 3, 1182, 3, 647, 0, 'marnie'): -3135,
    ('arch', 3, 1182, 3, 648, 0, 'marnie'): 0,
    ('arch', 3, 1182, 3, 860, 0, 'marnie'): -1946,
    ('arch', 30, 0, 6, 112, 0, 'unknown'): 1946,
    ('arch', 30, 0, 6, 648, 0, 'alakazam'): -336,
    ('arch', 30, 0, 6, 648, 0, 'crustle'): 588,
    ('arch', 30, 0, 6, 648, 0, 'marnie'): 198,
    ('arch', 30, 0, 6, 648, 0, 'rocket'): -336,
    ('arch', 30, 0, 6, 648, 0, 'unknown'): -368,
    ('arch', 34, 0, 15, 0, 0, 'alakazam'): 2565,
    ('arch', 34, 0, 15, 0, 0, 'crustle'): 2565,
    ('arch', 34, 0, 15, 0, 0, 'marnie'): 3611,
    ('arch', 34, 0, 15, 0, 0, 'rocket'): 3045,
    ('arch', 37, 1079, 9, 648, 646, 'alakazam'): 619,
    ('arch', 37, 1079, 9, 648, 646, 'cynthia'): -511,
    ('arch', 37, 1079, 9, 648, 646, 'marnie'): 666,
    ('arch', 37, 1079, 9, 648, 646, 'rocket'): -427,
    ('arch', 37, 1079, 9, 648, 646, 'unknown'): 310,
    ('arch', 38, 0, 0, 0, 0, 'unknown'): -3219,
    ('arch', 38, 0, 0, 1, 0, 'unknown'): 310,
    ('arch', 38, 0, 0, 2, 0, 'unknown'): 1099,
    ('arch', 4, 0, 3, 104, 0, 'crustle'): -2398,
    ('arch', 4, 0, 3, 104, 0, 'cynthia'): -1946,
    ('arch', 4, 0, 3, 104, 0, 'rocket'): -2708,
    ('arch', 4, 0, 3, 112, 0, 'alakazam'): -2565,
    ('arch', 4, 0, 3, 112, 0, 'crustle'): -3434,
    ('arch', 4, 0, 3, 112, 0, 'cynthia'): -1946,
    ('arch', 4, 0, 3, 112, 0, 'lucario'): -2197,
    ('arch', 4, 0, 3, 112, 0, 'marnie'): -4905,
    ('arch', 4, 0, 3, 112, 0, 'rocket'): -3497,
    ('arch', 4, 0, 3, 646, 0, 'alakazam'): 0,
    ('arch', 4, 0, 3, 646, 0, 'lucario'): -1946,
    ('arch', 4, 0, 3, 646, 0, 'marnie'): -869,
    ('arch', 4, 0, 3, 646, 0, 'rocket'): -1421,
    ('arch', 4, 0, 3, 647, 0, 'alakazam'): -511,
    ('arch', 4, 0, 3, 647, 0, 'crustle'): 956,
    ('arch', 4, 0, 3, 647, 0, 'lucario'): -511,
    ('arch', 4, 0, 3, 647, 0, 'marnie'): -452,
    ('arch', 4, 0, 3, 647, 0, 'rocket'): 2565,
    ('arch', 4, 0, 3, 647, 0, 'unknown'): -2398,
    ('arch', 4, 0, 3, 648, 0, 'alakazam'): 336,
    ('arch', 4, 0, 3, 648, 0, 'crustle'): -511,
    ('arch', 4, 0, 3, 648, 0, 'marnie'): 392,
    ('arch', 4, 0, 3, 648, 0, 'rocket'): 2197,
    ('arch', 4, 0, 3, 648, 0, 'unknown'): 847,
    ('arch', 4, 0, 3, 860, 0, 'alakazam'): -2565,
    ('arch', 4, 0, 3, 860, 0, 'crustle'): -1946,
    ('arch', 4, 0, 3, 860, 0, 'rocket'): -3045,
    ('arch', 40, 112, 0, 1, 0, 'alakazam'): -2565,
    ('arch', 40, 112, 0, 1, 0, 'crustle'): -4043,
    ('arch', 40, 112, 0, 1, 0, 'cynthia'): -1946,
    ('arch', 40, 112, 0, 1, 0, 'marnie'): -5565,
    ('arch', 40, 112, 0, 1, 0, 'rocket'): -3714,
    ('arch', 40, 112, 0, 1, 0, 'unknown'): -2565,
    ('arch', 40, 112, 0, 2, 0, 'alakazam'): -1299,
    ('arch', 40, 112, 0, 2, 0, 'crustle'): 138,
    ('arch', 40, 112, 0, 2, 0, 'cynthia'): -1946,
    ('arch', 40, 112, 0, 2, 0, 'marnie'): -4458,
    ('arch', 40, 112, 0, 2, 0, 'rocket'): 191,
    ('arch', 40, 112, 0, 2, 0, 'unknown'): 1299,
    ('arch', 40, 112, 0, 3, 0, 'alakazam'): 2398,
    ('arch', 40, 112, 0, 3, 0, 'crustle'): 3296,
    ('arch', 40, 112, 0, 3, 0, 'cynthia'): 1946,
    ('arch', 40, 112, 0, 3, 0, 'marnie'): 5557,
    ('arch', 40, 112, 0, 3, 0, 'rocket'): 2944,
    ('arch', 40, 646, 0, 1, 0, 'marnie'): -3219,
    ('arch', 40, 646, 0, 2, 0, 'marnie'): -3219,
    ('arch', 40, 646, 0, 3, 0, 'marnie'): 3219,
    ('arch', 40, 648, 0, 1, 0, 'alakazam'): -2565,
    ('arch', 40, 648, 0, 1, 0, 'crustle'): -3850,
    ('arch', 40, 648, 0, 1, 0, 'dragapult'): -1946,
    ('arch', 40, 648, 0, 1, 0, 'lucario'): -2197,
    ('arch', 40, 648, 0, 1, 0, 'marnie'): -4812,
    ('arch', 40, 648, 0, 1, 0, 'rocket'): -2833,
    ('arch', 40, 648, 0, 1, 0, 'unknown'): -2565,
    ('arch', 40, 648, 0, 2, 0, 'alakazam'): -2565,
    ('arch', 40, 648, 0, 2, 0, 'crustle'): -3850,
    ('arch', 40, 648, 0, 2, 0, 'dragapult'): 511,
    ('arch', 40, 648, 0, 2, 0, 'lucario'): 0,
    ('arch', 40, 648, 0, 2, 0, 'marnie'): -4812,
    ('arch', 40, 648, 0, 2, 0, 'rocket'): -956,
    ('arch', 40, 648, 0, 2, 0, 'unknown'): -2565,
    ('arch', 40, 648, 0, 3, 0, 'alakazam'): 2565,
    ('arch', 40, 648, 0, 3, 0, 'crustle'): 3850,
    ('arch', 40, 648, 0, 3, 0, 'marnie'): 4812,
    ('arch', 40, 648, 0, 3, 0, 'rocket'): 2565,
    ('arch', 40, 648, 0, 3, 0, 'unknown'): 2565,
    ('arch', 41, 0, 1, 0, 0, 'unknown'): 3892,
    ('arch', 41, 0, 2, 0, 0, 'unknown'): -3892,
    ('arch', 43, 648, 1, 0, 0, 'alakazam'): 3135,
    ('arch', 43, 648, 1, 0, 0, 'crustle'): 2944,
    ('arch', 43, 648, 1, 0, 0, 'lucario'): 2565,
    ('arch', 43, 648, 1, 0, 0, 'marnie'): 4796,
    ('arch', 43, 648, 1, 0, 0, 'rocket'): 3434,
    ('arch', 43, 648, 1, 0, 0, 'unknown'): 3296,
    ('arch', 43, 648, 2, 0, 0, 'alakazam'): -3135,
    ('arch', 43, 648, 2, 0, 0, 'crustle'): -2944,
    ('arch', 43, 648, 2, 0, 0, 'lucario'): -2565,
    ('arch', 43, 648, 2, 0, 0, 'marnie'): -4796,
    ('arch', 43, 648, 2, 0, 0, 'rocket'): -3434,
    ('arch', 43, 648, 2, 0, 0, 'unknown'): -3296,
    ('arch', 5, 1086, 3, 646, 0, 'lucario'): 1946,
    ('arch', 5, 1086, 3, 646, 0, 'rocket'): -201,
    ('arch', 5, 1086, 3, 646, 0, 'unknown'): 370,
    ('arch', 5, 1086, 3, 860, 0, 'lucario'): -847,
    ('arch', 5, 1086, 3, 860, 0, 'rocket'): 1299,
    ('arch', 5, 1086, 3, 860, 0, 'unknown'): -892,
    ('arch', 7, 0, 3, 0, 0, 'alakazam'): -813,
    ('arch', 7, 0, 3, 0, 0, 'crustle'): -529,
    ('arch', 7, 0, 3, 0, 0, 'cynthia'): -869,
    ('arch', 7, 0, 3, 0, 0, 'dragapult'): -236,
    ('arch', 7, 0, 3, 0, 0, 'lucario'): -215,
    ('arch', 7, 0, 3, 0, 0, 'marnie'): -665,
    ('arch', 7, 0, 3, 0, 0, 'rocket'): -851,
    ('arch', 7, 0, 3, 0, 0, 'unknown'): -936,
    ('arch', 7, 1097, 3, 104, 0, 'alakazam'): 0,
    ('arch', 7, 1097, 3, 104, 0, 'marnie'): -1946,
    ('arch', 7, 1097, 3, 104, 0, 'rocket'): 511,
    ('arch', 7, 1097, 3, 112, 0, 'lucario'): -847,
    ('arch', 7, 1097, 3, 112, 0, 'marnie'): -663,
    ('arch', 7, 1097, 3, 112, 0, 'rocket'): 1946,
    ('arch', 7, 1097, 3, 646, 0, 'alakazam'): -1099,
    ('arch', 7, 1097, 3, 646, 0, 'marnie'): -1335,
    ('arch', 7, 1097, 3, 646, 0, 'rocket'): -1609,
    ('arch', 7, 1097, 3, 647, 0, 'marnie'): -2565,
    ('arch', 7, 1097, 3, 647, 0, 'rocket'): 847,
    ('arch', 7, 1097, 3, 648, 0, 'marnie'): -1099,
    ('arch', 7, 1097, 3, 648, 0, 'rocket'): -2833,
    ('arch', 7, 1097, 3, 7, 0, 'alakazam'): -251,
    ('arch', 7, 1097, 3, 7, 0, 'crustle'): 0,
    ('arch', 7, 1097, 3, 7, 0, 'dragapult'): -588,
    ('arch', 7, 1097, 3, 7, 0, 'lucario'): -762,
    ('arch', 7, 1097, 3, 7, 0, 'marnie'): -755,
    ('arch', 7, 1097, 3, 7, 0, 'rocket'): -2104,
    ('arch', 7, 1097, 3, 7, 0, 'unknown'): -887,
    ('arch', 7, 1097, 3, 860, 0, 'alakazam'): -588,
    ('arch', 7, 1097, 3, 860, 0, 'crustle'): -511,
    ('arch', 7, 1097, 3, 860, 0, 'marnie'): -2398,
    ('arch', 7, 1097, 3, 860, 0, 'rocket'): 511,
    ('arch', 7, 1122, 3, 1182, 0, 'marnie'): -847,
    ('arch', 7, 1122, 3, 1219, 0, 'rocket'): 847,
    ('arch', 7, 1122, 3, 1219, 0, 'unknown'): 0,
    ('arch', 7, 1122, 3, 1227, 0, 'marnie'): 1099,
    ('arch', 7, 1122, 3, 1227, 0, 'unknown'): 847,
    ('arch', 7, 1152, 3, 104, 0, 'alakazam'): 0,
    ('arch', 7, 1152, 3, 104, 0, 'crustle'): 251,
    ('arch', 7, 1152, 3, 104, 0, 'dragapult'): -1946,
    ('arch', 7, 1152, 3, 104, 0, 'lucario'): -1299,
    ('arch', 7, 1152, 3, 104, 0, 'marnie'): -1640,
    ('arch', 7, 1152, 3, 104, 0, 'rocket'): -480,
    ('arch', 7, 1152, 3, 104, 0, 'unknown'): -2901,
    ('arch', 7, 1152, 3, 112, 0, 'alakazam'): -1316,
    ('arch', 7, 1152, 3, 112, 0, 'crustle'): -1350,
    ('arch', 7, 1152, 3, 112, 0, 'dragapult'): 847,
    ('arch', 7, 1152, 3, 112, 0, 'lucario'): -788,
    ('arch', 7, 1152, 3, 112, 0, 'marnie'): 0,
    ('arch', 7, 1152, 3, 112, 0, 'rocket'): -1435,
    ('arch', 7, 1152, 3, 112, 0, 'unknown'): -1086,
    ('arch', 7, 1152, 3, 646, 0, 'crustle'): -2197,
    ('arch', 7, 1152, 3, 646, 0, 'lucario'): -588,
    ('arch', 7, 1152, 3, 646, 0, 'marnie'): -762,
    ('arch', 7, 1152, 3, 646, 0, 'rocket'): -1526,
    ('arch', 7, 1152, 3, 646, 0, 'unknown'): -2136,
    ('arch', 7, 1152, 3, 647, 0, 'alakazam'): -1435,
    ('arch', 7, 1152, 3, 647, 0, 'crustle'): -2833,
    ('arch', 7, 1152, 3, 647, 0, 'cynthia'): -2197,
    ('arch', 7, 1152, 3, 647, 0, 'lucario'): -2565,
    ('arch', 7, 1152, 3, 647, 0, 'marnie'): -1157,
    ('arch', 7, 1152, 3, 647, 0, 'rocket'): -3850,
    ('arch', 7, 1152, 3, 647, 0, 'unknown'): -5056,
    ('arch', 7, 1152, 3, 860, 0, 'crustle'): 1946,
    ('arch', 7, 1152, 3, 860, 0, 'lucario'): -2398,
    ('arch', 7, 1152, 3, 860, 0, 'marnie'): -2979,
    ('arch', 7, 1152, 3, 860, 0, 'rocket'): -3135,
    ('arch', 7, 1152, 3, 860, 0, 'unknown'): -2398,
    ('arch', 7, 1219, 3, 1079, 0, 'alakazam'): -1488,
    ('arch', 7, 1219, 3, 1079, 0, 'crustle'): -3434,
    ('arch', 7, 1219, 3, 1079, 0, 'lucario'): -2833,
    ('arch', 7, 1219, 3, 1079, 0, 'marnie'): -2113,
    ('arch', 7, 1219, 3, 1079, 0, 'rocket'): -3664,
    ('arch', 7, 1219, 3, 1079, 0, 'unknown'): -3807,
    ('arch', 7, 1219, 3, 1080, 0, 'alakazam'): 847,
    ('arch', 7, 1219, 3, 1080, 0, 'crustle'): -588,
    ('arch', 7, 1219, 3, 1080, 0, 'marnie'): 1099,
    ('arch', 7, 1219, 3, 1080, 0, 'rocket'): 588,
    ('arch', 7, 1219, 3, 1080, 0, 'unknown'): 251,
    ('arch', 7, 1219, 3, 1086, 0, 'alakazam'): -4043,
    ('arch', 7, 1219, 3, 1086, 0, 'crustle'): -3555,
    ('arch', 7, 1219, 3, 1086, 0, 'cynthia'): -2197,
    ('arch', 7, 1219, 3, 1086, 0, 'dragapult'): -1946,
    ('arch', 7, 1219, 3, 1086, 0, 'lucario'): -3135,
    ('arch', 7, 1219, 3, 1086, 0, 'marnie'): -4949,
    ('arch', 7, 1219, 3, 1086, 0, 'rocket'): -2793,
    ('arch', 7, 1219, 3, 1086, 0, 'unknown'): -2282,
    ('arch', 7, 1219, 3, 1097, 0, 'alakazam'): -1825,
    ('arch', 7, 1219, 3, 1097, 0, 'crustle'): -3296,
    ('arch', 7, 1219, 3, 1097, 0, 'cynthia'): -2398,
    ('arch', 7, 1219, 3, 1097, 0, 'lucario'): -1435,
    ('arch', 7, 1219, 3, 1097, 0, 'marnie'): -2398,
    ('arch', 7, 1219, 3, 1097, 0, 'rocket'): -3664,
    ('arch', 7, 1219, 3, 1097, 0, 'unknown'): -2054,
    ('arch', 7, 1219, 3, 1122, 0, 'alakazam'): -2197,
    ('arch', 7, 1219, 3, 1122, 0, 'crustle'): -2398,
    ('arch', 7, 1219, 3, 1122, 0, 'lucario'): -1946,
    ('arch', 7, 1219, 3, 1122, 0, 'marnie'): -3611,
    ('arch', 7, 1219, 3, 1122, 0, 'rocket'): -2565,
    ('arch', 7, 1219, 3, 1122, 0, 'unknown'): -2398,
    ('arch', 7, 1219, 3, 1137, 0, 'alakazam'): -3045,
    ('arch', 7, 1219, 3, 1137, 0, 'crustle'): -2197,
    ('arch', 7, 1219, 3, 1137, 0, 'lucario'): -2398,
    ('arch', 7, 1219, 3, 1137, 0, 'marnie'): -3761,
    ('arch', 7, 1219, 3, 1137, 0, 'rocket'): -1466,
    ('arch', 7, 1219, 3, 1137, 0, 'unknown'): -2708,
    ('arch', 7, 1219, 3, 1152, 0, 'alakazam'): -2120,
    ('arch', 7, 1219, 3, 1152, 0, 'crustle'): -847,
    ('arch', 7, 1219, 3, 1152, 0, 'cynthia'): -2565,
    ('arch', 7, 1219, 3, 1152, 0, 'lucario'): -2269,
    ('arch', 7, 1219, 3, 1152, 0, 'marnie'): -3045,
    ('arch', 7, 1219, 3, 1152, 0, 'rocket'): -2793,
    ('arch', 7, 1219, 3, 1152, 0, 'unknown'): -4007,
    ('arch', 7, 1219, 3, 1182, 0, 'alakazam'): -3367,
    ('arch', 7, 1219, 3, 1182, 0, 'crustle'): -2197,
    ('arch', 7, 1219, 3, 1182, 0, 'cynthia'): -2197,
    ('arch', 7, 1219, 3, 1182, 0, 'lucario'): -2565,
    ('arch', 7, 1219, 3, 1182, 0, 'marnie'): -2317,
    ('arch', 7, 1219, 3, 1182, 0, 'rocket'): -3296,
    ('arch', 7, 1219, 3, 1182, 0, 'unknown'): -3367,
    ('arch', 7, 1219, 3, 1219, 0, 'alakazam'): -2335,
    ('arch', 7, 1219, 3, 1219, 0, 'crustle'): -3367,
    ('arch', 7, 1219, 3, 1219, 0, 'cynthia'): -2197,
    ('arch', 7, 1219, 3, 1219, 0, 'lucario'): -2197,
    ('arch', 7, 1219, 3, 1219, 0, 'marnie'): -4828,
    ('arch', 7, 1219, 3, 1219, 0, 'rocket'): -3892,
    ('arch', 7, 1219, 3, 1219, 0, 'unknown'): -3850,
    ('arch', 7, 1219, 3, 1227, 0, 'alakazam'): -3611,
    ('arch', 7, 1219, 3, 1227, 0, 'crustle'): -2457,
    ('arch', 7, 1219, 3, 1227, 0, 'cynthia'): -2565,
    ('arch', 7, 1219, 3, 1227, 0, 'lucario'): -3367,
    ('arch', 7, 1219, 3, 1227, 0, 'marnie'): -4844,
    ('arch', 7, 1219, 3, 1227, 0, 'rocket'): -1609,
    ('arch', 7, 1219, 3, 1227, 0, 'unknown'): -2595,
    ('arch', 7, 1219, 3, 1231, 0, 'alakazam'): -2565,
    ('arch', 7, 1219, 3, 1231, 0, 'crustle'): -2398,
    ('arch', 7, 1219, 3, 1231, 0, 'lucario'): -2197,
    ('arch', 7, 1219, 3, 1231, 0, 'marnie'): -3296,
    ('arch', 7, 1219, 3, 1231, 0, 'rocket'): -2833,
    ('arch', 7, 1219, 3, 1231, 0, 'unknown'): -2833,
    ('arch', 7, 1219, 3, 1259, 0, 'alakazam'): -3932,
    ('arch', 7, 1219, 3, 1259, 0, 'crustle'): -3664,
    ('arch', 7, 1219, 3, 1259, 0, 'cynthia'): -1466,
    ('arch', 7, 1219, 3, 1259, 0, 'dragapult'): -1946,
    ('arch', 7, 1219, 3, 1259, 0, 'lucario'): -1609,
    ('arch', 7, 1219, 3, 1259, 0, 'marnie'): -3919,
    ('arch', 7, 1219, 3, 1259, 0, 'rocket'): -2663,
    ('arch', 7, 1219, 3, 1259, 0, 'unknown'): -4007,
    ('arch', 7, 1231, 3, 104, 0, 'alakazam'): 511,
    ('arch', 7, 1231, 3, 104, 0, 'marnie'): -571,
    ('arch', 7, 1231, 3, 104, 0, 'unknown'): -452,
    ('arch', 7, 1231, 3, 112, 0, 'alakazam'): -336,
    ('arch', 7, 1231, 3, 112, 0, 'marnie'): 167,
    ('arch', 7, 1231, 3, 112, 0, 'rocket'): -251,
    ('arch', 7, 1231, 3, 112, 0, 'unknown'): -368,
    ('arch', 7, 1231, 3, 646, 0, 'marnie'): -847,
    ('arch', 7, 1231, 3, 646, 0, 'unknown'): -3045,
    ('arch', 7, 1231, 3, 647, 0, 'marnie'): 588,
    ('arch', 7, 1231, 3, 647, 0, 'unknown'): -1224,
    ('arch', 7, 1231, 3, 648, 0, 'marnie'): 1299,
    ('arch', 7, 1231, 3, 648, 0, 'rocket'): -511,
    ('arch', 7, 1231, 3, 648, 0, 'unknown'): 0,
    ('arch', 7, 1231, 3, 860, 0, 'marnie'): -2120,
    ('arch', 7, 1231, 3, 860, 0, 'unknown'): -1299,
    ('arch', 7, 1259, 3, 646, 0, 'crustle'): -1414,
    ('arch', 7, 1259, 3, 646, 0, 'lucario'): -1099,
    ('arch', 7, 1259, 3, 646, 0, 'marnie'): -376,
    ('arch', 7, 1259, 3, 646, 0, 'rocket'): -1237,
    ('arch', 7, 1259, 3, 646, 0, 'unknown'): -1231,
    ('arch', 7, 1259, 3, 647, 0, 'alakazam'): -251,
    ('arch', 7, 1259, 3, 647, 0, 'crustle'): 0,
    ('arch', 7, 1259, 3, 647, 0, 'cynthia'): -1846,
    ('arch', 7, 1259, 3, 647, 0, 'dragapult'): 511,
    ('arch', 7, 1259, 3, 647, 0, 'lucario'): -887,
    ('arch', 7, 1259, 3, 647, 0, 'marnie'): -1014,
    ('arch', 7, 1259, 3, 647, 0, 'rocket'): -1494,
    ('arch', 7, 1259, 3, 647, 0, 'unknown'): -1475,
    ('arch', 7, 1259, 3, 648, 0, 'alakazam'): -251,
    ('arch', 7, 1259, 3, 648, 0, 'crustle'): -956,
    ('arch', 7, 1259, 3, 648, 0, 'cynthia'): -452,
    ('arch', 7, 1259, 3, 648, 0, 'lucario'): -619,
    ('arch', 7, 1259, 3, 648, 0, 'marnie'): -210,
    ('arch', 7, 1259, 3, 648, 0, 'rocket'): -619,
    ('arch', 7, 1259, 3, 648, 0, 'unknown'): -1637,
    ('arch', 8, 1197, 3, 1182, 0, 'unknown'): 847,
    ('base', 0, 0, 10, 112, 0): 146,
    ('base', 0, 0, 10, 1257, 0): 588,
    ('base', 0, 0, 10, 1259, 0): -1053,
    ('base', 0, 0, 12, 0, 0): -3206,
    ('base', 0, 0, 13, 934, 0): -2223,
    ('base', 0, 0, 13, 935, 0): -5153,
    ('base', 0, 0, 13, 936, 0): -2979,
    ('base', 0, 0, 13, 937, 0): -1605,
    ('base', 0, 0, 14, 0, 0): -3087,
    ('base', 0, 0, 7, 1079, 0): -585,
    ('base', 0, 0, 7, 1080, 0): -847,
    ('base', 0, 0, 7, 1086, 0): -1393,
    ('base', 0, 0, 7, 1097, 0): -758,
    ('base', 0, 0, 7, 112, 0): 541,
    ('base', 0, 0, 7, 1122, 0): -788,
    ('base', 0, 0, 7, 1137, 0): -2037,
    ('base', 0, 0, 7, 1152, 0): -587,
    ('base', 0, 0, 7, 1182, 0): -2522,
    ('base', 0, 0, 7, 1219, 0): -2038,
    ('base', 0, 0, 7, 1227, 0): -1511,
    ('base', 0, 0, 7, 1231, 0): -1707,
    ('base', 0, 0, 7, 1259, 0): 73,
    ('base', 0, 0, 7, 646, 0): -275,
    ('base', 0, 0, 7, 860, 0): -2762,
    ('base', 0, 0, 8, 7, 104): -3892,
    ('base', 0, 0, 8, 7, 112): -2737,
    ('base', 0, 0, 8, 7, 646): -4716,
    ('base', 0, 0, 8, 7, 647): -5522,
    ('base', 0, 0, 8, 7, 648): -3950,
    ('base', 0, 0, 8, 7, 860): -5203,
    ('base', 0, 0, 9, 104, 860): 426,
    ('base', 0, 0, 9, 647, 646): -991,
    ('base', 0, 0, 9, 648, 647): -1703,
    ('base', 1, 0, 3, 112, 0): -108,
    ('base', 1, 0, 3, 646, 0): 1242,
    ('base', 1, 0, 3, 860, 0): 1846,
    ('base', 13, 112, 3, 104, 0): -2989,
    ('base', 13, 112, 3, 1071, 0): -2398,
    ('base', 13, 112, 3, 112, 0): -524,
    ('base', 13, 112, 3, 119, 0): -3219,
    ('base', 13, 112, 3, 120, 0): 1946,
    ('base', 13, 112, 3, 132, 0): 1946,
    ('base', 13, 112, 3, 140, 0): -3045,
    ('base', 13, 112, 3, 175, 0): -2197,
    ('base', 13, 112, 3, 235, 0): 511,
    ('base', 13, 112, 3, 272, 0): 2197,
    ('base', 13, 112, 3, 305, 0): -2565,
    ('base', 13, 112, 3, 341, 0): 511,
    ('base', 13, 112, 3, 342, 0): -956,
    ('base', 13, 112, 3, 343, 0): 368,
    ('base', 13, 112, 3, 344, 0): -731,
    ('base', 13, 112, 3, 345, 0): 752,
    ('base', 13, 112, 3, 380, 0): -1099,
    ('base', 13, 112, 3, 381, 0): -2833,
    ('base', 13, 112, 3, 400, 0): 636,
    ('base', 13, 112, 3, 401, 0): -111,
    ('base', 13, 112, 3, 414, 0): -1006,
    ('base', 13, 112, 3, 431, 0): -3135,
    ('base', 13, 112, 3, 434, 0): -1582,
    ('base', 13, 112, 3, 646, 0): -1046,
    ('base', 13, 112, 3, 647, 0): -2903,
    ('base', 13, 112, 3, 648, 0): -2278,
    ('base', 13, 112, 3, 673, 0): 336,
    ('base', 13, 112, 3, 675, 0): -2398,
    ('base', 13, 112, 3, 676, 0): -2398,
    ('base', 13, 112, 3, 678, 0): -2565,
    ('base', 13, 112, 3, 689, 0): -2565,
    ('base', 13, 112, 3, 741, 0): 310,
    ('base', 13, 112, 3, 742, 0): -1099,
    ('base', 13, 112, 3, 743, 0): -3664,
    ('base', 13, 112, 3, 756, 0): -1063,
    ('base', 13, 112, 3, 860, 0): -563,
    ('base', 15, 648, 3, 104, 0): -2361,
    ('base', 15, 648, 3, 1071, 0): -1946,
    ('base', 15, 648, 3, 112, 0): -265,
    ('base', 15, 648, 3, 119, 0): -788,
    ('base', 15, 648, 3, 140, 0): -2708,
    ('base', 15, 648, 3, 175, 0): -1846,
    ('base', 15, 648, 3, 184, 0): 1099,
    ('base', 15, 648, 3, 24, 0): -2398,
    ('base', 15, 648, 3, 272, 0): 1946,
    ('base', 15, 648, 3, 305, 0): -2241,
    ('base', 15, 648, 3, 341, 0): -847,
    ('base', 15, 648, 3, 343, 0): 788,
    ('base', 15, 648, 3, 345, 0): 762,
    ('base', 15, 648, 3, 379, 0): -847,
    ('base', 15, 648, 3, 380, 0): -511,
    ('base', 15, 648, 3, 400, 0): -143,
    ('base', 15, 648, 3, 401, 0): -511,
    ('base', 15, 648, 3, 414, 0): -1609,
    ('base', 15, 648, 3, 431, 0): -2565,
    ('base', 15, 648, 3, 434, 0): -788,
    ('base', 15, 648, 3, 646, 0): -1026,
    ('base', 15, 648, 3, 647, 0): -1989,
    ('base', 15, 648, 3, 648, 0): -1665,
    ('base', 15, 648, 3, 673, 0): 1299,
    ('base', 15, 648, 3, 675, 0): -2708,
    ('base', 15, 648, 3, 676, 0): -2565,
    ('base', 15, 648, 3, 677, 0): -1946,
    ('base', 15, 648, 3, 678, 0): -1946,
    ('base', 15, 648, 3, 741, 0): 111,
    ('base', 15, 648, 3, 742, 0): -1170,
    ('base', 15, 648, 3, 743, 0): -2708,
    ('base', 15, 648, 3, 756, 0): 2197,
    ('base', 15, 648, 3, 860, 0): -351,
    ('base', 16, 112, 3, 104, 0): -1299,
    ('base', 16, 112, 3, 112, 0): -403,
    ('base', 16, 112, 3, 646, 0): 1758,
    ('base', 16, 112, 3, 647, 0): -1099,
    ('base', 16, 112, 3, 648, 0): -630,
    ('base', 2, 0, 3, 112, 0): 3219,
    ('base', 2, 0, 3, 646, 0): 2398,
    ('base', 21, 7, 3, 646, 0): -1022,
    ('base', 21, 7, 3, 647, 0): -990,
    ('base', 21, 7, 3, 648, 0): 11,
    ('base', 22, 648, 3, 7, 0): 249,
    ('base', 3, 0, 3, 104, 0): -2944,
    ('base', 3, 0, 3, 112, 0): -4244,
    ('base', 3, 0, 3, 646, 0): -3497,
    ('base', 3, 0, 3, 647, 0): -2979,
    ('base', 3, 0, 3, 648, 0): 1588,
    ('base', 3, 0, 3, 860, 0): -1946,
    ('base', 3, 1182, 3, 104, 0): -1299,
    ('base', 3, 1182, 3, 112, 0): -671,
    ('base', 3, 1182, 3, 342, 0): 511,
    ('base', 3, 1182, 3, 344, 0): -511,
    ('base', 3, 1182, 3, 646, 0): -588,
    ('base', 3, 1182, 3, 647, 0): -3135,
    ('base', 3, 1182, 3, 648, 0): 0,
    ('base', 3, 1182, 3, 860, 0): -2398,
    ('base', 30, 0, 6, 104, 0): 2833,
    ('base', 30, 0, 6, 112, 0): 2565,
    ('base', 30, 0, 6, 646, 0): 847,
    ('base', 30, 0, 6, 647, 0): 511,
    ('base', 30, 0, 6, 648, 0): 112,
    ('base', 34, 0, 15, 0, 0): 4443,
    ('base', 37, 1079, 9, 648, 646): 276,
    ('base', 38, 0, 0, 0, 0): -3219,
    ('base', 38, 0, 0, 1, 0): 310,
    ('base', 38, 0, 0, 2, 0): 1099,
    ('base', 4, 0, 3, 104, 0): -3664,
    ('base', 4, 0, 3, 112, 0): -5425,
    ('base', 4, 0, 3, 646, 0): -995,
    ('base', 4, 0, 3, 647, 0): -76,
    ('base', 4, 0, 3, 648, 0): 588,
    ('base', 4, 0, 3, 860, 0): -3850,
    ('base', 40, 112, 0, 1, 0): -5974,
    ('base', 40, 112, 0, 2, 0): -1515,
    ('base', 40, 112, 0, 3, 0): 5778,
    ('base', 40, 646, 0, 1, 0): -3367,
    ('base', 40, 646, 0, 2, 0): -1609,
    ('base', 40, 646, 0, 3, 0): 3219,
    ('base', 40, 648, 0, 1, 0): -5416,
    ('base', 40, 648, 0, 2, 0): -2644,
    ('base', 40, 648, 0, 3, 0): 5352,
    ('base', 41, 0, 1, 0, 0): 3892,
    ('base', 41, 0, 2, 0, 0): -3892,
    ('base', 43, 648, 1, 0, 0): 5451,
    ('base', 43, 648, 2, 0, 0): -5451,
    ('base', 5, 1086, 3, 646, 0): 401,
    ('base', 5, 1086, 3, 860, 0): -652,
    ('base', 7, 0, 3, 0, 0): -712,
    ('base', 7, 1097, 3, 104, 0): -435,
    ('base', 7, 1097, 3, 112, 0): -368,
    ('base', 7, 1097, 3, 646, 0): -1494,
    ('base', 7, 1097, 3, 647, 0): -547,
    ('base', 7, 1097, 3, 648, 0): -1758,
    ('base', 7, 1097, 3, 7, 0): -902,
    ('base', 7, 1097, 3, 860, 0): -969,
    ('base', 7, 1122, 3, 1182, 0): -1735,
    ('base', 7, 1122, 3, 1219, 0): 100,
    ('base', 7, 1122, 3, 1227, 0): 1099,
    ('base', 7, 1122, 3, 1231, 0): 588,
    ('base', 7, 1152, 3, 104, 0): -1552,
    ('base', 7, 1152, 3, 112, 0): -936,
    ('base', 7, 1152, 3, 646, 0): -1859,
    ('base', 7, 1152, 3, 647, 0): -2979,
    ('base', 7, 1152, 3, 860, 0): -2445,
    ('base', 7, 1219, 3, 1079, 0): -2708,
    ('base', 7, 1219, 3, 1080, 0): 581,
    ('base', 7, 1219, 3, 1086, 0): -3948,
    ('base', 7, 1219, 3, 1097, 0): -2520,
    ('base', 7, 1219, 3, 1122, 0): -4466,
    ('base', 7, 1219, 3, 1137, 0): -3629,
    ('base', 7, 1219, 3, 1152, 0): -2736,
    ('base', 7, 1219, 3, 1182, 0): -3264,
    ('base', 7, 1219, 3, 1219, 0): -4588,
    ('base', 7, 1219, 3, 1227, 0): -3141,
    ('base', 7, 1219, 3, 1231, 0): -4554,
    ('base', 7, 1219, 3, 1259, 0): -3708,
    ('base', 7, 1231, 3, 104, 0): -336,
    ('base', 7, 1231, 3, 112, 0): -114,
    ('base', 7, 1231, 3, 646, 0): -2398,
    ('base', 7, 1231, 3, 647, 0): -511,
    ('base', 7, 1231, 3, 648, 0): 511,
    ('base', 7, 1231, 3, 860, 0): -2152,
    ('base', 7, 1259, 3, 646, 0): -1021,
    ('base', 7, 1259, 3, 647, 0): -1176,
    ('base', 7, 1259, 3, 648, 0): -799,
    ('base', 8, 1197, 3, 1097, 0): -1946,
    ('base', 8, 1197, 3, 1182, 0): 251,
    ('base', 8, 1197, 3, 1227, 0): -1946,
    ('base', 8, 1197, 3, 1231, 0): 511,
    ('base', 8, 1197, 3, 7, 0): -1099,
    ('exact', 0, 0, 10, 112, 0, 'alakazam', 'end'): 547,
    ('exact', 0, 0, 10, 112, 0, 'alakazam', 'late'): -762,
    ('exact', 0, 0, 10, 112, 0, 'alakazam', 'mid'): 788,
    ('exact', 0, 0, 10, 112, 0, 'crustle', 'end'): 0,
    ('exact', 0, 0, 10, 112, 0, 'crustle', 'late'): 521,
    ('exact', 0, 0, 10, 112, 0, 'crustle', 'mid'): 999,
    ('exact', 0, 0, 10, 112, 0, 'cynthia', 'end'): -336,
    ('exact', 0, 0, 10, 112, 0, 'cynthia', 'mid'): -762,
    ('exact', 0, 0, 10, 112, 0, 'dragapult', 'end'): 1099,
    ('exact', 0, 0, 10, 112, 0, 'dragapult', 'mid'): -847,
    ('exact', 0, 0, 10, 112, 0, 'lucario', 'late'): 847,
    ('exact', 0, 0, 10, 112, 0, 'marnie', 'end'): -89,
    ('exact', 0, 0, 10, 112, 0, 'marnie', 'late'): 120,
    ('exact', 0, 0, 10, 112, 0, 'marnie', 'mid'): 156,
    ('exact', 0, 0, 10, 112, 0, 'rocket', 'end'): 67,
    ('exact', 0, 0, 10, 112, 0, 'rocket', 'late'): 111,
    ('exact', 0, 0, 10, 112, 0, 'rocket', 'mid'): 511,
    ('exact', 0, 0, 10, 112, 0, 'unknown', 'late'): 1299,
    ('exact', 0, 0, 10, 112, 0, 'unknown', 'mid'): 368,
    ('exact', 0, 0, 10, 112, 0, 'unknown', 'open'): -588,
    ('exact', 0, 0, 10, 1257, 0, 'rocket', 'mid'): 847,
    ('exact', 0, 0, 10, 1259, 0, 'alakazam', 'end'): 1946,
    ('exact', 0, 0, 10, 1259, 0, 'alakazam', 'late'): -619,
    ('exact', 0, 0, 10, 1259, 0, 'alakazam', 'mid'): -1196,
    ('exact', 0, 0, 10, 1259, 0, 'crustle', 'end'): -1768,
    ('exact', 0, 0, 10, 1259, 0, 'crustle', 'late'): -1099,
    ('exact', 0, 0, 10, 1259, 0, 'crustle', 'mid'): -1036,
    ('exact', 0, 0, 10, 1259, 0, 'cynthia', 'end'): -847,
    ('exact', 0, 0, 10, 1259, 0, 'cynthia', 'mid'): -1224,
    ('exact', 0, 0, 10, 1259, 0, 'dragapult', 'end'): -847,
    ('exact', 0, 0, 10, 1259, 0, 'dragapult', 'late'): -847,
    ('exact', 0, 0, 10, 1259, 0, 'dragapult', 'mid'): -511,
    ('exact', 0, 0, 10, 1259, 0, 'lucario', 'late'): -999,
    ('exact', 0, 0, 10, 1259, 0, 'lucario', 'mid'): -938,
    ('exact', 0, 0, 10, 1259, 0, 'marnie', 'end'): -1162,
    ('exact', 0, 0, 10, 1259, 0, 'marnie', 'late'): -1526,
    ('exact', 0, 0, 10, 1259, 0, 'marnie', 'mid'): -1152,
    ('exact', 0, 0, 10, 1259, 0, 'marnie', 'open'): -1946,
    ('exact', 0, 0, 10, 1259, 0, 'rocket', 'end'): -788,
    ('exact', 0, 0, 10, 1259, 0, 'rocket', 'late'): -1273,
    ('exact', 0, 0, 10, 1259, 0, 'rocket', 'mid'): 268,
    ('exact', 0, 0, 10, 1259, 0, 'rocket', 'open'): 201,
    ('exact', 0, 0, 10, 1259, 0, 'unknown', 'late'): -379,
    ('exact', 0, 0, 10, 1259, 0, 'unknown', 'mid'): -423,
    ('exact', 0, 0, 10, 1259, 0, 'unknown', 'open'): -505,
    ('exact', 0, 0, 12, 0, 0, 'alakazam', 'end'): -3192,
    ('exact', 0, 0, 12, 0, 0, 'alakazam', 'late'): -3807,
    ('exact', 0, 0, 12, 0, 0, 'alakazam', 'mid'): -3153,
    ('exact', 0, 0, 12, 0, 0, 'crustle', 'end'): -3892,
    ('exact', 0, 0, 12, 0, 0, 'crustle', 'late'): -2663,
    ('exact', 0, 0, 12, 0, 0, 'crustle', 'mid'): -3932,
    ('exact', 0, 0, 12, 0, 0, 'cynthia', 'end'): -2708,
    ('exact', 0, 0, 12, 0, 0, 'cynthia', 'mid'): -3497,
    ('exact', 0, 0, 12, 0, 0, 'dragapult', 'end'): -3296,
    ('exact', 0, 0, 12, 0, 0, 'dragapult', 'mid'): -2398,
    ('exact', 0, 0, 12, 0, 0, 'lucario', 'end'): -2565,
    ('exact', 0, 0, 12, 0, 0, 'lucario', 'late'): -3932,
    ('exact', 0, 0, 12, 0, 0, 'lucario', 'mid'): -3135,
    ('exact', 0, 0, 12, 0, 0, 'marnie', 'end'): -3204,
    ('exact', 0, 0, 12, 0, 0, 'marnie', 'late'): -2985,
    ('exact', 0, 0, 12, 0, 0, 'marnie', 'mid'): -2823,
    ('exact', 0, 0, 12, 0, 0, 'rocket', 'end'): -2468,
    ('exact', 0, 0, 12, 0, 0, 'rocket', 'late'): -4443,
    ('exact', 0, 0, 12, 0, 0, 'rocket', 'mid'): -5242,
    ('exact', 0, 0, 12, 0, 0, 'rocket', 'open'): -4317,
    ('exact', 0, 0, 12, 0, 0, 'unknown', 'end'): -2398,
    ('exact', 0, 0, 12, 0, 0, 'unknown', 'late'): -2241,
    ('exact', 0, 0, 12, 0, 0, 'unknown', 'mid'): -1880,
    ('exact', 0, 0, 12, 0, 0, 'unknown', 'open'): -2314,
    ('exact', 0, 0, 13, 934, 0, 'alakazam', 'mid'): -3219,
    ('exact', 0, 0, 13, 934, 0, 'cynthia', 'end'): -1466,
    ('exact', 0, 0, 13, 934, 0, 'marnie', 'mid'): -2663,
    ('exact', 0, 0, 13, 934, 0, 'rocket', 'mid'): -3850,
    ('exact', 0, 0, 13, 934, 0, 'rocket', 'open'): -1526,
    ('exact', 0, 0, 13, 934, 0, 'unknown', 'open'): 251,
    ('exact', 0, 0, 13, 935, 0, 'alakazam', 'mid'): -3219,
    ('exact', 0, 0, 13, 935, 0, 'cynthia', 'end'): -2708,
    ('exact', 0, 0, 13, 935, 0, 'marnie', 'mid'): -3807,
    ('exact', 0, 0, 13, 935, 0, 'rocket', 'mid'): -3850,
    ('exact', 0, 0, 13, 935, 0, 'rocket', 'open'): -3296,
    ('exact', 0, 0, 13, 935, 0, 'unknown', 'open'): -2708,
    ('exact', 0, 0, 13, 936, 0, 'alakazam', 'mid'): -1946,
    ('exact', 0, 0, 13, 936, 0, 'cynthia', 'mid'): -2944,
    ('exact', 0, 0, 13, 936, 0, 'marnie', 'late'): -2120,
    ('exact', 0, 0, 13, 936, 0, 'rocket', 'late'): -2197,
    ('exact', 0, 0, 13, 937, 0, 'alakazam', 'end'): -1466,
    ('exact', 0, 0, 13, 937, 0, 'alakazam', 'late'): -1718,
    ('exact', 0, 0, 13, 937, 0, 'alakazam', 'mid'): -1273,
    ('exact', 0, 0, 13, 937, 0, 'crustle', 'end'): -1815,
    ('exact', 0, 0, 13, 937, 0, 'crustle', 'late'): -1350,
    ('exact', 0, 0, 13, 937, 0, 'crustle', 'mid'): -956,
    ('exact', 0, 0, 13, 937, 0, 'cynthia', 'mid'): -788,
    ('exact', 0, 0, 13, 937, 0, 'dragapult', 'end'): -1099,
    ('exact', 0, 0, 13, 937, 0, 'dragapult', 'late'): -1099,
    ('exact', 0, 0, 13, 937, 0, 'dragapult', 'mid'): -1099,
    ('exact', 0, 0, 13, 937, 0, 'lucario', 'end'): -1299,
    ('exact', 0, 0, 13, 937, 0, 'lucario', 'late'): -1861,
    ('exact', 0, 0, 13, 937, 0, 'lucario', 'mid'): -1815,
    ('exact', 0, 0, 13, 937, 0, 'marnie', 'end'): -1594,
    ('exact', 0, 0, 13, 937, 0, 'marnie', 'late'): -1821,
    ('exact', 0, 0, 13, 937, 0, 'marnie', 'mid'): -1616,
    ('exact', 0, 0, 13, 937, 0, 'rocket', 'end'): -1680,
    ('exact', 0, 0, 13, 937, 0, 'rocket', 'late'): -1713,
    ('exact', 0, 0, 13, 937, 0, 'rocket', 'mid'): -1421,
    ('exact', 0, 0, 13, 937, 0, 'rocket', 'open'): -847,
    ('exact', 0, 0, 13, 937, 0, 'unknown', 'end'): -1099,
    ('exact', 0, 0, 13, 937, 0, 'unknown', 'late'): -1695,
    ('exact', 0, 0, 13, 937, 0, 'unknown', 'mid'): -1318,
    ('exact', 0, 0, 13, 937, 0, 'unknown', 'open'): 1099,
    ('exact', 0, 0, 14, 0, 0, 'alakazam', 'end'): -4369,
    ('exact', 0, 0, 14, 0, 0, 'alakazam', 'late'): -3807,
    ('exact', 0, 0, 14, 0, 0, 'alakazam', 'mid'): -5130,
    ('exact', 0, 0, 14, 0, 0, 'crustle', 'end'): -4007,
    ('exact', 0, 0, 14, 0, 0, 'crustle', 'late'): -3377,
    ('exact', 0, 0, 14, 0, 0, 'crustle', 'mid'): -2001,
    ('exact', 0, 0, 14, 0, 0, 'cynthia', 'end'): -2708,
    ('exact', 0, 0, 14, 0, 0, 'cynthia', 'mid'): -3497,
    ('exact', 0, 0, 14, 0, 0, 'cynthia', 'open'): -2565,
    ('exact', 0, 0, 14, 0, 0, 'dragapult', 'end'): -3296,
    ('exact', 0, 0, 14, 0, 0, 'dragapult', 'late'): -2833,
    ('exact', 0, 0, 14, 0, 0, 'dragapult', 'mid'): -2398,
    ('exact', 0, 0, 14, 0, 0, 'lucario', 'end'): -2565,
    ('exact', 0, 0, 14, 0, 0, 'lucario', 'late'): -3932,
    ('exact', 0, 0, 14, 0, 0, 'lucario', 'mid'): -3245,
    ('exact', 0, 0, 14, 0, 0, 'lucario', 'open'): -1686,
    ('exact', 0, 0, 14, 0, 0, 'marnie', 'end'): -5808,
    ('exact', 0, 0, 14, 0, 0, 'marnie', 'late'): -5336,
    ('exact', 0, 0, 14, 0, 0, 'marnie', 'mid'): -3440,
    ('exact', 0, 0, 14, 0, 0, 'marnie', 'open'): -1946,
    ('exact', 0, 0, 14, 0, 0, 'rocket', 'end'): -2669,
    ('exact', 0, 0, 14, 0, 0, 'rocket', 'late'): -4489,
    ('exact', 0, 0, 14, 0, 0, 'rocket', 'mid'): -4244,
    ('exact', 0, 0, 14, 0, 0, 'rocket', 'open'): -2357,
    ('exact', 0, 0, 14, 0, 0, 'unknown', 'end'): -2565,
    ('exact', 0, 0, 14, 0, 0, 'unknown', 'late'): -2681,
    ('exact', 0, 0, 14, 0, 0, 'unknown', 'mid'): -3325,
    ('exact', 0, 0, 14, 0, 0, 'unknown', 'open'): -1841,
    ('exact', 0, 0, 7, 1079, 0, 'alakazam', 'mid'): -167,
    ('exact', 0, 0, 7, 1079, 0, 'crustle', 'late'): -847,
    ('exact', 0, 0, 7, 1079, 0, 'lucario', 'mid'): -511,
    ('exact', 0, 0, 7, 1079, 0, 'marnie', 'end'): -788,
    ('exact', 0, 0, 7, 1079, 0, 'marnie', 'late'): -788,
    ('exact', 0, 0, 7, 1079, 0, 'marnie', 'mid'): -950,
    ('exact', 0, 0, 7, 1079, 0, 'rocket', 'mid'): 1099,
    ('exact', 0, 0, 7, 1079, 0, 'rocket', 'open'): -336,
    ('exact', 0, 0, 7, 1079, 0, 'unknown', 'late'): -2398,
    ('exact', 0, 0, 7, 1079, 0, 'unknown', 'mid'): 847,
    ('exact', 0, 0, 7, 1079, 0, 'unknown', 'open'): -201,
    ('exact', 0, 0, 7, 1080, 0, 'alakazam', 'mid'): -636,
    ('exact', 0, 0, 7, 1080, 0, 'crustle', 'late'): -847,
    ('exact', 0, 0, 7, 1080, 0, 'crustle', 'mid'): -788,
    ('exact', 0, 0, 7, 1080, 0, 'cynthia', 'mid'): -511,
    ('exact', 0, 0, 7, 1080, 0, 'lucario', 'mid'): -1466,
    ('exact', 0, 0, 7, 1080, 0, 'marnie', 'end'): 847,
    ('exact', 0, 0, 7, 1080, 0, 'marnie', 'late'): -368,
    ('exact', 0, 0, 7, 1080, 0, 'marnie', 'mid'): -990,
    ('exact', 0, 0, 7, 1080, 0, 'rocket', 'late'): -1735,
    ('exact', 0, 0, 7, 1080, 0, 'rocket', 'mid'): -821,
    ('exact', 0, 0, 7, 1080, 0, 'unknown', 'mid'): -1099,
    ('exact', 0, 0, 7, 1086, 0, 'alakazam', 'mid'): -2565,
    ('exact', 0, 0, 7, 1086, 0, 'crustle', 'late'): -1946,
    ('exact', 0, 0, 7, 1086, 0, 'crustle', 'mid'): -1946,
    ('exact', 0, 0, 7, 1086, 0, 'lucario', 'late'): -1099,
    ('exact', 0, 0, 7, 1086, 0, 'lucario', 'mid'): -1099,
    ('exact', 0, 0, 7, 1086, 0, 'marnie', 'end'): -3245,
    ('exact', 0, 0, 7, 1086, 0, 'marnie', 'late'): -2737,
    ('exact', 0, 0, 7, 1086, 0, 'marnie', 'mid'): -802,
    ('exact', 0, 0, 7, 1086, 0, 'rocket', 'end'): -3045,
    ('exact', 0, 0, 7, 1086, 0, 'rocket', 'late'): -2335,
    ('exact', 0, 0, 7, 1086, 0, 'rocket', 'mid'): -1466,
    ('exact', 0, 0, 7, 1086, 0, 'rocket', 'open'): -201,
    ('exact', 0, 0, 7, 1086, 0, 'unknown', 'open'): -277,
    ('exact', 0, 0, 7, 1097, 0, 'alakazam', 'end'): -1825,
    ('exact', 0, 0, 7, 1097, 0, 'alakazam', 'mid'): -1099,
    ('exact', 0, 0, 7, 1097, 0, 'crustle', 'end'): -847,
    ('exact', 0, 0, 7, 1097, 0, 'crustle', 'late'): -2120,
    ('exact', 0, 0, 7, 1097, 0, 'crustle', 'mid'): -747,
    ('exact', 0, 0, 7, 1097, 0, 'cynthia', 'mid'): -1946,
    ('exact', 0, 0, 7, 1097, 0, 'marnie', 'end'): -738,
    ('exact', 0, 0, 7, 1097, 0, 'marnie', 'late'): -177,
    ('exact', 0, 0, 7, 1097, 0, 'marnie', 'mid'): -847,
    ('exact', 0, 0, 7, 1097, 0, 'rocket', 'end'): -1299,
    ('exact', 0, 0, 7, 1097, 0, 'rocket', 'late'): -167,
    ('exact', 0, 0, 7, 1097, 0, 'rocket', 'mid'): -1053,
    ('exact', 0, 0, 7, 1097, 0, 'unknown', 'late'): -336,
    ('exact', 0, 0, 7, 1097, 0, 'unknown', 'mid'): 511,
    ('exact', 0, 0, 7, 112, 0, 'alakazam', 'mid'): 788,
    ('exact', 0, 0, 7, 112, 0, 'crustle', 'end'): -511,
    ('exact', 0, 0, 7, 112, 0, 'crustle', 'late'): 511,
    ('exact', 0, 0, 7, 112, 0, 'crustle', 'mid'): 0,
    ('exact', 0, 0, 7, 112, 0, 'lucario', 'mid'): 847,
    ('exact', 0, 0, 7, 112, 0, 'lucario', 'open'): 511,
    ('exact', 0, 0, 7, 112, 0, 'marnie', 'end'): -938,
    ('exact', 0, 0, 7, 112, 0, 'marnie', 'late'): 129,
    ('exact', 0, 0, 7, 112, 0, 'marnie', 'mid'): 1017,
    ('exact', 0, 0, 7, 112, 0, 'rocket', 'end'): -588,
    ('exact', 0, 0, 7, 112, 0, 'rocket', 'mid'): 368,
    ('exact', 0, 0, 7, 112, 0, 'unknown', 'mid'): 167,
    ('exact', 0, 0, 7, 112, 0, 'unknown', 'open'): 969,
    ('exact', 0, 0, 7, 1122, 0, 'crustle', 'end'): -2944,
    ('exact', 0, 0, 7, 1122, 0, 'marnie', 'late'): -336,
    ('exact', 0, 0, 7, 1122, 0, 'marnie', 'mid'): -368,
    ('exact', 0, 0, 7, 1122, 0, 'rocket', 'mid'): 847,
    ('exact', 0, 0, 7, 1122, 0, 'unknown', 'open'): -1072,
    ('exact', 0, 0, 7, 1137, 0, 'marnie', 'mid'): -1466,
    ('exact', 0, 0, 7, 1137, 0, 'unknown', 'open'): -2398,
    ('exact', 0, 0, 7, 1152, 0, 'alakazam', 'end'): 336,
    ('exact', 0, 0, 7, 1152, 0, 'alakazam', 'mid'): 201,
    ('exact', 0, 0, 7, 1152, 0, 'crustle', 'end'): -336,
    ('exact', 0, 0, 7, 1152, 0, 'crustle', 'late'): 251,
    ('exact', 0, 0, 7, 1152, 0, 'crustle', 'mid'): 452,
    ('exact', 0, 0, 7, 1152, 0, 'cynthia', 'end'): -1099,
    ('exact', 0, 0, 7, 1152, 0, 'cynthia', 'mid'): -1466,
    ('exact', 0, 0, 7, 1152, 0, 'dragapult', 'end'): 0,
    ('exact', 0, 0, 7, 1152, 0, 'lucario', 'mid'): -762,
    ('exact', 0, 0, 7, 1152, 0, 'lucario', 'open'): -511,
    ('exact', 0, 0, 7, 1152, 0, 'marnie', 'end'): -1546,
    ('exact', 0, 0, 7, 1152, 0, 'marnie', 'late'): -1213,
    ('exact', 0, 0, 7, 1152, 0, 'marnie', 'mid'): -473,
    ('exact', 0, 0, 7, 1152, 0, 'rocket', 'late'): 511,
    ('exact', 0, 0, 7, 1152, 0, 'rocket', 'mid'): -435,
    ('exact', 0, 0, 7, 1152, 0, 'rocket', 'open'): -726,
    ('exact', 0, 0, 7, 1152, 0, 'unknown', 'mid'): -1335,
    ('exact', 0, 0, 7, 1152, 0, 'unknown', 'open'): -447,
    ('exact', 0, 0, 7, 1182, 0, 'alakazam', 'end'): -3219,
    ('exact', 0, 0, 7, 1182, 0, 'alakazam', 'late'): -2708,
    ('exact', 0, 0, 7, 1182, 0, 'alakazam', 'mid'): -2120,
    ('exact', 0, 0, 7, 1182, 0, 'crustle', 'late'): -2152,
    ('exact', 0, 0, 7, 1182, 0, 'crustle', 'mid'): -3497,
    ('exact', 0, 0, 7, 1182, 0, 'cynthia', 'mid'): -847,
    ('exact', 0, 0, 7, 1182, 0, 'dragapult', 'end'): -1299,
    ('exact', 0, 0, 7, 1182, 0, 'lucario', 'mid'): -2833,
    ('exact', 0, 0, 7, 1182, 0, 'marnie', 'end'): -1815,
    ('exact', 0, 0, 7, 1182, 0, 'marnie', 'late'): -2108,
    ('exact', 0, 0, 7, 1182, 0, 'marnie', 'mid'): -2197,
    ('exact', 0, 0, 7, 1182, 0, 'rocket', 'end'): -3850,
    ('exact', 0, 0, 7, 1182, 0, 'rocket', 'late'): -3296,
    ('exact', 0, 0, 7, 1182, 0, 'rocket', 'mid'): -2944,
    ('exact', 0, 0, 7, 1182, 0, 'rocket', 'open'): -2708,
    ('exact', 0, 0, 7, 1182, 0, 'unknown', 'mid'): -847,
    ('exact', 0, 0, 7, 1182, 0, 'unknown', 'open'): -2345,
    ('exact', 0, 0, 7, 1219, 0, 'alakazam', 'end'): -619,
    ('exact', 0, 0, 7, 1219, 0, 'alakazam', 'late'): -847,
    ('exact', 0, 0, 7, 1219, 0, 'alakazam', 'mid'): -1609,
    ('exact', 0, 0, 7, 1219, 0, 'crustle', 'end'): -1609,
    ('exact', 0, 0, 7, 1219, 0, 'crustle', 'late'): -2222,
    ('exact', 0, 0, 7, 1219, 0, 'crustle', 'mid'): -956,
    ('exact', 0, 0, 7, 1219, 0, 'cynthia', 'mid'): -1609,
    ('exact', 0, 0, 7, 1219, 0, 'cynthia', 'open'): -847,
    ('exact', 0, 0, 7, 1219, 0, 'dragapult', 'end'): -1846,
    ('exact', 0, 0, 7, 1219, 0, 'lucario', 'end'): -511,
    ('exact', 0, 0, 7, 1219, 0, 'lucario', 'late'): -2335,
    ('exact', 0, 0, 7, 1219, 0, 'lucario', 'mid'): -956,
    ('exact', 0, 0, 7, 1219, 0, 'lucario', 'open'): -1846,
    ('exact', 0, 0, 7, 1219, 0, 'marnie', 'end'): -2057,
    ('exact', 0, 0, 7, 1219, 0, 'marnie', 'late'): -2085,
    ('exact', 0, 0, 7, 1219, 0, 'marnie', 'mid'): -2208,
    ('exact', 0, 0, 7, 1219, 0, 'marnie', 'open'): -1946,
    ('exact', 0, 0, 7, 1219, 0, 'rocket', 'end'): -2398,
    ('exact', 0, 0, 7, 1219, 0, 'rocket', 'late'): -2132,
    ('exact', 0, 0, 7, 1219, 0, 'rocket', 'mid'): -1713,
    ('exact', 0, 0, 7, 1219, 0, 'rocket', 'open'): -1435,
    ('exact', 0, 0, 7, 1219, 0, 'unknown', 'end'): -847,
    ('exact', 0, 0, 7, 1219, 0, 'unknown', 'late'): -2457,
    ('exact', 0, 0, 7, 1219, 0, 'unknown', 'mid'): -2269,
    ('exact', 0, 0, 7, 1219, 0, 'unknown', 'open'): -1946,
    ('exact', 0, 0, 7, 1227, 0, 'alakazam', 'end'): -2872,
    ('exact', 0, 0, 7, 1227, 0, 'alakazam', 'late'): -2944,
    ('exact', 0, 0, 7, 1227, 0, 'alakazam', 'mid'): -2097,
    ('exact', 0, 0, 7, 1227, 0, 'crustle', 'end'): -1609,
    ('exact', 0, 0, 7, 1227, 0, 'crustle', 'late'): -1350,
    ('exact', 0, 0, 7, 1227, 0, 'crustle', 'mid'): -1170,
    ('exact', 0, 0, 7, 1227, 0, 'dragapult', 'late'): -1099,
    ('exact', 0, 0, 7, 1227, 0, 'lucario', 'late'): -1435,
    ('exact', 0, 0, 7, 1227, 0, 'lucario', 'mid'): -1099,
    ('exact', 0, 0, 7, 1227, 0, 'lucario', 'open'): -1299,
    ('exact', 0, 0, 7, 1227, 0, 'marnie', 'end'): -1946,
    ('exact', 0, 0, 7, 1227, 0, 'marnie', 'late'): -1747,
    ('exact', 0, 0, 7, 1227, 0, 'marnie', 'mid'): -1136,
    ('exact', 0, 0, 7, 1227, 0, 'rocket', 'end'): -1815,
    ('exact', 0, 0, 7, 1227, 0, 'rocket', 'late'): -1946,
    ('exact', 0, 0, 7, 1227, 0, 'rocket', 'mid'): -1466,
    ('exact', 0, 0, 7, 1227, 0, 'rocket', 'open'): -251,
    ('exact', 0, 0, 7, 1227, 0, 'unknown', 'mid'): -1099,
    ('exact', 0, 0, 7, 1227, 0, 'unknown', 'open'): -968,
    ('exact', 0, 0, 7, 1231, 0, 'alakazam', 'late'): -788,
    ('exact', 0, 0, 7, 1231, 0, 'alakazam', 'mid'): -3296,
    ('exact', 0, 0, 7, 1231, 0, 'crustle', 'late'): -847,
    ('exact', 0, 0, 7, 1231, 0, 'lucario', 'late'): -3045,
    ('exact', 0, 0, 7, 1231, 0, 'marnie', 'end'): -969,
    ('exact', 0, 0, 7, 1231, 0, 'marnie', 'late'): -2565,
    ('exact', 0, 0, 7, 1231, 0, 'marnie', 'mid'): -1810,
    ('exact', 0, 0, 7, 1231, 0, 'rocket', 'end'): -1335,
    ('exact', 0, 0, 7, 1231, 0, 'unknown', 'mid'): -1224,
    ('exact', 0, 0, 7, 1231, 0, 'unknown', 'open'): -336,
    ('exact', 0, 0, 7, 1259, 0, 'alakazam', 'end'): 511,
    ('exact', 0, 0, 7, 1259, 0, 'crustle', 'late'): -1224,
    ('exact', 0, 0, 7, 1259, 0, 'marnie', 'mid'): 511,
    ('exact', 0, 0, 7, 1259, 0, 'rocket', 'mid'): -571,
    ('exact', 0, 0, 7, 1259, 0, 'rocket', 'open'): -452,
    ('exact', 0, 0, 7, 1259, 0, 'unknown', 'late'): -511,
    ('exact', 0, 0, 7, 1259, 0, 'unknown', 'open'): 619,
    ('exact', 0, 0, 7, 646, 0, 'alakazam', 'mid'): 1946,
    ('exact', 0, 0, 7, 646, 0, 'crustle', 'end'): -1946,
    ('exact', 0, 0, 7, 646, 0, 'crustle', 'late'): -762,
    ('exact', 0, 0, 7, 646, 0, 'crustle', 'mid'): -847,
    ('exact', 0, 0, 7, 646, 0, 'lucario', 'open'): 1946,
    ('exact', 0, 0, 7, 646, 0, 'marnie', 'end'): -1686,
    ('exact', 0, 0, 7, 646, 0, 'marnie', 'late'): -1880,
    ('exact', 0, 0, 7, 646, 0, 'marnie', 'mid'): 0,
    ('exact', 0, 0, 7, 646, 0, 'rocket', 'late'): -1099,
    ('exact', 0, 0, 7, 646, 0, 'rocket', 'mid'): -201,
    ('exact', 0, 0, 7, 646, 0, 'rocket', 'open'): 2398,
    ('exact', 0, 0, 7, 646, 0, 'unknown', 'mid'): 511,
    ('exact', 0, 0, 7, 646, 0, 'unknown', 'open'): 815,
    ('exact', 0, 0, 7, 860, 0, 'alakazam', 'mid'): -511,
    ('exact', 0, 0, 7, 860, 0, 'lucario', 'late'): -2398,
    ('exact', 0, 0, 7, 860, 0, 'marnie', 'end'): -3761,
    ('exact', 0, 0, 7, 860, 0, 'marnie', 'late'): -5153,
    ('exact', 0, 0, 7, 860, 0, 'marnie', 'mid'): -4890,
    ('exact', 0, 0, 7, 860, 0, 'rocket', 'mid'): -1099,
    ('exact', 0, 0, 7, 860, 0, 'unknown', 'mid'): -2565,
    ('exact', 0, 0, 7, 860, 0, 'unknown', 'open'): -1609,
    ('exact', 0, 0, 8, 7, 104, 'alakazam', 'end'): -3434,
    ('exact', 0, 0, 8, 7, 104, 'alakazam', 'late'): -3296,
    ('exact', 0, 0, 8, 7, 104, 'alakazam', 'mid'): -1758,
    ('exact', 0, 0, 8, 7, 104, 'crustle', 'late'): -3821,
    ('exact', 0, 0, 8, 7, 104, 'crustle', 'mid'): -4796,
    ('exact', 0, 0, 8, 7, 104, 'dragapult', 'end'): -3555,
    ('exact', 0, 0, 8, 7, 104, 'dragapult', 'mid'): -3219,
    ('exact', 0, 0, 8, 7, 104, 'lucario', 'end'): -2197,
    ('exact', 0, 0, 8, 7, 104, 'lucario', 'late'): -3367,
    ('exact', 0, 0, 8, 7, 104, 'lucario', 'mid'): -2197,
    ('exact', 0, 0, 8, 7, 104, 'marnie', 'late'): -2833,
    ('exact', 0, 0, 8, 7, 104, 'marnie', 'mid'): -3271,
    ('exact', 0, 0, 8, 7, 104, 'rocket', 'end'): -2335,
    ('exact', 0, 0, 8, 7, 104, 'rocket', 'late'): -3497,
    ('exact', 0, 0, 8, 7, 104, 'rocket', 'mid'): -4812,
    ('exact', 0, 0, 8, 7, 104, 'unknown', 'late'): -3296,
    ('exact', 0, 0, 8, 7, 104, 'unknown', 'mid'): -1435,
    ('exact', 0, 0, 8, 7, 104, 'unknown', 'open'): -4078,
    ('exact', 0, 0, 8, 7, 112, 'alakazam', 'end'): -2061,
    ('exact', 0, 0, 8, 7, 112, 'alakazam', 'late'): -2398,
    ('exact', 0, 0, 8, 7, 112, 'alakazam', 'mid'): -2314,
    ('exact', 0, 0, 8, 7, 112, 'crustle', 'end'): -788,
    ('exact', 0, 0, 8, 7, 112, 'crustle', 'late'): -3476,
    ('exact', 0, 0, 8, 7, 112, 'crustle', 'mid'): -2854,
    ('exact', 0, 0, 8, 7, 112, 'cynthia', 'mid'): -511,
    ('exact', 0, 0, 8, 7, 112, 'dragapult', 'end'): -3045,
    ('exact', 0, 0, 8, 7, 112, 'dragapult', 'mid'): -4111,
    ('exact', 0, 0, 8, 7, 112, 'lucario', 'end'): -2833,
    ('exact', 0, 0, 8, 7, 112, 'lucario', 'late'): -3106,
    ('exact', 0, 0, 8, 7, 112, 'lucario', 'mid'): -452,
    ('exact', 0, 0, 8, 7, 112, 'lucario', 'open'): -1526,
    ('exact', 0, 0, 8, 7, 112, 'marnie', 'end'): -3634,
    ('exact', 0, 0, 8, 7, 112, 'marnie', 'late'): -3101,
    ('exact', 0, 0, 8, 7, 112, 'marnie', 'mid'): -2694,
    ('exact', 0, 0, 8, 7, 112, 'rocket', 'end'): -2152,
    ('exact', 0, 0, 8, 7, 112, 'rocket', 'late'): -1887,
    ('exact', 0, 0, 8, 7, 112, 'rocket', 'mid'): -2445,
    ('exact', 0, 0, 8, 7, 112, 'rocket', 'open'): -2501,
    ('exact', 0, 0, 8, 7, 112, 'unknown', 'end'): -2197,
    ('exact', 0, 0, 8, 7, 112, 'unknown', 'late'): -2856,
    ('exact', 0, 0, 8, 7, 112, 'unknown', 'mid'): -3204,
    ('exact', 0, 0, 8, 7, 112, 'unknown', 'open'): -2202,
    ('exact', 0, 0, 8, 7, 646, 'alakazam', 'late'): -2398,
    ('exact', 0, 0, 8, 7, 646, 'alakazam', 'mid'): -3497,
    ('exact', 0, 0, 8, 7, 646, 'crustle', 'late'): -3367,
    ('exact', 0, 0, 8, 7, 646, 'crustle', 'mid'): -3367,
    ('exact', 0, 0, 8, 7, 646, 'dragapult', 'end'): -3434,
    ('exact', 0, 0, 8, 7, 646, 'dragapult', 'mid'): -4007,
    ('exact', 0, 0, 8, 7, 646, 'lucario', 'mid'): -1946,
    ('exact', 0, 0, 8, 7, 646, 'lucario', 'open'): -2708,
    ('exact', 0, 0, 8, 7, 646, 'marnie', 'end'): -2565,
    ('exact', 0, 0, 8, 7, 646, 'marnie', 'late'): -4205,
    ('exact', 0, 0, 8, 7, 646, 'marnie', 'mid'): -6385,
    ('exact', 0, 0, 8, 7, 646, 'rocket', 'late'): -3664,
    ('exact', 0, 0, 8, 7, 646, 'rocket', 'mid'): -4272,
    ('exact', 0, 0, 8, 7, 646, 'rocket', 'open'): -2933,
    ('exact', 0, 0, 8, 7, 646, 'unknown', 'late'): -3219,
    ('exact', 0, 0, 8, 7, 646, 'unknown', 'mid'): -5293,
    ('exact', 0, 0, 8, 7, 646, 'unknown', 'open'): -4308,
    ('exact', 0, 0, 8, 7, 647, 'alakazam', 'end'): -1946,
    ('exact', 0, 0, 8, 7, 647, 'alakazam', 'late'): -3892,
    ('exact', 0, 0, 8, 7, 647, 'alakazam', 'mid'): -4043,
    ('exact', 0, 0, 8, 7, 647, 'crustle', 'late'): -4111,
    ('exact', 0, 0, 8, 7, 647, 'crustle', 'mid'): -5242,
    ('exact', 0, 0, 8, 7, 647, 'dragapult', 'end'): -3555,
    ('exact', 0, 0, 8, 7, 647, 'dragapult', 'mid'): -1946,
    ('exact', 0, 0, 8, 7, 647, 'lucario', 'late'): -3219,
    ('exact', 0, 0, 8, 7, 647, 'lucario', 'mid'): -2708,
    ('exact', 0, 0, 8, 7, 647, 'marnie', 'end'): -3761,
    ('exact', 0, 0, 8, 7, 647, 'marnie', 'late'): -5476,
    ('exact', 0, 0, 8, 7, 647, 'marnie', 'mid'): -6028,
    ('exact', 0, 0, 8, 7, 647, 'rocket', 'end'): -3219,
    ('exact', 0, 0, 8, 7, 647, 'rocket', 'late'): -2398,
    ('exact', 0, 0, 8, 7, 647, 'rocket', 'mid'): -4654,
    ('exact', 0, 0, 8, 7, 647, 'rocket', 'open'): -1609,
    ('exact', 0, 0, 8, 7, 647, 'unknown', 'late'): -5056,
    ('exact', 0, 0, 8, 7, 647, 'unknown', 'mid'): -4860,
    ('exact', 0, 0, 8, 7, 647, 'unknown', 'open'): -3995,
    ('exact', 0, 0, 8, 7, 648, 'alakazam', 'end'): -4111,
    ('exact', 0, 0, 8, 7, 648, 'alakazam', 'late'): -4394,
    ('exact', 0, 0, 8, 7, 648, 'alakazam', 'mid'): -4533,
    ('exact', 0, 0, 8, 7, 648, 'crustle', 'late'): -2752,
    ('exact', 0, 0, 8, 7, 648, 'crustle', 'mid'): -3135,
    ('exact', 0, 0, 8, 7, 648, 'dragapult', 'end'): -3761,
    ('exact', 0, 0, 8, 7, 648, 'dragapult', 'late'): -1946,
    ('exact', 0, 0, 8, 7, 648, 'dragapult', 'mid'): -3434,
    ('exact', 0, 0, 8, 7, 648, 'lucario', 'end'): -2037,
    ('exact', 0, 0, 8, 7, 648, 'lucario', 'late'): -2653,
    ('exact', 0, 0, 8, 7, 648, 'lucario', 'mid'): -2398,
    ('exact', 0, 0, 8, 7, 648, 'marnie', 'end'): -2925,
    ('exact', 0, 0, 8, 7, 648, 'marnie', 'late'): -3899,
    ('exact', 0, 0, 8, 7, 648, 'marnie', 'mid'): -5407,
    ('exact', 0, 0, 8, 7, 648, 'rocket', 'end'): -2197,
    ('exact', 0, 0, 8, 7, 648, 'rocket', 'late'): -3296,
    ('exact', 0, 0, 8, 7, 648, 'rocket', 'mid'): -4394,
    ('exact', 0, 0, 8, 7, 648, 'rocket', 'open'): -1946,
    ('exact', 0, 0, 8, 7, 648, 'unknown', 'late'): -2986,
    ('exact', 0, 0, 8, 7, 648, 'unknown', 'mid'): -4745,
    ('exact', 0, 0, 8, 7, 648, 'unknown', 'open'): -2565,
    ('exact', 0, 0, 8, 7, 860, 'alakazam', 'late'): -1946,
    ('exact', 0, 0, 8, 7, 860, 'alakazam', 'mid'): -3219,
    ('exact', 0, 0, 8, 7, 860, 'crustle', 'late'): -4263,
    ('exact', 0, 0, 8, 7, 860, 'crustle', 'mid'): -4920,
    ('exact', 0, 0, 8, 7, 860, 'dragapult', 'mid'): -1946,
    ('exact', 0, 0, 8, 7, 860, 'lucario', 'open'): -2398,
    ('exact', 0, 0, 8, 7, 860, 'marnie', 'mid'): -3761,
    ('exact', 0, 0, 8, 7, 860, 'rocket', 'late'): -2833,
    ('exact', 0, 0, 8, 7, 860, 'rocket', 'mid'): -4554,
    ('exact', 0, 0, 8, 7, 860, 'rocket', 'open'): -3892,
    ('exact', 0, 0, 8, 7, 860, 'unknown', 'late'): -3807,
    ('exact', 0, 0, 8, 7, 860, 'unknown', 'mid'): -4905,
    ('exact', 0, 0, 8, 7, 860, 'unknown', 'open'): -3993,
    ('exact', 0, 0, 9, 104, 860, 'alakazam', 'mid'): 788,
    ('exact', 0, 0, 9, 104, 860, 'crustle', 'late'): 1946,
    ('exact', 0, 0, 9, 104, 860, 'crustle', 'mid'): 1299,
    ('exact', 0, 0, 9, 104, 860, 'marnie', 'mid'): 511,
    ('exact', 0, 0, 9, 104, 860, 'rocket', 'late'): -1299,
    ('exact', 0, 0, 9, 104, 860, 'rocket', 'mid'): 251,
    ('exact', 0, 0, 9, 104, 860, 'rocket', 'open'): 511,
    ('exact', 0, 0, 9, 104, 860, 'unknown', 'open'): -435,
    ('exact', 0, 0, 9, 647, 646, 'alakazam', 'mid'): -726,
    ('exact', 0, 0, 9, 647, 646, 'crustle', 'late'): -452,
    ('exact', 0, 0, 9, 647, 646, 'crustle', 'mid'): -1237,
    ('exact', 0, 0, 9, 647, 646, 'lucario', 'late'): 511,
    ('exact', 0, 0, 9, 647, 646, 'lucario', 'mid'): -368,
    ('exact', 0, 0, 9, 647, 646, 'marnie', 'late'): -2468,
    ('exact', 0, 0, 9, 647, 646, 'marnie', 'mid'): -1305,
    ('exact', 0, 0, 9, 647, 646, 'rocket', 'end'): -788,
    ('exact', 0, 0, 9, 647, 646, 'rocket', 'late'): 0,
    ('exact', 0, 0, 9, 647, 646, 'rocket', 'mid'): -647,
    ('exact', 0, 0, 9, 647, 646, 'rocket', 'open'): -788,
    ('exact', 0, 0, 9, 647, 646, 'unknown', 'late'): -588,
    ('exact', 0, 0, 9, 647, 646, 'unknown', 'mid'): -236,
    ('exact', 0, 0, 9, 647, 646, 'unknown', 'open'): -563,
    ('exact', 0, 0, 9, 648, 647, 'alakazam', 'end'): -3850,
    ('exact', 0, 0, 9, 648, 647, 'alakazam', 'late'): -1609,
    ('exact', 0, 0, 9, 648, 647, 'alakazam', 'mid'): 336,
    ('exact', 0, 0, 9, 648, 647, 'crustle', 'end'): -511,
    ('exact', 0, 0, 9, 648, 647, 'crustle', 'late'): -1645,
    ('exact', 0, 0, 9, 648, 647, 'crustle', 'mid'): -1609,
    ('exact', 0, 0, 9, 648, 647, 'dragapult', 'end'): -2398,
    ('exact', 0, 0, 9, 648, 647, 'lucario', 'late'): -336,
    ('exact', 0, 0, 9, 648, 647, 'lucario', 'mid'): -762,
    ('exact', 0, 0, 9, 648, 647, 'marnie', 'end'): 0,
    ('exact', 0, 0, 9, 648, 647, 'marnie', 'late'): -1547,
    ('exact', 0, 0, 9, 648, 647, 'marnie', 'mid'): -1932,
    ('exact', 0, 0, 9, 648, 647, 'rocket', 'end'): -2398,
    ('exact', 0, 0, 9, 648, 647, 'rocket', 'mid'): 368,
    ('exact', 0, 0, 9, 648, 647, 'unknown', 'late'): -2752,
    ('exact', 0, 0, 9, 648, 647, 'unknown', 'mid'): -1946,
    ('exact', 1, 0, 3, 112, 0, 'unknown', 'end'): -108,
    ('exact', 1, 0, 3, 646, 0, 'unknown', 'end'): 1242,
    ('exact', 1, 0, 3, 860, 0, 'unknown', 'end'): 1846,
    ('exact', 13, 112, 3, 104, 0, 'marnie', 'end'): -3497,
    ('exact', 13, 112, 3, 104, 0, 'marnie', 'late'): -2197,
    ('exact', 13, 112, 3, 104, 0, 'marnie', 'mid'): -3807,
    ('exact', 13, 112, 3, 112, 0, 'dragapult', 'end'): -847,
    ('exact', 13, 112, 3, 112, 0, 'marnie', 'end'): -741,
    ('exact', 13, 112, 3, 112, 0, 'marnie', 'late'): -580,
    ('exact', 13, 112, 3, 112, 0, 'marnie', 'mid'): -230,
    ('exact', 13, 112, 3, 119, 0, 'unknown', 'mid'): -2708,
    ('exact', 13, 112, 3, 140, 0, 'alakazam', 'end'): -2197,
    ('exact', 13, 112, 3, 140, 0, 'dragapult', 'end'): -2197,
    ('exact', 13, 112, 3, 175, 0, 'unknown', 'late'): -2197,
    ('exact', 13, 112, 3, 305, 0, 'alakazam', 'end'): -1946,
    ('exact', 13, 112, 3, 305, 0, 'alakazam', 'late'): -1946,
    ('exact', 13, 112, 3, 305, 0, 'alakazam', 'mid'): -2565,
    ('exact', 13, 112, 3, 341, 0, 'cynthia', 'mid'): 511,
    ('exact', 13, 112, 3, 342, 0, 'cynthia', 'mid'): -2565,
    ('exact', 13, 112, 3, 343, 0, 'alakazam', 'end'): 2197,
    ('exact', 13, 112, 3, 343, 0, 'alakazam', 'mid'): -847,
    ('exact', 13, 112, 3, 344, 0, 'crustle', 'late'): -511,
    ('exact', 13, 112, 3, 344, 0, 'crustle', 'mid'): -2565,
    ('exact', 13, 112, 3, 345, 0, 'crustle', 'end'): 738,
    ('exact', 13, 112, 3, 345, 0, 'crustle', 'late'): 416,
    ('exact', 13, 112, 3, 345, 0, 'crustle', 'mid'): 2944,
    ('exact', 13, 112, 3, 380, 0, 'cynthia', 'mid'): -511,
    ('exact', 13, 112, 3, 381, 0, 'cynthia', 'end'): -2197,
    ('exact', 13, 112, 3, 381, 0, 'cynthia', 'mid'): -2197,
    ('exact', 13, 112, 3, 400, 0, 'rocket', 'late'): 0,
    ('exact', 13, 112, 3, 400, 0, 'rocket', 'mid'): 1946,
    ('exact', 13, 112, 3, 401, 0, 'rocket', 'end'): 1946,
    ('exact', 13, 112, 3, 401, 0, 'rocket', 'late'): 336,
    ('exact', 13, 112, 3, 401, 0, 'rocket', 'mid'): -1099,
    ('exact', 13, 112, 3, 414, 0, 'rocket', 'end'): -659,
    ('exact', 13, 112, 3, 414, 0, 'rocket', 'late'): -2565,
    ('exact', 13, 112, 3, 431, 0, 'rocket', 'end'): -2615,
    ('exact', 13, 112, 3, 431, 0, 'rocket', 'late'): -2565,
    ('exact', 13, 112, 3, 431, 0, 'rocket', 'mid'): -2833,
    ('exact', 13, 112, 3, 434, 0, 'rocket', 'end'): -1718,
    ('exact', 13, 112, 3, 434, 0, 'rocket', 'late'): -1609,
    ('exact', 13, 112, 3, 434, 0, 'rocket', 'mid'): -788,
    ('exact', 13, 112, 3, 646, 0, 'marnie', 'end'): -547,
    ('exact', 13, 112, 3, 646, 0, 'marnie', 'late'): -1367,
    ('exact', 13, 112, 3, 646, 0, 'marnie', 'mid'): -928,
    ('exact', 13, 112, 3, 647, 0, 'marnie', 'end'): -2708,
    ('exact', 13, 112, 3, 647, 0, 'marnie', 'late'): -2335,
    ('exact', 13, 112, 3, 647, 0, 'marnie', 'mid'): -3517,
    ('exact', 13, 112, 3, 648, 0, 'marnie', 'end'): -1426,
    ('exact', 13, 112, 3, 648, 0, 'marnie', 'late'): -2398,
    ('exact', 13, 112, 3, 648, 0, 'marnie', 'mid'): -4844,
    ('exact', 13, 112, 3, 673, 0, 'lucario', 'late'): -511,
    ('exact', 13, 112, 3, 675, 0, 'lucario', 'late'): -1946,
    ('exact', 13, 112, 3, 676, 0, 'lucario', 'late'): -1946,
    ('exact', 13, 112, 3, 678, 0, 'lucario', 'late'): -2197,
    ('exact', 13, 112, 3, 689, 0, 'marnie', 'late'): -1946,
    ('exact', 13, 112, 3, 689, 0, 'marnie', 'mid'): -1946,
    ('exact', 13, 112, 3, 741, 0, 'alakazam', 'late'): 0,
    ('exact', 13, 112, 3, 741, 0, 'alakazam', 'mid'): 588,
    ('exact', 13, 112, 3, 742, 0, 'alakazam', 'end'): -336,
    ('exact', 13, 112, 3, 742, 0, 'alakazam', 'mid'): -1946,
    ('exact', 13, 112, 3, 743, 0, 'alakazam', 'end'): -2944,
    ('exact', 13, 112, 3, 743, 0, 'alakazam', 'late'): -1946,
    ('exact', 13, 112, 3, 743, 0, 'alakazam', 'mid'): -2708,
    ('exact', 13, 112, 3, 756, 0, 'crustle', 'late'): -847,
    ('exact', 13, 112, 3, 756, 0, 'crustle', 'mid'): -2708,
    ('exact', 13, 112, 3, 756, 0, 'unknown', 'mid'): -1946,
    ('exact', 13, 112, 3, 860, 0, 'marnie', 'end'): 762,
    ('exact', 13, 112, 3, 860, 0, 'marnie', 'late'): -361,
    ('exact', 13, 112, 3, 860, 0, 'marnie', 'mid'): -3296,
    ('exact', 15, 648, 3, 104, 0, 'marnie', 'end'): -2833,
    ('exact', 15, 648, 3, 104, 0, 'marnie', 'late'): -1435,
    ('exact', 15, 648, 3, 104, 0, 'marnie', 'mid'): -2833,
    ('exact', 15, 648, 3, 112, 0, 'marnie', 'end'): -636,
    ('exact', 15, 648, 3, 112, 0, 'marnie', 'late'): -336,
    ('exact', 15, 648, 3, 112, 0, 'marnie', 'mid'): 209,
    ('exact', 15, 648, 3, 119, 0, 'unknown', 'mid'): 0,
    ('exact', 15, 648, 3, 140, 0, 'alakazam', 'mid'): -1946,
    ('exact', 15, 648, 3, 175, 0, 'unknown', 'late'): -2565,
    ('exact', 15, 648, 3, 184, 0, 'unknown', 'late'): 1946,
    ('exact', 15, 648, 3, 24, 0, 'unknown', 'late'): -1946,
    ('exact', 15, 648, 3, 305, 0, 'alakazam', 'end'): -588,
    ('exact', 15, 648, 3, 305, 0, 'alakazam', 'late'): -1946,
    ('exact', 15, 648, 3, 305, 0, 'alakazam', 'mid'): -3367,
    ('exact', 15, 648, 3, 343, 0, 'alakazam', 'mid'): 0,
    ('exact', 15, 648, 3, 345, 0, 'crustle', 'late'): 452,
    ('exact', 15, 648, 3, 379, 0, 'cynthia', 'open'): -511,
    ('exact', 15, 648, 3, 380, 0, 'cynthia', 'mid'): -511,
    ('exact', 15, 648, 3, 400, 0, 'rocket', 'mid'): 251,
    ('exact', 15, 648, 3, 400, 0, 'rocket', 'open'): -847,
    ('exact', 15, 648, 3, 401, 0, 'rocket', 'mid'): -511,
    ('exact', 15, 648, 3, 414, 0, 'rocket', 'end'): -956,
    ('exact', 15, 648, 3, 414, 0, 'rocket', 'late'): -1299,
    ('exact', 15, 648, 3, 414, 0, 'rocket', 'mid'): -2565,
    ('exact', 15, 648, 3, 431, 0, 'rocket', 'end'): -1299,
    ('exact', 15, 648, 3, 431, 0, 'rocket', 'late'): -2197,
    ('exact', 15, 648, 3, 431, 0, 'rocket', 'mid'): -3045,
    ('exact', 15, 648, 3, 434, 0, 'rocket', 'end'): 511,
    ('exact', 15, 648, 3, 434, 0, 'rocket', 'late'): -452,
    ('exact', 15, 648, 3, 434, 0, 'rocket', 'mid'): -1435,
    ('exact', 15, 648, 3, 646, 0, 'marnie', 'end'): 0,
    ('exact', 15, 648, 3, 646, 0, 'marnie', 'late'): -636,
    ('exact', 15, 648, 3, 646, 0, 'marnie', 'mid'): -1350,
    ('exact', 15, 648, 3, 647, 0, 'marnie', 'end'): -1099,
    ('exact', 15, 648, 3, 647, 0, 'marnie', 'late'): -2269,
    ('exact', 15, 648, 3, 647, 0, 'marnie', 'mid'): -2024,
    ('exact', 15, 648, 3, 648, 0, 'marnie', 'end'): -236,
    ('exact', 15, 648, 3, 648, 0, 'marnie', 'late'): -2228,
    ('exact', 15, 648, 3, 648, 0, 'marnie', 'mid'): -3367,
    ('exact', 15, 648, 3, 673, 0, 'lucario', 'mid'): 1946,
    ('exact', 15, 648, 3, 675, 0, 'lucario', 'late'): -1946,
    ('exact', 15, 648, 3, 675, 0, 'lucario', 'mid'): -1946,
    ('exact', 15, 648, 3, 676, 0, 'lucario', 'late'): -1946,
    ('exact', 15, 648, 3, 677, 0, 'lucario', 'mid'): -1946,
    ('exact', 15, 648, 3, 741, 0, 'alakazam', 'late'): -847,
    ('exact', 15, 648, 3, 741, 0, 'alakazam', 'mid'): 619,
    ('exact', 15, 648, 3, 741, 0, 'unknown', 'open'): -511,
    ('exact', 15, 648, 3, 742, 0, 'alakazam', 'end'): -847,
    ('exact', 15, 648, 3, 742, 0, 'alakazam', 'mid'): -1435,
    ('exact', 15, 648, 3, 743, 0, 'alakazam', 'mid'): -2197,
    ('exact', 15, 648, 3, 860, 0, 'marnie', 'late'): -201,
    ('exact', 15, 648, 3, 860, 0, 'marnie', 'mid'): -2833,
    ('exact', 16, 112, 3, 104, 0, 'alakazam', 'end'): -847,
    ('exact', 16, 112, 3, 112, 0, 'alakazam', 'end'): -847,
    ('exact', 16, 112, 3, 112, 0, 'alakazam', 'late'): -511,
    ('exact', 16, 112, 3, 112, 0, 'alakazam', 'mid'): -956,
    ('exact', 16, 112, 3, 112, 0, 'crustle', 'end'): -2625,
    ('exact', 16, 112, 3, 112, 0, 'crustle', 'late'): -1014,
    ('exact', 16, 112, 3, 112, 0, 'crustle', 'mid'): 236,
    ('exact', 16, 112, 3, 112, 0, 'cynthia', 'end'): 0,
    ('exact', 16, 112, 3, 112, 0, 'cynthia', 'mid'): 511,
    ('exact', 16, 112, 3, 112, 0, 'dragapult', 'end'): -588,
    ('exact', 16, 112, 3, 112, 0, 'lucario', 'late'): -1299,
    ('exact', 16, 112, 3, 112, 0, 'marnie', 'end'): -825,
    ('exact', 16, 112, 3, 112, 0, 'marnie', 'late'): 186,
    ('exact', 16, 112, 3, 112, 0, 'marnie', 'mid'): 147,
    ('exact', 16, 112, 3, 112, 0, 'rocket', 'end'): -174,
    ('exact', 16, 112, 3, 112, 0, 'rocket', 'late'): -571,
    ('exact', 16, 112, 3, 112, 0, 'rocket', 'mid'): 167,
    ('exact', 16, 112, 3, 112, 0, 'unknown', 'mid'): 251,
    ('exact', 16, 112, 3, 646, 0, 'marnie', 'late'): 1946,
    ('exact', 16, 112, 3, 646, 0, 'marnie', 'mid'): 1224,
    ('exact', 16, 112, 3, 647, 0, 'marnie', 'mid'): -1099,
    ('exact', 16, 112, 3, 648, 0, 'alakazam', 'end'): -747,
    ('exact', 16, 112, 3, 648, 0, 'alakazam', 'late'): 511,
    ('exact', 16, 112, 3, 648, 0, 'alakazam', 'mid'): 336,
    ('exact', 16, 112, 3, 648, 0, 'crustle', 'end'): 379,
    ('exact', 16, 112, 3, 648, 0, 'crustle', 'late'): 659,
    ('exact', 16, 112, 3, 648, 0, 'dragapult', 'end'): -788,
    ('exact', 16, 112, 3, 648, 0, 'lucario', 'end'): -511,
    ('exact', 16, 112, 3, 648, 0, 'lucario', 'late'): -956,
    ('exact', 16, 112, 3, 648, 0, 'marnie', 'end'): -874,
    ('exact', 16, 112, 3, 648, 0, 'marnie', 'late'): -1180,
    ('exact', 16, 112, 3, 648, 0, 'marnie', 'mid'): -619,
    ('exact', 16, 112, 3, 648, 0, 'rocket', 'end'): -547,
    ('exact', 16, 112, 3, 648, 0, 'rocket', 'late'): 1946,
    ('exact', 16, 112, 3, 648, 0, 'unknown', 'late'): 251,
    ('exact', 2, 0, 3, 112, 0, 'unknown', 'open'): 3219,
    ('exact', 2, 0, 3, 646, 0, 'unknown', 'open'): 2398,
    ('exact', 21, 7, 3, 646, 0, 'alakazam', 'mid'): -1237,
    ('exact', 21, 7, 3, 646, 0, 'cynthia', 'open'): -956,
    ('exact', 21, 7, 3, 646, 0, 'lucario', 'mid'): -452,
    ('exact', 21, 7, 3, 646, 0, 'marnie', 'mid'): -1062,
    ('exact', 21, 7, 3, 646, 0, 'rocket', 'mid'): -969,
    ('exact', 21, 7, 3, 646, 0, 'rocket', 'open'): -511,
    ('exact', 21, 7, 3, 646, 0, 'unknown', 'mid'): -1149,
    ('exact', 21, 7, 3, 646, 0, 'unknown', 'open'): -802,
    ('exact', 21, 7, 3, 647, 0, 'alakazam', 'late'): -1099,
    ('exact', 21, 7, 3, 647, 0, 'alakazam', 'mid'): 167,
    ('exact', 21, 7, 3, 647, 0, 'crustle', 'late'): -511,
    ('exact', 21, 7, 3, 647, 0, 'crustle', 'mid'): -588,
    ('exact', 21, 7, 3, 647, 0, 'lucario', 'mid'): -336,
    ('exact', 21, 7, 3, 647, 0, 'marnie', 'late'): -762,
    ('exact', 21, 7, 3, 647, 0, 'marnie', 'mid'): -1319,
    ('exact', 21, 7, 3, 647, 0, 'rocket', 'mid'): -251,
    ('exact', 21, 7, 3, 647, 0, 'unknown', 'mid'): -1099,
    ('exact', 21, 7, 3, 648, 0, 'alakazam', 'late'): -511,
    ('exact', 21, 7, 3, 648, 0, 'alakazam', 'mid'): -251,
    ('exact', 21, 7, 3, 648, 0, 'crustle', 'end'): 2197,
    ('exact', 21, 7, 3, 648, 0, 'crustle', 'late'): 211,
    ('exact', 21, 7, 3, 648, 0, 'crustle', 'mid'): 588,
    ('exact', 21, 7, 3, 648, 0, 'cynthia', 'open'): 0,
    ('exact', 21, 7, 3, 648, 0, 'lucario', 'mid'): -310,
    ('exact', 21, 7, 3, 648, 0, 'marnie', 'end'): -201,
    ('exact', 21, 7, 3, 648, 0, 'marnie', 'late'): -387,
    ('exact', 21, 7, 3, 648, 0, 'marnie', 'mid'): 206,
    ('exact', 21, 7, 3, 648, 0, 'rocket', 'end'): 0,
    ('exact', 21, 7, 3, 648, 0, 'rocket', 'late'): 511,
    ('exact', 21, 7, 3, 648, 0, 'rocket', 'mid'): 423,
    ('exact', 21, 7, 3, 648, 0, 'rocket', 'open'): 336,
    ('exact', 21, 7, 3, 648, 0, 'unknown', 'late'): -368,
    ('exact', 21, 7, 3, 648, 0, 'unknown', 'mid'): 191,
    ('exact', 21, 7, 3, 648, 0, 'unknown', 'open'): 143,
    ('exact', 22, 648, 3, 7, 0, 'alakazam', 'late'): 1299,
    ('exact', 22, 648, 3, 7, 0, 'alakazam', 'mid'): -137,
    ('exact', 22, 648, 3, 7, 0, 'crustle', 'end'): 2197,
    ('exact', 22, 648, 3, 7, 0, 'crustle', 'late'): 174,
    ('exact', 22, 648, 3, 7, 0, 'crustle', 'mid'): 956,
    ('exact', 22, 648, 3, 7, 0, 'cynthia', 'open'): 251,
    ('exact', 22, 648, 3, 7, 0, 'lucario', 'mid'): 336,
    ('exact', 22, 648, 3, 7, 0, 'marnie', 'end'): 2197,
    ('exact', 22, 648, 3, 7, 0, 'marnie', 'late'): 892,
    ('exact', 22, 648, 3, 7, 0, 'marnie', 'mid'): 111,
    ('exact', 22, 648, 3, 7, 0, 'rocket', 'end'): 511,
    ('exact', 22, 648, 3, 7, 0, 'rocket', 'late'): 847,
    ('exact', 22, 648, 3, 7, 0, 'rocket', 'mid'): -201,
    ('exact', 22, 648, 3, 7, 0, 'rocket', 'open'): -938,
    ('exact', 22, 648, 3, 7, 0, 'unknown', 'late'): 1099,
    ('exact', 22, 648, 3, 7, 0, 'unknown', 'mid'): 566,
    ('exact', 22, 648, 3, 7, 0, 'unknown', 'open'): 0,
    ('exact', 3, 0, 3, 104, 0, 'crustle', 'late'): -2197,
    ('exact', 3, 0, 3, 112, 0, 'alakazam', 'mid'): -1946,
    ('exact', 3, 0, 3, 112, 0, 'crustle', 'late'): -2944,
    ('exact', 3, 0, 3, 112, 0, 'marnie', 'end'): -3367,
    ('exact', 3, 0, 3, 112, 0, 'marnie', 'late'): -4043,
    ('exact', 3, 0, 3, 112, 0, 'marnie', 'mid'): -4043,
    ('exact', 3, 0, 3, 112, 0, 'rocket', 'end'): -2398,
    ('exact', 3, 0, 3, 112, 0, 'unknown', 'late'): -2398,
    ('exact', 3, 0, 3, 112, 0, 'unknown', 'mid'): -2398,
    ('exact', 3, 0, 3, 112, 0, 'unknown', 'open'): -847,
    ('exact', 3, 0, 3, 646, 0, 'marnie', 'mid'): -1946,
    ('exact', 3, 0, 3, 646, 0, 'unknown', 'mid'): -2398,
    ('exact', 3, 0, 3, 646, 0, 'unknown', 'open'): -2398,
    ('exact', 3, 0, 3, 647, 0, 'alakazam', 'mid'): -2197,
    ('exact', 3, 0, 3, 647, 0, 'marnie', 'late'): -2565,
    ('exact', 3, 0, 3, 647, 0, 'marnie', 'mid'): -2833,
    ('exact', 3, 0, 3, 647, 0, 'unknown', 'mid'): -1946,
    ('exact', 3, 0, 3, 648, 0, 'alakazam', 'mid'): 511,
    ('exact', 3, 0, 3, 648, 0, 'crustle', 'late'): 847,
    ('exact', 3, 0, 3, 648, 0, 'dragapult', 'late'): -511,
    ('exact', 3, 0, 3, 648, 0, 'marnie', 'end'): 2398,
    ('exact', 3, 0, 3, 648, 0, 'marnie', 'late'): 938,
    ('exact', 3, 0, 3, 648, 0, 'marnie', 'mid'): 3135,
    ('exact', 3, 0, 3, 648, 0, 'unknown', 'late'): 511,
    ('exact', 3, 0, 3, 648, 0, 'unknown', 'mid'): 2197,
    ('exact', 3, 0, 3, 648, 0, 'unknown', 'open'): 1946,
    ('exact', 3, 1182, 3, 104, 0, 'marnie', 'late'): -847,
    ('exact', 3, 1182, 3, 112, 0, 'marnie', 'end'): -1099,
    ('exact', 3, 1182, 3, 112, 0, 'marnie', 'late'): -747,
    ('exact', 3, 1182, 3, 112, 0, 'marnie', 'mid'): -547,
    ('exact', 3, 1182, 3, 646, 0, 'unknown', 'open'): -511,
    ('exact', 3, 1182, 3, 647, 0, 'marnie', 'late'): -1946,
    ('exact', 3, 1182, 3, 647, 0, 'marnie', 'mid'): -2565,
    ('exact', 3, 1182, 3, 648, 0, 'marnie', 'late'): -511,
    ('exact', 3, 1182, 3, 648, 0, 'marnie', 'mid'): -336,
    ('exact', 3, 1182, 3, 860, 0, 'marnie', 'late'): -1946,
    ('exact', 30, 0, 6, 648, 0, 'alakazam', 'end'): -336,
    ('exact', 30, 0, 6, 648, 0, 'crustle', 'late'): 588,
    ('exact', 30, 0, 6, 648, 0, 'marnie', 'end'): 100,
    ('exact', 30, 0, 6, 648, 0, 'marnie', 'late'): 251,
    ('exact', 30, 0, 6, 648, 0, 'marnie', 'mid'): 174,
    ('exact', 30, 0, 6, 648, 0, 'rocket', 'end'): -336,
    ('exact', 30, 0, 6, 648, 0, 'unknown', 'late'): -368,
    ('exact', 34, 0, 15, 0, 0, 'alakazam', 'end'): 2197,
    ('exact', 34, 0, 15, 0, 0, 'crustle', 'late'): 2197,
    ('exact', 34, 0, 15, 0, 0, 'marnie', 'end'): 2565,
    ('exact', 34, 0, 15, 0, 0, 'marnie', 'late'): 2565,
    ('exact', 34, 0, 15, 0, 0, 'marnie', 'mid'): 2565,
    ('exact', 34, 0, 15, 0, 0, 'rocket', 'end'): 2833,
    ('exact', 37, 1079, 9, 648, 646, 'alakazam', 'mid'): 788,
    ('exact', 37, 1079, 9, 648, 646, 'cynthia', 'open'): -511,
    ('exact', 37, 1079, 9, 648, 646, 'marnie', 'mid'): 423,
    ('exact', 37, 1079, 9, 648, 646, 'rocket', 'mid'): -636,
    ('exact', 37, 1079, 9, 648, 646, 'rocket', 'open'): -336,
    ('exact', 37, 1079, 9, 648, 646, 'unknown', 'mid'): -251,
    ('exact', 37, 1079, 9, 648, 646, 'unknown', 'open'): 1099,
    ('exact', 38, 0, 0, 0, 0, 'unknown', 'open'): -3219,
    ('exact', 38, 0, 0, 1, 0, 'unknown', 'open'): 310,
    ('exact', 38, 0, 0, 2, 0, 'unknown', 'open'): 1099,
    ('exact', 4, 0, 3, 104, 0, 'crustle', 'late'): -1946,
    ('exact', 4, 0, 3, 104, 0, 'rocket', 'late'): -1946,
    ('exact', 4, 0, 3, 104, 0, 'rocket', 'mid'): -1946,
    ('exact', 4, 0, 3, 112, 0, 'alakazam', 'mid'): -2398,
    ('exact', 4, 0, 3, 112, 0, 'crustle', 'late'): -3045,
    ('exact', 4, 0, 3, 112, 0, 'crustle', 'mid'): -1946,
    ('exact', 4, 0, 3, 112, 0, 'lucario', 'mid'): -2197,
    ('exact', 4, 0, 3, 112, 0, 'marnie', 'end'): -2398,
    ('exact', 4, 0, 3, 112, 0, 'marnie', 'late'): -3892,
    ('exact', 4, 0, 3, 112, 0, 'marnie', 'mid'): -4205,
    ('exact', 4, 0, 3, 112, 0, 'marnie', 'open'): -2398,
    ('exact', 4, 0, 3, 112, 0, 'rocket', 'late'): -1946,
    ('exact', 4, 0, 3, 112, 0, 'rocket', 'mid'): -3135,
    ('exact', 4, 0, 3, 646, 0, 'alakazam', 'mid'): -511,
    ('exact', 4, 0, 3, 646, 0, 'lucario', 'mid'): -1946,
    ('exact', 4, 0, 3, 646, 0, 'marnie', 'late'): -588,
    ('exact', 4, 0, 3, 646, 0, 'marnie', 'mid'): -1846,
    ('exact', 4, 0, 3, 646, 0, 'marnie', 'open'): 336,
    ('exact', 4, 0, 3, 646, 0, 'rocket', 'mid'): -1273,
    ('exact', 4, 0, 3, 647, 0, 'crustle', 'late'): 2197,
    ('exact', 4, 0, 3, 647, 0, 'crustle', 'mid'): 0,
    ('exact', 4, 0, 3, 647, 0, 'lucario', 'mid'): -511,
    ('exact', 4, 0, 3, 647, 0, 'marnie', 'late'): 847,
    ('exact', 4, 0, 3, 647, 0, 'marnie', 'mid'): -869,
    ('exact', 4, 0, 3, 647, 0, 'rocket', 'mid'): 2197,
    ('exact', 4, 0, 3, 647, 0, 'unknown', 'late'): -1946,
    ('exact', 4, 0, 3, 648, 0, 'alakazam', 'mid'): 511,
    ('exact', 4, 0, 3, 648, 0, 'marnie', 'end'): -511,
    ('exact', 4, 0, 3, 648, 0, 'marnie', 'late'): -379,
    ('exact', 4, 0, 3, 648, 0, 'marnie', 'mid'): 2037,
    ('exact', 4, 0, 3, 648, 0, 'rocket', 'mid'): 1946,
    ('exact', 4, 0, 3, 648, 0, 'unknown', 'late'): 511,
    ('exact', 4, 0, 3, 860, 0, 'alakazam', 'mid'): -2197,
    ('exact', 4, 0, 3, 860, 0, 'rocket', 'mid'): -2944,
    ('exact', 40, 112, 0, 1, 0, 'alakazam', 'end'): -2197,
    ('exact', 40, 112, 0, 1, 0, 'crustle', 'late'): -3664,
    ('exact', 40, 112, 0, 1, 0, 'crustle', 'mid'): -2708,
    ('exact', 40, 112, 0, 1, 0, 'marnie', 'end'): -3970,
    ('exact', 40, 112, 0, 1, 0, 'marnie', 'late'): -5017,
    ('exact', 40, 112, 0, 1, 0, 'marnie', 'mid'): -4078,
    ('exact', 40, 112, 0, 1, 0, 'rocket', 'end'): -3045,
    ('exact', 40, 112, 0, 1, 0, 'rocket', 'late'): -2197,
    ('exact', 40, 112, 0, 1, 0, 'rocket', 'mid'): -2565,
    ('exact', 40, 112, 0, 1, 0, 'unknown', 'mid'): -2197,
    ('exact', 40, 112, 0, 2, 0, 'alakazam', 'end'): -2197,
    ('exact', 40, 112, 0, 2, 0, 'crustle', 'late'): -302,
    ('exact', 40, 112, 0, 2, 0, 'crustle', 'mid'): 1466,
    ('exact', 40, 112, 0, 2, 0, 'marnie', 'end'): -3970,
    ('exact', 40, 112, 0, 2, 0, 'marnie', 'late'): -3905,
    ('exact', 40, 112, 0, 2, 0, 'marnie', 'mid'): -4078,
    ('exact', 40, 112, 0, 2, 0, 'rocket', 'end'): -762,
    ('exact', 40, 112, 0, 2, 0, 'rocket', 'late'): 847,
    ('exact', 40, 112, 0, 2, 0, 'rocket', 'mid'): 1299,
    ('exact', 40, 112, 0, 2, 0, 'unknown', 'mid'): 2197,
    ('exact', 40, 112, 0, 3, 0, 'alakazam', 'end'): 2197,
    ('exact', 40, 112, 0, 3, 0, 'crustle', 'late'): 3135,
    ('exact', 40, 112, 0, 3, 0, 'marnie', 'end'): 3970,
    ('exact', 40, 112, 0, 3, 0, 'marnie', 'late'): 5004,
    ('exact', 40, 112, 0, 3, 0, 'marnie', 'mid'): 4078,
    ('exact', 40, 112, 0, 3, 0, 'rocket', 'end'): 2708,
    ('exact', 40, 646, 0, 1, 0, 'marnie', 'late'): -1946,
    ('exact', 40, 646, 0, 1, 0, 'marnie', 'mid'): -2833,
    ('exact', 40, 646, 0, 2, 0, 'marnie', 'late'): -1946,
    ('exact', 40, 646, 0, 2, 0, 'marnie', 'mid'): -2833,
    ('exact', 40, 646, 0, 3, 0, 'marnie', 'late'): 1946,
    ('exact', 40, 646, 0, 3, 0, 'marnie', 'mid'): 2833,
    ('exact', 40, 648, 0, 1, 0, 'alakazam', 'end'): -1946,
    ('exact', 40, 648, 0, 1, 0, 'crustle', 'end'): -2944,
    ('exact', 40, 648, 0, 1, 0, 'crustle', 'late'): -3367,
    ('exact', 40, 648, 0, 1, 0, 'marnie', 'end'): -3970,
    ('exact', 40, 648, 0, 1, 0, 'marnie', 'late'): -3970,
    ('exact', 40, 648, 0, 1, 0, 'marnie', 'mid'): -2944,
    ('exact', 40, 648, 0, 1, 0, 'rocket', 'end'): -2398,
    ('exact', 40, 648, 0, 1, 0, 'unknown', 'late'): -2197,
    ('exact', 40, 648, 0, 2, 0, 'alakazam', 'end'): -1946,
    ('exact', 40, 648, 0, 2, 0, 'crustle', 'end'): -2944,
    ('exact', 40, 648, 0, 2, 0, 'crustle', 'late'): -3367,
    ('exact', 40, 648, 0, 2, 0, 'marnie', 'end'): -3970,
    ('exact', 40, 648, 0, 2, 0, 'marnie', 'late'): -3970,
    ('exact', 40, 648, 0, 2, 0, 'marnie', 'mid'): -2944,
    ('exact', 40, 648, 0, 2, 0, 'rocket', 'end'): -2398,
    ('exact', 40, 648, 0, 2, 0, 'unknown', 'late'): -2197,
    ('exact', 40, 648, 0, 3, 0, 'alakazam', 'end'): 1946,
    ('exact', 40, 648, 0, 3, 0, 'crustle', 'end'): 2944,
    ('exact', 40, 648, 0, 3, 0, 'crustle', 'late'): 3367,
    ('exact', 40, 648, 0, 3, 0, 'marnie', 'end'): 3970,
    ('exact', 40, 648, 0, 3, 0, 'marnie', 'late'): 3970,
    ('exact', 40, 648, 0, 3, 0, 'marnie', 'mid'): 2944,
    ('exact', 40, 648, 0, 3, 0, 'rocket', 'end'): 2398,
    ('exact', 40, 648, 0, 3, 0, 'unknown', 'late'): 2197,
    ('exact', 41, 0, 1, 0, 0, 'unknown', 'end'): 3892,
    ('exact', 41, 0, 2, 0, 0, 'unknown', 'end'): -3892,
    ('exact', 43, 648, 1, 0, 0, 'alakazam', 'late'): 1946,
    ('exact', 43, 648, 1, 0, 0, 'alakazam', 'mid'): 2833,
    ('exact', 43, 648, 1, 0, 0, 'crustle', 'late'): 2565,
    ('exact', 43, 648, 1, 0, 0, 'lucario', 'mid'): 2197,
    ('exact', 43, 648, 1, 0, 0, 'marnie', 'end'): 2833,
    ('exact', 43, 648, 1, 0, 0, 'marnie', 'late'): 3611,
    ('exact', 43, 648, 1, 0, 0, 'marnie', 'mid'): 4234,
    ('exact', 43, 648, 1, 0, 0, 'rocket', 'mid'): 3045,
    ('exact', 43, 648, 1, 0, 0, 'unknown', 'late'): 2197,
    ('exact', 43, 648, 1, 0, 0, 'unknown', 'mid'): 2398,
    ('exact', 43, 648, 1, 0, 0, 'unknown', 'open'): 2197,
    ('exact', 43, 648, 2, 0, 0, 'alakazam', 'late'): -1946,
    ('exact', 43, 648, 2, 0, 0, 'alakazam', 'mid'): -2833,
    ('exact', 43, 648, 2, 0, 0, 'crustle', 'late'): -2565,
    ('exact', 43, 648, 2, 0, 0, 'lucario', 'mid'): -2197,
    ('exact', 43, 648, 2, 0, 0, 'marnie', 'end'): -2833,
    ('exact', 43, 648, 2, 0, 0, 'marnie', 'late'): -3611,
    ('exact', 43, 648, 2, 0, 0, 'marnie', 'mid'): -4234,
    ('exact', 43, 648, 2, 0, 0, 'rocket', 'mid'): -3045,
    ('exact', 43, 648, 2, 0, 0, 'unknown', 'late'): -2197,
    ('exact', 43, 648, 2, 0, 0, 'unknown', 'mid'): -2398,
    ('exact', 43, 648, 2, 0, 0, 'unknown', 'open'): -2197,
    ('exact', 5, 1086, 3, 646, 0, 'rocket', 'open'): -588,
    ('exact', 5, 1086, 3, 646, 0, 'unknown', 'open'): 349,
    ('exact', 5, 1086, 3, 860, 0, 'rocket', 'open'): 1299,
    ('exact', 5, 1086, 3, 860, 0, 'unknown', 'open'): -892,
    ('exact', 7, 0, 3, 0, 0, 'alakazam', 'end'): 647,
    ('exact', 7, 0, 3, 0, 0, 'alakazam', 'late'): -659,
    ('exact', 7, 0, 3, 0, 0, 'alakazam', 'mid'): -1335,
    ('exact', 7, 0, 3, 0, 0, 'crustle', 'end'): 336,
    ('exact', 7, 0, 3, 0, 0, 'crustle', 'late'): -623,
    ('exact', 7, 0, 3, 0, 0, 'cynthia', 'end'): 511,
    ('exact', 7, 0, 3, 0, 0, 'cynthia', 'mid'): -999,
    ('exact', 7, 0, 3, 0, 0, 'cynthia', 'open'): -1299,
    ('exact', 7, 0, 3, 0, 0, 'dragapult', 'end'): 588,
    ('exact', 7, 0, 3, 0, 0, 'dragapult', 'late'): -511,
    ('exact', 7, 0, 3, 0, 0, 'dragapult', 'mid'): -788,
    ('exact', 7, 0, 3, 0, 0, 'lucario', 'end'): 1946,
    ('exact', 7, 0, 3, 0, 0, 'lucario', 'late'): 143,
    ('exact', 7, 0, 3, 0, 0, 'lucario', 'mid'): -1335,
    ('exact', 7, 0, 3, 0, 0, 'marnie', 'end'): 1200,
    ('exact', 7, 0, 3, 0, 0, 'marnie', 'late'): -752,
    ('exact', 7, 0, 3, 0, 0, 'marnie', 'mid'): -1302,
    ('exact', 7, 0, 3, 0, 0, 'rocket', 'end'): 821,
    ('exact', 7, 0, 3, 0, 0, 'rocket', 'late'): -835,
    ('exact', 7, 0, 3, 0, 0, 'rocket', 'mid'): -1376,
    ('exact', 7, 0, 3, 0, 0, 'rocket', 'open'): -1299,
    ('exact', 7, 0, 3, 0, 0, 'unknown', 'end'): 2197,
    ('exact', 7, 0, 3, 0, 0, 'unknown', 'late'): 0,
    ('exact', 7, 0, 3, 0, 0, 'unknown', 'mid'): -1125,
    ('exact', 7, 0, 3, 0, 0, 'unknown', 'open'): -1516,
    ('exact', 7, 1097, 3, 112, 0, 'marnie', 'end'): -2944,
    ('exact', 7, 1097, 3, 112, 0, 'marnie', 'late'): 0,
    ('exact', 7, 1097, 3, 112, 0, 'marnie', 'mid'): 511,
    ('exact', 7, 1097, 3, 646, 0, 'alakazam', 'mid'): -847,
    ('exact', 7, 1097, 3, 646, 0, 'marnie', 'end'): -2197,
    ('exact', 7, 1097, 3, 646, 0, 'marnie', 'late'): -2197,
    ('exact', 7, 1097, 3, 646, 0, 'marnie', 'mid'): 511,
    ('exact', 7, 1097, 3, 646, 0, 'rocket', 'late'): -2398,
    ('exact', 7, 1097, 3, 647, 0, 'marnie', 'late'): -1946,
    ('exact', 7, 1097, 3, 648, 0, 'marnie', 'end'): -511,
    ('exact', 7, 1097, 3, 648, 0, 'rocket', 'late'): -2398,
    ('exact', 7, 1097, 3, 7, 0, 'alakazam', 'mid'): -251,
    ('exact', 7, 1097, 3, 7, 0, 'dragapult', 'late'): -588,
    ('exact', 7, 1097, 3, 7, 0, 'lucario', 'end'): -847,
    ('exact', 7, 1097, 3, 7, 0, 'lucario', 'late'): -2197,
    ('exact', 7, 1097, 3, 7, 0, 'marnie', 'end'): -1609,
    ('exact', 7, 1097, 3, 7, 0, 'marnie', 'late'): -552,
    ('exact', 7, 1097, 3, 7, 0, 'marnie', 'mid'): 619,
    ('exact', 7, 1097, 3, 7, 0, 'rocket', 'late'): -3219,
    ('exact', 7, 1097, 3, 7, 0, 'rocket', 'mid'): -1609,
    ('exact', 7, 1097, 3, 7, 0, 'unknown', 'end'): -1299,
    ('exact', 7, 1097, 3, 7, 0, 'unknown', 'mid'): -511,
    ('exact', 7, 1097, 3, 860, 0, 'marnie', 'late'): -2197,
    ('exact', 7, 1122, 3, 1219, 0, 'rocket', 'mid'): 1946,
    ('exact', 7, 1122, 3, 1219, 0, 'unknown', 'open'): 0,
    ('exact', 7, 1122, 3, 1227, 0, 'marnie', 'late'): 511,
    ('exact', 7, 1122, 3, 1227, 0, 'unknown', 'open'): 511,
    ('exact', 7, 1152, 3, 104, 0, 'alakazam', 'mid'): 511,
    ('exact', 7, 1152, 3, 104, 0, 'crustle', 'mid'): -336,
    ('exact', 7, 1152, 3, 104, 0, 'lucario', 'mid'): -1946,
    ('exact', 7, 1152, 3, 104, 0, 'marnie', 'end'): 1099,
    ('exact', 7, 1152, 3, 104, 0, 'marnie', 'late'): -1099,
    ('exact', 7, 1152, 3, 104, 0, 'marnie', 'mid'): -2944,
    ('exact', 7, 1152, 3, 104, 0, 'rocket', 'mid'): -847,
    ('exact', 7, 1152, 3, 104, 0, 'rocket', 'open'): 0,
    ('exact', 7, 1152, 3, 104, 0, 'unknown', 'open'): -2856,
    ('exact', 7, 1152, 3, 112, 0, 'alakazam', 'end'): -336,
    ('exact', 7, 1152, 3, 112, 0, 'alakazam', 'late'): -511,
    ('exact', 7, 1152, 3, 112, 0, 'alakazam', 'mid'): -1825,
    ('exact', 7, 1152, 3, 112, 0, 'crustle', 'late'): -2197,
    ('exact', 7, 1152, 3, 112, 0, 'crustle', 'mid'): -1335,
    ('exact', 7, 1152, 3, 112, 0, 'lucario', 'mid'): -336,
    ('exact', 7, 1152, 3, 112, 0, 'marnie', 'late'): 847,
    ('exact', 7, 1152, 3, 112, 0, 'marnie', 'mid'): -138,
    ('exact', 7, 1152, 3, 112, 0, 'rocket', 'mid'): -511,
    ('exact', 7, 1152, 3, 112, 0, 'rocket', 'open'): -2752,
    ('exact', 7, 1152, 3, 112, 0, 'unknown', 'mid'): 0,
    ('exact', 7, 1152, 3, 112, 0, 'unknown', 'open'): -1139,
    ('exact', 7, 1152, 3, 646, 0, 'crustle', 'late'): -2398,
    ('exact', 7, 1152, 3, 646, 0, 'crustle', 'mid'): -2708,
    ('exact', 7, 1152, 3, 646, 0, 'lucario', 'mid'): -511,
    ('exact', 7, 1152, 3, 646, 0, 'lucario', 'open'): -511,
    ('exact', 7, 1152, 3, 646, 0, 'marnie', 'mid'): -1099,
    ('exact', 7, 1152, 3, 646, 0, 'rocket', 'open'): -1224,
    ('exact', 7, 1152, 3, 646, 0, 'unknown', 'open'): -2136,
    ('exact', 7, 1152, 3, 647, 0, 'alakazam', 'end'): -847,
    ('exact', 7, 1152, 3, 647, 0, 'alakazam', 'mid'): -1609,
    ('exact', 7, 1152, 3, 647, 0, 'crustle', 'late'): -1946,
    ('exact', 7, 1152, 3, 647, 0, 'crustle', 'mid'): -1946,
    ('exact', 7, 1152, 3, 647, 0, 'lucario', 'mid'): -1946,
    ('exact', 7, 1152, 3, 647, 0, 'lucario', 'open'): -1946,
    ('exact', 7, 1152, 3, 647, 0, 'marnie', 'mid'): -1299,
    ('exact', 7, 1152, 3, 647, 0, 'rocket', 'late'): -1946,
    ('exact', 7, 1152, 3, 647, 0, 'rocket', 'mid'): -2708,
    ('exact', 7, 1152, 3, 647, 0, 'rocket', 'open'): -3219,
    ('exact', 7, 1152, 3, 647, 0, 'unknown', 'open'): -5030,
    ('exact', 7, 1152, 3, 860, 0, 'lucario', 'mid'): -2197,
    ('exact', 7, 1152, 3, 860, 0, 'marnie', 'end'): -2197,
    ('exact', 7, 1152, 3, 860, 0, 'marnie', 'late'): -2398,
    ('exact', 7, 1152, 3, 860, 0, 'marnie', 'mid'): -2615,
    ('exact', 7, 1152, 3, 860, 0, 'rocket', 'open'): -2565,
    ('exact', 7, 1152, 3, 860, 0, 'unknown', 'open'): -2398,
    ('exact', 7, 1219, 3, 1079, 0, 'alakazam', 'end'): -1946,
    ('exact', 7, 1219, 3, 1079, 0, 'alakazam', 'mid'): -1099,
    ('exact', 7, 1219, 3, 1079, 0, 'crustle', 'late'): -3045,
    ('exact', 7, 1219, 3, 1079, 0, 'crustle', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1079, 0, 'lucario', 'mid'): -2197,
    ('exact', 7, 1219, 3, 1079, 0, 'marnie', 'end'): -2269,
    ('exact', 7, 1219, 3, 1079, 0, 'marnie', 'late'): -2197,
    ('exact', 7, 1219, 3, 1079, 0, 'marnie', 'mid'): -1665,
    ('exact', 7, 1219, 3, 1079, 0, 'rocket', 'end'): -1946,
    ('exact', 7, 1219, 3, 1079, 0, 'rocket', 'late'): -2197,
    ('exact', 7, 1219, 3, 1079, 0, 'rocket', 'mid'): -2833,
    ('exact', 7, 1219, 3, 1079, 0, 'rocket', 'open'): -2197,
    ('exact', 7, 1219, 3, 1079, 0, 'unknown', 'open'): -3611,
    ('exact', 7, 1219, 3, 1080, 0, 'alakazam', 'mid'): 511,
    ('exact', 7, 1219, 3, 1080, 0, 'crustle', 'late'): -511,
    ('exact', 7, 1219, 3, 1080, 0, 'marnie', 'end'): 2565,
    ('exact', 7, 1219, 3, 1080, 0, 'marnie', 'late'): 847,
    ('exact', 7, 1219, 3, 1080, 0, 'marnie', 'mid'): 251,
    ('exact', 7, 1219, 3, 1080, 0, 'rocket', 'mid'): 2197,
    ('exact', 7, 1219, 3, 1080, 0, 'unknown', 'open'): 0,
    ('exact', 7, 1219, 3, 1086, 0, 'alakazam', 'end'): -2944,
    ('exact', 7, 1219, 3, 1086, 0, 'alakazam', 'mid'): -3555,
    ('exact', 7, 1219, 3, 1086, 0, 'crustle', 'end'): -1946,
    ('exact', 7, 1219, 3, 1086, 0, 'crustle', 'late'): -2944,
    ('exact', 7, 1219, 3, 1086, 0, 'crustle', 'mid'): -2398,
    ('exact', 7, 1219, 3, 1086, 0, 'dragapult', 'end'): -1946,
    ('exact', 7, 1219, 3, 1086, 0, 'lucario', 'mid'): -2565,
    ('exact', 7, 1219, 3, 1086, 0, 'lucario', 'open'): -1946,
    ('exact', 7, 1219, 3, 1086, 0, 'marnie', 'end'): -3555,
    ('exact', 7, 1219, 3, 1086, 0, 'marnie', 'late'): -3850,
    ('exact', 7, 1219, 3, 1086, 0, 'marnie', 'mid'): -4111,
    ('exact', 7, 1219, 3, 1086, 0, 'rocket', 'end'): -1946,
    ('exact', 7, 1219, 3, 1086, 0, 'rocket', 'late'): -1299,
    ('exact', 7, 1219, 3, 1086, 0, 'rocket', 'mid'): -3135,
    ('exact', 7, 1219, 3, 1086, 0, 'rocket', 'open'): -2398,
    ('exact', 7, 1219, 3, 1086, 0, 'unknown', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1086, 0, 'unknown', 'open'): -2001,
    ('exact', 7, 1219, 3, 1097, 0, 'alakazam', 'end'): 0,
    ('exact', 7, 1219, 3, 1097, 0, 'alakazam', 'mid'): -3219,
    ('exact', 7, 1219, 3, 1097, 0, 'crustle', 'late'): -2565,
    ('exact', 7, 1219, 3, 1097, 0, 'crustle', 'mid'): -2398,
    ('exact', 7, 1219, 3, 1097, 0, 'cynthia', 'open'): -1946,
    ('exact', 7, 1219, 3, 1097, 0, 'lucario', 'mid'): -2565,
    ('exact', 7, 1219, 3, 1097, 0, 'lucario', 'open'): -1946,
    ('exact', 7, 1219, 3, 1097, 0, 'marnie', 'end'): -3045,
    ('exact', 7, 1219, 3, 1097, 0, 'marnie', 'late'): -1414,
    ('exact', 7, 1219, 3, 1097, 0, 'marnie', 'mid'): -3761,
    ('exact', 7, 1219, 3, 1097, 0, 'rocket', 'late'): -2398,
    ('exact', 7, 1219, 3, 1097, 0, 'rocket', 'mid'): -2944,
    ('exact', 7, 1219, 3, 1097, 0, 'rocket', 'open'): -2398,
    ('exact', 7, 1219, 3, 1097, 0, 'unknown', 'mid'): -511,
    ('exact', 7, 1219, 3, 1097, 0, 'unknown', 'open'): -3434,
    ('exact', 7, 1219, 3, 1122, 0, 'alakazam', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1122, 0, 'marnie', 'end'): -2197,
    ('exact', 7, 1219, 3, 1122, 0, 'marnie', 'late'): -2565,
    ('exact', 7, 1219, 3, 1122, 0, 'marnie', 'mid'): -2833,
    ('exact', 7, 1219, 3, 1122, 0, 'rocket', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1122, 0, 'unknown', 'open'): -2197,
    ('exact', 7, 1219, 3, 1137, 0, 'alakazam', 'end'): -1946,
    ('exact', 7, 1219, 3, 1137, 0, 'alakazam', 'mid'): -2565,
    ('exact', 7, 1219, 3, 1137, 0, 'crustle', 'late'): -1946,
    ('exact', 7, 1219, 3, 1137, 0, 'marnie', 'end'): -2708,
    ('exact', 7, 1219, 3, 1137, 0, 'marnie', 'late'): -2708,
    ('exact', 7, 1219, 3, 1137, 0, 'marnie', 'mid'): -2708,
    ('exact', 7, 1219, 3, 1137, 0, 'unknown', 'open'): -2398,
    ('exact', 7, 1219, 3, 1152, 0, 'alakazam', 'end'): -1946,
    ('exact', 7, 1219, 3, 1152, 0, 'alakazam', 'mid'): -1846,
    ('exact', 7, 1219, 3, 1152, 0, 'crustle', 'late'): -251,
    ('exact', 7, 1219, 3, 1152, 0, 'crustle', 'mid'): -1099,
    ('exact', 7, 1219, 3, 1152, 0, 'cynthia', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1152, 0, 'cynthia', 'open'): -1946,
    ('exact', 7, 1219, 3, 1152, 0, 'lucario', 'mid'): -1609,
    ('exact', 7, 1219, 3, 1152, 0, 'lucario', 'open'): -2197,
    ('exact', 7, 1219, 3, 1152, 0, 'marnie', 'end'): -3045,
    ('exact', 7, 1219, 3, 1152, 0, 'marnie', 'late'): -3664,
    ('exact', 7, 1219, 3, 1152, 0, 'marnie', 'mid'): -2241,
    ('exact', 7, 1219, 3, 1152, 0, 'rocket', 'late'): -1466,
    ('exact', 7, 1219, 3, 1152, 0, 'rocket', 'mid'): -3135,
    ('exact', 7, 1219, 3, 1152, 0, 'rocket', 'open'): -2565,
    ('exact', 7, 1219, 3, 1152, 0, 'unknown', 'open'): -3761,
    ('exact', 7, 1219, 3, 1182, 0, 'alakazam', 'end'): -2197,
    ('exact', 7, 1219, 3, 1182, 0, 'alakazam', 'mid'): -2944,
    ('exact', 7, 1219, 3, 1182, 0, 'crustle', 'late'): -1946,
    ('exact', 7, 1219, 3, 1182, 0, 'marnie', 'end'): -1846,
    ('exact', 7, 1219, 3, 1182, 0, 'marnie', 'late'): -1686,
    ('exact', 7, 1219, 3, 1182, 0, 'marnie', 'mid'): -3296,
    ('exact', 7, 1219, 3, 1182, 0, 'rocket', 'late'): -2197,
    ('exact', 7, 1219, 3, 1182, 0, 'rocket', 'mid'): -2398,
    ('exact', 7, 1219, 3, 1182, 0, 'unknown', 'open'): -3045,
    ('exact', 7, 1219, 3, 1219, 0, 'alakazam', 'end'): -2197,
    ('exact', 7, 1219, 3, 1219, 0, 'alakazam', 'mid'): -3135,
    ('exact', 7, 1219, 3, 1219, 0, 'crustle', 'late'): -2708,
    ('exact', 7, 1219, 3, 1219, 0, 'crustle', 'mid'): -2398,
    ('exact', 7, 1219, 3, 1219, 0, 'marnie', 'end'): -3497,
    ('exact', 7, 1219, 3, 1219, 0, 'marnie', 'late'): -3807,
    ('exact', 7, 1219, 3, 1219, 0, 'marnie', 'mid'): -3892,
    ('exact', 7, 1219, 3, 1219, 0, 'rocket', 'end'): -2197,
    ('exact', 7, 1219, 3, 1219, 0, 'rocket', 'late'): -2197,
    ('exact', 7, 1219, 3, 1219, 0, 'rocket', 'mid'): -3045,
    ('exact', 7, 1219, 3, 1219, 0, 'rocket', 'open'): -2565,
    ('exact', 7, 1219, 3, 1219, 0, 'unknown', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1219, 0, 'unknown', 'open'): -3611,
    ('exact', 7, 1219, 3, 1227, 0, 'alakazam', 'end'): -1946,
    ('exact', 7, 1219, 3, 1227, 0, 'alakazam', 'mid'): -3296,
    ('exact', 7, 1219, 3, 1227, 0, 'crustle', 'end'): -1946,
    ('exact', 7, 1219, 3, 1227, 0, 'crustle', 'late'): -2944,
    ('exact', 7, 1219, 3, 1227, 0, 'crustle', 'mid'): -1299,
    ('exact', 7, 1219, 3, 1227, 0, 'cynthia', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1227, 0, 'cynthia', 'open'): -1946,
    ('exact', 7, 1219, 3, 1227, 0, 'lucario', 'mid'): -2833,
    ('exact', 7, 1219, 3, 1227, 0, 'lucario', 'open'): -2197,
    ('exact', 7, 1219, 3, 1227, 0, 'marnie', 'end'): -3219,
    ('exact', 7, 1219, 3, 1227, 0, 'marnie', 'late'): -3807,
    ('exact', 7, 1219, 3, 1227, 0, 'marnie', 'mid'): -4078,
    ('exact', 7, 1219, 3, 1227, 0, 'rocket', 'late'): -1099,
    ('exact', 7, 1219, 3, 1227, 0, 'rocket', 'mid'): -3219,
    ('exact', 7, 1219, 3, 1227, 0, 'rocket', 'open'): -956,
    ('exact', 7, 1219, 3, 1227, 0, 'unknown', 'end'): -1946,
    ('exact', 7, 1219, 3, 1227, 0, 'unknown', 'open'): -2361,
    ('exact', 7, 1219, 3, 1231, 0, 'alakazam', 'mid'): -2197,
    ('exact', 7, 1219, 3, 1231, 0, 'crustle', 'late'): -1946,
    ('exact', 7, 1219, 3, 1231, 0, 'marnie', 'end'): -1946,
    ('exact', 7, 1219, 3, 1231, 0, 'marnie', 'late'): -2197,
    ('exact', 7, 1219, 3, 1231, 0, 'marnie', 'mid'): -2565,
    ('exact', 7, 1219, 3, 1231, 0, 'rocket', 'mid'): -2197,
    ('exact', 7, 1219, 3, 1231, 0, 'unknown', 'open'): -2565,
    ('exact', 7, 1219, 3, 1259, 0, 'alakazam', 'end'): -2708,
    ('exact', 7, 1219, 3, 1259, 0, 'alakazam', 'mid'): -3497,
    ('exact', 7, 1219, 3, 1259, 0, 'crustle', 'late'): -3045,
    ('exact', 7, 1219, 3, 1259, 0, 'crustle', 'mid'): -2708,
    ('exact', 7, 1219, 3, 1259, 0, 'cynthia', 'mid'): -1946,
    ('exact', 7, 1219, 3, 1259, 0, 'cynthia', 'open'): -847,
    ('exact', 7, 1219, 3, 1259, 0, 'dragapult', 'end'): -1946,
    ('exact', 7, 1219, 3, 1259, 0, 'lucario', 'late'): -1946,
    ('exact', 7, 1219, 3, 1259, 0, 'marnie', 'end'): -3892,
    ('exact', 7, 1219, 3, 1259, 0, 'marnie', 'late'): -4007,
    ('exact', 7, 1219, 3, 1259, 0, 'marnie', 'mid'): -2793,
    ('exact', 7, 1219, 3, 1259, 0, 'rocket', 'end'): -1946,
    ('exact', 7, 1219, 3, 1259, 0, 'rocket', 'late'): -2398,
    ('exact', 7, 1219, 3, 1259, 0, 'rocket', 'mid'): -1735,
    ('exact', 7, 1219, 3, 1259, 0, 'rocket', 'open'): -2398,
    ('exact', 7, 1219, 3, 1259, 0, 'unknown', 'open'): -3761,
    ('exact', 7, 1231, 3, 104, 0, 'alakazam', 'late'): 511,
    ('exact', 7, 1231, 3, 104, 0, 'marnie', 'end'): 0,
    ('exact', 7, 1231, 3, 104, 0, 'marnie', 'mid'): -788,
    ('exact', 7, 1231, 3, 104, 0, 'unknown', 'mid'): -511,
    ('exact', 7, 1231, 3, 104, 0, 'unknown', 'open'): -336,
    ('exact', 7, 1231, 3, 112, 0, 'alakazam', 'late'): -336,
    ('exact', 7, 1231, 3, 112, 0, 'marnie', 'late'): -511,
    ('exact', 7, 1231, 3, 112, 0, 'marnie', 'mid'): 251,
    ('exact', 7, 1231, 3, 112, 0, 'rocket', 'end'): 0,
    ('exact', 7, 1231, 3, 112, 0, 'rocket', 'mid'): -511,
    ('exact', 7, 1231, 3, 112, 0, 'unknown', 'mid'): 511,
    ('exact', 7, 1231, 3, 112, 0, 'unknown', 'open'): -788,
    ('exact', 7, 1231, 3, 646, 0, 'unknown', 'mid'): -2197,
    ('exact', 7, 1231, 3, 646, 0, 'unknown', 'open'): -2565,
    ('exact', 7, 1231, 3, 647, 0, 'marnie', 'mid'): 511,
    ('exact', 7, 1231, 3, 647, 0, 'unknown', 'mid'): -511,
    ('exact', 7, 1231, 3, 647, 0, 'unknown', 'open'): -1466,
    ('exact', 7, 1231, 3, 648, 0, 'marnie', 'mid'): 847,
    ('exact', 7, 1231, 3, 648, 0, 'rocket', 'mid'): -511,
    ('exact', 7, 1231, 3, 648, 0, 'unknown', 'open'): 0,
    ('exact', 7, 1231, 3, 860, 0, 'marnie', 'end'): -1299,
    ('exact', 7, 1231, 3, 860, 0, 'marnie', 'mid'): -2565,
    ('exact', 7, 1231, 3, 860, 0, 'unknown', 'open'): -847,
    ('exact', 7, 1259, 3, 646, 0, 'crustle', 'late'): -1350,
    ('exact', 7, 1259, 3, 646, 0, 'crustle', 'mid'): -1099,
    ('exact', 7, 1259, 3, 646, 0, 'lucario', 'mid'): -2398,
    ('exact', 7, 1259, 3, 646, 0, 'lucario', 'open'): -511,
    ('exact', 7, 1259, 3, 646, 0, 'marnie', 'end'): 336,
    ('exact', 7, 1259, 3, 646, 0, 'marnie', 'late'): 511,
    ('exact', 7, 1259, 3, 646, 0, 'marnie', 'mid'): -956,
    ('exact', 7, 1259, 3, 646, 0, 'rocket', 'mid'): -1466,
    ('exact', 7, 1259, 3, 646, 0, 'rocket', 'open'): -452,
    ('exact', 7, 1259, 3, 646, 0, 'unknown', 'mid'): -1099,
    ('exact', 7, 1259, 3, 646, 0, 'unknown', 'open'): -1201,
    ('exact', 7, 1259, 3, 647, 0, 'alakazam', 'mid'): -251,
    ('exact', 7, 1259, 3, 647, 0, 'crustle', 'late'): -452,
    ('exact', 7, 1259, 3, 647, 0, 'cynthia', 'mid'): -1099,
    ('exact', 7, 1259, 3, 647, 0, 'cynthia', 'open'): -1946,
    ('exact', 7, 1259, 3, 647, 0, 'lucario', 'mid'): -788,
    ('exact', 7, 1259, 3, 647, 0, 'marnie', 'end'): -847,
    ('exact', 7, 1259, 3, 647, 0, 'marnie', 'late'): -1170,
    ('exact', 7, 1259, 3, 647, 0, 'marnie', 'mid'): -908,
    ('exact', 7, 1259, 3, 647, 0, 'rocket', 'end'): -336,
    ('exact', 7, 1259, 3, 647, 0, 'rocket', 'late'): -511,
    ('exact', 7, 1259, 3, 647, 0, 'rocket', 'mid'): -2944,
    ('exact', 7, 1259, 3, 647, 0, 'rocket', 'open'): -1435,
    ('exact', 7, 1259, 3, 647, 0, 'unknown', 'mid'): -747,
    ('exact', 7, 1259, 3, 647, 0, 'unknown', 'open'): -1671,
    ('exact', 7, 1259, 3, 648, 0, 'alakazam', 'mid'): -588,
    ('exact', 7, 1259, 3, 648, 0, 'crustle', 'late'): -636,
    ('exact', 7, 1259, 3, 648, 0, 'crustle', 'mid'): -999,
    ('exact', 7, 1259, 3, 648, 0, 'cynthia', 'mid'): -847,
    ('exact', 7, 1259, 3, 648, 0, 'cynthia', 'open'): -511,
    ('exact', 7, 1259, 3, 648, 0, 'lucario', 'mid'): -336,
    ('exact', 7, 1259, 3, 648, 0, 'marnie', 'end'): 2197,
    ('exact', 7, 1259, 3, 648, 0, 'marnie', 'late'): 654,
    ('exact', 7, 1259, 3, 648, 0, 'marnie', 'mid'): -612,
    ('exact', 7, 1259, 3, 648, 0, 'rocket', 'late'): 511,
    ('exact', 7, 1259, 3, 648, 0, 'rocket', 'mid'): -125,
    ('exact', 7, 1259, 3, 648, 0, 'rocket', 'open'): -3045,
    ('exact', 7, 1259, 3, 648, 0, 'unknown', 'mid'): -762,
    ('exact', 7, 1259, 3, 648, 0, 'unknown', 'open'): -1961,
    ('exact', 8, 1197, 3, 1182, 0, 'unknown', 'mid'): 1946,
    ('phase', 0, 0, 10, 112, 0, 'end'): 32,
    ('phase', 0, 0, 10, 112, 0, 'late'): 199,
    ('phase', 0, 0, 10, 112, 0, 'mid'): 216,
    ('phase', 0, 0, 10, 112, 0, 'open'): -588,
    ('phase', 0, 0, 10, 1257, 0, 'mid'): 847,
    ('phase', 0, 0, 10, 1259, 0, 'end'): -1083,
    ('phase', 0, 0, 10, 1259, 0, 'late'): -1337,
    ('phase', 0, 0, 10, 1259, 0, 'mid'): -1004,
    ('phase', 0, 0, 10, 1259, 0, 'open'): -454,
    ('phase', 0, 0, 12, 0, 0, 'end'): -3403,
    ('phase', 0, 0, 12, 0, 0, 'late'): -3082,
    ('phase', 0, 0, 12, 0, 0, 'mid'): -3201,
    ('phase', 0, 0, 12, 0, 0, 'open'): -2944,
    ('phase', 0, 0, 13, 934, 0, 'end'): -1466,
    ('phase', 0, 0, 13, 934, 0, 'mid'): -3664,
    ('phase', 0, 0, 13, 934, 0, 'open'): -802,
    ('phase', 0, 0, 13, 935, 0, 'end'): -2708,
    ('phase', 0, 0, 13, 935, 0, 'mid'): -4779,
    ('phase', 0, 0, 13, 935, 0, 'open'): -3714,
    ('phase', 0, 0, 13, 936, 0, 'late'): -2398,
    ('phase', 0, 0, 13, 936, 0, 'mid'): -3296,
    ('phase', 0, 0, 13, 937, 0, 'end'): -1605,
    ('phase', 0, 0, 13, 937, 0, 'late'): -1763,
    ('phase', 0, 0, 13, 937, 0, 'mid'): -1510,
    ('phase', 0, 0, 13, 937, 0, 'open'): 368,
    ('phase', 0, 0, 14, 0, 0, 'end'): -4498,
    ('phase', 0, 0, 14, 0, 0, 'late'): -4435,
    ('phase', 0, 0, 14, 0, 0, 'mid'): -3479,
    ('phase', 0, 0, 14, 0, 0, 'open'): -1924,
    ('phase', 0, 0, 7, 1079, 0, 'end'): -619,
    ('phase', 0, 0, 7, 1079, 0, 'late'): -1099,
    ('phase', 0, 0, 7, 1079, 0, 'mid'): -547,
    ('phase', 0, 0, 7, 1079, 0, 'open'): -125,
    ('phase', 0, 0, 7, 1080, 0, 'end'): 788,
    ('phase', 0, 0, 7, 1080, 0, 'late'): -990,
    ('phase', 0, 0, 7, 1080, 0, 'mid'): -978,
    ('phase', 0, 0, 7, 1086, 0, 'end'): -3476,
    ('phase', 0, 0, 7, 1086, 0, 'late'): -2534,
    ('phase', 0, 0, 7, 1086, 0, 'mid'): -1290,
    ('phase', 0, 0, 7, 1086, 0, 'open'): -265,
    ('phase', 0, 0, 7, 1097, 0, 'end'): -1099,
    ('phase', 0, 0, 7, 1097, 0, 'late'): -326,
    ('phase', 0, 0, 7, 1097, 0, 'mid'): -900,
    ('phase', 0, 0, 7, 112, 0, 'end'): -666,
    ('phase', 0, 0, 7, 112, 0, 'late'): 373,
    ('phase', 0, 0, 7, 112, 0, 'mid'): 726,
    ('phase', 0, 0, 7, 112, 0, 'open'): 999,
    ('phase', 0, 0, 7, 1122, 0, 'end'): -1946,
    ('phase', 0, 0, 7, 1122, 0, 'late'): 201,
    ('phase', 0, 0, 7, 1122, 0, 'mid'): 0,
    ('phase', 0, 0, 7, 1122, 0, 'open'): -1099,
    ('phase', 0, 0, 7, 1137, 0, 'mid'): -1466,
    ('phase', 0, 0, 7, 1137, 0, 'open'): -2398,
    ('phase', 0, 0, 7, 1152, 0, 'end'): -922,
    ('phase', 0, 0, 7, 1152, 0, 'late'): -525,
    ('phase', 0, 0, 7, 1152, 0, 'mid'): -520,
    ('phase', 0, 0, 7, 1152, 0, 'open'): -502,
    ('phase', 0, 0, 7, 1182, 0, 'end'): -2446,
    ('phase', 0, 0, 7, 1182, 0, 'late'): -2429,
    ('phase', 0, 0, 7, 1182, 0, 'mid'): -2481,
    ('phase', 0, 0, 7, 1182, 0, 'open'): -2587,
    ('phase', 0, 0, 7, 1219, 0, 'end'): -1929,
    ('phase', 0, 0, 7, 1219, 0, 'late'): -2224,
    ('phase', 0, 0, 7, 1219, 0, 'mid'): -1935,
    ('phase', 0, 0, 7, 1219, 0, 'open'): -1946,
    ('phase', 0, 0, 7, 1227, 0, 'end'): -2176,
    ('phase', 0, 0, 7, 1227, 0, 'late'): -1760,
    ('phase', 0, 0, 7, 1227, 0, 'mid'): -1382,
    ('phase', 0, 0, 7, 1227, 0, 'open'): -966,
    ('phase', 0, 0, 7, 1231, 0, 'end'): -1262,
    ('phase', 0, 0, 7, 1231, 0, 'late'): -2120,
    ('phase', 0, 0, 7, 1231, 0, 'mid'): -1907,
    ('phase', 0, 0, 7, 1231, 0, 'open'): -452,
    ('phase', 0, 0, 7, 1259, 0, 'end'): 511,
    ('phase', 0, 0, 7, 1259, 0, 'late'): -847,
    ('phase', 0, 0, 7, 1259, 0, 'mid'): -160,
    ('phase', 0, 0, 7, 1259, 0, 'open'): 498,
    ('phase', 0, 0, 7, 646, 0, 'end'): -2001,
    ('phase', 0, 0, 7, 646, 0, 'late'): -1561,
    ('phase', 0, 0, 7, 646, 0, 'mid'): 48,
    ('phase', 0, 0, 7, 646, 0, 'open'): 1048,
    ('phase', 0, 0, 7, 860, 0, 'end'): -3761,
    ('phase', 0, 0, 7, 860, 0, 'late'): -3600,
    ('phase', 0, 0, 7, 860, 0, 'mid'): -2872,
    ('phase', 0, 0, 7, 860, 0, 'open'): -1511,
    ('phase', 0, 0, 8, 7, 104, 'end'): -3574,
    ('phase', 0, 0, 8, 7, 104, 'late'): -3970,
    ('phase', 0, 0, 8, 7, 104, 'mid'): -3606,
    ('phase', 0, 0, 8, 7, 104, 'open'): -4078,
    ('phase', 0, 0, 8, 7, 112, 'end'): -2989,
    ('phase', 0, 0, 8, 7, 112, 'late'): -3131,
    ('phase', 0, 0, 8, 7, 112, 'mid'): -2729,
    ('phase', 0, 0, 8, 7, 112, 'open'): -2214,
    ('phase', 0, 0, 8, 7, 646, 'end'): -3892,
    ('phase', 0, 0, 8, 7, 646, 'late'): -5142,
    ('phase', 0, 0, 8, 7, 646, 'mid'): -5930,
    ('phase', 0, 0, 8, 7, 646, 'open'): -3962,
    ('phase', 0, 0, 8, 7, 647, 'end'): -3714,
    ('phase', 0, 0, 8, 7, 647, 'late'): -6290,
    ('phase', 0, 0, 8, 7, 647, 'mid'): -6819,
    ('phase', 0, 0, 8, 7, 647, 'open'): -3567,
    ('phase', 0, 0, 8, 7, 648, 'end'): -3181,
    ('phase', 0, 0, 8, 7, 648, 'late'): -3723,
    ('phase', 0, 0, 8, 7, 648, 'mid'): -6347,
    ('phase', 0, 0, 8, 7, 648, 'open'): -2944,
    ('phase', 0, 0, 8, 7, 860, 'late'): -4949,
    ('phase', 0, 0, 8, 7, 860, 'mid'): -6089,
    ('phase', 0, 0, 8, 7, 860, 'open'): -4187,
    ('phase', 0, 0, 9, 104, 860, 'late'): 0,
    ('phase', 0, 0, 9, 104, 860, 'mid'): 1006,
    ('phase', 0, 0, 9, 104, 860, 'open'): -211,
    ('phase', 0, 0, 9, 647, 646, 'end'): -619,
    ('phase', 0, 0, 9, 647, 646, 'late'): -1224,
    ('phase', 0, 0, 9, 647, 646, 'mid'): -1047,
    ('phase', 0, 0, 9, 647, 646, 'open'): -604,
    ('phase', 0, 0, 9, 648, 647, 'end'): -1711,
    ('phase', 0, 0, 9, 648, 647, 'late'): -1800,
    ('phase', 0, 0, 9, 648, 647, 'mid'): -1588,
    ('phase', 1, 0, 3, 112, 0, 'end'): -108,
    ('phase', 1, 0, 3, 646, 0, 'end'): 1242,
    ('phase', 1, 0, 3, 860, 0, 'end'): 1846,
    ('phase', 13, 112, 3, 104, 0, 'end'): -3497,
    ('phase', 13, 112, 3, 104, 0, 'late'): -2197,
    ('phase', 13, 112, 3, 104, 0, 'mid'): -3807,
    ('phase', 13, 112, 3, 112, 0, 'end'): -757,
    ('phase', 13, 112, 3, 112, 0, 'late'): -580,
    ('phase', 13, 112, 3, 112, 0, 'mid'): -247,
    ('phase', 13, 112, 3, 119, 0, 'mid'): -2944,
    ('phase', 13, 112, 3, 140, 0, 'end'): -2833,
    ('phase', 13, 112, 3, 175, 0, 'late'): -2197,
    ('phase', 13, 112, 3, 305, 0, 'end'): -1946,
    ('phase', 13, 112, 3, 305, 0, 'late'): -1946,
    ('phase', 13, 112, 3, 305, 0, 'mid'): -2565,
    ('phase', 13, 112, 3, 341, 0, 'mid'): 511,
    ('phase', 13, 112, 3, 342, 0, 'mid'): -2565,
    ('phase', 13, 112, 3, 343, 0, 'end'): 2197,
    ('phase', 13, 112, 3, 343, 0, 'mid'): -847,
    ('phase', 13, 112, 3, 344, 0, 'late'): -511,
    ('phase', 13, 112, 3, 344, 0, 'mid'): -956,
    ('phase', 13, 112, 3, 345, 0, 'end'): 738,
    ('phase', 13, 112, 3, 345, 0, 'late'): 416,
    ('phase', 13, 112, 3, 345, 0, 'mid'): 2944,
    ('phase', 13, 112, 3, 380, 0, 'mid'): -511,
    ('phase', 13, 112, 3, 381, 0, 'end'): -2197,
    ('phase', 13, 112, 3, 381, 0, 'mid'): -2197,
    ('phase', 13, 112, 3, 400, 0, 'late'): 0,
    ('phase', 13, 112, 3, 400, 0, 'mid'): 1946,
    ('phase', 13, 112, 3, 401, 0, 'end'): 1946,
    ('phase', 13, 112, 3, 401, 0, 'late'): 336,
    ('phase', 13, 112, 3, 401, 0, 'mid'): -1099,
    ('phase', 13, 112, 3, 414, 0, 'end'): -659,
    ('phase', 13, 112, 3, 414, 0, 'late'): -2565,
    ('phase', 13, 112, 3, 431, 0, 'end'): -2615,
    ('phase', 13, 112, 3, 431, 0, 'late'): -2565,
    ('phase', 13, 112, 3, 431, 0, 'mid'): -2833,
    ('phase', 13, 112, 3, 434, 0, 'end'): -1718,
    ('phase', 13, 112, 3, 434, 0, 'late'): -1609,
    ('phase', 13, 112, 3, 434, 0, 'mid'): -788,
    ('phase', 13, 112, 3, 646, 0, 'end'): -547,
    ('phase', 13, 112, 3, 646, 0, 'late'): -1367,
    ('phase', 13, 112, 3, 646, 0, 'mid'): -928,
    ('phase', 13, 112, 3, 647, 0, 'end'): -2708,
    ('phase', 13, 112, 3, 647, 0, 'late'): -2335,
    ('phase', 13, 112, 3, 647, 0, 'mid'): -3517,
    ('phase', 13, 112, 3, 648, 0, 'end'): -1426,
    ('phase', 13, 112, 3, 648, 0, 'late'): -2398,
    ('phase', 13, 112, 3, 648, 0, 'mid'): -4844,
    ('phase', 13, 112, 3, 673, 0, 'late'): -511,
    ('phase', 13, 112, 3, 675, 0, 'late'): -1946,
    ('phase', 13, 112, 3, 676, 0, 'late'): -1946,
    ('phase', 13, 112, 3, 678, 0, 'late'): -2197,
    ('phase', 13, 112, 3, 689, 0, 'late'): -1946,
    ('phase', 13, 112, 3, 689, 0, 'mid'): -1946,
    ('phase', 13, 112, 3, 741, 0, 'late'): 0,
    ('phase', 13, 112, 3, 741, 0, 'mid'): 588,
    ('phase', 13, 112, 3, 742, 0, 'end'): -336,
    ('phase', 13, 112, 3, 742, 0, 'mid'): -1946,
    ('phase', 13, 112, 3, 743, 0, 'end'): -2944,
    ('phase', 13, 112, 3, 743, 0, 'late'): -1946,
    ('phase', 13, 112, 3, 743, 0, 'mid'): -2708,
    ('phase', 13, 112, 3, 756, 0, 'late'): -611,
    ('phase', 13, 112, 3, 756, 0, 'mid'): -3045,
    ('phase', 13, 112, 3, 860, 0, 'end'): 762,
    ('phase', 13, 112, 3, 860, 0, 'late'): -361,
    ('phase', 13, 112, 3, 860, 0, 'mid'): -3296,
    ('phase', 15, 648, 3, 104, 0, 'end'): -2833,
    ('phase', 15, 648, 3, 104, 0, 'late'): -1435,
    ('phase', 15, 648, 3, 104, 0, 'mid'): -2833,
    ('phase', 15, 648, 3, 112, 0, 'end'): -565,
    ('phase', 15, 648, 3, 112, 0, 'late'): -368,
    ('phase', 15, 648, 3, 112, 0, 'mid'): 78,
    ('phase', 15, 648, 3, 119, 0, 'mid'): -588,
    ('phase', 15, 648, 3, 140, 0, 'end'): -1946,
    ('phase', 15, 648, 3, 140, 0, 'mid'): -1946,
    ('phase', 15, 648, 3, 175, 0, 'late'): -2565,
    ('phase', 15, 648, 3, 184, 0, 'late'): 1946,
    ('phase', 15, 648, 3, 24, 0, 'late'): -1946,
    ('phase', 15, 648, 3, 305, 0, 'end'): -588,
    ('phase', 15, 648, 3, 305, 0, 'late'): -1946,
    ('phase', 15, 648, 3, 305, 0, 'mid'): -3367,
    ('phase', 15, 648, 3, 343, 0, 'mid'): 0,
    ('phase', 15, 648, 3, 345, 0, 'late'): 452,
    ('phase', 15, 648, 3, 379, 0, 'open'): -511,
    ('phase', 15, 648, 3, 380, 0, 'mid'): -511,
    ('phase', 15, 648, 3, 400, 0, 'mid'): 251,
    ('phase', 15, 648, 3, 400, 0, 'open'): -847,
    ('phase', 15, 648, 3, 401, 0, 'mid'): -511,
    ('phase', 15, 648, 3, 414, 0, 'end'): -956,
    ('phase', 15, 648, 3, 414, 0, 'late'): -1299,
    ('phase', 15, 648, 3, 414, 0, 'mid'): -2565,
    ('phase', 15, 648, 3, 431, 0, 'end'): -1299,
    ('phase', 15, 648, 3, 431, 0, 'late'): -2197,
    ('phase', 15, 648, 3, 431, 0, 'mid'): -3045,
    ('phase', 15, 648, 3, 434, 0, 'end'): 511,
    ('phase', 15, 648, 3, 434, 0, 'late'): -452,
    ('phase', 15, 648, 3, 434, 0, 'mid'): -1435,
    ('phase', 15, 648, 3, 646, 0, 'end'): 0,
    ('phase', 15, 648, 3, 646, 0, 'late'): -636,
    ('phase', 15, 648, 3, 646, 0, 'mid'): -1488,
    ('phase', 15, 648, 3, 647, 0, 'end'): -1099,
    ('phase', 15, 648, 3, 647, 0, 'late'): -2269,
    ('phase', 15, 648, 3, 647, 0, 'mid'): -2024,
    ('phase', 15, 648, 3, 648, 0, 'end'): -236,
    ('phase', 15, 648, 3, 648, 0, 'late'): -2228,
    ('phase', 15, 648, 3, 648, 0, 'mid'): -3367,
    ('phase', 15, 648, 3, 673, 0, 'mid'): 1946,
    ('phase', 15, 648, 3, 675, 0, 'late'): -1946,
    ('phase', 15, 648, 3, 675, 0, 'mid'): -1946,
    ('phase', 15, 648, 3, 676, 0, 'late'): -1946,
    ('phase', 15, 648, 3, 677, 0, 'mid'): -1946,
    ('phase', 15, 648, 3, 741, 0, 'late'): -847,
    ('phase', 15, 648, 3, 741, 0, 'mid'): 619,
    ('phase', 15, 648, 3, 741, 0, 'open'): -511,
    ('phase', 15, 648, 3, 742, 0, 'end'): -847,
    ('phase', 15, 648, 3, 742, 0, 'mid'): -1435,
    ('phase', 15, 648, 3, 743, 0, 'mid'): -2197,
    ('phase', 15, 648, 3, 860, 0, 'late'): -201,
    ('phase', 15, 648, 3, 860, 0, 'mid'): -1224,
    ('phase', 16, 112, 3, 104, 0, 'end'): -847,
    ('phase', 16, 112, 3, 112, 0, 'end'): -985,
    ('phase', 16, 112, 3, 112, 0, 'late'): -272,
    ('phase', 16, 112, 3, 112, 0, 'mid'): 88,
    ('phase', 16, 112, 3, 646, 0, 'late'): 1946,
    ('phase', 16, 112, 3, 646, 0, 'mid'): 1335,
    ('phase', 16, 112, 3, 647, 0, 'mid'): -1099,
    ('phase', 16, 112, 3, 648, 0, 'end'): -667,
    ('phase', 16, 112, 3, 648, 0, 'late'): -707,
    ('phase', 16, 112, 3, 648, 0, 'mid'): -196,
    ('phase', 2, 0, 3, 112, 0, 'open'): 3219,
    ('phase', 2, 0, 3, 646, 0, 'open'): 2398,
    ('phase', 21, 7, 3, 646, 0, 'late'): -956,
    ('phase', 21, 7, 3, 646, 0, 'mid'): -1027,
    ('phase', 21, 7, 3, 646, 0, 'open'): -862,
    ('phase', 21, 7, 3, 647, 0, 'late'): -1036,
    ('phase', 21, 7, 3, 647, 0, 'mid'): -941,
    ('phase', 21, 7, 3, 648, 0, 'end'): 53,
    ('phase', 21, 7, 3, 648, 0, 'late'): -274,
    ('phase', 21, 7, 3, 648, 0, 'mid'): 138,
    ('phase', 21, 7, 3, 648, 0, 'open'): 174,
    ('phase', 22, 648, 3, 7, 0, 'end'): 2054,
    ('phase', 22, 648, 3, 7, 0, 'late'): 760,
    ('phase', 22, 648, 3, 7, 0, 'mid'): 116,
    ('phase', 22, 648, 3, 7, 0, 'open'): -246,
    ('phase', 3, 0, 3, 104, 0, 'end'): -1946,
    ('phase', 3, 0, 3, 104, 0, 'late'): -2398,
    ('phase', 3, 0, 3, 112, 0, 'end'): -3761,
    ('phase', 3, 0, 3, 112, 0, 'late'): -4466,
    ('phase', 3, 0, 3, 112, 0, 'mid'): -4317,
    ('phase', 3, 0, 3, 112, 0, 'open'): -847,
    ('phase', 3, 0, 3, 646, 0, 'mid'): -2833,
    ('phase', 3, 0, 3, 646, 0, 'open'): -2398,
    ('phase', 3, 0, 3, 647, 0, 'end'): -1946,
    ('phase', 3, 0, 3, 647, 0, 'late'): -1846,
    ('phase', 3, 0, 3, 647, 0, 'mid'): -3555,
    ('phase', 3, 0, 3, 648, 0, 'end'): 2833,
    ('phase', 3, 0, 3, 648, 0, 'late'): 722,
    ('phase', 3, 0, 3, 648, 0, 'mid'): 2512,
    ('phase', 3, 0, 3, 648, 0, 'open'): 1946,
    ('phase', 3, 1182, 3, 104, 0, 'late'): -847,
    ('phase', 3, 1182, 3, 112, 0, 'end'): -1099,
    ('phase', 3, 1182, 3, 112, 0, 'late'): -747,
    ('phase', 3, 1182, 3, 112, 0, 'mid'): -547,
    ('phase', 3, 1182, 3, 646, 0, 'open'): -511,
    ('phase', 3, 1182, 3, 647, 0, 'late'): -1946,
    ('phase', 3, 1182, 3, 647, 0, 'mid'): -2565,
    ('phase', 3, 1182, 3, 648, 0, 'late'): -511,
    ('phase', 3, 1182, 3, 648, 0, 'mid'): -336,
    ('phase', 3, 1182, 3, 860, 0, 'late'): -1946,
    ('phase', 30, 0, 6, 104, 0, 'mid'): 2398,
    ('phase', 30, 0, 6, 112, 0, 'mid'): 1946,
    ('phase', 30, 0, 6, 646, 0, 'mid'): 847,
    ('phase', 30, 0, 6, 648, 0, 'end'): -67,
    ('phase', 30, 0, 6, 648, 0, 'late'): 179,
    ('phase', 30, 0, 6, 648, 0, 'mid'): 174,
    ('phase', 34, 0, 15, 0, 0, 'end'): 3714,
    ('phase', 34, 0, 15, 0, 0, 'late'): 3219,
    ('phase', 34, 0, 15, 0, 0, 'mid'): 3045,
    ('phase', 37, 1079, 9, 648, 646, 'end'): 1946,
    ('phase', 37, 1079, 9, 648, 646, 'late'): 1099,
    ('phase', 37, 1079, 9, 648, 646, 'mid'): 116,
    ('phase', 37, 1079, 9, 648, 646, 'open'): 143,
    ('phase', 38, 0, 0, 0, 0, 'open'): -3219,
    ('phase', 38, 0, 0, 1, 0, 'open'): 310,
    ('phase', 38, 0, 0, 2, 0, 'open'): 1099,
    ('phase', 4, 0, 3, 104, 0, 'end'): -2398,
    ('phase', 4, 0, 3, 104, 0, 'late'): -2833,
    ('phase', 4, 0, 3, 104, 0, 'mid'): -2565,
    ('phase', 4, 0, 3, 112, 0, 'end'): -3219,
    ('phase', 4, 0, 3, 112, 0, 'late'): -4344,
    ('phase', 4, 0, 3, 112, 0, 'mid'): -4762,
    ('phase', 4, 0, 3, 112, 0, 'open'): -2398,
    ('phase', 4, 0, 3, 646, 0, 'late'): -956,
    ('phase', 4, 0, 3, 646, 0, 'mid'): -1609,
    ('phase', 4, 0, 3, 646, 0, 'open'): 788,
    ('phase', 4, 0, 3, 647, 0, 'late'): 747,
    ('phase', 4, 0, 3, 647, 0, 'mid'): -373,
    ('phase', 4, 0, 3, 648, 0, 'end'): 0,
    ('phase', 4, 0, 3, 648, 0, 'late'): -302,
    ('phase', 4, 0, 3, 648, 0, 'mid'): 2054,
    ('phase', 4, 0, 3, 860, 0, 'late'): -1946,
    ('phase', 4, 0, 3, 860, 0, 'mid'): -3611,
    ('phase', 40, 112, 0, 1, 0, 'end'): -4533,
    ('phase', 40, 112, 0, 1, 0, 'late'): -5303,
    ('phase', 40, 112, 0, 1, 0, 'mid'): -4595,
    ('phase', 40, 112, 0, 2, 0, 'end'): -2021,
    ('phase', 40, 112, 0, 2, 0, 'late'): -1786,
    ('phase', 40, 112, 0, 2, 0, 'mid'): -708,
    ('phase', 40, 112, 0, 3, 0, 'end'): 4419,
    ('phase', 40, 112, 0, 3, 0, 'late'): 5153,
    ('phase', 40, 112, 0, 3, 0, 'mid'): 4205,
    ('phase', 40, 646, 0, 1, 0, 'late'): -1946,
    ('phase', 40, 646, 0, 1, 0, 'mid'): -2944,
    ('phase', 40, 646, 0, 2, 0, 'late'): -1946,
    ('phase', 40, 646, 0, 2, 0, 'mid'): -1735,
    ('phase', 40, 646, 0, 3, 0, 'late'): 1946,
    ('phase', 40, 646, 0, 3, 0, 'mid'): 2833,
    ('phase', 40, 648, 0, 1, 0, 'end'): -4554,
    ('phase', 40, 648, 0, 1, 0, 'late'): -4615,
    ('phase', 40, 648, 0, 1, 0, 'mid'): -3434,
    ('phase', 40, 648, 0, 2, 0, 'end'): -3434,
    ('phase', 40, 648, 0, 2, 0, 'late'): -2608,
    ('phase', 40, 648, 0, 2, 0, 'mid'): -1273,
    ('phase', 40, 648, 0, 3, 0, 'end'): 4533,
    ('phase', 40, 648, 0, 3, 0, 'late'): 4554,
    ('phase', 40, 648, 0, 3, 0, 'mid'): 3219,
    ('phase', 41, 0, 1, 0, 0, 'end'): 3892,
    ('phase', 41, 0, 2, 0, 0, 'end'): -3892,
    ('phase', 43, 648, 1, 0, 0, 'end'): 3045,
    ('phase', 43, 648, 1, 0, 0, 'late'): 4263,
    ('phase', 43, 648, 1, 0, 0, 'mid'): 4860,
    ('phase', 43, 648, 1, 0, 0, 'open'): 2708,
    ('phase', 43, 648, 2, 0, 0, 'end'): -3045,
    ('phase', 43, 648, 2, 0, 0, 'late'): -4263,
    ('phase', 43, 648, 2, 0, 0, 'mid'): -4860,
    ('phase', 43, 648, 2, 0, 0, 'open'): -2708,
    ('phase', 5, 1086, 3, 646, 0, 'late'): 847,
    ('phase', 5, 1086, 3, 646, 0, 'mid'): 1946,
    ('phase', 5, 1086, 3, 646, 0, 'open'): 317,
    ('phase', 5, 1086, 3, 860, 0, 'mid'): -1946,
    ('phase', 5, 1086, 3, 860, 0, 'open'): -619,
    ('phase', 7, 0, 3, 0, 0, 'end'): 1062,
    ('phase', 7, 0, 3, 0, 0, 'late'): -682,
    ('phase', 7, 0, 3, 0, 0, 'mid'): -1298,
    ('phase', 7, 0, 3, 0, 0, 'open'): -1546,
    ('phase', 7, 1097, 3, 104, 0, 'end'): -511,
    ('phase', 7, 1097, 3, 104, 0, 'late'): -336,
    ('phase', 7, 1097, 3, 104, 0, 'mid'): -336,
    ('phase', 7, 1097, 3, 112, 0, 'end'): -3045,
    ('phase', 7, 1097, 3, 112, 0, 'late'): 236,
    ('phase', 7, 1097, 3, 112, 0, 'mid'): 588,
    ('phase', 7, 1097, 3, 646, 0, 'end'): -2944,
    ('phase', 7, 1097, 3, 646, 0, 'late'): -3045,
    ('phase', 7, 1097, 3, 646, 0, 'mid'): 0,
    ('phase', 7, 1097, 3, 647, 0, 'end'): -2398,
    ('phase', 7, 1097, 3, 647, 0, 'late'): 0,
    ('phase', 7, 1097, 3, 647, 0, 'mid'): 511,
    ('phase', 7, 1097, 3, 648, 0, 'end'): -1299,
    ('phase', 7, 1097, 3, 648, 0, 'late'): -2708,
    ('phase', 7, 1097, 3, 648, 0, 'mid'): -511,
    ('phase', 7, 1097, 3, 7, 0, 'end'): -1341,
    ('phase', 7, 1097, 3, 7, 0, 'late'): -1048,
    ('phase', 7, 1097, 3, 7, 0, 'mid'): -188,
    ('phase', 7, 1097, 3, 860, 0, 'end'): -1099,
    ('phase', 7, 1097, 3, 860, 0, 'late'): -1735,
    ('phase', 7, 1097, 3, 860, 0, 'mid'): 336,
    ('phase', 7, 1122, 3, 1182, 0, 'late'): -2398,
    ('phase', 7, 1122, 3, 1182, 0, 'mid'): -847,
    ('phase', 7, 1122, 3, 1219, 0, 'mid'): 847,
    ('phase', 7, 1122, 3, 1219, 0, 'open'): 0,
    ('phase', 7, 1122, 3, 1227, 0, 'late'): 847,
    ('phase', 7, 1122, 3, 1227, 0, 'open'): 511,
    ('phase', 7, 1122, 3, 1231, 0, 'mid'): 847,
    ('phase', 7, 1152, 3, 104, 0, 'end'): -368,
    ('phase', 7, 1152, 3, 104, 0, 'late'): -167,
    ('phase', 7, 1152, 3, 104, 0, 'mid'): -1829,
    ('phase', 7, 1152, 3, 104, 0, 'open'): -2030,
    ('phase', 7, 1152, 3, 112, 0, 'end'): 762,
    ('phase', 7, 1152, 3, 112, 0, 'late'): -480,
    ('phase', 7, 1152, 3, 112, 0, 'mid'): -664,
    ('phase', 7, 1152, 3, 112, 0, 'open'): -1371,
    ('phase', 7, 1152, 3, 646, 0, 'end'): -336,
    ('phase', 7, 1152, 3, 646, 0, 'late'): -2565,
    ('phase', 7, 1152, 3, 646, 0, 'mid'): -1609,
    ('phase', 7, 1152, 3, 646, 0, 'open'): -1946,
    ('phase', 7, 1152, 3, 647, 0, 'end'): -1846,
    ('phase', 7, 1152, 3, 647, 0, 'late'): -1609,
    ('phase', 7, 1152, 3, 647, 0, 'mid'): -1997,
    ('phase', 7, 1152, 3, 647, 0, 'open'): -5209,
    ('phase', 7, 1152, 3, 860, 0, 'end'): -2833,
    ('phase', 7, 1152, 3, 860, 0, 'late'): -956,
    ('phase', 7, 1152, 3, 860, 0, 'mid'): -2398,
    ('phase', 7, 1152, 3, 860, 0, 'open'): -2565,
    ('phase', 7, 1219, 3, 1079, 0, 'end'): -2793,
    ('phase', 7, 1219, 3, 1079, 0, 'late'): -3076,
    ('phase', 7, 1219, 3, 1079, 0, 'mid'): -1968,
    ('phase', 7, 1219, 3, 1079, 0, 'open'): -3932,
    ('phase', 7, 1219, 3, 1080, 0, 'end'): 3045,
    ('phase', 7, 1219, 3, 1080, 0, 'late'): 251,
    ('phase', 7, 1219, 3, 1080, 0, 'mid'): 654,
    ('phase', 7, 1219, 3, 1080, 0, 'open'): -762,
    ('phase', 7, 1219, 3, 1086, 0, 'end'): -4317,
    ('phase', 7, 1219, 3, 1086, 0, 'late'): -3344,
    ('phase', 7, 1219, 3, 1086, 0, 'mid'): -5004,
    ('phase', 7, 1219, 3, 1086, 0, 'open'): -2434,
    ('phase', 7, 1219, 3, 1097, 0, 'end'): -1609,
    ('phase', 7, 1219, 3, 1097, 0, 'late'): -1546,
    ('phase', 7, 1219, 3, 1097, 0, 'mid'): -3646,
    ('phase', 7, 1219, 3, 1097, 0, 'open'): -3970,
    ('phase', 7, 1219, 3, 1122, 0, 'end'): -2565,
    ('phase', 7, 1219, 3, 1122, 0, 'late'): -2944,
    ('phase', 7, 1219, 3, 1122, 0, 'mid'): -3714,
    ('phase', 7, 1219, 3, 1122, 0, 'open'): -2833,
    ('phase', 7, 1219, 3, 1137, 0, 'end'): -2197,
    ('phase', 7, 1219, 3, 1137, 0, 'late'): -3434,
    ('phase', 7, 1219, 3, 1137, 0, 'mid'): -3714,
    ('phase', 7, 1219, 3, 1137, 0, 'open'): -2833,
    ('phase', 7, 1219, 3, 1152, 0, 'end'): -3714,
    ('phase', 7, 1219, 3, 1152, 0, 'late'): -2007,
    ('phase', 7, 1219, 3, 1152, 0, 'mid'): -2381,
    ('phase', 7, 1219, 3, 1152, 0, 'open'): -4234,
    ('phase', 7, 1219, 3, 1182, 0, 'end'): -2565,
    ('phase', 7, 1219, 3, 1182, 0, 'late'): -2241,
    ('phase', 7, 1219, 3, 1182, 0, 'mid'): -4234,
    ('phase', 7, 1219, 3, 1182, 0, 'open'): -3434,
    ('phase', 7, 1219, 3, 1219, 0, 'end'): -4078,
    ('phase', 7, 1219, 3, 1219, 0, 'late'): -3164,
    ('phase', 7, 1219, 3, 1219, 0, 'mid'): -4727,
    ('phase', 7, 1219, 3, 1219, 0, 'open'): -4007,
    ('phase', 7, 1219, 3, 1227, 0, 'end'): -2752,
    ('phase', 7, 1219, 3, 1227, 0, 'late'): -3271,
    ('phase', 7, 1219, 3, 1227, 0, 'mid'): -3878,
    ('phase', 7, 1219, 3, 1227, 0, 'open'): -2172,
    ('phase', 7, 1219, 3, 1231, 0, 'end'): -2708,
    ('phase', 7, 1219, 3, 1231, 0, 'late'): -3135,
    ('phase', 7, 1219, 3, 1231, 0, 'mid'): -3664,
    ('phase', 7, 1219, 3, 1231, 0, 'open'): -3045,
    ('phase', 7, 1219, 3, 1259, 0, 'end'): -4443,
    ('phase', 7, 1219, 3, 1259, 0, 'late'): -4595,
    ('phase', 7, 1219, 3, 1259, 0, 'mid'): -3219,
    ('phase', 7, 1219, 3, 1259, 0, 'open'): -2501,
    ('phase', 7, 1231, 3, 104, 0, 'end'): 201,
    ('phase', 7, 1231, 3, 104, 0, 'late'): -336,
    ('phase', 7, 1231, 3, 104, 0, 'mid'): -762,
    ('phase', 7, 1231, 3, 104, 0, 'open'): -336,
    ('phase', 7, 1231, 3, 112, 0, 'end'): 336,
    ('phase', 7, 1231, 3, 112, 0, 'late'): -201,
    ('phase', 7, 1231, 3, 112, 0, 'mid'): 143,
    ('phase', 7, 1231, 3, 112, 0, 'open'): -788,
    ('phase', 7, 1231, 3, 646, 0, 'late'): -1946,
    ('phase', 7, 1231, 3, 646, 0, 'mid'): -2565,
    ('phase', 7, 1231, 3, 646, 0, 'open'): -2565,
    ('phase', 7, 1231, 3, 647, 0, 'late'): -847,
    ('phase', 7, 1231, 3, 647, 0, 'mid'): 251,
    ('phase', 7, 1231, 3, 647, 0, 'open'): -1466,
    ('phase', 7, 1231, 3, 648, 0, 'late'): 1946,
    ('phase', 7, 1231, 3, 648, 0, 'mid'): 201,
    ('phase', 7, 1231, 3, 648, 0, 'open'): 0,
    ('phase', 7, 1231, 3, 860, 0, 'end'): -1466,
    ('phase', 7, 1231, 3, 860, 0, 'late'): -2197,
    ('phase', 7, 1231, 3, 860, 0, 'mid'): -2833,
    ('phase', 7, 1231, 3, 860, 0, 'open'): -847,
    ('phase', 7, 1259, 3, 646, 0, 'end'): -619,
    ('phase', 7, 1259, 3, 646, 0, 'late'): -578,
    ('phase', 7, 1259, 3, 646, 0, 'mid'): -1234,
    ('phase', 7, 1259, 3, 646, 0, 'open'): -1099,
    ('phase', 7, 1259, 3, 647, 0, 'end'): -268,
    ('phase', 7, 1259, 3, 647, 0, 'late'): -806,
    ('phase', 7, 1259, 3, 647, 0, 'mid'): -962,
    ('phase', 7, 1259, 3, 647, 0, 'open'): -1726,
    ('phase', 7, 1259, 3, 648, 0, 'end'): 956,
    ('phase', 7, 1259, 3, 648, 0, 'late'): 310,
    ('phase', 7, 1259, 3, 648, 0, 'mid'): -623,
    ('phase', 7, 1259, 3, 648, 0, 'open'): -2049,
    ('phase', 8, 1197, 3, 1182, 0, 'mid'): 2197,
    ('subject', 0, 0, 10, 112): 146,
    ('subject', 0, 0, 10, 1257): 588,
    ('subject', 0, 0, 10, 1259): -1053,
    ('subject', 0, 0, 12, 0): -3206,
    ('subject', 0, 0, 13, 934): -2223,
    ('subject', 0, 0, 13, 935): -5153,
    ('subject', 0, 0, 13, 936): -2979,
    ('subject', 0, 0, 13, 937): -1605,
    ('subject', 0, 0, 14, 0): -3087,
    ('subject', 0, 0, 7, 1079): -585,
    ('subject', 0, 0, 7, 1080): -847,
    ('subject', 0, 0, 7, 1086): -1393,
    ('subject', 0, 0, 7, 1097): -758,
    ('subject', 0, 0, 7, 112): 541,
    ('subject', 0, 0, 7, 1122): -788,
    ('subject', 0, 0, 7, 1137): -2037,
    ('subject', 0, 0, 7, 1152): -587,
    ('subject', 0, 0, 7, 1182): -2522,
    ('subject', 0, 0, 7, 1219): -2038,
    ('subject', 0, 0, 7, 1227): -1511,
    ('subject', 0, 0, 7, 1231): -1707,
    ('subject', 0, 0, 7, 1259): 73,
    ('subject', 0, 0, 7, 646): -275,
    ('subject', 0, 0, 7, 860): -2762,
    ('subject', 0, 0, 8, 7): -3500,
    ('subject', 0, 0, 9, 104): 426,
    ('subject', 0, 0, 9, 647): -991,
    ('subject', 0, 0, 9, 648): -1703,
    ('subject', 1, 0, 3, 112): -108,
    ('subject', 1, 0, 3, 646): 1242,
    ('subject', 1, 0, 3, 860): 1846,
    ('subject', 13, 112, 3, 104): -2989,
    ('subject', 13, 112, 3, 1071): -2398,
    ('subject', 13, 112, 3, 112): -524,
    ('subject', 13, 112, 3, 119): -3219,
    ('subject', 13, 112, 3, 120): 1946,
    ('subject', 13, 112, 3, 132): 1946,
    ('subject', 13, 112, 3, 140): -3045,
    ('subject', 13, 112, 3, 175): -2197,
    ('subject', 13, 112, 3, 235): 511,
    ('subject', 13, 112, 3, 272): 2197,
    ('subject', 13, 112, 3, 305): -2565,
    ('subject', 13, 112, 3, 341): 511,
    ('subject', 13, 112, 3, 342): -956,
    ('subject', 13, 112, 3, 343): 368,
    ('subject', 13, 112, 3, 344): -731,
    ('subject', 13, 112, 3, 345): 752,
    ('subject', 13, 112, 3, 380): -1099,
    ('subject', 13, 112, 3, 381): -2833,
    ('subject', 13, 112, 3, 400): 636,
    ('subject', 13, 112, 3, 401): -111,
    ('subject', 13, 112, 3, 414): -1006,
    ('subject', 13, 112, 3, 431): -3135,
    ('subject', 13, 112, 3, 434): -1582,
    ('subject', 13, 112, 3, 646): -1046,
    ('subject', 13, 112, 3, 647): -2903,
    ('subject', 13, 112, 3, 648): -2278,
    ('subject', 13, 112, 3, 673): 336,
    ('subject', 13, 112, 3, 675): -2398,
    ('subject', 13, 112, 3, 676): -2398,
    ('subject', 13, 112, 3, 678): -2565,
    ('subject', 13, 112, 3, 689): -2565,
    ('subject', 13, 112, 3, 741): 310,
    ('subject', 13, 112, 3, 742): -1099,
    ('subject', 13, 112, 3, 743): -3664,
    ('subject', 13, 112, 3, 756): -1063,
    ('subject', 13, 112, 3, 860): -563,
    ('subject', 15, 648, 3, 104): -2361,
    ('subject', 15, 648, 3, 1071): -1946,
    ('subject', 15, 648, 3, 112): -265,
    ('subject', 15, 648, 3, 119): -788,
    ('subject', 15, 648, 3, 140): -2708,
    ('subject', 15, 648, 3, 175): -1846,
    ('subject', 15, 648, 3, 184): 1099,
    ('subject', 15, 648, 3, 24): -2398,
    ('subject', 15, 648, 3, 272): 1946,
    ('subject', 15, 648, 3, 305): -2241,
    ('subject', 15, 648, 3, 341): -847,
    ('subject', 15, 648, 3, 343): 788,
    ('subject', 15, 648, 3, 345): 762,
    ('subject', 15, 648, 3, 379): -847,
    ('subject', 15, 648, 3, 380): -511,
    ('subject', 15, 648, 3, 400): -143,
    ('subject', 15, 648, 3, 401): -511,
    ('subject', 15, 648, 3, 414): -1609,
    ('subject', 15, 648, 3, 431): -2565,
    ('subject', 15, 648, 3, 434): -788,
    ('subject', 15, 648, 3, 646): -1026,
    ('subject', 15, 648, 3, 647): -1989,
    ('subject', 15, 648, 3, 648): -1665,
    ('subject', 15, 648, 3, 673): 1299,
    ('subject', 15, 648, 3, 675): -2708,
    ('subject', 15, 648, 3, 676): -2565,
    ('subject', 15, 648, 3, 677): -1946,
    ('subject', 15, 648, 3, 678): -1946,
    ('subject', 15, 648, 3, 741): 111,
    ('subject', 15, 648, 3, 742): -1170,
    ('subject', 15, 648, 3, 743): -2708,
    ('subject', 15, 648, 3, 756): 2197,
    ('subject', 15, 648, 3, 860): -351,
    ('subject', 16, 112, 3, 104): -1299,
    ('subject', 16, 112, 3, 112): -403,
    ('subject', 16, 112, 3, 646): 1758,
    ('subject', 16, 112, 3, 647): -1099,
    ('subject', 16, 112, 3, 648): -630,
    ('subject', 2, 0, 3, 112): 3219,
    ('subject', 2, 0, 3, 646): 2398,
    ('subject', 21, 7, 3, 646): -1022,
    ('subject', 21, 7, 3, 647): -990,
    ('subject', 21, 7, 3, 648): 11,
    ('subject', 22, 648, 3, 7): 249,
    ('subject', 3, 0, 3, 104): -2944,
    ('subject', 3, 0, 3, 112): -4244,
    ('subject', 3, 0, 3, 646): -3497,
    ('subject', 3, 0, 3, 647): -2979,
    ('subject', 3, 0, 3, 648): 1588,
    ('subject', 3, 0, 3, 860): -1946,
    ('subject', 3, 1182, 3, 104): -1299,
    ('subject', 3, 1182, 3, 112): -671,
    ('subject', 3, 1182, 3, 342): 511,
    ('subject', 3, 1182, 3, 344): -511,
    ('subject', 3, 1182, 3, 646): -588,
    ('subject', 3, 1182, 3, 647): -3135,
    ('subject', 3, 1182, 3, 648): 0,
    ('subject', 3, 1182, 3, 860): -2398,
    ('subject', 30, 0, 6, 104): 2833,
    ('subject', 30, 0, 6, 112): 2565,
    ('subject', 30, 0, 6, 646): 847,
    ('subject', 30, 0, 6, 647): 511,
    ('subject', 30, 0, 6, 648): 112,
    ('subject', 34, 0, 15, 0): 4443,
    ('subject', 37, 1079, 9, 648): 276,
    ('subject', 38, 0, 0, 0): -3219,
    ('subject', 38, 0, 0, 1): 310,
    ('subject', 38, 0, 0, 2): 1099,
    ('subject', 4, 0, 3, 104): -3664,
    ('subject', 4, 0, 3, 112): -5425,
    ('subject', 4, 0, 3, 646): -995,
    ('subject', 4, 0, 3, 647): -76,
    ('subject', 4, 0, 3, 648): 588,
    ('subject', 4, 0, 3, 860): -3850,
    ('subject', 40, 112, 0, 1): -5974,
    ('subject', 40, 112, 0, 2): -1515,
    ('subject', 40, 112, 0, 3): 5778,
    ('subject', 40, 646, 0, 1): -3367,
    ('subject', 40, 646, 0, 2): -1609,
    ('subject', 40, 646, 0, 3): 3219,
    ('subject', 40, 648, 0, 1): -5416,
    ('subject', 40, 648, 0, 2): -2644,
    ('subject', 40, 648, 0, 3): 5352,
    ('subject', 41, 0, 1, 0): 3892,
    ('subject', 41, 0, 2, 0): -3892,
    ('subject', 43, 648, 1, 0): 5451,
    ('subject', 43, 648, 2, 0): -5451,
    ('subject', 5, 1086, 3, 646): 401,
    ('subject', 5, 1086, 3, 860): -652,
    ('subject', 7, 0, 3, 0): -712,
    ('subject', 7, 1097, 3, 104): -435,
    ('subject', 7, 1097, 3, 112): -368,
    ('subject', 7, 1097, 3, 646): -1494,
    ('subject', 7, 1097, 3, 647): -547,
    ('subject', 7, 1097, 3, 648): -1758,
    ('subject', 7, 1097, 3, 7): -902,
    ('subject', 7, 1097, 3, 860): -969,
    ('subject', 7, 1122, 3, 1182): -1735,
    ('subject', 7, 1122, 3, 1219): 100,
    ('subject', 7, 1122, 3, 1227): 1099,
    ('subject', 7, 1122, 3, 1231): 588,
    ('subject', 7, 1152, 3, 104): -1552,
    ('subject', 7, 1152, 3, 112): -936,
    ('subject', 7, 1152, 3, 646): -1859,
    ('subject', 7, 1152, 3, 647): -2979,
    ('subject', 7, 1152, 3, 860): -2445,
    ('subject', 7, 1219, 3, 1079): -2708,
    ('subject', 7, 1219, 3, 1080): 581,
    ('subject', 7, 1219, 3, 1086): -3948,
    ('subject', 7, 1219, 3, 1097): -2520,
    ('subject', 7, 1219, 3, 1122): -4466,
    ('subject', 7, 1219, 3, 1137): -3629,
    ('subject', 7, 1219, 3, 1152): -2736,
    ('subject', 7, 1219, 3, 1182): -3264,
    ('subject', 7, 1219, 3, 1219): -4588,
    ('subject', 7, 1219, 3, 1227): -3141,
    ('subject', 7, 1219, 3, 1231): -4554,
    ('subject', 7, 1219, 3, 1259): -3708,
    ('subject', 7, 1231, 3, 104): -336,
    ('subject', 7, 1231, 3, 112): -114,
    ('subject', 7, 1231, 3, 646): -2398,
    ('subject', 7, 1231, 3, 647): -511,
    ('subject', 7, 1231, 3, 648): 511,
    ('subject', 7, 1231, 3, 860): -2152,
    ('subject', 7, 1259, 3, 646): -1021,
    ('subject', 7, 1259, 3, 647): -1176,
    ('subject', 7, 1259, 3, 648): -799,
    ('subject', 8, 1197, 3, 1097): -1946,
    ('subject', 8, 1197, 3, 1182): 251,
    ('subject', 8, 1197, 3, 1227): -1946,
    ('subject', 8, 1197, 3, 1231): 511,
    ('subject', 8, 1197, 3, 7): -1099,
    ('type', 0, 0, 10): -501,
    ('type', 0, 0, 12): -3206,
    ('type', 0, 0, 13): -1749,
    ('type', 0, 0, 14): -3087,
    ('type', 0, 0, 7): -1204,
    ('type', 0, 0, 8): -3500,
    ('type', 0, 0, 9): -1210,
    ('type', 1, 0, 3): 604,
    ('type', 13, 112, 3): -1169,
    ('type', 15, 648, 3): -919,
    ('type', 16, 112, 3): -464,
    ('type', 2, 0, 3): 3611,
    ('type', 21, 7, 3): -423,
    ('type', 22, 648, 3): 249,
    ('type', 3, 0, 3): -1257,
    ('type', 3, 1182, 3): -941,
    ('type', 30, 0, 6): 376,
    ('type', 34, 0, 15): 4443,
    ('type', 37, 1079, 9): 276,
    ('type', 38, 0, 0): -392,
    ('type', 4, 0, 3): -1220,
    ('type', 40, 104, 0): -511,
    ('type', 40, 112, 0): -598,
    ('type', 40, 646, 0): -603,
    ('type', 40, 647, 0): -511,
    ('type', 40, 648, 0): -659,
    ('type', 41, 0, 1): 3892,
    ('type', 41, 0, 2): -3892,
    ('type', 43, 648, 1): 5451,
    ('type', 43, 648, 2): -5451,
    ('type', 5, 1086, 3): 27,
    ('type', 7, 0, 3): -712,
    ('type', 7, 1097, 3): -912,
    ('type', 7, 1122, 3): 45,
    ('type', 7, 1152, 3): -1652,
    ('type', 7, 1219, 3): -2877,
    ('type', 7, 1231, 3): -594,
    ('type', 7, 1259, 3): -986,
    ('type', 8, 1197, 3): -394
}
# END REPLAY-TRAINED PRIORS
LEARNED_SCALE = 10
LEARNED_LEVEL_WEIGHTS = (30, 22, 18, 14, 10, 6)

# Archetype clues observed in supplied replays and prior benchmark agents.
ARCHETYPE_IDS = {
    "dragapult": {119, 120, 121},
    "marnie": {646, 647, 648},
    "alakazam": {741, 742, 743},
    "crustle": {344, 345, 756},
    "lucario": {673, 674, 675, 676, 677, 678},
    "rocket": {400, 401, 414, 431, 432, 433, 463, 473, 474, 891},
}

# Pre-evolution / engine targets that are especially valuable before they mature.
SETUP_TARGET_IDS = {
    119, 120,  # Dreepy, Drakloak
    344,       # Dwebble
    646, 647,  # Marnie line
    673, 674, 677,  # Lucario-family / supporting Fighting basics
    741, 742,  # Abra, Kadabra
    341, 342, 379, 380,  # Cynthia engines
    400, 401, 414,       # Rocket engines
}


def _norm(value: Any) -> str:
    # Card text uses “Pokémon”. Dropping non-ASCII characters directly turns it
    # into “pok mon”, which made damage-prevention checks miss Crustle and other
    # walls. Decompose accents first so capability detection remains generic.
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _card_data(cid: int):
    return CARD.get(cid)


def _card_name(cid: int) -> str:
    data = _card_data(cid)
    return data.name if data is not None else str(cid)


def _field(player) -> list[tuple[AreaType, int, Pokemon]]:
    out: list[tuple[AreaType, int, Pokemon]] = []
    for i, p in enumerate(player.active or []):
        if p is not None:
            out.append((AreaType.ACTIVE, i, p))
    for i, p in enumerate(player.bench or []):
        if p is not None:
            out.append((AreaType.BENCH, i, p))
    return out


def _get_card(obs: Observation, area, index: int, player_index: int):
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return (obs.select.deck or [])[index]
        if area == AreaType.HAND:
            return (player.hand or [])[index]
        if area == AreaType.DISCARD:
            return (player.discard or [])[index]
        if area == AreaType.ACTIVE:
            return (player.active or [])[index]
        if area == AreaType.BENCH:
            return (player.bench or [])[index]
        if area == AreaType.PRIZE:
            return (player.prize or [])[index]
        if area == AreaType.STADIUM:
            return (obs.current.stadium or [])[index]
        if area == AreaType.LOOKING:
            return (obs.current.looking or [])[index]
    except (IndexError, TypeError, AttributeError):
        return None
    return None


def _source_id(select) -> int:
    if getattr(select, "contextCard", None) is not None:
        return select.contextCard.id
    if getattr(select, "effect", None) is not None:
        return select.effect.id
    return 0


def _energy_count(p: Optional[Pokemon]) -> int:
    return len(p.energyCards or []) if p is not None else 0


def _is_basic_pokemon(p: Optional[Pokemon]) -> bool:
    if p is None:
        return False
    data = _card_data(p.id)
    return bool(getattr(data, "basic", False)) if data is not None else False


def _has_prism(p: Optional[Pokemon]) -> bool:
    return p is not None and any(e.id == PRISM_ENERGY for e in (p.energyCards or []))


def _dark_count(p: Optional[Pokemon]) -> int:
    if p is None:
        return 0
    count = sum(1 for e in (p.energyCards or []) if e.id == DARK_ENERGY)
    # Prism Energy supplies every type only while attached to a Basic Pokémon.
    if _is_basic_pokemon(p):
        count += sum(1 for e in (p.energyCards or []) if e.id == PRISM_ENERGY)
    return count


def _psychic_count(p: Optional[Pokemon]) -> int:
    if p is None:
        return 0
    count = sum(1 for e in (p.energyCards or []) if e.id == PSYCHIC_ENERGY)
    if _is_basic_pokemon(p):
        count += sum(1 for e in (p.energyCards or []) if e.id == PRISM_ENERGY)
    return count


def _munkidori_ready(p: Optional[Pokemon], extra_card_id: Optional[int] = None) -> bool:
    if p is None or p.id != MUNKIDORI:
        return False
    total = _energy_count(p)
    psychic = _psychic_count(p)
    if extra_card_id is not None:
        total += 1
        if extra_card_id in {PSYCHIC_ENERGY, PRISM_ENERGY}:
            psychic += 1
    return total >= 2 and psychic >= 1


def _fire_count(p: Optional[Pokemon]) -> int:
    if p is None:
        return 0
    return sum(1 for e in (p.energyCards or []) if e.id == FIRE_ENERGY)


def _chi_yu_ready(p: Optional[Pokemon]) -> bool:
    # Ground Melter requires one Fire plus one additional Energy.
    return p is not None and p.id == CHI_YU and _fire_count(p) >= 1 and _energy_count(p) >= 2


def _damage_on(p: Optional[Pokemon]) -> int:
    return max(0, p.maxHp - p.hp) if p is not None else 0


def _remaining_hp(p: Optional[Pokemon]) -> int:
    return max(0, p.hp) if p is not None else 0


def _prize_value(p: Optional[Pokemon]) -> int:
    if p is None:
        return 0
    data = _card_data(p.id)
    if data is None:
        return 1
    value = 3 if getattr(data, "megaEx", False) else 2 if getattr(data, "ex", False) else 1
    # Legacy Energy reduces prizes by one in CABT.
    if any(getattr(e, "id", -1) == 12 for e in (p.energyCards or [])):
        value -= 1
    return max(0, value)


def _retreat_cost(p: Optional[Pokemon]) -> int:
    data = _card_data(p.id) if p is not None else None
    return int(getattr(data, "retreatCost", 1) or 0) if data is not None else 1


def _has_ability(cid: int) -> bool:
    data = _card_data(cid)
    return bool(getattr(data, "skills", [])) if data is not None else cid in {GRIMMSNARL_EX, MUNKIDORI, FROSLASS}


def _min_attack_cost(cid: int) -> int:
    data = _card_data(cid)
    costs: list[int] = []
    for aid in (getattr(data, "attacks", []) if data is not None else []):
        attack = ATTACK.get(aid)
        if attack is not None:
            costs.append(len(attack.energies or []))
    return min(costs) if costs else 1


def _max_attack_damage(cid: int) -> int:
    data = _card_data(cid)
    values: list[int] = []
    for aid in (getattr(data, "attacks", []) if data is not None else []):
        attack = ATTACK.get(aid)
        if attack is not None:
            values.append(int(attack.damage or 0))
    return max(values) if values else 0


def _ready_to_attack(p: Optional[Pokemon], extra_energy: int = 0) -> bool:
    if p is None:
        return False
    # Munkidori specifically needs one Psychic plus one Colorless. Two Darkness
    # Energy do not satisfy Mind Bend; Prism + any second Energy does.
    if p.id == MUNKIDORI:
        if extra_energy <= 0:
            return _munkidori_ready(p)
        # Generic forecasts use one possible extra attachment. Prefer Prism when
        # Psychic is missing, otherwise any Energy completes the Colorless cost.
        hypothetical = PRISM_ENERGY if _psychic_count(p) == 0 else DARK_ENERGY
        return _munkidori_ready(p, hypothetical)
    return _energy_count(p) + extra_energy >= _min_attack_cost(p.id)


def _stage_value(cid: int) -> int:
    data = _card_data(cid)
    if data is None:
        return 0
    return 2 if getattr(data, "stage2", False) else 1 if getattr(data, "stage1", False) else 0


def _threat(p: Optional[Pokemon], damage_budget: int = 0, active: bool = False) -> int:
    if p is None:
        return NEG
    data = _card_data(p.id)
    score = _prize_value(p) * 3000 + _energy_count(p) * 650 + _stage_value(p.id) * 800
    score += 700 if _has_ability(p.id) else 0
    score += 900 if data is not None and getattr(data, "ex", False) else 0
    score += 2800 if p.id in SETUP_TARGET_IDS else 0
    score += min(2200, _damage_on(p) * 8)
    if damage_budget > 0:
        if p.hp <= damage_budget:
            score += 15000 + _prize_value(p) * 5000
        else:
            score += max(0, 1800 - (p.hp - damage_budget) * 8)
    if active:
        score += 300
    return score


def _stadium_id(state) -> int:
    return state.stadium[0].id if state.stadium else 0


def _safe_draw(player) -> bool:
    return player.deckCount > 7


def _attack_name(aid: Optional[int]) -> str:
    attack = ATTACK.get(aid) if aid is not None else None
    return _norm(attack.name) if attack is not None else ""


def _attack_damage_blocked(attacker: Optional[Pokemon], target: Optional[Pokemon]) -> bool:
    """Recognize common printed attack-damage prevention from live card text.

    This is intentionally capability based rather than matchup-name based, so a
    new deck using the same kind of wall cannot make the agent hallucinate a KO.
    Non-damage effects such as Shadow Bullet's 30 Bench damage remain valuable.
    """
    if attacker is None or target is None:
        return False
    attacker_data = _card_data(attacker.id)
    target_data = _card_data(target.id)
    if attacker_data is None or target_data is None:
        return False
    text = " ".join(_norm(getattr(skill, "text", "")) for skill in (getattr(target_data, "skills", []) or []))
    if "prevent all damage" not in text:
        return False
    if "pokemon ex" in text and bool(getattr(attacker_data, "ex", False)):
        return True
    if "pokemon that have an ability" in text and _has_ability(attacker.id):
        return True
    return False


def _has_rule_box(pokemon: Optional[Pokemon]) -> bool:
    if pokemon is None:
        return False
    data = _card_data(pokemon.id)
    if data is None:
        return False
    return any(
        bool(getattr(data, field, False))
        for field in ("ex", "megaEx", "v", "vmax", "vstar", "gx")
    )


def _bench_attack_damage_blocked(attacker: Optional[Pokemon], target: Optional[Pokemon], opponent) -> bool:
    """Recognize effects such as Shaymin's Flower Curtain.

    Shadow Bullet places attack damage, not damage counters, so these effects
    can stop its Bench damage even while the Active target is still damaged.
    """
    if attacker is None or target is None:
        return False
    if _attack_damage_blocked(attacker, target):
        return True
    if _has_rule_box(target):
        return False
    for _, _, pokemon in _field(opponent):
        data = _card_data(pokemon.id)
        text = " ".join(
            _norm(getattr(skill, "text", ""))
            for skill in (getattr(data, "skills", []) if data is not None else [])
        )
        if "prevent all damage done to your benched pokemon that don t have a rule box" in text:
            return True
    return False


def _learned_phase(obs: Observation) -> str:
    mine = obs.current.players[obs.current.yourIndex]
    prizes = len(mine.prize or [])
    if prizes <= 2:
        return "end"
    if obs.current.turn <= 3:
        return "open"
    if obs.current.turn <= 7:
        return "mid"
    return "late"


def _learned_option_subject(obs: Observation, option) -> tuple[int, int]:
    typ = option.type
    if typ == OptionType.ATTACK:
        return int(option.attackId or 0), 0
    if typ in {OptionType.NUMBER, OptionType.YES, OptionType.NO, OptionType.END,
               OptionType.SKILL, OptionType.SPECIAL_CONDITION}:
        return int(getattr(option, "number", 0) or 0), 0
    area = option.area
    if typ == OptionType.PLAY and area is None:
        area = AreaType.HAND
    card = _get_card(obs, area, option.index, option.playerIndex if option.playerIndex is not None else obs.current.yourIndex)
    target = _get_card(obs, option.inPlayArea, option.inPlayIndex, obs.current.yourIndex)
    return int(getattr(card, "id", 0) or 0), int(getattr(target, "id", 0) or 0)


def _residual_card_signature(cards) -> tuple:
    return tuple(sorted(int(c.id) for c in (cards or [])))


def _residual_count_signature(cards) -> tuple:
    return tuple(sorted(Counter(int(c.id) for c in (cards or [])).items()))


def _residual_field_signature(player) -> tuple:
    return tuple(sorted(
        (int(p.id or 0), _energy_count(p), _damage_on(p), int(getattr(p, "tool", 0) or 0))
        for _, _, p in _field(player)
    ))


def _residual_option_signature(obs: Observation) -> tuple:
    values = []
    for option in obs.select.option:
        subject, target = _learned_option_subject(obs, option)
        values.append((
            int(option.type), int(subject), int(target),
            int(getattr(option, "area", 0) or 0),
            int(getattr(option, "playerIndex", -1) if getattr(option, "playerIndex", None) is not None else -1),
        ))
    return tuple(values)


def _residual_state_fingerprint(obs: Observation, archetype: str, phase: str) -> str:
    cur = obs.current
    yi = int(cur.yourIndex or 0)
    mine = cur.players[yi]
    opp = cur.players[1 - yi]
    select = obs.select
    payload = (
        archetype, phase, int(cur.turn or 0), int(cur.turnActionCount or 0),
        bool(cur.energyAttached), bool(cur.retreated), bool(cur.stadiumPlayed), bool(cur.supporterPlayed),
        len(mine.prize or []), len(opp.prize or []), int(mine.deckCount or 0), int(opp.deckCount or 0),
        _residual_card_signature(mine.hand), _residual_count_signature(mine.discard),
        _residual_field_signature(mine), _residual_field_signature(opp), int(_stadium_id(cur)),
        int(select.context), int(select.type), int(_source_id(select)),
        int(select.minCount or 0), int(select.maxCount or 0), _residual_option_signature(obs),
    )
    return hashlib.blake2b(repr(payload).encode(), digest_size=10).hexdigest()


def _residual_bonus(obs: Observation, option, archetype: str) -> int:
    if RESIDUAL_SCALE <= 0 or archetype not in _RESIDUAL_ENABLED:
        return 0
    policy = _RESIDUAL_POLICIES.get(archetype) or {}
    entries = policy.get("entries", policy) if isinstance(policy, dict) else {}
    if not entries:
        return 0
    select = obs.select
    min_count = int(select.minCount or 0)
    max_count = int(select.maxCount or min_count)
    # Never learn mechanical multi-card combinations, numeric choices, or
    # energy-payment combinations.  v19 handles these deterministically.
    if max_count != 1 or min_count > 1 or int(select.type) in {4, 8}:
        return 0
    phase = _learned_phase(obs)
    context = int(select.context)
    source = _source_id(select)
    typ = int(option.type)
    subject, target = _learned_option_subject(obs, option)
    fp = _residual_state_fingerprint(obs, archetype, phase)
    key = "|".join(map(str, ("state", fp, context, source, typ, subject, target)))
    entry = entries.get(key)
    if not isinstance(entry, dict):
        return 0
    if int(entry.get("support", 0)) < 1 or float(entry.get("confidence", 0.0)) < 0.75:
        return 0
    return int(max(-200, min(200, float(entry.get("score", 0)) * RESIDUAL_SCALE)))


def _cf_coarse_key(obs: Observation, option, archetype: str) -> str:
    cur = obs.current
    yi = int(cur.yourIndex or 0)
    mine = cur.players[yi]
    opp = cur.players[1 - yi]
    phase = _learned_phase(obs)
    subject, target = _learned_option_subject(obs, option)
    my_field = Counter(p.id for _, _, p in _field(mine))
    opp_field = Counter(p.id for _, _, p in _field(opp))
    active = mine.active[0] if mine.active else None
    opp_active = opp.active[0] if opp.active else None
    payload = (
        "coarse", archetype, phase, int(obs.select.context), int(_source_id(obs.select)),
        int(option.type), int(subject), int(target),
        min(3, int(cur.turn or 0) // 3),
        min(3, len(mine.prize or []) // 2), min(3, len(opp.prize or []) // 2),
        min(2, my_field[GRIMMSNARL_EX]), min(3, my_field[MUNKIDORI]), min(2, my_field[FROSLASS]),
        min(2, opp_field[GRIMMSNARL_EX]),
        int(bool(active and _ready_to_attack(active))), int(bool(opp_active)),
        min(3, sum(_damage_on(p) for _, _, p in _field(mine)) // 30),
        int(bool(cur.energyAttached)), int(bool(cur.supporterPlayed)),
    )
    return "|".join(map(str, payload))


def _counterfactual_bonus(obs: Observation, option, archetype: str) -> int:
    global _CF_HIT_COUNT
    if CF_SCALE <= 0 or archetype not in _CF_ENABLED:
        return 0
    select = obs.select
    min_count = int(select.minCount or 0)
    max_count = int(select.maxCount or min_count)
    if max_count != 1 or min_count > 1 or int(select.type) in {4, 8}:
        return 0
    policy = _CF_POLICIES.get(archetype) or {}
    exact = policy.get("exact_entries", {})
    coarse = policy.get("coarse_entries", {})
    phase = _learned_phase(obs)
    context = int(select.context)
    source = _source_id(select)
    typ = int(option.type)
    subject, target = _learned_option_subject(obs, option)
    fp = _residual_state_fingerprint(obs, archetype, phase)
    key = "|".join(map(str, ("state", fp, context, source, typ, subject, target)))
    total = 0.0
    e = exact.get(key)
    if isinstance(e, dict) and int(e.get("support", 0)) >= 1 and float(e.get("confidence", 0.0)) >= 0.80:
        total += max(-70.0, min(70.0, float(e.get("score", 0)) * 0.65))
    c = coarse.get(_cf_coarse_key(obs, option, archetype))
    if isinstance(c, dict) and int(c.get("support", 0)) >= 3 and float(c.get("confidence", 0.0)) >= 0.62:
        total += max(-35.0, min(35.0, float(c.get("score", 0)) * 0.60))
    if total:
        _CF_HIT_COUNT += 1
    return int(max(-90, min(90, total * CF_SCALE)))


def _learned_bonus(obs: Observation, option, archetype: str) -> int:
    context = int(obs.select.context)
    source = _source_id(obs.select)
    typ = int(option.type)
    subject, target = _learned_option_subject(obs, option)
    phase = _learned_phase(obs)
    keys = (
        ("exact", context, source, typ, subject, target, archetype, phase),
        ("arch", context, source, typ, subject, target, archetype),
        ("phase", context, source, typ, subject, target, phase),
        ("base", context, source, typ, subject, target),
        ("subject", context, source, typ, subject),
        ("type", context, source, typ),
    )
    weighted = 0
    total_weight = 0
    correction_weighted = 0
    correction_total = 0
    for key, weight in zip(keys, LEARNED_LEVEL_WEIGHTS):
        value = LEARNED_PRIOR.get(key)
        if value is not None:
            weighted += value * weight
            total_weight += weight
        correction = _REPLAY_CORRECTION.get("|".join(map(str, key)))
        if correction is not None:
            correction_weighted += int(correction) * weight
            correction_total += weight
    # Only two old Dragapult wins were available, and arena evaluation showed
    # their sparse imitation prior overrode the stronger general tactical rules.
    # Keep replay learning for well-supported matchups and fall back to the
    # rule policy once Dragapult is publicly identified.
    scale = LEARNED_SCALE
    base_bonus = int(weighted * scale / total_weight) if total_weight else 0
    correction_bonus = (
        int(correction_weighted * REPLAY_CORRECTION_SCALE / correction_total)
        if correction_total else 0
    )
    return base_bonus + correction_bonus


@dataclass
class Forecast:
    active_ready: bool
    bench_ready: bool
    one_attach_attacker: bool
    evolution_threat: bool
    retreat_likely: bool
    boss_risk: bool
    disruption_risk: bool
    estimated_damage: int


@dataclass
class Plan:
    phase: str
    archetype: str
    forecast: Forecast
    desired_grimms: int
    desired_munkis: int
    desired_froslass: int
    lock_target_serial: Optional[int]
    lock_value: int
    boss_target_serial: Optional[int]
    energy_target_serial: Optional[int]
    next_goal: str


_MEMORY = {
    "episode": 0,
    "last_turn": -1,
    "archetype": "unknown",
    "revealed": set(),
    "phase": "OPENING",
    "last_prizes": 6,
    "recent_ko": False,
    "lock_target": None,
    "planned_next": "",
    "setup_active_id": None,
    # v33: remember a retreat until the replacement Active either attacks or
    # the turn changes.  This prevents a legal retreat line from ending idle.
    "retreat_followup_turn": -1,
}


def _reset_memory() -> None:
    _MEMORY["episode"] += 1
    _MEMORY["last_turn"] = -1
    _MEMORY["archetype"] = "unknown"
    _MEMORY["revealed"] = set()
    _MEMORY["phase"] = "OPENING"
    _MEMORY["last_prizes"] = 6
    _MEMORY["recent_ko"] = False
    _MEMORY["lock_target"] = None
    _MEMORY["planned_next"] = ""
    _MEMORY["setup_active_id"] = None
    _MEMORY["retreat_followup_turn"] = -1


def _recognize_from_ids(ids: Iterable[int]) -> str:
    ids = set(ids)
    # Explicit IDs first.
    order = ("crustle", "dragapult", "lucario", "alakazam", "rocket", "marnie")
    for archetype in order:
        if ids & ARCHETYPE_IDS[archetype]:
            return archetype
    # Dynamic names extend recognition to replay archetypes not hard-coded.
    names = " | ".join(_norm(_card_name(cid)) for cid in ids)
    if "dragapult" in names or "dreepy" in names or "drakloak" in names:
        return "dragapult"
    if "grimmsnarl" in names or "marnie s impidimp" in names:
        return "marnie"
    if "alakazam" in names or "kadabra" in names:
        return "alakazam"
    if "crustle" in names or "dwebble" in names:
        return "crustle"
    if "lucario" in names or "riolu" in names:
        return "lucario"
    if "team rocket" in names:
        return "rocket"
    if "cynthia s garchomp" in names or "cynthia s gabite" in names:
        return "cynthia"
    if "archaludon" in names or "duraludon" in names:
        return "archaludon"
    if "dipplin" in names or "festival grounds" in names:
        return "festival"
    return "unknown"


def _update_memory(obs: Observation) -> str:
    state = obs.current
    me = state.yourIndex
    opponent = state.players[1 - me]
    opponent_prizes = getattr(opponent, "prizeCount", None)
    if opponent_prizes is None:
        visible_prizes = getattr(opponent, "prize", None)
        opponent_prizes = len(visible_prizes or []) if visible_prizes is not None else _MEMORY["last_prizes"]
    opponent_prizes = int(opponent_prizes)
    if state.turn != _MEMORY["last_turn"]:
        _MEMORY["recent_ko"] = _MEMORY["last_turn"] >= 0 and opponent_prizes < _MEMORY["last_prizes"]
        _MEMORY["last_turn"] = state.turn
        _MEMORY["last_prizes"] = opponent_prizes
        # A retreat follow-up is valid only within the same turn.
        _MEMORY["retreat_followup_turn"] = -1
    revealed = set(_MEMORY["revealed"])
    for _, _, p in _field(opponent):
        revealed.add(p.id)
        for pre in (p.preEvolution or []):
            revealed.add(pre.id)
    for c in (opponent.discard or []):
        revealed.add(c.id)
    for log in (obs.logs or []):
        if getattr(log, "playerIndex", None) == 1 - me:
            for key in ("cardId", "cardIdTarget", "cardIdActive", "cardIdBench", "cardIdBefore", "cardIdAfter"):
                cid = getattr(log, key, None)
                if cid is not None:
                    revealed.add(cid)
    _MEMORY["revealed"] = revealed
    current = _recognize_from_ids(revealed)
    if current != "unknown":
        _MEMORY["archetype"] = current
    return str(_MEMORY["archetype"])


def _forecast(opponent, archetype: str) -> Forecast:
    active = opponent.active[0] if opponent.active else None
    bench = [p for p in (opponent.bench or []) if p is not None]
    active_ready = _ready_to_attack(active)
    bench_ready = any(_ready_to_attack(p) for p in bench)
    one_attach = _ready_to_attack(active, 1) or any(_ready_to_attack(p, 1) for p in bench)
    evolution_threat = any(
        _stage_value(p.id) < 2 and (p.id in SETUP_TARGET_IDS or _has_ability(p.id))
        for p in ([active] if active is not None else []) + bench
    )
    retreat_likely = bool(active is not None and bench_ready and (_retreat_cost(active) <= _energy_count(active) or archetype in {"lucario", "crustle", "rocket"}))
    boss_risk = opponent.handCount >= 5 and len(bench) >= 1
    disruption_risk = archetype in {"alakazam", "marnie", "dragapult", "rocket"} and opponent.handCount >= 4
    estimated = max((_max_attack_damage(p.id) for p in ([active] if active is not None else []) + bench if _ready_to_attack(p, 1)), default=0)
    return Forecast(active_ready, bench_ready, one_attach, evolution_threat, retreat_likely, boss_risk, disruption_risk, estimated)


def _lock_target_value(p: Pokemon, opponent, forecast: Forecast) -> int:
    data = _card_data(p.id)
    if p is None or _retreat_cost(p) <= 0:
        return NEG
    # Do not lock a Pokémon that is already a credible attacker.
    if _ready_to_attack(p) and _max_attack_damage(p.id) >= 90:
        return NEG
    score = 3000 + _retreat_cost(p) * 1800
    score += 3500 if p.id in SETUP_TARGET_IDS else 0
    score += 1800 if _has_ability(p.id) else 0
    score += 1000 if data is not None and getattr(data, "stage1", False) else 0
    score += 1300 if _energy_count(p) == 0 else 500 if _energy_count(p) == 1 else -1500
    score += 1800 if opponent.handCount <= 4 else -800
    score += 2000 if not forecast.bench_ready else -3500
    score += 1800 if _stadium_id_dummy() == SPIKEMUTH_GYM else 0
    return score


def _stadium_id_dummy() -> int:
    # Replaced by the live-state bonus in _make_plan; kept pure for target scoring.
    return 0


def _make_plan(obs: Observation, archetype: str) -> Plan:
    state = obs.current
    me = state.yourIndex
    mine = state.players[me]
    opponent = state.players[1 - me]
    my_field = _field(mine)
    op_field = _field(opponent)
    field = Counter(p.id for _, _, p in my_field)
    forecast = _forecast(opponent, archetype)
    active = mine.active[0] if mine.active else None
    op_active = opponent.active[0] if opponent.active else None
    ready_grimms = [p for _, _, p in my_field if p.id == GRIMMSNARL_EX and _dark_count(p) >= 2]
    charged_munkis = [p for _, _, p in my_field if p.id == MUNKIDORI and _dark_count(p) >= 1]
    damaged_pool = sum(_damage_on(p) for _, _, p in my_field)

    desired_grimms = 2
    if ready_grimms and archetype in {"dragapult", "marnie", "lucario", "archaludon", "cynthia"}:
        desired_grimms = 3
    # Winning top-table replays establish two or three Munkidori early and use
    # manual attachments on them; Punk Up supplies the Marnie's evolution line.
    desired_munkis = 2
    if archetype == "marnie" or (ready_grimms and archetype != "crustle"):
        desired_munkis = 3
    desired_froslass = 0
    opponent_abilities = sum(1 for _, _, p in op_field if _has_ability(p.id))
    if archetype in {"alakazam", "rocket", "cynthia"}:
        desired_froslass = 1
        if opponent_abilities >= 3:
            desired_froslass = 2
    elif archetype in {"dragapult", "lucario", "unknown"} and opponent_abilities >= 2:
        desired_froslass = 1
    elif archetype == "archaludon":
        # Distilled from replay 89291844: reserve one Froslass line before a third Munkidori.
        desired_froslass = 2 if opponent_abilities >= 3 else 1
        desired_munkis = 3 if field[FROSLASS] >= desired_froslass else min(desired_munkis, 2)
    # Crustle cannot be damaged by Grimmsnarl ex. Two Froslass plus charged
    # Munkidori are the replay-proven fixed-list counter route. Do not spend
    # the last Bench slot on a third Munkidori before both Froslass exist.
    if archetype == "crustle" and (344 in _MEMORY["revealed"] or 345 in _MEMORY["revealed"]):
        desired_froslass = 2
        desired_munkis = 3 if field[FROSLASS] >= desired_froslass else 2
    # The new Alakazam loss replay reached one Grimmsnarl but filled the Bench
    # with three Munkidori and Froslass before a second attacker existed.
    # Powerful Hand then took consecutive one-hit KOs. Keep this matchup to one
    # Froslass and two Munkidori until two Grimmsnarl are actually in play.
    if archetype == "alakazam":
        # Keep two Bench slots available for the second Grimmsnarl line. The
        # hammer-control variant is identified only from cards the opponent
        # has actually revealed, so the replay-focused Alakazam plan remains
        # unchanged until there is positive evidence.
        hammer_control = bool({ALAKAZAM_ALT, CRUSHING_HAMMER} & set(_MEMORY["revealed"]))
        desired_froslass = 0 if hammer_control else 1
        desired_munkis = 3 if field[GRIMMSNARL_EX] >= 2 else 1

    # v27 loss-replay corrections.  These are deliberately matchup-gated.
    # The opponent archive called ``crustle`` in the local test set actually
    # uses the same Duraludon/Archaludon/Cinderace deck as ``archaludon`` and
    # is detected from revealed cards as the same archetype.
    if archetype == "archaludon":
        # Wins establish two attacker lines before over-expanding support, then
        # retain one Froslass and two charged Munkidori as the control engine.
        desired_grimms = 3
        desired_froslass = 1
        desired_munkis = 2 if field[GRIMMSNARL_EX] < 2 else 3
    elif archetype == "lucario":
        # Preserve the existing attachment order; only strengthen the second
        # attacker search. Earlier manual-attachment forcing regressed badly.
        desired_grimms = 3
        desired_munkis = max(desired_munkis, 2)

    lock_target_serial = None
    lock_value = NEG
    for _, _, p in op_field:
        value = _lock_target_value(p, opponent, forecast)
        if _stadium_id(state) == SPIKEMUTH_GYM:
            value += 1500
        if value > lock_value:
            lock_value = value
            lock_target_serial = p.serial

    boss_target_serial = None
    best_boss = NEG
    movable_counters = min(damaged_pool, len(charged_munkis) * 30)
    for area, _, p in op_field:
        if area != AreaType.BENCH:
            continue
        value = _threat(p, 180, False)
        if p.id in SETUP_TARGET_IDS:
            value += 2500
        if archetype == "crustle" and p.id == 345 and 0 < p.hp <= movable_counters:
            value += 30000 + (movable_counters - p.hp) * 30
        if archetype == "alakazam" and len(mine.prize or []) > 2:
            if p.id == ALAKAZAM:
                value += 14000
            elif p.id == KADABRA:
                value += 10000
            elif p.id == ABRA:
                value += 6500
        if value > best_boss:
            best_boss = value
            boss_target_serial = p.serial

    # Energy plan: complete the first attacker before speculative lock pieces.
    energy_target_serial = None
    yveltal = next((p for _, _, p in my_field if p.id == YVELTAL), None)
    marnie_candidates = sorted(
        (p for _, _, p in my_field if p.id in MARNIE_LINE and _dark_count(p) < 2),
        key=lambda p: (p.id != GRIMMSNARL_EX, -_stage_value(p.id), _dark_count(p)),
    )
    uncharged_munki = next((p for _, _, p in my_field if p.id == MUNKIDORI and _dark_count(p) == 0), None)
    if uncharged_munki is not None:
        energy_target_serial = uncharged_munki.serial
    elif not ready_grimms and marnie_candidates:
        energy_target_serial = marnie_candidates[0].serial
    elif yveltal is not None and lock_value >= 8500 and not forecast.active_ready and not forecast.bench_ready and _dark_count(yveltal) < max(1, _min_attack_cost(YVELTAL)):
        energy_target_serial = yveltal.serial
    elif marnie_candidates:
        energy_target_serial = marnie_candidates[0].serial

    my_prizes = len(mine.prize or [])
    if my_prizes <= 2:
        phase = "ENDGAME"
        goal = "convert exact prizes with Boss, Shadow Bullet, and Adrena-Brain"
    elif archetype == "crustle" and ({344, 345} & set(_MEMORY["revealed"])):
        phase = "COUNTER_CONTROL"
        goal = "establish two Froslass and charged Munkidori; keep attacking to place Bench damage"
    elif (lock_value >= 9000 and yveltal is not None and not forecast.active_ready and not forecast.bench_ready
          and (ready_grimms or archetype == "crustle")
          and (_dark_count(yveltal) >= _min_attack_cost(YVELTAL) or not state.energyAttached)):
        phase = "YVELTAL_LOCK"
        goal = "trap a genuinely passive target after the primary attacker is secured"
    elif state.turn <= 2 or field[IMPIDIMP] + field[MORGREM] + field[GRIMMSNARL_EX] == 0:
        phase = "OPENING"
        goal = "establish two Impidimp, one Munkidori, and a searchable evolution route"
    elif not ready_grimms:
        phase = "EVOLUTION"
        goal = "evolve the first Grimmsnarl ex and activate Punk Up"
    elif archetype == "alakazam" and field[GRIMMSNARL_EX] < 2:
        phase = "EVOLUTION"
        goal = "reserve the Bench and establish a second Grimmsnarl before Powerful Hand scales"
    elif len(ready_grimms) < 2 and not any(_ready_to_attack(p) for _, _, p in my_field if p.id != GRIMMSNARL_EX):
        phase = "CHARGE"
        goal = "prepare a second attacker without consuming Munkidori/Yveltal manual energy"
    elif len(my_field) <= 2 or (field[GRIMMSNARL_EX] == 0 and state.turn >= 5):
        phase = "RECOVERY"
        goal = "recover the evolution line or energy with Night Stretcher"
    elif archetype == "marnie":
        phase = "MIRROR_CONTROL"
        goal = "remove opposing Morgrem/Munkidori and keep two attackers staggered"
    else:
        phase = "PRESSURE"
        goal = "attack while engineering a second knockout with bench 30 and moved counters"

    _MEMORY["phase"] = phase
    _MEMORY["lock_target"] = lock_target_serial
    _MEMORY["planned_next"] = goal
    return Plan(phase, archetype, forecast, desired_grimms, desired_munkis, desired_froslass,
                lock_target_serial, lock_value, boss_target_serial, energy_target_serial, goal)


def _choose(scores: list[int], select, forced_max: Optional[int] = None) -> list[int]:
    if not scores:
        return []
    max_count = min(select.maxCount, forced_max) if forced_max is not None else select.maxCount
    max_count = max(select.minCount, max_count)
    ranked = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)
    out: list[int] = []
    optional_contexts = {
        SelectContext.SETUP_BENCH_POKEMON,
        SelectContext.TO_BENCH,
        SelectContext.TO_HAND,
        SelectContext.ATTACH_FROM,
        SelectContext.ATTACH_TO,
        SelectContext.DISCARD,
    }
    for i in ranked:
        if len(out) >= max_count:
            break
        if scores[i] >= 0 or len(out) < select.minCount or select.context not in optional_contexts:
            out.append(i)
    for i in ranked:
        if len(out) >= select.minCount:
            break
        if i not in out:
            out.append(i)
    return out[:max_count]


def _discard_score(cid: int, hand: Counter, field: Counter, plan: Plan, state) -> int:
    # Higher score = more expendable.
    data = _card_data(cid)
    line_count = field[IMPIDIMP] + field[MORGREM] + field[GRIMMSNARL_EX]
    if cid == SECRET_BOX:
        return -30000
    if cid == DARK_ENERGY:
        return -18000 if plan.phase in {"EVOLUTION", "CHARGE", "YVELTAL_LOCK"} else -9000
    if cid == PRISM_ENERGY:
        # The single Prism is the only route to Munkidori's Psychic attack. Keep
        # it unless the Munkidori package is no longer relevant.
        if field[MUNKIDORI] > 0 or plan.desired_munkis > 0:
            return -22000
        return -7000
    if cid == FIRE_ENERGY:
        return -21000 if plan.phase == "WALL_BREAK" else 6500
    if cid == CHI_YU:
        return -23000 if plan.archetype == "crustle" else 7500
    if cid == UNFAIR_STAMP:
        return -26000
    if cid == GRIMMSNARL_EX and hand[cid] <= 1:
        return -17000
    if cid == RARE_CANDY and hand[cid] <= 1 and field[IMPIDIMP] > 0:
        return -16000
    if cid == IMPIDIMP and field[IMPIDIMP] + field[MORGREM] + field[GRIMMSNARL_EX] == 0:
        return -15000
    if cid == YVELTAL and plan.archetype == "crustle":
        return -14500
    if cid == AIR_BALLOON and plan.phase == "YVELTAL_LOCK":
        return -12000
    if plan.archetype == "alakazam" and field[GRIMMSNARL_EX] < 2:
        if cid in {IMPIDIMP, MORGREM, GRIMMSNARL_EX} and line_count < 2:
            return -22500
        if cid == RARE_CANDY and field[IMPIDIMP] > 0:
            return -21000
        if cid == BUDDY_BUDDY_POFFIN and line_count < 2:
            return -15000
    if (cid == SPIKEMUTH_GYM and plan.archetype == "alakazam"
            and _stadium_id(state) == BATTLE_CAGE):
        return -19000
    if ((plan.phase == "COUNTER_CONTROL" or plan.archetype == "archaludon")
            and cid in {SNORUNT, FROSLASS} and field[FROSLASS] < plan.desired_froslass):
        return -24000
    if (plan.phase == "COUNTER_CONTROL" and cid == BUDDY_BUDDY_POFFIN
            and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass):
        return -13000
    if cid == SPIKEMUTH_GYM and _stadium_id(state) == SPIKEMUTH_GYM:
        return 11000
    if data is not None and data.cardType == CardType.SUPPORTER and hand[cid] >= 2:
        return 8500
    if cid in {POKE_PAD, BUDDY_BUDDY_POFFIN} and state.turn >= 5:
        return 7000
    if cid in {SNORUNT, FROSLASS} and field[FROSLASS] >= plan.desired_froslass:
        return 6500
    return 1000 + max(0, hand[cid] - 1) * 1200


def _search_target_score(cid: int, source: int, hand: Counter, field: Counter, discard: Counter,
                         plan: Plan, mine, opponent, state) -> int:
    line_count = field[IMPIDIMP] + field[MORGREM] + field[GRIMMSNARL_EX]
    energy_punish = ALAKAZAM_ALT in set(_MEMORY["revealed"])
    if source == POKE_PAD:
        if plan.archetype == "lucario" and field[GRIMMSNARL_EX] < 2:
            if cid == GRIMMSNARL_EX and (field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0)):
                return 35000
            if cid == MORGREM and field[IMPIDIMP] > 0:
                return 33500
            if cid == IMPIDIMP and line_count < 3:
                return 32000
        if plan.archetype == "alakazam" and field[GRIMMSNARL_EX] < 2:
            if cid == GRIMMSNARL_EX and (field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0)):
                return 33500
            if cid == MORGREM and field[IMPIDIMP] > 0:
                return 32000
            if cid == IMPIDIMP and line_count < 2:
                return 30500
        if (plan.archetype == "archaludon" and line_count >= 2
                and field[FROSLASS] < plan.desired_froslass):
            if cid == FROSLASS and field[SNORUNT] > field[FROSLASS]:
                return 27500
            if cid == SNORUNT and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass:
                return 26750
        if (plan.phase == "COUNTER_CONTROL" and cid == FROSLASS
                and field[SNORUNT] > field[FROSLASS] and field[FROSLASS] < plan.desired_froslass):
            return 33000
        if (plan.phase == "COUNTER_CONTROL" and cid == SNORUNT
                and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass):
            return 31500
        if cid == CHI_YU and plan.archetype == "crustle" and field[CHI_YU] == 0:
            return 31000
        if cid == IMPIDIMP and line_count < 2:
            return 26000
        if cid == MORGREM and field[IMPIDIMP] > 0 and (hand[RARE_CANDY] == 0 or plan.archetype == "crustle"):
            return 25500
        if cid == GRIMMSNARL_EX and (field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0)):
            return 25000
        if cid == MUNKIDORI and field[MUNKIDORI] < plan.desired_munkis:
            return 23000
        if cid == FROSLASS and field[SNORUNT] > field[FROSLASS] and field[FROSLASS] < plan.desired_froslass:
            return 22000
        snorunt_target = plan.desired_froslass if energy_punish else max(1, plan.desired_froslass)
        if cid == SNORUNT and field[SNORUNT] + field[FROSLASS] < snorunt_target:
            return 19000
        if cid == YVELTAL and (plan.phase in {"WALL_BREAK", "YVELTAL_LOCK"} or plan.lock_value >= 7500):
            return 23500
        if cid == CHI_YU and plan.phase == "WALL_BREAK":
            return 30500
        if cid == BUDEW and state.turn <= 2 and not any(_ready_to_attack(p) for _, _, p in _field(mine)):
            return 14500
        return 1000

    if source == SPIKEMUTH_GYM:
        if plan.archetype == "alakazam" and field[GRIMMSNARL_EX] < 2:
            if cid == GRIMMSNARL_EX and (field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0)):
                return 35000
            if cid == MORGREM and field[IMPIDIMP] > 0:
                return 33000
            if cid == IMPIDIMP and line_count < 2:
                return 30000
        if cid == GRIMMSNARL_EX and (field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0)):
            return 28000
        if cid == MORGREM and field[IMPIDIMP] > 0:
            return 25000
        if cid == IMPIDIMP and line_count < plan.desired_grimms:
            return 20000
        return 3000

    if source == DAWN:
        # Dawn fills one card from each evolution stage. Score each offered card
        # by the live board rather than blindly selecting the highest stage.
        if plan.archetype == "archaludon" and field[FROSLASS] < plan.desired_froslass:
            if cid == FROSLASS and field[SNORUNT] > field[FROSLASS]:
                return 31200
            if cid == SNORUNT and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass:
                return 30500
        if (plan.phase == "COUNTER_CONTROL" and cid == SNORUNT
                and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass):
            return 33500
        if (plan.phase == "COUNTER_CONTROL" and cid == FROSLASS
                and field[SNORUNT] > field[FROSLASS] and field[FROSLASS] < plan.desired_froslass):
            return 34000
        if cid == MUNKIDORI and field[MUNKIDORI] < plan.desired_munkis:
            return 30000
        if cid == IMPIDIMP and line_count < plan.desired_grimms:
            return 28500
        if cid == SNORUNT and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass:
            return 26000
        if cid == MORGREM and field[IMPIDIMP] > 0:
            return 29000
        if cid == FROSLASS and field[SNORUNT] > field[FROSLASS] and field[FROSLASS] < plan.desired_froslass:
            return 29500
        if cid == GRIMMSNARL_EX and (field[MORGREM] > 0 or field[IMPIDIMP] > 0):
            return 31000
        return 4000

    if source == POKEGEAR:
        if cid == LILLIES_DETERMINATION:
            return 29000 if mine.handCount <= 5 or len(mine.prize or []) == 6 else 18000
        if cid == TEAM_ROCKETS_PETREL:
            return 28500 if plan.phase in {"OPENING", "EVOLUTION", "RECOVERY", "COUNTER_CONTROL"} else 21000
        if cid == DAWN:
            return 28000 if line_count < plan.desired_grimms or field[MUNKIDORI] < plan.desired_munkis else 19500
        if cid == BOSSS_ORDERS and plan.boss_target_serial is not None:
            return 26000
        return 5000

    if source == TEAM_ROCKETS_PETREL:
        if cid == UNFAIR_STAMP and _MEMORY["recent_ko"]:
            return 42000 if plan.archetype == "alakazam" and opponent.handCount >= 6 else 34000
        if cid == BOSSS_ORDERS and plan.archetype == "crustle" and plan.boss_target_serial is not None:
            target = next(
                (p for p in (opponent.bench or []) if p is not None and p.serial == plan.boss_target_serial),
                None,
            )
            charged = sum(1 for _, _, p in _field(mine) if p.id == MUNKIDORI and _dark_count(p) >= 1)
            movable = min(sum(_damage_on(p) for _, _, p in _field(mine)), charged * 30)
            if target is not None and target.id == 345 and 0 < target.hp <= movable:
                return 38000
        if (cid == BUDDY_BUDDY_POFFIN and plan.phase == "COUNTER_CONTROL"
                and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass
                and mine.benchMax > len(mine.bench)):
            return 30000
        if cid == RARE_CANDY and field[IMPIDIMP] > 0 and hand[GRIMMSNARL_EX] > 0:
            return 27000
        if cid == SPIKEMUTH_GYM and _stadium_id(state) != SPIKEMUTH_GYM:
            return 34000 if plan.archetype == "alakazam" and _stadium_id(state) == BATTLE_CAGE else 24500
        if cid == POKE_PAD:
            return 22500
        if cid == BUDDY_BUDDY_POFFIN and line_count < 2 and mine.benchMax > len(mine.bench):
            return 21500
        if cid == NIGHT_STRETCHER and any(discard[x] for x in {DARK_ENERGY, GRIMMSNARL_EX, MUNKIDORI, FROSLASS, IMPIDIMP}):
            return 20500
        if cid == POKEGEAR and not state.supporterPlayed:
            return 20000
        if cid == DAWN and line_count < plan.desired_grimms:
            return 21500
        if cid == TOOL_SCRAPPER and any(p.tools for _, _, p in _field(opponent)):
            return 22500
        if cid == AIR_BALLOON and plan.phase == "YVELTAL_LOCK":
            return 19000
        if cid == SECRET_BOX and mine.handCount >= 4 and mine.deckCount > 10:
            return 18500
        return 6000

    if source == SECRET_BOX:
        data = _card_data(cid)
        ctype = getattr(data, "cardType", None)
        if ctype == CardType.ITEM:
            if cid == RARE_CANDY and field[IMPIDIMP] > 0 and hand[GRIMMSNARL_EX] > 0:
                return 28000
            if cid == POKE_PAD:
                return 24000
            if cid == NIGHT_STRETCHER and discard:
                return 22000
            if cid == BUDDY_BUDDY_POFFIN and line_count < 2:
                return 21000
        elif ctype == CardType.TOOL:
            return 23000 if cid == AIR_BALLOON else 2000
        elif ctype == CardType.SUPPORTER:
            if cid == BOSSS_ORDERS and plan.boss_target_serial is not None:
                current_attacker = mine.active[0] if mine.active and _ready_to_attack(mine.active[0]) else None
                target = next((p for p in (opponent.bench or []) if p is not None and p.serial == plan.boss_target_serial), None)
                damage = 0 if _attack_damage_blocked(current_attacker, target) else _max_attack_damage(current_attacker.id) if current_attacker is not None else 0
                if plan.phase == "YVELTAL_LOCK" or (target is not None and target.hp <= damage):
                    return 26000
                return NEG
            if cid == JUDGE and (opponent.handCount >= 6 or plan.forecast.disruption_risk):
                return 23000
            if cid == LILLIES_DETERMINATION:
                return 21000
            if cid == TEAM_ROCKETS_PETREL:
                return 20000
        elif ctype == CardType.STADIUM:
            return 25000 if cid == SPIKEMUTH_GYM else 1000
        return 3000

    if source == NIGHT_STRETCHER:
        active = mine.active[0] if mine.active else None
        attack_energy_urgent = bool(
            active is not None
            and active.id == GRIMMSNARL_EX
            and not state.energyAttached
            and hand[DARK_ENERGY] == 0
            and not _ready_to_attack(active)
            and _ready_to_attack(active, 1)
        )
        if cid == DARK_ENERGY and attack_energy_urgent:
            return 50000
        if (
            cid == DARK_ENERGY
            and plan.archetype in {"marnie", "archaludon"}
            and hand[DARK_ENERGY] == 0
            and not state.energyAttached
            and any(p.id == MUNKIDORI and _dark_count(p) == 0 for _, _, p in _field(mine))
        ):
            return 39000
        if cid == FROSLASS and plan.phase == "COUNTER_CONTROL" and field[FROSLASS] < plan.desired_froslass:
            return 32500
        if (cid == SNORUNT and plan.phase == "COUNTER_CONTROL"
                and field[SNORUNT] + field[FROSLASS] < plan.desired_froslass):
            return 31500
        if cid == FIRE_ENERGY and plan.phase == "WALL_BREAK":
            return 29000
        if cid == CHI_YU and plan.phase == "WALL_BREAK" and field[CHI_YU] == 0:
            return 28500
        if cid == DARK_ENERGY and plan.phase in {"CHARGE", "YVELTAL_LOCK", "WALL_BREAK"}:
            return 25000
        if cid == GRIMMSNARL_EX and field[GRIMMSNARL_EX] < plan.desired_grimms:
            playable = field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0)
            if plan.archetype == "lucario" and playable and field[GRIMMSNARL_EX] < 2:
                return 35000
            if energy_punish and not playable:
                return NEG
            return 31500 if plan.archetype == "alakazam" else 24500
        if cid == MORGREM and energy_punish:
            return 25500 if field[IMPIDIMP] > 0 and hand[MORGREM] == 0 else NEG
        if cid == MUNKIDORI and field[MUNKIDORI] < plan.desired_munkis:
            return 22500
        if cid == YVELTAL and plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"}:
            return 23000
        if cid == FROSLASS and field[FROSLASS] < plan.desired_froslass:
            return 20500
        if cid == IMPIDIMP and (
            line_count == 0
            or (energy_punish and line_count < plan.desired_grimms and hand[IMPIDIMP] == 0)
        ):
            return 21000
        return NEG if energy_punish else 8000

    return _threat_from_id(cid)


def _threat_from_id(cid: int) -> int:
    data = _card_data(cid)
    if data is None:
        return 1000
    return 1500 + (2500 if getattr(data, "ex", False) else 0) + (1200 if getattr(data, "stage2", False) else 600 if getattr(data, "stage1", False) else 0) + (900 if getattr(data, "skills", []) else 0)


def _boss_target_score(p: Pokemon, plan: Plan) -> int:
    score = _threat(p, 180, False)
    if p.serial == plan.boss_target_serial:
        score += 7000
    if plan.phase == "YVELTAL_LOCK" and p.serial == plan.lock_target_serial:
        score += 18000
    if p.id in SETUP_TARGET_IDS:
        score += 3500
    return score


def _counter_target_score(p: Pokemon, plan: Plan, amount: int) -> int:
    score = _threat(p, amount, False)
    if p.id in SETUP_TARGET_IDS:
        score += 2500
    if p.serial == plan.boss_target_serial:
        score += 1200
    return score


def _play_score(cid: int, mine, opponent, field: Counter, hand: Counter, discard: Counter,
                plan: Plan, state) -> int:
    bench_free = mine.benchMax - len(mine.bench)
    line_count = field[IMPIDIMP] + field[MORGREM] + field[GRIMMSNARL_EX]
    data = _card_data(cid)
    ctype = getattr(data, "cardType", None)
    energy_punish = ALAKAZAM_ALT in set(_MEMORY["revealed"])

    if ctype == CardType.POKEMON:
        if bench_free <= 0:
            return NEG
        if cid == IMPIDIMP:
            return 24000 if line_count < plan.desired_grimms else NEG
        if cid == MUNKIDORI:
            return 22500 if field[MUNKIDORI] < plan.desired_munkis else NEG
        if cid == SNORUNT:
            snorunt_target = plan.desired_froslass if energy_punish else max(1, plan.desired_froslass)
            return 19500 if field[SNORUNT] + field[FROSLASS] < snorunt_target else NEG
        if cid == YVELTAL:
            return 25000 if plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"} or plan.lock_value >= 7500 else 6500
        if cid == CHI_YU:
            return 33000 if plan.phase == "WALL_BREAK" and field[CHI_YU] == 0 else NEG
        if cid == BUDEW:
            return 19000 if state.turn <= 2 and field[BUDEW] == 0 and not any(_ready_to_attack(p) for _, _, p in _field(mine)) else NEG
        return NEG

    if cid == BUDDY_BUDDY_POFFIN:
        snorunt_target = plan.desired_froslass if energy_punish else max(1, plan.desired_froslass)
        need = line_count < 2 or field[SNORUNT] + field[FROSLASS] < snorunt_target
        return 25500 if bench_free > 0 and need and _safe_draw(mine) else 2500
    if cid == POKE_PAD:
        need = line_count < plan.desired_grimms or field[MUNKIDORI] < plan.desired_munkis or field[FROSLASS] < plan.desired_froslass
        need = need or (plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"} and field[YVELTAL] == 0)
        need = need or (plan.phase == "WALL_BREAK" and field[CHI_YU] == 0)
        return 24500 if need and _safe_draw(mine) else 6500 if _safe_draw(mine) and mine.deckCount > 8 else NEG
    if cid == POKEGEAR:
        return 23500 if not state.supporterPlayed and mine.deckCount > 7 else NEG
    if cid == RARE_CANDY:
        return 32000 if field[IMPIDIMP] > 0 and hand[GRIMMSNARL_EX] > 0 else NEG
    if cid == NIGHT_STRETCHER:
        useful = any(discard[x] for x in {DARK_ENERGY, GRIMMSNARL_EX, IMPIDIMP, MUNKIDORI, FROSLASS})
        if energy_punish:
            active = mine.active[0] if mine.active else None
            urgent_energy = bool(
                discard[DARK_ENERGY]
                and hand[DARK_ENERGY] == 0
                and not state.energyAttached
                and active is not None
                and active.id == GRIMMSNARL_EX
                and not _ready_to_attack(active)
                and _ready_to_attack(active, 1)
            )
            playable_line = bool(
                discard[GRIMMSNARL_EX]
                and hand[GRIMMSNARL_EX] == 0
                and (field[MORGREM] > 0 or (field[IMPIDIMP] > 0 and hand[RARE_CANDY] > 0))
            )
            recover_basic = bool(
                discard[IMPIDIMP]
                and hand[IMPIDIMP] == 0
                and line_count < plan.desired_grimms
                and bench_free > 0
            )
            useful = urgent_energy or playable_line or recover_basic
        return 23000 if useful and mine.deckCount > 3 else NEG
    if cid == TOOL_SCRAPPER:
        opponent_tools = sum(len(p.tools or []) for _, _, p in _field(opponent))
        score = 24500 + opponent_tools * 2500
        # Removing a large-HP tool is especially valuable for the counter route.
        if any(any(getattr(tool, "id", 0) == 1159 for tool in (p.tools or [])) for _, _, p in _field(opponent)):
            score += 9000
        return score if opponent_tools > 0 else NEG
    if cid == SECRET_BOX:
        expendable = sum(1 for c in (mine.hand or []) if _discard_score(c.id, hand, field, plan, state) > 0)
        return 26000 if expendable >= 3 and mine.deckCount > 10 else NEG
    if cid == AIR_BALLOON:
        return 22000 if plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"} or (mine.active and mine.active[0].id not in {GRIMMSNARL_EX, YVELTAL, CHI_YU}) else 7000
    if cid == UNFAIR_STAMP:
        # The engine only offers Stamp when its KO timing condition is legal.
        score = 31500
        score += 6000 if opponent.handCount >= 6 else 2000 if opponent.handCount >= 4 else 0
        score += 3500 if mine.handCount <= 4 else 0
        if plan.archetype == "alakazam" and opponent.handCount >= 6:
            score += 9000
        return score if mine.deckCount > 5 else NEG
    if cid == HANDHELD_FAN:
        # A single Fan on the primary 320-HP attacker converts an opposing hit
        # into Energy displacement without spending the Supporter for the turn.
        has_target = any(p.id in MARNIE_LINE and not (p.tools or []) for _, _, p in _field(mine))
        return 23500 if has_target else NEG
    if cid == SPIKEMUTH_GYM:
        if plan.archetype == "alakazam" and _stadium_id(state) == BATTLE_CAGE:
            return 38500
        return 25000 if _stadium_id(state) != SPIKEMUTH_GYM else NEG

    if ctype == CardType.SUPPORTER:
        if state.supporterPlayed:
            return NEG
        if cid == BOSSS_ORDERS:
            if plan.phase == "YVELTAL_LOCK" and plan.lock_target_serial is not None and not plan.forecast.bench_ready:
                return 41000
            current_attacker = mine.active[0] if mine.active and _ready_to_attack(mine.active[0]) else None
            target = next((p for p in (opponent.bench or []) if p is not None and p.serial == plan.boss_target_serial), None)
            if target is not None and current_attacker is not None:
                max_damage = 0 if _attack_damage_blocked(current_attacker, target) else _max_attack_damage(current_attacker.id)
                if plan.archetype == "crustle":
                    charged = sum(1 for _, _, p in _field(mine) if p.id == MUNKIDORI and _dark_count(p) >= 1)
                    movable = min(sum(_damage_on(p) for _, _, p in _field(mine)), charged * 30)
                    if target.id == 345 and 0 < target.hp <= movable:
                        return 38500 + _prize_value(target) * 2500
                # Boss must produce a KO or remove a low-HP setup engine with the
                # Pokemon that is actually Active, not merely a charged Bench unit.
                if target.hp <= max_damage or (target.id in SETUP_TARGET_IDS and target.hp <= 180 and current_attacker.id == GRIMMSNARL_EX):
                    return 30000 + _prize_value(target) * 2500
            return NEG
        if cid == JUDGE:
            ready_primary = any(p.id == GRIMMSNARL_EX and _ready_to_attack(p) for _, _, p in _field(mine))
            if not ready_primary and plan.phase in {"OPENING", "EVOLUTION"} and mine.handCount >= 5:
                return NEG
            score = 11000
            score += 10000 if opponent.handCount >= 7 else 5000 if opponent.handCount >= 5 else -4000
            score += 4000 if plan.archetype in {"alakazam", "dragapult", "marnie"} else 0
            score += 2500 if mine.handCount <= 4 else -6500 if mine.handCount >= 7 else 0
            return score
        if cid == LILLIES_DETERMINATION:
            score = 19000 if len(mine.prize or []) == 6 else 14500
            score += 4500 if mine.handCount <= 4 else -6000 if mine.handCount >= 8 else 0
            return score if mine.deckCount > 7 else NEG
        if cid == DAWN:
            need = line_count < plan.desired_grimms or field[MUNKIDORI] < plan.desired_munkis
            need = need or field[FROSLASS] < plan.desired_froslass
            return 27000 if need and mine.deckCount > 6 else 12500 if mine.deckCount > 10 else NEG
        if cid == TEAM_ROCKETS_PETREL:
            score = 17500
            score += 6000 if _stadium_id(state) != SPIKEMUTH_GYM else 0
            score += 7000 if field[IMPIDIMP] > 0 and hand[GRIMMSNARL_EX] > 0 and hand[RARE_CANDY] == 0 else 0
            score += 3500 if plan.phase in {"RECOVERY", "COUNTER_CONTROL"} else 0
            return score
    return 1000


def _attach_score(card, target: Pokemon, plan: Plan, mine, opponent=None) -> int:
    if card is None or target is None or card.id not in {DARK_ENERGY, FIRE_ENERGY, PRISM_ENERGY}:
        return NEG
    score = 0
    if card.id == PRISM_ENERGY:
        # Keep the one-off Prism for an actual emergency attack. A normal
        # Darkness remains the preferred first attachment because it is
        # searchable by Punk Up and recoverable with Night Stretcher.
        if target.id != MUNKIDORI or _has_prism(target):
            return NEG
        total = _energy_count(target)
        active_target = bool(mine.active and mine.active[0].serial == target.serial)
        op_active = opponent.active[0] if opponent is not None and opponent.active else None
        other_ready = any(
            p.serial != target.serial and _ready_to_attack(p)
            for _, _, p in _field(mine)
        )
        # With one Darkness already attached, Prism is the emergency second
        # Energy that unlocks Mind Bend immediately.
        if total == 1 and _dark_count(target) >= 1:
            ko_now = bool(
                active_target and op_active is not None and op_active.hp <= 60
                and not _attack_damage_blocked(target, op_active)
            )
            forced_attack = active_target and not other_ready
            wall_break = bool(active_target and op_active is not None and op_active.id == 345 and not other_ready)
            endgame_backup = bool(not active_target and not plan.forecast.active_ready and not plan.forecast.bench_ready
                                  and plan.phase in {"ENDGAME", "RECOVERY", "WALL_BREAK"})
            if not (ko_now or forced_attack or wall_break or endgame_backup):
                return NEG
            score = 45000
            if active_target:
                score += 9000
            if ko_now:
                score += 32000 + _prize_value(op_active) * 6000
            elif wall_break:
                score += 15000
            elif forced_attack:
                score += 9000
            if endgame_backup:
                score += 6000
            return score
        # When no Darkness is available, Prism may serve as the first attachment
        # to keep Adrena-Brain online. This is allowed only when the Ability has
        # immediate counter value or Munkidori is stranded Active; otherwise the
        # card is held for the later attack conversion.
        if total == 0:
            dark_in_hand = any(getattr(c, "id", -1) == DARK_ENERGY for c in (mine.hand or []))
            movable = sum(_damage_on(p) for _, _, p in _field(mine))
            stranded = active_target and not other_ready
            counter_urgent = movable >= 20 and plan.phase in {"COUNTER_CONTROL", "RECOVERY", "ENDGAME", "WALL_BREAK"}
            crustle_engine = plan.archetype == "crustle" and (active_target or movable >= 10)
            if (dark_in_hand and not crustle_engine) or not (stranded or counter_urgent or crustle_engine):
                return NEG
            score = 40500 + (8000 if active_target else 0) + min(9000, movable * 120)
            if crustle_engine:
                score += 6500
            return score
        return NEG
    if target.id == CHI_YU:
        if plan.phase != "WALL_BREAK":
            return 4000 if mine.active and mine.active[0].serial == target.serial else NEG
        if card.id == FIRE_ENERGY and _fire_count(target) == 0:
            return 43000
        if _fire_count(target) >= 1 and _energy_count(target) < 2:
            return 39000
        if card.id == FIRE_ENERGY and _energy_count(target) < 2:
            return 36000
        return NEG
    if card.id == FIRE_ENERGY:
        # Fire is reserved for Chi-Yu. It can pay Yveltal's colorless third cost
        # only after two Darkness are already present.
        if target.id == YVELTAL and _dark_count(target) >= 2 and _energy_count(target) < 3:
            return 23500
        return NEG
    if (
        target.id == GRIMMSNARL_EX
        and mine.active
        and mine.active[0].serial == target.serial
        and not _ready_to_attack(target)
        and _ready_to_attack(target, 1)
    ):
        score += 55000
    if target.serial == plan.energy_target_serial:
        score += 18000
    if target.id == MUNKIDORI:
        # Manual Darkness belongs on Adrena-Brain; if Prism is already attached,
        # the Darkness attachment immediately completes Mind Bend.
        if _has_prism(target) and _energy_count(target) == 1:
            score += 53500
            if mine.active and mine.active[0].serial == target.serial:
                score += 9000
        else:
            score += 33500 if _dark_count(target) == 0 else NEG
    elif target.id == YVELTAL:
        need = max(1, _min_attack_cost(YVELTAL))
        score += 26000 if plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"} and _dark_count(target) < need else 8000 - _dark_count(target) * 2000
    elif target.id == GRIMMSNARL_EX:
        score += 20500 - _dark_count(target) * 7000
    elif target.id == MORGREM:
        score += 16500 - _dark_count(target) * 6000
    elif target.id == IMPIDIMP:
        score += 13500 - _dark_count(target) * 5500
    else:
        score = NEG
    return score


def _attack_score(aid: int, active: Optional[Pokemon], opponent, plan: Plan) -> int:
    op_active = opponent.active[0] if opponent.active else None
    name = _attack_name(aid)
    attack = ATTACK.get(aid)
    damage = int(getattr(attack, "damage", 0) or 0)

    if aid == SHADOW_BULLET or name == "shadow bullet":
        if op_active is None:
            return 7000
        blocked = _attack_damage_blocked(active, op_active)
        score = 12500 if blocked else 28000 + damage * 35
        if not blocked and op_active.hp <= 180:
            score += 22000 + _prize_value(op_active) * 6500
        bench_targets = [
            p for p in (opponent.bench or [])
            if p is not None and not _bench_attack_damage_blocked(active, p, opponent)
        ]
        if bench_targets:
            score += 6500
        if any(p.hp <= 30 for p in bench_targets):
            score += 26000 + max((_prize_value(p) for p in bench_targets if p.hp <= 30), default=0) * 5000
        if any(p.id in SETUP_TARGET_IDS and p.hp <= 60 for p in bench_targets):
            score += 5000
        if blocked and bench_targets:
            # Damage prevention on the Active does not prevent Shadow Bullet's
            # 30 Bench damage. Keep progressing the counter route.
            score += 9000
        if plan.phase == "COUNTER_CONTROL":
            score += 7000
        if plan.phase == "ENDGAME":
            score += 7000
        return score

    if aid == CLUTCH or name == "clutch":
        if op_active is None:
            return NEG
        value = _lock_target_value(op_active, opponent, plan.forecast)
        if _stadium_id_dummy() == SPIKEMUTH_GYM:
            value += 1200
        if op_active.serial == plan.lock_target_serial:
            value += 5000
        # Only lock when it buys a real development turn; never replace a winning KO.
        return 18000 + value if value >= 5000 else 4500

    if aid == GROUND_MELTER or name == "ground melter":
        if op_active is None:
            return NEG
        stadium_live = bool(getattr(opponent, "active", None))  # live-state value is added below by matchup
        # Printed damage is 60, plus 60 while any Stadium is in play. Against
        # Fire-Weak Crustle this becomes the clean one-Prize wall-breaking route.
        score = 25500
        if plan.phase == "WALL_BREAK":
            score += 30000
        if op_active.id == 345:
            score += 28000
        if op_active.hp <= 120:
            score += 15000 + _prize_value(op_active) * 4500
        return score

    if aid == DARK_FEATHER or name == "dark feather":
        score = 15500 + damage * 28
        if op_active is not None and op_active.hp <= max(damage, 120):
            score += 17000 + _prize_value(op_active) * 4500
        if plan.phase == "WALL_BREAK":
            score += 22000
        return score

    if aid == ITCHY_POLLEN or name == "itchy pollen":
        score = 22000 if plan.phase == "OPENING" and plan.forecast.evolution_threat else 9500
        if plan.archetype in {"dragapult", "lucario", "crustle", "alakazam"}:
            score += 4500
        return score

    if aid == MORGREM_PUNCH or name == "morgrem punch":
        score = 12000 + damage * 25
        if plan.phase == "WALL_BREAK":
            score += 19000
        if op_active is not None and op_active.hp <= damage:
            score += 16000
        return score

    if aid == MIND_BEND or name == "mind bend":
        if op_active is None:
            return 11500 + damage * 20
        blocked = _attack_damage_blocked(active, op_active)
        score = 13500 if blocked else 17500 + damage * 28
        if not blocked and op_active.hp <= damage:
            score += 30000 + _prize_value(op_active) * 6000
        elif not blocked:
            # Confusion can buy a development turn when Munkidori is forced Active
            # or the main attacker is not yet online.
            score += 7000
            if plan.phase in {"OPENING", "RECOVERY", "COUNTER_CONTROL", "WALL_BREAK"}:
                score += 5500
            if op_active.id == 345:  # Crustle is immune only to Pokémon ex.
                score += 10000
        if active is not None and active.id == MUNKIDORI and active.hp <= 50:
            score += 3500
        return score
    if aid == FROST_SMASH or name == "frost smash":
        return 9000 + damage * 20
    if aid == FILCH or name == "filch":
        return 6500 if plan.phase == "OPENING" and opponent.deckCount > 6 else NEG

    score = 9000 + damage * 22
    if op_active is not None and damage >= op_active.hp and damage > 0:
        score += 17000 + _prize_value(op_active) * 4500
    return score


def _main_score(obs: Observation, option, mine, opponent, field: Counter, hand: Counter,
                discard: Counter, plan: Plan) -> int:
    state = obs.current
    active = mine.active[0] if mine.active else None
    op_active = opponent.active[0] if opponent.active else None
    typ = option.type

    if typ == OptionType.PLAY:
        card = _get_card(obs, AreaType.HAND, option.index, state.yourIndex)
        return _play_score(card.id, mine, opponent, field, hand, discard, plan, state) if card is not None else NEG
    if typ == OptionType.EVOLVE:
        evo = _get_card(obs, option.area, option.index, state.yourIndex)
        target = _get_card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
        if evo is None or target is None:
            return NEG
        if evo.id == GRIMMSNARL_EX:
            if plan.archetype == "crustle" and field[GRIMMSNARL_EX] >= plan.desired_grimms:
                return NEG
            score = 39000 if field[GRIMMSNARL_EX] == 0 else 33000
            score += min(2, _dark_count(target)) * 22000
            if active is not None and target.serial == active.serial:
                score += 5000
            return score
        if evo.id == MORGREM:
            return 30000 if field[GRIMMSNARL_EX] == 0 else 24500
        if evo.id == FROSLASS:
            if field[FROSLASS] >= plan.desired_froslass:
                return NEG
            return 36500 if plan.archetype == "archaludon" else 28000
        return 12000
    if typ == OptionType.ATTACH:
        card = _get_card(obs, option.area, option.index, state.yourIndex)
        target = _get_card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
        return _attach_score(card, target, plan, mine, opponent)
    if typ == OptionType.ABILITY:
        card = _get_card(obs, option.area, option.index, state.yourIndex)
        if card is None:
            return NEG
        if card.id == SPIKEMUTH_GYM:
            if plan.archetype == "alakazam" and field[GRIMMSNARL_EX] < 2:
                return 34500
            return 28500 if field[GRIMMSNARL_EX] < plan.desired_grimms else 16500
        if card.id == MUNKIDORI:
            own_damage = sum(_damage_on(p) for _, _, p in _field(mine))
            return 35000 if own_damage >= 30 else 27500 if own_damage >= 10 else NEG
        if card.id == GRIMMSNARL_EX:
            return 34000
        return 17000
    if typ == OptionType.RETREAT:
        if active is None:
            return NEG
        if plan.phase == "YVELTAL_LOCK" and any(p.id == YVELTAL and _ready_to_attack(p) for p in (mine.bench or [])):
            return 31500
        if plan.phase == "WALL_BREAK" and active.id == GRIMMSNARL_EX:
            if any(p.id == CHI_YU and _chi_yu_ready(p) for p in (mine.bench or [])):
                return 42000
            wall = opponent.active[0] if opponent.active else None
            movable_damage = _damage_on(active)
            for p in (mine.bench or []):
                if p is None or p.id not in {YVELTAL, MORGREM, CHI_YU} or not (_chi_yu_ready(p) if p.id == CHI_YU else _ready_to_attack(p)):
                    continue
                damage = 120 if p.id == CHI_YU else 100 if p.id == YVELTAL and _dark_count(p) >= 3 else 30 if p.id == YVELTAL else 60
                if wall is not None and wall.hp <= damage:
                    return 36000
            # Keep the 320-HP ex Active as a damage reservoir until Adrena-Brain
            # has counters to move; an early Yveltal promotion is simply KO'd.
            if movable_damage >= 60:
                return 30000
            return NEG
        if active.id == BUDEW and state.turn <= 2 and plan.phase == "OPENING":
            return NEG
        if not _ready_to_attack(active) and any(_ready_to_attack(p) for p in (mine.bench or [])):
            return 24500
        if active.id != GRIMMSNARL_EX and any(p.id == GRIMMSNARL_EX and _ready_to_attack(p) for p in (mine.bench or [])):
            return 23500
        return NEG
    if typ == OptionType.ATTACK:
        score = _attack_score(option.attackId, active, opponent, plan)
        # Preserve the replay-derived order: complete critical evolution/attachment first.
        if plan.phase in {"EVOLUTION", "CHARGE"} and score < 50000:
            score -= 7000
        return score
    if typ == OptionType.END:
        return -5000
    if typ == OptionType.DISCARD:
        return -2000
    return 0


def _card_option_score(obs: Observation, option, mine, opponent, field: Counter, hand: Counter,
                       discard: Counter, plan: Plan, source: int) -> int:
    state = obs.current
    me = state.yourIndex
    player_index = option.playerIndex if option.playerIndex is not None else me
    card = _get_card(obs, option.area, option.index, player_index)
    if card is None:
        return NEG
    cid = card.id
    context = obs.select.context

    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        # We choose first, so protect Munkidori and lead the evolution line when possible.
        priority = {IMPIDIMP: 12000, YVELTAL: 8500, CHI_YU: 7200, MUNKIDORI: 7000, SNORUNT: 5500}
        return priority.get(cid, 0)

    if context == SelectContext.SETUP_BENCH_POKEMON:
        counts = Counter(c.id for c in (mine.hand or []))
        # A second expendable Snorunt beside a Snorunt opener gave the fast
        # Lucario replay a free extra Prize before the first attacker existed.
        # Keep it in hand; it can still be benched on our first turn if needed.
        if cid == SNORUNT and _MEMORY.get("setup_active_id") == SNORUNT:
            return NEG
        priority = {IMPIDIMP: 12000, MUNKIDORI: 10000, SNORUNT: 8500, YVELTAL: 3500, CHI_YU: -1000}
        return priority.get(cid, NEG) - max(0, counts[cid] - 1) * 300

    if context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        if player_index != me:
            return _boss_target_score(card, plan)
        if plan.phase == "YVELTAL_LOCK" and cid == YVELTAL and _ready_to_attack(card):
            return 36000
        if plan.phase == "WALL_BREAK" and cid == CHI_YU and _chi_yu_ready(card):
            return 41000
        if plan.phase == "WALL_BREAK" and cid in {YVELTAL, MORGREM} and _ready_to_attack(card):
            return 34500
        if cid == GRIMMSNARL_EX and _ready_to_attack(card):
            return 45000 + card.hp
        if cid == GRIMMSNARL_EX and _ready_to_attack(card, 1):
            return 35000 + card.hp
        if cid == YVELTAL and _ready_to_attack(card):
            return 24500
        if cid == BUDEW and plan.phase == "OPENING":
            return 18000
        offered = []
        for offered_option in obs.select.option:
            if offered_option.type != OptionType.CARD:
                continue
            offered_card = _get_card(
                obs,
                offered_option.area,
                offered_option.index,
                offered_option.playerIndex if offered_option.playerIndex is not None else me,
            )
            if isinstance(offered_card, Pokemon):
                offered.append(offered_card)
        expendable_available = any(
            p.id in {SNORUNT, BUDEW}
            or (p.id == MUNKIDORI and _dark_count(p) == 0)
            for p in offered
        )
        knockout_pressure = plan.forecast.active_ready or plan.forecast.estimated_damage > 0
        can_stabilize_as_grimms = (
            (cid == IMPIDIMP and hand[RARE_CANDY] > 0 and hand[GRIMMSNARL_EX] > 0)
            or (cid == MORGREM and hand[GRIMMSNARL_EX] > 0)
        )
        score = 7000 + card.hp
        if plan.forecast.estimated_damage > 0 and card.hp > plan.forecast.estimated_damage:
            score += 4500
        sacrificial_mode = expendable_available and knockout_pressure and not can_stabilize_as_grimms
        if sacrificial_mode:
            score -= _energy_count(card) * 4500
        else:
            # When no disposable pivot exists, preserve tempo by promoting the
            # invested copy rather than an identical zero-Energy evolution.
            score += _energy_count(card) * 14000
            if can_stabilize_as_grimms:
                score += 18000
        if cid == MORGREM:
            score -= 14500
        elif cid == IMPIDIMP:
            score -= 9500
        elif cid == GRIMMSNARL_EX:
            score -= 7000
        elif cid == SNORUNT:
            score += 8000 if plan.desired_froslass == 0 else 1500
        elif cid == MUNKIDORI:
            score += 5500 if _dark_count(card) == 0 else -2500
        elif cid == BUDEW:
            score += 6000
        if sacrificial_mode:
            if cid in MARNIE_LINE:
                score -= 42000
            elif cid in {SNORUNT, BUDEW} or (cid == MUNKIDORI and _dark_count(card) == 0):
                score += 22000
        return score

    if context in {SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
        if source == HANDHELD_FAN and isinstance(card, Pokemon) and player_index == me:
            if card.id == GRIMMSNARL_EX:
                return 33000 + card.hp
            if card.id in {MORGREM, IMPIDIMP}:
                return 26000 + card.hp
            if card.id == MUNKIDORI:
                return 18000 + card.hp
            return 8000 + card.hp
        if source == BUDDY_BUDDY_POFFIN:
            line_count = field[IMPIDIMP] + field[MORGREM] + field[GRIMMSNARL_EX]
            if cid == IMPIDIMP:
                return 28000 - line_count * 5000 if line_count < plan.desired_grimms else NEG
            energy_punish = ALAKAZAM_ALT in set(_MEMORY["revealed"])
            early_snorunt_target = 0 if energy_punish else 1 if state.turn <= 3 else 0
            snorunt_target = max(early_snorunt_target, plan.desired_froslass)
            if cid == SNORUNT:
                return 22000 if field[SNORUNT] + field[FROSLASS] < snorunt_target else NEG
            if cid == BUDEW and state.turn <= 2 and field[BUDEW] == 0:
                return 17000
            return NEG
        if cid == CHI_YU and plan.phase == "WALL_BREAK":
            return 32000
        if cid == YVELTAL and plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"}:
            return 28000
        if cid == MUNKIDORI and field[MUNKIDORI] < plan.desired_munkis:
            return 24500
        if cid == IMPIDIMP and field[GRIMMSNARL_EX] < plan.desired_grimms:
            return 25500
        if cid == SNORUNT and field[FROSLASS] < plan.desired_froslass:
            return 21000
        return 3000

    if context == SelectContext.TO_HAND:
        return _search_target_score(cid, source, hand, field, discard, plan, mine, opponent, state)

    if context in {SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO}:
        if cid == GRIMMSNARL_EX:
            return 30000
        if cid == MORGREM:
            return 26000
        if cid == IMPIDIMP:
            score = 24000 + min(2, _dark_count(card)) * 12000
            if mine.active and mine.active[0].serial == card.serial:
                score += 3500
            return score
        if cid == FROSLASS and field[FROSLASS] < plan.desired_froslass:
            return 23000
        if cid == SNORUNT:
            return 21000
        return 5000

    if context == SelectContext.DISCARD:
        return _discard_score(cid, hand, field, plan, state)

    if context in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE, SelectContext.EFFECT_TARGET}:
        if isinstance(card, Pokemon):
            amount = max(10, int(getattr(obs.select, "remainDamageCounter", 0) or 0) * 10)
            if source == GRIMMSNARL_EX and context == SelectContext.DAMAGE:
                amount = 30
            elif source == MUNKIDORI:
                amount = min(30, sum(_damage_on(p) for _, _, p in _field(mine)))
                amount = max(10, amount)
            if player_index == me:
                # Effects that target our board should generally pick safe, damaged Pokémon for removal/healing.
                return _damage_on(card) * 30 + (3000 if card.id == MUNKIDORI else 0)
            if (
                source == GRIMMSNARL_EX
                and context == SelectContext.DAMAGE
                and _bench_attack_damage_blocked(mine.active[0] if mine.active else None, card, opponent)
            ):
                return NEG
            score = _counter_target_score(card, plan, amount)
            # v29 candidate: in the mirror, Phantom-Dive-style 30 bench damage
            # from Grimmsnarl should seed an unevolved Snorunt before a full-HP
            # Munkidori when no immediate KO is available.  This creates a
            # future Adrena-Brain KO window and can deny the Froslass engine.
            if (
                plan.archetype == "marnie"
                and source == GRIMMSNARL_EX
                and context == SelectContext.DAMAGE
                and cid == SNORUNT
                and card.hp > amount
                and not any(p.id == FROSLASS for _, _, p in _field(opponent))
            ):
                score += 9000
            return score
        return NEG

    if context in {SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL}:
        if isinstance(card, Pokemon) and player_index == me:
            damage = _damage_on(card)
            score = damage * 45
            score += 5500 if card.id == MUNKIDORI else 3500 if card.id == GRIMMSNARL_EX else 0
            # v26/v27 validated matchup residual rule.  Preserve a damaged
            # Munkidori only in matchups where independent A/B tests showed a
            # repeatable gain.  All other matchups retain the exact v25 rule.
            if (
                source == MUNKIDORI
                and card.id == MUNKIDORI
                and damage > 0
                and plan.archetype in {"marnie", "dragapult"}
            ):
                score += 24000 if card.hp <= 40 else 15000 if card.hp <= 70 else 7000
            score += 2000 if option.area == AreaType.ACTIVE else 0
            return score
        return NEG

    if context == SelectContext.ATTACH_TO:
        return 20000 if cid == DARK_ENERGY else NEG

    if context == SelectContext.ATTACH_FROM:
        if not isinstance(card, Pokemon):
            return NEG
        deficit = max(0, 2 - _dark_count(card))
        if deficit <= 0:
            return NEG
        if card.id == GRIMMSNARL_EX:
            score = 27000 + deficit * 3500
            if mine.active and mine.active[0].serial == card.serial:
                score += 4500
            return score
        if card.id == MORGREM:
            score = 24500 + deficit * 3000
            if mine.active and mine.active[0].serial == card.serial:
                score += 4000
            return score
        if card.id == IMPIDIMP:
            return 22500 + deficit * 2500
        # Punk Up can only target Marnie's Pokémon; keep others illegal-low.
        return NEG

    if context in {SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.NOT_MOVE}:
        return -_discard_score(cid, hand, field, plan, state)

    return _threat(card if isinstance(card, Pokemon) else None) if isinstance(card, Pokemon) else 1000


def _forced_punk_up_count(obs: Observation, mine, field: Counter, plan: Plan, source: int) -> Optional[int]:
    if obs.select.context != SelectContext.ATTACH_TO or source != GRIMMSNARL_EX:
        return None
    dark_options = 0
    for o in obs.select.option:
        if o.type != OptionType.CARD:
            continue
        c = _get_card(obs, o.area, o.index, obs.current.yourIndex)
        if c is not None and c.id == DARK_ENERGY:
            dark_options += 1
    marnie_deficit = sum(max(0, 2 - _dark_count(p)) for _, _, p in _field(mine) if p.id in MARNIE_LINE)
    reserve = sum(1 for _, _, p in _field(mine) if p.id == MUNKIDORI and _dark_count(p) == 0)
    if plan.phase in {"YVELTAL_LOCK", "WALL_BREAK"} and any(p.id == YVELTAL and _dark_count(p) == 0 for _, _, p in _field(mine)):
        reserve += 1
    spendable = max(0, dark_options - min(2, reserve))
    wanted = min(5, marnie_deficit)
    # In the human Crustle replay, three Punk Up activations consumed nine of
    # ten Darkness Energy while both Munkidori stayed uncharged until turn 13.
    # Grimmsnarl cannot damage this wall, so keep Energy in the deck for manual
    # Munkidori attachments and stop each activation at the attack minimum.
    energy_punish = bool({ALAKAZAM_ALT, CRUSHING_HAMMER} & set(_MEMORY["revealed"]))
    if (plan.archetype == "crustle" or energy_punish) and reserve > 0:
        wanted = min(wanted, 2)
    if plan.archetype in {"marnie", "archaludon"} and reserve > 0:
        charged_munkis = sum(1 for _, _, p in _field(mine) if p.id == MUNKIDORI and _dark_count(p) >= 1)
        deployed_munkis = sum(1 for _, _, p in _field(mine) if p.id == MUNKIDORI)
        # Refined mirror correction: do not throttle the first Punk Up merely
        # because one lone Munkidori is uncharged. Reserve aggressively only
        # after the two-Munkidori engine is actually established.
        if (plan.archetype != "marnie" or deployed_munkis >= 2) and charged_munkis < 2:
            wanted = min(wanted, 2)
    return max(obs.select.minCount, min(obs.select.maxCount, dark_options, wanted, spendable))


def _validate_result(result: list[int], select) -> list[int]:
    valid = []
    seen = set()
    for i in result:
        if isinstance(i, int) and 0 <= i < len(select.option) and i not in seen:
            valid.append(i)
            seen.add(i)
    if len(valid) < select.minCount:
        for i in range(len(select.option)):
            if i not in seen:
                valid.append(i)
                seen.add(i)
                if len(valid) >= select.minCount:
                    break
    return valid[:select.maxCount]


def _immediate_winning_attack(obs: Observation, option, mine, opponent) -> bool:
    if option.type != OptionType.ATTACK or not opponent.active:
        return False
    target = opponent.active[0]
    attacker = mine.active[0] if mine.active else None
    remaining_prizes = len(mine.prize or [])
    if remaining_prizes <= 0:
        return True
    attack = ATTACK.get(option.attackId)
    damage = int(getattr(attack, "damage", 0) or 0)
    if option.attackId == SHADOW_BULLET:
        damage = 180
        if any(
            p is not None
            and not _bench_attack_damage_blocked(attacker, p, opponent)
            and p.hp <= 30
            and _prize_value(p) >= remaining_prizes
            for p in (opponent.bench or [])
        ):
            return True
    elif option.attackId == GROUND_MELTER:
        damage = 120
    if _attack_damage_blocked(attacker, target):
        damage = 0
    return damage > 0 and target.hp <= damage and _prize_value(target) >= remaining_prizes


def _apply_turn_order(scores: list[int], select, obs: Observation, mine, opponent, archetype: str) -> None:
    """Prevent an attack from ending the turn while useful setup remains.

    Replays consistently play/search/evolve/use abilities/attach first.  This is
    not imitation for its own sake: ATTACK ends the turn, so skipping those legal
    actions permanently loses tempo.  Immediate game-winning attacks are exempt.
    """
    if select.context != SelectContext.MAIN:
        return
    attack_indices = [i for i, o in enumerate(select.option) if o.type == OptionType.ATTACK]
    if not attack_indices:
        return
    priority_threshold = {
        OptionType.PLAY: 6000,
        OptionType.EVOLVE: 10000,
        OptionType.ABILITY: 12000,
        OptionType.ATTACH: 5000,
        OptionType.RETREAT: 22000,
    }
    candidates = []
    tier_bonus = {
        # Most play/search actions happen before evolution/attachment.  Adrena-Brain
        # is the exception: move counters before a shuffle, switch, heal or KO can
        # remove the opportunity.
        OptionType.PLAY: 400000,
        OptionType.EVOLVE: 320000,
        OptionType.ABILITY: 260000,
        OptionType.ATTACH: 180000,
        OptionType.RETREAT: 120000,
    }
    for i, o in enumerate(select.option):
        threshold = priority_threshold.get(o.type)
        if threshold is not None and scores[i] >= threshold:
            candidates.append(i)
            bonus = tier_bonus[o.type]
            if o.type == OptionType.ABILITY:
                source = _get_card(obs, o.area, o.index, obs.current.yourIndex)
                if source is not None and source.id == MUNKIDORI:
                    movable = min(30, sum(_damage_on(p) for _, _, p in _field(mine)))
                    counter_ko = movable > 0 and any(0 < p.hp <= movable for _, _, p in _field(opponent))
                    fragile_source = any(
                        p.id == MUNKIDORI and _damage_on(p) > 0 and p.hp <= 60
                        for _, _, p in _field(mine)
                    )
                    # Correctly aligned replay rule: move counters before
                    # evolution when doing so has immediate tactical value.
                    # Crustle keeps the proven v19 order because its control
                    # matchup lost ground in broad validation.
                    tactical = counter_ko or fragile_source or movable >= 20
                    if archetype == "archaludon":
                        # v28: against the fast Archaludon/Cinderace shell,
                        # develop/search/evolve before a routine counter move.
                        # Immediate KOs and saving a damaged Munkidori retain
                        # maximum priority; only low-consequence movement waits.
                        urgent = counter_ko or fragile_source
                        bonus = 650000 if urgent else 350000
                    else:
                        bonus = 650000 if tactical and archetype != "crustle" else 500000
                elif source is not None and source.id == GRIMMSNARL_EX:
                    bonus = 440000
                elif source is not None and source.id == SPIKEMUTH_GYM:
                    # v26 mirror-only sequencing rule.  Search/thin the
                    # evolution line before a non-tactical Adrena-Brain or
                    # ordinary evolution, while an immediate counter-KO keeps
                    # the higher 650k Munkidori priority.
                    mirror_field = Counter(p.id for _, _, p in _field(mine))
                    if archetype == "marnie" and mirror_field[GRIMMSNARL_EX] < 2:
                        bonus = 600000
            if o.type == OptionType.ATTACH:
                target = _get_card(obs, o.inPlayArea, o.inPlayIndex, obs.current.yourIndex)
                if (
                    target is not None
                    and target.id == GRIMMSNARL_EX
                    and mine.active
                    and mine.active[0].serial == target.serial
                    and not _ready_to_attack(target)
                    and _ready_to_attack(target, 1)
                ):
                    bonus = 560000
            elif o.type == OptionType.EVOLVE:
                evo = _get_card(obs, o.area, o.index, obs.current.yourIndex)
                target = _get_card(obs, o.inPlayArea, o.inPlayIndex, obs.current.yourIndex)
                if (
                    evo is not None
                    and target is not None
                    and evo.id == GRIMMSNARL_EX
                    and (
                        _dark_count(target) > 0
                        or any(
                            play.type == OptionType.PLAY
                            and (
                                (played := _get_card(obs, AreaType.HAND, play.index, obs.current.yourIndex))
                                is not None
                            )
                            and played.id in {LILLIES_DETERMINATION, UNFAIR_STAMP, JUDGE}
                            for play in select.option
                        )
                    )
                ):
                    bonus = 520000
            scores[i] += bonus
    if not candidates:
        return
    best_non_attack = max(scores[i] for i in candidates)
    for i in attack_indices:
        if _immediate_winning_attack(obs, select.option[i], mine, opponent):
            scores[i] += 900000
        else:
            scores[i] = min(scores[i], best_non_attack - 1)


def _promotion_attach_is_real(card: Pokemon, hand: Counter) -> bool:
    if card.id != MUNKIDORI:
        return False
    if _psychic_count(card) <= 0:
        return hand[PRISM_ENERGY] > 0 and _munkidori_ready(card, PRISM_ENERGY)
    return (hand[DARK_ENERGY] + hand[PRISM_ENERGY] + hand[PSYCHIC_ENERGY]) > 0


def _can_promote_and_evolve_grimms(card: Pokemon, hand: Counter) -> bool:
    return bool(
        not getattr(card, "appearThisTurn", False) and (
            (card.id == MORGREM and hand[GRIMMSNARL_EX] > 0) or
            (card.id == IMPIDIMP and hand[RARE_CANDY] > 0 and hand[GRIMMSNARL_EX] > 0)
        )
    )


def _apply_prism_munkidori_promotion_override(scores: list[int], obs: Observation, mine, hand: Counter) -> None:
    """Use the exact post-KO Prism tempo line learned from replay 89608633.

    When no offered Pokemon can attack immediately, a Basic Munkidori with one
    Darkness Energy becomes a legal Mind Bend attacker after attaching the
    visible Prism Energy.  Prefer it over a zero-Energy setup Pokemon.  The
    conditions are deliberately exact so this does not become a broad promotion
    preference or interfere with already-ready attackers.
    """
    select = obs.select
    if (
        select.context != SelectContext.TO_ACTIVE
        or (mine.active or [])
        or hand[PRISM_ENERGY] <= 0
        or obs.current.energyAttached
    ):
        return
    me = obs.current.yourIndex
    offered: list[tuple[int, Pokemon]] = []
    for i, option in enumerate(select.option):
        if option.type != OptionType.CARD:
            continue
        card = _get_card(obs, option.area, option.index, option.playerIndex if option.playerIndex is not None else me)
        if isinstance(card, Pokemon):
            offered.append((i, card))
    if not offered or any(_ready_to_attack(card) for _, card in offered):
        return
    candidates = [
        i for i, card in offered
        if card.id == MUNKIDORI
        and _energy_count(card) == 1
        and _dark_count(card) == 1
        and not _has_prism(card)
        and _munkidori_ready(card, PRISM_ENERGY)
    ]
    if not candidates:
        return
    best = max(candidates, key=lambda i: (scores[i], -i))
    # The replay line is a guaranteed same-turn attack conversion, not a small
    # preference.  Keep the intervention confined to this exact forced-promotion state.
    scores[best] += 90000


def _apply_post_retreat_attack_completion(scores: list[int], obs: Observation, mine, archetype: str) -> None:
    """Finish a retreat line with a legal Impidimp/Morgrem attack when idle is next.

    Replay 89635610 retreated a stranded Munkidori into a one-Energy Impidimp,
    then selected END despite two legal attacks.  Keep useful play/evolution/
    ability/attachment actions ahead of the attack, and intervene only when the
    pending retreat occurred this turn and no productive setup action remains.
    """
    state = obs.current
    select = obs.select
    if select.context != SelectContext.MAIN or _MEMORY["retreat_followup_turn"] != state.turn:
        return
    active = mine.active[0] if mine.active else None
    if (
        active is None
        or active.id not in {IMPIDIMP, MORGREM}
        or archetype == "crustle"
        or not state.retreated
        or mine.confused
    ):
        return
    attack_indices = [i for i, option in enumerate(select.option) if option.type == OptionType.ATTACK]
    if not attack_indices:
        return
    productive_types = {OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY, OptionType.ATTACH}
    productive = [
        i for i, option in enumerate(select.option)
        if option.type in productive_types and scores[i] >= 6000
    ]
    if productive:
        return
    best_attack = max(attack_indices, key=lambda i: (scores[i], -i))
    scores[best_attack] += 950000


def _apply_post_ko_promotion_override(scores: list[int], obs: Observation, mine, hand: Counter, archetype: str) -> None:
    """Replace only one proven inefficient v30 promotion pattern.

    The override is considered after ordinary and learned scores. It does nothing
    unless v30's current top choice is a one-Energy Munkidori that cannot attack
    now or after any visible attachment, while another offered card can evolve
    into Grimmsnarl ex immediately.
    """
    select = obs.select
    if archetype not in {"archaludon", "crustle", "dragapult"}:
        return
    if select.context != SelectContext.TO_ACTIVE or (mine.active or []) or not scores:
        return
    top = max(range(len(scores)), key=lambda i: (scores[i], -i))
    top_opt = select.option[top]
    if top_opt.type != OptionType.CARD:
        return
    me = obs.current.yourIndex
    top_card = _get_card(obs, top_opt.area, top_opt.index, top_opt.playerIndex if top_opt.playerIndex is not None else me)
    if not isinstance(top_card, Pokemon):
        return
    bad_top = bool(
        top_card.id == MUNKIDORI and _energy_count(top_card) == 1
        and not _ready_to_attack(top_card) and not _promotion_attach_is_real(top_card, hand)
    )
    if not bad_top:
        return
    stabilizers = []
    for i, option in enumerate(select.option):
        if option.type != OptionType.CARD:
            continue
        card = _get_card(obs, option.area, option.index, option.playerIndex if option.playerIndex is not None else me)
        if isinstance(card, Pokemon) and _can_promote_and_evolve_grimms(card, hand):
            stabilizers.append(i)
    if not stabilizers:
        return
    best = max(stabilizers, key=lambda i: (scores[i], -i))
    scores[top] -= 42000
    scores[best] += 32000




def _apply_v37_exact_loss_memory(scores: list[int], obs: Observation, archetype: str) -> None:
    global _LOSS_MEMORY_V37_HITS
    if not _LOSS_MEMORY_V37 or obs.select.context != SelectContext.MAIN:
        return
    phase = _learned_phase(obs)
    fp = _residual_state_fingerprint(obs, archetype, phase)
    entry = _LOSS_MEMORY_V37.get(fp)
    if not isinstance(entry, dict) or entry.get("archetype") != archetype:
        return
    desired = entry.get("desired") or {}
    avoid = entry.get("avoid") or {}
    hit = False
    for i, option in enumerate(obs.select.option):
        subject, target = _learned_option_subject(obs, option)
        signature = {
            "type": int(option.type),
            "subject": int(subject),
            "target": int(target),
            "attack": int(getattr(option, "attackId", 0) or 0),
        }
        if signature == desired:
            scores[i] += 260000
            hit = True
        elif signature == avoid:
            scores[i] -= 130000
    if hit:
        _LOSS_MEMORY_V37_HITS += 1


def _strategy(obs: Observation) -> list[int]:
    state = obs.current
    select = obs.select
    me = state.yourIndex
    mine = state.players[me]
    opponent = state.players[1 - me]
    archetype = _update_memory(obs)
    plan = _make_plan(obs, archetype)
    field = Counter(p.id for _, _, p in _field(mine))
    hand = Counter(c.id for c in (mine.hand or []))
    discard = Counter(c.id for c in (mine.discard or []))
    source = _source_id(select)

    if select.context == SelectContext.IS_FIRST:
        scores = [1000 if o.type == OptionType.YES else 0 for o in select.option]
        return _choose(scores, select)

    if select.context in {SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT}:
        # Use Punk Up, Adrena-Brain, useful stadium/search effects; decline only when no board value.
        yes_value = 1000
        if source == GRIMMSNARL_EX:
            # v30 efficiency candidate: decline Punk Up when no Marnie's
            # Pokémon can use another Darkness Energy.
            deficits = sum(max(0, 2 - _dark_count(p)) for _, _, p in _field(mine) if p.id in MARNIE_LINE)
            yes_value = 10000 if deficits > 0 else -1000
        elif source == MUNKIDORI:
            yes_value = 9000 if sum(_damage_on(p) for _, _, p in _field(mine)) >= 10 else -1000
        elif source in {SPIKEMUTH_GYM, FROSLASS}:
            yes_value = 8000
        scores = [yes_value if o.type == OptionType.YES else 0 for o in select.option]
        return _choose(scores, select)

    if select.context == SelectContext.MULLIGAN:
        scores = [0 if o.type == OptionType.NO else -100 for o in select.option]
        return _choose(scores, select)

    scores: list[int] = []
    if select.context == SelectContext.MAIN:
        scores = [_main_score(obs, o, mine, opponent, field, hand, discard, plan) for o in select.option]
    elif select.type == 6 or select.context == SelectContext.ATTACK:
        active = mine.active[0] if mine.active else None
        scores = [_attack_score(o.attackId, active, opponent, plan) if o.type == OptionType.ATTACK else NEG for o in select.option]
    elif select.type == 8:
        # Counts: maximize legal useful effect except preserve low-deck draw.
        for o in select.option:
            n = int(o.number or 0)
            score = n * 100
            if select.context == SelectContext.DRAW_COUNT and mine.deckCount <= n + 2:
                score -= 10000
            scores.append(score)
    elif select.type == 9:
        scores = [100 if o.type == OptionType.YES else 0 for o in select.option]
    elif select.type == 4:
        # Energy payment/discard: prefer smallest sufficient option count.
        remain = int(getattr(select, "remainEnergyCost", 0) or 0)
        for o in select.option:
            count = int(getattr(o, "count", 1) or 1)
            scores.append(1000 - abs(remain - count) * 100 - count)
    elif select.type == 5:
        scores = [1000 if getattr(o, "cardId", None) in {MUNKIDORI, GRIMMSNARL_EX, FROSLASS, SPIKEMUTH_GYM, CHI_YU} else 0 for o in select.option]
    else:
        scores = [_card_option_score(obs, o, mine, opponent, field, hand, discard, plan, source) if o.type == OptionType.CARD else 0 for o in select.option]

    learned = [_learned_bonus(obs, option, archetype) + _residual_bonus(obs, option, archetype) + _counterfactual_bonus(obs, option, archetype) for option in select.option]
    if learned:
        ordered = sorted(learned)
        center = ordered[len(ordered) // 2]
        adjusted = []
        for score, bonus in zip(scores, learned):
            delta = bonus - center
            if select.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
                delta = max(-3500, min(3500, delta))
            adjusted.append(score + delta)
        scores = adjusted

    _apply_v37_exact_loss_memory(scores, obs, archetype)
    _apply_prism_munkidori_promotion_override(scores, obs, mine, hand)
    _apply_post_ko_promotion_override(scores, obs, mine, hand, archetype)

    if select.context == SelectContext.MAIN:
        _apply_turn_order(scores, select, obs, mine, opponent, archetype)
        _apply_post_retreat_attack_completion(scores, obs, mine, archetype)

    forced_max = _forced_punk_up_count(obs, mine, field, plan, source)
    result = _choose(scores, select, forced_max)
    if result:
        chosen_types = {select.option[i].type for i in result if 0 <= i < len(select.option)}
        if OptionType.RETREAT in chosen_types:
            _MEMORY["retreat_followup_turn"] = state.turn
        if OptionType.ATTACK in chosen_types:
            _MEMORY["retreat_followup_turn"] = -1
    if select.context == SelectContext.SETUP_ACTIVE_POKEMON and result:
        chosen = select.option[result[0]]
        selected_card = _get_card(
            obs,
            chosen.area,
            chosen.index,
            chosen.playerIndex if chosen.playerIndex is not None else me,
        )
        _MEMORY["setup_active_id"] = getattr(selected_card, "id", None)
    return result


# Exactly one public callable must remain last for Kaggle's raw loader.
def agent(observation: dict) -> list[int]:
    if observation.get("select") is None:
        _reset_memory()
        return MY_DECK
    try:
        obs = to_observation_class(observation)
        result = _strategy(obs)
        return _validate_result(result, obs.select)
    except Exception:
        # Submission safety: always return a legal deterministic fallback.
        select = observation.get("select") or {}
        options = select.get("option") or []
        min_count = int(select.get("minCount", 0) or 0)
        max_count = int(select.get("maxCount", min_count) or min_count)
        if not options:
            return []
        count = max(min_count, min(max_count, 1 if min_count == 0 else min_count))
        return list(range(min(count, len(options))))
