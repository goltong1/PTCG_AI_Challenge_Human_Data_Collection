from __future__ import annotations

"""Fast conservative sequence-conditioned IQL inference.

NumPy is used when available; a pure-Python scorer is retained as a fallback.
The model is a residual over the submitted base policy, never a source of legal
moves on its own.
"""

import array
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import Counter, deque

try:
    import numpy as np
except Exception:  # Competition fallback.
    np = None


def _resolve_root():
    f = globals().get("__file__")
    candidates = []
    if f:
        candidates.append(os.path.dirname(os.path.abspath(f)))
    candidates.extend([globals().get("_HERE"), globals().get("R"), "/kaggle_simulations/agent", os.getcwd()])
    for p in candidates:
        if p and os.path.isfile(os.path.join(p, "offline_csiql_config.json")):
            return p
    return candidates[0] if candidates else os.getcwd()


ROOT = _resolve_root()
_COMMON_PATH = os.path.join(ROOT, "offline_model_common.py")
_COMMON_NAME = "_csiql_common_" + hashlib.sha1(ROOT.encode()).hexdigest()[:12]
_sp = importlib.util.spec_from_file_location(_COMMON_NAME, _COMMON_PATH)
if _sp is None or _sp.loader is None:
    raise RuntimeError(_COMMON_PATH)
C = importlib.util.module_from_spec(_sp); sys.modules[_COMMON_NAME] = C; _sp.loader.exec_module(C)
try:
    from cg.api import all_attack, all_card_data
    C.configure_metadata(all_card_data(), all_attack())
except Exception:
    pass

CFG = json.load(open(os.path.join(ROOT, "offline_csiql_config.json"), encoding="utf-8"))
DIM = int(CFG["dim"]); HIDDEN = int(CFG["hidden"]); E = int(CFG["ensemble"]); MASK = DIM - 1
HISTORY_LEN = int(CFG.get("history_len", 6))
SUPPORT = CFG.get("support") or {}
WEIGHTS_PATH = os.path.join(ROOT, CFG.get("weights_file", "offline_csiql_embeddings.bin"))
_EXPECTED = 2 * E * DIM * HIDDEN

if np is not None:
    _NP_EMB = np.fromfile(WEIGHTS_PATH, dtype="<i2")
    if int(_NP_EMB.size) != _EXPECTED:
        raise ValueError(f"CS-IQL embedding length {_NP_EMB.size} != {_EXPECTED}")
    _NP_EMB = _NP_EMB.reshape(2, E, DIM, HIDDEN)
    _NP_SCALE = (
        np.asarray(CFG["actor_scales"], dtype=np.float32),
        np.asarray(CFG["q_scales"], dtype=np.float32),
    )
    def _np_heads(kind):
        return (
            np.asarray([h[kind]["fc_weight"] for h in CFG["heads"]], dtype=np.float32),
            np.asarray([h[kind]["fc_bias"] for h in CFG["heads"]], dtype=np.float32),
            np.asarray([h[kind]["out_weight"] for h in CFG["heads"]], dtype=np.float32),
            np.asarray([h[kind]["out_bias"] for h in CFG["heads"]], dtype=np.float32),
        )
    _NP_HEAD = (_np_heads("actor"), _np_heads("q"))
    ACTOR = Q = None
else:
    _buf = array.array("h")
    with open(WEIGHTS_PATH, "rb") as f:
        _buf.frombytes(f.read())
    if sys.byteorder != "little":
        _buf.byteswap()
    if len(_buf) != _EXPECTED:
        raise ValueError(f"CS-IQL embedding length {len(_buf)} != {_EXPECTED}")
    stride = DIM * HIDDEN
    ACTOR = [_buf[i * stride:(i + 1) * stride] for i in range(E)]
    Q = [_buf[(E + i) * stride:(E + i + 1) * stride] for i in range(E)]
    _NP_SCALE = (tuple(float(x) for x in CFG["actor_scales"]), tuple(float(x) for x in CFG["q_scales"]))
    _NP_HEAD = None

HEADS = CFG["heads"]


def _h(ns: int, *vals: int):
    idx, sign = C.hashed(ns, *vals)
    return int(idx) & MASK, int(sign)


def _turn(raw):
    return int(((raw or {}).get("current") or {}).get("turn") or 0)


def _context(raw):
    sel = (raw or {}).get("select") or {}
    return int(sel.get("context", -1) or -1), int(sel.get("type", -1) or -1)


def _action_semantics(raw, action):
    if not action:
        return [C.semantic_option(raw, None)]
    opts = ((raw or {}).get("select") or {}).get("option") or []
    out = []
    for i in action[:2]:
        if isinstance(i, int) and 0 <= i < len(opts):
            out.append(C.semantic_option(raw, opts[i]))
    return out or [C.semantic_option(raw, None)]


def _history_features(history, current_turn, candidate_sem=None):
    out = []
    recent = list(reversed(list(history)[-HISTORY_LEN:]))
    type_counts = Counter()
    for lag, item in enumerate(recent, 1):
        dt = max(0, min(10, current_turn - int(item.get("turn", current_turn))))
        ctx = int(item.get("context", -1)); stype = int(item.get("stype", -1)); length = int(item.get("length", 0))
        out.append(_h(180, lag, dt, ctx + 3, stype + 3, min(length, 3)))
        for j, sem in enumerate((item.get("semantics") or [])[:2]):
            typ, src, tgt, aid, num, special, area = tuple(int(x) for x in sem)
            type_counts[typ] += 1
            out.append(_h(181, lag, j, typ + 3, src, tgt, aid))
            out.append(_h(182, lag, typ + 3, src, aid))
            out.append(_h(183, lag, typ + 3, ctx + 3, dt))
    for typ, count in type_counts.items():
        out.append(_h(184, typ + 3, min(count, 4)))
    if candidate_sem is not None and recent:
        ctyp, csrc, ctgt, caid, *_ = tuple(int(x) for x in candidate_sem)
        psem = (recent[0].get("semantics") or [(-2, 0, 0, 0, 0, -1, -1)])[0]
        ptyp, psrc, ptgt, paid, *_ = tuple(int(x) for x in psem)
        out.append(_h(185, ctyp + 3, csrc, caid, ptyp + 3, psrc, paid))
        out.append(_h(186, ctyp + 3, ctgt, ptyp + 3, ptgt))
    return out


def _action_bag(raw, action, history):
    opt = C.option_for_action(raw, action)
    sem = C.semantic_option(raw, opt)
    feats = [((int(i) & MASK), int(s)) for i, s in C.action_features(raw, opt, include_state=True)]
    feats.extend(_history_features(history, _turn(raw), sem))
    return feats


def _score_np(kind, features):
    if not features:
        return _NP_HEAD[kind][3].copy()
    idx = np.fromiter((int(i) & MASK for i, _ in features), dtype=np.int64, count=len(features))
    signs = np.fromiter((float(s) for _, s in features), dtype=np.float32, count=len(features))
    x = (_NP_EMB[kind][:, idx, :].astype(np.float32) * signs[None, :, None]).sum(axis=1)
    x *= _NP_SCALE[kind][:, None] / math.sqrt(max(1, len(features)))
    fw, fb, ow, ob = _NP_HEAD[kind]
    y = np.tanh(np.einsum("eoi,ei->eo", fw, x, optimize=True) + fb)
    return (y * ow).sum(axis=1) + ob


def _head_score_py(vec, scale, head, features):
    x = [0.0] * HIDDEN
    for idx, sign in features:
        base = (int(idx) & MASK) * HIDDEN; sg = int(sign)
        for j in range(HIDDEN):
            x[j] += int(vec[base + j]) * sg
    norm = float(scale) / math.sqrt(max(1, len(features)))
    for j in range(HIDDEN):
        x[j] *= norm
    w = head["fc_weight"]; b = head["fc_bias"]; y = []
    for r in range(HIDDEN):
        z = float(b[r]); wr = w[r]
        for j in range(HIDDEN): z += float(wr[j]) * x[j]
        y.append(math.tanh(z))
    z = float(head["out_bias"]); ow = head["out_weight"]
    for j in range(HIDDEN): z += float(ow[j]) * y[j]
    return z


def _scores(kind, features):
    if np is not None:
        return _score_np(kind, features).tolist()
    vecs = ACTOR if kind == 0 else Q
    scales = _NP_SCALE[kind]
    key = "actor" if kind == 0 else "q"
    return [_head_score_py(vecs[m], scales[m], HEADS[m][key], features) for m in range(E)]


def _obs_key(raw):
    cur = (raw or {}).get("current") or {}; sel = (raw or {}).get("select") or {}
    opts = sel.get("option") or []; sems = []
    for o in opts[:24]:
        try: sems.append(C.semantic_option(raw, o))
        except Exception: sems.append((-9, 0, 0, 0, 0, -1, -1))
    return (
        int((raw or {}).get("step", -1) or -1), int(cur.get("turn", -1) or -1), int(cur.get("yourIndex", -1) or -1),
        int(sel.get("context", -1) or -1), int(sel.get("type", -1) or -1), int(sel.get("minCount", 0) or 0),
        int(sel.get("maxCount", 0) or 0), int(C.context_source(raw)), tuple(sems),
    )


def _visible_opponent_ids(raw):
    """Return only card ids already visible in the opponent public zones."""
    cur, ps, me = C._players(raw)
    if len(ps) < 2:
        return set()
    opp = ps[1 - me]
    ids = set()
    for zone in ("active", "bench", "discard"):
        for card in C._cards(opp.get(zone)):
            cid = C._cid(card)
            if cid:
                ids.add(cid)
            for pre in C._cards(card.get("preEvolution")):
                pcid = C._cid(pre)
                if pcid:
                    ids.add(pcid)
    return ids


def _deployment_scope(raw):
    """Check an optional public-evidence deployment scope.

    This gate is intentionally separate from C.public_matchup(): the latter is
    part of the learned feature representation and preserves its training-time
    precedence.  Deployment can therefore recognize an exact deck signature
    without changing any model feature.
    """
    any_ids = {int(x) for x in (CFG.get("allowed_public_any_ids") or [])}
    all_groups = CFG.get("allowed_public_all_groups") or []
    if any_ids or all_groups:
        ids = _visible_opponent_ids(raw)
        if ids & any_ids:
            return True, {"public_ids": sorted(ids), "scope_evidence": sorted(ids & any_ids)}
        for group in all_groups:
            group_ids = {int(x) for x in (group or [])}
            if group_ids and group_ids.issubset(ids):
                return True, {"public_ids": sorted(ids), "scope_evidence": sorted(group_ids)}
        return False, {"public_ids": sorted(ids), "scope_evidence": []}
    allowed = CFG.get("allowed_matchup_names") or []
    if allowed:
        fam = int(C.public_matchup(raw))
        allowed_ids = {int(C.MATCHUP_ID.get(str(name), -999)) for name in allowed}
        return fam in allowed_ids, {"public_matchup": fam}
    return True, {}


def _support(raw, action):
    opt = C.option_for_action(raw, action)
    typ, src, tgt, aid, *_ = C.semantic_option(raw, opt)
    sel = raw.get("select") or {}; ctx = int(sel.get("context", -1)); stype = int(sel.get("type", -1)); fam = int(C.public_matchup(raw))
    keys = (
        f"{fam}|{ctx}|{stype}|{typ}|{src}|{aid}", f"{fam}|{ctx}|*|{typ}|{src}|{aid}",
        f"{fam}|{ctx}|*|{typ}|{src}|*", f"{fam}|*|*|{typ}|{src}|*",
        f"*|{ctx}|*|{typ}|{src}|{aid}", f"*|{ctx}|*|{typ}|*|*",
    )
    return max([int(SUPPORT.get(k, 0)) for k in keys] + [0])


class OfflineCSIQLPolicy:
    def __init__(self, root=None):
        self.root = root or ROOT
        self.history = deque(maxlen=HISTORY_LEN)
        self.last_key = None; self.last_choice = None; self.last_step = -1
        self.stats = Counter(); self.last_diagnosis = None

    def reset(self):
        self.history.clear(); self.last_key = None; self.last_choice = None; self.last_step = -1
        self.stats = Counter(); self.last_diagnosis = None

    def get_stats(self):
        out = dict(self.stats); out["history_len"] = len(self.history); out["numpy"] = bool(np is not None)
        if self.last_diagnosis:
            out["last"] = {k: v for k, v in self.last_diagnosis.items() if k not in ("actor_scores", "q_scores", "combined_scores")}
        return out

    def diagnose(self, raw, base):
        in_scope, scope_info = _deployment_scope(raw)
        if not in_scope:
            out = {"eligible": False, "reason": "outside_deployment_matchup"}
            out.update(scope_info)
            return out
        candidates = C.candidates_for(raw)
        if not candidates or len(candidates) <= 1 or base not in candidates:
            return {"eligible": False, "reason": "unsupported_selection"}
        bags = [_action_bag(raw, a, self.history) for a in candidates]
        actor_scores = [None] * len(candidates); q_scores = [None] * len(candidates)
        for i, bag in enumerate(bags):
            actor_scores[i] = _scores(0, bag); q_scores[i] = _scores(1, bag)
        # candidate-major -> model-major
        actor_scores = [[actor_scores[i][m] for i in range(len(candidates))] for m in range(E)]
        q_scores = [[q_scores[i][m] for i in range(len(candidates))] for m in range(E)]
        base_i = candidates.index(base); combined = []; votes = []
        qcoef = float(CFG.get("q_coefficient", 0.55)); bonus = float(CFG.get("base_bonus", 0.18))
        for m in range(E):
            zz = [actor_scores[m][i] + qcoef * q_scores[m][i] + (bonus if i == base_i else 0.0) for i in range(len(candidates))]
            combined.append(zz); votes.append(max(range(len(zz)), key=lambda i: zz[i]))
        vote_counts = Counter(votes)
        winner = max(range(len(candidates)), key=lambda i: (vote_counts.get(i, 0), sum(z[i] for z in combined) / E))
        adiff = [actor_scores[m][winner] - actor_scores[m][base_i] for m in range(E)]
        qdiff = [q_scores[m][winner] - q_scores[m][base_i] for m in range(E)]
        cdiff = [combined[m][winner] - combined[m][base_i] for m in range(E)]
        qmean = sum(qdiff) / E; qsd = statistics.pstdev(qdiff) if E > 1 else 0.0
        return {
            "eligible": True, "candidates": candidates, "base_index": base_i, "winner_index": winner,
            "winner_action": candidates[winner], "votes": int(vote_counts.get(winner, 0)),
            "actor_margin": sum(adiff) / E, "q_margin": qmean,
            "q_lower_margin": qmean - float(CFG.get("risk_z", 0.75)) * qsd,
            "combined_margin": sum(cdiff) / E,
            "combined_uncertainty": statistics.pstdev(cdiff) if E > 1 else 0.0,
            "support": _support(raw, candidates[winner]), "model_votes": votes,
            "actor_scores": actor_scores, "q_scores": q_scores, "combined_scores": combined,
        }

    def _record(self, raw, choice):
        ctx, stype = _context(raw)
        self.history.append({"turn": _turn(raw), "context": ctx, "stype": stype,
            "length": len(choice) if isinstance(choice, list) else 0,
            "semantics": _action_semantics(raw, choice if isinstance(choice, list) else [])})

    def choose(self, raw, base):
        if not isinstance(raw, dict) or (raw.get("current") is None and raw.get("select") is None):
            self.reset(); return base
        key = _obs_key(raw); step = int(raw.get("step", -1) or -1)
        if self.last_key == key and self.last_choice is not None:
            # A repeated engine call is not a new decision.  Return the freshly
            # computed retained-baseline choice and do not append history twice.
            self.stats["cached_calls"] += 1; return base
        if self.last_step >= 0 and step >= 0 and step < self.last_step: self.reset()
        self.stats["calls"] += 1
        try: d = self.diagnose(raw, base)
        except Exception: self.stats["diagnose_error"] += 1; d = {"eligible": False, "reason": "exception"}
        choice = base
        if d.get("eligible"):
            self.stats["eligible"] += 1; candidate = d["winner_action"]
            if candidate != base:
                if d["support"] < int(CFG.get("min_support", 3)): self.stats["blocked_support"] += 1
                elif d["votes"] < int(CFG.get("min_votes", 4)): self.stats["blocked_vote"] += 1
                elif d["actor_margin"] < float(CFG.get("actor_margin", 0.12)): self.stats["blocked_actor_margin"] += 1
                elif d["combined_margin"] < float(CFG.get("combined_margin", 0.16)): self.stats["blocked_combined_margin"] += 1
                elif d["q_lower_margin"] < float(CFG.get("q_lower_margin", 0.015)): self.stats["blocked_q"] += 1
                elif d["combined_uncertainty"] > float(CFG.get("max_combined_uncertainty", 0.35)): self.stats["blocked_uncertainty"] += 1
                else: choice = candidate; self.stats["overrides"] += 1
            else: self.stats["model_kept_base"] += 1
        else: self.stats["unsupported"] += 1
        self._record(raw, choice); self.last_key = key
        self.last_choice = list(choice) if isinstance(choice, list) else choice; self.last_step = step; self.last_diagnosis = d
        return choice


_DEFAULT_POLICY = OfflineCSIQLPolicy(ROOT)
def reset(): _DEFAULT_POLICY.reset()
def diagnose(raw, base): return _DEFAULT_POLICY.diagnose(raw, base)
def choose(raw, base): return _DEFAULT_POLICY.choose(raw, base)
def diagnostics():
    out = _DEFAULT_POLICY.get_stats(); out.update({"loaded": True, "schema": str(CFG.get("schema", "offline_csiql_runtime_v1"))}); return out
