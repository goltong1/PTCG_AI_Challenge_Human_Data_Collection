from pathlib import Path
import sys


def _find_submission_root() -> Path:
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.extend([Path("/kaggle_simulations/agent"), Path.cwd()])
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            pass
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "cg" / "hybrid_runtime" / "search.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate cg/hybrid_runtime/search.json")


_SUBMISSION_ROOT = _find_submission_root()
_RUNTIME_ROOT = _SUBMISSION_ROOT / "cg" / "hybrid_runtime"
for _path in (_SUBMISSION_ROOT, _RUNTIME_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cabt_hybrid.hybrid_agent import HybridAgent

_runtime = HybridAgent(_RUNTIME_ROOT, _RUNTIME_ROOT / "search.json")


def agent(obs_dict):
    return _runtime.act(obs_dict)
