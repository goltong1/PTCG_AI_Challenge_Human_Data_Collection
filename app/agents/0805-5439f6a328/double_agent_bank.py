from __future__ import annotations

import importlib.util
import inspect
import os
import random
import sys
import types
from collections import Counter, defaultdict
from copy import deepcopy


class OpponentPolicyHandle:
    """Isolated, lazily loaded deck-policy module used only inside search.

    The real submitted policy is called on the simulated opponent observation.  Its
    small mutable globals are snapshotted/restored for every candidate branch, so
    one hypothetical line cannot leak state into another line or the real game.
    """

    def __init__(self, root: str, name: str):
        self.root = root
        self.name = name
        self.module = self._load()
        self._initial = self.snapshot()
        self.calls = 0
        self.errors = 0

    def _load(self):
        path = os.path.join(self.root, self.name, "deck_policy.py")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        mod_name = f"_opponent_policy_{self.name}_{abs(hash(path)) & 0xffffffff:x}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        old_path = list(sys.path)
        old_cwd = os.getcwd()
        try:
            sys.path.insert(0, os.path.dirname(path))
            os.chdir(os.path.dirname(path))
            assert spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
            os.chdir(old_cwd)
        if not callable(getattr(module, "agent", None)):
            raise TypeError(f"{self.name}: deck_policy.agent missing")
        return module

    @staticmethod
    def _copyable(value):
        if isinstance(value, (types.ModuleType, type)) or inspect.isfunction(value) or inspect.ismethod(value):
            return False
        if isinstance(value, (int, float, str, bool, type(None), list, tuple, dict, set, Counter, defaultdict)):
            return True
        return value.__class__.__name__ in {"AttackPlan", "Forecast", "Plan", "MachineState"}

    def snapshot(self):
        state = {}
        for key, value in self.module.__dict__.items():
            if key.startswith("__") or key.isupper() or key in {"CARDS", "ATTACKS", "CARD_DB", "ALL_ATTACKS", "MY_DECK"}:
                continue
            if not self._copyable(value):
                continue
            try:
                state[key] = deepcopy(value)
            except Exception:
                pass
        return state

    def restore(self, state=None):
        for key, value in (self._initial if state is None else state).items():
            try:
                setattr(self.module, key, deepcopy(value))
            except Exception:
                pass

    def act(self, observation: dict):
        self.calls += 1
        rng = random.getstate()
        old_path = list(sys.path)
        old_cwd = os.getcwd()
        try:
            sys.path.insert(0, os.path.join(self.root, self.name))
            os.chdir(os.path.join(self.root, self.name))
            return self.module.agent(observation)
        except Exception:
            self.errors += 1
            return []
        finally:
            random.setstate(rng)
            sys.path[:] = old_path
            os.chdir(old_cwd)


class OpponentPolicyBank:
    def __init__(self, root: str, mapping: dict[str, str] | None = None):
        self.root = root
        self.mapping = dict(mapping or {})
        self._handles: dict[str, OpponentPolicyHandle | None] = {}

    def available(self, archetype: str | None) -> bool:
        if not archetype:
            return False
        folder = self.mapping.get(archetype, archetype)
        return os.path.exists(os.path.join(self.root, folder, "deck_policy.py"))

    def get(self, archetype: str | None):
        if not archetype:
            return None
        if archetype not in self._handles:
            folder = self.mapping.get(archetype, archetype)
            try:
                self._handles[archetype] = OpponentPolicyHandle(self.root, folder)
            except Exception:
                self._handles[archetype] = None
        return self._handles[archetype]

    def stats(self):
        out = {}
        for name, handle in self._handles.items():
            if handle is not None:
                out[name] = {"calls": handle.calls, "errors": handle.errors}
        return out
