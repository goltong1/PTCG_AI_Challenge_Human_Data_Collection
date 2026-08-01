from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def load_agent(agent_dir: Path, module_name: str) -> ModuleType:
    """Load an agent's main.py while resolving deck.csv relative to the agent folder."""
    agent_dir = agent_dir.resolve()
    main_path = agent_dir / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(main_path)

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {main_path}")

    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    os.chdir(agent_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def read_deck(agent_dir: Path) -> list[int]:
    deck_path = agent_dir / "deck.csv"
    values = [int(x) for x in deck_path.read_text(encoding="utf-8").split()]
    if len(values) != 60:
        raise ValueError(f"Deck must contain 60 cards, got {len(values)}: {deck_path}")
    return values
