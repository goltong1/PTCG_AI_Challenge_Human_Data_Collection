from __future__ import annotations
import os
from memory_policy_v19 import build
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd();_MODEL=build(R,'crustle');MY_DECK=_MODEL.MY_DECK
def agent(observation):return _MODEL.agent(observation)
