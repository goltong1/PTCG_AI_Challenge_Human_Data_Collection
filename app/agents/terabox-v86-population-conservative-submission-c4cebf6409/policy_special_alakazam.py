from __future__ import annotations
import os
from specialist_v19 import build
R=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
_MODEL=build(R,'marnie','alakazam')
for _name,_value in _MODEL.__dict__.items():
    if not _name.startswith('__'):
        globals()[_name]=_value
MY_DECK=_MODEL.MY_DECK
def agent(observation):return _MODEL.agent(observation)
