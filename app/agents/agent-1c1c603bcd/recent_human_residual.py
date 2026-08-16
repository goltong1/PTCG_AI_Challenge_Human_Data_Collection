"""Conservative game-level support residual trained from the user's newest 37 replays.
Only exact public matchup/turn/action signatures with repeated game-level evidence
can change the frozen v157 action.  This layer never learns a losing replay as a
positive target and is intentionally bounded to one intervention per game.
"""
from __future__ import annotations
import json,os
from cg.api import AreaType,OptionType,SelectContext,to_observation_class

def _i(x,d=0):
    try:return int(x if x is not None else d)
    except:return d

def _tb(t):return 'early' if _i(t)<=3 else 'mid' if _i(t)<=7 else 'late'

def _field_ids(p):
    out=set()
    for a in ('active','bench','discard'):
        for q in list(getattr(p,a,None) or []):
            if q is not None:out.add(_i(getattr(q,'id',0)))
    return out

def _family(obs,plan=None):
    a=str(getattr(plan,'archetype','') or '')
    if a and a!='unknown':return a
    try:
        me=obs.current.yourIndex;ids=_field_ids(obs.current.players[1-me])
    except:return 'unknown'
    for n,s in [('crustle',{344,345}),('dragapult',{119,120,121}),('alakazam',{741,742,743,245}),('archaludon',{169,190,666}),('marnie',{646,647,648}),('cynthia',{379,380,381,341,342})]:
        if ids&s:return n
    if ids&{333,677,678}:return 'lucario'
    return 'unknown'

def _card(obs,o,me):
    try:
        if o.type==OptionType.PLAY:
            arr=obs.current.players[me].hand or [];return _i(arr[o.index].id) if 0<=o.index<len(arr) and arr[o.index] is not None else 0
        area=getattr(o,'area',None);idx=_i(getattr(o,'index',-1));pidx=_i(getattr(o,'playerIndex',me),me)
        if area is None:return 0
        p=obs.current.players[pidx]
        mp={AreaType.DECK:getattr(p,'deck',None),AreaType.HAND:getattr(p,'hand',None),AreaType.DISCARD:getattr(p,'discard',None),AreaType.ACTIVE:getattr(p,'active',None),AreaType.BENCH:getattr(p,'bench',None),AreaType.PRIZE:getattr(p,'prize',None)}
        arr=mp.get(area) or []
        return _i(arr[idx].id) if 0<=idx<len(arr) and arr[idx] is not None else 0
    except:return 0

def _target(obs,o,me):
    try:
        ia=getattr(o,'inPlayArea',None);ii=_i(getattr(o,'inPlayIndex',-1));pidx=_i(getattr(o,'playerIndex',me),me)
        if ia is None:return 0
        p=obs.current.players[pidx];arr=(p.active or []) if ia==AreaType.ACTIVE else (p.bench or []) if ia==AreaType.BENCH else []
        return _i(arr[ii].id) if 0<=ii<len(arr) and arr[ii] is not None else 0
    except:return 0

def _sig(obs,o,me):return f'{_i(o.type,-1)}:{_card(obs,o,me)}:{_target(obs,o,me)}:{_i(getattr(o,"attackId",0))}'

class RecentHumanResidual:
    def __init__(self,root):
        try:self.model=json.load(open(os.path.join(root,'recent_human_support_model.json'),encoding='utf8'))
        except Exception:self.model={'support':{},'gate':{}}
        self.support=self.model.get('support') or {};self.gate=self.model.get('gate') or {};self.reset()
    def reset(self):self.overrides=0;self.last_turn=-1;self.stats={'calls':0,'eligible':0,'overrides':0,'base_risk':0,'candidate_support':0,'families':{},'reasons':{}}
    @staticmethod
    def _rate(r):
        w=_i((r or {}).get('win_games'));l=_i((r or {}).get('loss_games'));return (w+1.0)/(w+l+2.0)
    def choose(self,obs_dict,base,plan=None):
        self.stats['calls']+=1
        try:obs=to_observation_class(obs_dict)
        except Exception:return base
        if obs.current is None or obs.select is None or obs.select.context!=SelectContext.MAIN or len(base)!=1:return base
        if self.overrides>=_i(self.gate.get('max_game_overrides'),1):return base
        opts=list(obs.select.option or []);bi=_i(base[0],-1)
        if not 0<=bi<len(opts) or len(opts)<2:return base
        fam=_family(obs,plan);allowed=set(self.gate.get('allowed_families') or [])
        if fam not in allowed:return base
        turn=_i(obs.current.turn);tb=_tb(turn);me=obs.current.yourIndex;bo=opts[bi];bs=_sig(obs,bo,me);br=self.support.get(f'{fam}|{tb}|{bs}')
        if not br:return base
        bw=_i(br.get('win_games'));bl=_i(br.get('loss_games'));brate=self._rate(br)
        if bl<_i(self.gate.get('min_base_loss_games'),3) or brate>float(self.gate.get('max_base_rate',.35)):return base
        self.stats['base_risk']+=1
        bt=_i(bo.type,-1);cands=[]
        for i,o in enumerate(opts):
            if i==bi:continue
            ot=_i(o.type,-1)
            # Preserve phase semantics.  Only END can be replaced cross-type.
            if bt!=_i(OptionType.END) and ot!=bt:continue
            if bt==_i(OptionType.END) and ot not in {_i(OptionType.PLAY),_i(OptionType.ATTACK)}:continue
            s=_sig(obs,o,me);r=self.support.get(f'{fam}|{tb}|{s}')
            if not r:continue
            w=_i(r.get('win_games'));l=_i(r.get('loss_games'));rate=self._rate(r)
            if w<_i(self.gate.get('min_candidate_win_games'),2) or rate<float(self.gate.get('min_candidate_rate',.67)):continue
            if rate-brate<float(self.gate.get('min_rate_gap',.4)):continue
            # Do not promote Black Belt from correlation when no attack is legal.
            if _card(obs,o,me)==1211 and not any(x.type==OptionType.ATTACK for x in opts):continue
            cands.append((rate,w,-l,int(ot==_i(OptionType.ATTACK)),-i,i,s))
        if not cands:return base
        self.stats['candidate_support']+=1;cands.sort(reverse=True);pick=cands[0]
        self.overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;self.stats['families'][fam]=self.stats['families'].get(fam,0)+1
        key=f'{fam}:{tb}:{bs}->{pick[6]}';self.stats['reasons'][key]=self.stats['reasons'].get(key,0)+1
        return [pick[5]]
    def get_stats(self):return dict(self.stats)
