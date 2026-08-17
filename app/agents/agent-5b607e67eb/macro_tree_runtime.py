from __future__ import annotations
import json,math,os
from dataclasses import dataclass
from typing import Dict
from cg.api import all_card_data,all_attack

CARD={c.cardId:c for c in all_card_data()}; ATT={a.attackId:a for a in all_attack()}
RI={333,677};LUC=678;SOL=676;LUN=675;DUN=305;DUD=66;DUX=306;OG=117;ROCK=20
HILDA=1225;POFFIN=1086;GONG=1142;LILLIE=1227;XERO=1197;JUDGE=1213;BOSS=1182
KEYHAND={1141:'ppp',GONG:'gong',1152:'pad',POFFIN:'poffin',JUDGE:'judge',XERO:'xero',LILLIE:'lillie',HILDA:'hilda',BOSS:'boss',1252:'mountain',1211:'belt'}
BRIDGES={
 'marnie':{646,647},'alakazam':{741,742},'dragapult':{119,120},'archaludon':{169},'crustle':{344},
}

@dataclass
class TreeDecision:
    mode:str='BASELINE'
    score:float=0.0
    base_value:float=0.5
    margin:float=0.0
    pv:tuple[str,...]=()
    turn:int=-1

class MacroValue:
    def __init__(self,root:str):
        p=os.path.join(root,'macro_value_model.json')
        try:self.model=json.load(open(p))
        except Exception:self.model={'intercept':0.0,'numeric':{},'arch':{}}
    def value(self,x:Dict[str,float],arch:str)->float:
        if x.get('my_prize',1)<=0:return .995
        if x.get('opp_prize',1)<=0:return .005
        z=float(self.model.get('intercept',0.0))+float(self.model.get('arch',{}).get(arch,0.0))
        for k,w in self.model.get('numeric',{}).items():z+=float(w)*float(x.get(k,0.0) or 0.0)
        if z>12:return .999994
        if z<-12:return .000006
        return 1.0/(1.0+math.exp(-z))

class MacroTreePlanner:
    """Depth-3 public-state strategic lookahead.

    This is intentionally an *abstract* tree, not a fake perfect-information card rollout.
    Nodes are strategic states (prize race, attacker waves, disruption, bypass progress).
    Leaf values are learned from current-deck self-play plus historical replay states.
    The tactical executor still chooses the exact legal card/action.
    """
    def __init__(self,root:str,depth:int=3):
        self.value_model=MacroValue(root);self.depth=depth;self.reset()
    def reset(self):
        self.turn=-1;self.cache_key=None;self.cached=TreeDecision();self.stats={'calls':0,'modes':{},'pv':{},'margins':[]}
    def _field(self,p):return [q for q in list(p.active or [])+list(p.bench or []) if q is not None]
    def _prize(self,q):
        c=CARD.get(q.id) if q else None
        return 3 if c and c.megaEx else 2 if c and c.ex else 1
    def _maxbase(self,q):
        if q is None:return 0
        c=CARD.get(q.id);es=len(q.energies or []);best=0
        if c:
            for aid in c.attacks or []:
                a=ATT.get(aid)
                if a and len(a.energies or [])<=es:best=max(best,int(a.damage or 0))
        if q.id==DUX and es>=1:best=max(best,120)
        return best
    def features(self,obs,arch:str,history=None):
        st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me]
        mf=self._field(mine);of=self._field(op);a=mine.active[0] if mine.active else None;oa=op.active[0] if op.active else None
        luc=[q for q in mf if q.id==LUC];ri=[q for q in mf if q.id in RI]
        dux=next((q for q in mf if q.id==DUX),None);og=next((q for q in mf if q.id==OG),None)
        hand=[c.id for c in (mine.hand or [])]
        ready=sum(len(q.energies or [])>=2 for q in luc);one=sum(len(q.energies or [])==1 for q in luc)
        # ``energies`` contains EnergyType integers; card identity (and therefore
        # Rock Fighting protection) lives in ``energyCards``.
        rock=sum(any(getattr(e,'id',None)==ROCK for e in (q.energyCards or [])) for q in luc)
        x={'turn':int(st.turn or 0),'my_prize':len(mine.prize or []),'opp_prize':len(op.prize or []),'prize_adv':len(op.prize or [])-len(mine.prize or []),
           'my_hand':int(getattr(mine,'handCount',len(mine.hand or [])) or 0),'opp_hand':int(getattr(op,'handCount',len(op.hand or [])) or 0),
           'luc':len(luc),'riolu':len(ri),'luc_lines':len(luc)+len(ri),'ready_luc':ready,'one_luc':one,'rock_luc':rock,
           'sol_luna':int(any(q.id==SOL for q in mf) and any(q.id==LUN for q in mf)),'dun_base':sum(q.id==DUN for q in mf),'dud_ex':int(dux is not None),'dud_energy':len(dux.energies or []) if dux else 0,
           'oger':int(og is not None),'oger_energy':len(og.energies or []) if og else 0,
           'active_prize':self._prize(a) if a else 0,'active_energy':len(a.energies or []) if a else 0,'active_damage':self._maxbase(a),
           'opp_active_prize':self._prize(oa) if oa else 0,'opp_active_hp':int(oa.hp or 0) if oa else 0,'opp_active_energy':len(oa.energies or []) if oa else 0,'opp_active_damage':self._maxbase(oa),
           'opp_stage2':sum(bool(CARD.get(q.id) and CARD[q.id].stage2) for q in of),'opp_ex':sum(bool(CARD.get(q.id) and (CARD[q.id].ex or CARD[q.id].megaEx)) for q in of),
           'can_ko':int(bool(a and oa and self._maxbase(a)>=int(oa.hp or 0))),
           'supporter_played':int(bool(st.supporterPlayed)),'energy_attached':int(bool(st.energyAttached))}
        for cid,nm in KEYHAND.items():x['hand_'+nm]=hand.count(cid)
        # tree-only meta; not part of learned leaf model
        x['_my_active_hp']=int(a.hp or 0) if a else 0
        x['_opp_bridge']=sum(q.id in BRIDGES.get(arch,set()) for q in of)
        x['_one_prize_pivots']=sum(self._prize(q)==1 for q in mf if a is None or q.serial!=a.serial)
        x['_under_luc']=sum(q.id==LUC and len(q.energies or [])<2 for q in mf)
        x['_ready_bench_luc']=sum(q.id==LUC and len(q.energies or [])>=2 and (a is None or q.serial!=a.serial) for q in mf)
        x['_has_mega_hand']=int(LUC in hand)
        x['_has_riolu_field']=int(bool(ri))
        x['_opp_can_ko']=int(bool(a and oa and self._maxbase(oa)>=int(a.hp or 0)))
        x['_active_riolu']=int(bool(a and a.id in RI))
        x['_active_appear']=int(bool(a and getattr(a,'appearThisTurn',False)))
        # Sequence-conditioned threat and hand belief.  Opponent card identities
        # come only from public history; hidden hand cards are never fabricated.
        if history is not None:
            try:
                observed=int(history.recent_attack_damage(1-me,oa.id if oa else 0) or 0)
                known=list(history.known_hand_ids(1-me))
                x['_opp_observed_damage']=observed
                x['_opp_history_attacks']=int(history.attack_count(1-me))
                x['_own_history_attacks']=int(history.attack_count(me))
                x['_opp_known_hand']=len(known)
                x['_opp_known_evolution_hand']=sum(bool(CARD.get(cid) and (CARD[cid].stage1 or CARD[cid].stage2)) for cid in known)
                x['_opp_can_ko']=int(bool(a and oa and max(self._maxbase(oa),observed)>=int(a.hp or 0)))
                x['history_attack_pressure']=min(6,x['_opp_history_attacks'])/6.0
                x['history_known_hand_pressure']=min(4,x['_opp_known_evolution_hand'])/4.0
            except Exception:
                pass
        return x
    def _advance(self,s):
        n=dict(s);n['turn']=s.get('turn',0)+1;n['supporter_played']=0;n['energy_attached']=0;return n
    def _branches(self,s,arch):
        out=[]
        # 0) FINISH_WAVE: Hilda can turn a mature Active Riolu into Mega Lucario
        # and fetch the missing Energy in the same supporter action.  Model this as a
        # complete strategic branch rather than rating Judge/Hilda as isolated cards.
        # This is the counterfactual missed in replay 92275837.
        if arch in {'dragapult','lucario'} and s.get('_active_riolu',0) and not s.get('_active_appear',0) and s.get('hand_hilda',0)>0 and not s.get('_has_mega_hand',0) and not s.get('supporter_played',0):
            future_en=int(s.get('active_energy',0))+(0 if s.get('energy_attached',0) else 1)
            dmg=270 if future_en>=2 else 130 if future_en>=1 else 0
            tp=max(1,int(s.get('opp_active_prize',1)))
            urgent=(int(s.get('my_prize',6))<=tp) or (tp>=2 and (int(s.get('my_prize',6))<=3 or int(s.get('opp_prize',6))<=2))
            if urgent and dmg>=int(s.get('opp_active_hp',9999)):
                n=self._advance(s);n['hand_hilda']=0;n['supporter_played']=1
                n['riolu']=max(0,n.get('riolu',0)-1);n['luc']=n.get('luc',0)+1;n['luc_lines']=n.get('luc_lines',0)
                if future_en>=2:n['ready_luc']=n.get('ready_luc',0)+1
                elif future_en==1:n['one_luc']=n.get('one_luc',0)+1
                n['active_prize']=3;n['active_energy']=future_en;n['active_damage']=270 if future_en>=2 else 130
                n['my_prize']=max(0,int(n.get('my_prize',6))-tp);n['prize_adv']=int(n.get('opp_prize',6))-int(n['my_prize']);n['opp_active_hp']=0
                out.append(('FINISH_WAVE',n,.065+.02*tp))

        # 1) convert current pressure into prizes; exact current KO gets strong priority.
        if s.get('active_damage',0)>0:
            n=self._advance(s);bonus=0.0
            if s.get('can_ko',0):
                take=max(1,int(s.get('opp_active_prize',1)));n['my_prize']=max(0,n['my_prize']-take);n['prize_adv']=n['opp_prize']-n['my_prize'];bonus=.045*take
            else:
                n['opp_active_hp']=max(10,n.get('opp_active_hp',0)-max(0,n.get('active_damage',0)));bonus=-.015
            # multi-prize Active is exposed after attacking; learned state value handles most of it.
            out.append(('PRIZE_RACE',n,bonus))
        # 2) build next Lucario wave. Hilda converts an existing Riolu into a one-energy Lucario;
        # Poffin/Gong establishes a new Riolu line. Repeating the branch advances it to ready.
        can_wave=(s.get('luc_lines',0)<2 or s.get('ready_luc',0)<2)
        if can_wave and (s.get('riolu',0)>0 or s.get('hand_poffin',0)>0 or s.get('hand_gong',0)>0):
            n=self._advance(s);bonus=.015
            if s.get('riolu',0)>0 and s.get('hand_hilda',0)>0:
                n['riolu']=max(0,n['riolu']-1);n['luc']=n.get('luc',0)+1;n['one_luc']=n.get('one_luc',0)+1;n['hand_hilda']=max(0,n['hand_hilda']-1);n['supporter_played']=1
            elif s.get('hand_poffin',0)>0:
                n['riolu']=n.get('riolu',0)+1;n['luc_lines']=n.get('luc_lines',0)+1;n['hand_poffin']=max(0,n['hand_poffin']-1);n['my_hand']=max(0,n.get('my_hand',0)-1)
            elif s.get('hand_gong',0)>0:
                n['riolu']=n.get('riolu',0)+1;n['luc_lines']=n.get('luc_lines',0)+1;n['hand_gong']=max(0,n['hand_gong']-1)
            # abstract follow-up attachment: a one-energy Lucario becomes the next ready wave.
            if s.get('one_luc',0)>0:
                n['one_luc']=max(0,n.get('one_luc',0)-1);n['ready_luc']=n.get('ready_luc',0)+1
            n['luc_lines']=n.get('luc',0)+n.get('riolu',0)
            out.append(('WAVE_BUILD',n,bonus))
        # 3) engine establishment if Solrock/Lunatone pair is missing and search resources exist.
        if not s.get('sol_luna',0) and (s.get('hand_poffin',0)>0 or s.get('hand_gong',0)>0 or s.get('my_hand',0)>=6):
            n=self._advance(s);n['sol_luna']=1;out.append(('ENGINE_BUILD',n,.005))
        # 4) hand disruption only when there is real opponent pressure; otherwise supporter tempo dominates.
        if s.get('opp_hand',0)>=6 and (s.get('hand_xero',0)>0 or s.get('hand_judge',0)>0) and (s.get('opp_stage2',0)>0 or s.get('_opp_bridge',0)>0 or s.get('opp_active_damage',0)>=120 or s.get('_opp_known_evolution_hand',0)>0):
            n=self._advance(s)
            if s.get('hand_xero',0)>0:n['opp_hand']=3;n['hand_xero']=0
            else:n['opp_hand']=4;n['hand_judge']=0
            n['supporter_played']=1;out.append(('DISRUPT',n,.008))
        # 5) trade a one-Prize body for one turn to preserve an unready three-Prize Lucario.
        if s.get('active_prize',0)==1 and s.get('_under_luc',0)>0 and s.get('_opp_can_ko',0) and s.get('opp_prize',6)>1:
            n=self._advance(s);n['opp_prize']=max(0,n['opp_prize']-1);n['prize_adv']=n['opp_prize']-n['my_prize'];n['active_prize']=3
            if n.get('one_luc',0)>0:n['one_luc']-=1;n['ready_luc']=n.get('ready_luc',0)+1;n['active_energy']=2
            out.append(('PRIZE_SHIELD',n,.025))
        # 6) bypass route is a real long-horizon plan; Hilda is especially valuable because the deck has one Dudunsparce ex.
        if arch=='crustle' and s.get('dud_energy',0)<3:
            n=self._advance(s);bonus=.02
            if not s.get('dud_ex',0) and s.get('dun_base',0)>0 and s.get('hand_hilda',0)>0:
                n['dud_ex']=1;n['dud_energy']=max(1,n.get('dud_energy',0));n['hand_hilda']=max(0,n['hand_hilda']-1);n['supporter_played']=1;bonus=.045
            elif s.get('dud_ex',0):n['dud_energy']=min(3,n.get('dud_energy',0)+1)
            elif s.get('dun_base',0)>0:n['dud_ex']=1
            out.append(('BYPASS_BUILD',n,bonus))
        # 7) draw/stabilize branch competes with greedy Hilda/disruption when hand is genuinely poor.
        if s.get('hand_lillie',0)>0 and s.get('my_hand',0)<=4:
            n=self._advance(s);n['my_hand']=8 if s.get('my_prize',6)==6 else 6;n['hand_lillie']=0;n['supporter_played']=1;out.append(('STABILIZE',n,.008))
        if not out:
            n=self._advance(s);out.append(('BASELINE',n,0.0))
        return out
    def _search(self,s,arch,depth):
        if depth<=0 or s.get('my_prize',1)<=0 or s.get('opp_prize',1)<=0:
            return self.value_model.value(s,arch),()
        vals=[]
        for mode,n,bonus in self._branches(s,arch):
            leaf,pv=self._search(n,arch,depth-1)
            # Small discount prevents repeatedly choosing slow development merely because each
            # synthetic node has a good static value.
            q=bonus+(0.94*leaf)
            vals.append((q,mode,(mode,)+pv))
        vals.sort(reverse=True,key=lambda z:z[0]);return vals[0][0],vals[0][2]
    def decide(self,obs,arch:str,force:bool=False,history=None)->TreeDecision:
        turn=int(obs.current.turn or 0);self.stats['calls']+=1
        # A macro commitment is intentionally stable within one turn.  The full
        # sequence is re-encoded at the next turn boundary; intra-turn choices are
        # still history-conditioned by the decision gate and residual input.
        step=turn
        if not force and self.cache_key==step:return self.cached
        s=self.features(obs,arch,history);base=self.value_model.value(s,arch)
        roots=[]
        for mode,n,bonus in self._branches(s,arch):
            leaf,pv=self._search(n,arch,self.depth-1);roots.append((bonus+0.94*leaf,mode,(mode,)+pv))
        roots.sort(reverse=True,key=lambda z:z[0])
        if not roots:d=TreeDecision('BASELINE',base,base,0.0,('BASELINE',),turn)
        else:
            best=roots[0];second=roots[1][0] if len(roots)>1 else base
            d=TreeDecision(best[1],best[0],base,best[0]-second,best[2],turn)
        self.turn=turn;self.cache_key=step;self.cached=d
        self.stats['modes'][d.mode]=self.stats['modes'].get(d.mode,0)+1
        pv='>'.join(d.pv);self.stats['pv'][pv]=self.stats['pv'].get(pv,0)+1
        if len(self.stats['margins'])<2000:self.stats['margins'].append(round(d.margin,6))
        return d
    def get_stats(self):return dict(self.stats)
