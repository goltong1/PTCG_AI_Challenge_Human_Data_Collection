"""Hierarchical Mega Lucario policy.

Active architecture (the only code path that should receive future edits):
  1. StrategyPlanner interprets the public board and commits to a 2-3 turn plan.
  2. FrozenV118Executor supplies the validated tactical action for the current legal state.
  3. StrategyArbiter changes that action only when a legal alternative directly advances
     the persistent plan with a high-confidence, matchup-specific reason.
  4. ReplayLossRepairGate is the final legal-action guard for audited human-loss
     failures such as resistance-blind KOs, prize-feed promotions, and wall abandonment.

The old v118 patch stack is quarantined in frozen_v118_tactical_executor.py and treated as a frozen
compatibility executor.  It is not monkey-patched here and no new strategic rule is added to
it.  This keeps the proven tactical baseline while preventing the policy stack from growing
another layer of intertwined version patches.
"""
from __future__ import annotations

import hashlib,importlib.util,os,sys
from dataclasses import dataclass
from typing import Optional

from cg.api import AreaType,CardType,EnergyType,OptionType,Pokemon,SelectContext,all_card_data,to_observation_class

HERE=os.path.dirname(os.path.abspath(__file__)) if globals().get('__file__') else '/kaggle_simulations/agent'
if HERE not in sys.path:sys.path.insert(0,HERE)

# ---- frozen validated tactical executor ---------------------------------
def _load(tag,file):
    n=tag+'_'+hashlib.sha1((HERE+file).encode()).hexdigest()[:12]
    s=importlib.util.spec_from_file_location(n,os.path.join(HERE,file));m=importlib.util.module_from_spec(s);sys.modules[n]=m
    assert s.loader is not None;s.loader.exec_module(m);return m

_legacy=_load('lucario_frozen_v118','frozen_v118_tactical_executor.py')
_rr_mod=_load('lucario_regret','regret_residual.py')
_td_mod=_load('lucario_terminal','terminal_distill_runtime.py')
_rr=_rr_mod.ResidualPolicy(HERE)
_td=_td_mod.TerminalDistillGate('lucario')
_mt_mod=_load('lucario_macro_tree','macro_tree_runtime.py')
_MT=_mt_mod.MacroTreePlanner(HERE,depth=3)
_close_mod=_load('lucario_closeout','closeout_runtime.py')
_CLOSE=_close_mod.CloseoutPlanner()
_hist_mod=_load('lucario_history','history_context_runtime.py')
_hr_mod=_load('lucario_history_replay','history_replay_policy.py')
_tg_mod=_load('lucario_temporal_gru','temporal_gru_policy.py')
_ta_mod=_load('lucario_temporal_attention_observer','temporal_attention_observer.py')
_ts_mod=_load('lucario_temporal_safety','temporal_safety_runtime.py')
_lr_mod=_load('lucario_loss_repair','loss_repair_runtime.py')

CARD={c.cardId:c for c in all_card_data()}

class C:
    RIOLU70=333; RIOLU80=677; LUCARIO=678; SOLROCK=676; LUNATONE=675
    DUNSPARCE=305; DUDUN=66; DUDUN_EX=306; OGERPON=117
    PPP=1141; GONG=1142; POKE_PAD=1152; POFFIN=1086; JUDGE=1213; XEROSIC=1197
    AIR_BALLOON=1174; HERO_CAPE=1159; LILLIE=1227; HILDA=1225; BOSS=1182
    MOUNTAIN=1252; BLACK_BELT=1211; BASIC_F=6; ROCK_F=20
    DREEPY=119; DRAKLOAK=120; DRAGAPULT=121
    ABRA=741; KADABRA=742; ALAKAZAM=743; ALAKAZAM_ALT=245
    DWEBBLE=344; CRUSTLE=345
    DURALUDON=169; ARCHALUDON=190; CINDERACE=666; STARYU=1030; STARMIE=1031
    IMPIDIMP=646; MORGREM=647; GRIMMSNARL=648
    TEAL=96; WELLSPRING=108; CLEFAIRY=272; LATIAS=184; MEOWTH=1071; PECHARUNT=230; FEZ=140; CHIYU=31; KANG=756
    HYDRAPPLE=150; MEGANIUM=710; CHIKORITA=917; BAYLEEF=709; APPLIN=149; DIPPLIN=93
    CYN_GIBLE=379; CYN_GABITE=380; CYN_GARCHOMP=381; CYN_ROSELIA=341; CYN_ROSERADE=342
    SLOWPOKE=162; SLOWKING=163; KYUREM=144


_HISTORY=_hist_mod.HistoryContext()
_HIST_GATE=_hist_mod.HistoryDecisionGate(_HISTORY,CARD,C.JUDGE,C.XEROSIC,C.LILLIE)
_HISTORY_REPLAY=_hr_mod.HistoryReplayPolicy(HERE,_HISTORY)
_LEAGUE_REPLAY=_hr_mod.HistoryReplayPolicy(HERE,_HISTORY,'history_league_model.json')
_TEMPORAL_GRU=_tg_mod.TemporalGRUPolicy(HERE,_HISTORY,_hr_mod)
_TEMPORAL_ATTENTION=_ta_mod.TemporalAttentionObserver(HERE,_HISTORY,_tg_mod,_hr_mod)
_TEMPORAL_SAFETY=_ts_mod.TemporalSafetyGate(_HISTORY,CARD)
_LOSS_REPAIR=_lr_mod.ReplayLossRepairGate()

AURA_JAB=982; MEGA_BRAVE=983; TENACIOUS=425; DRILL=426; DEMOLISH=148

SIG={
 'crustle':{C.DWEBBLE,C.CRUSTLE},
 'dragapult':{C.DREEPY,C.DRAKLOAK,C.DRAGAPULT},
 'alakazam':{C.ABRA,C.KADABRA,C.ALAKAZAM,245},
 'starmie':{C.STARYU,C.STARMIE},
 'archaludon':{C.DURALUDON,C.ARCHALUDON,C.CINDERACE},
 'marnie':{C.IMPIDIMP,C.MORGREM,C.GRIMMSNARL},
 'lucario':{C.RIOLU70,C.RIOLU80,C.LUCARIO},
 'hydrapple':{C.HYDRAPPLE,C.MEGANIUM,C.CHIKORITA,C.BAYLEEF,C.APPLIN,C.DIPPLIN},
 'cynthia':{C.CYN_GIBLE,C.CYN_GABITE,C.CYN_GARCHOMP},
 'slowking':{C.SLOWPOKE,C.SLOWKING},
 'terabox':{C.TEAL,C.WELLSPRING,C.CLEFAIRY,C.LATIAS,C.MEOWTH,C.PECHARUNT,C.FEZ,C.CHIYU,C.KANG},
}

@dataclass
class Plan:
    archetype:str='unknown'
    strategy:str='LUCARIO_CHAIN'
    start_turn:int=-1
    horizon:int=3
    primary:int=C.LUCARIO
    secondary:int=C.LUCARIO
    target_ids:tuple[int,...]=()
    objective:str='build two Lucario attackers and convert prizes efficiently'


class StrategyPlanner:
    def __init__(self):
        self.reset()
    def reset(self):
        self.revealed=set();self.plan=Plan();self.last_turn=-1;self.own_deck_scan=None;self.stats={'plans':{},'overrides':{},'calls':0}
    def _field(self,p):return [q for q in list(p.active or [])+list(p.bench or []) if q is not None]
    def update_revealed(self,obs):
        me=obs.current.yourIndex;op=obs.current.players[1-me]
        # ``obs.logs`` contains only the events since the previous selection.
        # The persistent history encoder contributes every card revealed earlier
        # in the game, including cards no longer present on the public board.
        self.revealed.update(_HISTORY.revealed_ids(1-me))
        for p in self._field(op):
            self.revealed.add(p.id)
            for pre in (p.preEvolution or []):self.revealed.add(pre.id)
        for c in (op.discard or []):self.revealed.add(c.id)
        for log in (obs.logs or []):
            if getattr(log,'playerIndex',None)==1-me:
                for k in ('cardId','cardIdTarget','cardIdActive','cardIdBench','cardIdBefore','cardIdAfter'):
                    x=getattr(log,k,None)
                    if x:self.revealed.add(int(x))
    def recognize(self):
        ids=self.revealed
        # Dedicated evolution-line signatures outrank splashable tech Basics.  The
        # previous recognizer treated a single Latias ex as definitive Tera Box and
        # could flip a correctly identified Dragapult game into DUDUN_PRESSURE midgame.
        # That happened in the current human replay 20260812_205541_57cb856f.
        # Core-evolution evidence outranks a lone splash Basic.  Replay 92558889
        # is a 4-4 Archaludon + 4 Cinderace deck with a single Dwebble; treating that
        # Dwebble as definitive Crustle sent the whole game into BYPASS_CRUSTLE.
        # A revealed Crustle itself still has highest priority because its immunity is
        # strategically decisive.
        if C.CRUSTLE in ids:return 'crustle'
        if ids & {C.DURALUDON,C.ARCHALUDON}:return 'archaludon'
        for a in ('crustle','dragapult','alakazam','starmie','archaludon','marnie','lucario','hydrapple','cynthia','slowking'):
            if ids & SIG[a]:return a
        # Tera Box needs a strong anchor (its Ogerpon/Kangaskhan core) or at least
        # two independent splash-tech signatures.  A lone Latias/Fezandipiti/Meowth
        # is not enough to redefine the whole opponent archetype.
        if ids & {C.TEAL,C.WELLSPRING,C.KANG}:return 'terabox'
        weak={C.CLEFAIRY,C.LATIAS,C.MEOWTH,C.PECHARUNT,C.FEZ,C.CHIYU}
        if len(ids & weak)>=2:return 'terabox'
        return 'unknown'
    def _find(self,player,cid):return next((p for p in self._field(player) if p.id==cid),None)
    def build(self,obs):
        self.stats['calls']+=1;self.update_revealed(obs)
        st=obs.current;me=st.yourIndex;mine=st.players[me];op=st.players[1-me];a=self.recognize()
        # Exact own-deck scans exposed by search effects are strategic information.
        # Cache them so a route whose only evolution is known to be prized/absent
        # is assigned zero completion probability instead of being charged blindly.
        try:
            sdeck=list(obs.select.deck or []) if obs.select is not None else []
            own_opts=[o for o in list(obs.select.option or []) if getattr(o,'playerIndex',me)==me] if obs.select is not None else []
            if sdeck and own_opts:
                self.own_deck_scan={int(c.id) for c in sdeck if c is not None}
        except Exception:
            pass
        old=self.plan
        # proposed macro plan
        p=Plan(archetype=a,start_turn=st.turn,horizon=3)
        if a=='crustle' and ({C.DWEBBLE,C.CRUSTLE}&self.revealed):
            dud=self._find(mine,C.DUDUN_EX);oger=self._find(mine,C.OGERPON)
            hand_ids={int(c.id) for c in (mine.hand or [])};discard_ids={int(c.id) for c in (mine.discard or [])}
            # A cached exact deck scan can prove the one-copy Dudunsparce ex route dead.
            # Only then re-plan to Cornerstone; before that, preserve normal uncertainty.
            dud_proven_dead=bool(self.own_deck_scan is not None and C.DUDUN_EX not in self.own_deck_scan and C.DUDUN_EX not in hand_ids and dud is None)
            oger_available=bool(oger is not None or C.OGERPON in hand_ids or self.own_deck_scan is None or C.OGERPON in self.own_deck_scan)
            oger_committed=bool(oger and len(oger.energies)>=2 and not(dud and len(dud.energies)>=3))
            force_oger=bool(dud_proven_dead and oger_available)
            p.strategy='BYPASS_CRUSTLE';p.primary=C.OGERPON if (force_oger or oger_committed) else C.DUDUN_EX
            p.secondary=C.DUDUN_EX if p.primary==C.OGERPON else C.OGERPON
            p.target_ids=(C.CRUSTLE,C.DWEBBLE)
            p.objective='re-plan from exact deck visibility, then commit one live bypass attacker without route oscillation'
            if force_oger:self.stats['overrides']['crustle:route_replan_dudun_unavailable']=self.stats['overrides'].get('crustle:route_replan_dudun_unavailable',0)+1
        elif a=='terabox':
            opp_field=self._field(op)
            # Real replays 92673565, 92729436 and 92883371 exposed the same
            # four-Teal-Mask deck.  Dudunsparce ex lost two of the three games
            # while our Cornerstone Mask Ogerpon is a direct damage lock against
            # Teal Dance attackers.  Require three public Teal Mask bodies and no
            # other public Pokemon before committing to that hard-wall route.
            teal_only=len(opp_field)>=3 and all(q.id==C.TEAL for q in opp_field)
            exn=sum(1 for q in opp_field if CARD[q.id].ex or CARD[q.id].megaEx)
            if teal_only:
                p.strategy='TEAL_ABILITY_WALL';p.primary=C.OGERPON;p.secondary=C.LUCARIO
                p.target_ids=(C.TEAL,);p.objective='establish Cornerstone Mask Ogerpon to lock the pure Teal Mask damage plan'
            elif exn>=2:
                p.strategy='DUDUN_PRESSURE';p.primary=C.DUDUN_EX;p.secondary=C.LUCARIO
                p.target_ids=(C.KANG,C.TEAL,C.WELLSPRING,C.CLEFAIRY,C.LATIAS,C.MEOWTH);p.objective='turn the wide ex board into Tenacious Tail damage'
        elif a=='starmie':
            p.strategy='LUCARIO_CHAIN';p.primary=C.LUCARIO;p.secondary=C.LUCARIO
            p.target_ids=(C.STARYU,C.STARMIE);p.objective='preserve Lucario tempo and deny the Starmie evolution line'
        elif a=='archaludon':
            p.strategy='ABILITY_WALL';p.primary=C.OGERPON;p.secondary=C.LUCARIO
            p.target_ids=(C.DURALUDON,C.ARCHALUDON,C.CINDERACE);p.objective='establish Cornerstone as a safe anchor while preserving Lucario prize tempo'
        elif a=='alakazam':
            p.strategy='HAND_DENIAL';p.primary=C.LUCARIO;p.secondary=C.LUCARIO
            p.target_ids=(C.KADABRA,C.ABRA,C.ALAKAZAM,C.ALAKAZAM_ALT);p.objective='Rock-Energy protection plus hand compression, then deny the evolution bridge'
        elif a=='dragapult':
            p.strategy='EVOLUTION_DENIAL';p.target_ids=(C.DRAKLOAK,C.DREEPY,C.DRAGAPULT)
            p.objective='keep two Lucario lines and convert bridge KOs before Dragapult stabilizes'
        elif a=='marnie':
            # The three current human losses show that evolution denial alone is
            # insufficient once Punk Up is online.  Grimmsnarl ex attacks from a
            # Pokemon with an Ability, so Cornerstone Mask Ogerpon is the durable
            # primary route after the Stage-2 is publicly revealed.
            if C.GRIMMSNARL in self.revealed:
                p.strategy='MARNIE_ABILITY_WALL';p.primary=C.OGERPON;p.secondary=C.LUCARIO
                p.target_ids=(C.GRIMMSNARL,C.MORGREM,C.IMPIDIMP)
                p.objective='lock Punk Up attackers with Cornerstone, then remove Froslass and Munkidori support'
            else:
                p.strategy='EVOLUTION_DENIAL';p.target_ids=(C.MORGREM,C.IMPIDIMP,C.GRIMMSNARL)
                p.objective='remove the one-prize evolution bridge before Punk Up creates a second attacker'
        elif a=='hydrapple':
            p.strategy='HYDRAPPLE_PRIZE';p.primary=C.LUCARIO;p.secondary=C.LUCARIO
            p.target_ids=(C.TEAL,C.FEZ,C.MEOWTH,C.HYDRAPPLE,C.CHIKORITA,C.APPLIN)
            p.objective='preserve Lucario attack continuity and map prizes through exposed two-prize grass support'
        elif a=='cynthia':
            p.strategy='CYNTHIA_PRIZE';p.primary=C.LUCARIO;p.secondary=C.LUCARIO
            p.target_ids=(C.CYN_GARCHOMP,C.CYN_GABITE,C.CYN_GIBLE,C.CYN_ROSERADE,C.CYN_ROSELIA)
            p.objective='convert exposed damaged Garchomp ex prizes without sacrificing attacker continuity'
        elif a=='slowking':
            p.strategy='SPREAD_SAFETY';p.primary=C.LUCARIO;p.secondary=C.OGERPON
            p.target_ids=(C.SLOWKING,C.SLOWPOKE,C.KYUREM)
            p.objective='do not expose a third low-HP prize to a ready Seek Inspiration into Trifrost window'
        elif a=='lucario':
            p.strategy='MIRROR_PRIZE';p.target_ids=(C.RIOLU80,C.RIOLU70,C.LUCARIO)
            p.objective='deny Riolu and spend damage modifiers only when they cross a KO threshold'
        elif st.turn<=3:
            p.strategy='ENGINE_BUILD';p.horizon=2;p.objective='establish Riolu plus Solrock/Lunatone before committing excess resources'
        # game-ending prize window always interrupts a long plan
        if len(mine.prize or [])<=2:
            p.strategy='EXACT_PRIZE';p.horizon=1;p.objective='take the smallest-resource line that ends the game'
        # Persistence: exact plan identity is held for 2-3 turns.  Crustle primary is also
        # held once resources are committed, preventing Dudunsparce/Ogerpon oscillation.
        live=old.start_turn>=0 and st.turn<=old.start_turn+max(1,old.horizon)
        if live and old.archetype==p.archetype and p.strategy!='EXACT_PRIZE':
            if old.strategy==p.strategy:
                p.start_turn=old.start_turn;p.horizon=old.horizon
                if p.strategy=='BYPASS_CRUSTLE':
                    # Do not preserve a primary route after exact search information has
                    # proved its one-copy evolution unavailable.
                    dud_dead=bool(self.own_deck_scan is not None and C.DUDUN_EX not in self.own_deck_scan and C.DUDUN_EX not in {int(c.id) for c in (mine.hand or [])} and self._find(mine,C.DUDUN_EX) is None)
                    if not (old.primary==C.DUDUN_EX and dud_dead):
                        p.primary=old.primary;p.secondary=old.secondary
            elif {old.strategy,p.strategy}<={'ENGINE_BUILD','LUCARIO_CHAIN'}:
                p=old
        if p.strategy!=old.strategy or p.archetype!=old.archetype or p.primary!=old.primary:
            key=f'{p.archetype}:{p.strategy}:{p.primary}';self.stats['plans'][key]=self.stats['plans'].get(key,0)+1
        self.plan=p;return p
    def note_override(self,key):self.stats['overrides'][key]=self.stats['overrides'].get(key,0)+1


_PLANNER=StrategyPlanner()


def _field(player):return [p for p in list(player.active or [])+list(player.bench or []) if p is not None]
def _card(obs,area,index,player):
    try:
        area=AreaType(int(area));pl=obs.current.players[player]
        if area==AreaType.DECK:return obs.select.deck[index]
        if area==AreaType.HAND:return pl.hand[index]
        if area==AreaType.DISCARD:return pl.discard[index]
        if area==AreaType.ACTIVE:return pl.active[index]
        if area==AreaType.BENCH:return pl.bench[index]
        if area==AreaType.PRIZE:return pl.prize[index]
        if area==AreaType.LOOKING:return obs.current.looking[index]
    except Exception:return None
    return None

def _is_lucario_line(p):return p is not None and p.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO}
def _has_rock(p):return p is not None and any(getattr(e,'id',None)==C.ROCK_F for e in (p.energyCards or []))
def _find_field(player,cid):return next((p for p in _field(player) if p.id==cid),None)
def _opt_card(obs,o,me):
    if o.type==OptionType.PLAY:return _card(obs,AreaType.HAND,o.index,me)
    if o.type in {OptionType.CARD,OptionType.ABILITY}:return _card(obs,o.area,o.index,getattr(o,'playerIndex',me))
    if o.type==OptionType.ATTACH:return _card(obs,AreaType.HAND,o.index,me)
    if o.type==OptionType.EVOLVE:return _card(obs,o.area,o.index,me)
    return None

def _choose_best(indices,score):
    return max(indices,key=lambda i:(score(i),-i)) if indices else None


class StrategyArbiter:
    """High-confidence strategic overrides only; otherwise retain validated v118 action."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.finish_turn=-1;self.finish_target_serial=-1;self.finish_damage=0;self.alak_ppp_turn=-1;self.alak_ppp_used=0;self.cynthia_boss_turn=-1;self.cynthia_boss_target=-1;self.cynthia_boss_attack=-1;_CLOSE.reset()
    def _finish_live(self,obs):
        return self.finish_turn==obs.current.turn and self.finish_target_serial>=0
    def choose(self,obs_dict,base):
        if obs_dict.get('select') is None and 'current' not in obs_dict:return base
        try:obs=to_observation_class(obs_dict)
        except Exception:return base
        if obs.current is None or obs.select is None:return base
        plan=_PLANNER.build(obs);opts=list(obs.select.option or [])
        tree=None
        try:
            if obs.select.context==SelectContext.MAIN: tree=_MT.decide(obs,plan.archetype,history=_HISTORY)
        except Exception: tree=None
        if not opts:return base
        me=obs.current.yourIndex;mine=obs.current.players[me];op=obs.current.players[1-me];ctx=obs.select.context

        # Track already-played PPP copies from the previous action log so a
        # multi-PPP exact threshold is evaluated as a sequence rather than each copy
        # in isolation.
        if self.alak_ppp_turn!=obs.current.turn:
            self.alak_ppp_turn=obs.current.turn;self.alak_ppp_used=0
        try:
            for lg in (obs.logs or []):
                if getattr(lg,'playerIndex',None)==me and getattr(lg,'cardId',None)==C.PPP:
                    self.alak_ppp_used+=1
        except Exception:pass

        # Proved same-turn wins are handled by a separate end-game planner.  If it
        # cannot prove mate, the normal macro tree / matchup strategy remains untouched.
        try:
            z=_CLOSE.choose(obs,base)
            if z is not None:return z
        except Exception:
            pass

        # UNIVERSAL DECK-RUNWAY RECYCLE ---------------------------------------
        # Replays 92671590 and 92689590 reached 0-5 cards with Judge/Lillie legal
        # and a large hand, yet the old Crustle-only guard kept consuming the deck.
        # A same-turn mate has already been claimed by CloseoutPlanner above, so at
        # this point a hand of at least eight cards should be recycled in every
        # matchup when doing so adds at least five cards of runway.  The Supporter
        # is followed by the ordinary tactical executor, so this does not forfeit a
        # legal attack; it only prevents a forced future deck-out.
        if ctx==SelectContext.MAIN and not obs.current.supporterPlayed:
            deck_n=int(getattr(mine,'deckCount',0) or 0); hand_n=len(mine.hand or [])
            if deck_n<=5 and hand_n>=8:
                recycle=[]
                for i,o in enumerate(opts):
                    if o.type!=OptionType.PLAY:continue
                    c0=_card(obs,AreaType.HAND,o.index,me)
                    if c0 is None or c0.id not in {C.JUDGE,C.LILLIE}:continue
                    draw_n=4 if c0.id==C.JUDGE else (8 if len(mine.prize or [])==6 else 6)
                    post=max(0,deck_n+(hand_n-1)-draw_n);gain=post-deck_n
                    if gain>=5:recycle.append((post,gain,1 if c0.id==C.JUDGE else 0,i,c0.id))
                if recycle:
                    recycle.sort(reverse=True);_,_,_,j,cid=recycle[0]
                    _PLANNER.note_override('universal:deck_runway_recycle_judge' if cid==C.JUDGE else 'universal:deck_runway_recycle_lillie')
                    return [j]

        # ALAKAZAM EXACT-THREE-PRIZE PROMOTION SHIELD -------------------------
        # In replay 92759896, Powerful Hand had just taken a KO while the opponent
        # had exactly three Prizes remaining.  Promoting an unprotected Mega Lucario
        # exposed the entire game; a powered one-Prize Solrock was legal and bought
        # another full turn.  Scope this to that exact public prize/lethal window.
        if plan.archetype=='alakazam' and ctx==SelectContext.TO_ACTIVE and base and len(base)==1 and op.active and len(op.prize or [])==3:
            try:
                t0=op.active[0];td=CARD.get(t0.id)
                bo=opts[base[0]];bq=_card(obs,bo.area,bo.index,getattr(bo,'playerIndex',me))
                incoming=20*(int(op.handCount or 0)+1) if t0.id==C.ALAKAZAM and len(t0.energies or [])>=1 else 0
                bcd=CARD.get(bq.id) if bq is not None else None
                protected=bool(bq is not None and any(getattr(e,'id',None)==C.ROCK_F for e in (bq.energyCards or [])))
                if bq is not None and bcd is not None and bcd.megaEx and not protected and incoming>=int(bq.hp or 0):
                    shields=[]
                    luna_live=_find_field(mine,C.LUNATONE) is not None
                    for i,o in enumerate(opts):
                        q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me));cd=CARD.get(q.id) if q is not None else None
                        if q is None or cd is None or cd.ex or cd.megaEx:continue
                        rock=any(getattr(e,'id',None)==C.ROCK_F for e in (q.energyCards or []))
                        attack_value=0
                        if q.id==C.SOLROCK and luna_live and len(q.energies or [])>=1:attack_value=500
                        elif q.id in {C.RIOLU70,C.RIOLU80} and len(q.energies or [])>=1:attack_value=400
                        elif q.id==C.DUNSPARCE and len(q.energies or [])>=1:attack_value=300
                        shields.append((1000 if rock else 0,attack_value,int(q.hp or 0),-i,i))
                    if shields:
                        shields.sort(reverse=True);_PLANNER.note_override('alakazam:exact_three_prize_promotion_shield');return [shields[0][-1]]
            except Exception:pass

        # SLOWKING / KYUREM SPREAD-SAFETY GATE -------------------------------
        # Replay 92762959 publicly showed an Energy-attached mature Slowpoke before
        # our setup turn.  Poffin raised the number of <=110 HP targets from two to
        # four; Seek Inspiration copied Kyurem's Trifrost and took three Prizes at
        # once.  The deck identity is inferred only from public Slowpoke/Slowking.
        # When that attack window is live, charge a >110 HP attacker once and end
        # rather than add the third guaranteed spread target.
        if plan.strategy=='SPREAD_SAFETY' and ctx==SelectContext.MAIN and base and len(base)==1:
            try:
                threat=any(q.id in {C.SLOWPOKE,C.SLOWKING} and len(q.energies or [])>=1 and not bool(getattr(q,'appearThisTurn',False)) for q in _field(op))
                vulnerable=sum(int(q.hp or 0)<=110 for q in _field(mine))
                bo=opts[base[0]];bc=_card(obs,AreaType.HAND,bo.index,me) if bo.type==OptionType.PLAY else None
                adds_target=bool(bc is not None and (bc.id==C.POFFIN or (CARD.get(bc.id) and CARD[bc.id].cardType==CardType.POKEMON and int(CARD[bc.id].hp or 0)<=110)))
                if threat and vulnerable>=2 and adds_target:
                    if not obs.current.energyAttached:
                        safe=[]
                        for i,o in enumerate(opts):
                            if o.type!=OptionType.ATTACH:continue
                            e=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                            if e is None or q is None or e.id not in {C.BASIC_F,C.ROCK_F} or int(q.hp or 0)<=110:continue
                            role=300 if q.id==C.OGERPON else 250 if _is_lucario_line(q) else 100
                            safe.append((role,-len(q.energies or []),-i,i))
                        if safe:
                            safe.sort(reverse=True);_PLANNER.note_override('slowking:spread_safe_attach');return [safe[0][-1]]
                    ends=[i for i,o in enumerate(opts) if o.type==OptionType.END]
                    if ends:
                        _PLANNER.note_override('slowking:spread_safe_end');return [ends[0]]
            except Exception:pass

        # FINISH_WAVE: a replay-derived multi-step strategy sequence, not a one-action
        # bonus.  In 92275837 the policy had a mature 2-Energy Riolu and Hilda versus
        # a 50-HP Dragapult ex, but spent Judge first, lost Hilda, then used Quick
        # Attack and lost.  When Hilda can turn an established Active Riolu into an
        # immediate multi-Prize KO in a critical prize window, commit to the whole
        # Hilda -> Mega -> Energy -> exact attack sequence before disruption.
        if self.finish_turn!=obs.current.turn:
            self.finish_turn=-1;self.finish_target_serial=-1;self.finish_damage=0
        if plan.archetype in {'dragapult','lucario'} and ctx==SelectContext.MAIN and mine.active and op.active and not self._finish_live(obs):
            a0=mine.active[0];t0=op.active[0]
            if a0.id in {C.RIOLU70,C.RIOLU80} and not bool(getattr(a0,'appearThisTurn',False)) and not obs.current.supporterPlayed:
                hilda_i=None
                for i,o in enumerate(opts):
                    if o.type==OptionType.PLAY:
                        c0=_card(obs,AreaType.HAND,o.index,me)
                        if c0 is not None and c0.id==C.HILDA:hilda_i=i;break
                has_mega=any(c0.id==C.LUCARIO for c0 in (mine.hand or []))
                en=len(a0.energies or []);future_en=en+(0 if obs.current.energyAttached else 1)
                dmg=270 if future_en>=2 else 130 if future_en>=1 else 0
                cd=CARD.get(t0.id);tp=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
                urgent=(len(mine.prize or [])<=tp) or (tp>=2 and (len(mine.prize or [])<=3 or len(op.prize or [])<=2))
                if hilda_i is not None and not has_mega and dmg>=int(t0.hp or 0) and urgent and tree is not None and tree.mode=='FINISH_WAVE':
                    self.finish_turn=obs.current.turn;self.finish_target_serial=t0.serial;self.finish_damage=dmg
                    _PLANNER.note_override('macro_sequence:finish_wave_hilda');return [hilda_i]
        if self._finish_live(obs):
            # Hilda searches Evolution first, then Energy. Keep both pieces of the
            # committed line instead of letting generic card value redirect the search.
            if ctx==SelectContext.TO_HAND and getattr(getattr(obs.select,'effect',None),'id',-1)==C.HILDA:
                mega=[];energy=[]
                for i,o in enumerate(opts):
                    c0=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if c0 is None:continue
                    if c0.id==C.LUCARIO:mega.append(i)
                    elif c0.id in {C.BASIC_F,C.ROCK_F}:energy.append(i)
                if mega:
                    _PLANNER.note_override('macro_sequence:finish_wave_search_mega');return [mega[0]]
                if energy:
                    # Basic Fighting is the safest generic exact-KO energy; Rock is
                    # retained as fallback when it is the only Hilda energy option.
                    j=next((i for i in energy if _card(obs,opts[i].area,opts[i].index,getattr(opts[i],'playerIndex',me)).id==C.BASIC_F),energy[0])
                    _PLANNER.note_override('macro_sequence:finish_wave_search_energy');return [j]
            if ctx==SelectContext.MAIN and mine.active:
                a0=mine.active[0];t0=op.active[0] if op.active else None
                # Evolve the committed Active Riolu as soon as Hilda supplies Mega.
                if a0.id in {C.RIOLU70,C.RIOLU80}:
                    ev=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.EVOLVE:continue
                        c0=_card(obs,o.area,o.index,me);basep=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if c0 is not None and basep is not None and c0.id==C.LUCARIO and basep.serial==a0.serial:ev.append(i)
                    if ev:
                        _PLANNER.note_override('macro_sequence:finish_wave_evolve');return [ev[0]]
                if a0.id==C.LUCARIO:
                    en=len(a0.energies or [])
                    # Attach only if it opens the exact attack threshold.
                    if en<2 and not obs.current.energyAttached:
                        need=1 if (t0 is not None and int(t0.hp or 0)<=130) else 2
                        if en<need:
                            att=[]
                            for i,o in enumerate(opts):
                                if o.type!=OptionType.ATTACH:continue
                                p0=_card(obs,o.inPlayArea,o.inPlayIndex,me);c0=_card(obs,AreaType.HAND,o.index,me)
                                if p0 is not None and p0.serial==a0.serial and c0 is not None and c0.id in {C.BASIC_F,C.ROCK_F}:att.append(i)
                            if att:
                                _PLANNER.note_override('macro_sequence:finish_wave_attach');return [att[0]]
                    attacks={o.attackId:i for i,o in enumerate(opts) if o.type==OptionType.ATTACK}
                    hp=int(t0.hp or 0) if t0 is not None else 9999
                    if AURA_JAB in attacks and hp<=130:
                        _PLANNER.note_override('macro_sequence:finish_wave_aura');return [attacks[AURA_JAB]]
                    if MEGA_BRAVE in attacks and hp<=270:
                        _PLANNER.note_override('macro_sequence:finish_wave_mega');return [attacks[MEGA_BRAVE]]

        # Universal survival invariant: Run Away Draw must never shuffle the last
        # Pokemon in play into the deck.  The frozen v118 executor can occasionally
        # evolve a lone Active Dunsparce and immediately use the Ability, which loses
        # the game on the spot.  Establish another Basic first, otherwise end/attack.
        if ctx==SelectContext.MAIN and len(_field(mine))==1 and mine.active and mine.active[0].id==C.DUDUN:
            # Some Ability options do not carry a resolvable area/index.  The
            # invariant is about the board state itself, so any selected Ability on a
            # lone Active Dudunsparce is treated as Run Away Draw and blocked.
            base_ability=any(0<=bi<len(opts) and opts[bi].type==OptionType.ABILITY for bi in list(base or []))
            if base_ability:
                basic_priority={C.RIOLU80:100,C.RIOLU70:95,C.SOLROCK:85,C.LUNATONE:80,C.DUNSPARCE:70,C.OGERPON:60}
                plays=[]
                for i,o in enumerate(opts):
                    if o.type!=OptionType.PLAY:continue
                    c=_card(obs,AreaType.HAND,o.index,me)
                    if c is not None and c.id in basic_priority:plays.append(i)
                if plays:
                    j=_choose_best(plays,lambda i:basic_priority.get(_card(obs,AreaType.HAND,opts[i].index,me).id,0))
                    _PLANNER.note_override('universal:runaway_survival');return [j]
                attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
                if attacks:
                    _PLANNER.note_override('universal:runaway_survival');return [attacks[0]]
                ends=[i for i,o in enumerate(opts) if o.type==OptionType.END]
                if ends:
                    _PLANNER.note_override('universal:runaway_survival');return [ends[0]]

        # Universal one-turn resource invariant: Premium Power Pro only lasts for
        # this turn.  If no attack is legal *right now*, spending it has zero immediate
        # damage value and there is never a reason to burn it before the action that
        # actually opens an attack (attach/evolve/retreat).  Defer PPP until an attack
        # option exists, and use the best development action instead.  This directly
        # prevents the observed turn-1 going-first PPP burn in replay 92275837.
        if ctx==SelectContext.MAIN and base and len(base)==1:
            try:
                bi=base[0];bo=opts[bi]
                bc=_card(obs,AreaType.HAND,bo.index,me) if bo.type==OptionType.PLAY else None
                if bc is not None and bc.id==C.PPP and not any(o.type==OptionType.ATTACK for o in opts):
                    def tempo_score(i):
                        o=opts[i]
                        if o.type==OptionType.EVOLVE:
                            q=_card(obs,o.area,o.index,me)
                            return 1200+(250 if q is not None and q.id==C.LUCARIO else 0)
                        if o.type==OptionType.ATTACH:
                            q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                            if q is None:return 500
                            if q.id==C.LUCARIO:return 1150-100*len(q.energies or [])
                            if q.id in {C.RIOLU70,C.RIOLU80}:return 1100
                            if q.id==C.SOLROCK:return 760
                            if q.id==C.DUDUN_EX:return 720
                            if q.id==C.DUNSPARCE:return 540
                            return 500
                        if o.type==OptionType.PLAY:
                            c0=_card(obs,AreaType.HAND,o.index,me)
                            if c0 is None or c0.id==C.PPP:return -10**9
                            if c0.id in {C.RIOLU80,C.RIOLU70}:return 1080
                            if c0.id==C.SOLROCK:return 980
                            if c0.id==C.LUNATONE:return 940
                            if c0.id==C.DUNSPARCE:return 900
                            if c0.id==C.POFFIN:return 930
                            if c0.id==C.GONG:return 900
                            if c0.id==C.HILDA:return 860
                            if c0.id==C.LILLIE:return 820
                            if c0.id==C.POKE_PAD:return 720
                            if c0.id in {C.JUDGE,C.XEROSIC}:return 500
                            if c0.id==C.BOSS:return 120
                            cd=CARD.get(c0.id)
                            if cd is not None and cd.cardType==CardType.POKEMON:return 700
                            return 450
                        if o.type==OptionType.ABILITY:return 650
                        if o.type==OptionType.RETREAT:return 420
                        if o.type==OptionType.END:return 0
                        return 100
                    cand=[i for i in range(len(opts)) if i!=bi and tempo_score(i)>-10**8]
                    if cand:
                        j=max(cand,key=lambda i:(tempo_score(i),-i))
                        _PLANNER.note_override('universal:ppp_no_attack_guard');return [j]
            except Exception:pass

        # CYNTHIA GUARANTEED PRIZE CONVERSION -------------------------------
        # Scoped to the public Cynthia Garchomp line.  Commit Boss -> target -> an
        # attack that is already legal now only when the current Active cannot be KO'd
        # and a damaged 2/3-Prize Bench Pokemon can. No future search/PPP/attachment.
        if plan.archetype=='cynthia':
            if self.cynthia_boss_turn!=int(obs.current.turn or 0):
                self.cynthia_boss_turn=-1;self.cynthia_boss_target=-1;self.cynthia_boss_attack=-1
            if self.cynthia_boss_target>=0 and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
                for i,o in enumerate(opts):
                    q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is not None and q.serial==self.cynthia_boss_target:
                        _PLANNER.note_override('cynthia:boss_prize_target');return [i]
            if self.cynthia_boss_target>=0 and ctx==SelectContext.MAIN:
                for i,o in enumerate(opts):
                    if o.type==OptionType.ATTACK and o.attackId==self.cynthia_boss_attack:
                        _PLANNER.note_override('cynthia:boss_prize_attack')
                        self.cynthia_boss_target=-1;self.cynthia_boss_attack=-1
                        return [i]
            if ctx==SelectContext.MAIN and not obs.current.supporterPlayed and mine.active and op.active:
                boss_i=None
                for i,o in enumerate(opts):
                    if o.type==OptionType.PLAY:
                        c0=_card(obs,AreaType.HAND,o.index,me)
                        if c0 is not None and c0.id==C.BOSS:boss_i=i;break
                attacks=[o for o in opts if o.type==OptionType.ATTACK]
                a0=mine.active[0]
                def _dmg(aid,target):
                    try:return int(_close_mod.attack_damage(a0,aid,op,target))
                    except Exception:return 0
                active_kill=any(_dmg(o.attackId,op.active[0])>=int(op.active[0].hp or 0) for o in attacks)
                if boss_i is not None and attacks and not active_kill:
                    cand=[]
                    for q in (op.bench or []):
                        if q is None:continue
                        cd=CARD.get(q.id);pr=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
                        if pr<2:continue
                        for o in attacks:
                            d=_dmg(o.attackId,q)
                            if d>=int(q.hp or 0):
                                lowcost=1 if o.attackId==AURA_JAB else 0
                                cand.append((pr,lowcost,-(d-int(q.hp or 0)),q.serial,o.attackId))
                    if cand:
                        cand.sort(reverse=True);_,_,_,ts,aid=cand[0]
                        self.cynthia_boss_turn=int(obs.current.turn or 0);self.cynthia_boss_target=ts;self.cynthia_boss_attack=aid
                        _PLANNER.note_override('cynthia:boss_prize_start');return [boss_i]

        # SEARCH OPTION-VALUE CONSERVATION ---------------------------------
        # Poffin is the narrower setup resource (<=70 HP Basics directly to Bench),
        # while Poké Pad can later find Riolu 80 / Solrock / Lunatone / Dudunsparce.
        # Use the narrower resource first only where the current macro plan makes the
        # dominance clear; broad Poffin-first rules regressed Crustle/Archaludon/mirror.
        if ctx==SelectContext.MAIN and base and len(base)==1 and len(mine.prize or [])>2:
            try:
                bo=opts[base[0]];bc=_card(obs,AreaType.HAND,bo.index,me) if bo.type==OptionType.PLAY else None
                if bc is not None and bc.id==C.POKE_PAD:
                    poffin=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.PLAY:continue
                        c0=_card(obs,AreaType.HAND,o.index,me)
                        if c0 is not None and c0.id==C.POFFIN:poffin.append(i)
                    bench_room=int(getattr(mine,'benchMax',5) or 5)-len([q for q in (mine.bench or []) if q is not None])
                    luc_lines=sum(1 for q in _field(mine) if q.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})
                    use_poffin=False
                    # Dragapult: repeated 1000-game validation favored establishing the
                    # second Lucario body with Poffin while preserving Pad for later.
                    if plan.archetype=='dragapult' and bench_room>=2 and luc_lines<2:
                        use_poffin=True
                    # Mirror: only the strongest dominance state. Engine pair already
                    # exists and Mega is in hand, so Pad's unique immediate targets are
                    # not needed; Poffin can create the missing second Riolu directly.
                    elif plan.archetype=='lucario' and bench_room>=2 and luc_lines<2:
                        sol=_find_field(mine,C.SOLROCK) is not None;lun=_find_field(mine,C.LUNATONE) is not None
                        mega_hand=any(c0.id==C.LUCARIO for c0 in (mine.hand or []))
                        if sol and lun and mega_hand:use_poffin=True
                    # Marnie replay 92682034 spent the flexible Pad before Poffin,
                    # then Poffin produced two Dunsparce while only one Lucario line
                    # existed.  Preserve Pad and establish the second Riolu first.
                    elif plan.archetype=='marnie' and bench_room>=2 and luc_lines<2:
                        use_poffin=True
                    if use_poffin and poffin:
                        _PLANNER.note_override('search_value:poffin_before_pad');return [poffin[0]]
            except Exception:pass

        # MARNIE POFFIN ATTACKER-CONTINUITY RESCUE ---------------------------
        # Losses 92682034 and 92777659 selected duplicate Dunsparce while a legal
        # Riolu and fewer than two Lucario lines were present.  Replace only one
        # utility duplicate; existing mixed targets and all other matchups remain.
        if plan.archetype=='marnie' and ctx==SelectContext.TO_BENCH and getattr(getattr(obs.select,'effect',None),'id',-1)==C.POFFIN and base:
            try:
                lines=sum(1 for q in _field(mine) if q.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})
                if lines<2:
                    chosen=[]
                    for bi in list(base):
                        if 0<=bi<len(opts):
                            q=_card(obs,opts[bi].area,opts[bi].index,getattr(opts[bi],'playerIndex',me))
                            if q is not None:chosen.append((bi,q.id))
                    duns=[bi for bi,cid in chosen if cid==C.DUNSPARCE]
                    ri=[]
                    for i,o in enumerate(opts):
                        q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                        if q is not None and q.id==C.RIOLU70 and i not in list(base):ri.append(i)
                    if len(duns)>=2 and ri:
                        new=list(base);new.remove(duns[-1]);new.append(ri[0])
                        _PLANNER.note_override('marnie:poffin_continuity_rescue');return self._merge_multi([],new,obs)
            except Exception:pass

        # MARNIE HERO'S CAPE ROLE ASSIGNMENT --------------------------------
        # Replay 92694296 put the only +100 HP tool on Solrock while the lone
        # Riolu line was exposed and subsequently removed.  In this matchup the
        # Cape protects three-Prize attack continuity, not the draw engine.
        if plan.archetype=='marnie' and ctx==SelectContext.MAIN and base and len(base)==1:
            try:
                bo=opts[base[0]]
                if bo.type==OptionType.ATTACH:
                    tool=_card(obs,AreaType.HAND,bo.index,me);target=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                    if tool is not None and tool.id==C.HERO_CAPE and target is not None and target.id not in {C.RIOLU70,C.RIOLU80,C.LUCARIO}:
                        cap=[]
                        for i,o in enumerate(opts):
                            if o.type!=OptionType.ATTACH:continue
                            c0=_card(obs,AreaType.HAND,o.index,me);q=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                            if c0 is None or q is None or c0.id!=C.HERO_CAPE or q.id not in {C.RIOLU70,C.RIOLU80,C.LUCARIO}:continue
                            cap.append((100 if q.id==C.LUCARIO else 50,20*len(q.energies or []),-i,i))
                        if cap:
                            cap.sort(reverse=True);_PLANNER.note_override('marnie:cape_attacker_continuity');return [cap[0][-1]]
            except Exception:pass

        # Poffin target diversity is helpful in Dragapult WAVE_BUILD only. If the
        # frozen executor selects two Dunsparce while the second Lucario line is still
        # missing and Riolu 70 is legal, replace just one duplicate with Riolu. Do not
        # rewrite Poffin targets in other matchups.
        if plan.archetype=='dragapult' and ctx==SelectContext.TO_BENCH and getattr(getattr(obs.select,'effect',None),'id',-1)==C.POFFIN and base:
            try:
                luc_lines=sum(1 for q in _field(mine) if q.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})
                if luc_lines<2:
                    chosen=[]
                    for bi in list(base):
                        if 0<=bi<len(opts):
                            q=_card(obs,opts[bi].area,opts[bi].index,getattr(opts[bi],'playerIndex',me))
                            if q is not None:chosen.append((bi,q.id))
                    duns=[bi for bi,cid in chosen if cid==C.DUNSPARCE]
                    ri=[]
                    for i,o in enumerate(opts):
                        q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                        if q is not None and q.id==C.RIOLU70 and i not in list(base):ri.append(i)
                    if len(duns)>=2 and ri:
                        new=list(base);new.remove(duns[-1]);new.append(ri[0])
                        _PLANNER.note_override('search_value:poffin_diversify_riolu');return new
            except Exception:pass

        # REPLAY-500 PAD CONTINUITY FLOOR -----------------------------------
        # Pad chose draw utility / an engine half in hundreds of recorded loss
        # searches while fewer than two Lucario bodies existed and a legal Riolu was
        # present.  Poffin cannot fetch the 80-HP Riolu, so preserve Pad's unique
        # attacker-body value.  The 1,000-game Dragapult gate rejected this override,
        # so Dragapult and mirror retain their v143 search policy unchanged.
        if plan.archetype=='marnie' and ctx==SelectContext.TO_HAND and getattr(getattr(obs.select,'effect',None),'id',-1)==C.POKE_PAD:
            try:
                lines=sum(1 for q in _field(mine) if q.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})
                hand_has_riolu=any(c0.id in {C.RIOLU70,C.RIOLU80} for c0 in (mine.hand or []))
                bench_room=int(getattr(mine,'benchMax',5) or 5)-len([q for q in (mine.bench or []) if q is not None])
                if lines<2 and not hand_has_riolu and bench_room>0:
                    ri=[]
                    for i,o in enumerate(opts):
                        q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                        if q is not None and q.id in {C.RIOLU80,C.RIOLU70}:ri.append((1 if q.id==C.RIOLU80 else 0,-i,i))
                    if ri:
                        ri.sort(reverse=True);_PLANNER.note_override('replay500:pad_second_riolu');return [ri[0][-1]] if obs.select.maxCount==1 else self._merge_multi(base,[ri[0][-1]],obs)
            except Exception:pass

        # REAL-REPLAY MIRROR LONE-BOARD RESCUE ------------------------------
        # Replay 92877716 had exactly one Pokemon in play: an Active Dunsparce.
        # Poké Pad selected Dudunsparce, leaving no Bench target and losing
        # before the deck could attack.  The broader zero-Lucario-line version
        # regressed the 1,000-game mirror A/B, so retain only this exact survival
        # state and fetch the more durable 80-HP Riolu when it is legal.
        if plan.archetype=='lucario' and ctx==SelectContext.TO_HAND and getattr(getattr(obs.select,'effect',None),'id',-1)==C.POKE_PAD:
            try:
                own_field=_field(mine)
                hand_has_riolu=any(c0.id in {C.RIOLU70,C.RIOLU80} for c0 in (mine.hand or []))
                lone_active_dun=(len(own_field)==1 and bool(mine.active) and mine.active[0].id==C.DUNSPARCE and len([q for q in (mine.bench or []) if q is not None])==0)
                if lone_active_dun and not hand_has_riolu:
                    ri=[]
                    for i,o in enumerate(opts):
                        q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                        if q is not None and q.id in {C.RIOLU80,C.RIOLU70}:ri.append((1 if q.id==C.RIOLU80 else 0,-i,i))
                    if ri:
                        ri.sort(reverse=True);_PLANNER.note_override('replay92877716:mirror_lone_dunsparce_pad_riolu');return [ri[0][-1]] if obs.select.maxCount==1 else self._merge_multi(base,[ri[0][-1]],obs)
            except Exception:pass

        # TOP-RANK ENGINE FLOOR ------------------------------------------------
        # Transfer the strategic invariant, not the teacher's old card package:
        # in early Dragapult turns with at most one Lucario line, finish the missing
        # Solrock/Lunatone half before spending Pad on utility.  This is deliberately
        # narrow; once two attack lines exist, Pad keeps its normal wider option value.
        if plan.archetype=='dragapult' and ctx==SelectContext.TO_HAND and getattr(getattr(obs.select,'effect',None),'id',-1)==C.POKE_PAD and int(obs.current.turn or 0)<=5:
            try:
                lines=sum(1 for q in _field(mine) if q.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})
                if lines<=1:
                    hand_ids={c0.id for c0 in (mine.hand or [])}
                    field_ids={q.id for q in _field(mine)}
                    wanted=[]
                    if C.SOLROCK not in field_ids and C.SOLROCK not in hand_ids:wanted.append(C.SOLROCK)
                    if C.LUNATONE not in field_ids and C.LUNATONE not in hand_ids:wanted.append(C.LUNATONE)
                    if wanted:
                        cand=[]
                        for i,o in enumerate(opts):
                            c0=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                            if c0 is not None and c0.id in wanted:cand.append((wanted.index(c0.id),i,c0.id))
                        if cand:
                            cand.sort();_PLANNER.note_override('top_replay:early_engine_floor');return [cand[0][1]] if obs.select.maxCount==1 else self._merge_multi(base,[cand[0][1]],obs)
            except Exception:pass

        # Never override a base action that already wins the game with a direct attack.
        if base and len(base)==1:
            try:
                bo=opts[base[0]]
                if bo.type==OptionType.ATTACK and op.active and len(mine.prize or [])<= (3 if CARD[op.active[0].id].megaEx else 2 if CARD[op.active[0].id].ex else 1):
                    return base
            except Exception:pass

        # --- MACRO TREE: future attacker wave --------------------------------
        # Use the learned depth-3 tree only in matchups where replay/self-play data
        # consistently rewards multiple ready Lucario waves and no stronger dedicated
        # route (Crustle bypass / Archaludon Ogerpon / Alakazam disruption) exists.
        if tree is not None and tree.mode=='WAVE_BUILD' and tree.margin>=0.018 and plan.archetype in {'dragapult','lucario'} and ctx==SelectContext.MAIN:
            if not obs.current.supporterPlayed:
                has_riolu=any(p0.id in {C.RIOLU70,C.RIOLU80} for p0 in _field(mine))
                has_mega_hand=any(c0.id==C.LUCARIO for c0 in (mine.hand or []))
                if has_riolu and not has_mega_hand:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.PLAY:
                            c0=_card(obs,AreaType.HAND,o.index,me)
                            if c0 is not None and c0.id==C.HILDA:
                                _PLANNER.note_override('macro_tree:hilda_wave');return [i]
            if sum(1 for p0 in _field(mine) if p0.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})<2:
                for i,o in enumerate(opts):
                    if o.type==OptionType.PLAY:
                        c0=_card(obs,AreaType.HAND,o.index,me)
                        if c0 is not None and c0.id==C.POFFIN:
                            _PLANNER.note_override('macro_tree:poffin_wave');return [i]
        if _MT.cached.mode=='WAVE_BUILD' and plan.archetype in {'dragapult','lucario'} and ctx==SelectContext.TO_HAND:
            if any(p0.id in {C.RIOLU70,C.RIOLU80} for p0 in _field(mine)):
                for i,o in enumerate(opts):
                    c0=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if c0 is not None and c0.id==C.LUCARIO:
                        _PLANNER.note_override('macro_tree:search_wave');return [i] if obs.select.maxCount==1 else self._merge_multi(base,[i],obs)
        if _MT.cached.mode=='WAVE_BUILD' and plan.archetype in {'dragapult','lucario'} and ctx==SelectContext.TO_BENCH:
            lines=sum(1 for p0 in _field(mine) if p0.id in {C.RIOLU70,C.RIOLU80,C.LUCARIO})
            if lines<2:
                for i,o in enumerate(opts):
                    q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is not None and q.id in {C.RIOLU80,C.RIOLU70}:
                        _PLANNER.note_override('macro_tree:bench_riolu_wave');return [i] if obs.select.maxCount==1 else self._merge_multi(base,[i],obs)

        # --- REPLAY-DERIVED BRIDGE TEMPO -----------------------------------
        # Historical high-performing Lucario games farm exposed one-Prize evolution
        # bridges with the cheapest attack, then save Mega Brave / damage modifiers for
        # the real Stage-2 target.  The old list used Hariyama as a free catcher; the
        # current list transfers only that *targeting role*: Boss timing stays with the
        # frozen executor, and this layer intervenes only after a forced-active selection opens.
        if plan.archetype in {'marnie','alakazam'} and ctx==SelectContext.MAIN and mine.active and op.active:
            a0=mine.active[0];t0=op.active[0]
            attacks={o.attackId:i for i,o in enumerate(opts) if o.type==OptionType.ATTACK}
            bridge_ids={C.MORGREM,C.IMPIDIMP} if plan.archetype=='marnie' else {C.KADABRA,C.ABRA}
            # If the bridge is already Active, use the lowest-cost exact attack and do
            # not burn PPP/Mega Brave for damage that Aura Jab/Cosmic Beam already covers.
            if t0.id in bridge_ids:
                base_ppp=any(0<=bi<len(opts) and opts[bi].type==OptionType.PLAY and (_card(obs,AreaType.HAND,opts[bi].index,me) is not None and _card(obs,AreaType.HAND,opts[bi].index,me).id==C.PPP) for bi in list(base or []))
                if a0.id==C.SOLROCK and _find_field(mine,C.LUNATONE) is not None and 980 in attacks and t0.hp<=70:
                    if base_ppp:_PLANNER.note_override(f'{plan.archetype}:save_ppp_bridge')
                    else:_PLANNER.note_override(f'{plan.archetype}:cheap_bridge_attack')
                    return [attacks[980]]
                if a0.id==C.LUCARIO and AURA_JAB in attacks and t0.hp<=130:
                    base_mega=any(0<=bi<len(opts) and opts[bi].type==OptionType.ATTACK and opts[bi].attackId==MEGA_BRAVE for bi in list(base or []))
                    if base_ppp or base_mega:
                        _PLANNER.note_override(f'{plan.archetype}:aura_bridge_exact');return [attacks[AURA_JAB]]


        # --- ALAKAZAM: VOLUNTARY ACTIVE-EVOLUTION PRIZE SHIELD ----------
        # Extremely narrow replay-proven invariant.  When Powerful Hand is online,
        # the opponent needs exactly 2 Prizes, has a very large (15+) hand, and the one-Prize Riolu is already a mature 2-Energy shield, and a one-Prize Active Riolu would become a
        # guaranteed-lethal three-Prize Mega without ending the game this turn, keep
        # the one-Prize shield.  Forced promotions are never touched.
        if plan.archetype=='alakazam' and ctx==SelectContext.MAIN and base and len(base)==1 and mine.active and op.active:
            try:
                bi=base[0];bo=opts[bi];a0=mine.active[0];t0=op.active[0]
                if bo.type==OptionType.EVOLVE and a0.id in {C.RIOLU70,C.RIOLU80} and t0.id==C.ALAKAZAM and len(t0.energies or [])>=1 and len(op.prize or [])==2 and int(op.handCount or 0)>=15 and len(a0.energies or [])>=2:
                    evo=_card(obs,bo.area,bo.index,me);bp=_card(obs,bo.inPlayArea,bo.inPlayIndex,me)
                    if evo is not None and bp is not None and evo.id==C.LUCARIO and bp.serial==a0.serial:
                        protected=any(getattr(e,'id',None)==C.ROCK_F for e in (a0.energyCards or []))
                        damage=max(0,int(CARD[a0.id].hp)-int(a0.hp or 0));mega_hp=max(1,int(CARD[C.LUCARIO].hp)-damage)
                        incoming=20*(int(op.handCount or 0)+1)
                        td=CARD[t0.id];tp=3 if td.megaEx else 2 if td.ex else 1
                        en=len(a0.energies or []);resist=30 if getattr(td,'resistance',None)==EnergyType.FIGHTING else 0
                        ppp=sum(1 for c0 in (mine.hand or []) if c0.id==C.PPP)
                        best=max(0,(270 if en>=2 else 130 if en>=1 else 0)+30*ppp-resist)
                        ends_now=(len(mine.prize or [])<=tp and best>=int(t0.hp or 0))
                        if not protected and incoming>=mega_hp and not ends_now:
                            attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
                            if attacks:
                                _PLANNER.note_override('alakazam:hold_one_prize_active');return [attacks[0]]
                            ends=[i for i,o in enumerate(opts) if o.type==OptionType.END]
                            if ends:
                                _PLANNER.note_override('alakazam:hold_one_prize_active');return [ends[0]]
            except Exception:pass

        # --- ALAKAZAM: THREE-PRIZE EXPOSURE SUBGOAL -------------------------
        # Preserve a one-Prize Active against a double-digit Powerful-Hand hand
        # unless the incoming Mega Lucario is protected by Rock Energy or actually
        # ends the Alakazam exchange. Forced promotions and ready Mega Brave lines
        # remain entirely in the frozen tactical executor.
        if plan.archetype=='alakazam' and ctx==SelectContext.MAIN and op.active and op.active[0] is not None and op.active[0].id==C.ALAKAZAM and op.handCount>=10 and mine.active and mine.active[0] is not None:
            a0=mine.active[0];t0=op.active[0]
            if not (CARD[a0.id].ex or CARD[a0.id].megaEx):
                base_retreat=any(0<=bi<len(opts) and opts[bi].type==OptionType.RETREAT for bi in list(base or []))
                if base_retreat:
                    ppp=sum(1 for c0 in (mine.hand or []) if c0.id==C.PPP)
                    mountain=any(c0.id==C.MOUNTAIN for c0 in (mine.hand or []))
                    safe=False
                    for q in (mine.bench or []):
                        if q is None or q.id!=C.LUCARIO:continue
                        en=len(q.energies or [])
                        rock=_has_rock(q)
                        aura_net=100+30*ppp
                        target_hp=t0.hp-(30 if mountain and CARD[t0.id].stage2 else 0)
                        if rock or (en>=2 and 240>=target_hp) or (en>=1 and aura_net>=target_hp):
                            safe=True;break
                    if not safe:
                        for i,o in enumerate(opts):
                            if o.type==OptionType.ATTACK and o.attackId==980:
                                _PLANNER.note_override('alakazam:high_hand_hold_cosmic');return [i]
                        for i,o in enumerate(opts):
                            if o.type==OptionType.END:
                                _PLANNER.note_override('alakazam:block_high_hand_lucario_entry');return [i]

        # Replay-derived role assignment: outside matchups where Dudunsparce ex
        # is itself the win condition, Dunsparce/Dudunsparce are draw-pivot pieces, not
        # Fighting-Energy sinks.  Alakazam in particular punishes the 3-Energy line with
        # Psychic while the same Energy on Lucario advances the actual Prize plan.
        if plan.archetype=='alakazam' and ctx==SelectContext.MAIN and not obs.current.energyAttached and base:
            try:
                bo=opts[base[0]]
                if bo.type==OptionType.ATTACH:
                    tc=_card(obs,bo.inPlayArea,bo.inPlayIndex,me);ec=_card(obs,AreaType.HAND,bo.index,me)
                    if tc is not None and ec is not None and tc.id in {C.DUNSPARCE,C.DUDUN,C.DUDUN_EX} and ec.id in {C.BASIC_F,C.ROCK_F}:
                        cand=[]
                        for i,o in enumerate(opts):
                            if o.type!=OptionType.ATTACH:continue
                            q=_card(obs,o.inPlayArea,o.inPlayIndex,me);e=_card(obs,AreaType.HAND,o.index,me)
                            if q is None or e is None or e.id!=ec.id:continue
                            if q.id==C.LUCARIO and len(q.energies)<2:cand.append((120-20*len(q.energies),i))
                            elif q.id in {C.RIOLU70,C.RIOLU80}:cand.append((90-10*len(q.energies),i))
                            elif q.id==C.SOLROCK and _find_field(mine,C.LUNATONE) is not None:cand.append((70-10*len(q.energies),i))
                        if cand:
                            cand.sort(reverse=True);_PLANNER.note_override(f'{plan.archetype}:redirect_dudun_energy');return [cand[0][1]]
            except Exception:pass

        # --- ALAKAZAM: PPP ATTACK-COUNT CONSERVATION --------------------
        # Cosmic Beam is fixed 70 damage and ignores Weakness/Resistance.  Do not
        # spend an additional one-turn PPP when even all remaining PPP copies leave
        # the number of Cosmic Beam hits-to-KO unchanged.  Already-played PPP copies
        # are tracked above so legitimate multi-PPP exact thresholds remain allowed.
        if plan.archetype=='alakazam' and ctx==SelectContext.MAIN and base and len(base)==1 and mine.active and op.active:
            try:
                bi=base[0];bo=opts[bi];bc=_card(obs,AreaType.HAND,bo.index,me) if bo.type==OptionType.PLAY else None
                a0=mine.active[0];t0=op.active[0]
                if bc is not None and bc.id==C.PPP and a0.id==C.SOLROCK and _find_field(mine,C.LUNATONE) is not None and t0.id in {C.ALAKAZAM,C.ALAKAZAM_ALT}:
                    cosmic=next((i for i,o in enumerate(opts) if o.type==OptionType.ATTACK and o.attackId==980),None)
                    if cosmic is not None:
                        remain=sum(1 for c0 in (mine.hand or []) if c0.id==C.PPP)
                        hp=max(1,int(t0.hp or 0));now_d=70+30*self.alak_ppp_used;max_d=now_d+30*remain
                        now_hits=(hp+now_d-1)//now_d;max_hits=(hp+max_d-1)//max_d
                        if max_hits>=now_hits:
                            _PLANNER.note_override('alakazam:ppp_no_attack_count_gain');return [cosmic]
            except Exception:pass

        # --- MARNIE: NEXT-LUCARIO-WAVE SUBGOAL ----------------------------
        # Riolu is future three-Prize attack capital.  The legacy tactical stack can
        # incorrectly consider a Dudunsparce evolution in hand as satisfying the need
        # for an evolution card.  In the Marnie matchup, if a Riolu is waiting and no
        # Mega Lucario is available in hand, Hilda deterministically completes the next
        # attacking wave.  Do not interrupt an EXACT_PRIZE finish.
        if plan.archetype=='marnie' and plan.strategy!='EXACT_PRIZE' and ctx==SelectContext.MAIN and not obs.current.supporterPlayed:
            riolu_wait=any(p0.id in {C.RIOLU70,C.RIOLU80} for p0 in _field(mine))
            mega_hand=any(c0.id==C.LUCARIO for c0 in (mine.hand or []))
            luc_field=sum(1 for p0 in _field(mine) if p0.id==C.LUCARIO)
            if riolu_wait and not mega_hand and luc_field<2:
                for i,o in enumerate(opts):
                    if o.type==OptionType.PLAY:
                        c0=_card(obs,AreaType.HAND,o.index,me)
                        if c0 is not None and c0.id==C.HILDA:
                            _PLANNER.note_override('marnie:hilda_next_lucario_wave');return [i]
        if plan.archetype=='marnie' and ctx==SelectContext.TO_HAND:
            mega=[]
            if any(p0.id in {C.RIOLU70,C.RIOLU80} for p0 in _field(mine)):
                for i,o in enumerate(opts):
                    c0=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if c0 is not None and c0.id==C.LUCARIO:mega.append(i)
            if mega:
                _PLANNER.note_override('marnie:search_next_lucario_wave');return [mega[0]] if obs.select.maxCount==1 else self._merge_multi(base,mega,obs)

        # A second Rock can still be a legitimate second attack Energy when every
        # Lucario line is already protected.  A third Energy on that same line is
        # pure protection/tempo waste: redirect it, attack, or end instead.
        if plan.archetype=='alakazam' and ctx==SelectContext.MAIN and base and len(base)==1 and not obs.current.energyAttached:
            try:
                bo=opts[base[0]];bc=_card(obs,AreaType.HAND,bo.index,me) if bo.type==OptionType.ATTACH else None
                bp=_card(obs,bo.inPlayArea,bo.inPlayIndex,me) if bo.type==OptionType.ATTACH else None
                if bc is not None and bp is not None and bc.id==C.ROCK_F and _is_lucario_line(bp) and _has_rock(bp) and len(bp.energies or [])>=2:
                    fresh=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        c0=_card(obs,AreaType.HAND,o.index,me);p0=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if c0 is not None and p0 is not None and c0.id==C.ROCK_F and _is_lucario_line(p0) and (not _has_rock(p0) or len(p0.energies or [])<2):fresh.append(i)
                    if fresh:
                        _PLANNER.note_override('alakazam:rock_spread_before_stack');return [fresh[0]]
                    attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
                    if attacks:
                        attacks.sort(key=lambda i:(opts[i].attackId==MEGA_BRAVE,opts[i].attackId==AURA_JAB),reverse=True)
                        _PLANNER.note_override('alakazam:block_third_rock');return [attacks[0]]
                    ends=[i for i,o in enumerate(opts) if o.type==OptionType.END]
                    if ends:
                        _PLANNER.note_override('alakazam:block_third_rock');return [ends[0]]
            except Exception:pass

        # --- HAND_DENIAL: Rock Fighting Energy is strategic protection against
        # Alakazam's damage-counter attack, not just another one-turn attachment.
        if plan.strategy=='HAND_DENIAL':
            if ctx==SelectContext.MAIN and not obs.current.energyAttached:
                rock=[]
                for i,o in enumerate(opts):
                    if o.type!=OptionType.ATTACH:continue
                    c=_card(obs,AreaType.HAND,o.index,me);p=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                    if c is not None and p is not None and c.id==C.ROCK_F and _is_lucario_line(p) and not _has_rock(p):rock.append(i)
                if rock:
                    def rs(i):
                        o=opts[i];p=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        return (100 if o.inPlayArea==AreaType.ACTIVE else 0)+(20 if p.id==C.LUCARIO else 0)-10*len(p.energies)
                    j=_choose_best(rock,rs);_PLANNER.note_override('alakazam:rock_attach');return [j]
            if ctx==SelectContext.MAIN and not obs.current.supporterPlayed and op.handCount>=5:
                # Hand denial is a defensive turn only when Powerful Hand is actually
                # online. Before that, spend the supporter on development instead.
                threat=bool(op.active and op.active[0].id==C.ALAKAZAM and len(op.active[0].energies)>=1)
                protected=bool(mine.active and mine.active[0].id==C.LUCARIO and _has_rock(mine.active[0]))
                if threat and not protected:
                    xs=[]
                    for i,o in enumerate(opts):
                        if o.type==OptionType.PLAY:
                            c=_card(obs,AreaType.HAND,o.index,me)
                            if c is not None and c.id==C.XEROSIC:xs.append(i)
                    if xs:_PLANNER.note_override('alakazam:xerosic_threat_gate');return [xs[0]]
            if ctx==SelectContext.TO_HAND and not obs.current.energyAttached:
                rock=[]
                for i,o in enumerate(opts):
                    c=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if c is not None and c.id==C.ROCK_F:rock.append(i)
                # Hilda can fetch the special Energy. Choose it only when the current board
                # already has a Lucario line to receive it.
                if rock and any(_is_lucario_line(p) and not _has_rock(p) for p in _field(mine)):
                    _PLANNER.note_override('alakazam:search_rock');return [rock[0]] if obs.select.maxCount==1 else self._merge_multi(base,rock,obs)

        # Alakazam resource invariant: when Solrock + Lunatone already takes
        # the bridge KO, keep Premium Power Pro for a later Lucario threshold.
        if plan.strategy=='HAND_DENIAL' and ctx==SelectContext.MAIN and mine.active and op.active:
            base_ppp=any(0<=bi<len(opts) and opts[bi].type==OptionType.PLAY and (_card(obs,AreaType.HAND,opts[bi].index,me) is not None and _card(obs,AreaType.HAND,opts[bi].index,me).id==C.PPP) for bi in list(base or []))
            if base_ppp:
                a0=mine.active[0];t0=op.active[0]
                if a0.id==C.SOLROCK and _find_field(mine,C.LUNATONE) is not None and t0.hp<=70:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.ATTACK and o.attackId==980:
                            _PLANNER.note_override('alakazam:save_ppp_solrock_exact');return [i]

        # --- BYPASS_CRUSTLE: one committed route for several turns.
        if plan.strategy=='BYPASS_CRUSTLE':
            primary=plan.primary
            oppa=op.active[0] if op.active else None
            # Current deck has only one Dudunsparce ex, so reserve one Poffin search
            # slot for its 70-HP Dunsparce base whenever no bypass body exists yet.
            if ctx==SelectContext.TO_BENCH and _find_field(mine,C.DUDUN_EX) is None and _find_field(mine,C.OGERPON) is None and _find_field(mine,C.DUNSPARCE) is None:
                ds=[]
                for i,o in enumerate(opts):
                    if getattr(o,'playerIndex',me)!=me:continue
                    q=_card(obs,o.area,o.index,me)
                    if q is not None and q.id==C.DUNSPARCE:ds.append(i)
                if ds:
                    _PLANNER.note_override('crustle:poffin_dunsparce_route');return [ds[0]] if obs.select.maxCount==1 else self._merge_multi(base,ds,obs)
            # Tempo routing before the bypass logic: opposing Cornerstone is a
            # two-Prize target that Mega Lucario can hit normally.  Do not leave a
            # charged Lucario stranded behind Dunsparce/Dudunsparce while taking
            # 60-damage Tail turns.
            ready_luc=None
            near_luc=None
            for p0 in (mine.bench or []):
                if p0 is None:continue
                if p0.id==C.LUCARIO and len(p0.energies)>=2 and ready_luc is None:ready_luc=p0
                if p0.id in {C.LUCARIO,C.RIOLU80,C.RIOLU70} and len(p0.energies)>=1:
                    if near_luc is None or len(p0.energies)>len(near_luc.energies):near_luc=p0
            if ctx==SelectContext.MAIN and mine.active:
                a0=mine.active[0]
                # Air Balloon is a matchup-level mobility resource.  If a utility
                # Active is blocking any charged Lucario line, free that Active now
                # even if the Riolu evolution arrives one turn later.
                base_balloon=False
                for bi in list(base or []):
                    if 0<=bi<len(opts) and opts[bi].type==OptionType.ATTACH:
                        c0=_card(obs,AreaType.HAND,opts[bi].index,me)
                        if c0 is not None and c0.id==C.AIR_BALLOON:base_balloon=True
                if base_balloon and near_luc is not None and a0.id in {C.DUNSPARCE,C.DUDUN,C.DUDUN_EX,C.SOLROCK,C.LUNATONE}:
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        c0=_card(obs,AreaType.HAND,o.index,me);p0=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if c0 is not None and p0 is not None and c0.id==C.AIR_BALLOON and p0.serial==a0.serial:
                            _PLANNER.note_override('crustle:balloon_active_pivot');return [i]
                # Opposing Cornerstone is a clean two-Prize Lucario target.
                if oppa is not None and oppa.id==C.OGERPON and ready_luc is not None and a0.id!=C.LUCARIO:
                    retreats=[i for i,o in enumerate(opts) if o.type==OptionType.RETREAT]
                    if retreats:
                        _PLANNER.note_override('crustle:pivot_to_lucario');return [retreats[0]]
            # Promotion after that retreat must land on the charged Lucario when the
            # target is Cornerstone Ogerpon.
            if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE} and oppa is not None and oppa.id==C.OGERPON and ready_luc is not None:
                for i,o in enumerate(opts):
                    if getattr(o,'playerIndex',me)!=me:continue
                    p0=_card(obs,o.area,o.index,me)
                    if p0 is not None and p0.serial==ready_luc.serial:
                        _PLANNER.note_override('crustle:promote_lucario_vs_oger');return [i]
            # Evolve the reserved Dunsparce into Dudunsparce ex; never consume the only
            # Dunsparce with Run Away Draw while the bypass route still needs it.
            if ctx==SelectContext.MAIN:
                # Hilda is the direct Evolution+Energy bridge for the one-copy
                # Dudunsparce ex. Promote only this deterministic completion step;
                # Fighting Gong remains unconstrained so Ogerpon stays a live backup.
                if primary==C.DUDUN_EX and _find_field(mine,C.DUNSPARCE) is not None and _find_field(mine,C.DUDUN_EX) is None and not any(c0.id==C.DUDUN_EX for c0 in (mine.hand or [])) and not any(c0.id==C.DUDUN_EX for c0 in (mine.discard or [])) and not obs.current.supporterPlayed:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.PLAY:
                            c0=_card(obs,AreaType.HAND,o.index,me)
                            if c0 is not None and c0.id==C.HILDA:
                                _PLANNER.note_override('crustle:hilda_complete_bypass');return [i]
                evol=[]
                for i,o in enumerate(opts):
                    if o.type!=OptionType.EVOLVE:continue
                    evo=_card(obs,o.area,o.index,me);basep=_card(obs,o.inPlayArea,o.inPlayIndex,me)
                    if primary==C.DUDUN_EX and evo is not None and basep is not None and evo.id==C.DUDUN_EX and basep.id==C.DUNSPARCE:evol.append(i)
                if evol:_PLANNER.note_override('crustle:evolve_dudun_ex');return [evol[0]]
                # Manual Energy stays on the committed bypass attacker until its threshold.
                target=_find_field(mine,primary)
                goal=3
                if target is not None and len(target.energies)<goal and not obs.current.energyAttached:
                    att=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        p=_card(obs,o.inPlayArea,o.inPlayIndex,me);c=_card(obs,AreaType.HAND,o.index,me)
                        if p is not None and c is not None and p.serial==target.serial and c.id in {C.BASIC_F,C.ROCK_F}:att.append(i)
                    if att:_PLANNER.note_override('crustle:charge_primary');return [att[0]]
                # Once the bypass attacker is ready, put it Active instead of continuing
                # Lucario setup into Mysterious Rock Inn.
                if target is not None and len(target.energies)>=3 and mine.active and mine.active[0].serial!=target.serial:
                    retreats=[i for i,o in enumerate(opts) if o.type==OptionType.RETREAT]
                    if retreats:_PLANNER.note_override('crustle:retreat_to_bypass');return [retreats[0]]
            # Aura Jab allocation: all accelerated Basic Fighting Energy goes to the
            # committed bypass attacker before speculative Lucario charging.
            if ctx==SelectContext.ATTACH_FROM:
                target=_find_field(mine,primary)
                if target is not None and len(target.energies)<3:
                    hits=[]
                    for i,o in enumerate(opts):
                        p=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                        if p is not None and p.serial==target.serial:hits.append(i)
                    if hits:_PLANNER.note_override('crustle:aura_to_primary');return [hits[0]]
            if ctx==SelectContext.TO_HAND:
                wanted=[]
                for i,o in enumerate(opts):
                    c=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if c is None:continue
                    if primary==C.DUDUN_EX and c.id==C.DUDUN_EX and _find_field(mine,C.DUNSPARCE):wanted.append(i)
                    if primary==C.OGERPON and c.id==C.OGERPON and _find_field(mine,C.OGERPON) is None:wanted.append(i)
                if wanted:_PLANNER.note_override('crustle:search_primary');return [wanted[0]] if obs.select.maxCount==1 else self._merge_multi(base,wanted,obs)
            if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
                target=_find_field(mine,primary)
                if target is not None and len(target.energies)>=3:
                    for i,o in enumerate(opts):
                        p=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                        if p is not None and p.serial==target.serial:
                            _PLANNER.note_override('crustle:promote_primary');return [i]
            # PPP only boosts attacks from Fighting Pokemon. Cornerstone Ogerpon is
            # eligible; Colorless Dudunsparce ex is not. Keep the exact-KO resource
            # gate only on the legal Demolish route.
            if ctx==SelectContext.MAIN and mine.active and op.active and op.active[0].id==C.CRUSTLE and mine.active[0].id==C.OGERPON:
                a0=mine.active[0];t0=op.active[0];aid0=DEMOLISH;base0=140
                if any(o.type==OptionType.ATTACK and o.attackId==aid0 for o in opts) and base0<t0.hp<=base0+30:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.PLAY:
                            c0=_card(obs,AreaType.HAND,o.index,me)
                            if c0 is not None and c0.id==C.PPP:
                                _PLANNER.note_override('crustle:ppp_exact_ogerpon');return [i]
            # Exact bypass attack when Crustle is Active.
            if ctx==SelectContext.MAIN and mine.active and op.active and op.active[0].id==C.CRUSTLE:
                aid=DRILL if mine.active[0].id==C.DUDUN_EX else DEMOLISH if mine.active[0].id==C.OGERPON else None
                if aid is not None:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.ATTACK and o.attackId==aid:
                            _PLANNER.note_override('crustle:bypass_attack');return [i]

        # --- DUDUN_PRESSURE: Tera Box with several ex Pokemon.  One Energy is enough
        # for Tenacious Tail, so do not waste three turns charging Drill first.
        if plan.strategy=='DUDUN_PRESSURE':
            if ctx==SelectContext.MAIN:
                target=_find_field(mine,C.DUDUN_EX)
                if target is not None and len(target.energies)<1 and not obs.current.energyAttached:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.ATTACH:
                            p=_card(obs,o.inPlayArea,o.inPlayIndex,me);c=_card(obs,AreaType.HAND,o.index,me)
                            if p is not None and c is not None and p.serial==target.serial and c.id in {C.BASIC_F,C.ROCK_F}:
                                _PLANNER.note_override('terabox:one_energy_dudun');return [i]
                if mine.active and mine.active[0].id==C.DUDUN_EX and len(mine.active[0].energies)>=1:
                    for i,o in enumerate(opts):
                        if o.type==OptionType.ATTACK and o.attackId==TENACIOUS:
                            _PLANNER.note_override('terabox:tenacious');return [i]

        # --- TEAL_ABILITY_WALL: the public all-Teal board is damage-locked by
        # Cornerstone Stance.  Commit search, deployment, charging and promotion
        # as one route so the old Tera-box Dudunsparce preference cannot split it.
        if plan.strategy=='TEAL_ABILITY_WALL':
            oger=_find_field(mine,C.OGERPON)
            hand_has_oger=any(c0.id==C.OGERPON for c0 in (mine.hand or []))
            if ctx==SelectContext.TO_HAND and oger is None and not hand_has_oger:
                wanted=[]
                for i,o in enumerate(opts):
                    q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is not None and q.id==C.OGERPON:wanted.append(i)
                if wanted:
                    _PLANNER.note_override('replay92883371:teal_wall_search_ogerpon');return [wanted[0]] if obs.select.maxCount==1 else self._merge_multi(base,wanted,obs)
            if ctx==SelectContext.MAIN:
                if oger is None and hand_has_oger:
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.PLAY:continue
                        q=_card(obs,AreaType.HAND,o.index,me)
                        if q is not None and q.id==C.OGERPON:
                            _PLANNER.note_override('replay92883371:teal_wall_play_ogerpon');return [i]
                if oger is not None and len(oger.energies)<3 and not obs.current.energyAttached:
                    attach=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.ATTACH:continue
                        q=_card(obs,o.inPlayArea,o.inPlayIndex,me);e=_card(obs,AreaType.HAND,o.index,me)
                        if q is not None and e is not None and q.serial==oger.serial and e.id in {C.BASIC_F,C.ROCK_F}:attach.append(i)
                    if attach:
                        _PLANNER.note_override('replay92883371:teal_wall_charge_ogerpon');return [attach[0]]
                if oger is not None and mine.active and mine.active[0].serial!=oger.serial:
                    retreats=[i for i,o in enumerate(opts) if o.type==OptionType.RETREAT]
                    if retreats:
                        _PLANNER.note_override('replay92883371:teal_wall_retreat');return [retreats[0]]
            if ctx==SelectContext.ATTACH_FROM and oger is not None and len(oger.energies)<3:
                wanted=[]
                for i,o in enumerate(opts):
                    q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is not None and q.serial==oger.serial:wanted.append(i)
                if wanted:
                    _PLANNER.note_override('replay92883371:teal_wall_accelerate_ogerpon');return [wanted[0]]
            if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE} and oger is not None:
                for i,o in enumerate(opts):
                    q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is not None and q.serial==oger.serial:
                        _PLANNER.note_override('replay92883371:teal_wall_promote_ogerpon');return [i]

        # Boss/forced-active target selection is a Prize-map decision, not a static
        # card priority.  Prefer an immediately KO-able 2/3-Prize target over a one-Prize
        # bridge, but add a denial bonus to Kadabra/Morgrem so cheap evolution KOs still
        # win when the larger target cannot be finished this turn.
        if (plan.strategy == 'HAND_DENIAL' or (plan.strategy == 'EVOLUTION_DENIAL' and plan.archetype == 'marnie')) and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
            a0=mine.active[0] if mine.active else None
            dmg=0
            if a0 is not None:
                en=len(a0.energies or [])
                if a0.id==C.LUCARIO:dmg=270 if en>=2 else 130 if en>=1 else 0
                elif a0.id==C.SOLROCK and en>=1 and _find_field(mine,C.LUNATONE) is not None:dmg=70
                elif a0.id in {C.RIOLU70,C.RIOLU80} and en>=1:dmg=30
                elif a0.id==C.DUDUN_EX:dmg=150 if en>=3 else 0
                elif a0.id==C.OGERPON:dmg=140 if en>=3 else 0
            scored=[];fallback=[]
            for i,o in enumerate(opts):
                if getattr(o,'playerIndex',me)!=1-me:continue
                q=_card(obs,o.area,o.index,1-me)
                if q is None:continue
                if q.id in plan.target_ids:fallback.append((plan.target_ids.index(q.id),i))
                if dmg>0 and q.hp<=dmg:
                    cd=CARD[q.id];pr=3 if cd.megaEx else 2 if cd.ex else 1
                    bonus=0
                    if plan.archetype=='alakazam':
                        if q.id in {C.ALAKAZAM,C.ALAKAZAM_ALT}:bonus=70
                        elif q.id==C.KADABRA:bonus=50
                        elif q.id==C.ABRA:bonus=30
                    elif plan.archetype=='marnie':
                        if q.id==C.GRIMMSNARL:bonus=70
                        elif q.id==C.MORGREM:bonus=50
                        elif q.id==C.IMPIDIMP:bonus=30
                    scored.append((pr*100+bonus,-q.hp,-i,i))
            if scored:
                scored.sort(reverse=True);_PLANNER.note_override(f'{plan.archetype}:prize_aware_target');return [scored[0][-1]]
            if fallback:
                fallback.sort();_PLANNER.note_override(f'{plan.archetype}:target_route_fallback');return [fallback[0][1]]

        return base

    def _merge_multi(self,base,wanted,obs):
        maxc=int(obs.select.maxCount or 0);minc=int(obs.select.minCount or 0)
        out=[]
        for i in wanted+list(base or []):
            if i not in out:out.append(i)
            if len(out)>=maxc:break
        if len(out)<minc:
            for i in range(len(obs.select.option or [])):
                if i not in out:out.append(i)
                if len(out)>=minc:break
        return out


_ARBITER=StrategyArbiter()


def agent(obs_dict:dict)->list[int]:
    # Reset all active strategic state at the deck-selection boundary.
    if obs_dict.get('select') is None and obs_dict.get('current') is None:
        # The frozen executor contains legacy per-game counters in addition to its
        # documented turn state.  Re-instantiating it once at the deck boundary is
        # safer than enumerating private counters and fixes cross-game leakage seen
        # when 45 replays were audited in one interpreter.
        global _legacy
        _legacy=_load('lucario_frozen_v118','frozen_v118_tactical_executor.py')
        _PLANNER.reset();_MT.reset();_ARBITER.reset();_rr.reset();_td.reset();_HISTORY.reset();_HIST_GATE.reset();_HISTORY_REPLAY.reset();_LEAGUE_REPLAY.reset();_TEMPORAL_GRU.reset();_TEMPORAL_ATTENTION.reset();_TEMPORAL_SAFETY.reset();_LOSS_REPAIR.reset()
    base=_legacy.agent(obs_dict)
    if obs_dict.get('select') is None and obs_dict.get('current') is None:return base
    # Build one history-conditioned input before any learned or strategic stage.
    # It contains the complete public action stream, our previous decisions, the
    # exact current self hand/field, and the opponent's public hand belief/field.
    try:_HISTORY.observe(obs_dict)
    except Exception:pass
    # Preserve the two validated tactical correction stages as one frozen executor.
    try:base=_rr.choose(obs_dict,base,_HISTORY.features(obs_dict))
    except Exception:pass
    try:base=_td.choose(obs_dict,base)
    except Exception:pass
    try:base=_HIST_GATE.choose(obs_dict,base)
    except Exception:pass
    # Replay-trained residual sees the complete action sequence and may replace
    # one risky supported action per turn.  Exact closeout/matchup constraints in
    # StrategyArbiter still run afterwards and remain authoritative.
    try:base=_HISTORY_REPLAY.choose(obs_dict,base)
    except Exception:pass
    # League residual was fitted on 900 complete self/opponent histories.  It is
    # empirically enabled only for Dragapult; the original official-replay layer
    # above remains authoritative in every other matchup.
    try:base=_LEAGUE_REPLAY.choose(obs_dict,base)
    except Exception:pass
    # The GRU consumes the entire ordered public prefix and every prior emitted
    # decision.  Its model file controls whether it may change an action; when
    # the validation gate is disabled it still maintains the recurrent state for
    # diagnostics without overriding the established policy.
    try:base=_TEMPORAL_GRU.choose(obs_dict,base)
    except Exception:pass
    # The stronger causal attention ensemble observes exactly the same ordered
    # prefix and emits risk/phase telemetry only.  ACTION_AUTHORITY is hardcoded
    # False in its runtime, so this call returns the exact base action object.
    try:base=_TEMPORAL_ATTENTION.choose(obs_dict,base)
    except Exception:pass
    out=_ARBITER.choose(obs_dict,base)
    try:out=_TEMPORAL_SAFETY.choose(obs_dict,out)
    except Exception:pass
    # Final authoritative gate distilled from the current nine human losses.
    # It uses only legal actions and repairs objectively dominated decisions
    # (resistance-blind Boss KOs, prize-feed promotions, and wall abandonment).
    try:out=_LOSS_REPAIR.choose(obs_dict,out,_PLANNER.plan)
    except Exception:pass
    try:_HISTORY.record_choice(obs_dict,out)
    except Exception:pass
    try:_CLOSE.record_action(to_observation_class(obs_dict),out)
    except Exception:pass
    return out


def get_strategy_stats():
    return {'planner':dict(_PLANNER.stats),'macro_tree':_MT.get_stats(),'closeout':_CLOSE.get_stats(),'regret':_rr.get_stats(),'terminal':_td.get_stats(),'history':_HISTORY.summary(),'history_gate':_HIST_GATE.get_stats(),'history_replay':_HISTORY_REPLAY.get_stats(),'league_replay':_LEAGUE_REPLAY.get_stats(),'temporal_gru':_TEMPORAL_GRU.get_stats(),'temporal_attention':_TEMPORAL_ATTENTION.get_stats(),'temporal_safety':_TEMPORAL_SAFETY.get_stats(),'loss_repair':_LOSS_REPAIR.get_stats()}

POLICY_RELEASE='v153_human_loss_reflection_ability_wall_prize_guard'
