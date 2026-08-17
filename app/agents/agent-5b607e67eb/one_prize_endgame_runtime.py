"""Replay-distilled one-Prize matchup and terminal-board safety guard.

This layer is deliberately narrow and deterministic.  It was added after human
Alakazam testing and a Lucario-vs-Team-Rocket loss replay exposed two failures:
(1) feeding a damaged 3-Prize Mega Lucario into a guaranteed one-Prize return KO,
and (2) manually attaching a second Fighting Energy to an already mobile Dunsparce
and then immediately Trading Places.  It also proves the Pokemon-TCG board-empty
win condition independently of remaining Prizes.
"""
from __future__ import annotations
from cg.api import AreaType,EnergyType,OptionType,SelectContext,all_attack,all_card_data,to_observation_class

CARD={c.cardId:c for c in all_card_data()}
ATT={a.attackId:a for a in all_attack()}

RIOLU70=333;RIOLU80=677;LUCARIO=678;SOLROCK=676;LUNATONE=675
DUNSPARCE=305;DUDUN=66;DUDUN_EX=306;OGERPON=117
BASIC_F=6;ROCK_F=20;PPP=1141;POFFIN=1086;POKE_PAD=1152;LILLIE=1227
AURA=982;MEGA=983;TRADING=423;RAM=424;COSMIC=980
ALAKAZAM=743;ALAKAZAM_ALT=245
TR_ARTICUNO=414;TR_MURKROW=463;TR_PORYGON=473;TR_PORYGON2=474;TR_PORYGONZ=475;TR_HONCHKROW=891
TR_SUPPORTERS={1216,1217,1218,1219,1220}
TR_ENERGY=15
ONE_PRIZE_ALA={741,742,743,245}
ONE_PRIZE_ROCKET={TR_ARTICUNO,TR_MURKROW,TR_PORYGON,TR_PORYGON2,TR_PORYGONZ,TR_HONCHKROW}


def _field(p):return [q for q in list(p.active or [])+list(p.bench or []) if q is not None]
def _pv(q):
    if q is None:return 1
    c=CARD.get(int(q.id))
    if c is None:return 1
    return 3 if bool(getattr(c,'megaEx',False)) else 2 if bool(getattr(c,'ex',False)) else 1

def _card(obs,area,index,player):
    try:
        ar=AreaType(int(area));p=obs.current.players[player]
        if ar==AreaType.HAND:return p.hand[index]
        if ar==AreaType.ACTIVE:return p.active[index]
        if ar==AreaType.BENCH:return p.bench[index]
        if ar==AreaType.DISCARD:return p.discard[index]
        if ar==AreaType.DECK:return obs.select.deck[index]
        if ar==AreaType.LOOKING:return obs.current.looking[index]
    except Exception:return None
    return None

def _target(obs,o,me):return _card(obs,o.inPlayArea,o.inPlayIndex,me)
def _source(obs,o,me):return _card(obs,AreaType.HAND,o.index,me)

def _fighting_resist(q):
    try:return getattr(CARD.get(int(q.id)),'resistance',None)==EnergyType.FIGHTING
    except Exception:return False

def _aura_damage(q):return max(0,130-(30 if _fighting_resist(q) else 0))

def _revealed_family(op):
    ids={int(q.id) for q in _field(op)}|{int(c.id) for c in (op.discard or [])}
    if ids & ONE_PRIZE_ROCKET:return 'rocket'
    if ids & ONE_PRIZE_ALA:return 'alakazam'
    return 'unknown'

def _has_rock(q):return bool(q is not None and any(int(getattr(e,'id',-1))==ROCK_F for e in (q.energyCards or [])))

def _known_incoming(op,defender):
    """Guaranteed/public minimum next attack damage/effect counters.

    Hidden Rocket-Feathers supporters are intentionally not guessed.  This guard
    only calls a loss forced when public information already proves the KO.
    """
    if not op.active or defender is None:return 0
    a=op.active[0];cid=int(a.id);en=len(a.energies or [])
    if cid==ALAKAZAM and en>=1:
        if _has_rock(defender):return 0
        # Powerful Hand places counters (an attack effect); Rock Fighting blocks it.
        return 20*(int(op.handCount or 0)+1)
    if cid==ALAKAZAM_ALT and en>=1:
        # Psychic: 10 + 50 per Energy attached to our Active.
        return 10+50*len(defender.energies or [])
    if cid==TR_ARTICUNO and en>=3:
        return 120 if any(int(getattr(e,'id',-1))==TR_ENERGY for e in (a.energyCards or [])) else 60
    if cid==TR_MURKROW and en>=2:return 30
    if cid in {TR_PORYGON2,TR_PORYGONZ} and en>=(3 if cid==TR_PORYGON2 else 2):
        n=sum(1 for c in (op.discard or []) if int(c.id) in TR_SUPPORTERS)
        return 20*n
    if cid==TR_HONCHKROW and en>=3:return 100
    return 0

class OnePrizeEndgameGuard:
    def __init__(self):self.reset()
    def reset(self):
        self.pending_shield_turn=-1
        self.pending_mega_rotation_turn=-1
        self.stats={'calls':0,'families':{},'overrides':{},'board_clear':0,'dunsparce_redirects':0,'shield_pivot_blocks':0,'mega_saves':0,'aura_economy':0,'board_survival_blocks':0,'poke_pad_survival_redirects':0,'mega_rotations':0,'last':{}}
    def _note(self,k,**kw):
        self.stats['overrides'][k]=self.stats['overrides'].get(k,0)+1;self.stats['last']={'reason':k,**kw}
    def _emit(self,a,k,**kw):self._note(k,action=list(a),**kw);return list(a)
    def _terminal_attack(self,obs,opts,mine,op):
        # If the opponent has exactly one Pokemon in play, KOing it wins even when
        # our remaining Prize count is larger than that Pokemon's Prize value.
        if len(_field(op))!=1 or not mine.active or not op.active:return None
        a0=mine.active[0];t0=op.active[0];cands=[]
        for i,o in enumerate(opts):
            if o.type!=OptionType.ATTACK:continue
            aid=int(o.attackId or -1);d=0
            if int(a0.id)==LUCARIO:
                if aid==AURA:d=_aura_damage(t0)
                elif aid==MEGA:d=max(0,270-(30 if _fighting_resist(t0) else 0))
            elif int(a0.id)==DUNSPARCE:
                if aid==RAM:d=max(0,20-(30 if _fighting_resist(t0) else 0))
            elif int(a0.id)==DUDUN:
                if aid==76:d=max(0,90-(30 if _fighting_resist(t0) else 0))
            elif int(a0.id)==SOLROCK and aid==COSMIC and any(int(q.id)==LUNATONE for q in (mine.bench or []) if q is not None):d=70
            else:
                # Unknown/dynamic attacks are not claimed terminal here.  The
                # semantic FinalDay proof layer handles them; this last guard is
                # intentionally conservative so a printed damage number cannot
                # bypass immunity, attack text, or other special effects.
                d=0
            if d>=int(t0.hp or 0)>0:cands.append((d,i,aid))
        if not cands:return None
        cands.sort(key=lambda x:(x[0],x[1]))
        self.stats['board_clear']+=1
        return self._emit([cands[0][1]],'oneprize:board_clear_attack',target=int(t0.id),hp=int(t0.hp or 0),attack=cands[0][2])
    def _terminal_now(self,obs,opts,mine,op):
        if not mine.active or not op.active:return False
        need=len(mine.prize or []);t=op.active[0];field_clear=(len(_field(op))==1)
        for o in opts:
            if o.type!=OptionType.ATTACK:continue
            d=0;aid=int(o.attackId or -1);a=mine.active[0]
            if int(a.id)==LUCARIO:
                d=_aura_damage(t) if aid==AURA else max(0,270-(30 if _fighting_resist(t) else 0)) if aid==MEGA else 0
            elif int(a.id)==SOLROCK and aid==COSMIC and any(int(q.id)==LUNATONE for q in (mine.bench or []) if q is not None):d=70
            else:d=int(getattr(ATT.get(aid),'damage',0) or 0)
            if d>=int(t.hp or 0) and (field_clear or _pv(t)>=need):return True
        return False
    def _shield_choice(self,obs,opts,mine):
        cand=[];luna=any(int(q.id)==LUNATONE for q in _field(mine))
        for i,o in enumerate(opts):
            try:q=_card(obs,o.area,o.index,getattr(o,'playerIndex',obs.current.yourIndex))
            except Exception:q=None
            if q is None:continue
            c=CARD.get(int(q.id))
            if c is None or bool(getattr(c,'ex',False)) or bool(getattr(c,'megaEx',False)):continue
            en=len(q.energies or []);role=0
            if int(q.id)==SOLROCK and luna and en>=1:role=700
            elif int(q.id) in {RIOLU70,RIOLU80} and en>=1:role=620
            elif int(q.id)==DUNSPARCE and en>=1:role=560
            elif int(q.id)==DUDUN:role=500
            else:role=300
            cand.append((role,int(q.hp or 0),-en,-i,i,int(q.id)))
        if not cand:return None
        cand.sort(reverse=True);return cand[0][-2],cand[0][-1]
    def choose(self,obs_dict,base):
        self.stats['calls']+=1
        try:obs=to_observation_class(obs_dict)
        except Exception:return base
        if obs.current is None or obs.select is None:return base
        opts=list(obs.select.option or []);me=int(obs.current.yourIndex);mine=obs.current.players[me];op=obs.current.players[1-me]
        fam=_revealed_family(op);self.stats['families'][fam]=self.stats['families'].get(fam,0)+1
        if not opts:return base
        ctx=obs.select.context;turn=int(obs.current.turn or 0)

        # Live replay 93744849: with only one Pokemon in play, Poké Pad searched
        # Dudunsparce twice instead of a Basic, leaving no Bench and eventually
        # losing by board clear.  When Poké Pad is resolving in this emergency
        # state, an immediately benchable Basic strictly dominates an Evolution.
        # Keep this generic across matchups; the option list itself proves legality.
        if ctx==SelectContext.TO_HAND and len(_field(mine))<=1:
            effect_id=int(getattr(getattr(obs.select,'effect',None),'id',-1) or -1)
            if effect_id==POKE_PAD:
                # Preserve an already-good Basic choice; this guard exists only to
                # stop the dominated Evolution pickup seen in the Rocket loss.
                if isinstance(base,list) and len(base)==1 and 0<=int(base[0])<len(opts):
                    bo=opts[int(base[0])];bq=_card(obs,bo.area,bo.index,getattr(bo,'playerIndex',me))
                    bd=CARD.get(int(bq.id)) if bq is not None else None
                    if bd is not None and bool(getattr(bd,'basic',False)) and not bool(getattr(bd,'ex',False)) and not bool(getattr(bd,'megaEx',False)):
                        return base
                priority={RIOLU80:1000,RIOLU70:990,SOLROCK:900,LUNATONE:850,DUNSPARCE:800}
                basics=[]
                for i,o in enumerate(opts):
                    q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is None:continue
                    cd=CARD.get(int(q.id))
                    if cd is None or not bool(getattr(cd,'basic',False)):continue
                    # Poké Pad already masks Rule-Box Pokemon, but keep the guard
                    # self-contained in case the engine exposes a broader menu.
                    if bool(getattr(cd,'ex',False)) or bool(getattr(cd,'megaEx',False)):continue
                    basics.append((priority.get(int(q.id),500),-i,i,int(q.id)))
                if basics:
                    basics.sort(reverse=True);ii=basics[0][2];cid=basics[0][3]
                    if not (isinstance(base,list) and base==[ii]):
                        self.stats['board_survival_blocks']+=1
                        self.stats['poke_pad_survival_redirects']+=1
                        return self._emit([ii],'oneprize:poke_pad_basic_for_board_survival',target=cid,field=len(_field(mine)))

        # Absolute terminal dominance: no Prize arithmetic is required when the KO
        # removes the opponent's final Pokemon from the field.
        if ctx==SelectContext.MAIN:
            z=self._terminal_attack(obs,opts,mine,op)
            if z is not None:return z

        # Finish a healthy-Mega rotation requested on the previous MAIN choice.
        # This is a multi-action plan: retreat the nearly-KO'd three-Prize body,
        # then promote the healthier charged Mega before continuing the turn.
        if self.pending_mega_rotation_turn==turn and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
            cand=[]
            for i,o in enumerate(opts):
                q=_card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                if q is None or int(q.id)!=LUCARIO:continue
                cand.append((len(q.energies or []),int(q.hp or 0),-i,i))
            if cand:
                cand.sort(reverse=True);self.pending_mega_rotation_turn=-1
                return self._emit([cand[0][-1]],'oneprize:promote_healthy_mega',family=fam,hp=cand[0][1],energies=cand[0][0])

        # Finish a safety retreat by selecting a one-Prize shield, not the same
        # three-Prize Mega we just spent resources to protect.
        if self.pending_shield_turn==turn and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
            z=self._shield_choice(obs,opts,mine)
            if z is not None:
                idx,cid=z;return self._emit([idx],'oneprize:retreat_to_shield',target=cid)

        if fam not in {'alakazam','rocket'}:return base

        # Human Rocket replay 20260816_165836: do not greedily consume a
        # Dudunsparce draw/pivot when that would collapse a two-Pokemon field to
        # a single body.  The game can be lost by board clear regardless of Prize
        # count.  First make a deterministic board-building action (bench a Basic,
        # Poffin, or Lillie's Determination); if none exists, keep both bodies.
        # This is intentionally limited to the revealed one-Prize families.
        if ctx==SelectContext.MAIN and len(_field(mine))<=2 and isinstance(base,list) and len(base)==1:
            bi=int(base[0])
            if 0<=bi<len(opts) and opts[bi].type==OptionType.ABILITY:
                bo=opts[bi];src=_card(obs,bo.area,bo.index,me)
                if src is not None and int(src.id)==DUDUN:
                    setup=[]
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.PLAY:continue
                        c=_source(obs,o,me)
                        if c is None:continue
                        cid=int(c.id);cd=CARD.get(cid)
                        # A Basic Pokemon enlarges the board immediately.
                        is_basic=bool(cd is not None and int(getattr(cd,'stage',-1) or -1)==0)
                        if is_basic:
                            setup.append((1000,-i,i,cid,'basic'))
                        elif cid==POFFIN:
                            setup.append((900,-i,i,cid,'poffin'))
                        elif cid==LILLIE and not bool(obs.current.supporterPlayed):
                            setup.append((800,-i,i,cid,'lillie'))
                    if setup:
                        setup.sort(reverse=True);self.stats['board_survival_blocks']+=1
                        _,_,ii,cid,kind=setup[0]
                        return self._emit([ii],'oneprize:build_board_before_dudun',family=fam,setup=kind,card=cid,field=len(_field(mine)))
                    # No deterministic way to replace the disappearing body.  Use
                    # an attack if one exists; otherwise end while two Pokemon remain.
                    attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
                    if attacks:
                        self.stats['board_survival_blocks']+=1
                        return self._emit([attacks[0]],'oneprize:keep_two_body_attack',family=fam,field=len(_field(mine)))
                    end=next((i for i,o in enumerate(opts) if o.type==OptionType.END),None)
                    if end is not None:
                        self.stats['board_survival_blocks']+=1
                        return self._emit([end],'oneprize:keep_two_body_end',family=fam,field=len(_field(mine)))

        # Cross-play distilled sequence safety.  If a damaged three-Prize Mega
        # can retreat and a substantially healthier, already-charged Mega is on the
        # Bench, rotate before attacking when the opponent is within a single Mega
        # KO of winning.  The prior policy saw only the current attack and missed
        # the superior retreat -> promote -> attack sequence.
        if ctx==SelectContext.MAIN and mine.active and int(mine.active[0].id)==LUCARIO and 0 < len(op.prize or []) <= 3 and not self._terminal_now(obs,opts,mine,op):
            a0=mine.active[0];retreat=next((i for i,o in enumerate(opts) if o.type==OptionType.RETREAT),None)
            healthy=[]
            for q in (mine.bench or []):
                if q is None or int(q.id)!=LUCARIO:continue
                if len(q.energies or [])<1:continue
                if int(q.hp or 0) >= int(a0.hp or 0)+140:
                    healthy.append((len(q.energies or []),int(q.hp or 0),int(q.serial)))
            if retreat is not None and healthy and int(a0.hp or 0)<=160:
                healthy.sort(reverse=True);self.pending_mega_rotation_turn=turn;self.stats['mega_rotations']+=1
                return self._emit([retreat],'oneprize:rotate_damaged_mega',family=fam,active_hp=int(a0.hp or 0),bench_hp=healthy[0][1],bench_energy=healthy[0][0],opp_prizes=len(op.prize or []))

        # Cheap-KO economy.  Against a one-Prize board, Aura Jab is strictly better
        # than Mega Brave/PPP when Aura already KOs: it avoids the Mega-Brave lock
        # and keeps Aura-Jab acceleration available.
        if ctx==SelectContext.MAIN and mine.active and op.active and int(mine.active[0].id)==LUCARIO and _pv(op.active[0])==1:
            aura=next((i for i,o in enumerate(opts) if o.type==OptionType.ATTACK and int(o.attackId or -1)==AURA),None)
            if aura is not None and _aura_damage(op.active[0])>=int(op.active[0].hp or 0):
                bi=base[0] if isinstance(base,list) and len(base)==1 else -1
                if bi!=aura:
                    self.stats['aura_economy']+=1
                    return self._emit([aura],'oneprize:aura_exact_economy',family=fam,target=int(op.active[0].id),hp=int(op.active[0].hp or 0))

        # Replay-distilled Dunsparce role: once it has the one Energy needed for
        # Trading Places/retreat utility, another manual Fighting Energy is a sink
        # in Alakazam/Rocket matchups unless that second Energy itself proves a
        # terminal 20-damage Ram.  Prefer the real attacker or keep the attachment.
        if ctx==SelectContext.MAIN and not obs.current.energyAttached and isinstance(base,list) and len(base)==1:
            bi=int(base[0])
            if 0<=bi<len(opts) and opts[bi].type==OptionType.ATTACH:
                bo=opts[bi];q=_target(obs,bo,me);e=_source(obs,bo,me)
                if q is not None and e is not None and int(q.id)==DUNSPARCE and int(e.id) in {BASIC_F,ROCK_F} and len(q.energies or [])>=1:
                    ram_terminal=bool(mine.active and int(mine.active[0].serial)==int(q.serial) and op.active and int(op.active[0].hp or 0)<=20 and (len(_field(op))==1 or len(mine.prize or [])<=1))
                    if not ram_terminal:
                        cand=[];luna=any(int(x.id)==LUNATONE for x in _field(mine))
                        for i,o in enumerate(opts):
                            if o.type!=OptionType.ATTACH:continue
                            qq=_target(obs,o,me);ee=_source(obs,o,me)
                            if qq is None or ee is None or int(ee.id)!=int(e.id) or int(qq.serial)==int(q.serial):continue
                            en=len(qq.energies or []);cid=int(qq.id);score=-999
                            if cid==LUCARIO and en<2:score=900-50*en
                            elif cid in {RIOLU70,RIOLU80} and en<2:score=820-40*en
                            elif cid==SOLROCK and luna and en<1:score=760
                            elif cid==OGERPON and en<3:score=500-40*en
                            elif cid not in {DUNSPARCE,DUDUN} and en<2:score=300-30*en
                            if score>-999:cand.append((score,-i,i,cid))
                        if cand:
                            cand.sort(reverse=True);self.stats['dunsparce_redirects']+=1
                            return self._emit([cand[0][2]],'oneprize:dunsparce_energy_redirect',family=fam,target=cand[0][3])
                        trading=next((i for i,o in enumerate(opts) if o.type==OptionType.ATTACK and int(o.attackId or -1)==TRADING),None)
                        if trading is not None and any(x is not None for x in (mine.bench or [])):
                            self.stats['dunsparce_redirects']+=1
                            return self._emit([trading],'oneprize:dunsparce_keep_attachment',family=fam)
                        end=next((i for i,o in enumerate(opts) if o.type==OptionType.END),None)
                        if end is not None:
                            self.stats['dunsparce_redirects']+=1
                            return self._emit([end],'oneprize:dunsparce_no_sink_end',family=fam)

        # Preserve a one-Prize Dunsparce shield.  The human Team-Rocket loss
        # showed a dominated line: Dunsparce used Trading Places only to expose a
        # two-Prize Ogerpon while the opponent had three Prizes left.  That attack
        # ends our turn, so unless a one-Prize bench target exists it is usually
        # strictly better to keep the one-Prize Active; if Ram is legal, take the
        # free chip instead of donating a multi-Prize body.
        if ctx==SelectContext.MAIN and mine.active and int(mine.active[0].id)==DUNSPARCE and 0 < len(op.prize or []) <= 3 and isinstance(base,list) and len(base)==1:
            bi=int(base[0])
            if 0<=bi<len(opts):
                bo=opts[bi]
                unsafe_pivot=(bo.type==OptionType.ATTACK and int(bo.attackId or -1)==TRADING) or bo.type==OptionType.RETREAT
                if unsafe_pivot:
                    one_bench=any(q is not None and _pv(q)==1 for q in (mine.bench or []))
                    multi_bench=any(q is not None and _pv(q)>1 for q in (mine.bench or []))
                    if multi_bench and not one_bench:
                        ram=next((i for i,o in enumerate(opts) if o.type==OptionType.ATTACK and int(o.attackId or -1)==RAM),None)
                        if ram is not None:
                            self.stats['shield_pivot_blocks']+=1
                            return self._emit([ram],'oneprize:dunsparce_hold_shield_ram',family=fam,opp_prizes=len(op.prize or []))
                        end=next((i for i,o in enumerate(opts) if o.type==OptionType.END),None)
                        if end is not None:
                            self.stats['shield_pivot_blocks']+=1
                            return self._emit([end],'oneprize:dunsparce_hold_shield_end',family=fam,opp_prizes=len(op.prize or []))

        # Guaranteed 3-Prize feed avoidance.  If the current Mega is certainly KO'd
        # by the opponent's already-public attack and that KO would take all their
        # remaining Prizes, retreat to a one-Prize body unless we end the game now.
        if ctx==SelectContext.MAIN and mine.active and int(mine.active[0].id)==LUCARIO and len(op.prize or [])<=3 and not self._terminal_now(obs,opts,mine,op):
            a0=mine.active[0];incoming=_known_incoming(op,a0)
            retreat=next((i for i,o in enumerate(opts) if o.type==OptionType.RETREAT),None)
            has_shield=any(q is not None and _pv(q)==1 for q in (mine.bench or []))
            if retreat is not None and has_shield and incoming>=int(a0.hp or 0)>0:
                self.pending_shield_turn=turn;self.stats['mega_saves']+=1
                return self._emit([retreat],'oneprize:avoid_forced_three_prize_loss',family=fam,incoming=incoming,hp=int(a0.hp or 0),opp_prizes=len(op.prize or []))

        # The same invariant applies before voluntarily evolving an Active one-Prize
        # Riolu into the three-Prize Mega.  Carry current damage through evolution and
        # refuse the evolution when the public return attack already proves game loss.
        if ctx==SelectContext.MAIN and isinstance(base,list) and len(base)==1 and mine.active and int(mine.active[0].id) in {RIOLU70,RIOLU80} and len(op.prize or [])<=3:
            bi=int(base[0])
            if 0<=bi<len(opts) and opts[bi].type==OptionType.EVOLVE:
                evo=_card(obs,opts[bi].area,opts[bi].index,me);bp=_card(obs,opts[bi].inPlayArea,opts[bi].inPlayIndex,me)
                if evo is not None and bp is not None and int(evo.id)==LUCARIO and int(bp.serial)==int(mine.active[0].serial):
                    damage=max(0,int(CARD[int(bp.id)].hp)-int(bp.hp or 0));fake=type('P',(),{})();fake.id=LUCARIO;fake.hp=max(1,int(CARD[LUCARIO].hp)-damage);fake.energies=list(bp.energies or []);fake.energyCards=list(bp.energyCards or [])
                    incoming=_known_incoming(op,fake)
                    if incoming>=int(fake.hp) and not self._terminal_now(obs,opts,mine,op):
                        # Keep the one-Prize body and use an existing attack/retreat/end.
                        attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
                        if attacks:
                            self.stats['mega_saves']+=1;return self._emit([attacks[0]],'oneprize:hold_riolu_shield',family=fam,incoming=incoming,mega_hp=int(fake.hp))
                        retreat=next((i for i,o in enumerate(opts) if o.type==OptionType.RETREAT),None)
                        if retreat is not None:
                            self.stats['mega_saves']+=1;self.pending_shield_turn=turn;return self._emit([retreat],'oneprize:hold_riolu_retreat',family=fam,incoming=incoming,mega_hp=int(fake.hp))
                        end=next((i for i,o in enumerate(opts) if o.type==OptionType.END),None)
                        if end is not None:
                            self.stats['mega_saves']+=1;return self._emit([end],'oneprize:hold_riolu_end',family=fam,incoming=incoming,mega_hp=int(fake.hp))
        return base
    def get_stats(self):return dict(self.stats)
