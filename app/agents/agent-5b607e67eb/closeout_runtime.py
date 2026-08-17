from __future__ import annotations
from cg.api import AreaType,EnergyType,OptionType,SelectContext,all_attack,all_card_data

CARD={c.cardId:c for c in all_card_data()}
ATTACK={a.attackId:a for a in all_attack()}

RIOLU70=333; RIOLU80=677; LUCARIO=678; DUDUN_EX=306; OGERPON=117; CRUSTLE=345
TEAL=96; RAGING_BOLT=63; WELLSPRING=108; KANG=756
PPP=1141; BLACK_BELT=1211; MOUNTAIN=1252; BOSS=1182; BASIC_F=6; ROCK_F=20
AURA=982; MEGA=983; TENACIOUS=425; DRILL=426; DEMOLISH=148; COSMIC=980


def field(p): return [q for q in list(p.active or [])+list(p.bench or []) if q is not None]

def prize_value(p):
    if p is None:return 0
    c=CARD.get(p.id)
    return 3 if c and c.megaEx else 2 if c and c.ex else 1

def card(obs,area,index,player):
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

def attack_damage(active,aid,opponent,target=None,ppp=0,black_belt=0):
    """Conservative deterministic damage used only to prove a closeout line."""
    if active is None:return 0
    if target is None: target=opponent.active[0] if opponent.active else None
    ac=CARD.get(active.id)
    # Known defending effects in the current league.
    if target is not None and target.id==CRUSTLE and ac and (ac.ex or ac.megaEx) and aid not in {DRILL,DEMOLISH}:return 0
    if target is not None and target.id==OGERPON and ac and bool(ac.skills) and aid not in {DRILL,DEMOLISH}:return 0
    if aid==TENACIOUS:
        dmg=60*sum(1 for q in field(opponent) if CARD.get(q.id) and (CARD[q.id].ex or CARD[q.id].megaEx))
    else:
        a=ATTACK.get(aid);dmg=int(getattr(a,'damage',0) or 0) if a else 0
    if ac and int(getattr(ac,'energyType',-1))==int(EnergyType.FIGHTING):dmg+=30*int(ppp)
    tc=CARD.get(target.id) if target is not None else None
    if black_belt and tc and (tc.ex or tc.megaEx):dmg+=40*int(black_belt)
    # Demolish / Drill / Cosmic explicitly ignore Weakness/Resistance.
    if target is not None and aid not in {DRILL,DEMOLISH,COSMIC} and tc and ac and getattr(tc,'resistance',None) is not None and int(tc.resistance)==int(ac.energyType):dmg=max(0,dmg-30)
    return dmg

class CloseoutPlanner:
    """Proves only same-turn wins. If it cannot prove mate, it returns None."""
    def __init__(self):self.reset()
    def reset(self):
        self.turn=-1;self.boss_target=-1;self.ppp_used=0;self.belt_used=0
        self.last_mega_brave_serial=-1;self.last_mega_brave_turn=-999
        self.stats={'calls':0,'overrides':{}}
    def _note(self,k):self.stats['overrides'][k]=self.stats['overrides'].get(k,0)+1
    def _reset_turn(self,t):
        if self.turn!=t:self.turn=t;self.boss_target=-1;self.ppp_used=0;self.belt_used=0
    def record_action(self,obs,action):
        """Record the final action actually emitted by the full policy stack."""
        try:
            if obs.current is None or obs.select is None or not action or len(action)!=1:return
            i=int(action[0]);opts=list(obs.select.option or [])
            if i<0 or i>=len(opts):return
            o=opts[i]
            if o.type==OptionType.ATTACK and o.attackId==MEGA and obs.current.players[obs.current.yourIndex].active:
                a0=obs.current.players[obs.current.yourIndex].active[0]
                self.last_mega_brave_serial=int(a0.serial)
                self.last_mega_brave_turn=int(obs.current.turn or 0)
        except Exception:pass

    def choose(self,obs,base):
        self.stats['calls']+=1
        if obs.current is None or obs.select is None:return None
        st=obs.current;self._reset_turn(int(st.turn or 0));opts=list(obs.select.option or [])
        if not opts:return None
        me=st.yourIndex;mine=st.players[me];op=st.players[1-me];ctx=obs.select.context
        # Finish the target-selection half of a committed Boss mate.
        if self.boss_target>=0 and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
            for i,o in enumerate(opts):
                q=card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                if q is not None and q.serial==self.boss_target:
                    self._note('closeout:boss_target');return [i]
            self.boss_target=-1

        # Forced-promotion closeout: if the current opponent Active itself is the
        # final prize target, promote a Bench attacker that already has a proven KO.
        # No future search/supporter/manual attachment is assumed here.
        if ctx==SelectContext.TO_ACTIVE and op.active:
            # Replay 92543804 exposed this exact failure in the Raging Bolt / Ogerpon
            # family. Keep the new promotion planner scoped to that public identity so
            # Archaludon/Marnie/Crustle forced promotions remain byte-for-byte legacy.
            opp_ids={q.id for q in field(op)} | {c.id for c in (op.discard or [])}
            raging_tera=bool(opp_ids & {TEAL,RAGING_BOLT,WELLSPRING,KANG})
            if not raging_tera:
                return None
            # Only treat this as a next-turn mate when the promotion follows an
            # opponent attack.  A promotion caused during our own turn (recoil,
            # self-KO effects, etc.) does not grant another attack this turn.
            last_attack_player=None
            try:
                for lg in (obs.logs or []):
                    if getattr(lg,'attackId',None) is not None:
                        last_attack_player=getattr(lg,'playerIndex',None)
            except Exception:pass
            my_pr=len(mine.prize or []); target=op.active[0]; tp=prize_value(target)
            if last_attack_player==1-me and 0<my_pr<=tp:
                mates=[]
                for i,o in enumerate(opts):
                    q=card(obs,o.area,o.index,getattr(o,'playerIndex',me))
                    if q is None:continue
                    en=len(q.energies or []); dmg=0
                    if q.id==LUCARIO and en>=1:
                        # If this exact Lucario used Mega Brave on our immediately
                        # previous turn, only Aura Jab is guaranteed legal now.
                        locked=bool(q.serial==self.last_mega_brave_serial and self.last_mega_brave_turn==int(st.turn or 0)-1)
                        aid=MEGA if en>=2 and not locked else AURA
                        dmg=attack_damage(q,aid,op,target)
                        # TO_ACTIVE does not expose the next attack menu.  Require a
                        # 30-damage cushion for Mega Brave so stadium/field reductions
                        # cannot turn a marginal 'mate' into a false promotion.
                        if aid==MEGA:dmg=max(0,dmg-30)
                    elif q.id==DUDUN_EX and en>=1:
                        dmg=attack_damage(q,TENACIOUS,op,target)
                        if en>=3:dmg=max(dmg,attack_damage(q,DRILL,op,target))
                    elif q.id==OGERPON and en>=3:
                        dmg=attack_damage(q,DEMOLISH,op,target)
                    elif q.id==676 and en>=1 and any(x is not None and x.id==675 for x in (mine.bench or [])):
                        dmg=attack_damage(q,COSMIC,op,target)
                    elif q.id==675 and en>=2:
                        dmg=50
                    if dmg>=int(target.hp or 0):
                        # Prefer the lowest-prize body among proven mates, then more
                        # damage.  In replay 92543804 the only mate is E2 Lucario.
                        mates.append((-prize_value(q),dmg,-i,i))
                if mates:
                    mates.sort(reverse=True);self._note('closeout:promotion_mate');return [mates[0][-1]]
        if ctx!=SelectContext.MAIN or not mine.active or not op.active:return None
        my_pr=len(mine.prize or [])
        if my_pr<=0:return None
        a0=mine.active[0];t0=op.active[0];tp=prize_value(t0)
        board_clear=(len(field(op))==1)
        terminal_target=(tp>=my_pr or board_clear)
        attacks=[(i,o.attackId) for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
        # Direct attack mate, after modifiers already committed by this planner.
        direct=[]
        for i,aid in attacks:
            d=attack_damage(a0,aid,op,t0,self.ppp_used,self.belt_used)
            if terminal_target and d>=int(t0.hp or 0):direct.append((d,i,aid))
        if direct:
            direct.sort(key=lambda x:(x[0],x[1]));self._note('closeout:direct_mate');return [direct[0][1]]
        # PPP: Item, so it can precede either a normal mate or a Boss mate. Only use
        # one when the +30 actually crosses the current target's final KO threshold.
        ppp_idx=[]
        for i,o in enumerate(opts):
            if o.type==OptionType.PLAY:
                c=card(obs,AreaType.HAND,o.index,me)
                if c is not None and c.id==PPP:ppp_idx.append(i)
        if ppp_idx and attacks and terminal_target:
            now=max((attack_damage(a0,aid,op,t0,self.ppp_used,self.belt_used) for _,aid in attacks),default=0)
            plus=max((attack_damage(a0,aid,op,t0,self.ppp_used+1,self.belt_used) for _,aid in attacks),default=0)
            if now<int(t0.hp or 0)<=plus:
                self.ppp_used+=1;self._note('closeout:ppp_threshold');return [ppp_idx[0]]
        # Black Belt is a Supporter; use it only when it alone proves the current ex KO.
        if not st.supporterPlayed and CARD.get(t0.id) and (CARD[t0.id].ex or CARD[t0.id].megaEx) and attacks and terminal_target:
            now=max((attack_damage(a0,aid,op,t0,self.ppp_used,self.belt_used) for _,aid in attacks),default=0)
            plus=max((attack_damage(a0,aid,op,t0,self.ppp_used,self.belt_used+1) for _,aid in attacks),default=0)
            if now<int(t0.hp or 0)<=plus:
                for i,o in enumerate(opts):
                    if o.type==OptionType.PLAY:
                        c=card(obs,AreaType.HAND,o.index,me)
                        if c is not None and c.id==BLACK_BELT:
                            self.belt_used+=1;self._note('closeout:black_belt_threshold');return [i]
        # Gravity Mountain can turn a Stage-2 Active into an immediate exact KO.
        tc=CARD.get(t0.id)
        if tc and tc.stage2 and attacks and terminal_target:
            best=max((attack_damage(a0,aid,op,t0,self.ppp_used,self.belt_used) for _,aid in attacks),default=0)
            if best<int(t0.hp or 0) and best>=max(0,int(t0.hp or 0)-30):
                for i,o in enumerate(opts):
                    if o.type==OptionType.PLAY:
                        c=card(obs,AreaType.HAND,o.index,me)
                        if c is not None and c.id==MOUNTAIN:
                            self._note('closeout:mountain_threshold');return [i]
        # Evolve a mature Active Riolu when that evolution itself opens mate.
        if a0.id in {RIOLU70,RIOLU80} and not bool(getattr(a0,'appearThisTurn',False)) and terminal_target:
            en=len(a0.energies or []);future_aid=MEGA if en>=2 else AURA if en>=1 else None
            if future_aid is not None:
                # Lucario has no Ability; construct conservative damage directly.
                fake=type('P',(),{'id':LUCARIO})()
                dmg=attack_damage(fake,future_aid,op,t0,self.ppp_used,self.belt_used)
                if dmg>=int(t0.hp or 0):
                    for i,o in enumerate(opts):
                        if o.type!=OptionType.EVOLVE:continue
                        evo=card(obs,o.area,o.index,me);bp=card(obs,o.inPlayArea,o.inPlayIndex,me)
                        if evo is not None and bp is not None and evo.id==LUCARIO and bp.serial==a0.serial:
                            self._note('closeout:evolve_mate');return [i]
        # One Energy can open Aura (0->1) or Mega Brave (1->2). Extra Energy never
        # re-enables a Mega Brave locked by the previous turn.
        if a0.id==LUCARIO and not st.energyAttached and terminal_target:
            en=len(a0.energies or []);future_aid=AURA if en==0 else MEGA if en==1 else None
            if future_aid is not None and attack_damage(a0,future_aid,op,t0,self.ppp_used,self.belt_used)>=int(t0.hp or 0):
                for i,o in enumerate(opts):
                    if o.type!=OptionType.ATTACH:continue
                    q=card(obs,o.inPlayArea,o.inPlayIndex,me);e=card(obs,AreaType.HAND,o.index,me)
                    if q is not None and q.serial==a0.serial and e is not None and e.id in {BASIC_F,ROCK_F}:
                        self._note('closeout:attach_mate');return [i]
        # Boss mate. Include PPP still in hand because Item can be played after Boss;
        # do not include Black Belt because both it and Boss are Supporters.
        if not st.supporterPlayed and attacks:
            boss_i=None
            for i,o in enumerate(opts):
                if o.type==OptionType.PLAY:
                    c=card(obs,AreaType.HAND,o.index,me)
                    if c is not None and c.id==BOSS:boss_i=i;break
            if boss_i is not None:
                ppp_avail=sum(1 for c in (mine.hand or []) if c.id==PPP)
                kill=[]
                for q in (op.bench or []):
                    if q is None or prize_value(q)<my_pr:continue
                    bd=max((attack_damage(a0,aid,op,q,self.ppp_used+ppp_avail,self.belt_used) for _,aid in attacks),default=0)
                    if bd>=int(q.hp or 0):kill.append((prize_value(q),-int(q.hp or 0),q.serial))
                if kill:
                    kill.sort(reverse=True);self.boss_target=kill[0][2];self._note('closeout:boss_mate');return [boss_i]
        return None
    def get_stats(self):return {'calls':self.stats['calls'],'overrides':dict(self.stats['overrides'])}
