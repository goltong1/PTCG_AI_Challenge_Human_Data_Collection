from __future__ import annotations
import json,os,hashlib
from cg.api import all_card_data
from cg.sim import lib as _simlib
import card_text_semantics as sem

MAIN=0; PLAY=7; ATTACH=8; EVOLVE=9; ABILITY=10; RETREAT=12; ATTACK=13; END=14; CARDOPT=3
BASIC_F=6; ROCK_F=20; AIR_BALLOON=1174; HERO_CAPE=1159; BOSS=1182
POKE_PAD=1152
POFFIN=1086
SOLROCK=676; LUNATONE=675; DUNSPARCE=305; DUDUN=66; DUDUN_EX=306; RIOLU70=333; RIOLU80=677; LUCARIO=678; OGERPON=117
COSMIC=980; AURA=982; MEGA_BRAVE=983; TENACIOUS=425; DRILL=426; DEMOLISH=148
UTILITY={SOLROCK,LUNATONE,DUNSPARCE,DUDUN}
ATTACKERS={RIOLU70,RIOLU80,LUCARIO,DUDUN_EX,OGERPON}
CARD={int(c.cardId):c for c in all_card_data()}
try: ATTACK_DB={int(a['attackId']):a for a in json.loads(_simlib.AllAttack())}
except Exception: ATTACK_DB={}

def _i(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d

def _ctx_main(sel):
    c=sel.get('context')
    return c==0 or str(c).lower()=='main'

def _players(obs):
    cur=obs.get('current') or {}; ps=cur.get('players') or []; me=_i(cur.get('yourIndex'),-1)
    if me not in (0,1) or len(ps)<2:return None,None,None
    return cur,ps[me],ps[1-me]

def _field(p):return [q for q in (p.get('active') or [])+(p.get('bench') or []) if q]

def _energy_n(q):return len(q.get('energies') or q.get('energyCards') or []) if q else 0

def _hand_card(mine,o):
    try:
        if _i(o.get('area'),2)!=2:return None
        return (mine.get('hand') or [])[_i(o.get('index'),-1)]
    except Exception:return None

def _target_card(mine,o):
    try:
        a=_i(o.get('inPlayArea'),-1);j=_i(o.get('inPlayIndex'),-1)
        if a==4:return (mine.get('active') or [])[j]
        if a==5:return (mine.get('bench') or [])[j]
    except Exception:pass
    return None

def _opt_card_from_area(mine,o):
    try:
        a=_i(o.get('area'),-1);j=_i(o.get('index'),-1)
        if a==5:return (mine.get('bench') or [])[j]
        if a==4:return (mine.get('active') or [])[j]
        if a==2:return (mine.get('hand') or [])[j]
    except Exception:pass
    return None

def _opt_any_card(cur,o):
    try:
        ps=cur.get('players') or [];pi=_i(o.get('playerIndex'),_i(cur.get('yourIndex'),0));a=_i(o.get('area'),-1);j=_i(o.get('index'),-1)
        if not 0<=pi<len(ps):return None
        p=ps[pi]
        if a==5:return (p.get('bench') or [])[j]
        if a==4:return (p.get('active') or [])[j]
        if a==2:return (p.get('hand') or [])[j]
    except Exception:pass
    return None

def _has_luna(mine):return any(_i(q.get('id'),-1)==LUNATONE for q in (mine.get('bench') or []) if q)

def _special_energy(q):
    for e in (q.get('energyCards') or []) if q else []:
        if _i(e.get('id'),-1)!=BASIC_F:return True
    return False

def _effect_ids(cur,op):
    ids=[]
    for z in (cur.get('stadium') or []):
        if z:ids.append(_i(z.get('id')))
    for z in _field(op):ids.append(_i(z.get('id')))
    return ids

def _attack_damage(aid,active,opp_active,cur,op):
    aid=_i(aid,-1); attacker=_i(active.get('id'),-1); defender=_i(opp_active.get('id'),-1)
    rec=ATTACK_DB.get(aid) or {}; dmg=_i(rec.get('damage'),0)
    if aid==TENACIOUS:
        exn=0
        for q in _field(op):
            c=CARD.get(_i(q.get('id'),-1))
            if c is not None and (bool(getattr(c,'ex',False)) or bool(getattr(c,'megaEx',False))):exn+=1
        dmg=60*exn
    if aid==COSMIC and not any(_i(q.get('id'),-1)==LUNATONE for q in (cur.get('players') or [])[cur.get('yourIndex',0)].get('bench',[]) if q):return 0
    bypass=sem.attack_bypasses_active_effects(aid)
    if not bypass:
        if sem.damage_prevention_applies(defender,attacker,attacker_has_special_energy=_special_energy(active),raw_damage=dmg):return 0
        if sem.global_damage_prevention_applies(_effect_ids(cur,op),defender,attacker,attacker_has_special_energy=_special_energy(active),raw_damage=dmg):return 0
    tags=set(sem.attack_tags(aid))
    # Weakness only helps, so ignore it.  Resistance is conservatively treated as -30
    # unless the printed attack explicitly ignores Weakness/Resistance.
    dcard=CARD.get(defender)
    if 'IGNORE_WEAKNESS_RESISTANCE' not in tags and dcard is not None and getattr(dcard,'resistance',None) is not None:
        dmg=max(0,dmg-30)
    return dmg

def _prizes_for(cid):
    c=CARD.get(_i(cid,-1))
    if c is None:return 1
    if bool(getattr(c,'megaEx',False)):return 3
    if bool(getattr(c,'ex',False)):return 2
    return 1

class FinalDayProofGuard:
    """Last-day invariant/proof layer.

    It never predicts an archetype.  It only overrides when a public-state invariant is
    exact (game-winning attack, no-effect Cosmic Beam) or when a same-hidden-state
    two-ply CABT verifier proves an attacker-pivot chain superior.
    """
    def __init__(self,root=None):
        self.verifier=None;self.generic_verifier=None;self.root=root
        self.exact={}
        try:
            if root:
                self.exact=(json.load(open(os.path.join(root,'final_day_exact_cf_model.json'),encoding='utf-8')).get('states') or {})
        except Exception:
            self.exact={}
        self.reset()
    def reset(self):
        self.pending=None;self.search_calls=0
        self.stats={'calls':0,'overrides':{},'search_calls':0,'search_accept':0,'search_reject':0,'exact_hits':0,'exact_accept':0,'exact_reject':0,'errors':0,'last':{}}
    def _emit(self,a,r,**kw):
        self.stats['overrides'][r]=self.stats['overrides'].get(r,0)+1;self.stats['last']={'reason':r,'action':list(a),**kw};return list(a)
    def _clear(self):self.pending=None
    def _exact_fp(self,obs):
        try:
            sel=obs.get('select') or {};raw=(obs.get('search_begin_input') or '')+'|'+json.dumps(sel.get('option') or [],sort_keys=True,separators=(',',':'))
            return hashlib.sha1(raw.encode()).hexdigest()
        except Exception:return ''
    def _robust_exact_cf(self,obs,base):
        if not self.exact or not isinstance(base,list) or len(base)!=1 or not callable(self.generic_verifier):return None
        mem=self.exact.get(self._exact_fp(obs))
        if not mem:return None
        self.stats['exact_hits']+=1;bi=_i(base[0],-1);bad=_i(mem.get('bad_index'),-2);gi=_i(mem.get('good_index'),-1)
        opts=(obs.get('select') or {}).get('option') or []
        if bi!=bad or not 0<=gi<len(opts):return None
        try:
            ok,detail=self.generic_verifier(obs,base,[gi])
        except Exception as e:
            self.stats['errors']+=1;self.stats['last']={'reason':'exact_cf_verify_error','error':repr(e)[:160]};return None
        if ok and float((detail or {}).get('gain',-1e9))>=25.0:
            self.stats['exact_accept']+=1
            return self._emit([gi],'proof:robust_exact_cf',mistake_class=mem.get('mistake_class'),offline_median_gain=mem.get('offline_median_gain'),runtime=detail)
        self.stats['exact_reject']+=1;return None

    def _game_winning_attack(self,obs,base):
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not mine.get('active') or not op.get('active'):return None
        remain=len(mine.get('prize') or []);target=op['active'][0];pv=_prizes_for(target.get('id'))
        board_clear=(len(_field(op))==1)
        if remain<=0 or (pv<remain and not board_clear):return None
        active=mine['active'][0];cand=[]
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)!=ATTACK:continue
            aid=_i(o.get('attackId'),-1);dmg=_attack_damage(aid,active,target,cur,op)
            if dmg>=_i(target.get('hp'),9999):cand.append((dmg,i,aid))
        if not cand:return None
        if isinstance(base,list) and len(base)==1:
            bi=_i(base[0],-1)
            if any(i==bi for _,i,_ in cand):return None
        cand.sort(key=lambda x:(x[0],x[1]))
        reason='proof:board_clear_attack' if board_clear and pv<remain else 'proof:game_winning_attack'
        return self._emit([cand[0][1]],reason,attack=cand[0][2],damage=cand[0][0],target=_i(target.get('id')),prizes=remain,board_clear=board_clear)
    def _no_effect_cosmic(self,obs,base):
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not isinstance(base,list) or len(base)!=1:return None
        bi=_i(base[0],-1)
        if not 0<=bi<len(opts):return None
        bo=opts[bi]
        if _i(bo.get('type'),-1)!=ATTACK or _i(bo.get('attackId'),-1)!=COSMIC or _has_luna(mine):return None
        # Cosmic Beam is text-proven to do nothing without a benched Lunatone.
        # First repair the missing condition if Lunatone is already in hand.
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)!=PLAY:continue
            c=_hand_card(mine,o)
            if c and _i(c.get('id'),-1)==LUNATONE:
                self.pending={'kind':'cosmic','turn':_i(cur.get('turn'))}
                return self._emit([i],'proof:no_effect_cosmic_play_lunatone')
        # Otherwise, if a ready damaging bench attacker can be reached immediately,
        # prefer the real attack path rather than a zero-effect attack.
        retreat=next((i for i,o in enumerate(opts) if _i(o.get('type'),-1)==RETREAT),None)
        if retreat is not None:
            ready=[]
            target=op['active'][0] if op.get('active') else None
            if target:
                for j,q in enumerate(mine.get('bench') or []):
                    if not q:continue
                    cid=_i(q.get('id'),-1)
                    # Mega Lucario needs one Fighting; Dudunsparce ex needs 3 total.
                    if cid==LUCARIO and _energy_n(q)>=1 and _attack_damage(AURA,q,target,cur,op)>0:ready.append((3,_i(q.get('hp')),j,_i(q.get('serial'))))
                    elif cid==DUDUN_EX and _energy_n(q)>=3 and max(_attack_damage(TENACIOUS,q,target,cur,op),_attack_damage(DRILL,q,target,cur,op))>0:ready.append((2,_i(q.get('hp')),j,_i(q.get('serial'))))
                    elif cid==OGERPON and _energy_n(q)>=3 and _attack_damage(DEMOLISH,q,target,cur,op)>0:ready.append((1,_i(q.get('hp')),j,_i(q.get('serial'))))
            if ready:
                ready.sort(reverse=True);self.pending={'kind':'generic_retreat_attack','turn':_i(cur.get('turn')),'target_serial':ready[0][3]}
                return self._emit([retreat],'proof:no_effect_cosmic_retreat_to_attacker')
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)==END:return self._emit([i],'proof:no_effect_cosmic_to_end')
        return None

    def _boss_game_win(self,obs,base):
        """Prove a Boss -> bench target -> attack chain that immediately ends the game.

        This is independent of archetype recognition and uses only public HP/prizes/card text.
        """
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not mine.get('active') or not op.get('bench'):return None
        remain=len(mine.get('prize') or [])
        if remain<=0:return None
        boss=[]
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)!=PLAY:continue
            c=_hand_card(mine,o)
            if c and _i(c.get('id'),-1)==BOSS:boss.append(i)
        if not boss:return None
        active=mine['active'][0];legal_attacks=[_i(o.get('attackId'),-1) for o in opts if _i(o.get('type'),-1)==ATTACK]
        if not legal_attacks:return None
        wins=[]
        for q in op.get('bench') or []:
            if not q or _prizes_for(q.get('id'))<remain:continue
            for aid in legal_attacks:
                dmg=_attack_damage(aid,active,q,cur,op)
                if dmg>=_i(q.get('hp'),9999):
                    wins.append((_prizes_for(q.get('id')),dmg,_i(q.get('hp')), _i(q.get('serial')),_i(q.get('id')),aid))
        if not wins:return None
        wins.sort(reverse=True);_,dmg,hp,serial,cid,aid=wins[0]
        self.pending={'kind':'boss_win','turn':_i(cur.get('turn')),'target_serial':serial,'attack':aid}
        return self._emit([boss[0]],'proof:boss_game_win_start',target=cid,target_hp=hp,attack=aid,damage=dmg,prizes=remain)

    def _pad_before_utility_energy(self,obs,base):
        """Ordering proof: free setup search before spending the manual attachment.

        If Solrock is already public, Lunatone is missing, Poke Pad is legal, and
        the base policy wants to attach Fighting to an Active utility Pokemon,
        use Pad first.  This does not consume the manual attachment and can only
        reveal/complete the Solrock-Lunatone engine before the resource decision.
        It is deliberately limited to early/midgame with a non-empty deck.
        """
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not isinstance(base,list) or len(base)!=1:return None
        if _i(cur.get('turn'),99)>6 or _i(mine.get('deckCount'),0)<8 or not mine.get('active'):return None
        bi=_i(base[0],-1)
        if not 0<=bi<len(opts):return None
        bo=opts[bi]
        if _i(bo.get('type'),-1)!=ATTACH or _i(bo.get('inPlayArea'),-1)!=4:return None
        active=mine['active'][0]
        if _i(active.get('id'),-1) not in UTILITY:return None
        ec=_hand_card(mine,bo)
        if not ec or _i(ec.get('id'),-1) not in (6,20):return None
        field=[q for q in (mine.get('active') or [])+(mine.get('bench') or []) if q]
        if not any(_i(q.get('id'),-1)==SOLROCK for q in field) or any(_i(q.get('id'),-1)==LUNATONE for q in field):return None
        # Preserve the stronger generic setup order: if early Poffin can still
        # develop the board without filling the bench, do that before Pad.
        bench_n=sum(1 for q in (mine.get('bench') or []) if q)
        if _i(cur.get('turn'),99)<=3 and bench_n<=3:
            for i,o in enumerate(opts):
                if _i(o.get('type'),-1)!=PLAY:continue
                c=_hand_card(mine,o)
                if c and _i(c.get('id'),-1)==POFFIN:
                    return self._emit([i],'proof:setup_before_utility_energy',setup='poffin',active=_i(active.get('id')),energy=_i(ec.get('id')))
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)!=PLAY:continue
            c=_hand_card(mine,o)
            if c and _i(c.get('id'),-1)==POKE_PAD:
                return self._emit([i],'proof:setup_before_utility_energy',setup='pad',active=_i(active.get('id')),energy=_i(ec.get('id')))
        return None

    def _end_with_real_damage(self,obs,base):
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not isinstance(base,list) or len(base)!=1 or not callable(self.generic_verifier):return None
        bi=_i(base[0],-1)
        if not 0<=bi<len(opts) or _i(opts[bi].get('type'),-1)!=END or not mine.get('active') or not op.get('active'):return None
        active=mine['active'][0];target=op['active'][0];cand=[]
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)!=ATTACK:continue
            dmg=_attack_damage(o.get('attackId'),active,target,cur,op)
            if dmg>0:cand.append((dmg,i,_i(o.get('attackId'),-1)))
        cand.sort(reverse=True)
        for dmg,ci,aid in cand[:2]:
            try:ok,detail=self.generic_verifier(obs,base,[ci])
            except Exception:ok=False;detail={}
            if ok and float((detail or {}).get('gain',-1e9))>=35.0:
                return self._emit([ci],'proof:end_abandons_real_attack',attack=aid,damage=dmg,runtime=detail)
        return None

    def _candidate_ready_retreat(self,obs,base):
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not isinstance(base,list) or len(base)!=1 or not mine.get('active') or not op.get('active'):return None
        bi=_i(base[0],-1)
        if not 0<=bi<len(opts) or _i(opts[bi].get('type'),-1)!=END:return None
        active=mine['active'][0]
        if _i(active.get('id'),-1) not in UTILITY:return None
        # Do not leave a utility attacker that can already make meaningful progress.
        for o in opts:
            if _i(o.get('type'),-1)==ATTACK and _attack_damage(o.get('attackId'),active,op['active'][0],cur,op)>0:return None
        ri=next((i for i,o in enumerate(opts) if _i(o.get('type'),-1)==RETREAT),None)
        if ri is None:return None
        megas=[]
        for j,q in enumerate(mine.get('bench') or []):
            if not q or _i(q.get('id'),-1)!=LUCARIO or _energy_n(q)<1:continue
            defender=_i(op['active'][0].get('id'),-1)
            if sem.damage_prevention_applies(defender,LUCARIO,raw_damage=130):continue
            if sem.global_damage_prevention_applies(_effect_ids(cur,op),defender,LUCARIO,raw_damage=130):continue
            if _attack_damage(AURA,q,op['active'][0],cur,op)<=0:continue
            cape=any(_i(t.get('id'),-1)==HERO_CAPE for t in (q.get('tools') or []));megas.append((1 if cape else 0,_i(q.get('hp')),j,_i(q.get('serial'))))
        if not megas:return None
        megas.sort(reverse=True);return {'candidate':[ri],'active_serial':_i(active.get('serial')),'mega_serial':megas[0][3],'turn':_i(cur.get('turn')),'energy':None}

    def _candidate_pivot(self,obs,base):
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or not _ctx_main(sel) or not isinstance(base,list) or len(base)!=1:return None
        bi=_i(base[0],-1)
        if not 0<=bi<len(opts) or not mine.get('active') or not op.get('active'):return None
        bo=opts[bi]
        if _i(bo.get('type'),-1)!=ATTACH:return None
        src=_hand_card(mine,bo);dst=_target_card(mine,bo);active=mine['active'][0]
        if not src or not dst or _i(src.get('id'),-1) not in {BASIC_F,ROCK_F}:return None
        if _i(active.get('id'),-1) not in UTILITY or _i(dst.get('serial'),-1)!=_i(active.get('serial'),-2):return None
        # Do not abandon a guaranteed KO from the current utility attacker.
        for o in opts:
            if _i(o.get('type'),-1)==ATTACK:
                if _attack_damage(o.get('attackId'),active,op['active'][0],cur,op)>=_i(op['active'][0].get('hp'),9999):return None
        # Candidate Mega Lucario must become immediately attack-ready with this one Energy.
        megas=[]
        for j,q in enumerate(mine.get('bench') or []):
            if q and _i(q.get('id'),-1)==LUCARIO and _energy_n(q)<1:
                # Text-proven ex/Ability wall makes the pivot invalid.
                defender=_i(op['active'][0].get('id'),-1)
                if sem.damage_prevention_applies(defender,LUCARIO,raw_damage=130):continue
                if sem.global_damage_prevention_applies(_effect_ids(cur,op),defender,LUCARIO,raw_damage=130):continue
                # Prefer Cape/full-HP bodies; serial is the stable switch target.
                cape=any(_i(t.get('id'),-1)==HERO_CAPE for t in (q.get('tools') or []))
                megas.append((1 if cape else 0,_i(q.get('hp')), -j, j, _i(q.get('serial'))))
        if not megas:return None
        # The utility Active must have a public same-turn exit after redirecting Energy.
        retreat_now=any(_i(o.get('type'),-1)==RETREAT for o in opts)
        balloon=False
        for o in opts:
            if _i(o.get('type'),-1)!=ATTACH:continue
            sc=_hand_card(mine,o);tc=_target_card(mine,o)
            if sc and tc and _i(sc.get('id'),-1)==AIR_BALLOON and _i(tc.get('serial'),-1)==_i(active.get('serial'),-2):balloon=True
        if not (retreat_now or balloon):return None
        megas.sort(reverse=True);target_slot=megas[0][3];target_serial=megas[0][4]
        # Prefer Basic Fighting when the same state offers it; otherwise preserve the source.
        cand=[]
        for i,o in enumerate(opts):
            if _i(o.get('type'),-1)!=ATTACH:continue
            sc=_hand_card(mine,o);tc=_target_card(mine,o)
            if not sc or not tc or _i(tc.get('serial'),-1)!=target_serial:continue
            sid=_i(sc.get('id'),-1)
            if sid in {BASIC_F,ROCK_F}:cand.append((0 if sid==BASIC_F else 1,i,sid))
        if not cand:return None
        cand.sort();ci=cand[0][1]
        if ci==bi:return None
        return {'candidate':[ci],'active_serial':_i(active.get('serial')),'mega_serial':target_serial,'turn':_i(cur.get('turn')),'energy':cand[0][2]}
    def _continue_pending(self,obs,base):
        if not self.pending:return None
        cur,mine,op=_players(obs);sel=obs.get('select') or {};opts=sel.get('option') or []
        if cur is None or _i(cur.get('turn'))!=self.pending['turn']:
            self._clear();return None
        active=(mine.get('active') or [None])[0]
        if active is None:self._clear();return None
        kind=self.pending.get('kind')
        if kind=='boss_win':
            # Boss target selection is usually a CARD/SWITCH-like context.
            if not _ctx_main(sel):
                for i,o in enumerate(opts):
                    q=_opt_any_card(cur,o)
                    if q and _i(q.get('serial'),-1)==self.pending.get('target_serial'):
                        return self._emit([i],'proof:boss_game_win_target')
                self._clear();return None
            # After the switch, finish with the pre-proven attack.
            for i,o in enumerate(opts):
                if _i(o.get('type'),-1)==ATTACK and _i(o.get('attackId'),-1)==self.pending.get('attack'):
                    self._clear();return self._emit([i],'proof:boss_game_win_attack')
            # Let harmless setup continue, but never overwrite with a different supporter/end.
            if isinstance(base,list) and len(base)==1 and 0<=_i(base[0],-1)<len(opts):
                bt=_i(opts[_i(base[0])].get('type'),-1)
                if bt in {END,ATTACK}:self._clear()
            return None
        if kind=='cosmic':
            if _ctx_main(sel):
                # Once Lunatone is benched, allow setup but prevent ending before the real Cosmic Beam.
                if _has_luna(mine) and isinstance(base,list) and len(base)==1 and 0<=_i(base[0],-1)<len(opts):
                    bo=opts[_i(base[0])];bt=_i(bo.get('type'),-1)
                    if bt==END:
                        for i,o in enumerate(opts):
                            if _i(o.get('type'),-1)==ATTACK and _i(o.get('attackId'),-1)==COSMIC:
                                self._clear();return self._emit([i],'proof:no_effect_cosmic_finish')
                    if bt==ATTACK:
                        if _i(bo.get('attackId'),-1)==COSMIC:self._clear()
                        else:self._clear()
            return None
        if kind=='generic_retreat_attack':
            if not _ctx_main(sel):
                for i,o in enumerate(opts):
                    q=_opt_any_card(cur,o)
                    if q and _i(q.get('serial'),-1)==self.pending.get('target_serial'):
                        return self._emit([i],'proof:generic_retreat_choose_attacker')
                self._clear();return None
            if _i(active.get('serial'),-1)==self.pending.get('target_serial'):
                if isinstance(base,list) and len(base)==1 and 0<=_i(base[0],-1)<len(opts) and _i(opts[_i(base[0])].get('type'),-1)==END:
                    damage=[]
                    target=op['active'][0] if op.get('active') else None
                    if target:
                        for i,o in enumerate(opts):
                            if _i(o.get('type'),-1)==ATTACK:
                                dmg=_attack_damage(o.get('attackId'),active,target,cur,op)
                                if dmg>0:damage.append((dmg,i))
                    if damage:
                        damage.sort(reverse=True);self._clear();return self._emit([damage[0][1]],'proof:generic_retreat_finish_attack')
                return None
            self._clear();return None
        # Retreat target selection.
        if not _ctx_main(sel):
            for i,o in enumerate(opts):
                q=_opt_card_from_area(mine,o)
                if q and _i(q.get('serial'),-1)==self.pending['mega_serial']:
                    return self._emit([i],'proof:pivot_choose_mega')
            return None
        aid=_i(active.get('serial'),-1)
        if aid==self.pending['active_serial']:
            # Air Balloon first if it is the remaining exit enabler.
            for i,o in enumerate(opts):
                if _i(o.get('type'),-1)!=ATTACH:continue
                sc=_hand_card(mine,o);tc=_target_card(mine,o)
                if sc and tc and _i(sc.get('id'),-1)==AIR_BALLOON and _i(tc.get('serial'),-1)==aid:
                    return self._emit([i],'proof:pivot_air_balloon')
            for i,o in enumerate(opts):
                if _i(o.get('type'),-1)==RETREAT:return self._emit([i],'proof:pivot_retreat')
            self._clear();return None
        if aid==self.pending['mega_serial']:
            # Let useful setup/PPP proceed, but never end the turn without Aura Jab.
            if isinstance(base,list) and len(base)==1 and 0<=_i(base[0],-1)<len(opts):
                bo=opts[_i(base[0])]
                if _i(bo.get('type'),-1)==END:
                    for i,o in enumerate(opts):
                        if _i(o.get('type'),-1)==ATTACK and _i(o.get('attackId'),-1)==AURA:
                            self._clear();return self._emit([i],'proof:pivot_finish_aura')
                if _i(bo.get('type'),-1)==ATTACK:
                    self._clear()
            return None
        self._clear();return None
    def choose(self,obs,base):
        self.stats['calls']+=1
        try:
            if not isinstance(base,list):return base
            if (obs.get('select') is None and obs.get('current') is None):self.reset();return base
            x=self._continue_pending(obs,base)
            if x is not None:return x
            x=self._game_winning_attack(obs,base)
            if x is not None:return x
            x=self._boss_game_win(obs,base)
            if x is not None:return x
            x=self._robust_exact_cf(obs,base)
            if x is not None:return x
            x=self._no_effect_cosmic(obs,base)
            if x is not None:return x
            x=self._pad_before_utility_energy(obs,base)
            if x is not None:return x
            x=self._end_with_real_damage(obs,base)
            if x is not None:return x
            if self.search_calls>=2 or not callable(self.verifier):return base
            meta=self._candidate_ready_retreat(obs,base)
            if meta is None:meta=self._candidate_pivot(obs,base)
            if meta:
                self.search_calls+=1;self.stats['search_calls']+=1
                try:
                    ok,detail=self.verifier(obs,base,meta['candidate'],meta)
                except Exception as e:
                    self.stats['errors']+=1;self.stats['last']={'reason':'pivot_verify_error','error':repr(e)[:160]};return base
                if ok:
                    self.stats['search_accept']+=1;self.pending=dict(meta)
                    return self._emit(meta['candidate'],'proof:two_ply_utility_to_lucario',gain=(detail or {}).get('gain'))
                self.stats['search_reject']+=1
            return base
        except Exception as e:
            self.stats['errors']+=1;self.stats['last']={'reason':'error','error':repr(e)[:160]};return base
    def get_stats(self):return dict(self.stats)
