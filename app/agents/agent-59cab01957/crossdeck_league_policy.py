"""Cross-deck league residual for the Dragapult submission.

This layer is trained/gated on local cross-play rather than replacing the
validated policy.  It compiles visible card text into causal tags, then repairs
only decisions with a provable dominated alternative:

* an attack whose direct damage is prevented and whose Bench-counter effect is
  also prevented;
* evolving away the last non-ex wall breaker while a visible wall plus a
  Bench-counter lock remains;
* wasting an attachment on an already-ready ex attacker when the same Energy
  advances a legal non-ex wall breaker;
* escaping a currently effective text-defined wall with a same-turn Boss KO.

No network/model runtime is required.  Every output is one of the engine's
legal option indices, and the existing deterministic safety guards still run
later in main.py.
"""
from __future__ import annotations

import json,re
from collections import Counter,defaultdict,deque


def _i(v,d=0):
    try:return int(v) if v is not None else d
    except Exception:return d

def _norm(s):
    return re.sub(r"\s+"," ",str(s or "").replace("’","'")).strip().lower()

class CrossDeckLeaguePolicy:
    def __init__(self,base,api,config_path=None):
        self.base=base; self.api=api
        self.cards=getattr(base,'card_table',{})
        self.attacks=getattr(base,'attack_table',{})
        self.cfg={
            'block_zero_wall_attacks':True,
            'preserve_last_wall_breaker':True,
            'redirect_wall_attachments':True,
            'boss_wall_escape':True,
            'boss_evolution_denial':True,
            'boss_threat_threshold':5200,
            'prefer_drakloak_over_munkidori':True,
            'wall_seen_board_only':True,
        }
        if config_path:
            try:
                with open(config_path,encoding='utf-8') as f:self.cfg.update(json.load(f))
            except Exception:pass
        self.stats=Counter(); self._text_cache={}; self._wall_cache={};self._bench_lock_cache={};self._effect_lock_cache={}
        self._descendants=self._build_descendants()
        self.pending_wall_switch=False
        self.pending_boss=False

    def reset(self):
        self.pending_wall_switch=False;self.pending_boss=False

    def get_stats(self):
        return {'league_'+str(k):int(v) for k,v in self.stats.items()}

    def _build_descendants(self):
        children=defaultdict(list)
        for c in self.cards.values():
            parent=_norm(getattr(c,'evolvesFrom',None))
            if parent:children[parent].append(c)
        out={}
        for c in self.cards.values():
            root=_norm(getattr(c,'name',''))
            q=deque(children.get(root,[]));seen=set();rows=[]
            while q:
                x=q.popleft();cid=_i(getattr(x,'cardId',0))
                if cid in seen:continue
                seen.add(cid);rows.append(x);q.extend(children.get(_norm(getattr(x,'name','')),[]))
            out[_i(getattr(c,'cardId',0))]=rows
        return out

    def _text(self,cid):
        cid=_i(cid)
        if cid in self._text_cache:return self._text_cache[cid]
        c=self.cards.get(cid);parts=[]
        if c:
            for s in list(getattr(c,'skills',[]) or []):parts += [getattr(s,'name',''),getattr(s,'text','')]
            for aid in list(getattr(c,'attacks',[]) or []):
                a=self.attacks.get(_i(aid))
                if a:parts += [getattr(a,'name',''),getattr(a,'text','')]
        self._text_cache[cid]=_norm(' '.join(parts));return self._text_cache[cid]

    def _wall_tags(self,cid):
        cid=_i(cid)
        if cid in self._wall_cache:return self._wall_cache[cid]
        t=self._text(cid)
        tags={
          'all': 'prevent all damage done to this pokémon by attacks' in t and 'pokémon {ex}' not in t and 'pokemon {ex}' not in t and 'have an ability' not in t,
          'ex': 'prevent all damage done to this pokémon by attacks' in t and ('pokémon {ex}' in t or 'pokemon {ex}' in t or 'pokémon ex' in t or 'pokemon ex' in t),
          'ability': 'prevent all damage from attacks done to this pokémon' in t and 'have an ability' in t,
        }
        self._wall_cache[cid]=tags;return tags

    def _bench_counter_lock(self,cid):
        cid=_i(cid)
        if cid in self._bench_lock_cache:return self._bench_lock_cache[cid]
        t=self._text(cid);v=(('prevent all damage counters' in t or 'damage counters from being placed' in t) and 'benched pok' in t)
        self._bench_lock_cache[cid]=bool(v);return bool(v)

    def _effect_lock(self,cid):
        cid=_i(cid)
        if cid in self._effect_lock_cache:return self._effect_lock_cache[cid]
        t=self._text(cid);v=('prevent all effects of attacks' in t)
        self._effect_lock_cache[cid]=bool(v);return bool(v)

    def _board(self,p):return [x for x in list(p.active or [])+list(p.bench or []) if x is not None]

    def _source(self,obs,o):
        try:return self.base.get_card(obs,o.area,o.index,o.playerIndex)
        except Exception:
            try:
                me=obs.current.yourIndex;p=obs.current.players[me]
                if _i(o.type) in (_i(self.api.OptionType.PLAY),_i(self.api.OptionType.ATTACH),_i(self.api.OptionType.EVOLVE)):
                    return list(p.hand or [])[o.index]
            except Exception:pass
        return None

    def _cid(self,obs,o):
        c=self._source(obs,o)
        return _i(getattr(c,'id',getattr(o,'cardId',0)))

    def _target(self,obs,o):
        try:
            p=obs.current.players[obs.current.yourIndex]
            ar=_i(getattr(o,'inPlayArea',-1),-1);ix=_i(getattr(o,'inPlayIndex',-1),-1)
            arr=p.active if ar==4 else p.bench if ar==5 else []
            return list(arr)[ix] if 0<=ix<len(arr) else None
        except Exception:return None

    def _option_card(self,obs,o):
        try:return self.base.get_card(obs,o.area,o.index,o.playerIndex)
        except Exception:return None

    def _has_ability(self,p):
        c=self.cards.get(_i(getattr(p,'id',0)))
        return bool(c and list(getattr(c,'skills',[]) or []))

    def _damage_blocked(self,attacker,target):
        if attacker is None or target is None:return False
        tags=self._wall_tags(getattr(target,'id',0));ac=self.cards.get(_i(getattr(attacker,'id',0)))
        if tags['all']:return True
        if tags['ex'] and ac is not None and (bool(getattr(ac,'ex',False)) or bool(getattr(ac,'megaEx',False))):return True
        if tags['ability'] and self._has_ability(attacker):return True
        return False

    def _stadium_lock(self,obs):
        if obs.current is None:return False
        return any(s is not None and self._bench_counter_lock(getattr(s,'id',0)) for s in list(obs.current.stadium or []))

    def _provided_types(self,p):
        vals=[]
        for e in list(getattr(p,'energyCards',[]) or []):
            c=self.cards.get(_i(getattr(e,'id',e)))
            et=_i(getattr(c,'energyType',0)) if c else 0
            vals.append(et)
        return vals

    def _can_pay(self,cost,provided):
        pool=list(provided);colors=[];colorless=0
        for e in list(cost or []):
            x=_i(e)
            if x==0:colorless+=1
            else:colors.append(x)
        for x in colors:
            if x in pool:pool.remove(x)
            elif 10 in pool:pool.remove(10)
            elif 11 in pool and x in (5,7):pool.remove(11)
            else:return False
        return len(pool)>=colorless

    def _ready_nonex_attacks(self,p):
        c=self.cards.get(_i(getattr(p,'id',0)))
        if c is None or bool(getattr(c,'ex',False)) or bool(getattr(c,'megaEx',False)):return []
        provided=self._provided_types(p);out=[]
        for aid in list(getattr(c,'attacks',[]) or []):
            a=self.attacks.get(_i(aid))
            if a and self._can_pay(getattr(a,'energies',[]),provided):out.append(a)
        return out

    def _breaker_score(self,p):
        ats=self._ready_nonex_attacks(p)
        if not ats:return -1
        best=max((_i(getattr(a,'damage',0)) + (35 if 'confused' in _norm(getattr(a,'text','')) else 0) for a in ats),default=0)
        # Drakloak is retained ahead of Munkidori in the accepted wall plan:
        # 70 deterministic damage beats 60 plus coin-dependent confusion.
        if _i(getattr(p,'id',0))==120:best+=18
        elif _i(getattr(p,'id',0))==112:best+=10
        return best

    def _wall_visible(self,opp,attacker):
        return any(self._damage_blocked(attacker,p) for p in self._board(opp))

    def _wall_evidence(self,obs,opp,attacker=None):
        """True only after the opponent publicly reveals a text-defined wall.

        Discard evidence is retained because a second copy can still be in the
        deck/hand and the matchup plan should not forget the revealed mechanic.
        A Bench-counter-lock Stadium is also direct public evidence.
        """
        rows=self._board(opp)+[x for x in list(getattr(opp,'discard',[]) or []) if x is not None]
        for p in rows:
            tags=self._wall_tags(getattr(p,'id',0))
            if any(tags.values()):return True
        return self._stadium_lock(obs)

    def _phantom_bench_value(self,obs,opp):
        if self._stadium_lock(obs):return 0
        val=0
        for p in list(opp.bench or []):
            if p is None:continue
            # Special Energy such as Mist Energy can protect a target from the
            # effect even when the Stadium does not.
            if any(self._effect_lock(getattr(e,'id',0)) for e in list(getattr(p,'energyCards',[]) or [])):continue
            hp=_i(getattr(p,'hp',0));val+=max(1,70-min(60,hp))
            if hp<=60:val+=8000*(3 if bool(getattr(self.cards.get(_i(p.id)),'megaEx',False)) else 2 if bool(getattr(self.cards.get(_i(p.id)),'ex',False)) else 1)
        return val

    def _legal_attacks(self,obs):
        out=[]
        for i,o in enumerate(list(obs.select.option or [])):
            if _i(o.type)==_i(self.api.OptionType.ATTACK):
                a=self.attacks.get(_i(getattr(o,'attackId',0)))
                if a:out.append((i,a))
        return out

    def _direct_damage(self,a):return _i(getattr(a,'damage',0))

    def _prize(self,p):
        c=self.cards.get(_i(getattr(p,'id',0)))
        return 3 if c and bool(getattr(c,'megaEx',False)) else 2 if c and bool(getattr(c,'ex',False)) else 1

    def _evolution_threat(self,p):
        score=0
        for c in self._descendants.get(_i(getattr(p,'id',0)),[]):
            if bool(getattr(c,'megaEx',False)):score=max(score,9200)
            elif bool(getattr(c,'ex',False)):score=max(score,5400)
            txt=self._text(getattr(c,'cardId',0))
            if 'attach up to' in txt and 'energy' in txt:score=max(score,7600)
            for aid in list(getattr(c,'attacks',[]) or []):
                a=self.attacks.get(_i(aid));score=max(score,_i(getattr(a,'damage',0))*22 if a else 0)
        return score

    def _boss_targets(self,obs,attacker,attacks,wall_escape=False):
        if obs.current is None:return []
        st=obs.current;me=st.yourIndex;opp=st.players[1-me]
        maxd=max((self._direct_damage(a) for _,a in attacks),default=0)
        out=[]
        for p in list(opp.bench or []):
            if p is None or self._damage_blocked(attacker,p):continue
            hp=_i(getattr(p,'hp',0));ko=maxd>=hp and maxd>0
            if not ko:continue
            threat=self._evolution_threat(p)
            score=self._prize(p)*10000 + threat + max(0,maxd-hp)
            if wall_escape:score+=9000
            out.append((score,p))
        return sorted(out,key=lambda z:(-z[0],_i(getattr(z[1],'hp',999)),_i(getattr(z[1],'serial',0))))

    def _boss_play(self,obs):
        for i,o in enumerate(list(obs.select.option or [])):
            if _i(o.type)==_i(self.api.OptionType.PLAY) and self._cid(obs,o)==1182:return i
        return None

    def _safe_stadium(self,obs):
        cand=[]
        for i,o in enumerate(list(obs.select.option or [])):
            if _i(o.type)!=_i(self.api.OptionType.PLAY):continue
            cid=self._cid(obs,o);c=self.cards.get(cid)
            if c is None or _i(getattr(c,'cardType',-1),-1)!=_i(self.api.CardType.STADIUM):continue
            if not self._bench_counter_lock(cid):cand.append((0 if cid==1246 else 1,i))
        return min(cand)[1] if cand else None

    def _attach_gain(self,p,eid):
        c=self.cards.get(_i(getattr(p,'id',0)))
        if c is None or bool(getattr(c,'ex',False)) or bool(getattr(c,'megaEx',False)):return -1
        before=max((self._breaker_score(p),0))
        types=self._provided_types(p)
        ec=self.cards.get(_i(eid));types2=types+([_i(getattr(ec,'energyType',0))] if ec else [0])
        after=0
        for aid in list(getattr(c,'attacks',[]) or []):
            a=self.attacks.get(_i(aid))
            if a and self._can_pay(getattr(a,'energies',[]),types2):
                after=max(after,_i(getattr(a,'damage',0))+(35 if 'confused' in _norm(getattr(a,'text','')) else 0))
        gain=(after-before)*100
        # Partial progress toward the deck's two principal non-ex breakers.
        if _i(getattr(p,'id',0))==120:
            have=set(types);et=_i(getattr(ec,'energyType',0)) if ec else 0
            if et in (2,5) and et not in have:gain+=2100
        elif _i(getattr(p,'id',0))==112:
            have=set(types);et=_i(getattr(ec,'energyType',0)) if ec else 0
            if et==5 and 5 not in have:gain+=2300
            elif et==7 and 7 not in have:gain+=1900
            elif len(types)<2:gain+=700
        elif _i(getattr(p,'id',0))==119:
            have=set(types);et=_i(getattr(ec,'energyType',0)) if ec else 0
            if et in (2,5) and et not in have:gain+=1000
        return gain

    def _best_breaker_attach(self,obs):
        rows=[]
        for i,o in enumerate(list(obs.select.option or [])):
            if _i(o.type)!=_i(self.api.OptionType.ATTACH):continue
            p=self._target(obs,o);eid=self._cid(obs,o)
            if p is None:continue
            g=self._attach_gain(p,eid)
            if g>0:rows.append((g, 2 if _i(p.id)==120 else 1 if _i(p.id)==112 else 0, i))
        return max(rows)[2] if rows else None

    def _ready_breakers(self,mine):
        rows=[]
        for p in list(mine.bench or []):
            if p is not None:
                s=self._breaker_score(p)
                if s>=0:rows.append((s,p))
        return sorted(rows,key=lambda z:(-z[0],_i(z[1].serial)))

    def _chosen_is_last_breaker_evolve(self,obs,chosen,opp,attacker):
        if not self.cfg.get('preserve_last_wall_breaker') or not isinstance(chosen,list) or len(chosen)!=1:return False
        opts=list(obs.select.option or []);i=chosen[0]
        if not (0<=i<len(opts)) or _i(opts[i].type)!=_i(self.api.OptionType.EVOLVE) or self._cid(obs,opts[i])!=121:return False
        target=self._target(obs,opts[i])
        if target is None or _i(target.id)!=120:return False
        mine=obs.current.players[obs.current.yourIndex]
        other=[p for p in self._board(mine) if _i(p.serial)!=_i(target.serial) and self._breaker_score(p)>=0]
        return (not other) and self._wall_visible(opp,attacker) and self._stadium_lock(obs)

    def _nonzero_fallback(self,obs,chosen,attacker,opp):
        opts=list(obs.select.option or [])
        # Text-grounded Stadium answer first.
        s=self._safe_stadium(obs)
        if s is not None:self.stats['wall_stadium_answer']+=1;return [s]
        # Value abilities are free before a later attack/END.
        for i,o in enumerate(opts):
            if _i(o.type)==_i(self.api.OptionType.ABILITY) and self._cid(obs,o)==120:
                self.stats['wall_recon_before_pass']+=1;return [i]
        a=self._best_breaker_attach(obs)
        if a is not None:self.stats['wall_breaker_attach']+=1;return [a]
        # Evolve a Dreepy into the non-ex breaker, not the breaker into an ex.
        for i,o in enumerate(opts):
            if _i(o.type)==_i(self.api.OptionType.EVOLVE) and self._cid(obs,o)==120:
                self.stats['wall_build_drakloak']+=1;return [i]
        mine=obs.current.players[obs.current.yourIndex]
        if len(list(mine.bench or []))<_i(mine.benchMax):
            for want in (112,119):
                for i,o in enumerate(opts):
                    if _i(o.type)==_i(self.api.OptionType.PLAY) and self._cid(obs,o)==want:
                        self.stats['wall_bench_breaker']+=1;return [i]
        for i,o in enumerate(opts):
            if _i(o.type)==_i(self.api.OptionType.END):self.stats['wall_end_over_zero']+=1;return [i]
        return chosen

    def _selection_override(self,obs,chosen):
        if obs.select is None or obs.current is None:return None
        ctx=_i(obs.select.context);opts=list(obs.select.option or [])
        st=obs.current;me=st.yourIndex;mine=st.players[me];opp=st.players[1-me]
        if ctx in (_i(self.api.SelectContext.SWITCH),_i(self.api.SelectContext.TO_ACTIVE)):
            # Boss target selection is identifiable from the effect card or the
            # pending main action.  Choose an exact-KO target and avoid another
            # text-defined wall.
            eff=_i(getattr(getattr(obs.select,'effect',None),'id',0))
            attacker=mine.active[0] if mine.active and mine.active[0] is not None else None
            scoped_base_boss=(eff==1182 and self.cfg.get('wall_scoped_base_boss_target',False) and self._wall_evidence(obs,opp,attacker))
            if self.pending_boss or scoped_base_boss:
                ats=self._legal_attacks(obs) if ctx==_i(self.api.SelectContext.MAIN) else []
                # The attack prompt is not current during Boss resolution; use
                # printed attacks that are already payable on the active.
                rows=[]
                if attacker is not None:
                    ac=self.cards.get(_i(attacker.id));provided=self._provided_types(attacker)
                    for aid in list(getattr(ac,'attacks',[]) or []) if ac else []:
                        a=self.attacks.get(_i(aid))
                        if a and self._can_pay(getattr(a,'energies',[]),provided):rows.append((0,a))
                maxd=max((self._direct_damage(a) for _,a in rows),default=0)
                cand=[]
                for i,o in enumerate(opts):
                    p=self._option_card(obs,o)
                    if p is None or _i(getattr(o,'playerIndex',me))==me or self._damage_blocked(attacker,p):continue
                    if maxd>=_i(p.hp):cand.append((self._prize(p)*10000+self._evolution_threat(p),-_i(p.hp),i))
                self.pending_boss=False
                if cand:
                    self.stats['boss_target_exact_ko']+=1
                    if scoped_base_boss:self.stats['wall_scoped_base_boss_target']+=1
                    return [max(cand)[2]]
            if self.pending_wall_switch:
                cand=[]
                for i,o in enumerate(opts):
                    if _i(getattr(o,'playerIndex',me))!=me:continue
                    p=self._option_card(obs,o)
                    if p is not None:
                        s=self._breaker_score(p)
                        if s>=0:cand.append((s,i))
                self.pending_wall_switch=False
                if cand:self.stats['wall_switch_to_breaker']+=1;return [max(cand)[1]]
        return None

    def rerank(self,observation,chosen,history=None):
        try:
            if not observation.get('select'):
                if observation.get('current') is None:self.reset()
                return chosen
            obs=self.api.to_observation_class(observation)
            if obs.select is None or obs.current is None:return chosen
            # A pending choice is valid only for the immediate non-MAIN
            # resolution generated by the action that armed it.  If the
            # downstream deterministic guard replaced that action, the next
            # prompt is MAIN again and the intent must not leak into an
            # unrelated later switch/Boss selection.
            if _i(obs.select.context)==_i(self.api.SelectContext.MAIN):
                if self.pending_boss:
                    self.pending_boss=False;self.stats['cleared_stale_boss_intent']+=1
                if self.pending_wall_switch:
                    self.pending_wall_switch=False;self.stats['cleared_stale_switch_intent']+=1
            ov=self._selection_override(obs,chosen)
            if ov is not None:return ov
            if _i(obs.select.context)!=_i(self.api.SelectContext.MAIN) or not isinstance(chosen,list) or len(chosen)!=1:return chosen
            st=obs.current;me=st.yourIndex;mine=st.players[me];opp=st.players[1-me]
            attacker=mine.active[0] if mine.active and mine.active[0] is not None else None
            target=opp.active[0] if opp.active and opp.active[0] is not None else None
            if attacker is None or target is None:return chosen
            opts=list(obs.select.option or []);ci=chosen[0]
            if not (0<=ci<len(opts)):return chosen
            legal_attacks=self._legal_attacks(obs)
            direct_block=self._damage_blocked(attacker,target)
            wall_visible=self._wall_visible(opp,attacker)
            lock=self._stadium_lock(obs)

            # Empirical wall-preservation gate learned from the 62-loss league:
            # do not turn the only ready Drakloak into an ex while the opponent
            # can restore an ex-damage wall and the Stadium already removes the
            # Phantom counter plan.
            if self._chosen_is_last_breaker_evolve(obs,chosen,opp,attacker):
                fb=self._nonzero_fallback(obs,chosen,attacker,opp)
                if fb!=chosen:self.stats['preserve_last_wall_breaker']+=1;return fb

            # Redirect a wasted third/fourth Energy from a ready ex toward the
            # non-ex answer whenever that attachment makes measurable progress.
            if self.cfg.get('redirect_wall_attachments') and wall_visible and lock and _i(opts[ci].type)==_i(self.api.OptionType.ATTACH):
                dest=self._target(obs,opts[ci]);best=self._best_breaker_attach(obs)
                if best is not None and (dest is None or _i(dest.id)==121 or self._attach_gain(dest,self._cid(obs,opts[ci]))<=0) and best!=ci:
                    self.stats['redirect_wall_attachment']+=1;return [best]

            if not direct_block:return chosen
            phantom=next(((i,a) for i,a in legal_attacks if _i(getattr(a,'attackId',0))==154),None)
            bench_value=self._phantom_bench_value(obs,opp) if phantom else 0
            chosen_attack=_i(opts[ci].type)==_i(self.api.OptionType.ATTACK)
            chosen_aid=_i(getattr(opts[ci],'attackId',0)) if chosen_attack else 0
            zero_attack=chosen_attack and (chosen_aid!=154 or bench_value<=0)

            # Replace a global Bench-counter lock before any zero attack.
            if lock:
                s=self._safe_stadium(obs)
                if s is not None and s!=ci:self.stats['wall_stadium_answer']+=1;return [s]

            # A Boss KO on a non-wall bench target converts the otherwise dead
            # attack and frequently removes the next Crustle/Duraludon/Riolu.
            boss=self._boss_play(obs)
            if self.cfg.get('boss_wall_escape') and boss is not None:
                tg=self._boss_targets(obs,attacker,legal_attacks,wall_escape=True)
                if tg:
                    self.pending_boss=True;self.stats['boss_wall_escape']+=1;return [boss]

            # Switch a ready non-ex attacker into the wall.
            breakers=self._ready_breakers(mine)
            if breakers:
                ret=next((i for i,o in enumerate(opts) if _i(o.type)==_i(self.api.OptionType.RETREAT)),None)
                if ret is not None:
                    self.pending_wall_switch=True;self.stats['wall_retreat_to_breaker']+=1;return [ret]

            # If Phantom can still place counters, preserve that real value.
            if phantom is not None and bench_value>0:
                if chosen_attack and chosen_aid!=154:self.stats['wall_phantom_over_jet']+=1;return [phantom[0]]
                return chosen

            if zero_attack and self.cfg.get('block_zero_wall_attacks'):
                self.stats['blocked_zero_wall_attack']+=1
                return self._nonzero_fallback(obs,chosen,attacker,opp)

            # Even when the base chose a non-attack, advance a breaker before
            # arbitrary low-value setup if the attack route is completely shut.
            if lock and phantom is not None and bench_value<=0:
                best=self._best_breaker_attach(obs)
                if best is not None and best!=ci:
                    self.stats['wall_breaker_attach']+=1;return [best]
            return chosen
        except Exception:
            self.stats['exceptions']+=1;return chosen
