from __future__ import annotations
import math, inspect, types, random, hashlib
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict

class UnifiedController:
    """Conservative common meta-controller.

    The deck-specific policy proposes the default action and handles effect choices.
    This controller owns opponent recognition, hidden-state reconstruction, candidate
    generation, shallow turn search, generic state valuation, replay priors and
    regression-safe override thresholds.
    """
    def __init__(self, base, api, config, deck_catalog, replay_prior=None, opponent_bank=None):
        self.base=base
        self.api=api
        self.cfg=config
        self.catalog={k:list(v) for k,v in deck_catalog.items()}
        self.prior=replay_prior or {}
        self.opponent_bank=opponent_bank
        self.my_deck=list(config['my_deck'])
        self._seen=set(); self._turn_searched=-1; self._native_root_turn=-1; self._loop_sig=None; self._loop_count=0; self.stats={'searches':0,'overrides':0,'recognized':Counter(),'opponent_turns':0,'opponent_actions':0,'opponent_errors':0,'native_root_calls':0,'hybrid_searches':0}
        self._cards={c.cardId:c for c in api.all_card_data()}
        self._attacks={a.attackId:a for a in api.all_attack()}
        self._variant_names=set((config.get('exact_opponent_variants') or {}).keys())
        self._deck_sets={k:set(v) for k,v in self.catalog.items() if k not in self._variant_names}
        occ=Counter()
        for s in self._deck_sets.values():
            for x in s: occ[x]+=1
        self._idf={x:math.log((len(self.catalog)+1)/(n+0.35))+0.25 for x,n in occ.items()}
        self._base_search_names=(
            'USE_TURN_LOOKAHEAD','USE_LOOKAHEAD','ENABLE_SEARCH','SEARCH_ENABLED',
            '_LOOKAHEAD_ACTIVE','_searched_turn','LOOKAHEAD_MAX_CANDIDATES',
        )

    def reset(self):
        self._seen.clear(); self._turn_searched=-1; self._native_root_turn=-1; self._loop_sig=None; self._loop_count=0

    def fallback(self,d):
        s=d.get('select') or {}; opts=s.get('option') or []
        mn=int(s.get('minCount',0) or 0); mx=int(s.get('maxCount',mn) or mn)
        if not opts or mx<=0:return []
        n=mn if mn>0 else 1
        return list(range(min(len(opts),mx,n)))

    def validate(self,a,sel):
        if not isinstance(a,list):return self._fallback_sel(sel)
        z=[]
        for x in a:
            if isinstance(x,int) and 0<=x<len(sel.option) and x not in z:z.append(x)
        if len(z)<sel.minCount:return self._fallback_sel(sel)
        return z[:sel.maxCount]

    def _fallback_sel(self,sel):
        if not sel.option or sel.maxCount<=0:return []
        n=sel.minCount if sel.minCount>0 else 1
        return list(range(min(len(sel.option),sel.maxCount,n)))

    def _search_flags(self,turn):
        old={}
        for n in self._base_search_names:
            if hasattr(self.base,n):
                old[n]=getattr(self.base,n)
                if n in ('USE_TURN_LOOKAHEAD','USE_LOOKAHEAD','ENABLE_SEARCH','SEARCH_ENABLED'):
                    setattr(self.base,n,False)
                elif n=='_LOOKAHEAD_ACTIVE': setattr(self.base,n,True)
                elif n=='_searched_turn': setattr(self.base,n,turn)
                elif n=='LOOKAHEAD_MAX_CANDIDATES': setattr(self.base,n,0)
        return old

    def _restore_flags(self,old):
        for n,v in old.items():setattr(self.base,n,v)

    def call_base(self,d,turn=0,disable_search=True):
        if not disable_search:
            return self.base.agent(d)
        old=self._search_flags(turn)
        try:return self.base.agent(d)
        finally:self._restore_flags(old)

    def _param(self,name,key,default=None):
        ov=(self.cfg.get('matchup_overrides') or {}).get(name,{})
        return ov.get(key,self.cfg.get(key,default))

    def snapshot(self):
        out={}; skip={'my_deck','MY_DECK','all_card','card_table','_cards','_attacks'}
        for n,v in self.base.__dict__.items():
            if n in skip or n.startswith('__') or n.isupper():continue
            if isinstance(v,(types.ModuleType,type)) or inspect.isfunction(v):continue
            if isinstance(v,(int,bool,str,float,list,dict,set,Counter,defaultdict,tuple)) or v.__class__.__name__ in {'AttackPlan','Forecast','Plan','MachineState'}:
                try:out[n]=deepcopy(v)
                except Exception:pass
        return out

    def restore(self,s):
        for n,v in s.items():setattr(self.base,n,v)

    def _card_data(self,cid):return self._cards.get(int(cid or 0))
    def _card_weight(self,cid):
        cd=self._card_data(cid); w=self._idf.get(cid,0.05)
        if cd:
            if int(cd.cardType)==0:w*=3.0
            elif int(cd.cardType) in (5,6):w*=0.12
            else:w*=0.45
        return w

    def recognize(self,obs):
        s=obs.current
        if s is None:return None,0.0
        if s.turn==0:self._seen.clear()
        me=s.yourIndex; op=s.players[1-me]
        for p in list(op.active)+list(op.bench):
            if p:
                self._seen.add(p.id)
                self._seen.update(x.id for x in p.preEvolution)
                self._seen.update(x.id for x in p.energyCards)
                self._seen.update(x.id for x in p.tools)
        self._seen.update(x.id for x in op.discard if x)
        for l in obs.logs:
            if l.playerIndex==1-me:
                for x in (l.cardId,l.cardIdAfter,l.cardIdBefore,l.cardIdActive,l.cardIdBench,l.cardIdTarget):
                    if x and x>0:self._seen.add(x)
        # Exact variants are considered only after distinctive signature cards are visible.
        # They do not compete in the normal archetype classifier, avoiding early false positives.
        for vname,spec in (self.cfg.get('exact_opponent_variants') or {}).items():
            sig=set(int(x) for x in (spec.get('signature') or []))
            hits=len(sig & self._seen)
            if hits>=int(spec.get('min_hits',2)):
                return vname,10.0+hits
        scores=[]
        for name,ds in self._deck_sets.items():
            hit=sum(self._card_weight(x) for x in self._seen if x in ds)
            miss=sum(min(1.8,self._card_weight(x))*0.75 for x in self._seen if x not in ds)
            scores.append((hit-miss,name))
        scores.sort(reverse=True)
        if not scores:return None,0.0
        best,name=scores[0]; second=scores[1][0] if len(scores)>1 else -10
        conf=max(0.0,best-second)
        return name,conf

    def _known(self,pl,include_hand=True):
        out=[]
        if include_hand and pl.hand:out += [x.id for x in pl.hand if x]
        out += [x.id for x in pl.discard if x]
        for p in list(pl.active)+list(pl.bench):
            if not p:continue
            out.append(p.id);out += [x.id for x in p.preEvolution]
            out += [x.id for x in p.energyCards];out += [x.id for x in p.tools]
        return out

    def _remaining(self,full,known,n,rot=0):
        c=Counter(full)
        for x in known:
            if c[x]>0:c[x]-=1
        arr=[]
        for k in sorted(c):arr += [k]*c[k]
        if arr:
            rot%=len(arr);arr=arr[rot:]+arr[:rot]
        basics=[x for x in full if (self._card_data(x) and self._card_data(x).basic)]
        fill=(basics[0] if basics else full[-1]) if full else 1
        if len(arr)<n:arr += [fill]*(n-len(arr))
        return arr[:n]

    def _public_seed(self,obs,name,sample_idx,side):
        s=obs.current;me=s.yourIndex;a=s.players[me];b=s.players[1-me]
        def board(pl):
            return tuple((p.id,p.hp,tuple(x.id for x in p.energyCards),tuple(x.id for x in p.tools)) for p in list(pl.active)+list(pl.bench) if p)
        sig=(name,side,s.turn,s.turnActionCount,me,a.deckCount,b.deckCount,len(a.prize),len(b.prize),a.handCount,b.handCount,board(a),board(b),tuple(sorted(self._seen)),sample_idx)
        return int(hashlib.sha1(repr(sig).encode()).hexdigest()[:16],16)

    def _remaining_sample(self,full,known,n,seed):
        c=Counter(full)
        for x in known:
            if c[x]>0:c[x]-=1
        arr=[]
        for k in sorted(c):arr += [k]*c[k]
        rng=random.Random(seed);rng.shuffle(arr)
        basics=[x for x in full if (self._card_data(x) and self._card_data(x).basic)]
        fill=(basics[0] if basics else full[-1]) if full else 1
        if len(arr)<n:arr += [fill]*(n-len(arr))
        return arr[:n]

    def hidden(self,obs,name,sample_idx=0):
        s=obs.current;me=s.yourIndex;a=s.players[me];b=s.players[1-me]
        if sample_idx<0:
            own=self._remaining(self.my_deck,self._known(a,True),a.deckCount+len(a.prize),s.turn*3+me)
        else:
            own=self._remaining_sample(self.my_deck,self._known(a,True),a.deckCount+len(a.prize),self._public_seed(obs,name,sample_idx,'own'))
        yp=own[:len(a.prize)];yd=own[len(a.prize):]
        odfull=self.catalog[name]
        if sample_idx<0:
            if name != 'alakazam':
                raise RuntimeError('single-sample search retained off outside verified Alakazam policy')
            rem=self._remaining(odfull,self._known(b,False),b.deckCount+len(b.prize)+b.handCount,s.turn*5+me)
        else:
            rem=self._remaining_sample(odfull,self._known(b,False),b.deckCount+len(b.prize)+b.handCount,self._public_seed(obs,name,sample_idx,'opp'))
        oh=rem[:b.handCount];op=rem[b.handCount:b.handCount+len(b.prize)];od=rem[b.handCount+len(b.prize):]
        active=[]
        if b.active and b.active[0] is None:
            basics=[x for x in rem if self._card_data(x) and self._card_data(x).basic]
            if basics:active=[basics[0]]
        return yd,yp,od,op,oh,active

    def _source_card(self,obs,o):
        s=obs.current;me=s.yourIndex; pi=o.playerIndex if o.playerIndex is not None else me
        pl=s.players[pi]
        try:
            if o.area is None and o.type==self.api.OptionType.PLAY:return pl.hand[o.index]
            ar=int(o.area) if o.area is not None else -1
            if ar==1 and obs.select.deck:return obs.select.deck[o.index]
            if ar==2 and pl.hand:return pl.hand[o.index]
            if ar==3:return pl.discard[o.index]
            if ar==4:return pl.active[o.index]
            if ar==5:return pl.bench[o.index]
            if ar==7:return s.stadium[o.index]
            if ar==12 and s.looking:return s.looking[o.index]
        except Exception:return None
        return None

    def _target(self,obs,o):
        try:
            pl=obs.current.players[obs.current.yourIndex]
            ar=int(o.inPlayArea) if o.inPlayArea is not None else -1
            if ar==4:return pl.active[o.inPlayIndex]
            if ar==5:return pl.bench[o.inPlayIndex]
        except Exception:pass
        return None

    def _phase(self,turn):
        return 'early' if int(turn)<=3 else 'mid' if int(turn)<=8 else 'late'

    def _replay_residual(self,obs,o,name=None):
        c=self._source_card(obs,o);cid=getattr(c,'id',getattr(o,'cardId',0)) or 0
        typ=int(o.type);ph=self._phase(obs.current.turn if obs.current else 0)
        keys=[f't{typ}']
        if cid:keys.append(f't{typ}:c{cid}')
        target=self._target(obs,o);tcid=getattr(target,'id',0) or 0
        if tcid:keys.append(f't{typ}:target{tcid}')
        aid=int(getattr(o,'attackId',0) or 0)
        if aid:keys.append(f't{typ}:a{aid}')
        ctx=int(obs.select.context) if obs.select is not None else -1
        base_keys=list(keys)
        for k in base_keys:
            if k.startswith(f't{typ}'):
                keys.append(f'ctx{ctx}:'+k)
        def total(table):
            z=(table or {}).get(ph,{})
            return sum(float(z.get(k,0.0)) for k in keys)
        r=total(self.prior.get('global_phase_bias'))*float(self.cfg.get('global_replay_scale',1.0))
        r+=total(self.prior.get('phase_bias'))*float(self.cfg.get('family_replay_scale',2.0))
        r+=total(self.prior.get('choice_phase_bias'))*float(self.cfg.get('choice_replay_scale',0.0))
        mb=(self.prior.get('matchup_phase_bias') or {}).get(name,{})
        r+=total(mb)*float(self.cfg.get('matchup_replay_scale',2.5))
        cb=(self.cfg.get('matchup_action_bias') or {}).get(name,{})
        r+=sum(float(cb.get(k,0.0)) for k in keys)
        # Backward-compatible compact priors.
        r+=float(self.prior.get('card_bias',{}).get(str(cid),0.0))*float(self.cfg.get('replay_scale',3.0))
        r+=float(self.prior.get('type_bias',{}).get(str(typ),0.0))
        return r

    def _option_prior(self,obs,o,name=None):
        T=self.api.OptionType
        base={T.ATTACK:125,T.EVOLVE:113,T.ABILITY:106,T.ATTACH:101,T.RETREAT:78,T.PLAY:72,T.END:5}.get(o.type,20)
        c=self._source_card(obs,o);cid=getattr(c,'id',getattr(o,'cardId',0)) or 0
        cd=self._card_data(cid)
        if cd:
            if cd.cardType==0:base+=10+(12 if cd.stage2 else 6 if cd.stage1 else 0)
            elif cd.cardType==3:base+=7
            elif cd.cardType==1:base+=4
        return base+self._replay_residual(obs,o,name)

    def candidates(self,obs,base,name=None):
        sel=obs.select
        if sel.minCount!=1 or sel.maxCount!=1:return []
        vals=[]
        for i,o in enumerate(sel.option):
            if o.type in {self.api.OptionType.PLAY,self.api.OptionType.ATTACH,self.api.OptionType.EVOLVE,
                          self.api.OptionType.ABILITY,self.api.OptionType.RETREAT,self.api.OptionType.ATTACK,self.api.OptionType.END}:
                vals.append((self._option_prior(obs,o,name),i))
        vals.sort(reverse=True)
        ids=[]
        if base and len(base)==1:ids.append(base[0])
        for _,i in vals:
            if i not in ids:ids.append(i)
            if len(ids)>=int(self._param(name,'candidates',4)):break
        return ids

    def _energy_ready(self,p):
        cd=self._card_data(p.id)
        if not cd:return (0,0)
        have=list(p.energies or [])
        best=0;ready=0
        for aid in cd.attacks:
            at=self._attacks.get(aid)
            if not at:continue
            best=max(best,int(at.damage or 0))
            pool=list(have);ok=True
            for req in at.energies:
                if int(req)==0:continue
                found=None
                for j,x in enumerate(pool):
                    if int(x) in (int(req),10,11) or (int(x)==11 and int(req) in (5,7)):
                        found=j;break
                if found is None:ok=False;break
                pool.pop(found)
            if ok and len(have)>=len(at.energies):ready=max(ready,int(at.damage or 1))
        return ready,best

    def _poke_value(self,p,own=True):
        cd=self._card_data(p.id)
        if not cd:return max(0,p.hp)*20
        w=self.cfg['weights']
        stage=2 if cd.stage2 else 1 if cd.stage1 else 0
        ready,best=self._energy_ready(p)
        energy=len(p.energyCards)
        rem=max(0,p.hp); dmg=max(0,p.maxHp-p.hp)
        ability=sum(1 for x in cd.skills if x.name)
        rule=(3 if cd.megaEx else 2 if cd.ex else 1)
        val=(rem*w['hp'] - dmg*w.get('damage_liability',0) + stage*w['stage'] + energy*w['energy']
             + (1 if ready>0 else 0)*w['ready'] + ready*w['attack'] + best*w.get('potential_attack',0)
             + ability*w.get('ability',0))
        # A fragile multi-prize body is a liability, but an energized ready one remains valuable.
        val -= max(0,rule-1)*w.get('rulebox_liability',0)*(1.0 if rem < p.maxHp*0.55 else 0.35)
        val += float((self.prior.get('state_card_bias') or {}).get(str(p.id),0.0))*float(self.cfg.get('state_prior_scale',0.0))
        return val

    def _learned_side(self,pl):
        board=[p for p in list(pl.active)+list(pl.bench) if p]
        hp=sum(max(0,p.hp) for p in board);maxhp=sum(max(0,p.maxHp) for p in board);damage=maxhp-hp
        energy=sum(len(p.energyCards) for p in board);stage=0;ready=0;ready_dmg=0;potential=0;ability=0;rulebox=0
        for p in board:
            cd=self._card_data(p.id)
            if cd:
                stage += 2 if cd.stage2 else 1 if cd.stage1 else 0
                ability += sum(1 for x in cd.skills if x.name)
                rulebox += 3 if cd.megaEx else 2 if cd.ex else 0
            r,b=self._energy_ready(p);ready += int(r>0);ready_dmg=max(ready_dmg,r);potential=max(potential,b)
        active=pl.active[0] if pl.active and pl.active[0] else None
        ar,ab=self._energy_ready(active) if active else (0,0)
        return [len(pl.prize)/6.0,pl.handCount/12.0,pl.deckCount/60.0,len(pl.bench)/5.0,
                hp/1800.0,damage/1800.0,energy/12.0,stage/10.0,ready/5.0,ready_dmg/350.0,
                potential/350.0,ability/8.0,rulebox/12.0,(active.hp/400.0 if active else 0.0),
                ar/350.0,ab/350.0,float(pl.poisoned),float(pl.burned),float(pl.asleep),float(pl.paralyzed),float(pl.confused)]

    def _model_logit(self,model,x):
        if not model:return 0.0
        coef=model.get('coef') or []
        if len(coef)!=len(x):return 0.0
        z=float(model.get('intercept',0.0))+sum(float(c)*float(v) for c,v in zip(coef,x))
        return max(-12.0,min(12.0,z))

    def _learned_logits(self,s,me):
        a=s.players[me];b=s.players[1-me];af=self._learned_side(a);bf=self._learned_side(b)
        x=[s.turn/30.0,s.turnActionCount/25.0,float(s.firstPlayer==me),float(s.supporterPlayed),float(s.energyAttached),float(s.retreated)]
        x += [u-v for u,v in zip(af,bf)]+af+bf
        return self._model_logit(self.cfg.get('value_model'),x),self._model_logit(self.cfg.get('global_value_model'),x)

    def evaluate(self,obs,me,root,name=None,root_turn=0):
        s=obs.current
        if s is None:return -10**15
        if s.result>=0:return 10**15 if s.result==me else -10**15
        native=set(self.cfg.get('native_value_matchups') or [])
        if name in native and hasattr(self.base,'_eval'):
            try:return float(self.base._eval(obs,root_turn,me,name))
            except Exception:pass
        a=s.players[me];b=s.players[1-me];w=self.cfg['weights']
        pa=len(a.prize);pb=len(b.prize)
        val=(pb-pa)*w['prize']
        val += (a.handCount-b.handCount)*w['hand']
        val += (a.deckCount-b.deckCount)*w.get('deck',0)
        aval=sum(self._poke_value(p,True) for p in list(a.active)+list(a.bench) if p)
        bval=sum(self._poke_value(p,False) for p in list(b.active)+list(b.bench) if p)
        val += aval - bval*w.get('opp_board_scale',0.82)
        ms=(self.cfg.get('matchup_state_bias') or {}).get(name,{})
        if ms:
            val += sum(float(ms.get(str(p.id),ms.get(p.id,0.0))) for p in list(a.active)+list(a.bench) if p)
            om=(self.cfg.get('matchup_opponent_state_bias') or {}).get(name,{})
            val -= sum(float(om.get(str(p.id),om.get(p.id,0.0))) for p in list(b.active)+list(b.bench) if p)
        val += (len([p for p in a.bench if p])-len([p for p in b.bench if p]))*w['bench']
        # Reward concrete progress made during the searched turn.
        ra=root['prize_me']-pa; rb=root['prize_op']-pb
        val += ra*w.get('prize_progress',w['prize']*0.8)-rb*w.get('prize_progress',w['prize']*0.8)
        opp_damage=sum(max(0,p.maxHp-p.hp) for p in list(b.active)+list(b.bench) if p)
        my_damage=sum(max(0,p.maxHp-p.hp) for p in list(a.active)+list(a.bench) if p)
        val += (opp_damage-root['opp_damage'])*w.get('damage_progress',0)
        val -= (my_damage-root['my_damage'])*w.get('damage_progress',0)
        dz,gz=self._learned_logits(s,me)
        val += dz*float(self.cfg.get('learned_scale',0.0))
        val += gz*float(self.cfg.get('global_learned_scale',0.0))
        return val

    def _simulate_opponent(self,st,sid,me,name,max_actions):
        if not self.opponent_bank or not self.opponent_bank.available(name):
            return st,sid
        handle=self.opponent_bank.get(name)
        if handle is None:return st,sid
        snap=handle.snapshot();self.stats['opponent_turns']+=1
        try:
            for _ in range(max_actions):
                o=st.observation
                if o.current is None or o.current.result>=0 or o.current.yourIndex==me or o.select is None:break
                before_errors=handle.errors
                act=handle.act(asdict(o))
                if handle.errors>before_errors:self.stats['opponent_errors']+=1
                act=self.validate(act,o.select)
                if not act:act=self._fallback_sel(o.select)
                if not act:break
                st2=self.api.search_step(sid,act)
                try:self.api.search_release(sid)
                except Exception:pass
                st=st2;sid=st.searchId;self.stats['opponent_actions']+=1
            return st,sid
        finally:handle.restore(snap)

    def _simulate(self,root_state,first_idx,obs,turn,me,root_info,max_actions,name):
        st=self.api.search_step(root_state.searchId,[first_idx]);sid=st.searchId
        try:
            for _ in range(max_actions):
                o=st.observation
                if o.current is None or o.current.result>=0 or o.current.turn!=turn or o.current.yourIndex!=me or o.select is None:break
                d=asdict(o)
                act=self.call_base(d,turn)
                act=self.validate(act,o.select)
                if not act:break
                st2=self.api.search_step(sid,act)
                try:self.api.search_release(sid)
                except Exception:pass
                st=st2;sid=st.searchId
            pre_value=self.evaluate(st.observation,me,root_info,name,turn)
            if bool(self._param(name,'double_agent_enabled',self.cfg.get('double_agent_enabled',False))):
                st,sid=self._simulate_opponent(st,sid,me,name,int(self._param(name,'opponent_max_actions',7)))
                post_value=self.evaluate(st.observation,me,root_info,name,turn)
                response_weight=float(self._param(name,'opponent_response_weight',self.cfg.get('opponent_response_weight',0.65)))
                response_weight=max(0.0,min(1.0,response_weight))
                return pre_value*(1.0-response_weight)+post_value*response_weight
            return pre_value
        finally:
            try:self.api.search_release(sid)
            except Exception:pass

    def search(self,obs,base,name,conf):
        s=obs.current
        if obs.select.context!=self.api.SelectContext.MAIN:return base
        if s.turn>int(self._param(name,'max_turn',8)) or s.turn==self._turn_searched:return base
        if conf<float(self._param(name,'min_confidence',0.05)):return base
        inds=self.candidates(obs,base,name)
        if len(inds)<2:return base
        self.stats['searches']+=1
        a=s.players[s.yourIndex];b=s.players[1-s.yourIndex]
        root_info={'prize_me':len(a.prize),'prize_op':len(b.prize),
                   'my_damage':sum(max(0,p.maxHp-p.hp) for p in list(a.active)+list(a.bench) if p),
                   'opp_damage':sum(max(0,p.maxHp-p.hp) for p in list(b.active)+list(b.bench) if p)}
        snap=self.snapshot();sample_vals={i:[] for i in inds}
        samples=max(1,int(self._param(name,'hidden_samples',1)))
        # Low-confidence recognition should not spend extra compute on a dubious archetype.
        if conf<float(self._param(name,'multi_sample_min_confidence',0.12)):samples=1
        try:
            for sample_idx in range(samples):
                # Preserve the original single-determinization behavior outside
                # explicitly probabilistic matchups.
                h=self.hidden(obs,name,sample_idx if samples>1 else -1)
                root=self.api.search_begin(obs,*h)
                try:
                    for i in inds:
                        self.restore(deepcopy(snap))
                        try:
                            v=self._simulate(root,i,obs,s.turn,s.yourIndex,root_info,int(self._param(name,'max_actions',10)),name)
                            v+=self._replay_residual(obs,obs.select.option[i],name)*float(self._param(name,'replay_value_scale',0.0))
                            sample_vals[i].append(v)
                        except Exception:continue
                finally:
                    try:self.api.search_end()
                    except Exception:pass
        except Exception:
            try:self.api.search_end()
            except Exception:pass
            return base
        finally:self.restore(snap)
        self._turn_searched=s.turn
        bi=base[0] if base and len(base)==1 else None
        if bi not in sample_vals or not sample_vals[bi]:return base
        risk=max(0.0,float(self._param(name,'uncertainty_risk',0.0)))
        tail=max(0.0,min(1.0,float(self._param(name,'downside_weight',0.0))))
        agg={};detail={}
        for i,vs in sample_vals.items():
            if not vs:continue
            mean=sum(vs)/len(vs)
            var=sum((x-mean)**2 for x in vs)/len(vs)
            sd=math.sqrt(max(0.0,var));worst=min(vs)
            agg[i]=(1.0-tail)*(mean-risk*sd)+tail*worst
            detail[i]=(mean,sd,worst,len(vs))
        if bi not in agg:return base
        best=max(agg,key=agg.get)
        margin=float(self._param(name,'override_margin',100000))
        paired=min(len(sample_vals.get(best,[])),len(sample_vals.get(bi,[])))
        wins=sum(1 for x,y in zip(sample_vals.get(best,[])[:paired],sample_vals.get(bi,[])[:paired]) if x>=y+margin*float(self._param(name,'sample_margin_fraction',0.25)))
        win_prob=wins/max(1,paired)
        req=float(self._param(name,'override_probability',0.0 if samples==1 else 0.66))
        if best!=bi and agg[best]>=agg[bi]+margin and win_prob+1e-12>=req:
            self.stats['overrides']+=1
            return [best]
        return base

    def loop_break(self,obs,act):
        if obs.select.context!=self.api.SelectContext.MAIN:
            self._loop_sig=None;self._loop_count=0;return act
        s=obs.current;a=s.players[s.yourIndex]
        opts=tuple((int(o.type),o.index,o.inPlayIndex,o.attackId,o.cardId,o.serial) for o in obs.select.option)
        field=tuple((p.id,p.hp,tuple(x.id for x in p.energyCards)) for p in list(a.active)+list(a.bench) if p)
        sig=(s.turn,s.turnActionCount,a.handCount,a.deckCount,len(a.prize),field,opts,tuple(act))
        if sig==self._loop_sig:self._loop_count+=1
        else:self._loop_sig=sig;self._loop_count=1
        if self._loop_count>=3:
            for i,o in enumerate(obs.select.option):
                if o.type==self.api.OptionType.END:
                    self._loop_count=0;return [i]
        return act

    def agent(self,d):
        if d.get('select') is None:
            self.reset()
            try:self.call_base(d,0,disable_search=False)
            except Exception:pass
            return list(self.my_deck)
        try:
            obs=self.api.to_observation_class(d)
            if obs.current is not None and obs.current.turn==0:self.reset()
            turn=obs.current.turn if obs.current else 0
            name,conf=self.recognize(obs)
            allowed=set(self.cfg.get('search_matchups') or [])
            da_allowed=set(self.cfg.get('double_agent_matchups') or [])
            eligible=bool(self.cfg.get('enabled',True) and name and ((not allowed and not da_allowed) or name in allowed or name in da_allowed))
            preserve_root=bool(self.cfg.get('preserve_base_search_root',False))
            # Hybrid mode: let the deck's own/native search choose the root proposal once.
            # Native search remains disabled inside hypothetical continuations to prevent
            # recursive search_begin calls and exponential branch growth.
            native_root = bool(eligible and preserve_root and obs.select.context==self.api.SelectContext.MAIN and turn!=self._native_root_turn)
            if native_root:
                self.stats['native_root_calls']+=1
                self._native_root_turn=turn
            # Run deck-native search at most once per real turn. Later decisions in the
            # same turn use the native policy without recursively reopening its search.
            base=self.validate(self.call_base(d,turn,disable_search=(eligible and not native_root)),obs.select)
            if name:self.stats['recognized'][name]+=1
            if eligible:
                self.stats['hybrid_searches']+=1
                base=self.search(obs,base,name,conf)
            return self.loop_break(obs,self.validate(base,obs.select))
        except Exception:
            return self.fallback(d)
