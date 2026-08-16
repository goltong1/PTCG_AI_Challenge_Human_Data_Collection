"""v170 mistake-elimination layer learned from same-hidden-state counterfactual replay audits.

Terminal wins/losses are never used as mistake labels. Known replay mistakes are exact-state
memories; unseen states use the counterfactual quality critic only to nominate candidates.
Every action change is re-proved in the current CABT hidden state by exact branch search.
"""
from __future__ import annotations
import os,json,hashlib,time,math
MAIN=0

def _i(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d

def _phase(t):return 'early' if _i(t)<=3 else 'mid' if _i(t)<=7 else 'late'
def _coarse(d):return f"{_i(d.get('type'),-1)}:{_i(d.get('cardId'))}:{_i(d.get('attackId'))}"
def _sigmoid(x):
    x=max(-30.,min(30.,float(x)));return 1.0/(1.0+math.exp(-x))

def _normfam(f):
    f=str(f or 'unknown').lower()
    if f.startswith('drag'):return 'dragapult'
    if f.startswith('lucario'):return 'lucario'
    if f.startswith('nzoroark') or f=='n_zoroark':return 'n_zoroark'
    if f.startswith('clefairy'):return 'clefairy'
    if f in ('grass5','hydrapple'):return 'grass'
    return f

class MistakeEliminator:
    def __init__(self,root,history,replay_mod,tok_mod,quality_model):
        self.root=root;self.history=history;self.replay=replay_mod;self.tok=tok_mod;self.quality=quality_model;self.verifier=None;self.load_error=None
        try:self.cfg=json.load(open(os.path.join(root,'mistake_eliminator_model.json'),encoding='utf-8'))
        except Exception as e:self.cfg={};self.load_error=repr(e)
        self.gate=self.cfg.get('gate') or {};self.exact=self.cfg.get('exact_mistake_memory') or {};self.priors=self.cfg.get('mistake_priors') or {};self.enabled=bool(self.cfg and self.quality is not None);self.reset()
    def reset(self):
        self.overrides=0;self.search_calls=0;self.last_turn=-1;self.stats={'calls':0,'known_exact_hits':0,'known_exact_candidate_missing':0,'learned_suspects':0,'quality_scored':0,'search_calls':0,'search_accept':0,'search_reject':0,'overrides':0,'errors':0,'classes':{},'reasons':{},'decisions':[]}
    def _note(self,k):self.stats['reasons'][k]=self.stats['reasons'].get(k,0)+1
    def _family(self,obs):
        # Detect the Crustle axis from public IDs before generic recognition so the
        # learned mistake search can never interfere with the immutable Drill route.
        try:
            cur=obs.get('current') or {};me=_i(cur.get('yourIndex'));ps=cur.get('players') or [];ids=set()
            if len(ps)>=2:
                op=ps[1-me]
                for z in (op.get('active') or [])+(op.get('bench') or [])+(op.get('discard') or []):
                    if z:ids.add(_i(z.get('id')))
            for e in obs.get('logs') or []:
                if _i(e.get('playerIndex'),me)==1-me:
                    for k in ('cardId','cardIdAfter','cardIdBefore','cardIdActive','cardIdBench','cardIdTarget'):
                        if e.get(k) is not None:ids.add(_i(e.get(k)))
            if ids & {344,345,756}:return 'crustle'
        except Exception:pass
        try:return _normfam(self.replay.recognize(self.history,obs))
        except Exception:return 'unknown'
    def _descs(self,obs):
        try:return self.quality._descs(obs)
        except Exception:return []
    def _state_key(self,obs,fam,descs):
        ctx=self.tok.context_tokens(self.history,obs,fam);intent=self.tok.intent_context_tokens(ctx,112);acts=[self.tok.action_signature(d) for d in descs]
        raw=json.dumps([intent,acts],ensure_ascii=False,separators=(',',':')).encode('utf-8');return hashlib.sha1(raw).hexdigest()
    def _prior(self,fam,turn,d):return self.priors.get(f'{fam}|{_phase(turn)}|{_coarse(d)}')
    def _search(self,obs,base,ci,min_gain,rec):
        if not callable(self.verifier):rec['reason']='no_verifier';return None
        if self.search_calls>=_i(self.gate.get('max_game_search_calls'),3):rec['reason']='search_quota';return None
        self.search_calls+=1;self.stats['search_calls']+=1
        try:
            out=self.verifier(obs,base,[ci]);ok,meta=(bool(out[0]),out[1]) if isinstance(out,tuple) else (bool(out),{});gain=float((meta or {}).get('gain',-1e9));rec.setdefault('searches',[]).append({'candidate':ci,'gain':round(gain,3),'ok':ok})
            if ok and gain>=float(min_gain):self.stats['search_accept']+=1;return [ci]
            self.stats['search_reject']+=1;return None
        except Exception as e:self.stats['errors']+=1;rec.setdefault('search_errors',[]).append(repr(e)[:120]);return None
    def choose(self,obs,base):
        if not self.enabled or not isinstance(base,list) or len(base)!=1:return base
        sel=obs.get('select') or {};opts=sel.get('option') or [];cur=obs.get('current') or {};turn=_i(cur.get('turn'));self.stats['calls']+=1
        if _i(sel.get('context'),-1)!=MAIN or _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or len(opts)<2 or len(opts)>_i(self.gate.get('max_options'),24):return base
        bi=_i(base[0],-1)
        if not 0<=bi<len(opts) or self.overrides>=_i(self.gate.get('max_game_overrides'),2) or turn==self.last_turn:return base
        try:
            fam=self._family(obs);descs=self._descs(obs)
            if len(descs)!=len(opts):return base
            bsig=self.tok.action_signature(descs[bi]);rec={'family':fam,'turn':turn,'base':bi,'base_sig':bsig}
            # 1) Exact replay-mistake memory. This does not trust terminal outcome: every
            # remembered entry came from same-hidden-state branch regret and is rechecked now.
            key=self._state_key(obs,fam,descs);mem=self.exact.get(key)
            if mem and mem.get('bad')==bsig:
                self.stats['known_exact_hits']+=1;rec.update({'mode':'exact','severity':mem.get('severity'),'mistake_class':mem.get('mistake_class'),'historical_regret':mem.get('regret')});self.stats['classes'][mem.get('mistake_class','other')]=self.stats['classes'].get(mem.get('mistake_class','other'),0)+1
                sig2i={self.tok.action_signature(d):i for i,d in enumerate(descs)};cands=[sig2i[s] for s in (mem.get('good') or []) if s in sig2i and sig2i[s]!=bi]
                if not cands:self.stats['known_exact_candidate_missing']+=1;rec['reason']='exact_candidate_missing';self.stats['decisions'].append(rec);return base
                for ci in cands[:2]:
                    got=self._search(obs,base,ci,self.gate.get('exact_search_min_gain',10),rec)
                    if got is not None:
                        self.overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;rec['reason']='exact_mistake_repaired';self.stats['decisions'].append(rec);return got
                rec['reason']='exact_search_reject';self.stats['decisions'].append(rec);return base
            # 2) Unseen state. Crustle remains governed by immutable text/Drill safety layers.
            if fam=='crustle' and not bool(self.gate.get('allow_crustle_learned',False)):return base
            inds=list(range(len(descs)));p,q,o=self.quality._score(obs,inds,descs,fam);self.stats['quality_scored']+=1;score=.25*p+q;order=sorted((i for i in inds if i!=bi),key=lambda i:(float(score[i]),float(q[i]),float(p[i])),reverse=True)
            if not order:return base
            ci=order[0];sm=float(score[ci]-score[bi]);qm=float(q[ci]-q[bi]);prior=self._prior(fam,turn,descs[bi]);rec.update({'mode':'learned','candidate':ci,'score_margin':round(sm,4),'quality_margin':round(qm,4),'prior':prior})
            supported=bool(prior and _i(prior.get('samples'))>=_i(self.gate.get('prior_min_samples'),2) and float(prior.get('mistake_rate',0))>=float(self.gate.get('prior_min_mistake_rate',.55)) and float(prior.get('mean_mistake_regret',0))>=float(self.gate.get('prior_min_mean_regret',50)))
            strong=sm>=max(.42,float(self.gate.get('min_score_margin',.12))*2.5) and qm>=max(.22,float(self.gate.get('min_quality_margin',.08))*2.0)
            if not (supported or strong):self._note('not_suspect');return base
            if sm<float(self.gate.get('min_score_margin',.12)) or qm<float(self.gate.get('min_quality_margin',.08)):self._note('low_margin');return base
            self.stats['learned_suspects']+=1
            for alt in order[:2]:
                got=self._search(obs,base,alt,self.gate.get('learned_search_min_gain',35),rec)
                if got is not None:
                    self.overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;rec['candidate']=alt;rec['reason']='learned_mistake_repaired';self.stats['decisions'].append(rec);return got
            rec['reason']='learned_search_reject';self.stats['decisions'].append(rec);return base
        except Exception as e:
            self.stats['errors']+=1;self.stats['decisions'].append({'reason':'error','error':repr(e)[:160]});return base
    def get_stats(self):
        out=dict(self.stats);out.update({'enabled':self.enabled,'version':self.cfg.get('version'),'known_mistakes':len(self.exact),'game_overrides':self.overrides,'game_search_calls':self.search_calls,'load_error':self.load_error});return out
