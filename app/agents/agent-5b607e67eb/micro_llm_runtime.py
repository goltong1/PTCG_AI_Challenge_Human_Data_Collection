"""Hybrid recent-preserving domain-specific decoder policy for Lucario v157.

This is a medium four-layer, chunk-compressed causal Transformer trained on
complete public action prefixes. It is not an external API client and requires no
network access. The validated v152 tactical executor remains the anchor. In veto
release mode the decoder may only restore that exact anchor over a learned residual;
it cannot invent a third action. In shadow release mode it scores and logs the same
comparison without changing play. Independent support, exact search, the strategy
arbiter, and the temporal safety gate remain authoritative.
"""
from __future__ import annotations
import json,math,os
try:
    import numpy as np
except Exception:
    np=None

MAIN=0;PLAY=7;ATTACK=13
BLACK_BELT=1211

def _int(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d

def _sigmoid(x):
    x=max(-30.0,min(30.0,float(x)));return 1.0/(1.0+math.exp(-x))

def _turnbin(t):return 'early' if int(t)<=3 else 'mid' if int(t)<=7 else 'late'

def _coarse(desc):
    return f"{_int(desc.get('type'),-1)}:{_int(desc.get('cardId'))}:{_int(desc.get('targetId'))}:{_int(desc.get('attackId'))}"

class MicroDecoderPolicy:
    def __init__(self,root,history,replay_module,token_module,config_file='micro_llm_model.json',weights_file='micro_llm_model.npz'):
        self.root=root;self.history=history;self.replay=replay_module;self.tokens=token_module;self.verifier=None;self.load_error=None
        try:self.config=json.load(open(os.path.join(root,config_file),encoding='utf-8'))
        except Exception as e:self.config={'enabled':False,'gate':{}};self.load_error=f'config:{type(e).__name__}'
        self.gate=self.config.get('gate') or {};self.weights={}
        if np is not None and self.config.get('enabled',False):
            try:
                z=np.load(os.path.join(root,weights_file),allow_pickle=False)
                self.weights={k:z[k] for k in z.files};z.close()
            except Exception as e:self.load_error=f'weights:{type(e).__name__}';self.weights={}
        elif np is None:self.load_error='numpy_unavailable'
        self.enabled=bool(self.config.get('enabled',False) and self.weights and np is not None)
        self.vocab=max(8,_int(self.config.get('vocab_size'),4096));self.max_context=max(8,_int(self.config.get('max_context'),96));self.max_action=max(4,_int(self.config.get('max_action'),12))
        self.layers=max(1,_int(self.config.get('layers'),2));self.heads=max(1,_int(self.config.get('heads'),3));self.d=max(1,_int(self.config.get('d_model'),24));self.architecture=str(self.config.get('architecture','causal_decoder'));self.chunk_size=max(1,_int(self.config.get('context_chunk_size'),4));self.recent_context=max(0,min(self.max_context,_int(self.config.get('recent_context_tokens'),32)));self.max_len=max(self.max_context+self.max_action,_int(self.config.get('max_len'),self.max_context+self.max_action));self.action_authority=bool(self.config.get('action_authority',_int(self.gate.get('max_game_vetoes'),1)>0));self.reset()

    def reset(self):
        self.last_turn=-1;self.shadow_last_turn=-1;self.game_overrides=0;self.game_vetoes=0
        self.stats={'calls':0,'scored':0,'overrides':0,'suggestions':0,'family_block':0,'context_block':0,'type_block':0,'support_block':0,'risk_block':0,'margin_block':0,'policy_block':0,'errors':0,'latency_ms_sum':0.0,'search_calls':0,'search_accept':0,'search_reject':0,'search_errors':0,'search_latency_ms_sum':0.0,'veto_calls':0,'veto_suggestions':0,'vetoes':0,'veto_type_block':0,'veto_support_block':0,'veto_margin_block':0,'veto_search_reject':0,'veto_errors':0,'veto_latency_ms_sum':0.0,'veto_keys':{},'veto_decisions':[],'keys':{},'suggestion_keys':{},'decisions':[]}
        self.last={'enabled':self.enabled,'action_authority':self.action_authority,'family':'unknown','base':[],'candidate':[]}

    @staticmethod
    def _ln(x,w,b,eps=1e-5):
        mean=x.mean(axis=-1,keepdims=True);var=((x-mean)*(x-mean)).mean(axis=-1,keepdims=True)
        return (x-mean)/np.sqrt(var+eps)*w+b

    @staticmethod
    def _gelu(x):
        # PyTorch-compatible tanh approximation; the difference from exact GELU is
        # far below the conservative action margins used by the gate.
        return 0.5*x*(1.0+np.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*x*x*x)))

    def _forward(self,sequences):
        n=len(sequences);max_len=max(len(s) for s in sequences)
        ids=np.zeros((n,max_len),dtype=np.int64);lengths=np.empty(n,dtype=np.int64)
        for i,s in enumerate(sequences):ids[i,:len(s)]=s;lengths[i]=len(s)
        x=self.weights['emb.weight'][ids]+self.weights['pos.weight'][np.arange(max_len)][None,:,:]
        pad=(ids==0);causal=np.triu(np.ones((max_len,max_len),dtype=bool),1)
        dh=self.d//self.heads
        for li in range(self.layers):
            p=f'blocks.{li}.'
            h=self._ln(x,self.weights[p+'ln1.weight'],self.weights[p+'ln1.bias'])
            q=h@self.weights[p+'q.weight'].T+self.weights[p+'q.bias'];k=h@self.weights[p+'k.weight'].T+self.weights[p+'k.bias'];v=h@self.weights[p+'v.weight'].T+self.weights[p+'v.bias']
            q=q.reshape(n,max_len,self.heads,dh).transpose(0,2,1,3);k=k.reshape(n,max_len,self.heads,dh).transpose(0,2,1,3);v=v.reshape(n,max_len,self.heads,dh).transpose(0,2,1,3)
            score=np.matmul(q,k.transpose(0,1,3,2))/math.sqrt(dh)
            score=np.where(causal[None,None,:,:],-1e4,score);score=np.where(pad[:,None,None,:],-1e4,score)
            score=score-score.max(axis=-1,keepdims=True);att=np.exp(score);att=att/np.maximum(att.sum(axis=-1,keepdims=True),1e-12)
            y=np.matmul(att,v).transpose(0,2,1,3).reshape(n,max_len,self.d)
            x=x+y@self.weights[p+'o.weight'].T+self.weights[p+'o.bias']
            h=self._ln(x,self.weights[p+'ln2.weight'],self.weights[p+'ln2.bias'])
            ff=self._gelu(h@self.weights[p+'fc1.weight'].T+self.weights[p+'fc1.bias'])
            x=x+ff@self.weights[p+'fc2.weight'].T+self.weights[p+'fc2.bias']
        x=self._ln(x,self.weights['final_ln.weight'],self.weights['final_ln.bias'])
        last=x[np.arange(n),lengths-1]
        policy=(last@self.weights['policy.weight'].T+self.weights['policy.bias']).reshape(-1)
        value=(last@self.weights['value.weight'].T+self.weights['value.bias']).reshape(-1)
        return policy,value

    def _forward_hybrid(self,context_ids,action_sequences):
        """Pool older context 4:1 while preserving the newest tokens exactly.

        v154 averaged every eight tokens.  v157 keeps the newest configured recent tokens and
        pools only the older prefix in groups of four.  Card-text semantics are
        appended at the end of the context, so they always remain uncompressed.
        """
        n=len(action_sequences);c_len=len(context_ids);a_len=max(len(s) for s in action_sequences)
        if n<=0 or c_len<=0 or a_len<=0:raise ValueError('empty hybrid batch')
        clen=min(c_len,self.max_context);ids=np.asarray(context_ids[-clen:],dtype=np.int64)
        recent_len=min(clen,self.recent_context);old_len=clen-recent_len
        old_cap=max(0,self.max_context-self.recent_context)
        chunks=(old_cap+self.chunk_size-1)//self.chunk_size if old_cap else 0
        target=chunks*self.chunk_size
        if chunks:
            ctx=np.zeros((target,self.d),dtype=np.float32);valid=np.zeros(target,dtype=bool)
            if old_len:
                ctx[:old_len]=self.weights['emb.weight'][ids[:old_len]]+self.weights['pos.weight'][np.arange(old_len)]
                valid[:old_len]=True
            cr=ctx.reshape(chunks,self.chunk_size,self.d);cm=valid.reshape(chunks,self.chunk_size);counts=cm.sum(axis=1)
            pooled=cr.sum(axis=1)/np.maximum(counts[:,None],1)
            pooled=self._ln(pooled,self.weights['chunk_ln.weight'],self.weights['chunk_ln.bias'])+self.weights['segment.weight'][0]
            chunk_valid=counts>0
        else:
            pooled=np.zeros((0,self.d),dtype=np.float32);chunk_valid=np.zeros(0,dtype=bool)
        if recent_len:
            rids=ids[old_len:]
            rpos=np.arange(old_len,clen)
            recent=self.weights['emb.weight'][rids]+self.weights['pos.weight'][rpos]
            recent=self._ln(recent,self.weights['chunk_ln.weight'],self.weights['chunk_ln.bias'])+self.weights['segment.weight'][0]
            recent_valid=np.ones(recent_len,dtype=bool)
        else:
            recent=np.zeros((0,self.d),dtype=np.float32);recent_valid=np.zeros(0,dtype=bool)
        context=np.concatenate((pooled,recent),axis=0)
        context_valid=np.concatenate((chunk_valid,recent_valid),axis=0)

        aids=np.zeros((n,a_len),dtype=np.int64);lengths=np.empty(n,dtype=np.int64)
        for i,a in enumerate(action_sequences):aids[i,:len(a)]=a;lengths[i]=len(a)
        apos=np.clip(clen+np.arange(a_len),0,self.max_len-1)
        action=self.weights['emb.weight'][aids]+self.weights['pos.weight'][apos][None,:,:]+self.weights['segment.weight'][1]
        x=np.concatenate((np.broadcast_to(context[None,:,:],(n,len(context),self.d)),action),axis=1)
        key_valid=np.concatenate((np.broadcast_to(context_valid[None,:],(n,len(context))),aids!=0),axis=1)
        total=x.shape[1];causal=np.triu(np.ones((total,total),dtype=bool),1);dh=self.d//self.heads
        for li in range(self.layers):
            p=f'blocks.{li}.';h=self._ln(x,self.weights[p+'ln1.weight'],self.weights[p+'ln1.bias'])
            q=h@self.weights[p+'q.weight'].T+self.weights[p+'q.bias'];k=h@self.weights[p+'k.weight'].T+self.weights[p+'k.bias'];v=h@self.weights[p+'v.weight'].T+self.weights[p+'v.bias']
            q=q.reshape(n,total,self.heads,dh).transpose(0,2,1,3);k=k.reshape(n,total,self.heads,dh).transpose(0,2,1,3);v=v.reshape(n,total,self.heads,dh).transpose(0,2,1,3)
            score=np.matmul(q,k.transpose(0,1,3,2))/math.sqrt(dh)
            score=np.where(causal[None,None,:,:],-1e4,score);score=np.where((~key_valid)[:,None,None,:],-1e4,score)
            score=score-score.max(axis=-1,keepdims=True);att=np.exp(score);att=att/np.maximum(att.sum(axis=-1,keepdims=True),1e-12)
            y=np.matmul(att,v).transpose(0,2,1,3).reshape(n,total,self.d)
            x=x+y@self.weights[p+'o.weight'].T+self.weights[p+'o.bias']
            h=self._ln(x,self.weights[p+'ln2.weight'],self.weights[p+'ln2.bias']);ff=self._gelu(h@self.weights[p+'fc1.weight'].T+self.weights[p+'fc1.bias'])
            x=x+ff@self.weights[p+'fc2.weight'].T+self.weights[p+'fc2.bias']
        x=self._ln(x,self.weights['final_ln.weight'],self.weights['final_ln.bias']);last=x[np.arange(n),len(context)+lengths-1]
        policy=(last@self.weights['policy.weight'].T+self.weights['policy.bias']).reshape(-1);value=(last@self.weights['value.weight'].T+self.weights['value.bias']).reshape(-1)
        return policy,value

    def _score_actions(self,context_ids,action_sequences):
        if self.architecture in {'chunked_causal_decoder','hybrid_recent_chunked_causal_decoder'}:return self._forward_hybrid(context_ids,action_sequences)
        return self._forward([list(context_ids)+list(a) for a in action_sequences])

    def _support(self,family,turn,desc):
        return (self.config.get('support') or {}).get(f'{family}|{_turnbin(turn)}|{_coarse(desc)}')

    def _descs(self,obs):
        opts=((obs.get('select') or {}).get('option') or []);out=[]
        for i in range(len(opts)):
            d=self.replay.action_desc(self.history,obs,[i]);d=d[0] if d else {'index':i,'type':-1}
            out.append(self.tokens.enrich_desc(obs,i,d))
        return out

    def choose(self,obs,base):
        if str(self.gate.get('mode','challenger'))=='veto_only':return base
        if not self.enabled or not isinstance(obs,dict) or not isinstance(base,list):return base
        sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
        if _int(sel.get('context'),-1)!=MAIN or _int(sel.get('minCount'))!=1 or _int(sel.get('maxCount'))!=1 or len(base)!=1 or not opts or len(opts)>_int(self.gate.get('max_options'),24):
            self.stats['context_block']+=1;return base
        bi=_int(base[0],-1);turn=_int(cur.get('turn'))
        # The recent-preserving observer is intentionally sampled once per game turn.
        # Subsequent sub-decisions in the same turn remain governed by the fast
        # tactical stack and the deterministic card-text reasoner.
        if str(self.gate.get('mode','challenger'))=='shadow' and turn==self.shadow_last_turn:return base
        self.stats['calls']+=1
        if not 0<=bi<len(opts):return base
        family=self.replay.recognize(self.history,obs);allowed=set(self.gate.get('allowed_families') or [])
        if allowed and family not in allowed:self.stats['family_block']+=1;return base
        try:
            import time;t0=time.perf_counter()
            descs=self._descs(obs);context=self.tokens.context_tokens(self.history,obs,family);context_ids=self.tokens.encode(context,self.vocab,self.max_context,True)
            actions=[self.tokens.encode(self.tokens.action_tokens(d),self.vocab,self.max_action,False) for d in descs]
            policy,value=self._score_actions(context_ids,actions);self.stats['scored']+=1;self.stats['latency_ms_sum']+=(time.perf_counter()-t0)*1000.0
            if str(self.gate.get('mode','challenger'))=='shadow':self.shadow_last_turn=turn
        except Exception:
            self.stats['errors']+=1;return base
        bdesc=descs[bi];btype=_int(bdesc.get('type'),-1);allowed_types={_int(x) for x in (self.gate.get('allowed_types') or [PLAY])}
        if btype not in allowed_types:self.stats['type_block']+=1;return base
        alpha=float(self.gate.get('alpha',.35));combined=policy+alpha*value;candidates=[]
        for i,d in enumerate(descs):
            if i==bi or _int(d.get('type'),-1)!=btype or self.tokens.action_signature(d)==self.tokens.action_signature(bdesc):continue
            if _int(d.get('cardId'))==BLACK_BELT and not any(_int(x.get('type'),-1)==ATTACK for x in opts):continue
            candidates.append(i)
        if not candidates:self.stats['type_block']+=1;return base
        ci=max(candidates,key=lambda i:(float(combined[i]),float(value[i]),float(policy[i]),-i));cdesc=descs[ci]
        self.last={'enabled':True,'action_authority':self.action_authority,'family':family,'turn':turn,'base':list(base),'candidate':[ci],
                   'base_desc':bdesc,'candidate_desc':cdesc,'policy_delta':round(float(policy[ci]-policy[bi]),6),'value_margin':round(float(value[ci]-value[bi]),6),
                   'base_value_prob':round(_sigmoid(value[bi]),6),'candidate_value_prob':round(_sigmoid(value[ci]),6),'combined_margin':round(float(combined[ci]-combined[bi]),6)}
        if ci==bi or combined[ci]<=combined[bi]:return base
        self.stats['suggestions']+=1
        pdelta=float(policy[ci]-policy[bi]);vm=float(value[ci]-value[bi]);bp=_sigmoid(value[bi]);cp=_sigmoid(value[ci])
        cs=self._support(family,turn,cdesc);bs=self._support(family,turn,bdesc)
        key=f'{family}:{_coarse(bdesc)}>{_coarse(cdesc)}';self.stats['suggestion_keys'][key]=self.stats['suggestion_keys'].get(key,0)+1
        cw=_int((cs or {}).get('win_games'));cl=_int((cs or {}).get('loss_games'));crate=(cw+1.0)/(cw+cl+2.0) if cs else None
        bw=_int((bs or {}).get('win_games'));bl=_int((bs or {}).get('loss_games'));brate=(bw+1.0)/(bw+bl+2.0) if bs else None
        rec={'turn':turn,'key':key,'policy_delta':round(pdelta,6),'value_margin':round(vm,6),'base_prob':round(bp,6),'candidate_prob':round(cp,6),
             'combined_margin':round(float(combined[ci]-combined[bi]),6),'candidate_support':[cw,cl,round(crate,4) if crate is not None else None],
             'base_support':[bw,bl,round(brate,4) if brate is not None else None],'reason':'pending'}
        if turn==self.last_turn or self.game_overrides>=_int(self.gate.get('max_game_overrides'),1):rec['reason']='quota';self.stats['decisions'].append(rec);return base
        if pdelta<float(self.gate.get('min_policy_delta',-.15)):self.stats['policy_block']+=1;rec['reason']='policy';self.stats['decisions'].append(rec);return base
        if vm<float(self.gate.get('min_value_margin',.55)):self.stats['margin_block']+=1;rec['reason']='margin';self.stats['decisions'].append(rec);return base
        if bp>float(self.gate.get('max_base_value_prob',.48)) or cp<float(self.gate.get('min_candidate_value_prob',.54)):
            self.stats['risk_block']+=1;rec['reason']='risk';self.stats['decisions'].append(rec);return base
        if not cs:self.stats['support_block']+=1;rec['reason']='no_support';self.stats['decisions'].append(rec);return base
        if cw<_int(self.gate.get('min_candidate_win_games'),3) or crate<float(self.gate.get('min_candidate_rate',.58)):
            self.stats['support_block']+=1;rec['reason']='weak_support';self.stats['decisions'].append(rec);return base
        if bs and brate>=crate-float(self.gate.get('min_support_rate_gap',.05)):
            self.stats['support_block']+=1;rec['reason']='support_gap';self.stats['decisions'].append(rec);return base
        if callable(self.verifier):
            try:
                import time;tv=time.perf_counter();self.stats['search_calls']+=1
                verdict=self.verifier(obs,base,[ci]);self.stats['search_latency_ms_sum']+=(time.perf_counter()-tv)*1000.0
                if isinstance(verdict,tuple):ok,meta=bool(verdict[0]),verdict[1]
                else:ok,meta=bool(verdict),{}
                rec['search']=meta
                if not ok:self.stats['search_reject']+=1;rec['reason']='search_reject';self.stats['decisions'].append(rec);return base
                self.stats['search_accept']+=1
            except Exception as e:
                self.stats['search_errors']+=1;rec['reason']='search_error';rec['search']={'error':type(e).__name__};self.stats['decisions'].append(rec);return base
        self.last_turn=turn;self.game_overrides+=1;self.stats['overrides']+=1;self.stats['keys'][key]=self.stats['keys'].get(key,0)+1;rec['reason']='override';self.stats['decisions'].append(rec)
        return [ci]

    def veto(self,obs,anchor,base):
        """Choose between the pre-replay tactical action and replay-residual action.

        This never invents a third move.  It can only restore the validated tactical
        anchor when the decoder, game-level support, and exact one-turn verifier agree.
        """
        if not self.enabled or not isinstance(obs,dict) or not isinstance(anchor,list) or not isinstance(base,list):return base
        if len(anchor)!=1 or len(base)!=1 or anchor==base:return base
        sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
        if _int(sel.get('context'),-1)!=MAIN or _int(sel.get('minCount'))!=1 or _int(sel.get('maxCount'))!=1 or not opts:return base
        ai=_int(anchor[0],-1);bi=_int(base[0],-1)
        if not (0<=ai<len(opts) and 0<=bi<len(opts)):return base
        self.stats['veto_calls']+=1;turn=_int(cur.get('turn'))
        family=self.replay.recognize(self.history,obs);allowed=set(self.gate.get('veto_allowed_families') or self.gate.get('allowed_families') or [])
        if allowed and family not in allowed:return base
        try:
            import time;t0=time.perf_counter();descs=self._descs(obs);ad=descs[ai];bd=descs[bi]
            at=_int(ad.get('type'),-1);bt=_int(bd.get('type'),-1)
            allowed_types={_int(x) for x in (self.gate.get('veto_allowed_types') or self.gate.get('allowed_types') or [PLAY])}
            if at!=bt or at not in allowed_types:self.stats['veto_type_block']+=1;return base
            context=self.tokens.context_tokens(self.history,obs,family);context_ids=self.tokens.encode(context,self.vocab,self.max_context,True)
            actions=[self.tokens.encode(self.tokens.action_tokens(d),self.vocab,self.max_action,False) for d in (bd,ad)]
            policy,value=self._score_actions(context_ids,actions);self.stats['veto_latency_ms_sum']+=(time.perf_counter()-t0)*1000.0
        except Exception:
            self.stats['veto_errors']+=1;return base
        alpha=float(self.gate.get('veto_alpha',self.gate.get('alpha',.35)));bscore=float(policy[0]+alpha*value[0]);ascore=float(policy[1]+alpha*value[1])
        pd=float(policy[1]-policy[0]);vm=float(value[1]-value[0]);margin=ascore-bscore
        key=f'{family}:{_coarse(bd)}>{_coarse(ad)}';rec={'turn':turn,'key':key,'policy_delta':round(pd,6),'value_margin':round(vm,6),'combined_margin':round(margin,6),'reason':'pending'}
        if margin<float(self.gate.get('veto_min_combined_margin',.015)) or vm<float(self.gate.get('veto_min_value_margin',-.02)):
            self.stats['veto_margin_block']+=1;rec['reason']='margin';self.stats['veto_decisions'].append(rec);return base
        self.stats['veto_suggestions']+=1
        a_sup=self._support(family,turn,ad);b_sup=self._support(family,turn,bd)
        aw=_int((a_sup or {}).get('win_games'));al=_int((a_sup or {}).get('loss_games'));ar=(aw+1.0)/(aw+al+2.0) if a_sup else None
        bw=_int((b_sup or {}).get('win_games'));bl=_int((b_sup or {}).get('loss_games'));br=(bw+1.0)/(bw+bl+2.0) if b_sup else None
        rec['anchor_support']=[aw,al,round(ar,4) if ar is not None else None];rec['residual_support']=[bw,bl,round(br,4) if br is not None else None]
        minw=_int(self.gate.get('veto_min_anchor_win_games',3));minr=float(self.gate.get('veto_min_anchor_rate',.56));gap=float(self.gate.get('veto_min_support_gap',.04))
        if not a_sup or aw<minw or ar<minr or (b_sup and ar<br+gap):
            self.stats['veto_support_block']+=1;rec['reason']='support';self.stats['veto_decisions'].append(rec);return base
        if self.game_vetoes>=_int(self.gate.get('max_game_vetoes'),1):rec['reason']='quota';self.stats['veto_decisions'].append(rec);return base
        if callable(self.verifier):
            try:
                verdict=self.verifier(obs,base,anchor);ok,meta=(bool(verdict[0]),verdict[1]) if isinstance(verdict,tuple) else (bool(verdict),{})
                rec['search']=meta
                if not ok:self.stats['veto_search_reject']+=1;rec['reason']='search_reject';self.stats['veto_decisions'].append(rec);return base
            except Exception as e:
                self.stats['veto_errors']+=1;rec['reason']='search_error';rec['search']={'error':type(e).__name__};self.stats['veto_decisions'].append(rec);return base
        self.game_vetoes+=1;self.stats['vetoes']+=1;self.stats['veto_keys'][key]=self.stats['veto_keys'].get(key,0)+1;rec['reason']='veto';self.stats['veto_decisions'].append(rec)
        return list(anchor)

    def get_stats(self):
        out=dict(self.stats);out.update({'version':self.config.get('version','unavailable'),'architecture':self.architecture,'parameter_count':_int(self.config.get('parameter_count')),'enabled':self.enabled,'action_authority':self.action_authority,'game_overrides':self.game_overrides,'game_vetoes':self.game_vetoes,'load_error':self.load_error,'last':dict(self.last)})
        out['mean_latency_ms']=round(out['latency_ms_sum']/max(1,out['scored']),4)
        out['mean_search_latency_ms']=round(out['search_latency_ms_sum']/max(1,out['search_calls']),4)
        out['mean_veto_latency_ms']=round(out['veto_latency_ms_sum']/max(1,out['veto_calls']),4)
        return out
