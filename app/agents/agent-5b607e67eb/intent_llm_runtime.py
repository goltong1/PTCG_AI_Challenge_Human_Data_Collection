"""Lucario v160 strategic-intent sidecar Transformer.

A compact second language-model stream that reads strategic/card-text intent tokens
on top of the validated v159 long-context model. It can only replace a same-type
PLAY/ATTACH action after game-level support and exact CABT search both approve it.
"""
from __future__ import annotations
import json, math, os, time
try:
    import numpy as np
except Exception:
    np=None

MAIN=0
BLACK_BELT=1211
ATTACK=13

def _int(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d

def _turnbin(t):return 'early' if int(t)<=3 else 'mid' if int(t)<=7 else 'late'
def _coarse(d):return f"{_int(d.get('type'),-1)}:{_int(d.get('cardId'))}:{_int(d.get('targetId'))}:{_int(d.get('attackId'))}"

def _sigmoid(x):
    x=max(-30.0,min(30.0,float(x)))
    return 1.0/(1.0+math.exp(-x))

class IntentDecoderPolicy:
    def __init__(self,root,history,replay_module,token_module,config_file='intent_llm_model.json',weights_file='intent_llm_model.npz'):
        self.root=root;self.history=history;self.replay=replay_module;self.tokens=token_module;self.verifier=None;self.load_error=None
        try:self.config=json.load(open(os.path.join(root,config_file),encoding='utf-8'))
        except Exception as e:self.config={'enabled':False,'gate':{}};self.load_error=f'config:{type(e).__name__}'
        self.gate=self.config.get('gate') or {};self.weights={}
        if np is not None and self.config.get('enabled',False):
            try:
                z=np.load(os.path.join(root,weights_file),allow_pickle=False);self.weights={k:z[k] for k in z.files};z.close()
            except Exception as e:self.load_error=f'weights:{type(e).__name__}'
        elif np is None:self.load_error='numpy_unavailable'
        self.enabled=bool(np is not None and self.weights and self.config.get('enabled',False))
        self.vocab=max(8,_int(self.config.get('vocab_size'),8192));self.max_context=max(8,_int(self.config.get('max_context'),72));self.max_action=max(4,_int(self.config.get('max_action'),40));self.max_len=max(self.max_context+self.max_action,_int(self.config.get('max_len'),112));self.d=max(1,_int(self.config.get('d_model'),64));self.heads=max(1,_int(self.config.get('heads'),4));self.layers=max(1,_int(self.config.get('layers'),2));self.reset()

    def reset(self):
        self.game_overrides=0;self.last_turn=-1
        self.stats={'calls':0,'scored':0,'suggestions':0,'overrides':0,'family_block':0,'context_block':0,'type_block':0,'support_block':0,'margin_block':0,'policy_block':0,'quota_block':0,'search_calls':0,'search_accept':0,'search_reject':0,'search_errors':0,'errors':0,'latency_ms_sum':0.0,'search_latency_ms_sum':0.0,'keys':{},'decisions':[]}
        self.last={'enabled':self.enabled,'family':'unknown','base':[],'candidate':[]}

    @staticmethod
    def _ln(x,w,b,eps=1e-5):
        m=x.mean(axis=-1,keepdims=True);v=((x-m)*(x-m)).mean(axis=-1,keepdims=True)
        return (x-m)/np.sqrt(v+eps)*w+b
    @staticmethod
    def _gelu(x):return .5*x*(1.0+np.tanh(math.sqrt(2.0/math.pi)*(x+.044715*x*x*x)))

    def _forward(self,context_ids,action_ids):
        n=len(action_ids);cl=min(len(context_ids),self.max_context)
        if n<=0 or cl<=0:raise ValueError('empty intent batch')
        c=np.zeros((n,self.max_context),dtype=np.int64);c[:,:cl]=np.asarray(context_ids[-cl:],dtype=np.int64)[None,:]
        a=np.zeros((n,self.max_action),dtype=np.int64);al=np.zeros(n,dtype=np.int64)
        for i,s in enumerate(action_ids):
            q=list(s[:self.max_action]);al[i]=len(q);a[i,:len(q)]=q
        if np.any(al<=0):raise ValueError('empty action')
        ci=np.arange(self.max_context);ai=np.arange(self.max_action)
        cx=self.weights['emb.weight'][c]+self.weights['pos.weight'][ci][None,:,:]
        apos=np.minimum(cl+ai,self.max_len-1)
        ax=self.weights['emb.weight'][a]+self.weights['pos.weight'][apos][None,:,:]+self.weights['seg.weight'][1][None,None,:]
        x=np.concatenate((cx,ax),axis=1)
        valid=np.concatenate((np.broadcast_to((ci<cl)[None,:],(n,self.max_context)),ai[None,:]<al[:,None]),axis=1)
        total=self.max_context+self.max_action;causal=np.triu(np.ones((total,total),dtype=bool),1);dh=self.d//self.heads
        for li in range(self.layers):
            p=f'blocks.{li}.';h=self._ln(x,self.weights[p+'ln1.weight'],self.weights[p+'ln1.bias'])
            q=h@self.weights[p+'q.weight'].T+self.weights[p+'q.bias'];k=h@self.weights[p+'k.weight'].T+self.weights[p+'k.bias'];v=h@self.weights[p+'v.weight'].T+self.weights[p+'v.bias']
            q=q.reshape(n,total,self.heads,dh).transpose(0,2,1,3);k=k.reshape(n,total,self.heads,dh).transpose(0,2,1,3);v=v.reshape(n,total,self.heads,dh).transpose(0,2,1,3)
            score=np.matmul(q,k.transpose(0,1,3,2))/math.sqrt(dh);score=np.where(causal[None,None,:,:],-1e4,score);score=np.where((~valid)[:,None,None,:],-1e4,score)
            score-=score.max(axis=-1,keepdims=True);att=np.exp(score);att/=np.maximum(att.sum(axis=-1,keepdims=True),1e-12)
            y=np.matmul(att,v).transpose(0,2,1,3).reshape(n,total,self.d)
            x=x+y@self.weights[p+'o.weight'].T+self.weights[p+'o.bias']
            h=self._ln(x,self.weights[p+'ln2.weight'],self.weights[p+'ln2.bias']);ff=self._gelu(h@self.weights[p+'fc1.weight'].T+self.weights[p+'fc1.bias']);x=x+ff@self.weights[p+'fc2.weight'].T+self.weights[p+'fc2.bias']
        x=self._ln(x,self.weights['ln.weight'],self.weights['ln.bias']);idx=self.max_context+al-1;last=x[np.arange(n),idx]
        policy=(last@self.weights['policy.weight'].T+self.weights['policy.bias']).reshape(-1);value=(last@self.weights['value.weight'].T+self.weights['value.bias']).reshape(-1)
        return policy,value

    def _descs(self,obs):
        opts=((obs.get('select') or {}).get('option') or []);out=[]
        for i in range(len(opts)):
            d=self.replay.action_desc(self.history,obs,[i]);d=d[0] if d else {'index':i,'type':-1};out.append(self.tokens.enrich_desc(obs,i,d))
        return out
    def _support(self,family,turn,d):return (self.config.get('support') or {}).get(f'{family}|{_turnbin(turn)}|{_coarse(d)}')

    def choose(self,obs,base):
        if not self.enabled or not isinstance(obs,dict) or not isinstance(base,list):return base
        sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
        if _int(sel.get('context'),-1)!=MAIN or _int(sel.get('minCount'))!=1 or _int(sel.get('maxCount'))!=1 or len(base)!=1 or not opts or len(opts)>_int(self.gate.get('max_options'),24):self.stats['context_block']+=1;return base
        bi=_int(base[0],-1);turn=_int(cur.get('turn'))
        if not 0<=bi<len(opts):return base
        self.stats['calls']+=1;family=self.replay.recognize(self.history,obs);allowed=set(self.gate.get('allowed_families') or [])
        if allowed and family not in allowed:self.stats['family_block']+=1;return base
        if family=='unknown' and turn<_int(self.gate.get('unknown_min_turn'),4):self.stats['family_block']+=1;return base
        try:
            t0=time.perf_counter();descs=self._descs(obs);bdesc=descs[bi];btype=_int(bdesc.get('type'),-1);allowed_types={_int(x) for x in (self.gate.get('allowed_types') or [7,8])}
            if btype not in allowed_types:self.stats['type_block']+=1;return base
            context=self.tokens.context_tokens(self.history,obs,family);intent=self.tokens.intent_context_tokens(context,self.max_context);cids=self.tokens.encode(intent,self.vocab,self.max_context,True)
            cand=[]
            for i,d in enumerate(descs):
                if i==bi or _int(d.get('type'),-1)!=btype or self.tokens.action_signature(d)==self.tokens.action_signature(bdesc):continue
                if _int(d.get('cardId'))==BLACK_BELT and not any(_int(x.get('type'),-1)==ATTACK for x in opts):continue
                cand.append(i)
            if not cand:self.stats['type_block']+=1;return base
            inds=[bi]+cand;acts=[self.tokens.encode(self.tokens.action_tokens(descs[i]),self.vocab,self.max_action,False) for i in inds];policy,value=self._forward(cids,acts);self.stats['scored']+=1;self.stats['latency_ms_sum']+=(time.perf_counter()-t0)*1000.0
        except Exception as e:
            self.stats['errors']+=1;self.last={'enabled':self.enabled,'error':type(e).__name__};return base
        alpha=float(self.gate.get('alpha',.55));score=policy+alpha*value;j=max(range(1,len(inds)),key=lambda z:(float(score[z]),float(value[z]),float(policy[z]),-inds[z]));ci=inds[j];cdesc=descs[ci];pd=float(policy[j]-policy[0]);vm=float(value[j]-value[0]);cm=float(score[j]-score[0])
        self.last={'enabled':True,'family':family,'turn':turn,'base':list(base),'candidate':[ci],'base_desc':bdesc,'candidate_desc':cdesc,'policy_delta':round(pd,6),'value_margin':round(vm,6),'combined_margin':round(cm,6),'base_prob':round(_sigmoid(value[0]),6),'candidate_prob':round(_sigmoid(value[j]),6)}
        if cm<=0:return base
        self.stats['suggestions']+=1;key=f'{family}:{_coarse(bdesc)}>{_coarse(cdesc)}';cs=self._support(family,turn,cdesc);bs=self._support(family,turn,bdesc)
        cw=_int((cs or {}).get('win_games'));cl=_int((cs or {}).get('loss_games'));crate=(cw+1)/(cw+cl+2) if cs else None;bw=_int((bs or {}).get('win_games'));bl=_int((bs or {}).get('loss_games'));brate=(bw+1)/(bw+bl+2) if bs else None
        rec={'turn':turn,'key':key,'policy_delta':round(pd,6),'value_margin':round(vm,6),'combined_margin':round(cm,6),'candidate_support':[cw,cl,round(crate,4) if crate is not None else None],'base_support':[bw,bl,round(brate,4) if brate is not None else None],'reason':'pending'}
        if self.game_overrides>=_int(self.gate.get('max_game_overrides'),1) or turn==self.last_turn:self.stats['quota_block']+=1;rec['reason']='quota';self.stats['decisions'].append(rec);return base
        if pd<float(self.gate.get('min_policy_delta',.05)):self.stats['policy_block']+=1;rec['reason']='policy';self.stats['decisions'].append(rec);return base
        if vm<float(self.gate.get('min_value_margin',.25)):self.stats['margin_block']+=1;rec['reason']='margin';self.stats['decisions'].append(rec);return base
        if not cs or cw<_int(self.gate.get('min_candidate_win_games'),4) or crate<float(self.gate.get('min_candidate_rate',.60)):
            self.stats['support_block']+=1;rec['reason']='candidate_support';self.stats['decisions'].append(rec);return base
        if bs and crate<brate+float(self.gate.get('min_support_rate_gap',.07)):
            self.stats['support_block']+=1;rec['reason']='support_gap';self.stats['decisions'].append(rec);return base
        if callable(self.verifier):
            try:
                tv=time.perf_counter();self.stats['search_calls']+=1;verdict=self.verifier(obs,base,[ci]);self.stats['search_latency_ms_sum']+=(time.perf_counter()-tv)*1000.0;ok,meta=(bool(verdict[0]),verdict[1]) if isinstance(verdict,tuple) else (bool(verdict),{})
                need=float(self.gate.get('search_min_gain',30.0));gain=float((meta or {}).get('gain',-1e9));rec['search']=meta
                if not ok or gain<need:self.stats['search_reject']+=1;rec['reason']='search_reject';self.stats['decisions'].append(rec);return base
                self.stats['search_accept']+=1
            except Exception as e:
                self.stats['search_errors']+=1;rec['reason']='search_error';rec['search']={'error':type(e).__name__};self.stats['decisions'].append(rec);return base
        self.game_overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;self.stats['keys'][key]=self.stats['keys'].get(key,0)+1;rec['reason']='override';self.stats['decisions'].append(rec);return [ci]

    def get_stats(self):
        out=dict(self.stats);out.update({'version':self.config.get('version','unavailable'),'parameter_count':_int(self.config.get('parameter_count')),'enabled':self.enabled,'load_error':self.load_error,'game_overrides':self.game_overrides,'last':dict(self.last)})
        out['mean_latency_ms']=round(out['latency_ms_sum']/max(1,out['scored']),4);out['mean_search_latency_ms']=round(out['search_latency_ms_sum']/max(1,out['search_calls']),4);return out
