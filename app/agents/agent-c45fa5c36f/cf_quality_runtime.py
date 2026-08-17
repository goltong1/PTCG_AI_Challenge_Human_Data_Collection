"""Luck-aware temporal quality residual for Dragapult.

The model is trained only from same-hidden-state CABT counterfactual branches.
Terminal wins/losses are metadata, not direct action labels.  Runtime inputs are
strictly public observation + causal public history.
"""
from __future__ import annotations
import os,json,math
from collections import Counter
try: import numpy as np
except Exception: np=None
import transformer_intent_policy as tip

def _i(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d

def _phase(t):
    t=_i(t);return 'early' if t<=3 else 'mid' if t<=8 else 'late'

def _ln(x,g,b,eps=1e-5):
    m=x.mean(-1,keepdims=True);v=((x-m)**2).mean(-1,keepdims=True)
    return (x-m)/np.sqrt(v+eps)*g+b

def _gelu(x):return .5*x*(1+np.tanh(0.7978845608*(x+0.044715*x*x*x)))

class CFQualityTransformer:
    def __init__(self,root,base_transformer,model='cf_quality_model.npz'):
        self.root=root;self.base=base_transformer;self.meta={};self.w={};self.stats=Counter();self.game_overrides=0;self.last_turn=-1
        if np is None:return
        try:
            z=np.load(os.path.join(root,model),allow_pickle=False)
            raw=bytes(z['meta_json'].astype(np.uint8).tolist()).decode('utf8');self.meta=json.loads(raw)
            self.w={k:np.asarray(z[k],dtype=np.float32) for k in z.files if k!='meta_json'};z.close();self.stats['loaded']+=1
        except Exception:self.stats['load_error']+=1
    @property
    def enabled(self):return bool(np is not None and self.w and self.meta.get('enabled') and getattr(self.base,'weights',None))
    def reset(self):self.game_overrides=0;self.last_turn=-1;self.stats['games']+=1
    def get_stats(self):
        x={'cfq_'+str(k):int(v) for k,v in self.stats.items()};x['cfq_game_overrides']=int(self.game_overrides);return x
    def _block(self,x,W,p):
        y=_ln(x,W[p+'ln1_g'],W[p+'ln1_b']);qkv=y@W[p+'qkv_w']+W[p+'qkv_b'];q,k,v=np.split(qkv,3,-1)
        H=_i(self.meta.get('heads'),4);D=x.shape[-1];dh=D//H;L=len(x);scale=1/math.sqrt(max(1,dh))
        q=q.reshape(L,H,dh).transpose(1,0,2);k=k.reshape(L,H,dh).transpose(1,0,2);v=v.reshape(L,H,dh).transpose(1,0,2)
        a=q@k.transpose(0,2,1)*scale;a=np.exp(a-a.max(-1,keepdims=True));a/=np.maximum(a.sum(-1,keepdims=True),1e-12)
        z=(a@v).transpose(1,0,2).reshape(L,D);x=x+z@W[p+'out_w']+W[p+'out_b']
        y=_ln(x,W[p+'ln2_g'],W[p+'ln2_b']);y=_gelu(y@W[p+'ff1_w']+W[p+'ff1_b']);return x+y@W[p+'ff2_w']+W[p+'ff2_b']
    def _score(self,ids):
        B=self.base.weights;ids=ids[:_i(self.meta.get('max_tokens'),96)];L=len(ids);pos=np.empty((L,B['position_embedding'].shape[1]),np.float32)
        n=min(L,len(B['position_embedding']));pos[:n]=B['position_embedding'][:n]
        if L>n:
            e=self.w.get('position_extra');pos[n:]=e[:L-n] if e is not None and len(e)>=L-n else B['position_embedding'][-1]
        x=B['token_embedding'][np.asarray(ids,np.int64)]+pos
        for j in range(_i(self.meta.get('base_layers'),4)):x=self._block(x,B,'l%d_'%j)
        upper=_i(self.meta.get('upper_tokens'),24)
        if len(x)>upper:x=np.concatenate([x[:1],x[-(upper-1):]],axis=0)
        for j in range(_i(self.meta.get('extra_layers'),2)):x=self._block(x,self.w,'x%d_'%j)
        z=_ln(x[:1],self.w['q_ln_g'],self.w['q_ln_b'])[0]
        return float(z@self.w['quality_w']+self.w['quality_b'])
    def _cand_tokens(self,sem,intent):
        return ['[CAND]','cand_intent='+str(intent),f"cand_type={_i(sem.get('option_type'),-1)}",f"cand_card={_i(sem.get('card_id'),0)}",f"cand_target={_i(sem.get('target_card_id'),0)}",f"cand_attack={_i(sem.get('attack_id'),0)}",f"cand_area={_i(sem.get('target_area'),-1)}",'cand_num='+str(sem.get('number'))]
    def _key(self,matchup,turn,basei,candi):return f"{matchup}|{_phase(turn)}|{basei}|{candi}"
    def choose(self,observation,chosen,history,matchup=None,confidence=0.0):
        self.stats['calls']+=1
        if not self.enabled or not isinstance(chosen,list) or len(chosen)!=1:return chosen
        sel=observation.get('select') or {};cur=observation.get('current') or {};opts=sel.get('option') or []
        if _i(sel.get('context'),-1)!=0 or _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or len(opts)<2 or len(opts)>_i(self.meta.get('max_options'),24):return chosen
        turn=_i(cur.get('turn'));matchup=str(matchup or 'unknown');bi=_i(chosen[0],-1)
        if not 0<=bi<len(opts):return chosen
        dec=list(getattr(history,'decisions',[]) or []);pub=list(getattr(history,'public_events',[]) or [])
        sems=[];intents=[]
        for i in range(len(opts)):
            try:s=tip.semantic_option(observation,i);it=tip.intent_key(observation,s,dec) if s else 'invalid'
            except Exception:s={'index':i,'option_type':-1};it='invalid'
            sems.append(s);intents.append(it)
        basei=intents[bi];support=self.meta.get('support') or {};eligible=[]
        for i,it in enumerate(intents):
            if i==bi:continue
            rec=support.get(self._key(matchup,turn,basei,it))
            if not rec:continue
            typ=_i(sems[i].get('option_type'),-1);btyp=_i(sems[bi].get('option_type'),-1)
            need=_i(self.meta.get('min_support'),2)
            if btyp in (tip.OPT_ATTACK,tip.OPT_RETREAT) or typ in (tip.OPT_ATTACK,tip.OPT_RETREAT):need=max(need,_i(self.meta.get('commitment_support'),4))
            if _i(rec.get('positive'))<need or float(rec.get('positive_rate',0))<float(self.meta.get('min_positive_rate',.60)) or float(rec.get('mean_gain',0))<float(self.meta.get('min_mean_gain',45000)):continue
            if typ==tip.OPT_END:continue
            if btyp==tip.OPT_ATTACK and typ!=tip.OPT_ATTACK and (_i(rec.get('positive'))<_i(self.meta.get('attack_leave_support'),6) or float(rec.get('mean_gain',0))<float(self.meta.get('attack_leave_gain',120000))):continue
            eligible.append((i,rec))
        if not eligible:self.stats['support_block']+=1;return chosen
        legal=sorted(set(intents));ctx=tip.build_tokens(observation,dec,pub,matchup,confidence,legal,_i(self.meta.get('context_tokens'),84))
        vocab=_i(self.meta.get('vocab_size'),4096);mt=_i(self.meta.get('max_tokens'),96)
        try:
            b_ids=tip.tokens_to_ids(ctx+self._cand_tokens(sems[bi],basei),vocab,mt);bs=self._score(b_ids)
            scored=[]
            for i,rec in eligible:
                ids=tip.tokens_to_ids(ctx+self._cand_tokens(sems[i],intents[i]),vocab,mt);scored.append((self._score(ids),float(rec.get('mean_gain',0)),i,rec))
            cs,g,ci,rec=max(scored,key=lambda x:(x[0]-bs,x[1]))
        except Exception:self.stats['inference_error']+=1;return chosen
        margin=float(cs-bs);self.stats['evaluated']+=1
        if margin<float(self.meta.get('min_model_margin',0.10)):self.stats['margin_block']+=1;return chosen
        if self.game_overrides>=_i(self.meta.get('max_game_overrides'),2) or turn==self.last_turn:self.stats['quota_block']+=1;return chosen
        self.game_overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;self.stats['override_'+matchup]+=1
        return [int(ci)]
