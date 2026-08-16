"""v162 Token-Cross luck-aware counterfactual local-action quality PlanFormer.

Trained on same-hidden-state CABT branch comparisons. Terminal win/loss is only a weak
auxiliary target, so an unlucky loss does not automatically make its actions negative.
"""
from __future__ import annotations
import os,json,math,time
try: import numpy as np
except Exception: np=None
MAIN=0

def _i(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d
def _tb(t):return 'early' if _i(t)<=3 else 'mid' if _i(t)<=7 else 'late'
def _coarse(d):return f"{_i(d.get('type'),-1)}:{_i(d.get('cardId'))}:{_i(d.get('targetId'))}:{_i(d.get('attackId'))}"
def _sig(x):x=max(-30.,min(30.,float(x)));return 1/(1+math.exp(-x))
class QualityPlanFormer:
 def __init__(self,root,history,replay,token_module,config='quality_planformer_model.json',weights='quality_planformer_model.npz'):
  self.root=root;self.history=history;self.replay=replay;self.tok=token_module;self.verifier=None;self.load_error=None
  try:self.cfg=json.load(open(os.path.join(root,config),encoding='utf8'))
  except Exception as e:self.cfg={'enabled':False};self.load_error='cfg:'+type(e).__name__
  self.gate=self.cfg.get('gate') or {};self.w={}
  if np is not None and self.cfg.get('enabled'):
   try:
    z=np.load(os.path.join(root,weights),allow_pickle=False);self.w={k:z[k] for k in z.files};z.close()
   except Exception as e:self.load_error='weights:'+type(e).__name__
  self.enabled=bool(np is not None and self.w and self.cfg.get('enabled'));self.vocab=_i(self.cfg.get('vocab_size'),8192);self.mc=_i(self.cfg.get('max_context'),112);self.ma=_i(self.cfg.get('max_action'),40);self.d=_i(self.cfg.get('d_model'),64);self.h=_i(self.cfg.get('heads'),4);self.reset()
 def reset(self):
  self.overrides=0;self.last_turn=-1;self.stats={'calls':0,'scored':0,'suggestions':0,'overrides':0,'vetoes':0,'support_block':0,'margin_block':0,'family_block':0,'search_calls':0,'search_accept':0,'search_reject':0,'errors':0,'latency_ms_sum':0.,'search_latency_ms_sum':0.,'decisions':[]}
 def _ln(self,x,w,b,eps=1e-5):m=x.mean(-1,keepdims=True);v=((x-m)**2).mean(-1,keepdims=True);return (x-m)/np.sqrt(v+eps)*w+b
 def _gelu(self,x):return .5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))
 def _softmax(self,x):x=x-x.max(-1,keepdims=True);e=np.exp(x);return e/np.maximum(e.sum(-1,keepdims=True),1e-12)
 def _self_block(self,x,pad,prefix):
  h=self._ln(x,self.w[prefix+'ln1.weight'],self.w[prefix+'ln1.bias']);W=self.w[prefix+'att.in_proj_weight'];b=self.w[prefix+'att.in_proj_bias'];qkv=h@W.T+b;q,k,v=np.split(qkv,3,-1);B,L,D=q.shape;dh=D//self.h
  q=q.reshape(B,L,self.h,dh).transpose(0,2,1,3);k=k.reshape(B,L,self.h,dh).transpose(0,2,1,3);v=v.reshape(B,L,self.h,dh).transpose(0,2,1,3);s=q@k.transpose(0,1,3,2)/math.sqrt(dh);s=np.where(pad[:,None,None,:],-1e4,s);a=self._softmax(s);y=(a@v).transpose(0,2,1,3).reshape(B,L,D);y=y@self.w[prefix+'att.out_proj.weight'].T+self.w[prefix+'att.out_proj.bias'];x=x+y;h=self._ln(x,self.w[prefix+'ln2.weight'],self.w[prefix+'ln2.bias']);ff=self._gelu(h@self.w[prefix+'fc1.weight'].T+self.w[prefix+'fc1.bias']);return x+ff@self.w[prefix+'fc2.weight'].T+self.w[prefix+'fc2.bias']
 def _forward(self,cids,action_ids):
  cl=min(len(cids),self.mc);N=len(action_ids);ci=np.arange(self.mc);cpad=np.broadcast_to((ci>=cl)[None,:],(1,self.mc));c=np.zeros((1,self.mc),np.int64);c[0,:cl]=np.asarray(cids[-cl:]);cx=self.w['emb.weight'][c]+self.w['cpos.weight'][ci][None,:,:];cx=self._self_block(cx,cpad,'ctx.0.');cx=self._ln(cx,self.w['c_ln.weight'],self.w['c_ln.bias']);cvec=np.broadcast_to(cx[0,max(0,cl-1)][None,:],(N,self.d))
  a=np.zeros((N,self.ma),np.int64);al=np.zeros(N,np.int64)
  for j,s in enumerate(action_ids):q=list(s[:self.ma]);al[j]=len(q);a[j,:len(q)]=q
  ai=np.arange(self.ma);apad=ai[None,:]>=al[:,None];ax=self.w['emb.weight'][a]+self.w['apos.weight'][ai][None,:,:];ax=self._self_block(ax,apad,'act.0.');ax=self._ln(ax,self.w['a_ln.weight'],self.w['a_ln.bias']);mask=(~apad).astype(np.float32);avec=(ax*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
  # v162 Token-Cross: every action token queries the shared context, then the attended
  # token sequence is masked-mean pooled.  This preserves target/card/attack detail that
  # was blurred by the old pre-cross action mean.
  W=self.w['cross.in_proj_weight'];bb=self.w['cross.in_proj_bias'];Wq,Wk,Wv=np.split(W,3,0);bq,bk,bv=np.split(bb,3);q=ax@Wq.T+bq;k=cx@Wk.T+bk;v=cx@Wv.T+bv;dh=self.d//self.h;q=q.reshape(N,self.ma,self.h,dh).transpose(0,2,1,3);k=k.reshape(1,self.mc,self.h,dh).transpose(0,2,1,3);v=v.reshape(1,self.mc,self.h,dh).transpose(0,2,1,3);scr=q@k.transpose(0,1,3,2)/math.sqrt(dh);scr=np.where(cpad[:,None,None,:],-1e4,scr);att=self._softmax(scr);xt=(att@v).transpose(0,2,1,3).reshape(N,self.ma,self.d);xt=xt@self.w['cross.out_proj.weight'].T+self.w['cross.out_proj.bias'];cross=(xt*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
  z=np.concatenate([avec,cvec,cross],-1);z=self._gelu(z@self.w['f0.weight'].T+self.w['f0.bias']);z=z@self.w['f2.weight'].T+self.w['f2.bias'];z=self._ln(z,self.w['fln.weight'],self.w['fln.bias']);p=(z@self.w['policy.weight'].T+self.w['policy.bias']).reshape(-1);qv=(z@self.w['quality.weight'].T+self.w['quality.bias']).reshape(-1);o=(z@self.w['outcome.weight'].T+self.w['outcome.bias']).reshape(-1);return p,qv,o
 def _descs(self,obs):
  out=[];opts=((obs.get('select') or {}).get('option') or [])
  for j in range(len(opts)):
   d=self.replay.action_desc(self.history,obs,[j]);d=d[0] if d else {'index':j,'type':-1};out.append(self.tok.enrich_desc(obs,j,d))
  return out
 def _support(self,fam,turn,d):
  keyfam='grass' if fam=='hydrapple' else fam; bucket=_tb(turn); typ=_i(d.get('type'),-1); cid=_i(d.get('cardId'));
  exact=(self.cfg.get('quality_support') or {}).get(f'{keyfam}|{bucket}|{_coarse(d)}')
  if exact:return exact,'exact'
  card=(self.cfg.get('quality_support_card') or {}).get(f'{keyfam}|{bucket}|{typ}:{cid}')
  if card and _i(card.get('games'))>=_i(self.gate.get('backoff_card_min_games'),2):return card,'card'
  typv=(self.cfg.get('quality_support_type') or {}).get(f'{keyfam}|{bucket}|{typ}')
  if typv and _i(typv.get('games'))>=_i(self.gate.get('backoff_type_min_games'),5):return typv,'type'
  return None,'none'
 def _score(self,obs,inds,descs,fam):
  ctx=self.tok.context_tokens(self.history,obs,fam);it=self.tok.intent_context_tokens(ctx,self.mc);cids=self.tok.encode(it,self.vocab,self.mc,True);acts=[self.tok.encode(self.tok.action_tokens(descs[i]),self.vocab,self.ma,False) for i in inds];return self._forward(cids,acts)
 def guard(self,obs,anchor,candidate):
  if not self.enabled or anchor==candidate or not isinstance(anchor,list) or not isinstance(candidate,list) or len(anchor)!=1 or len(candidate)!=1:return candidate
  try:
   fam=self.replay.recognize(self.history,obs);descs=self._descs(obs);ai,ci=_i(anchor[0],-1),_i(candidate[0],-1)
   if not (0<=ai<len(descs) and 0<=ci<len(descs)):return candidate
   p,q,o=self._score(obs,[ai,ci],descs,fam);self.stats['scored']+=1;ap,cp=_sig(q[0]),_sig(q[1]);m=float(q[0]-q[1])
   if m>=float(self.gate.get('strong_veto_margin',.8)) and ap>=float(self.gate.get('strong_veto_anchor_prob',.65)) and cp<=float(self.gate.get('strong_veto_candidate_prob',.35)):
    self.stats['vetoes']+=1;self.stats['decisions'].append({'reason':'strong_quality_veto','family':fam,'anchor_q':round(ap,4),'candidate_q':round(cp,4)});return anchor
  except Exception:self.stats['errors']+=1
  return candidate
 def choose(self,obs,base):
  if not self.enabled or not isinstance(base,list) or len(base)!=1:return base
  sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
  if _i(sel.get('context'),-1)!=MAIN or _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or not opts or len(opts)>_i(self.gate.get('max_options'),24):return base
  bi=_i(base[0],-1);turn=_i(cur.get('turn'));self.stats['calls']+=1
  if not 0<=bi<len(opts):return base
  fam=self.replay.recognize(self.history,obs);allowed=set(self.gate.get('allowed_families') or [])
  if fam=='crustle' or (allowed and fam not in allowed):self.stats['family_block']+=1;return base
  if fam=='unknown' and turn<_i(self.gate.get('unknown_min_turn'),5):self.stats['family_block']+=1;return base
  try:
   t=time.perf_counter();descs=self._descs(obs);bt=_i(descs[bi].get('type'),-1);types={_i(x) for x in self.gate.get('allowed_types',[7,8])}
   if bt not in types:return base
   cand=[i for i,d in enumerate(descs) if i!=bi and _i(d.get('type'),-2)==bt and self.tok.action_signature(d)!=self.tok.action_signature(descs[bi])]
   if not cand:return base
   inds=[bi]+cand;p,q,o=self._score(obs,inds,descs,fam);self.stats['scored']+=1;self.stats['latency_ms_sum']+=(time.perf_counter()-t)*1000
   score=.65*p+1.15*q;j=max(range(1,len(inds)),key=lambda x:(float(score[x]),float(q[x]),float(p[x])));ci=inds[j];pd=float(p[j]-p[0]);qm=float(q[j]-q[0]);cal=self.cfg.get('quality_calibration') or {};qp=_sig(float(cal.get('scale',1.0))*q[j]+float(cal.get('bias',0.0)));self.stats['suggestions']+=1
   rec={'family':fam,'turn':turn,'base':bi,'candidate':ci,'policy_margin':round(pd,4),'quality_margin':round(qm,4),'candidate_quality_prob':round(qp,4)}
   if self.overrides>=_i(self.gate.get('max_game_overrides'),1) or turn==self.last_turn:rec['reason']='quota';self.stats['decisions'].append(rec);return base
   if qp<float(self.gate.get('min_quality_prob',.68)) or qm<float(self.gate.get('min_quality_margin',.45)) or pd<float(self.gate.get('min_policy_margin',.02)):
    self.stats['margin_block']+=1;rec['reason']='margin';self.stats['decisions'].append(rec);return base
   s,slevel=self._support(fam,turn,descs[ci]);g=_i((s or {}).get('games'));good=_i((s or {}).get('good_games'));rate=good/max(1,g);rec['support_level']=slevel
   need=(_i(self.gate.get('unknown_min_support_games'),2) if fam=='unknown' else _i(self.gate.get('min_support_games'),2))
   if not s or g<need or rate<float(self.gate.get('min_support_good_rate',.67)):
    self.stats['support_block']+=1;rec['reason']='quality_support';rec['support']=s;self.stats['decisions'].append(rec);return base
   maxsearch=_i(self.gate.get('unknown_max_game_search_calls'),2) if fam=='unknown' else _i(self.gate.get('max_game_search_calls'),4)
   if self.stats['search_calls']>=maxsearch:rec['reason']='search_quota';self.stats['decisions'].append(rec);return base
   if callable(self.verifier):
    tv=time.perf_counter();self.stats['search_calls']+=1;ver=self.verifier(obs,base,[ci]);self.stats['search_latency_ms_sum']+=(time.perf_counter()-tv)*1000;ok,meta=(bool(ver[0]),ver[1]) if isinstance(ver,tuple) else (bool(ver),{});gain=float((meta or {}).get('gain',-1e9));rec['search']=meta
    if not ok or gain<float(self.gate.get('search_min_gain',30)):
     self.stats['search_reject']+=1;rec['reason']='search_reject';self.stats['decisions'].append(rec);return base
    self.stats['search_accept']+=1
   self.overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;rec['reason']='override';self.stats['decisions'].append(rec);return [ci]
  except Exception as e:self.stats['errors']+=1;return base
 def get_stats(self):
  x=dict(self.stats);x.update({'enabled':self.enabled,'version':self.cfg.get('version'),'parameter_count':_i(self.cfg.get('parameter_count')),'load_error':self.load_error,'game_overrides':self.overrides});x['mean_latency_ms']=round(x['latency_ms_sum']/max(1,x['scored']),4);x['mean_search_latency_ms']=round(x['search_latency_ms_sum']/max(1,x['search_calls']),4);return x
