from __future__ import annotations
import hashlib,json,math,os,time
import numpy as np

MAIN=0; PLAY=7; ATTACH=8; EVOLVE=9; ABILITY=10; RETREAT=12; ATTACK=13; END=14

def _i(x,d=0):
 try:return int(x if x is not None else d)
 except Exception:return d

def _softmax(x,axis=-1):
 x=x-np.max(x,axis=axis,keepdims=True);e=np.exp(x);return e/np.maximum(e.sum(axis=axis,keepdims=True),1e-12)

def _erf(x):
 # Abramowitz-Stegun 7.1.26, vectorized; sufficient to reproduce exact GELU closely.
 s=np.sign(x);a=np.abs(x);t=1.0/(1.0+0.3275911*a)
 y=1.0-(((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t)*np.exp(-a*a)
 return s*y

def _gelu(x):return 0.5*x*(1.0+_erf(x/math.sqrt(2.0)))

def _ln(x,w,b,eps=1e-5):
 m=x.mean(-1,keepdims=True);v=((x-m)**2).mean(-1,keepdims=True);return (x-m)/np.sqrt(v+eps)*w+b

def _mha(qin,kin,vin,W,b,Wo,bo,heads,key_pad=None):
 Wq,Wk,Wv=np.split(W,3,0);bq,bk,bv=np.split(b,3)
 q=qin@Wq.T+bq;k=kin@Wk.T+bk;v=vin@Wv.T+bv
 B,Lq,D=q.shape;Lk=k.shape[1];dh=D//heads
 q=q.reshape(B,Lq,heads,dh).transpose(0,2,1,3);k=k.reshape(B,Lk,heads,dh).transpose(0,2,1,3);v=v.reshape(B,Lk,heads,dh).transpose(0,2,1,3)
 s=q@k.transpose(0,1,3,2)/math.sqrt(dh)
 if key_pad is not None:s=np.where(key_pad[:,None,None,:],-1e4,s)
 a=_softmax(s,-1);y=(a@v).transpose(0,2,1,3).reshape(B,Lq,D);return y@Wo.T+bo

def _phase(t):return 'early' if t<=3 else 'mid' if t<=7 else 'late'

def _coarse(d):return f"{_i(d.get('type'),-1)}:{_i(d.get('cardId'))}:{_i(d.get('attackId'))}"

def _action_tags(d,fam):
 typ=_i(d.get('type'),-1);cid=_i(d.get('cardId'));tid=_i(d.get('targetId'));aid=_i(d.get('attackId'));s=set()
 if cid in {333,677,678,676,675,1142,1141,20,6} or tid in {333,677,678,676,675}:s.add(0)
 if cid in {305,66,306} or tid in {305,66,306} or aid==426:s.add(1)
 if cid in {1152,1086,1225,1227,1142,1141,1097}:s.add(2)
 if typ in {RETREAT,ATTACK} or cid==1182 or aid in {982,983,426,148,464}:s.add(3)
 if cid==117 or aid in {148,426}:s.add(4)
 if cid in {1213,1197,1227}:s.add(5)
 if not s:s.add(2 if typ==PLAY else 3 if typ in {RETREAT,ATTACK} else 0)
 return sorted(s)

def _hier_tokens(src,max_len=192):
 src=list(src or []);head=[];events=[];decisions=[];board=[];strat=[]
 headpref=('FAMILY=','CTX=','TURN_BIN=','PHASE=','MY_PRIZE=','OPP_PRIZE=','PRIZE_DIFF=','MY_HAND=','OPP_HAND=','MY_DECK=','OPP_DECK=','FLAG_','FIRST_REL=','HIST_','SEM_VERSION=')
 boardpref=('OWN_HAND_CARD=','OPP_KNOWN_CARD=','OWN_ACTIVE_','OWN_BENCH_','OPP_ACTIVE_','OPP_BENCH_','STADIUM=','SEM_CARD_','SEM_ATTACK_')
 for t in src:
  s=str(t)
  if s=='<BOS>' or s.startswith(headpref):head.append(s)
  elif s=='<EVENT>' or s.startswith('EV_'):events.append(s)
  elif s=='<DECISION>' or s.startswith('DEC_'):decisions.append(s)
  elif s.startswith('STRAT_'):strat.append(s)
  elif s=='<STATE>' or s.startswith(boardpref):board.append(s)
 out=['<BOS>']+[x for x in head if x!='<BOS>'][:26]+events[-72:]+decisions[-24:]+board[-44:]+strat[-40:]
 if len(out)>max_len:
  fixed=out[:27];tail=[x for x in out if x.startswith('STRAT_')][-36:];mid=[x for x in out[27:] if not x.startswith('STRAT_')];rem=max_len-len(fixed)-len(tail);out=fixed+mid[-max(0,rem):]+tail
 return out[:max_len]

def _semantic_family(obs,fallback):
 try:
  cur=obs.get('current') or {};me=_i(cur.get('yourIndex'));ps=cur.get('players') or []
  if len(ps)<2:return fallback
  op=ps[1-me];ids=set()
  for z in (op.get('active') or [])+(op.get('bench') or [])+(op.get('discard') or []):
   if not z:continue
   ids.add(_i(z.get('id')))
   for q in z.get('preEvolution') or []:
    if q:ids.add(_i(q.get('id')))
  for l in obs.get('logs') or []:
   if _i(l.get('playerIndex'),me)==1-me:
    for k in ('cardId','cardIdAfter','cardIdBefore','cardIdActive','cardIdBench','cardIdTarget'):
     if l.get(k) is not None:ids.add(_i(l.get(k)))
  if ids & {292,293,257,258,906,303}:return 'n_zoroark'
  if ids & {149,150,93}:return 'hydrapple'
  if ids & {402,403,404,709,918,917,710,96}:return 'grass'
  return fallback
 except Exception:return fallback

class HierStrategyMoE:
 def __init__(self,root,history,replay_mod,tok_mod):
  self.root=root;self.history=history;self.replay=replay_mod;self.tok=tok_mod;self.enabled=False;self.load_error=None
  try:
   self.cfg=json.load(open(os.path.join(root,'hier_strategy_moe_model.json'),encoding='utf-8'));z=np.load(os.path.join(root,'hier_strategy_moe_model.npz'));self.w={k:z[k].astype(np.float32) for k in z.files};self.enabled=bool(self.cfg.get('enabled',True));self.V=_i(self.cfg.get('vocab_size'),12288);self.D=_i(self.cfg.get('d_model'),64);self.H=_i(self.cfg.get('heads'),4);self.C=_i(self.cfg.get('max_context'),192);self.A=_i(self.cfg.get('max_action'),24);self.E=len(self.cfg.get('experts') or []);self.gate=self.cfg.get('gate') or {};self.support=self.cfg.get('strategy_support') or {}
  except Exception as e:self.load_error=repr(e);self.cfg={};self.w={};self.gate={};self.support={}
  self.verifier=None;self.reset()
 def reset(self):
  self.overrides=0;self.last_turn=-1;self.stats={'calls':0,'scored':0,'suggestions':0,'overrides':0,'support_block':0,'router_block':0,'margin_block':0,'family_block':0,'search_calls':0,'search_accept':0,'search_reject':0,'quality_veto':0,'errors':0,'latency_ms_sum':0.0,'search_latency_ms_sum':0.0,'expert_hits':[0]*6,'decisions':[]}
 def _block(self,x,pad,prefix):
  h=_ln(x,self.w[prefix+'norm1.weight'],self.w[prefix+'norm1.bias']);y=_mha(h,h,h,self.w[prefix+'self_attn.in_proj_weight'],self.w[prefix+'self_attn.in_proj_bias'],self.w[prefix+'self_attn.out_proj.weight'],self.w[prefix+'self_attn.out_proj.bias'],self.H,pad);x=x+y;h=_ln(x,self.w[prefix+'norm2.weight'],self.w[prefix+'norm2.bias']);ff=_gelu(h@self.w[prefix+'linear1.weight'].T+self.w[prefix+'linear1.bias']);return x+ff@self.w[prefix+'linear2.weight'].T+self.w[prefix+'linear2.bias']
 def _encode_ctx(self,cids):
  L=min(len(cids),self.C);ids=np.zeros((1,self.C),np.int64);ids[0,:L]=np.asarray(cids[-L:]);pad=np.arange(self.C)[None,:]>=L;x=self.w['emb.weight'][ids]+self.w['cpos.weight'][np.arange(self.C)][None]
  x=self._block(x,pad,'ctx.0.');x=self._block(x,pad,'ctx.1.');x=_ln(x,self.w['cln.weight'],self.w['cln.bias']);v=(~pad).astype(np.float32);cv=(x*v[:,:,None]).sum(1)/np.maximum(v.sum(1,keepdims=True),1)
  q=np.broadcast_to(self.w['expert_queries'][None,:,:],(1,self.E,self.D));q0=_ln(q,self.w['memory.lnq.weight'],self.w['memory.lnq.bias']);kv=_ln(x,self.w['memory.lnkv.weight'],self.w['memory.lnkv.bias']);y=_mha(q0,kv,kv,self.w['memory.att.in_proj_weight'],self.w['memory.att.in_proj_bias'],self.w['memory.att.out_proj.weight'],self.w['memory.att.out_proj.bias'],self.H,pad);mem=q+y;h=_ln(mem,self.w['memory.ln2.weight'],self.w['memory.ln2.bias']);ff=_gelu(h@self.w['memory.ff.0.weight'].T+self.w['memory.ff.0.bias']);mem=mem+ff@self.w['memory.ff.2.weight'].T+self.w['memory.ff.2.bias'];return x,pad,cv,mem
 def _encode_actions(self,actions):
  N=len(actions);ids=np.zeros((N,self.A),np.int64);lens=[]
  for i,a in enumerate(actions):aa=list(a[:self.A]);lens.append(len(aa));ids[i,:len(aa)]=aa
  pad=np.arange(self.A)[None,:]>=np.asarray(lens)[:,None];x=self.w['emb.weight'][ids]+self.w['apos.weight'][np.arange(self.A)][None];x=self._block(x,pad,'act.0.');x=_ln(x,self.w['aln.weight'],self.w['aln.bias']);v=(~pad).astype(np.float32);av=(x*v[:,:,None]).sum(1)/np.maximum(v.sum(1,keepdims=True),1);return av
 def _forward(self,cids,actions,tagsets=None):
  _,_,cv,mem=self._encode_ctx(cids);av=self._encode_actions(actions);N=len(actions);slog=cv@self.w['srouter.weight'].T+self.w['srouter.bias'];alog=av@self.w['arouter.weight'].T+self.w['arouter.bias'];rlog=.45*slog+alog
  if tagsets is not None:
   for ii,tg in enumerate(tagsets):
    if tg:
     allow=set(tg)
     for ee in range(self.E):
      if ee not in allow:rlog[ii,ee]=-8.0
  rw=_softmax(rlog,-1);ex=[]
  for e in range(self.E):
   mm=np.broadcast_to(mem[0,e][None,:],(N,self.D));z=np.concatenate([av,mm],-1);z=_ln(z,self.w[f'expert_ff.{e}.0.weight'],self.w[f'expert_ff.{e}.0.bias']);z=_gelu(z@self.w[f'expert_ff.{e}.1.weight'].T+self.w[f'expert_ff.{e}.1.bias']);z=z@self.w[f'expert_ff.{e}.3.weight'].T+self.w[f'expert_ff.{e}.3.bias'];ex.append(z)
  ex=np.stack(ex,1);mix=(ex*rw[:,:,None]).sum(1);cc=np.broadcast_to(cv,(N,self.D));z=np.concatenate([av,cc,mix],-1);z=_gelu(z@self.w['fuse.0.weight'].T+self.w['fuse.0.bias']);z=_ln(z,self.w['fuse.2.weight'],self.w['fuse.2.bias']);zp=z[None,:,:];pad=np.zeros((1,N),bool);zp=self._block(zp,pad,'cand.');z=_ln(zp[0],self.w['oln.weight'],self.w['oln.bias']);p=(z@self.w['policy.weight'].T+self.w['policy.bias']).reshape(-1);q=(z@self.w['quality.weight'].T+self.w['quality.bias']).reshape(-1);return p,q,rlog
 def _descs(self,obs):
  out=[];opts=((obs.get('select') or {}).get('option') or [])
  for j in range(len(opts)):
   d=self.replay.action_desc(self.history,obs,[j]);d=d[0] if d else {'index':j,'type':-1};out.append(self.tok.enrich_desc(obs,j,d))
  return out
 def _support(self,fam,turn,d):
  phase=_phase(turn);co=_coarse(d)
  # Prefer matchup-specific evidence. Back off only to cross-matchup evidence
  # for the same card/action/expert; current-state CABT search is still mandatory.
  for ff,pp in ((fam,phase),('*',phase),('*','*')):
   best=None;besttag=None
   for e in _action_tags(d,fam):
    x=self.support.get(f'{ff}|{pp}|{e}|{co}')
    if x and (best is None or (_i(x.get('games')),-float(x.get('mean_regret',999)))>(_i(best.get('games')),-float(best.get('mean_regret',999)))):
     best=x;besttag=e
   if best is not None:return best,besttag
  return None,None
 def choose(self,obs,base):
  if not self.enabled or not isinstance(base,list) or len(base)!=1:return base
  sel=obs.get('select') or {};cur=obs.get('current') or {};opts=sel.get('option') or []
  if _i(sel.get('context'),-1)!=MAIN or _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or len(opts)<2 or len(opts)>_i(self.gate.get('max_options'),24):return base
  bi=_i(base[0],-1);turn=_i(cur.get('turn'));self.stats['calls']+=1
  if not 0<=bi<len(opts):return base
  try:
   fam=_semantic_family(obs,self.replay.recognize(self.history,obs));disabled=set(self.gate.get('disable_families') or []);allowed=set(self.gate.get('allowed_families') or [])
   if fam in disabled or (allowed and fam not in allowed):self.stats['family_block']+=1;return base
   if fam=='unknown' and turn<_i(self.gate.get('unknown_min_turn'),4):self.stats['family_block']+=1;return base
   t0=time.perf_counter();descs=self._descs(obs);bt=_i(descs[bi].get('type'),-1)
   dev={PLAY,ATTACH,EVOLVE,ABILITY}
   # Two-cycle CF repair: in late mirror/Grimmsnarl states, a development action can be
   # dominated by attacking now or pivoting out.  Let the hierarchy propose those
   # cross-class candidates, but current-state CABT exact search remains mandatory.
   cross_late=(fam=='lucario' and turn>=6) or (fam=='marnie' and turn>=8)
   if bt in dev:
    pool=dev|({ATTACK,RETREAT} if cross_late else set())
    cand=[i for i,d in enumerate(descs) if i!=bi and _i(d.get('type'),-2) in pool]
   elif bt==END:cand=[i for i,d in enumerate(descs) if _i(d.get('type'),-2) in dev|{ATTACK}]
   elif bt==ATTACK:cand=[i for i,d in enumerate(descs) if i!=bi and _i(d.get('type'),-2)==ATTACK]
   else:return base
   if not cand:return base
   inds=[bi]+cand;full=self.tok.context_tokens(self.history,obs,fam);ct=_hier_tokens(full,self.C);cids=self.tok.encode(ct,self.V,self.C,True);acts=[self.tok.encode(self.tok.action_tokens(descs[i]),self.V,self.A,False) for i in inds];tagsets=[_action_tags(descs[i],fam) for i in inds];p,q,rlog=self._forward(cids,acts,tagsets);self.stats['scored']+=1;self.stats['latency_ms_sum']+=(time.perf_counter()-t0)*1000;score=p+.20*q;j=max(range(1,len(inds)),key=lambda k:(float(score[k]),float(p[k])));ci=inds[j];pm=float(p[j]-p[0]);self.stats['suggestions']+=1;tags=_action_tags(descs[ci],fam);rw=_softmax(rlog[j],-1);rmass=float(sum(rw[e] for e in tags));rec={'family':fam,'turn':turn,'base':bi,'candidate':ci,'base_desc':descs[bi],'candidate_desc':descs[ci],'policy_margin':round(pm,4),'router_mass':round(rmass,4),'expert':int(np.argmax(rw))}
   self.stats['expert_hits'][int(np.argmax(rw))]+=1
   if self.overrides>=_i(self.gate.get('max_game_overrides'),1) or turn==self.last_turn:rec['reason']='quota';self.stats['decisions'].append(rec);return base
   if pm<float(self.gate.get('min_policy_margin',.08)):self.stats['margin_block']+=1;rec['reason']='margin';self.stats['decisions'].append(rec);return base
   if rmass<float(self.gate.get('min_router_mass',.38)):self.stats['router_block']+=1;rec['reason']='router';self.stats['decisions'].append(rec);return base
   sup,etag=self._support(fam,turn,descs[ci]);need=_i(self.gate.get('unknown_min_support_games'),3) if fam=='unknown' else _i(self.gate.get('min_support_games'),2)
   if not sup or _i(sup.get('games'))<need or float(sup.get('mean_regret',999))>float(self.gate.get('max_support_mean_regret',80)):
    self.stats['support_block']+=1;rec['reason']='support';rec['support']=sup;self.stats['decisions'].append(rec);return base
   rec['support']=sup;rec['support_expert']=etag
   if self.stats['search_calls']>=_i(self.gate.get('max_game_search_calls'),3):rec['reason']='search_quota';self.stats['decisions'].append(rec);return base
   if callable(self.verifier):
    tv=time.perf_counter();self.stats['search_calls']+=1;ver=self.verifier(obs,base,[ci]);self.stats['search_latency_ms_sum']+=(time.perf_counter()-tv)*1000;ok,meta=(bool(ver[0]),ver[1]) if isinstance(ver,tuple) else (bool(ver),{});gain=float((meta or {}).get('gain',-1e9));rec['search']=meta
    if not ok or gain<float(self.gate.get('search_min_gain',45)):
     self.stats['search_reject']+=1;rec['reason']='search_reject';self.stats['decisions'].append(rec);return base
    self.stats['search_accept']+=1
   self.overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;rec['reason']='override';self.stats['decisions'].append(rec);return [ci]
  except Exception as e:
   self.stats['errors']+=1;self.stats['decisions'].append({'reason':'error','error':repr(e)[:160]});return base
 def get_stats(self):
  x=dict(self.stats);x.update({'enabled':self.enabled,'version':self.cfg.get('version'),'parameter_count':_i(self.cfg.get('parameter_count')),'load_error':self.load_error,'game_overrides':self.overrides});x['mean_latency_ms']=round(x['latency_ms_sum']/max(1,x['scored']),4);x['mean_search_latency_ms']=round(x['search_latency_ms_sum']/max(1,x['search_calls']),4);return x
