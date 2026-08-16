from __future__ import annotations
from collections import Counter
import math

FAMILY_SIGNATURES={
 'dragapult':{119,120,121,235},
 'terabox':{96,108,272,184,230,31,756},
 'lucario':{333,677,678},
 'marnie':{646,647,648,860,104},
 'crustle':{343,344,345},
}

def cid(x):
 if x is None:return 0
 return int((x.get('id') if isinstance(x,dict) else getattr(x,'id',0)) or 0)

def visible_ids(obs):
 cur=obs.get('current') or {};me=int(cur.get('yourIndex',0));ps=cur.get('players') or []
 if len(ps)<2:return set()
 op=ps[1-me];ids=set()
 def add(x):
  if not x:return
  z=cid(x)
  if z:ids.add(z)
  for q in x.get('preEvolution') or []:add(q)
 for zone in ('active','bench','discard','lostZone'):
  for x in op.get(zone) or []:add(x)
 return ids

def infer_family(obs,seen=None):
 ids=visible_ids(obs)
 if seen is not None:
  seen.update(ids);ids=set(seen)
 scores=[]
 for name,sig in FAMILY_SIGNATURES.items():
  hit=ids&sig
  if hit:scores.append((len(hit),sum(2 if x in sig else 1 for x in hit),name))
 if not scores:return 'unknown'
 scores.sort(reverse=True)
 if len(scores)>1 and scores[0][:2]==scores[1][:2]:return 'unknown'
 return scores[0][2]

def area_card(obs,area,index,player):
 try:
  if area is None or index is None:return None
  area=int(area);index=int(index)
  if area==1:a=(obs.get('select') or {}).get('deck') or []
  elif area==7:a=(obs.get('current') or {}).get('stadium') or []
  elif area==12:a=(obs.get('current') or {}).get('looking') or []
  else:a=(obs.get('current') or {}).get('players')[player].get({2:'hand',3:'discard',4:'active',5:'bench',6:'prize'}.get(area,'_')) or []
  return a[index] if 0<=index<len(a) else None
 except Exception:return None

def _enrich_subject(d,q,zone,prefix='subject'):
 if not q:return
 d[f'{prefix}_zone']=int(zone or 0);d[f'{prefix}_active']=1 if int(zone or 0)==4 else 0
 if isinstance(q,dict) and q.get('hp') is not None:
  hp=float(q.get('hp') or 0);mh=float(q.get('maxHp') or hp or 1);es=q.get('energyCards') or q.get('energies') or []
  d[f'{prefix}_energy']=min(6,len(es));d[f'{prefix}_damage_bin']=min(5,int(max(0,mh-hp)//30));d[f'{prefix}_hp_bin']=min(5,int((hp/max(1,mh))*5))

def option_desc(obs,i):
 opts=(obs.get('select') or {}).get('option') or [];actor=int((obs.get('current') or {}).get('yourIndex',0))
 if not 0<=int(i)<len(opts):return {'option_index':int(i),'type':-1}
 i=int(i);o=opts[i];t=int(o.get('type',-1));d={'option_index':i,'type':t}
 if t==7:d['card_id']=cid(area_card(obs,2,o.get('index'),actor))
 elif t in (8,9):
  d['card_id']=cid(area_card(obs,o.get('area'),o.get('index'),actor));q=area_card(obs,o.get('inPlayArea'),o.get('inPlayIndex'),actor);d['target_id']=cid(q);_enrich_subject(d,q,o.get('inPlayArea'),'target')
 elif t in (10,11):
  q=area_card(obs,o.get('area'),o.get('index'),actor);d['card_id']=cid(q);_enrich_subject(d,q,o.get('area'),'subject')
 elif t==13:d['attack_id']=int(o.get('attackId') or 0)
 elif t in (3,4,5,6):
  q=area_card(obs,o.get('area'),o.get('index'),int(o.get('playerIndex',actor)));d['card_id']=cid(q);_enrich_subject(d,q,o.get('area'),'subject')
 return d

def candidates(obs,base,maxn=6):
 sel=obs.get('select') or {};opts=sel.get('option') or [];mn=int(sel.get('minCount') or 0);mx=int(sel.get('maxCount') or 0);bt=tuple(int(x) for x in base);out=[bt]
 if mx==1:
  seen=set();groups=[]
  for i in range(len(opts)):
   d=option_desc(obs,i);k=(d.get('type'),d.get('card_id'),d.get('target_id'),d.get('attack_id'),d.get('target_active'),d.get('target_energy'),d.get('target_damage_bin'),d.get('subject_active'),d.get('subject_energy'),d.get('subject_damage_bin'))
   if k not in seen:seen.add(k);groups.append((i,))
  groups.sort(key=lambda x:(0 if int(opts[x[0]].get('type',-1)) in (13,9,12) else 1,x[0]))
  for g in groups:
   if g not in out:out.append(g)
   if len(out)>=maxn:break
 else:
  chosen=set(bt);un=[i for i in range(len(opts)) if i not in chosen]
  for pos in range(len(bt)):
   for j in un:
    z=list(bt);z[pos]=j
    if len(set(z))==len(z) and mn<=len(z)<=mx:
     z=tuple(sorted(z))
     if z not in out:out.append(z)
     if len(out)>=maxn:break
   if len(out)>=maxn:break
  if mn==0 and () not in out:out.append(())
 return [list(x) for x in out[:maxn]]

def desc_sig(descs):
 def one(d):
  return f"{int(d.get('type',-1))}:{int(d.get('card_id',0) or 0)}:{int(d.get('target_id',0) or 0)}:{int(d.get('attack_id',0) or 0)}:ta{int(d.get('target_active',0) or 0)}:te{int(d.get('target_energy',0) or 0)}:td{int(d.get('target_damage_bin',0) or 0)}:sa{int(d.get('subject_active',0) or 0)}:se{int(d.get('subject_energy',0) or 0)}:sd{int(d.get('subject_damage_bin',0) or 0)}"
 return '+'.join(one(d) for d in sorted(descs,key=one)) or 'empty'

def support_key(family,context,base_desc,cand_desc):return f'{family}|{int(context)}|{desc_sig(base_desc)}|{desc_sig(cand_desc)}'

def _poke_features(f,p,pfx,active=False):
 if not p:return
 z=cid(p);hp=float(p.get('hp') or 0);mh=float(p.get('maxHp') or hp or 1);es=p.get('energyCards') or p.get('energies') or []
 f[f'{pfx}_field={z}']=f.get(f'{pfx}_field={z}',0.0)+1.0
 if active:f[f'{pfx}_active={z}']=1.0
 f[f'{pfx}_hpfrac_{z}']=hp/max(1.0,mh);f[f'{pfx}_energy_{z}']=min(6,len(es))/6.0
 for e in es:
  f[f'{pfx}_energyid={cid(e) if isinstance(e,dict) else int(e or 0)}']=f.get(f'{pfx}_energyid={cid(e) if isinstance(e,dict) else int(e or 0)}',0.0)+1/6

def state_features(obs,family=None):
 cur=obs.get('current') or {};me=int(cur.get('yourIndex',0));ps=cur.get('players') or [{},{}];a=ps[me];b=ps[1-me];sel=obs.get('select') or {};turn=int(cur.get('turn') or 0);ctx=int(sel.get('context',-1));effect=int(((sel.get('effect') or {}).get('id')) or 0);family=family or infer_family(obs)
 tb='early' if turn<=3 else 'mid' if turn<=6 else 'late'
 ap=len(a.get('prize') or []);bp=len(b.get('prize') or [])
 f={'bias':1.0,f'family={family}':1.0,f'context={ctx}':1.0,f'turnbin={tb}':1.0,'turn_norm':min(turn,20)/20.0,'own_prize_norm':ap/6.0,'opp_prize_norm':bp/6.0,'prize_lead_norm':(bp-ap)/6.0,'own_hand_norm':min(int(a.get('handCount') or 0),15)/15.0,'opp_hand_norm':min(int(b.get('handCount') or 0),15)/15.0,'own_deck_norm':min(int(a.get('deckCount') or 0),60)/60.0,'opp_deck_norm':min(int(b.get('deckCount') or 0),60)/60.0,'own_bench_norm':len(a.get('bench') or [])/5.0,'opp_bench_norm':len(b.get('bench') or [])/5.0,'supporter_played':1.0 if cur.get('supporterPlayed') else 0.0,'energy_attached':1.0 if cur.get('energyAttached') else 0.0,'retreated':1.0 if cur.get('retreated') else 0.0}
 if effect:f[f'effect={effect}']=1.0
 for p in a.get('active') or []:_poke_features(f,p,'own',True)
 for p in a.get('bench') or []:_poke_features(f,p,'own',False)
 for p in b.get('active') or []:_poke_features(f,p,'opp',True)
 for p in b.get('bench') or []:_poke_features(f,p,'opp',False)
 for s in cur.get('stadium') or []:f[f'stadium={cid(s)}']=1.0
 return f

def _action_tokens(descs,pfx):
 out={};parts=[]
 for d in descs:
  t=int(d.get('type',-1));c=int(d.get('card_id',0) or 0);g=int(d.get('target_id',0) or 0);a=int(d.get('attack_id',0) or 0)
  out[f'{pfx}_type={t}']=out.get(f'{pfx}_type={t}',0.0)+1.0
  if c:out[f'{pfx}_card={c}']=out.get(f'{pfx}_card={c}',0.0)+1.0
  if g:out[f'{pfx}_target={g}']=out.get(f'{pfx}_target={g}',0.0)+1.0
  if a:out[f'{pfx}_attack={a}']=out.get(f'{pfx}_attack={a}',0.0)+1.0
  for k in ('target_active','target_energy','target_damage_bin','target_hp_bin','subject_active','subject_energy','subject_damage_bin','subject_hp_bin','subject_zone'):
   if d.get(k) is not None:out[f'{pfx}_{k}={int(d.get(k) or 0)}']=out.get(f'{pfx}_{k}={int(d.get(k) or 0)}',0.0)+1.0
  parts.append(desc_sig([d]))
 out[f'{pfx}_sig={"+".join(sorted(parts)) or "empty"}']=1.0
 return out

def pair_features(obs,base_desc,cand_desc,family=None,history_features=None):
 family=family or infer_family(obs);f=state_features(obs,family);sel=obs.get('select') or {};ctx=int(sel.get('context',-1));turn=int((obs.get('current') or {}).get('turn') or 0);tb='early' if turn<=3 else 'mid' if turn<=6 else 'late'
 if history_features:f.update({str(k):float(v) for k,v in history_features.items() if isinstance(v,(int,float))})
 bt=_action_tokens(base_desc,'base');ct=_action_tokens(cand_desc,'cand');f.update(bt);f.update(ct)
 bs=desc_sig(base_desc);cs=desc_sig(cand_desc)
 f[f'pair={bs}->{cs}']=1.0;f[f'family_pair={family}|{bs}->{cs}']=1.0;f[f'context_pair={ctx}|{bs}->{cs}']=1.0;f[f'family_candsig={family}|{cs}']=1.0;f[f'turn_candsig={tb}|{cs}']=1.0
 for d in cand_desc:
  t=int(d.get('type',-1));c=int(d.get('card_id',0) or 0);g=int(d.get('target_id',0) or 0);a=int(d.get('attack_id',0) or 0)
  f[f'family_candtype={family}|{t}']=1.0;f[f'context_candtype={ctx}|{t}']=1.0;f[f'turn_candtype={tb}|{t}']=1.0
  if c:f[f'family_candcard={family}|{c}']=1.0
  if g:f[f'family_candtarget={family}|{g}']=1.0
  if a:f[f'family_candattack={family}|{a}']=1.0
 return f
