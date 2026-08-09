from __future__ import annotations
import collections, math
IDS=[7,16,104,112,646,647,648,860,1079,1080,1086,1097,1122,1137,1152,1182,1219,1227,1231,1259]
DECK=collections.Counter({16:1,7:9,104:2,112:4,646:4,647:3,648:3,860:2,1079:3,1080:1,1086:4,1097:3,1122:1,1137:1,1152:4,1182:2,1219:4,1227:4,1231:1,1259:4})
IMP,MORG,GRIM,SNOR,FROS,MUNK=646,647,648,860,104,112
DARK,PRISM,CANDY,LILLIE=7,16,1079,1227
SEARCH={1086,1097,1122,1152,1219,1231}

def ids(cards):
 out=[]
 for x in cards or []:
  if x is None: continue
  if isinstance(x,dict): out.append(int(x.get('id') or 0))
  else: out.append(int(getattr(x,'id',0) or 0))
 return out

def pokemon_id(x):
 if x is None:return 0
 return int((x.get('id') if isinstance(x,dict) else getattr(x,'id',0)) or 0)

def energies(x):
 if x is None:return []
 v=x.get('energyCards') if isinstance(x,dict) else getattr(x,'energyCards',[])
 return ids(v)

def preevo(x):
 if x is None:return []
 v=x.get('preEvolution') if isinstance(x,dict) else getattr(x,'preEvolution',[])
 return ids(v)

def board_counter(p):
 b=collections.Counter();e=collections.Counter()
 for x in list(p.get('active') or [])+list(p.get('bench') or []):
  if not x:continue
  b[pokemon_id(x)]+=1;b.update(preevo(x));e.update(energies(x))
 return b,e

def feature_from_current(c,seat,opp='other',hand_override=None):
 ps=c.get('players') or [];p=ps[seat];q=ps[1-seat]
 h=collections.Counter(ids(p.get('hand') or [])) if hand_override is None else (collections.Counter({int(k):float(v) for k,v in hand_override.items()}) if hasattr(hand_override,'items') else collections.Counter(hand_override))
 b,e=board_counter(p);d=collections.Counter(ids(p.get('discard') or []));known=h+b+d+e;rem=DECK.copy()
 for k,v in known.items():rem[k]=max(0,rem[k]-v)
 x=[];names=[]
 def add(n,v):names.append(n);x.append(float(v))
 add('bias',1);add('turn',min(20,int(c.get('turn') or 0))/10);add('hand_n',sum(h.values())/10);add('deck_n',int(p.get('deckCount') or 0)/60);add('my_prize',len(p.get('prize') or [])/6);add('op_prize',len(q.get('prize') or [])/6);add('bench_n',len([z for z in p.get('bench') or [] if z])/5);add('supporter_used',int(bool(c.get('supporterPlayed'))));add('energy_used',int(bool(c.get('energyAttached'))))
 for cid in IDS:
  add(f'h{cid}',min(4,h[cid])/4);add(f'b{cid}',min(4,b[cid])/4);add(f'd{cid}',min(4,d[cid])/4);add(f'r{cid}',min(4,rem[cid])/4)
 ready=0
 for z in list(p.get('active') or [])+list(p.get('bench') or []):
  if pokemon_id(z)==GRIM and len(energies(z))>=1:ready+=1
 add('grim_ready',min(2,ready)/2);add('grim_count',min(3,b[GRIM])/3)
 add('candy_route',int(b[IMP]>0 and h[CANDY]>0 and h[GRIM]>0));add('stage1_route',int(b[MORG]>0 and h[GRIM]>0));add('fros_route',int(b[SNOR]>0 and h[FROS]>0));add('munk_energy',int(b[MUNK]>0 and (h[DARK]+h[PRISM]+e[DARK]+e[PRISM]>0)))
 add('backup_routes',min(4,int(b[IMP]>0)+int(b[MORG]>0)+h[GRIM]+h[CANDY])/4);add('search_density',sum(rem[z] for z in SEARCH)/max(1,sum(rem.values())));add('energy_density',(rem[DARK]+rem[PRISM])/max(1,sum(rem.values())));add('dead_duplicate',sum(max(0,h[z]-1) for z in [GRIM,CANDY,FROS,MUNK,LILLIE])/5)
 add('hand_grim_package',min(1,(h[GRIM]+h[CANDY]+h[MORG])/3));add('hand_support_package',min(1,(h[FROS]+h[MUNK]+h[DARK]+h[PRISM])/4));add('recovery_access',min(1,(h[1097]+h[1231]+h[1080])/2))
 for o in ['marnie','dragapult','lucario','alakazam','archaludon','crustle','other']:add('opp_'+o,int(opp==o))
 return x,names,rem

def board_quality(p):
 h=collections.Counter(ids(p.get('hand') or []));b,e=board_counter(p);ready=0
 for z in list(p.get('active') or [])+list(p.get('bench') or []):
  if pokemon_id(z)==GRIM and len(energies(z))>=1:ready+=1
 primary=min(1,ready);backup=min(1,max(0,ready-1)+int(b[MORG]>0 and h[GRIM]>0)+int(b[IMP]>0 and h[CANDY]>0 and h[GRIM]>0));support=.5*int(b[FROS]>0)+.45*int(b[MUNK]>0 and (e[DARK]+e[PRISM]+h[DARK]+h[PRISM]>0));routes=min(1.5,.3*(h[GRIM]+h[CANDY]+h[MORG]+h[IMP]+h[1097]));return 1.25*primary+.95*backup+support+.25*routes

def classify(deck_ids):
 s=set(deck_ids)
 if 648 in s:return 'marnie'
 if 121 in s:return 'dragapult'
 if 678 in s:return 'lucario'
 if 743 in s or 245 in s:return 'alakazam'
 if 354 in s:return 'archaludon'
 if 72 in s:return 'crustle'
 return 'other'
