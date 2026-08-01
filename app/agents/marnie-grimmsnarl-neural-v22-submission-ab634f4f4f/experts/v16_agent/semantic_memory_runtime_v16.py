from __future__ import annotations
import gzip,json,os,hashlib
class SemanticPolicyMemory:
    def __init__(self,base_dir='.'):
        p=os.path.join(base_dir,'semantic_policy_memory_v16.json.gz')
        if not os.path.exists(p):p='/kaggle_simulations/agent/semantic_policy_memory_v16.json.gz'
        with gzip.open(p,'rt') as f:self.m=json.load(f)
        self.levels=[list(map(int,x)) for x in self.m['levels']];self.tabs=self.m['tables']
    @staticmethod
    def q(i,v):
        v=float(v)
        if i<66:return int(round(v*2)) if i in (0,7,8,34,35,36,37,53,54,55,56) else int(round(v))
        if 104<=i<416:return int(round(v*5))
        return int(round(v))
    @staticmethod
    def sem(d):
        d=list(map(int,d))+[-1]*17;c,s,t,u,v=d[:5];area,index,player,toolidx,energyidx,count,ipa,ipi,attack,cardid,number,special=d[5:17]
        return (c,s,t,u,v,area,player,count,ipa,attack,cardid,number,special)
    @staticmethod
    def skey(x):return ','.join(map(str,x))
    def option_signature(self,opts):return ';'.join(sorted(self.skey(self.sem(d)) for d in opts))
    def key(self,rx,arch,phase,opts,lev):
        vals=','.join(str(self.q(i,rx[i])) for i in self.levels[lev]);ctx=int(opts[0][0]) if opts else 0
        raw=f'{arch}|{phase}|{ctx}|{self.option_signature(opts)}|{vals}'
        return hashlib.blake2b(raw.encode(),digest_size=10).hexdigest()
    def recall_semantics(self,rx,arch,phase,opts):
        if len(opts)<2:return None
        for lev in range(len(self.levels)-1,len(self.levels)-2,-1):
            z=self.tabs[lev].get(self.key(rx,arch,phase,opts,lev))
            if not z:continue
            raw,sup,conf=z
            try:
                ss=[tuple(map(int,x.split(','))) for x in raw.split(';') if x]
                return ss
            except Exception:return None
        return None
    def map_indices(self,semantic_selection,opts,scores=None):
        if not semantic_selection:return None
        sems=[self.sem(d) for d in opts];used=set();out=[]
        for wanted in semantic_selection:
            cand=[i for i,x in enumerate(sems) if x==tuple(wanted) and i not in used]
            if not cand:return None
            i=max(cand,key=lambda j:scores[j]) if scores is not None and len(cand)>1 else cand[0]
            out.append(i);used.add(i)
        return out
