from __future__ import annotations
import gzip,json,os,hashlib
class PolicyMemory:
    def __init__(self,base_dir='.'):
        p=os.path.join(base_dir,'policy_memory_v16.json.gz')
        if not os.path.exists(p):p='/kaggle_simulations/agent/policy_memory_v16.json.gz'
        with gzip.open(p,'rt') as f:self.m=json.load(f)
        self.levels=[list(map(int,x)) for x in self.m['levels']];self.tabs=self.m['tables']
    @staticmethod
    def q(i,v):
        v=float(v)
        if i<66:return int(round(v*2)) if i in (0,7,8,34,35,36,37,53,54,55,56) else int(round(v))
        if 104<=i<416:return int(round(v*5))
        return int(round(v))
    def key(self,rx,arch,phase,options,lev):
        vals=','.join(str(self.q(i,rx[i])) for i in self.levels[lev]);opts=';'.join(','.join(map(str,map(int,d))) for d in options);ctx=int(options[0][0]) if options else 0
        return hashlib.blake2b(f'{arch}|{phase}|{ctx}|{opts}|{vals}'.encode(),digest_size=8).hexdigest()
    def recall(self,rx,arch,phase,options):
        if len(options)<2:return None
        for lev in range(len(self.levels)-1,-1,-1):
            z=self.tabs[lev].get(self.key(rx,arch,phase,options,lev))
            if not z:continue
            sel,sup,conf=z
            need=1.0 if lev==len(self.levels)-1 and float(conf)>=.995 else 2.5 if lev>=3 else 4.0
            if float(sup)>=need and float(conf)>=.75:
                try:return [int(x) for x in sel.split(',') if x!='']
                except Exception:return None
        return None
class CountModel:
    def __init__(self,base_dir='.'):
        p=os.path.join(base_dir,'count_model_v16.json.gz')
        if not os.path.exists(p):p='/kaggle_simulations/agent/count_model_v16.json.gz'
        with gzip.open(p,'rt') as f:self.m=json.load(f)
    def predict(self,arch,phase,context,source,n,minc,maxc):
        keys=[f'{arch}|{phase}|{context}|{source}|{n}',f'{arch}|{context}|{source}|{n}',f'{phase}|{context}|{source}|{n}',f'{context}|{source}|{n}']
        for tab,k in zip(self.m['tables'],keys):
            z=tab.get(k)
            if z and float(z[2])>=.55:return max(minc,min(maxc,int(z[0])))
        return minc if minc==maxc else max(minc,min(maxc,1 if minc==0 else minc))
