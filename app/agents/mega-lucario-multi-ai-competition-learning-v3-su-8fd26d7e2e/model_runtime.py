from __future__ import annotations
import gzip,json,math,os
from pathlib import Path
import numpy as np

class TreeDumpModel:
    def __init__(self,path):
        with gzip.open(path,'rt') as f:self.m=json.load(f)
        self.state_idx=list(map(int,self.m['state_idx']));self.state_dim=int(self.m['state_dim']);self.arches=self.m['arches'];self.phases=self.m['phases'];self.own=list(map(int,self.m['own_ids']));self.ctx_n=int(self.m.get('ctx_n',64));self.type_n=int(self.m.get('type_n',24));self.area_n=int(self.m.get('area_n',14));self.index_n=int(self.m.get('index_n',8));self.a2={x:i for i,x in enumerate(self.arches)};self.p2={x:i for i,x in enumerate(self.phases)};self.trees=[]
        for root in self.m['trees']:
            mp={}
            def walk(n):
                mp[int(n['nodeid'])]=n
                for c in n.get('children',[]):walk(c)
            walk(root);self.trees.append((root,mp))
    def avec(self,arch,phase,d):
        d=list(map(int,d))+[-1]*17;c,s,t,u,v=d[:5];area,index,player,toolidx,energyidx,count,ipa,ipi,attack,cardid,number,special=d[5:17]
        dim=len(self.arches)+len(self.phases)+self.ctx_n+self.type_n+self.area_n*2+self.index_n*2+18+55+3*len(self.own);z=[0.0]*dim;o=0
        z[o+self.a2.get(arch,0)]=1;o+=len(self.arches);z[o+self.p2.get(phase,1)]=1;o+=len(self.phases);z[o+max(0,min(self.ctx_n-1,c))]=1;o+=self.ctx_n;z[o+max(0,min(self.type_n-1,t))]=1;o+=self.type_n
        z[o+max(0,min(self.area_n-1,area+1))]=1;o+=self.area_n;z[o+max(0,min(self.area_n-1,ipa+1))]=1;o+=self.area_n;z[o+max(0,min(self.index_n-1,index+1))]=1;o+=self.index_n;z[o+max(0,min(self.index_n-1,ipi+1))]=1;o+=self.index_n
        z[o:o+18]=[s/1400.,u/1400.,v/1400.,area/12.,index/7.,player/2.,toolidx/5.,energyidx/5.,count/10.,ipa/12.,ipi/7.,attack/1400.,cardid/1400.,number/10.,special/10.,float(u==v and u>0),float(index==ipi and index>=0),float(t in (13,14))];o+=18
        for q in (s,u,v,attack,cardid):
            q=max(0,q)
            for b in range(11):z[o+b]=(q>>b)&1
            o+=11
        for q in (s,u,v):
            for j,k in enumerate(self.own):
                if q==k:z[o+j]=1
            o+=len(self.own)
        return z
    @staticmethod
    def tree(root,mp,x):
        n=root
        while 'leaf' not in n:
            f=int(str(n['split']).lstrip('f'));val=x[f]
            if val is None or (isinstance(val,float) and math.isnan(val)):nid=int(n.get('missing',n['yes']))
            else:nid=int(n['yes'] if val<float(n['split_condition']) else n['no'])
            n=mp[nid]
        return float(n['leaf'])
    def scores(self,rx,arch,phase,descs):
        st=[float(rx[i]) for i in self.state_idx];out=[]
        for d in descs:
            x=st+self.avec(arch,phase,d);out.append(sum(self.tree(r,m,x) for r,m in self.trees))
        return out

class CountModel:
    def __init__(self,path):
        with gzip.open(path,'rt') as f:self.m=json.load(f)
        self.state_idx=list(map(int,self.m['state_idx']));self.arches=self.m['arches'];self.phases=self.m['phases'];self.own=list(map(int,self.m['own_ids']));self.ctx_n=int(self.m.get('ctx_n',64));self.a2={x:i for i,x in enumerate(self.arches)};self.p2={x:i for i,x in enumerate(self.phases)};self.trees=[]
        for root in self.m['trees']:
            mp={}
            def walk(n):
                mp[int(n['nodeid'])]=n
                for c in n.get('children',[]):walk(c)
            walk(root);self.trees.append((root,mp))
    def cvec(self,arch,phase,ctx,source,n,minc,maxc,k):
        dim=len(self.arches)+len(self.phases)+self.ctx_n+11+len(self.own)+10+7+7;z=[0.0]*dim;o=0
        z[o+self.a2.get(arch,0)]=1;o+=len(self.arches);z[o+self.p2.get(phase,1)]=1;o+=len(self.phases);z[o+max(0,min(self.ctx_n-1,ctx))]=1;o+=self.ctx_n;q=max(0,int(source))
        for b in range(11):z[o+b]=(q>>b)&1
        o+=11
        for j,x in enumerate(self.own):z[o+j]=float(q==x)
        o+=len(self.own);z[o+max(0,min(9,n))]=1;o+=10;z[o+max(0,min(6,k))]=1;o+=7;z[o:o+7]=[n/20.,minc/6.,maxc/6.,k/6.,float(k==0),float(k==minc),float(k==maxc)];return z
    def pick(self,rx,arch,phase,ctx,source,n,minc,maxc):
        lo=max(0,int(minc));hi=max(lo,min(6,int(maxc)));ks=list(range(lo,hi+1));st=[float(rx[i]) for i in self.state_idx];scores=[]
        for k in ks:
            x=st+self.cvec(arch,phase,ctx,source,n,lo,hi,k);scores.append(sum(TreeDumpModel.tree(r,m,x) for r,m in self.trees))
        return ks[max(range(len(scores)),key=lambda i:scores[i])]

def zscore(a):
    if not a:return []
    m=sum(a)/len(a);v=(sum((x-m)**2 for x in a)/len(a))**.5
    return [(x-m)/(v+1e-6) for x in a]
