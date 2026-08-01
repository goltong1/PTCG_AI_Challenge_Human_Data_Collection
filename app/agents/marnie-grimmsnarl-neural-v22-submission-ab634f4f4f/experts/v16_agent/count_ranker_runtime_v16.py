from __future__ import annotations
import gzip,json,os
class CountRanker:
    def __init__(self,base_dir='.'):
        p=os.path.join(base_dir,'count_ranker_v16.json.gz')
        if not os.path.exists(p):p='/kaggle_simulations/agent/count_ranker_v16.json.gz'
        with gzip.open(p,'rt') as f:self.m=json.load(f)
        self.si=list(map(int,self.m['state_idx']));self.arch={x:i for i,x in enumerate(self.m['arches'])};self.phase={x:i for i,x in enumerate(self.m['phases'])};self.own=list(map(int,self.m['own_ids']));self.ctx=int(self.m['ctx_n']);self.nmax=int(self.m['nmax']);self.arches=self.m['arches'];self.phases=self.m['phases'];self.trees=[]
        for tree in self.m['trees']:
            mp={}
            def walk(n):
                mp[int(n['nodeid'])]=n
                for c in n.get('children',[]):walk(c)
            walk(tree);self.trees.append((tree,mp))
    def cvec(self,arch,phase,ctx,source,n,k):
        dim=len(self.arches)+len(self.phases)+self.ctx+11+len(self.own)+10+self.nmax+4;z=[0.0]*dim;o=0
        z[o+self.arch.get(arch,0)]=1;o+=len(self.arches);z[o+self.phase.get(phase,1)]=1;o+=len(self.phases);z[o+max(0,min(self.ctx-1,int(ctx)))]=1;o+=self.ctx
        q=max(0,int(source))
        for b in range(11):z[o+b]=(q>>b)&1
        o+=11
        for j,x in enumerate(self.own):z[o+j]=float(q==x)
        o+=len(self.own);z[o+max(0,min(9,int(n)))]=1;o+=10;z[o+max(0,min(self.nmax-1,int(k)))]=1;o+=self.nmax
        z[o:o+4]=[float(n)/10.,float(k)/6.,float(k==0),float(k==n)];return z
    @staticmethod
    def score_tree(root,mp,x):
        node=root
        while 'leaf' not in node:
            f=int(str(node['split']).lstrip('f'));v=x[f];cond=float(node['split_condition']);node=mp[int(node['yes'] if v<cond else node['no'])]
        return float(node['leaf'])
    def predict(self,rx,arch,phase,ctx,source,n,minc,maxc):
        minc=max(0,int(minc));maxc=max(minc,min(int(n),int(maxc)))
        if minc==maxc:return minc
        if minc>6:return minc
        ks=list(range(minc,min(maxc,6)+1))
        st=[float(rx[i]) for i in self.si];best=ks[0];bs=None
        for k in ks:
            x=st+self.cvec(arch,phase,ctx,source,n,k);sc=sum(self.score_tree(r,m,x) for r,m in self.trees)
            if bs is None or sc>bs:best=k;bs=sc
        return best
