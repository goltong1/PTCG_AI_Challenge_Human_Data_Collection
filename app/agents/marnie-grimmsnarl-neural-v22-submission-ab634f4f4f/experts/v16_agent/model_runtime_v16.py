from __future__ import annotations
import gzip,json,os,math

class TreeModel:
    def __init__(self,path):
        with gzip.open(path,'rt') as f:self.m=json.load(f)
        self.si=list(map(int,self.m['state_idx']));self.arch={x:i for i,x in enumerate(self.m['arches'])};self.phase={x:i for i,x in enumerate(self.m['phases'])};self.own=list(map(int,self.m['own_ids']));self.ctx=int(self.m['ctx_n']);self.typ=int(self.m['type_n']);self.area=int(self.m['area_n']);self.index=int(self.m['index_n']);self.arches=self.m['arches'];self.phases=self.m['phases'];self.trees=[]
        for tree in self.m['trees']:
            mp={}
            def walk(n):
                mp[int(n['nodeid'])]=n
                for c in n.get('children',[]):walk(c)
            walk(tree);self.trees.append((tree,mp))
    def avec(self,arch,phase,d):
        d=list(map(int,d))+[-1]*17;c,s,t,u,v=d[:5];area,index,player,toolidx,energyidx,count,ipa,ipi,attack,cardid,number,special=d[5:17]
        dim=len(self.arches)+len(self.phases)+self.ctx+self.typ+self.area*2+self.index*2+17+55+3*len(self.own);z=[0.0]*dim;o=0
        z[o+self.arch.get(arch,0)]=1;o+=len(self.arches);z[o+self.phase.get(phase,1)]=1;o+=len(self.phases);z[o+max(0,min(self.ctx-1,c))]=1;o+=self.ctx;z[o+max(0,min(self.typ-1,t))]=1;o+=self.typ
        z[o+max(0,min(self.area-1,area+1))]=1;o+=self.area;z[o+max(0,min(self.area-1,ipa+1))]=1;o+=self.area;z[o+max(0,min(self.index-1,index+1))]=1;o+=self.index;z[o+max(0,min(self.index-1,ipi+1))]=1;o+=self.index
        z[o:o+17]=[s/1400.,u/1400.,v/1400.,area/10.,index/6.,player/2.,toolidx/5.,energyidx/5.,count/10.,ipa/10.,ipi/6.,attack/1400.,cardid/1400.,number/10.,special/10.,float(u==v and u>0),float(index==ipi and index>=0)];o+=17
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
    def _score_tree(root,mp,x):
        n=root
        while 'leaf' not in n:
            f=int(str(n['split']).lstrip('f'));v=x[f];cond=float(n['split_condition']);nid=int(n['yes'] if v<cond else n['no']);n=mp[nid]
        return float(n['leaf'])
    def scores(self,rx,arch,phase,descs):
        st=[float(rx[i]) for i in self.si];out=[]
        for d in descs:
            x=st+self.avec(arch,phase,d);out.append(sum(self._score_tree(r,m,x) for r,m in self.trees))
        return out

def _z(v):
    if not v:return []
    m=sum(v)/len(v);s=(sum((x-m)*(x-m) for x in v)/max(1,len(v)))**.5
    if s<1e-8:return [0.0]*len(v)
    return [(x-m)/s for x in v]

class Ensemble:
    def __init__(self,base_dir='.'):
        self.models={};names=['generic','lucario','dragapult','alakazam','archaludon','hydrapple','marnie','movement','handselect']
        for n in names:
            p=os.path.join(base_dir,f'{n}_v16.json.gz')
            if not os.path.exists(p):p=os.path.join('/kaggle_simulations/agent',f'{n}_v16.json.gz')
            if os.path.exists(p):self.models[n]=TreeModel(p)
    def scores(self,rx,arch,phase,descs):
        g=self.models['generic'].scores(rx,arch,phase,descs);gz=_z(g)
        sp=self.models.get(arch)
        if sp is None:return gz
        sz=_z(sp.scores(rx,arch,phase,descs))
        # Learned ensemble: broad, opponent-distribution, and selection-context rankers.
        spec_weight=3.0 if arch=='dragapult' else 0.72
        out=[a+spec_weight*b for a,b in zip(gz,sz)]
        ctx=int(descs[0][0]) if descs else -1
        aux=None;weight=0.0
        if ctx in (3,4,5,21,22) and 'movement' in self.models:
            aux=self.models['movement'];weight=1.0
        elif arch in ('archaludon','hydrapple') and ctx in (7,8,9,10,12,29) and 'handselect' in self.models:
            aux=self.models['handselect'];weight=1.5
        if aux is not None:
            az=_z(aux.scores(rx,arch,phase,descs));out=[a+weight*b for a,b in zip(out,az)]
        return out
