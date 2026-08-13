from __future__ import annotations
from typing import Any
import os

SUPPORT_PROMOTION_MODE='support_low_midlate_healthy'

PROTOTYPES = {
    'lucario': {
        ('crustle',0,'mid','9:678:333:0:ta0:te1:td0:sa0:se0:sd0'): '10:66:0:0:ta0:te0:td0:sa1:se1:sd0',
        ('crustle',4,'late','3:676:0:0:ta0:te0:td0:sa0:se0:sd0'): '3:678:0:0:ta0:te0:td0:sa0:se2:sd0',
    },
    'terabox': {
        ('lucario',0,'late','7:1182:0:0:ta0:te0:td0:sa0:se0:sd0'): '13:0:0:120:ta0:te0:td0:sa0:se0:sd0',
        ('lucario',0,'late','10:96:0:0:ta0:te0:td0:sa1:se3:sd0'): '8:16:96:0:ta1:te3:td0:sa0:se0:sd0',
    },
}

class TerminalDistillGate:
    def __init__(self,target:str):
        self.target=target;self.seen=set();self.last_override_turn=None
        self.stats={'calls':0,'recognized':{},'overrides':0,'prototype_hits':{},'errors':0}
    def reset(self):
        self.seen.clear();self.last_override_turn=None
    @staticmethod
    def _id(x:Any)->int:
        if not x:return 0
        try:return int(x.get('id',0) if isinstance(x,dict) else getattr(x,'id',0) or 0)
        except Exception:return 0
    def _observe(self,d):
        cur=d.get('current') or {};ps=cur.get('players') or []
        if len(ps)<2:return
        me=int(cur.get('yourIndex',0));op=ps[1-me]
        def add(c):
            if not c:return
            z=self._id(c)
            if z:self.seen.add(z)
            xs=(c.get('preEvolution') if isinstance(c,dict) else getattr(c,'preEvolution',[])) or []
            for q in xs:add(q)
        for zone in ('active','bench','discard','lostZone'):
            for c in op.get(zone) or []:add(c)
    def _family(self):
        s=self.seen
        if s & {344,345}:return 'crustle'
        if s & {333,677,678}:return 'lucario'
        if s & {119,120,121}:return 'dragapult'
        if s & {646,647,648,860}:return 'marnie'
        # Avoid classifying a generic Teal Mask Ogerpon alone as Terabox.
        if s & {756,108,184,230,31,272}:return 'terabox'
        return 'unknown'
    @staticmethod
    def _card(d,area,index,player):
        try:
            if area is None or index is None:return None
            area=int(area);index=int(index);cur=d.get('current') or {};ps=cur.get('players') or []
            if area==1:a=(d.get('select') or {}).get('deck') or []
            elif area==7:a=cur.get('stadium') or []
            elif area==12:a=cur.get('looking') or []
            else:a=ps[player].get({2:'hand',3:'discard',4:'active',5:'bench',6:'prize'}.get(area,'_')) or []
            return a[index] if 0<=index<len(a) else None
        except Exception:return None
    @staticmethod
    def _energy_count(c):
        if not c:return 0
        try:return len((c.get('energyCards') or c.get('energies') or []) if isinstance(c,dict) else (getattr(c,'energyCards',[]) or getattr(c,'energies',[]) or []))
        except Exception:return 0
    @staticmethod
    def _damage_bin(c):
        if not c:return 0
        try:
            hp=float(c.get('hp') or 0);mh=float(c.get('maxHp') or hp or 1);return max(0,int((mh-hp)//30))
        except Exception:return 0
    def _sig(self,d,i):
        sel=d.get('select') or {};opts=sel.get('option') or [];cur=d.get('current') or {};me=int(cur.get('yourIndex',0))
        if not 0<=i<len(opts):return None
        o=opts[i];t=int(o.get('type',-1));card=target=attack=0;ta=te=td=sa=se=sd=0
        if t==7:
            q=self._card(d,2,o.get('index'),me);card=self._id(q)
        elif t in (8,9):
            q=self._card(d,o.get('area'),o.get('index'),me);card=self._id(q)
            z=self._card(d,o.get('inPlayArea'),o.get('inPlayIndex'),me);target=self._id(z)
            ta=int(int(o.get('inPlayArea',-1))==4);te=self._energy_count(z);td=self._damage_bin(z)
        elif t in (10,11):
            q=self._card(d,o.get('area'),o.get('index'),me);card=self._id(q)
            sa=int(int(o.get('area',-1))==4);se=self._energy_count(q);sd=self._damage_bin(q)
        elif t==13:
            attack=int(o.get('attackId') or 0)
        elif t in (3,4,5,6):
            pi=int(o.get('playerIndex',me));q=self._card(d,o.get('area'),o.get('index'),pi);card=self._id(q)
            sa=int(int(o.get('area',-1))==4);se=self._energy_count(q);sd=self._damage_bin(q)
        return f'{t}:{card}:{target}:{attack}:ta{ta}:te{te}:td{td}:sa{sa}:se{se}:sd{sd}'
    def choose(self,d,base):
        self.stats['calls']+=1
        if os.environ.get('TERMINAL_DISTILL_DISABLE')=='1':return base
        try:
            if not isinstance(d,dict) or d.get('current') is None:
                self.reset();return base
            if not isinstance(base,list) or len(base)!=1 or not d.get('select'):return base
            self._observe(d);fam=self._family();self.stats['recognized'][fam]=self.stats['recognized'].get(fam,0)+1
            cur=d.get('current') or {};turn=int(cur.get('turn',0) or 0);tb='early' if turn<=3 else 'mid' if turn<=7 else 'late';ctx=int((d.get('select') or {}).get('context',-1))
            if self.last_override_turn==turn:return base
            bs=self._sig(d,int(base[0]))
            if self.target=='lucario' and fam=='crustle' and ctx==4 and bs:
                try:
                    pp=bs.split(':');bcard=int(pp[1]);benergy=int(next(x[2:] for x in pp if x.startswith('se')))
                except Exception:bcard=benergy=-1
                late=turn>=8;midlate=turn>=4
                allow=((SUPPORT_PROMOTION_MODE=='lunatone_late_healthy' and late and bcard==675 and benergy==0) or
                       (SUPPORT_PROMOTION_MODE=='lunar_low_late_healthy' and late and bcard in (675,676) and benergy<=1) or
                       (SUPPORT_PROMOTION_MODE=='support_low_late_healthy' and late and bcard in (305,675,676) and benergy<=1) or
                       (SUPPORT_PROMOTION_MODE=='support_low_midlate_healthy' and midlate and bcard in (305,675,676) and benergy<=1) or
                       (SUPPORT_PROMOTION_MODE=='duns_late_healthy' and late and bcard==305 and benergy==0))
                if allow:
                    best_i=None;best_e=-1
                    for i,_ in enumerate((d.get('select') or {}).get('option') or []):
                        ss=self._sig(d,i)
                        if not ss:continue
                        try:
                            qq=ss.split(':');cc=int(qq[1]);ee=int(next(x[2:] for x in qq if x.startswith('se')));dd=int(next(x[2:] for x in qq if x.startswith('sd')))
                        except Exception:continue
                        if cc==678 and ee>=2 and dd==0 and ee>best_e:best_i=i;best_e=ee
                    if best_i is not None:
                        self.last_override_turn=turn;self.stats['overrides']+=1;key=f'support:{SUPPORT_PROMOTION_MODE}|{fam}|{ctx}|{tb}|{bs}->678e{best_e}d0';self.stats['prototype_hits'][key]=self.stats['prototype_hits'].get(key,0)+1;return [best_i]
            want=PROTOTYPES.get(self.target,{}).get((fam,ctx,tb,bs))
            if not want:return base
            for i,_ in enumerate((d.get('select') or {}).get('option') or []):
                if self._sig(d,i)==want:
                    self.last_override_turn=turn;self.stats['overrides']+=1;key=f'{fam}|{ctx}|{tb}|{bs}->{want}';self.stats['prototype_hits'][key]=self.stats['prototype_hits'].get(key,0)+1;return [i]
            return base
        except Exception:
            self.stats['errors']+=1;return base
    def get_stats(self):return {k:(dict(v) if isinstance(v,dict) else v) for k,v in self.stats.items()}
