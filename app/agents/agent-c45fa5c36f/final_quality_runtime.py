from __future__ import annotations
import os,json
from collections import Counter
import transformer_intent_policy as tip

def _i(x,d=0):
    try:return int(x if x is not None else d)
    except Exception:return d

def _phase(t):
    t=_i(t);return 'early' if t<=3 else 'mid' if t<=8 else 'late'

class FinalGeneralizedCFGuard:
    """Precision-first final arbiter distilled from repeated exact counterfactuals.

    Runtime uses only public observation/history.  It abstains unless a semantic
    action-pair has repeated exact-CF support *and* the trained 6-layer quality
    model clears a pair-specific margin.  This layer runs after all strategic
    residuals so a validated correction cannot be overwritten downstream.
    """
    def __init__(self,root,quality_model,filename='final_quality_support.json'):
        self.root=root;self.quality=quality_model;self.cfg={};self.stats=Counter();self.game_overrides=0;self.last_turn=-1;self.last_event=None
        try:
            self.cfg=json.load(open(os.path.join(root,filename),encoding='utf8'));self.stats['loaded']+=1
        except Exception:self.stats['load_error']+=1
    def reset(self):
        self.game_overrides=0;self.last_turn=-1;self.last_event=None;self.stats['games']+=1
    def get_stats(self):
        d={'finalcf_'+str(k):int(v) for k,v in self.stats.items()};d['finalcf_game_overrides']=int(self.game_overrides);return d
    def _family(self,sem,intent):
        typ=_i(sem.get('option_type'),-1);card=_i(sem.get('card_id'));targ=_i(sem.get('target_card_id'));aid=_i(sem.get('attack_id'))
        if typ==8:
            nm={119:'dreepy',120:'drakloak',121:'dragapult',305:'dunsparce',66:'dudunsparce',112:'munkidori',140:'fezandipiti',235:'budew',1071:'meowth'}.get(targ,'other')
            # Dreepy setup is color-sensitive: manual Psychic is repeatedly
            # superior to manual Fire because Crispin can supply Fire later.
            if targ==119:
                en={2:'fire',5:'psychic',7:'dark'}.get(card)
                if en:return 'attach_dreepy_'+en
            return 'attach_'+nm
        if typ==7:
            return {1086:'poffin',1121:'ultraball',1152:'pokepad',1227:'lillie',1198:'crispin',1182:'boss',1213:'judge',1120:'hammer',1097:'stretcher',1080:'unfair',1260:'risky',119:'play_dreepy',112:'play_munkidori',140:'play_fez',305:'play_dunsparce',235:'play_budew',1071:'play_meowth'}.get(card,'play_'+str(card))
        if typ==10:return 'ability_'+{120:'drakloak',112:'munkidori',66:'dudunsparce',140:'fez'}.get(card,str(card))
        if typ==9:return 'evolve_'+{120:'drakloak',121:'dragapult',66:'dudunsparce'}.get(card,str(card))
        if typ==12:return 'retreat'
        if typ==13:return 'attack_'+str(aid)
        if typ==14:return 'end'
        return str(intent or 'invalid').split('__')[0]
    def choose(self,observation,chosen,history,matchup=None,confidence=0.0):
        self.stats['calls']+=1;self.last_event=None
        if not self.cfg or self.quality is None or not getattr(self.quality,'enabled',False):return chosen
        if not isinstance(chosen,list) or len(chosen)!=1:return chosen
        sel=observation.get('select') or {};cur=observation.get('current') or {};opts=sel.get('option') or []
        if _i(sel.get('context'),-1)!=0 or _i(sel.get('minCount'))!=1 or _i(sel.get('maxCount'))!=1 or len(opts)<2 or len(opts)>24:return chosen
        bi=_i(chosen[0],-1);turn=_i(cur.get('turn'));match=str(matchup or 'unknown')
        if not 0<=bi<len(opts):return chosen
        dec=list(getattr(history,'decisions',[]) or []);pub=list(getattr(history,'public_events',[]) or [])
        sems=[];ints=[];fams=[]
        for i in range(len(opts)):
            try:s=tip.semantic_option(observation,i);it=tip.intent_key(observation,s,dec) if s else 'invalid'
            except Exception:s={'index':i,'option_type':-1};it='invalid'
            sems.append(s);ints.append(it);fams.append(self._family(s,it))
        basefam=fams[bi];records=self.cfg.get('records') or {};eligible=[]
        for i in range(len(opts)):
            if i==bi:continue
            key=f"{match}|{_phase(turn)}|{basefam}|{fams[i]}";rec=records.get(key)
            if not rec:continue
            allowed=rec.get('allowed_turns')
            if allowed is not None and int(turn) not in {int(x) for x in allowed}:continue
            if bool(rec.get('same_target_required',False)):
                if _i(sems[i].get('target_serial'),-999)!=_i(sems[bi].get('target_serial'),-998):continue
            # Standard support is strict-positive based.  For precision-audited
            # ordering rules we instead treat tiny (<50k) branch differences as
            # near-ties rather than false negatives.  This prevents numerical /
            # short-rollout noise from blocking an action family that has repeated
            # large gains and zero observed material harm.
            if str(rec.get('precision_mode',''))=='material_safe':
                if _i(rec.get('material_harm'))>0:continue
                if _i(rec.get('material_positive'))<_i(rec.get('min_material_positive'),3):continue
                if _i(rec.get('opportunities'))<_i(rec.get('min_opportunities'),4):continue
                if float(rec.get('mean_gain',0))<float(rec.get('min_mean_gain',40000)):continue
            else:
                if _i(rec.get('positive'))<4 or _i(rec.get('opportunities'))<5:continue
                if float(rec.get('positive_rate',0))<0.75 or float(rec.get('mean_gain',0))<60000:continue
            if fams[i]=='end':continue
            eligible.append((i,rec,key))
        if not eligible:
            self.stats['support_block']+=1;return chosen
        try:
            legal=sorted(set(ints));ctx=tip.build_tokens(observation,dec,pub,match,float(confidence),legal,84)
            q=self.quality;vocab=_i(q.meta.get('vocab_size'),4096);mt=_i(q.meta.get('max_tokens'),96)
            bscore=q._score(tip.tokens_to_ids(ctx+q._cand_tokens(sems[bi],ints[bi]),vocab,mt));ranked=[]
            for i,rec,key in eligible:
                sc=q._score(tip.tokens_to_ids(ctx+q._cand_tokens(sems[i],ints[i]),vocab,mt))
                ranked.append((float(sc-bscore),float(rec.get('positive_rate',0)),float(rec.get('mean_gain',0)),i,rec,key))
            margin,pr,gain,ci,rec,key=max(ranked)
        except Exception:
            self.stats['inference_error']+=1;return chosen
        self.stats['evaluated']+=1
        if str(rec.get('precision_mode',''))=='material_safe':
            need=float(rec.get('min_model_margin',self.cfg.get('min_model_margin',1.5)))
        else:
            need=max(float(self.cfg.get('min_model_margin',1.5)),float(rec.get('min_model_margin',0.0)))
        if margin<need:
            self.stats['margin_block']+=1;return chosen
        limits=self.cfg.get('max_game_overrides_by_matchup') or {}
        limit=_i(limits.get(match),_i(self.cfg.get('max_game_overrides'),1))
        same_turn_ok=(str(rec.get('precision_mode',''))=='material_safe' and bool(rec.get('allow_same_turn',False)))
        if self.game_overrides>=limit or (turn==self.last_turn and not same_turn_ok):
            self.stats['quota_block']+=1;return chosen
        self.game_overrides+=1;self.last_turn=turn;self.stats['overrides']+=1;self.stats['override_'+match]+=1;self.stats['from_'+basefam]+=1;self.stats['to_'+fams[ci]]+=1
        self.last_event={'turn':int(turn),'matchup':match,'before':int(bi),'after':int(ci),'base_family':basefam,'alt_family':fams[ci],'margin':float(margin),'positive_rate':float(pr),'mean_gain':float(gain),'key':key}
        return [int(ci)]
