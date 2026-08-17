"""Persistent public-history encoder for the Lucario policy.

The CABT observation only contains logs produced since the previous selection.
This module retains the complete public event stream for the current game, the
agent's emitted decisions, and the current public/private-to-us state.  It never
reads hidden opponent cards: opponent hand identity is represented by handCount
plus the lower-bound set of cards that public movements prove are still known.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any


LOG_FIELDS=(
    'type','playerIndex','hasBasicPokemon','cardId','serial','fromArea','toArea',
    'cardIdActive','serialActive','cardIdBench','serialBench','cardIdBefore',
    'serialBefore','cardIdAfter','serialAfter','cardIdTarget','serialTarget',
    'attackId','value','putDamageCounter','isRecover','head','result','reason',
)

# cg.api integer values.  Keeping integers here avoids another module import and
# makes this runtime safe under Kaggle raw-exec loading.
LOG_SHUFFLE=0;LOG_TURN_START=2;LOG_TURN_END=3;LOG_DRAW=4;LOG_DRAW_REVERSE=5
LOG_MOVE=6;LOG_MOVE_REVERSE=7;LOG_SWITCH=8;LOG_CHANGE=9;LOG_PLAY=10
LOG_ATTACH=11;LOG_EVOLVE=12;LOG_DEVOLVE=13;LOG_MOVE_ATTACHED=14
LOG_ATTACK=15;LOG_HP_CHANGE=16;LOG_RESULT=23
AREA_DECK=1;AREA_HAND=2;AREA_DISCARD=3;AREA_ACTIVE=4;AREA_BENCH=5
OPT_PLAY=7;OPT_ATTACH=8;OPT_EVOLVE=9;OPT_ABILITY=10;OPT_RETREAT=12
OPT_ATTACK=13;OPT_END=14
CTX_MAIN=0;CTX_SWITCH=3;CTX_TO_ACTIVE=4


def _int(value:Any,default:int=0)->int:
    try:return int(value if value is not None else default)
    except Exception:return default


def _card_id(card:Any)->int:
    if not card:return 0
    return _int(card.get('id',0) if isinstance(card,dict) else getattr(card,'id',0),0)


def _serial(card:Any)->int:
    if not card:return 0
    return _int(card.get('serial',0) if isinstance(card,dict) else getattr(card,'serial',0),0)


def _norm_log(log:Any)->dict:
    out={}
    for key in LOG_FIELDS:
        value=log.get(key) if isinstance(log,dict) else getattr(log,key,None)
        if value is not None:out[key]=value
    return out


def _player(current:dict,index:int)->dict:
    players=current.get('players') or []
    return players[index] if 0<=index<len(players) and isinstance(players[index],dict) else {}


def _zone(current:dict,select:dict,area:Any,index:Any,player:int):
    try:
        area=_int(area,-1);index=_int(index,-1)
        if area==AREA_DECK:items=select.get('deck') or []
        elif area==7:items=current.get('stadium') or []
        elif area==12:items=current.get('looking') or []
        else:items=_player(current,player).get({2:'hand',3:'discard',4:'active',5:'bench',6:'prize'}.get(area,'_')) or []
        return items[index] if 0<=index<len(items) else None
    except Exception:return None


def _pokemon_state(card:Any)->dict:
    if not card:return {}
    if not isinstance(card,dict):
        return {
            'id':_card_id(card),'serial':_serial(card),'hp':_int(getattr(card,'hp',0)),
            'maxHp':_int(getattr(card,'maxHp',0)),
            'energies':[_card_id(x) or _int(x) for x in (getattr(card,'energyCards',None) or getattr(card,'energies',None) or [])],
            'tools':[_card_id(x) for x in (getattr(card,'tools',None) or [])],
            'preEvolution':[_card_id(x) for x in (getattr(card,'preEvolution',None) or [])],
        }
    return {
        'id':_card_id(card),'serial':_serial(card),'hp':_int(card.get('hp')),
        'maxHp':_int(card.get('maxHp')),
        'energies':[_card_id(x) or _int(x) for x in (card.get('energyCards') or card.get('energies') or [])],
        'tools':[_card_id(x) for x in (card.get('tools') or [])],
        'preEvolution':[_card_id(x) for x in (card.get('preEvolution') or [])],
    }


class HistoryContext:
    """Full-game public event memory plus a bounded sequence feature encoder."""
    def __init__(self):self.reset()

    def reset(self):
        self.me_index=None
        self.events=[]
        self.decisions=[]
        self.event_counts=Counter()
        self.turn_event_counts=Counter()
        self.card_events=Counter()
        self.attack_events=Counter()
        self.revealed={0:set(),1:set()}
        self.known_hand={0:{},1:{}}
        self.hand_timeline=[]
        self.last_turn=-1
        self.last_obs_key=None
        self.last_state={}
        self.last_input={}
        self.pending_attack=None
        self.attack_damage=Counter()
        self.stats={'observations':0,'events':0,'decisions':0,'duplicate_observations':0,'known_opponent_hand_peak':0}

    def _observation_key(self,obs:dict)->str:
        cur=obs.get('current') or {};sel=obs.get('select') or {}
        slim={
            'turn':cur.get('turn'),'turnActionCount':cur.get('turnActionCount'),
            'yourIndex':cur.get('yourIndex'),'context':sel.get('context'),
            'logs':[_norm_log(x) for x in (obs.get('logs') or [])],
            'options':sel.get('option') or [],
            'hands':[[(_card_id(c),_serial(c)) for c in ((_player(cur,i).get('hand') or []))] for i in range(2)],
        }
        raw=json.dumps(slim,sort_keys=True,separators=(',',':'),default=str).encode('utf-8')
        return hashlib.sha1(raw).hexdigest()

    def _remember_card(self,player:int,card_id:int):
        if player in (0,1) and card_id:self.revealed[player].add(int(card_id))

    def _known_add(self,player:int,serial:int,card_id:int):
        if player not in (0,1) or not serial or not card_id:return
        self.known_hand[player][int(serial)]=int(card_id)

    def _known_remove(self,player:int,serial:int=0,card_id:int=0):
        if player not in (0,1):return
        known=self.known_hand[player]
        if serial and serial in known:known.pop(serial,None);return
        if card_id:
            hit=next((s for s,c in known.items() if c==card_id),None)
            if hit is not None:known.pop(hit,None)

    def _trim_known_hand(self,player:int,count:int):
        known=self.known_hand[player]
        while len(known)>max(0,count):known.pop(next(iter(known)),None)

    def _ingest(self,log:dict,turn:int,action_count:int):
        event=dict(log);event['historyIndex']=len(self.events);event['turn']=turn;event['turnActionCount']=action_count
        self.events.append(event);self.stats['events']+=1
        typ=_int(log.get('type'),-1);player=_int(log.get('playerIndex'),-1)
        self.event_counts[(player,typ)]+=1;self.turn_event_counts[(turn,player,typ)]+=1
        for key in ('cardId','cardIdActive','cardIdBench','cardIdBefore','cardIdAfter','cardIdTarget'):
            cid=_int(log.get(key),0)
            if cid:self._remember_card(player,cid)
        cid=_int(log.get('cardId'),0);serial=_int(log.get('serial'),0)
        if cid:self.card_events[(player,typ,cid)]+=1
        aid=_int(log.get('attackId'),0)
        if aid:self.attack_events[(player,aid)]+=1

        if typ==LOG_MOVE:
            fr=_int(log.get('fromArea'),-1);to=_int(log.get('toArea'),-1)
            if fr==AREA_HAND:self._known_remove(player,serial,cid)
            if to==AREA_HAND:self._known_add(player,serial,cid)
        elif typ in {LOG_PLAY,LOG_ATTACH,LOG_EVOLVE}:
            self._known_remove(player,serial,cid)
        elif typ==LOG_MOVE_REVERSE and _int(log.get('fromArea'),-1)==AREA_HAND:
            # Identity is hidden; retain only a conservative lower bound.
            if self.known_hand.get(player):self.known_hand[player].pop(next(iter(self.known_hand[player])),None)

        if typ==LOG_ATTACK:
            self.pending_attack={'player':player,'card_id':cid,'serial':serial,'attack_id':aid,'turn':turn,'damage':0}
        elif typ==LOG_HP_CHANGE and self.pending_attack is not None:
            victim=player;value=abs(_int(log.get('value'),0))
            if victim!=self.pending_attack['player'] and value:
                self.pending_attack['damage']+=value
                key=(self.pending_attack['player'],self.pending_attack['card_id'],self.pending_attack['attack_id'])
                self.attack_damage[key]+=value
        elif typ in {LOG_TURN_START,LOG_TURN_END,LOG_RESULT}:
            self.pending_attack=None

    def observe(self,obs:dict)->dict:
        if not isinstance(obs,dict):return {}
        cur=obs.get('current')
        if cur is None:return {}
        key=self._observation_key(obs)
        duplicate=key==self.last_obs_key
        self.last_obs_key=key
        turn=_int(cur.get('turn'),0);action_count=_int(cur.get('turnActionCount'),0)
        if self.me_index is None:self.me_index=_int(cur.get('yourIndex'),0)
        if turn!=self.last_turn:self.last_turn=turn
        if duplicate:self.stats['duplicate_observations']+=1
        else:
            self.stats['observations']+=1
            for log in (obs.get('logs') or []):self._ingest(_norm_log(log),turn,action_count)

        for player in (0,1):
            p=_player(cur,player)
            for zone in ('active','bench','discard','lostZone'):
                for card in (p.get(zone) or []):
                    self._remember_card(player,_card_id(card))
                    for pre in ((card or {}).get('preEvolution') or []) if isinstance(card,dict) else []:
                        self._remember_card(player,_card_id(pre))
            if p.get('hand') is not None:
                self.known_hand[player]={_serial(c):_card_id(c) for c in (p.get('hand') or []) if _serial(c) and _card_id(c)}
            self._trim_known_hand(player,_int(p.get('handCount'),len(p.get('hand') or [])))
        me=self.me_index if self.me_index in (0,1) else _int(cur.get('yourIndex'),0)
        op=1-me
        known_opp=len(self.known_hand[op]);self.stats['known_opponent_hand_peak']=max(self.stats['known_opponent_hand_peak'],known_opp)
        self.hand_timeline.append((turn,action_count,_int(_player(cur,me).get('handCount'),0),_int(_player(cur,op).get('handCount'),0),known_opp))
        self.last_state=self.current_state(obs)
        self.last_input={'history':self.events,'decisions':self.decisions,'current':self.last_state,'features':self.features(obs)}
        return self.last_input

    def _option_desc(self,obs:dict,index:int)->dict:
        cur=obs.get('current') or {};sel=obs.get('select') or {};opts=sel.get('option') or []
        if not 0<=index<len(opts):return {'index':index,'type':-1}
        o=opts[index];typ=_int(o.get('type'),-1);me=_int(cur.get('yourIndex'),0)
        out={'index':index,'type':typ}
        card=None;target=None
        if typ==OPT_PLAY:card=_zone(cur,sel,AREA_HAND,o.get('index'),me)
        elif typ in {OPT_ATTACH,OPT_EVOLVE}:
            card=_zone(cur,sel,o.get('area'),o.get('index'),me)
            target=_zone(cur,sel,o.get('inPlayArea'),o.get('inPlayIndex'),me)
        elif typ in {OPT_ABILITY}:
            card=_zone(cur,sel,o.get('area'),o.get('index'),me)
        elif typ==OPT_ATTACK:out['attackId']=_int(o.get('attackId'),0)
        elif o.get('area') is not None:
            card=_zone(cur,sel,o.get('area'),o.get('index'),_int(o.get('playerIndex'),me))
        if card:out.update({'cardId':_card_id(card),'serial':_serial(card)})
        if target:out.update({'targetId':_card_id(target),'targetSerial':_serial(target)})
        if o.get('playerIndex') is not None:out['playerIndex']=_int(o.get('playerIndex'),me)
        return out

    def record_choice(self,obs:dict,action:list[int]):
        if not isinstance(obs,dict) or not isinstance(action,list):return
        cur=obs.get('current') or {};sel=obs.get('select') or {}
        entry={
            'historyIndex':len(self.events),'decisionIndex':len(self.decisions),
            'turn':_int(cur.get('turn'),0),'turnActionCount':_int(cur.get('turnActionCount'),0),
            'context':_int(sel.get('context'),-1),'selected':list(action),
            'actions':[self._option_desc(obs,_int(i,-1)) for i in action],
            'stateKey':self.state_key(obs),
        }
        if self.decisions and self.decisions[-1]==entry:return
        self.decisions.append(entry);self.stats['decisions']+=1

    def current_state(self,obs:dict)->dict:
        cur=obs.get('current') or {};me=self.me_index if self.me_index in (0,1) else _int(cur.get('yourIndex'),0);op=1-me
        def side(index:int,own:bool)->dict:
            p=_player(cur,index);hand=p.get('hand') or []
            return {
                'active':[_pokemon_state(x) for x in (p.get('active') or []) if x],
                'bench':[_pokemon_state(x) for x in (p.get('bench') or []) if x],
                'discard':[_card_id(x) for x in (p.get('discard') or [])],
                'prizeCount':len(p.get('prize') or []),'deckCount':_int(p.get('deckCount'),0),
                'handCount':_int(p.get('handCount'),len(hand)),
                'handExact':[_card_id(x) for x in hand] if own else None,
                'handKnownLowerBound':list(self.known_hand[index].values()) if not own else [_card_id(x) for x in hand],
                'conditions':{k:bool(p.get(k)) for k in ('poisoned','burned','asleep','paralyzed','confused')},
            }
        return {
            'turn':_int(cur.get('turn'),0),'turnActionCount':_int(cur.get('turnActionCount'),0),
            'yourIndex':me,'firstPlayer':_int(cur.get('firstPlayer'),-1),
            'supporterPlayed':bool(cur.get('supporterPlayed')),'stadiumPlayed':bool(cur.get('stadiumPlayed')),
            'energyAttached':bool(cur.get('energyAttached')),'retreated':bool(cur.get('retreated')),
            'stadium':[_card_id(x) for x in (cur.get('stadium') or [])],
            'self':side(me,True),'opponent':side(op,False),
        }

    def state_key(self,obs:dict)->str:
        state=self.current_state(obs);state.pop('turnActionCount',None)
        raw=json.dumps(state,sort_keys=True,separators=(',',':')).encode('utf-8')
        return hashlib.sha1(raw).hexdigest()[:16]

    def revealed_ids(self,player:int)->set[int]:return set(self.revealed.get(player,set()))

    def known_hand_ids(self,player:int)->list[int]:return list(self.known_hand.get(player,{}).values())

    def action_count(self,player:int,log_type:int|None=None,card_id:int|None=None)->int:
        if card_id is not None:
            if log_type is None:return sum(n for (p,_t,c),n in self.card_events.items() if p==player and c==card_id)
            return int(self.card_events.get((player,log_type,card_id),0))
        if log_type is None:return sum(n for (p,_t),n in self.event_counts.items() if p==player)
        return int(self.event_counts.get((player,log_type),0))

    def attack_count(self,player:int,attack_id:int|None=None)->int:
        if attack_id is None:return sum(n for (p,_a),n in self.attack_events.items() if p==player)
        return int(self.attack_events.get((player,attack_id),0))

    def card_threat(self,player:int,card_id:int)->float:
        attacks=self.action_count(player,LOG_ATTACK,card_id)
        evolves=self.action_count(player,LOG_EVOLVE,card_id)
        plays=self.action_count(player,LOG_PLAY,card_id)
        damage=sum(v for (p,c,_a),v in self.attack_damage.items() if p==player and c==card_id)
        return 4.0*attacks+2.0*evolves+1.25*plays+min(8.0,damage/60.0)

    def recent_attack_damage(self,player:int,card_id:int=0)->int:
        values=[(i,e) for i,e in enumerate(self.events) if _int(e.get('type'),-1)==LOG_ATTACK and _int(e.get('playerIndex'),-1)==player and (not card_id or _int(e.get('cardId'),0)==card_id)]
        if not values:return 0
        idx,attack=values[-1];total=0
        for e in self.events[idx+1:]:
            if _int(e.get('type'),-1) in {LOG_ATTACK,LOG_TURN_START,LOG_TURN_END}:break
            if _int(e.get('type'),-1)==LOG_HP_CHANGE and _int(e.get('playerIndex'),-1)!=player:total+=abs(_int(e.get('value'),0))
        return total

    def features(self,obs:dict|None=None)->dict:
        obs=obs or {};cur=obs.get('current') or {};me=self.me_index if self.me_index in (0,1) else _int(cur.get('yourIndex'),0);op=1-me
        turn=_int(cur.get('turn'),self.last_turn if self.last_turn>=0 else 0)
        f={
            'hist_event_count':min(len(self.events),512)/512.0,
            'hist_decision_count':min(len(self.decisions),256)/256.0,
            'hist_own_actions':min(self.action_count(me),256)/256.0,
            'hist_opp_actions':min(self.action_count(op),256)/256.0,
            'hist_own_attacks':min(self.attack_count(me),20)/20.0,
            'hist_opp_attacks':min(self.attack_count(op),20)/20.0,
            'hist_opp_known_hand':min(len(self.known_hand.get(op,{})),10)/10.0,
        }
        for (player,typ),count in self.event_counts.items():
            side='own' if player==me else 'opp' if player==op else 'global'
            f[f'hist_{side}_logtype={typ}']=min(count,20)/20.0
        for (player,typ,cid),count in self.card_events.items():
            side='own' if player==me else 'opp' if player==op else 'global'
            f[f'hist_{side}_event={typ}|card={cid}']=min(count,8)/8.0
        for (player,aid),count in self.attack_events.items():
            side='own' if player==me else 'opp' if player==op else 'global'
            f[f'hist_{side}_attack={aid}']=min(count,8)/8.0
        # Ordered recent sequence.  The complete unbounded sequence remains in
        # ``events``; these recency slots are the fixed-cost policy representation.
        recent=self.events[-32:]
        tokens=[]
        for pos,event in enumerate(recent):
            side='own' if _int(event.get('playerIndex'),-1)==me else 'opp' if _int(event.get('playerIndex'),-1)==op else 'global'
            token=f"{side}:{_int(event.get('type'),-1)}:{_int(event.get('cardId'),0)}:{_int(event.get('attackId'),0)}:{_int(event.get('cardIdTarget'),0)}"
            tokens.append(token);f[f'hist_recent_{len(recent)-1-pos}={token}']=1.0
        for a,b in zip(tokens,tokens[1:]):f[f'hist_transition={a}>{b}']=f.get(f'hist_transition={a}>{b}',0.0)+1.0/31.0
        mine=_player(cur,me);enemy=_player(cur,op)
        for cid,count in Counter(_card_id(x) for x in (mine.get('hand') or []) if _card_id(x)).items():f[f'current_own_hand={cid}']=min(count,4)/4.0
        for cid,count in Counter(self.known_hand_ids(op)).items():f[f'current_opp_known_hand={cid}']=min(count,4)/4.0
        f['current_own_hand_count']=min(_int(mine.get('handCount'),0),20)/20.0
        f['current_opp_hand_count']=min(_int(enemy.get('handCount'),0),20)/20.0
        for side,p in (('own',mine),('opp',enemy)):
            for zone in ('active','bench'):
                for card in (p.get(zone) or []):
                    cid=_card_id(card);hp=_int((card or {}).get('hp') if isinstance(card,dict) else 0);mh=max(1,_int((card or {}).get('maxHp') if isinstance(card,dict) else hp,1))
                    f[f'current_{side}_{zone}={cid}']=f.get(f'current_{side}_{zone}={cid}',0.0)+1.0
                    f[f'current_{side}_hpfrac={cid}']=hp/mh
                    f[f'current_{side}_energy={cid}']=min(len((card or {}).get('energyCards') or (card or {}).get('energies') or []),6)/6.0 if isinstance(card,dict) else 0.0
        f['hist_turn_norm']=min(turn,30)/30.0
        return f

    def summary(self)->dict:
        me=self.me_index if self.me_index in (0,1) else 0;op=1-me
        return {
            'events':len(self.events),'decisions':len(self.decisions),
            'own_attacks':self.attack_count(me),'opponent_attacks':self.attack_count(op),
            'revealed_opponent_cards':len(self.revealed.get(op,set())),
            'known_opponent_hand_lower_bound':len(self.known_hand.get(op,{})),
            'feature_count':len((self.last_input or {}).get('features') or {}),
            'stats':dict(self.stats),
        }


class HistoryDecisionGate:
    """Conservative legal-action corrections driven by the persistent history."""
    def __init__(self,history:HistoryContext,card_table:dict,judge_id:int,xerosic_id:int,lillie_id:int):
        self.history=history;self.card_table=card_table;self.judge_id=judge_id;self.xerosic_id=xerosic_id;self.lillie_id=lillie_id
        self.stats={'calls':0,'overrides':{},'stutter_hits':0}

    def reset(self):self.stats={'calls':0,'overrides':{},'stutter_hits':0}

    def _note(self,key:str):self.stats['overrides'][key]=self.stats['overrides'].get(key,0)+1

    def _card(self,obs:dict,option:dict):
        cur=obs.get('current') or {};sel=obs.get('select') or {};me=_int(cur.get('yourIndex'),0)
        typ=_int(option.get('type'),-1)
        if typ==OPT_PLAY:return _zone(cur,sel,AREA_HAND,option.get('index'),me)
        return _zone(cur,sel,option.get('area'),option.get('index'),_int(option.get('playerIndex'),me))

    def choose(self,obs:dict,base:list[int])->list[int]:
        if not isinstance(obs,dict) or not isinstance(base,list) or obs.get('select') is None:return base
        self.stats['calls']+=1
        cur=obs.get('current') or {};sel=obs.get('select') or {};opts=sel.get('option') or []
        if not opts:return base
        me=_int(cur.get('yourIndex'),0);op=1-me;ctx=_int(sel.get('context'),-1)

        # Publicly known evolution cards in the opponent's hand are a real
        # sequence signal.  Interrupt only when the baseline would end the turn or
        # spend the Supporter on generic draw, never over an attack/closeout line.
        if ctx==CTX_MAIN and not bool(cur.get('supporterPlayed')):
            known=self.history.known_hand_ids(op)
            threat_known=[]
            for cid in known:
                data=self.card_table.get(cid)
                if data is not None and (bool(getattr(data,'stage1',False)) or bool(getattr(data,'stage2',False))):threat_known.append(cid)
            enemy=_player(cur,op);mine=_player(cur,me)
            base_types={_int(opts[i].get('type'),-1) for i in base if 0<=i<len(opts)}
            base_cards={_card_id(self._card(obs,opts[i])) for i in base if 0<=i<len(opts)}
            low_commit=not base or base_types<={OPT_END,OPT_PLAY} and (not base_cards or base_cards<={self.lillie_id})
            if threat_known and _int(enemy.get('handCount'),0)>=5 and _int(mine.get('handCount'),0)>=4 and low_commit:
                candidates=[]
                for i,o in enumerate(opts):
                    if _int(o.get('type'),-1)!=OPT_PLAY:continue
                    cid=_card_id(self._card(obs,o))
                    if cid in {self.judge_id,self.xerosic_id}:candidates.append((1 if cid==self.xerosic_id and _int(enemy.get('handCount'),0)>=6 else 0,i,cid))
                if candidates:
                    candidates.sort(reverse=True);self._note('known_hand_evolution_disrupt');return [candidates[0][1]]

        # When selecting an opposing target, historical attacks/evolutions break a
        # tie between equal-prize bodies.  This uses all prior public actions, not
        # only the currently visible board.
        if ctx in {CTX_SWITCH,CTX_TO_ACTIVE} and base and len(base)==1 and 0<=base[0]<len(opts):
            def info(i:int):
                o=opts[i]
                if _int(o.get('playerIndex'),me)!=op:return None
                card=self._card(obs,o)
                if not card:return None
                cid=_card_id(card);data=self.card_table.get(cid);prize=3 if data is not None and bool(getattr(data,'megaEx',False)) else 2 if data is not None and bool(getattr(data,'ex',False)) else 1
                return (prize,self.history.card_threat(op,cid),cid)
            current=info(base[0]);best=None
            if current is not None:
                for i in range(len(opts)):
                    candidate=info(i)
                    if candidate is None or candidate[0]!=current[0]:continue
                    if candidate[1]>=current[1]+4.0 and (best is None or candidate[1]>best[0]):best=(candidate[1],i,candidate[2])
            if best is not None:self._note('historical_threat_target');return [best[1]]

        # Exact same state + exact same emitted action twice indicates a stalled
        # action loop, not a purposeful sequence (hand/field state would otherwise
        # have changed).  Prefer a legal attack, then END, as a bounded failsafe.
        if base:
            state_key=self.history.state_key(obs);desc=[self.history._option_desc(obs,i) for i in base]
            repeats=sum(1 for d in self.history.decisions if d.get('turn')==_int(cur.get('turn'),0) and d.get('stateKey')==state_key and d.get('actions')==desc)
            if repeats>=2:
                rescue=next((i for i,o in enumerate(opts) if _int(o.get('type'),-1)==OPT_ATTACK),None)
                if rescue is None:rescue=next((i for i,o in enumerate(opts) if _int(o.get('type'),-1)==OPT_END),None)
                if rescue is not None and rescue not in base:
                    self.stats['stutter_hits']+=1;self._note('stalled_sequence_guard');return [rescue]
        return base

    def get_stats(self)->dict:return dict(self.stats)
