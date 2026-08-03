from __future__ import annotations
import os,json,math
from collections import Counter
from cg.api import (to_observation_class, all_card_data, all_attack, OptionType, CardType,
                    SelectContext, SelectType, AreaType)
_HERE=os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

def _load_deck():
    p=os.path.join(_HERE,'deck.csv')
    txt=open(p,encoding='utf-8').read().replace('\n',',')
    return [int(x.strip()) for x in txt.split(',') if x.strip()]
MY_DECK=_load_deck()
try:
    FAMILY=json.load(open(os.path.join(_HERE,'unified_config.json'),encoding='utf-8')).get('deck','generic')
except Exception:
    FAMILY='generic'
CARDS={c.cardId:c for c in all_card_data()}
ATTACKS={a.attackId:a for a in all_attack()}
try: PRIOR=json.load(open(os.path.join(_HERE,'replay_prior.json'),encoding='utf-8'))
except Exception: PRIOR={}

try:
    BEHAVIOR_MODEL=json.load(open(os.path.join(_HERE,'behavior_model.json'),encoding='utf-8'))
except Exception:
    BEHAVIOR_MODEL={}
try:
    BEHAVIOR_SCALE=float(json.load(open(os.path.join(_HERE,'unified_config.json'),encoding='utf-8')).get('behavior_scale',0.0))
except Exception:
    BEHAVIOR_SCALE=0.0

def _behavior_board_summary(pl):
    ps=[x for x in list(pl.active)+list(pl.bench) if x]
    return len(ps),sum(int(x.hp or 0) for x in ps),sum(max(0,int(x.maxHp or 0)-int(x.hp or 0)) for x in ps),sum(len(x.energies or []) for x in ps)

def _behavior_features(obs,o):
    s=obs.current; me=s.yourIndex; a=s.players[me]; b=s.players[1-me]
    ctx=int(obs.select.context) if obs.select else -1; typ=int(o.type); ph=_phase(int(s.turn or 0))
    an,ah,ad,ae=_behavior_board_summary(a);bn,bh,bd,be=_behavior_board_summary(b)
    d={'turn':float(s.turn or 0)/30,'turn_action':float(s.turnActionCount or 0)/25,
       'hand':float(a.handCount or 0)/12,'opp_hand':float(b.handCount or 0)/12,
       'deck':float(a.deckCount or 0)/60,'opp_deck':float(b.deckCount or 0)/60,
       'prize_diff':(len(b.prize or [])-len(a.prize or []))/6,
       'bench_diff':(an-bn)/5,'hp_diff':(ah-bh)/1800,'damage_diff':(bd-ad)/1800,
       'energy_diff':(ae-be)/12,'energy_attached':float(bool(s.energyAttached)),
       'supporter_played':float(bool(s.supporterPlayed)),'retreated':float(bool(s.retreated)),
       'first':float(s.firstPlayer==me),f'ctx={ctx}':1,f'type={typ}':1,
       f'ctx_type={ctx}:{typ}':1,f'phase={ph}':1,f'phase_type={ph}:{typ}':1}
    active=next((x for x in a.active if x),None);oppactive=next((x for x in b.active if x),None)
    aa=int(getattr(active,'id',0) or 0);ba=int(getattr(oppactive,'id',0) or 0)
    stadium=next((x for x in s.stadium if x),None);stad=int(getattr(stadium,'id',0) or 0)
    if aa:d[f'active={aa}']=1;d[f'ctx_active={ctx}:{aa}']=1;d[f'type_active={typ}:{aa}']=1
    if ba:d[f'oppactive={ba}']=1;d[f'ctx_oppactive={ctx}:{ba}']=1;d[f'type_oppactive={typ}:{ba}']=1
    if stad:d[f'stadium={stad}']=1;d[f'ctx_stadium={ctx}:{stad}']=1
    for x,n in Counter(int(getattr(z,'id',0) or 0) for z in (a.hand or []) if getattr(z,'id',0)).items():
        d[f'handcard={x}']=min(n,3);d[f'type_handcard={typ}:{x}']=min(n,2)
    for x,n in Counter(int(getattr(z,'id',0) or 0) for z in (a.bench or []) if getattr(z,'id',0)).items():d[f'benchcard={x}']=min(n,3)
    for x,n in Counter(int(getattr(z,'id',0) or 0) for z in (b.bench or []) if getattr(z,'id',0)).items():d[f'oppbenchcard={x}']=min(n,3)
    for k in ('area','inPlayArea'):
        v=getattr(o,k,None)
        if v is not None:d[f'{k}={int(v)}']=1
    c=_source(obs,o);cid=int(getattr(c,'id',0) or getattr(o,'cardId',0) or 0)
    if cid:
        d[f'card={cid}']=1;d[f'ctx_card={ctx}:{cid}']=1;d[f'type_card={typ}:{cid}']=1;d[f'phase_card={ph}:{cid}']=1
        if aa:d[f'card_active={cid}:{aa}']=1
        if ba:d[f'card_oppactive={cid}:{ba}']=1
        cd=CARDS.get(cid)
        if cd:
            d[f'card_type={int(cd.cardType)}']=1;d['card_hp']=float(cd.hp or 0)/400
            d['card_stage']=2 if cd.stage2 else 1 if cd.stage1 else 0
            d['card_basic']=float(bool(cd.basic));d['card_ex']=2 if cd.megaEx else 1 if cd.ex else 0
            d['card_skills']=len(cd.skills)/4
            aa=[ATTACKS[x] for x in cd.attacks if x in ATTACKS]
            d['card_best_attack']=max([int(x.damage or 0) for x in aa] or [0])/350
            d['card_min_cost']=min([len(x.energies) for x in aa] or [5])/5
    t=_target(obs,o);tid=int(getattr(t,'id',0) or 0)
    if tid:
        d[f'target={tid}']=1;d[f'ctx_target={ctx}:{tid}']=1;d[f'type_target={typ}:{tid}']=1
        if aa:d[f'target_active={tid}:{aa}']=1
        d['target_hp']=float(t.hp or 0)/400;d['target_damage']=max(0,float(t.maxHp or 0)-float(t.hp or 0))/400
        d['target_energy']=len(t.energies or [])/5
    aid=int(getattr(o,'attackId',0) or 0)
    if aid:
        d[f'attack={aid}']=1;d[f'ctx_attack={ctx}:{aid}']=1
        if ba:d[f'attack_oppactive={aid}:{ba}']=1
        at=ATTACKS.get(aid)
        if at:d['attack_damage']=int(at.damage or 0)/350;d['attack_cost']=len(at.energies)/5;d['attack_effect']=float(bool((at.text or '').strip()))
    num=getattr(o,'number',None)
    if num is not None:d['number']=float(num)/10;d[f'ctx_number={ctx}:{num}']=1
    cnt=getattr(o,'count',None)
    if cnt is not None:d['energy_count']=float(cnt)/5
    return d

def _behavior_score(obs,o):
    if not BEHAVIOR_MODEL or BEHAVIOR_SCALE==0:return 0.0
    d=_behavior_features(obs,o);co=BEHAVIOR_MODEL.get('coef') or {}
    return float(BEHAVIOR_MODEL.get('intercept',0.0))+sum(float(v)*float(co.get(k,0.0)) for k,v in d.items())

SETUP_NAMES={'buddy-buddy poffin','ultra ball','rare candy','poké pad','pokégear 3.0','bug catching set','fighting gong','team rocket\'s transceiver','tera orb'}
DRAW_NAMES={'lillie\'s determination','hilda','dawn','wally\'s compassion','cynthia\'s roselia','explorer’s guidance','team rocket\'s petrel','judge','ciphermaniac’s codebreaking'}
DISRUPT_NAMES={'boss’s orders','crushing hammer','enhanced hammer','xerosic’s machinations','eri','team rocket\'s ariana','team rocket\'s proton','hand trimmer','unfair stamp'}
SWITCH_NAMES={'switch','air balloon'}
RECOVERY_NAMES={'night stretcher','sacred ash','energy retrieval','jumbo ice cream','cook','bianca’s devotion','wally\'s compassion'}


def _phase(turn:int)->str:return 'early' if turn<=3 else 'mid' if turn<=8 else 'late'

def _source(obs,o):
    s=obs.current; me=s.yourIndex; pi=o.playerIndex if o.playerIndex is not None else me; pl=s.players[pi]
    try:
        ar=int(o.area) if o.area is not None else (int(AreaType.HAND) if o.type==OptionType.PLAY else -1)
        if ar==int(AreaType.DECK) and obs.select.deck:return obs.select.deck[o.index]
        if ar==int(AreaType.HAND) and pl.hand:return pl.hand[o.index]
        if ar==int(AreaType.DISCARD):return pl.discard[o.index]
        if ar==int(AreaType.ACTIVE):return pl.active[o.index]
        if ar==int(AreaType.BENCH):return pl.bench[o.index]
        if ar==int(AreaType.STADIUM):return s.stadium[o.index]
        if ar==int(AreaType.LOOKING) and s.looking:return s.looking[o.index]
    except Exception:return None
    return None

def _target(obs,o):
    try:
        pl=obs.current.players[obs.current.yourIndex]
        ar=int(o.inPlayArea) if o.inPlayArea is not None else -1
        if ar==int(AreaType.ACTIVE):return pl.active[o.inPlayIndex]
        if ar==int(AreaType.BENCH):return pl.bench[o.inPlayIndex]
    except Exception:pass
    return None

def _keys(obs,o):
    typ=int(o.type);ctx=int(obs.select.context) if obs.select else -1;ks=[f't{typ}',f'ctx{ctx}:t{typ}']; c=_source(obs,o);cid=getattr(c,'id',getattr(o,'cardId',0)) or 0
    if cid:ks.extend([f't{typ}:c{cid}',f'ctx{ctx}:t{typ}:c{cid}'])
    t=_target(obs,o);tcid=getattr(t,'id',0) or 0
    if tcid:ks.extend([f't{typ}:target{tcid}',f'ctx{ctx}:t{typ}:target{tcid}'])
    aid=int(getattr(o,'attackId',0) or 0)
    if aid:ks.extend([f't{typ}:a{aid}',f'ctx{ctx}:t{typ}:a{aid}'])
    return ks

def _prior(obs,o):
    ph=_phase(obs.current.turn if obs.current else 0);keys=_keys(obs,o)
    def total(table):
        d=(table or {}).get(ph,{})
        return sum(float(d.get(k,0.0)) for k in keys)
    return 45.0*total(PRIOR.get('phase_bias'))+10.0*total(PRIOR.get('global_phase_bias'))+120.0*total(PRIOR.get('choice_phase_bias'))

def _attack_stats(cid):
    cd=CARDS.get(int(cid or 0)); best=0;cheap=99;effects=0
    if not cd:return 0,99,0
    for aid in cd.attacks:
        a=ATTACKS.get(aid)
        if not a:continue
        best=max(best,int(a.damage or 0));cheap=min(cheap,len(a.energies)); effects+=int(bool((a.text or '').strip()))
    return best,cheap,effects

def _pokemon_score_id(cid,active=False):
    cd=CARDS.get(int(cid or 0))
    if not cd:return 0
    best,cheap,effects=_attack_stats(cid)
    stage=2 if cd.stage2 else 1 if cd.stage1 else 0
    rule=3 if cd.megaEx else 2 if cd.ex else 0
    v=cd.hp*1.2+best*1.4+effects*18+stage*75+len(cd.skills)*55+rule*35
    if active:v-=cd.retreatCost*24;v+=max(0,3-cheap)*18
    return v

def _ready_damage(p):
    cd=CARDS.get(p.id)
    if not cd:return 0
    have=list(p.energies or []);best=0
    for aid in cd.attacks:
        a=ATTACKS.get(aid)
        if not a:continue
        pool=list(have);ok=True
        for req in a.energies:
            if int(req)==0:continue
            j=next((j for j,x in enumerate(pool) if int(x) in (int(req),10,11) or (int(x)==11 and int(req) in (5,7))),None)
            if j is None:ok=False;break
            pool.pop(j)
        if ok and len(have)>=len(a.energies):best=max(best,int(a.damage or 1))
    return best

def _board_pokemon(obs,own=True):
    s=obs.current;pl=s.players[s.yourIndex if own else 1-s.yourIndex]
    return [p for p in list(pl.active)+list(pl.bench) if p]


def _stadium_name(obs):
    try:
        s=obs.current
        if s.stadium:
            cd=CARDS.get(s.stadium[0].id);return cd.name.lower() if cd else ''
    except Exception:pass
    return ''

def _family_card_bonus(obs,card):
    if card is None:return 0.0
    cd=CARDS.get(card.id);name=cd.name.lower() if cd else ''
    turn=obs.current.turn if obs.current else 0
    bonus=0.0
    if FAMILY=='lopunny':
        if name=='mega lopunny ex':bonus+=320
        elif name=='buneary':bonus+=150 if turn<=5 else 55
        elif name=='dunsparce':bonus+=125 if turn<=5 else 35
        elif name=='dudunsparce':bonus+=180
        elif name=='fan rotom':bonus+=220 if turn<=2 else -50
        elif name in {'air balloon','hilda','buddy-buddy poffin'}:bonus+=110 if turn<=5 else 25
        elif name=="wally's compassion":
            damaged=any(CARDS.get(p.id) and CARDS[p.id].megaEx and p.hp<p.maxHp for p in _board_pokemon(obs,True));bonus+=220 if damaged else -80
    elif FAMILY=='grass_ogerpon':
        if name=='teal mask ogerpon ex':bonus+=300
        elif name in {'basic {g} energy','grow grass energy','energy search','bug catching set','tera orb'}:bonus+=150
        elif name=='n\'s plan':bonus+=110 if turn>=5 else -30
        elif name=='briar':bonus+=170 if len(obs.current.players[1-obs.current.yourIndex].prize)==2 else -150
    elif FAMILY=='dipplin':
        if name=='dipplin':bonus+=300
        elif name=='applin':bonus+=180 if turn<=5 else 40
        elif name=='thwackey':bonus+=250
        elif name=='grookey':bonus+=150 if turn<=5 else 25
        elif name=='festival grounds':bonus+=260 if _stadium_name(obs)!='festival grounds' else -80
        elif name in {'buddy-buddy poffin','bug catching set'}:bonus+=120 if turn<=5 else 20
    elif FAMILY=='cynthia':
        if name=="cynthia's garchomp ex":bonus+=360
        elif name=="cynthia's gabite":bonus+=270
        elif name=="cynthia's gible":bonus+=180 if turn<=5 else 40
        elif name=="cynthia's roserade":bonus+=280
        elif name=="cynthia's roselia":bonus+=150 if turn<=5 else 35
        elif name=="cynthia's power weight":bonus+=180
        elif name in {'fighting gong','hilda','buddy-buddy poffin'}:bonus+=120 if turn<=5 else 25
        elif name=='forest of vitality':bonus+=170 if _stadium_name(obs)!='forest of vitality' else -60
    elif FAMILY=='spidops':
        if name=="team rocket's spidops":bonus+=320
        elif name=="team rocket's tarountula":bonus+=180 if turn<=5 else 45
        elif name=="team rocket's mewtwo ex":bonus+=210
        elif name=="team rocket's articuno":bonus+=170
        elif name=="team rocket's energy":bonus+=190
        elif name=="team rocket's proton":bonus+=260 if turn<=2 else -40
        elif name=="team rocket's ariana":bonus+=160
        elif name in {"team rocket's transceiver","team rocket's factory","poké pad"}:bonus+=135
    return bonus

def _family_main_bonus(obs,o):
    s=obs.current;me=s.yourIndex;a=s.players[me];c=_source(obs,o);cd=CARDS.get(c.id) if c else None;name=cd.name.lower() if cd else ''
    t=_target(obs,o);tn=CARDS.get(t.id).name.lower() if t and CARDS.get(t.id) else ''
    b=0.0
    if FAMILY=='lopunny':
        if o.type==OptionType.ABILITY and name=='dudunsparce':b+=520
        if o.type==OptionType.EVOLVE and name=='mega lopunny ex':b+=520
        if o.type==OptionType.ATTACH and tn=='mega lopunny ex':b+=360 if len(t.energyCards)==0 else 80
        if o.type==OptionType.RETREAT:
            ready_lop=any(CARDS.get(p.id) and CARDS[p.id].name.lower()=='mega lopunny ex' and len(p.energyCards)>=1 for p in a.bench)
            if ready_lop:b+=650
        if o.type==OptionType.ATTACK and int(o.attackId or 0)==1225:
            switched=bool(s.retreated) or any(int(getattr(x,'type',-1))==8 and getattr(x,'playerIndex',None)==me for x in obs.logs)
            b+=700 if switched else -160
    elif FAMILY=='grass_ogerpon':
        if o.type==OptionType.ABILITY and name=='teal mask ogerpon ex':b+=650
        if o.type==OptionType.ATTACH and tn=='teal mask ogerpon ex':
            b+=420-(len(t.energyCards)*55)
            if t in (a.active or []):b+=80
        if o.type==OptionType.ATTACK and int(o.attackId or 0)==120:
            own=len((a.active[0].energyCards if a.active and a.active[0] else []));op=s.players[1-me];opp=len((op.active[0].energyCards if op.active and op.active[0] else []));b+=(own+opp)*85
    elif FAMILY=='dipplin':
        if o.type==OptionType.EVOLVE and name in {'dipplin','thwackey'}:b+=480
        if o.type==OptionType.ABILITY and name=='thwackey':
            active=a.active[0] if a.active else None;an=CARDS.get(active.id).name.lower() if active and CARDS.get(active.id) else ''
            b+=650 if an=='dipplin' and _stadium_name(obs)=='festival grounds' else -120
        if o.type==OptionType.ATTACH and tn=='dipplin':b+=430
        if o.type==OptionType.ATTACK and int(o.attackId or 0)==115:
            dmg=20*len(a.bench)*(2 if _stadium_name(obs)=='festival grounds' else 1);b+=dmg*2.5
        if o.type==OptionType.PLAY and name=='festival grounds':b+=500
    elif FAMILY=='cynthia':
        if o.type==OptionType.EVOLVE and name in {"cynthia's gabite","cynthia's garchomp ex","cynthia's roserade"}:b+=520
        if o.type==OptionType.ABILITY and name=="cynthia's gabite":b+=600
        if o.type==OptionType.ATTACH and tn in {"cynthia's garchomp ex","cynthia's gabite","cynthia's gible"}:b+=420-(len(t.energyCards)*70)
        if o.type==OptionType.ATTACK and int(o.attackId or 0)==532:b+=620
        if o.type==OptionType.ATTACK and int(o.attackId or 0)==531 and a.handCount<=3:b+=260
    elif FAMILY=='spidops':
        team_count=sum(1 for p in _board_pokemon(obs,True) if CARDS.get(p.id) and "team rocket's" in CARDS[p.id].name.lower())
        if o.type==OptionType.EVOLVE and name=="team rocket's spidops":b+=520
        if o.type==OptionType.ABILITY and name=="team rocket's spidops":b+=600
        if o.type==OptionType.ATTACH:
            if name=="team rocket's energy" and tn in {"team rocket's mewtwo ex","team rocket's articuno"}:b+=520
            if tn=="team rocket's spidops":b+=260
        if o.type==OptionType.ATTACK and int(o.attackId or 0)==560:b+=team_count*105
        if o.type==OptionType.PLAY and name in {"team rocket's proton","team rocket's factory","team rocket's transceiver"}:b+=300
    return b

def _card_value(obs,card,for_discard=False):
    if card is None:return -999
    cd=CARDS.get(card.id)
    if not cd:return 0
    s=obs.current;me=s.yourIndex;pl=s.players[me];name=cd.name.lower();v=0.0
    if cd.cardType==CardType.POKEMON:
        v=_pokemon_score_id(cd.cardId)
        # Evolution that can immediately land.
        if cd.evolvesFrom:
            for p in _board_pokemon(obs,True):
                pc=CARDS.get(p.id)
                if pc and pc.name==cd.evolvesFrom:v+=230
        elif len(pl.bench)<pl.benchMax:v+=90 if s.turn<=4 else 35
        # Duplicate penalty.
        visible=sum(1 for p in _board_pokemon(obs,True) if p.id==cd.cardId)
        v-=visible*35
    elif cd.cardType in (CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY):
        v=105 if not s.energyAttached else 42
        if sum(len(p.energyCards) for p in _board_pokemon(obs,True))<2:v+=65
    elif cd.cardType==CardType.SUPPORTER:
        v=135 if not s.supporterPlayed else 12
        if name in DRAW_NAMES:v+=max(0,7-pl.handCount)*18
        if name in DISRUPT_NAMES:v+=40
    elif cd.cardType==CardType.ITEM:
        v=95
        if name in SETUP_NAMES:v+=80 if s.turn<=5 else 15
        if name in RECOVERY_NAMES:v+=35 if pl.discard else -15
        if name in SWITCH_NAMES:v+=25
        if name in DISRUPT_NAMES:v+=35
    elif cd.cardType==CardType.TOOL:v=70
    elif cd.cardType==CardType.STADIUM:v=78 if not s.stadiumPlayed else 8
    v+=_family_card_bonus(obs,card)
    if for_discard:
        # Lower is better for discard caller.
        if cd.cardType in (CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY) and sum(1 for x in (pl.hand or []) if CARDS.get(x.id) and CARDS[x.id].cardType in (CardType.BASIC_ENERGY,CardType.SPECIAL_ENERGY))>2:v-=55
        if cd.cardType==CardType.POKEMON and cd.basic and len(pl.bench)>=pl.benchMax:v-=45
    return v

def _main_score(obs,o):
    s=obs.current;me=s.yourIndex;a=s.players[me];b=s.players[1-me];score=_prior(obs,o)+BEHAVIOR_SCALE*_behavior_score(obs,o)
    if o.type==OptionType.ATTACK:
        at=ATTACKS.get(int(o.attackId or 0));dmg=int(at.damage or 0) if at else 0
        score+=900+dmg*3.2
        active=b.active[0] if b.active and b.active[0] else None
        if active and dmg>=active.hp:
            cd=CARDS.get(active.id);pr=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
            score+=650+pr*260
        if at and at.text:score+=80
    elif o.type==OptionType.EVOLVE:
        c=_source(obs,o);score+=820+_card_value(obs,c)*0.55
    elif o.type==OptionType.ABILITY:
        c=_source(obs,o);score+=780+(_pokemon_score_id(c.id) if c else 0)*0.12
    elif o.type==OptionType.ATTACH:
        t=_target(obs,o);score+=700
        if t:
            before=_ready_damage(t);score+=_pokemon_score_id(t.id)*0.25+max(0,220-before)
            if t in (a.active or []):score+=35
    elif o.type==OptionType.PLAY:
        c=_source(obs,o);score+=500+_card_value(obs,c)
        if c:
            cd=CARDS.get(c.id);nm=cd.name.lower() if cd else ''
            if cd and cd.cardType==CardType.POKEMON and not cd.basic:score-=120 # evolutions normally use EVOLVE
            if nm in DRAW_NAMES and a.handCount<=4:score+=120
            if nm in SETUP_NAMES and s.turn<=4:score+=110
            if nm in DISRUPT_NAMES and b.handCount>=6:score+=80
    elif o.type==OptionType.RETREAT:
        score+=360
        act=a.active[0] if a.active and a.active[0] else None
        if act and _ready_damage(act)==0:score+=220
        if max([_ready_damage(p) for p in a.bench] or [0])>0:score+=260
    elif o.type==OptionType.END:
        score+=5
        # Ending is reasonable if already attacked unavailable and no setup options.
        if s.turnActionCount>7:score+=90
    elif o.type==OptionType.DISCARD:score+=100
    score+=_family_main_bonus(obs,o)
    return score

def _context_score(obs,o):
    ctx=obs.select.context;c=_source(obs,o);cd=CARDS.get(c.id) if c else None;score=_prior(obs,o)+BEHAVIOR_SCALE*_behavior_score(obs,o)
    if o.type==OptionType.NUMBER:
        n=int(o.number or 0)
        if ctx in (SelectContext.DRAW_COUNT,SelectContext.DAMAGE_COUNTER_COUNT,SelectContext.REMOVE_DAMAGE_COUNTER_COUNT):return n*100
        return -n
    if o.type==OptionType.YES:return 30
    if o.type==OptionType.NO:return 0
    if o.type==OptionType.ATTACK:
        a=ATTACKS.get(int(o.attackId or 0));return 400+(a.damage if a else 0)*4+(80 if a and a.text else 0)+score
    if o.type==OptionType.SKILL:return score+10-(o.serial or 0)*1e-6
    if o.type==OptionType.SPECIAL_CONDITION:return score+{3:50,4:40,0:35,1:25,2:20}.get(int(o.specialConditionType or 0),0)
    if o.type in (OptionType.ENERGY,OptionType.ENERGY_CARD,OptionType.TOOL_CARD):return score+50+(o.count or 0)*15
    if o.type==OptionType.CARD:
        val=_card_value(obs,c)
        if ctx==SelectContext.SETUP_ACTIVE_POKEMON:return _pokemon_score_id(c.id,True) if c else -999
        if ctx==SelectContext.SETUP_BENCH_POKEMON:return _pokemon_score_id(c.id,False) if c else -999
        if ctx in (SelectContext.SWITCH,SelectContext.TO_ACTIVE):
            if c and cd and cd.cardType==CardType.POKEMON:return _pokemon_score_id(c.id,True)+(_ready_damage(c) if hasattr(c,'energies') else 0)*4
        if ctx in (SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH,SelectContext.EVOLVES_TO,SelectContext.ATTACH_TO):return val
        if ctx in (SelectContext.DISCARD,SelectContext.TO_DECK,SelectContext.TO_DECK_BOTTOM,SelectContext.TO_PRIZE,SelectContext.NOT_MOVE):return -_card_value(obs,c,True)
        if ctx in (SelectContext.DAMAGE_COUNTER,SelectContext.DAMAGE_COUNTER_ANY,SelectContext.DAMAGE,SelectContext.EFFECT_TARGET):
            # Opponent targets: low HP/high prize are preferred. Own effect targets still get generic value.
            if c and hasattr(c,'hp'):
                rule=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
                return (500-c.hp)+rule*180+_pokemon_score_id(c.id)*0.15
        if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER,SelectContext.HEAL):
            if c and hasattr(c,'hp'):return (c.maxHp-c.hp)*4+_pokemon_score_id(c.id)*0.2
        if ctx in (SelectContext.EVOLVES_FROM,SelectContext.ATTACH_FROM,SelectContext.DETACH_FROM):return val
        return val
    return score

def _choose_count(obs):
    mn=max(0,int(obs.select.minCount));mx=min(len(obs.select.option),int(obs.select.maxCount))
    ctx=obs.select.context
    # Beneficial acquisition/placement contexts use the maximum; discard/return uses the minimum.
    if ctx in (SelectContext.SETUP_BENCH_POKEMON,SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH,SelectContext.DAMAGE_COUNTER_ANY):return mx
    return mn if mn>0 else (1 if mx>0 else 0)

def agent(observation:dict)->list[int]:
    if observation.get('select') is None:return list(MY_DECK)
    try:obs=to_observation_class(observation);sel=obs.select
    except Exception:
        s=observation.get('select') or {};n=int(s.get('minCount',0) or 0);return list(range(min(n,len(s.get('option') or []))))
    if sel is None or not sel.option or sel.maxCount<=0:return []
    n=_choose_count(obs)
    if n<=0:return []
    vals=[]
    for i,o in enumerate(sel.option):
        try:v=_main_score(obs,o) if sel.context==SelectContext.MAIN else _context_score(obs,o)
        except Exception:v=-1e9+i*1e-6
        vals.append((v,-i,i))
    vals.sort(reverse=True)
    return [i for _,__,i in vals[:min(n,len(vals))]]
