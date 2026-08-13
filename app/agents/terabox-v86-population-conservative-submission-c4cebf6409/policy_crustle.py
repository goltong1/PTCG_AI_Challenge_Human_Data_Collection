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


# === Tera Box v1 strategy layer ===
# This layer keeps the generic probabilistic policy as a fallback and adds
# deck-specific value comparisons for Area Zero, draw engines, energy routing,
# Ciphermaniac top-deck plans, and matchup tech attackers.
_TB_OLD_CARD_VALUE=_card_value
_TB_OLD_MAIN_SCORE=_main_score
_TB_OLD_CONTEXT_SCORE=_context_score
_TB_OLD_CHOOSE_COUNT=_choose_count

TB={
 'TEAL':96,'TERAPAGOS':176,'KANGA':756,'LATIAS':184,'CORNER':117,'WELLSPRING':108,
 'CHIYU':31,'PECH':230,'PECH_EX':141,'BUG':1094,'LILLIE':1227,'AREA':1250,
 'ESWITCH':1116,'NPLAN':1221,'CIPHER':1188,'TRUMPET':1098,'ULTRA':1121,
 'TERA_ORB':1127,'BOSS':1182,'PRIME':1088,'NIGHT':1097,'SWITCH':1123,'CYRANO':1205,
 'G':1,'R':2,'W':3,'L':4,'F':6,'D':7,
}

def _tb_ids(xs):return [int(getattr(x,'id',0) or 0) for x in (xs or []) if x]
def _tb_board(obs,own=True):return _board_pokemon(obs,own)
def _tb_hand(obs):
 s=obs.current;return [x for x in (s.players[s.yourIndex].hand or []) if x]
def _tb_has(obs,cid,own=True):return any(int(getattr(p,'id',0) or 0)==cid for p in _tb_board(obs,own))
def _tb_count(obs,cid,own=True):return sum(int(getattr(p,'id',0) or 0)==cid for p in _tb_board(obs,own))
def _tb_hand_count(obs,cid):return sum(int(getattr(x,'id',0) or 0)==cid for x in _tb_hand(obs))
def _tb_stadium_id(obs):
 try:return int(obs.current.stadium[0].id) if obs.current.stadium else 0
 except Exception:return 0

def _tb_matchup(obs):
 ids=set(_tb_ids(_tb_board(obs,False)))
 if ids & {344,345,58}:return 'crustle'
 if ids & {646,647,648,860,104}:return 'marnie'
 if ids & {119,120,121}:return 'dragapult'
 if ids & {169,190,666,57}:return 'archaludon'
 if ids & {741,742,743}:return 'alakazam'
 # An all-single-prize public board is a soft signal only after two bodies appear.
 opp=_tb_board(obs,False)
 if len(opp)>=2 and not any((CARDS.get(p.id) and (CARDS[p.id].ex or CARDS[p.id].megaEx)) for p in opp):return 'single_prize'
 return 'generic'

def _tb_type(cid):
 # Energy type requirements used by this deck.  0 means colorless/no typed need.
 return {TB['TEAL']:TB['G'],TB['TERAPAGOS']:TB['G'],TB['CORNER']:TB['F'],TB['WELLSPRING']:TB['W'],TB['CHIYU']:TB['R'],TB['PECH']:TB['D'],TB['PECH_EX']:TB['D'],TB['LATIAS']:5}.get(int(cid),0)

def _tb_dynamic_damage(obs,p,aid):
 if p is None:return 0
 aid=int(aid or 0);s=obs.current;me=s.yourIndex;opp=s.players[1-me]
 if aid==232:return 30*len([x for x in s.players[me].bench if x])
 if aid==120:
  oe=len(opp.active[0].energyCards or []) if opp.active and opp.active[0] else 0
  return 30+30*(len(p.energyCards or [])+oe)
 if aid==1092:return 250  # expected Rapid-Fire Combo value
 if aid==184:return 60*(6-len(opp.prize or []))
 if aid==20:return 120 if _tb_stadium_id(obs) else 60
 if aid==19:return 0
 if aid==148:return 140
 if aid==136:return 100
 if aid==243:return 200
 if aid==1296:return 20
 if aid==233:return 180
 a=ATTACKS.get(aid);return int(a.damage or 0) if a else 0

def _tb_attack_ready(p,aid,extra_energy=None):
 a=ATTACKS.get(int(aid or 0))
 if not a:return False
 pool=list(getattr(p,'energies',[]) or [])
 if extra_energy is not None:pool.append(extra_energy)
 # Engine energy IDs use type integers; universal energies are accepted by base policy.
 for req in a.energies:
  req=int(req)
  if req==0:
   if not pool:return False
   pool.pop(0);continue
  j=next((j for j,x in enumerate(pool) if int(x)==req),None)
  if j is None:return False
  pool.pop(j)
 return True

def _tb_best_ready_damage(obs,p):
 cd=CARDS.get(int(getattr(p,'id',0) or 0));best=0
 if not cd:return 0
 for aid in cd.attacks:
  if _tb_attack_ready(p,aid):best=max(best,_tb_dynamic_damage(obs,p,aid))
 return best

def _tb_role_value(obs,card,instance=False):
 if card is None:return -1000
 cid=int(getattr(card,'id',0) or 0);turn=int(obs.current.turn or 0);m=_tb_matchup(obs)
 v=_TB_OLD_CARD_VALUE(obs,card)
 inplay=_tb_has(obs,cid)
 if cid==TB['BUG']:v+=800 if turn<=5 else 180
 elif cid==TB['LATIAS']:v+=850 if not inplay and turn<=4 else (-180 if inplay else 180)
 elif cid==TB['TEAL']:v+=620 if _tb_count(obs,cid)<2 and turn<=4 else 180
 elif cid==TB['KANGA']:v+=520 if not inplay and turn<=4 else 140
 elif cid==TB['TERAPAGOS']:v+=520 if _tb_count(obs,cid)==0 else 170
 elif cid==TB['AREA']:v+=520 if _tb_stadium_id(obs)!=TB['AREA'] and any(CARDS.get(p.id) and (p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']}) for p in _tb_board(obs,True)) else -80
 elif cid==TB['CIPHER']:v+=220
 elif cid==TB['LILLIE']:v+=max(0,6-obs.current.players[obs.current.yourIndex].handCount)*55
 elif cid==TB['ESWITCH']:v+=260 if any(len(p.energyCards or [])>=2 for p in _tb_board(obs,True)) else -40
 elif cid==TB['NPLAN']:v+=300 if any(len(p.energyCards or [])>=2 for p in _tb_board(obs,True)[1:]) else -100
 elif cid==TB['TRUMPET']:v+=280 if _tb_stadium_id(obs)==TB['AREA'] else -80
 elif cid==TB['CORNER']:v+=700 if m in {'marnie','archaludon','alakazam'} else 70
 elif cid==TB['CHIYU']:v+=620 if m in {'crustle','single_prize'} else 40
 elif cid==TB['PECH']:v+=520 if m in {'crustle','single_prize'} else 25
 elif cid==TB['PECH_EX']:v+=420 if m=='single_prize' and len(obs.current.players[1-obs.current.yourIndex].prize or [])<=3 else -30
 elif cid==TB['WELLSPRING']:v+=280 if m=='dragapult' else 50
 if instance and hasattr(card,'energyCards'):
  v+=len(card.energyCards or [])*180
  v+=max(0,int(card.hp or 0))*0.25
  if cid==TB['LATIAS'] and any((CARDS.get(p.id) and CARDS[p.id].retreatCost>0) for p in _tb_board(obs,True)):v+=260
  if cid==TB['TEAL'] and len(card.energyCards or [])==0:v-=80
  if cid==TB['KANGA'] and card not in (obs.current.players[obs.current.yourIndex].active or []):v-=40
 return v

def _tb_unknown_deck_values(obs):
 # Hypergeometric-style expectation from the known 60-card list minus public own cards.
 counts=Counter(MY_DECK);s=obs.current;pl=s.players[s.yourIndex]
 known=list(pl.hand or [])+list(pl.discard or [])+list(pl.active or [])+list(pl.bench or [])+list(pl.prize or [])
 for c in known:
  cid=int(getattr(c,'id',0) or 0)
  if counts[cid]>0:counts[cid]-=1
 vals=[]
 for cid,n in counts.items():
  if n<=0:continue
  dummy=CARDS.get(cid)
  if dummy:vals += [_tb_role_value(obs,dummy)]*n
 return vals

def _tb_cycle_delta(obs,draw_n):
 hand=[x for x in _tb_hand(obs) if int(getattr(x,'id',0) or 0)!=TB['LILLIE']]
 keep=sum(_tb_role_value(obs,x) for x in hand)
 vals=_tb_unknown_deck_values(obs);avg=(sum(vals)/len(vals)) if vals else 90
 expected=draw_n*avg
 # Existing exact pieces are worth more than generic average because they can be used now.
 immediate=0
 if _tb_hand_count(obs,TB['BUG']):immediate+=500
 if _tb_hand_count(obs,TB['AREA']) and _tb_stadium_id(obs)!=TB['AREA']:immediate+=220
 if _tb_hand_count(obs,TB['LATIAS']) and not _tb_has(obs,TB['LATIAS']):immediate+=280
 if any(_tb_best_ready_damage(obs,p)>0 for p in _tb_board(obs,True)):immediate+=180
 return expected-(keep+immediate)

def _tb_guaranteed_draws(obs):
 s=obs.current;pl=s.players[s.yourIndex];n=0
 active=pl.active[0] if pl.active else None
 if active and active.id==TB['KANGA']:n+=2
 grass_in_hand=_tb_hand_count(obs,TB['G'])
 unused_teal=sum(1 for p in _tb_board(obs,True) if p.id==TB['TEAL'] and len(p.energyCards or [])<4)
 n+=min(grass_in_hand,unused_teal)
 return n

def _tb_card_value(obs,card,for_discard=False):
 v=_tb_role_value(obs,card,instance=hasattr(card,'energyCards'))
 if for_discard:
  cid=int(getattr(card,'id',0) or 0)
  # Preserve unique engines and matchup tech; duplicate empty bodies are expendable.
  if cid==TB['LATIAS'] and _tb_count(obs,cid)<=1:v+=700
  if cid in {TB['CHIYU'],TB['PECH'],TB['CORNER']} and _tb_matchup(obs) in {'crustle','single_prize','marnie','archaludon','alakazam'}:v+=650
  if cid==TB['TEAL'] and _tb_count(obs,cid)>2 and hasattr(card,'energyCards') and not card.energyCards:v-=260
  if cid==TB['KANGA'] and _tb_count(obs,cid)>1 and hasattr(card,'energyCards') and not card.energyCards:v-=180
 return v

def _tb_main_score(obs,o):
 score=_TB_OLD_MAIN_SCORE(obs,o);c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);t=_target(obs,o);tid=int(getattr(t,'id',0) or 0);turn=int(obs.current.turn or 0)
 m=_tb_matchup(obs);pl=obs.current.players[obs.current.yourIndex]
 if o.type==OptionType.PLAY:
  if cid==TB['BUG']:
   # Bug Catching Set precedes other shuffle/draw actions unless Cipher already set the top deck.
   score+=3200 if turn<=6 else 900
  elif cid==TB['AREA']:
   tera=any(p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']} for p in _tb_board(obs,True))
   score+=1800 if tera and _tb_stadium_id(obs)!=TB['AREA'] else (-500 if _tb_stadium_id(obs)==TB['AREA'] else 100)
  elif cid==TB['LILLIE']:
   draw=8 if len(pl.prize or [])==6 else 6;delta=_tb_cycle_delta(obs,draw)
   score+=delta*1.6
   if _tb_hand_count(obs,TB['BUG']):score-=1700
   if pl.handCount>=7 and delta<0:score-=1800
  elif cid==TB['CIPHER']:
   draws=_tb_guaranteed_draws(obs)
   score+=(1200+draws*500) if draws>=1 else -1200
   if _tb_hand_count(obs,TB['BUG']):score-=1000
   if _tb_hand_count(obs,TB['LILLIE']) and _tb_cycle_delta(obs,8 if len(pl.prize or [])==6 else 6)>250:score-=800
  elif cid==TB['ESWITCH']:
   ready=max([_tb_best_ready_damage(obs,p) for p in _tb_board(obs,True)] or [0])
   score+=900 if ready==0 and any(len(p.energyCards or [])>=2 for p in _tb_board(obs,True)) else 100
  elif cid==TB['NPLAN']:
   active=pl.active[0] if pl.active else None
   score+=1000 if active and _tb_best_ready_damage(obs,active)==0 and sum(len(p.energyCards or []) for p in pl.bench if p)>=2 else -200
  elif cid==TB['TRUMPET']:
   score+=900 if _tb_stadium_id(obs)==TB['AREA'] and any(p.id in {TB['KANGA'],TB['TERAPAGOS']} for p in pl.bench if p) else -350
  elif cid in {TB['ULTRA'],TB['TERA_ORB'],TB['CYRANO']}:
   if not _tb_has(obs,TB['LATIAS']) and turn<=3:score+=500
  elif cid==TB['SWITCH']:
   active=pl.active[0] if pl.active else None
   if active and active.id==TB['KANGA'] and _tb_has(obs,TB['LATIAS']):score-=300
 elif o.type==OptionType.ABILITY:
  if cid==TB['TEAL']:score+=2200
  elif cid==TB['KANGA']:score+=2000
 elif o.type==OptionType.ATTACH:
  if t:
   # Typed manual energy goes to the corresponding tech; Grass first feeds Teal Dance/Terapagos.
   et=int(getattr(c,'id',0) or 0);need=_tb_type(tid)
   if et==need and need:score+=850
   if tid==TB['TEAL'] and et==TB['G']:score+=650-len(t.energyCards or [])*90
   if tid==TB['TERAPAGOS']:score+=480
   if tid==TB['KANGA'] and len(t.energyCards or [])<3:score+=250
 elif o.type==OptionType.RETREAT:
  active=pl.active[0] if pl.active else None
  if active and active.id==TB['KANGA'] and _tb_has(obs,TB['LATIAS']) and max([_tb_best_ready_damage(obs,p) for p in pl.bench if p] or [0])>0:score+=1500
 elif o.type==OptionType.ATTACK:
  active=pl.active[0] if pl.active else None;dmg=_tb_dynamic_damage(obs,active,o.attackId)
  score+=dmg*4
  opp=obs.current.players[1-obs.current.yourIndex];oa=opp.active[0] if opp.active else None
  if oa and dmg>=oa.hp:score+=1600
  if active:
   if active.id==TB['TERAPAGOS']:score+=len([p for p in pl.bench if p])*120
   if active.id in {TB['CHIYU'],TB['PECH']} and m in {'crustle','single_prize'}:score+=1800
   if active.id==TB['CORNER'] and m in {'marnie','archaludon','alakazam'}:score+=2000
   if active.id==TB['PECH_EX'] and m=='single_prize':score+=900
 return score

def _tb_context_score(obs,o):
 ctx=obs.select.context;eff=int(getattr(getattr(obs.select,'effect',None),'id',0) or 0);c=_source(obs,o);cid=int(getattr(c,'id',0) or 0)
 base=_TB_OLD_CONTEXT_SCORE(obs,o)
 # Ciphermaniac: selected deck cards go on top, so high-value immediate roles are positive.
 if ctx==SelectContext.TO_DECK and eff==TB['CIPHER'] and o.type==OptionType.CARD:
  v=_tb_role_value(obs,c)
  draws=_tb_guaranteed_draws(obs)
  if cid==TB['BUG'] and draws<=1:v-=250
  if cid==TB['LATIAS'] and not _tb_has(obs,TB['LATIAS']):v+=700
  if cid==TB['AREA'] and _tb_stadium_id(obs)!=TB['AREA']:v+=450
  if cid==TB['G'] and _tb_hand_count(obs,TB['G'])==0:v+=300
  return v+draws*80
 if o.type==OptionType.CARD:
  # Setup and search target hierarchy.
  if ctx==SelectContext.SETUP_ACTIVE_POKEMON:
   if cid==TB['KANGA']:return 1800
   if cid==TB['LATIAS']:return 950
   if cid==TB['TEAL']:return 850
   if cid==TB['TERAPAGOS']:return 700
  if ctx==SelectContext.SETUP_BENCH_POKEMON:
   if cid==TB['LATIAS']:return 2200
   if cid==TB['TEAL']:return 1700-250*_tb_count(obs,cid)
   if cid==TB['KANGA']:return 1350
   if cid==TB['TERAPAGOS']:return 1200
  if ctx in {SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH}:
   return _tb_role_value(obs,c)
  if ctx in {SelectContext.DISCARD,SelectContext.NOT_MOVE,SelectContext.TO_DECK_BOTTOM}:
   return -_tb_card_value(obs,c,True)
  if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
   if cid==TB['KANGA'] and _tb_has(obs,TB['LATIAS']):return 1600
   # Matchup-specific attacker selection.
   m=_tb_matchup(obs)
   if cid in {TB['CHIYU'],TB['PECH']} and m in {'crustle','single_prize'}:return 2200
   if cid==TB['CORNER'] and m in {'marnie','archaludon','alakazam'}:return 2300
   return _tb_role_value(obs,c,instance=True)+_tb_best_ready_damage(obs,c)*4
  if ctx==SelectContext.ATTACH_FROM:
   # Move surplus energy, not the only typed energy of an active attacker.
   if hasattr(c,'energyCards'):
    ready=_tb_best_ready_damage(obs,c);surplus=len(c.energyCards or [])-(3 if c.id==TB['TEAL'] else 2)
    return surplus*500-ready*2-_tb_role_value(obs,c)*0.1
  if ctx==SelectContext.ATTACH_TO:
   if hasattr(c,'energyCards'):
    before=_tb_best_ready_damage(obs,c)
    # Target active attacker first, then Terapagos/Kangaskhan.
    active=obs.current.players[obs.current.yourIndex].active
    return _tb_role_value(obs,c,instance=True)+(900 if active and c is active[0] else 0)+(700 if before==0 else 0)
 return base

def _card_value(obs,card,for_discard=False):return _tb_card_value(obs,card,for_discard)
def _main_score(obs,o):return _tb_main_score(obs,o)
def _context_score(obs,o):return _tb_context_score(obs,o)

def _choose_count(obs):
 # Area Zero collapse and other discard contexts should discard only the required count;
 # Bug Catching/Ciphermaniac acquisitions use their full legal maximum.
 if obs.select.context in {SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH}:return min(len(obs.select.option),int(obs.select.maxCount))
 return _TB_OLD_CHOOSE_COUNT(obs)

# === Tera Box v2 connected-action planner ===

def _tb_unknown_deck_values(obs):
 counts=Counter(MY_DECK);s=obs.current;pl=s.players[s.yourIndex]
 known=list(pl.hand or [])+list(pl.discard or [])+list(pl.active or [])+list(pl.bench or [])+list(pl.prize or [])
 for c in known:
  cid=int(getattr(c,'id',0) or 0)
  if counts[cid]>0:counts[cid]-=1
 vals=[]
 for cid,n in counts.items():
  if n<=0:continue
  proxy=type('TBCard',(),{'id':cid})()
  vals += [_tb_role_value(obs,proxy)]*n
 return vals

_TB_V1_AGENT=agent
_TB_MEM={'turn':-1,'move':None,'played':set()}

def _tb_energy_types(p):return list(getattr(p,'energies',[]) or [])
def _tb_attacks(p):
 cd=CARDS.get(int(getattr(p,'id',0) or 0));return list(cd.attacks) if cd else []
def _tb_ready_after(p,add=None,remove_index=None):
 class Q:pass
 q=Q();q.id=p.id;q.energies=list(_tb_energy_types(p));q.energyCards=list(getattr(p,'energyCards',[]) or []);q.hp=p.hp;q.maxHp=p.maxHp
 if remove_index is not None and 0<=remove_index<len(q.energies):q.energies.pop(remove_index)
 if add is not None:q.energies.append(add)
 return max([_tb_dynamic_damage(_TB_CUR_OBS,q,aid) for aid in _tb_attacks(q) if _tb_attack_ready(q,aid)] or [0])

def _tb_transfer_plan(obs,active_only=False):
 global _TB_CUR_OBS;_TB_CUR_OBS=obs
 pl=obs.current.players[obs.current.yourIndex];board=[p for p in list(pl.active)+list(pl.bench) if p]
 sources=[p for p in board if _tb_energy_types(p)];targets=[pl.active[0]] if active_only and pl.active else board
 best=None
 oa=obs.current.players[1-obs.current.yourIndex].active;ohp=oa[0].hp if oa and oa[0] else 999
 for sp in sources:
  for ei,et in enumerate(_tb_energy_types(sp)):
   sb=_tb_best_ready_damage(obs,sp);sa=_tb_ready_after(sp,remove_index=ei)
   loss=max(0,sb-sa)
   # Moving the only typed energy from a tech attacker is expensive.
   st=_tb_type(sp.id)
   if st and et==st and _tb_energy_types(sp).count(et)<=1:loss+=220
   for tp in targets:
    if tp is sp:continue
    tb=_tb_best_ready_damage(obs,tp);ta=_tb_ready_after(tp,add=et)
    gain=ta-tb
    if tb==0 and ta>0:gain+=700
    if ta>=ohp and tb<ohp:gain+=900
    if tp.id==TB['TERAPAGOS'] and ta>0:gain+=180
    if tp.id==TB['CORNER'] and _tb_matchup(obs) in {'marnie','archaludon','alakazam'} and ta>0:gain+=350
    if tp.id in {TB['CHIYU'],TB['PECH']} and _tb_matchup(obs) in {'crustle','single_prize'} and ta>0:gain+=450
    val=gain-loss
    if best is None or val>best[0]:best=(val,sp,ei,et,tp,tb,ta)
 return best

def _tb_manual_attach_score(obs,o):
 c=_source(obs,o);t=_target(obs,o)
 if c is None or t is None:return -9999
 et=int(getattr(c,'id',0) or 0);before=_tb_best_ready_damage(obs,t)
 global _TB_CUR_OBS;_TB_CUR_OBS=obs
 after=_tb_ready_after(t,add=et)
 need=_tb_type(t.id);v=(after-before)*6
 if before==0 and after>0:v+=1500
 if need and et==need:v+=850
 elif need and et!=need and len(t.energies or [])==0:v-=450
 if t.id==TB['TEAL'] and et==TB['G']:v+=850-100*len(t.energies or [])
 if t.id==TB['TERAPAGOS']:v+=650
 if t.id==TB['KANGA'] and len(t.energies or [])<3:v+=350
 # Avoid scattering a lone typed energy onto an attacker that remains multiple turns away.
 if after==0 and need and et!=need:v-=500
 return v

def _tb_portfolio_score(obs,c,effect_id,selected_ids):
 cid=int(getattr(c,'id',0) or 0);v=_tb_role_value(obs,c)
 # Search selections must cover different roles before taking duplicates.
 role={TB['LATIAS']:'mobility',TB['TEAL']:'engine',TB['KANGA']:'draw',TB['TERAPAGOS']:'main',TB['CORNER']:'wall',TB['WELLSPRING']:'snipe',TB['CHIYU']:'single',TB['PECH']:'single',TB['PECH_EX']:'late'}.get(cid,'other')
 selected_roles={ {TB['LATIAS']:'mobility',TB['TEAL']:'engine',TB['KANGA']:'draw',TB['TERAPAGOS']:'main',TB['CORNER']:'wall',TB['WELLSPRING']:'snipe',TB['CHIYU']:'single',TB['PECH']:'single',TB['PECH_EX']:'late'}.get(x,'other') for x in selected_ids}
 if role in selected_roles:v-=850
 if cid in selected_ids:v-=1400
 turn=obs.current.turn;m=_tb_matchup(obs)
 if cid==TB['LATIAS'] and not _tb_has(obs,cid):v+=1100
 if cid==TB['TEAL'] and _tb_count(obs,cid)<2:v+=700
 if cid==TB['KANGA'] and not _tb_has(obs,cid):v+=600
 if cid==TB['TERAPAGOS'] and not _tb_has(obs,cid):v+=650
 if cid==TB['CORNER'] and m in {'marnie','archaludon','alakazam'}:v+=1000
 if cid in {TB['CHIYU'],TB['PECH']} and m in {'crustle','single_prize'}:v+=1000
 if effect_id==TB['BUG']:
  if cid==TB['TEAL'] and _tb_count(obs,cid)<2:v+=500
  if cid==TB['G']:v+=450 if _tb_hand_count(obs,TB['G'])<2 else 120
 if effect_id==TB['CIPHER']:
  # Top deck is only valuable if a deterministic draw follows.
  v+=_tb_guaranteed_draws(obs)*120
 return v

def _tb_select_diverse(obs,n):
 eff=int(getattr(getattr(obs.select,'effect',None),'id',0) or 0);remaining=list(range(len(obs.select.option)));out=[];ids=[]
 for _ in range(min(n,len(remaining))):
  best=None
  for i in remaining:
   o=obs.select.option[i];c=_source(obs,o)
   v=_tb_portfolio_score(obs,c,eff,ids)
   if best is None or v>best[0]:best=(v,i,c)
  if best is None:break
  _,i,c=best;out.append(i);remaining.remove(i);ids.append(int(getattr(c,'id',0) or 0))
 return out

def _tb_main_override(obs):
 sel=obs.select;pl=obs.current.players[obs.current.yourIndex];opts=sel.option;m=_tb_matchup(obs)
 # 1) Bug Catching Set is the first shuffle action, before Cipher/Lillie.
 for i,o in enumerate(opts):
  if o.type==OptionType.PLAY:
   c=_source(obs,o)
   if c and c.id==TB['BUG'] and TB['CIPHER'] not in _TB_MEM['played']:return [i]
 # 2) Use deterministic draw abilities before probabilistic hand cycling.
 for wanted in (TB['TEAL'],TB['KANGA']):
  for i,o in enumerate(opts):
   if o.type==OptionType.ABILITY:
    c=_source(obs,o)
    if c and c.id==wanted:
     if wanted!=TB['TEAL'] or _tb_hand_count(obs,TB['G'])>0:return [i]
 # 3) Latias is a unique infrastructure card; deploy it immediately.
 if not _tb_has(obs,TB['LATIAS']):
  for i,o in enumerate(opts):
   if o.type==OptionType.PLAY:
    c=_source(obs,o)
    if c and c.id==TB['LATIAS']:return [i]
 # 4) Area Zero only after a Tera body exists.
 if _tb_stadium_id(obs)!=TB['AREA'] and any(p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']} for p in _tb_board(obs,True)):
  for i,o in enumerate(opts):
   if o.type==OptionType.PLAY:
    c=_source(obs,o)
    if c and c.id==TB['AREA']:return [i]
 # 5) Energy movement starts only if the simulated transfer creates real attack value.
 for i,o in enumerate(opts):
  if o.type==OptionType.PLAY:
   c=_source(obs,o)
   if c and c.id in {TB['ESWITCH'],TB['NPLAN']}:
    plan=_tb_transfer_plan(obs,active_only=(c.id==TB['NPLAN']))
    if plan and plan[0]>=500:
     _TB_MEM['move']=plan;return [i]
 # 6) Ciphermaniac requires a guaranteed draw and must happen after Bug Set.
 for i,o in enumerate(opts):
  if o.type==OptionType.PLAY:
   c=_source(obs,o)
   if c and c.id==TB['CIPHER'] and _tb_guaranteed_draws(obs)>=1 and not _tb_hand_count(obs,TB['BUG']):return [i]
 # 7) Lillie is chosen by expected post-shuffle hand value, not raw hand count.
 for i,o in enumerate(opts):
  if o.type==OptionType.PLAY:
   c=_source(obs,o)
   if c and c.id==TB['LILLIE']:
    draw=8 if len(pl.prize or [])==6 else 6
    if _tb_cycle_delta(obs,draw)>350 and not _tb_hand_count(obs,TB['BUG']):return [i]
 # 8) Manual attachment is a one-step lookahead; use it only when it improves readiness.
 attaches=[]
 for i,o in enumerate(opts):
  if o.type==OptionType.ATTACH:attaches.append((_tb_manual_attach_score(obs,o),i))
 if attaches and max(attaches)[0]>=300:return [max(attaches)[1]]
 # 9) Do not deploy duplicate attackers without a distinct board role.
 plays=[]
 for i,o in enumerate(opts):
  if o.type!=OptionType.PLAY:continue
  c=_source(obs,o)
  if not c or not CARDS.get(c.id) or CARDS[c.id].cardType!=CardType.POKEMON:continue
  cid=c.id;cnt=_tb_count(obs,cid)
  cap={TB['TEAL']:3,TB['TERAPAGOS']:2,TB['KANGA']:1,TB['LATIAS']:1,TB['CORNER']:1,TB['WELLSPRING']:1,TB['CHIYU']:1,TB['PECH']:1,TB['PECH_EX']:1}.get(cid,1)
  if cnt>=cap:continue
  v=_tb_role_value(obs,c)
  if cid in {TB['CHIYU'],TB['PECH']} and m not in {'crustle','single_prize'}:v-=350
  if cid==TB['CORNER'] and m not in {'marnie','archaludon','alakazam'}:v-=250
  plays.append((v,i))
 if plays and max(plays)[0]>=450:return [max(plays)[1]]
 # 10) If a ready attacker is stranded on the Bench, switch/retreat into it.
 active=pl.active[0] if pl.active else None
 ready_bench=[p for p in pl.bench if p and _tb_best_ready_damage(obs,p)>0]
 if active and _tb_best_ready_damage(obs,active)==0 and ready_bench:
  for i,o in enumerate(opts):
   if o.type in {OptionType.RETREAT}:
    return [i]
   if o.type==OptionType.PLAY:
    c=_source(obs,o)
    if c and c.id==TB['SWITCH']:return [i]
 return None

def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  if observation.get('current') is None:
   _TB_MEM.update({'turn':-1,'move':None,'played':set()})
  return list(MY_DECK)
 try:obs=to_observation_class(observation);sel=obs.select
 except Exception:return _TB_V1_AGENT(observation)
 if sel is None or not sel.option:return []
 turn=int(obs.current.turn or 0) if obs.current else -1
 if turn!=_TB_MEM['turn']:_TB_MEM.update({'turn':turn,'move':None,'played':set()})
 # Resolve an Energy Switch/N plan deterministically across follow-up contexts.
 if _TB_MEM.get('move'):
  val,sp,ei,et,tp,_,__=_TB_MEM['move']
  if sel.context==SelectContext.SWITCH_ENERGY_CARD:
   for i,o in enumerate(sel.option):
    c=_source(obs,o)
    if c is not None and int(getattr(c,'id',-1))==int(getattr(sp,'id',-2)) and int(getattr(o,'index',-9))==ei:return [i]
  if sel.context in {SelectContext.ATTACH_FROM,SelectContext.DETACH_FROM}:
   for i,o in enumerate(sel.option):
    c=_source(obs,o)
    if c is not None and int(getattr(c,'id',-1))==int(getattr(sp,'id',-2)):return [i]
  if sel.context==SelectContext.ATTACH_TO:
   for i,o in enumerate(sel.option):
    c=_source(obs,o)
    if c is not None and int(getattr(c,'id',-1))==int(getattr(tp,'id',-2)):
     _TB_MEM['move']=None;return [i]
 # Diverse multi-card searches: Cyrano/Cipher/Bug should not take 3 copies of one role.
 if sel.context in {SelectContext.TO_HAND,SelectContext.TO_DECK,SelectContext.SETUP_BENCH_POKEMON} and sel.maxCount>1:
  n=min(len(sel.option),int(sel.maxCount));return _tb_select_diverse(obs,n)
 if sel.context==SelectContext.MAIN:
  x=_tb_main_override(obs)
  if x:
   try:
    o=sel.option[x[0]];c=_source(obs,o)
    if c and o.type==OptionType.PLAY:_TB_MEM['played'].add(c.id)
   except Exception:pass
   return x
 # Search/switch/discard contexts use the v2 value functions.
 n=_choose_count(obs)
 vals=[]
 for i,o in enumerate(sel.option):
  try:v=_main_score(obs,o) if sel.context==SelectContext.MAIN else _context_score(obs,o)
  except Exception:v=-1e9
  vals.append((v,-i,i))
 vals.sort(reverse=True)
 return [i for _,__,i in vals[:min(n,len(vals))]]

# === Tera Box v3 action-efficiency residual ===
_TB_V2_MAIN_SCORE=_main_score
_TB_V2_GDRAWS=_tb_guaranteed_draws
_TB_V2_AGENT=agent

def _tb_guaranteed_draws(obs):
 pl=obs.current.players[obs.current.yourIndex];n=0
 active=pl.active[0] if pl.active else None
 if active and active.id==TB['KANGA'] and not _TB_MEM.get('kanga_used',False):n+=2
 grass=_tb_hand_count(obs,TB['G'])
 remaining=max(0,sum(1 for p in _tb_board(obs,True) if p.id==TB['TEAL'])-int(_TB_MEM.get('teal_used',0)))
 n+=min(grass,remaining)
 return n

def _tb_main_score(obs,o):
 score=_TB_V2_MAIN_SCORE(obs,o);c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);pl=obs.current.players[obs.current.yourIndex]
 if o.type==OptionType.PLAY and cid in {TB['ESWITCH'],TB['NPLAN']}:
  plan=_tb_transfer_plan(obs,active_only=(cid==TB['NPLAN']))
  if not plan or plan[0]<500:score-=6000
 if o.type==OptionType.PLAY and cid==TB['SWITCH']:
  active=pl.active[0] if pl.active else None;ready=[p for p in pl.bench if p and _tb_best_ready_damage(obs,p)>0]
  draw_kanga=any(p and p.id==TB['KANGA'] for p in pl.bench) and not _TB_MEM.get('kanga_used',False) and _tb_has(obs,TB['LATIAS'])
  if not ready and not draw_kanga:score-=4500
 if o.type==OptionType.RETREAT:
  active=pl.active[0] if pl.active else None;ready=[p for p in pl.bench if p and _tb_best_ready_damage(obs,p)>0]
  if not ready:score-=3500
 if o.type==OptionType.PLAY and cid==TB['CIPHER'] and _tb_guaranteed_draws(obs)<=0:score-=5000
 return score

def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  if observation.get('current') is None:
   _TB_MEM.update({'turn':-1,'move':None,'played':set(),'kanga_used':False,'teal_used':0})
  return list(MY_DECK)
 try:obs=to_observation_class(observation)
 except Exception:return _TB_V2_AGENT(observation)
 turn=int(obs.current.turn or 0) if obs.current else -1
 if turn!=_TB_MEM.get('turn'):
  _TB_MEM.update({'turn':turn,'move':None,'played':set(),'kanga_used':False,'teal_used':0})
 out=_TB_V2_AGENT(observation)
 try:
  if obs.select and out and len(out)==1:
   o=obs.select.option[out[0]];c=_source(obs,o)
   if o.type==OptionType.ABILITY and c:
    if c.id==TB['KANGA']:_TB_MEM['kanga_used']=True
    elif c.id==TB['TEAL']:_TB_MEM['teal_used']=int(_TB_MEM.get('teal_used',0))+1
 except Exception:pass
 return out


# === Tera Box v4 attack-opening and probability planner ===
# The previous layers remain as a generic fallback.  v4 only commits to a search,
# cycle, switch, gust, or energy move when the simulated post-action board has a
# higher attack-ready value than the current board.
_TB_V3_AGENT=agent
_TB_V3_CONTEXT=_context_score
_TB_V3_MAIN=_main_score

_V4={'turn':-1,'teal_used':0,'kanga_used':False,'transfer':None,'own_switch':None,'gust':None,'cipher':None,'played':set(),'last_sig':None,'repeat':0}

_V4_ROLE={TB['LATIAS']:'mobility',TB['TEAL']:'engine',TB['KANGA']:'draw',TB['TERAPAGOS']:'main',TB['CORNER']:'wall',TB['WELLSPRING']:'snipe',TB['CHIYU']:'single',TB['PECH']:'single',TB['PECH_EX']:'late'}

def _v4_pl(obs,own=True):
 s=obs.current;return s.players[s.yourIndex if own else 1-s.yourIndex]
def _v4_locs(obs,own=True):
 pl=_v4_pl(obs,own);out=[]
 for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
  for i,p in enumerate(xs or []):
   if p:out.append((int(area),i,p))
 return out

def _v4_energy_need(p):
 cid=int(getattr(p,'id',0) or 0)
 # Exact typed + total requirements for the preferred attack.
 return {
  TB['TERAPAGOS']:(None,2), TB['KANGA']:(None,3), TB['TEAL']:(TB['G'],3),
  TB['CORNER']:(TB['F'],3), TB['WELLSPRING']:(TB['W'],3), TB['CHIYU']:(TB['R'],2),
  TB['PECH']:(TB['D'],1), TB['PECH_EX']:(TB['D'],2), TB['LATIAS']:(5,3),
 }.get(cid,(None,99))

def _v4_missing(p,energies=None):
 es=list(_tb_energy_types(p) if energies is None else energies);typed,total=_v4_energy_need(p)
 miss_total=max(0,total-len(es));miss_typed=1 if typed and typed not in es else 0
 return miss_typed,miss_total

def _v4_attack_value(obs,p,energies=None):
 if p is None:return 0.0
 # Clone only the fields used by readiness/damage helpers.
 class Q:pass
 q=Q();q.id=p.id;q.energies=list(_tb_energy_types(p) if energies is None else energies);q.energyCards=list(getattr(p,'energyCards',[]) or []);q.hp=p.hp;q.maxHp=p.maxHp
 vals=[];opp=_v4_pl(obs,False);oa=opp.active[0] if opp.active else None;m=_tb_matchup(obs)
 cd=CARDS.get(q.id)
 if not cd:return 0.0
 for aid in cd.attacks:
  if not _tb_attack_ready(q,aid):continue
  dmg=_tb_dynamic_damage(obs,q,aid);v=float(dmg)
  if oa and dmg>=oa.hp:
   oc=CARDS.get(oa.id);v+=650+(250 if oc and (oc.ex or oc.megaEx) else 80)
  if q.id==TB['TERAPAGOS']:v+=35*len([x for x in _v4_pl(obs,True).bench if x])
  if q.id==TB['CORNER'] and m in {'marnie','archaludon','alakazam'}:v+=500
  if q.id in {TB['CHIYU'],TB['PECH']} and m in {'crustle','single_prize'}:v+=520
  if q.id==TB['WELLSPRING'] and m=='dragapult':
   # Bench spread is valuable only when there is a real low-HP target.
   low=sum(1 for x in opp.bench if x and x.hp<=120);v+=min(2,low)*180
  if aid==19:v+=80 # Chi-Yu draw is a fallback, not an attack-ready line.
  vals.append(v)
 return max(vals or [0.0])

def _v4_board_attack_value(obs):
 pl=_v4_pl(obs,True);active=pl.active[0] if pl.active else None
 av=_v4_attack_value(obs,active)
 bench=max([_v4_attack_value(obs,p) for p in pl.bench if p] or [0])
 # A benched attacker is actionable with Latias or a switch card.
 mobile=_tb_has(obs,TB['LATIAS']) or _tb_hand_count(obs,TB['SWITCH'])>0 or _tb_hand_count(obs,TB['PRIME'])>0
 return max(av,bench-(0 if mobile else 280))

def _v4_role_score(obs,c,for_search=True):
 if c is None:return -99999
 cid=int(getattr(c,'id',0) or 0);turn=int(obs.current.turn or 0);m=_tb_matchup(obs);pl=_v4_pl(obs,True)
 cnt=_tb_count(obs,cid);hand=_tb_hand_count(obs,cid);v=_tb_role_value(obs,c)
 # Unique infrastructure and engines.
 if cid==TB['LATIAS']:v+=2200 if cnt==0 and turn<=4 else (-1800 if cnt else 100)
 elif cid==TB['TEAL']:v+=1500 if cnt==0 else 850 if cnt<2 else -250*(cnt-1)
 elif cid==TB['KANGA']:v+=1200 if cnt==0 and turn<=5 else -500*cnt
 elif cid==TB['TERAPAGOS']:v+=1250 if cnt==0 else 250 if cnt==1 else -900
 elif cid==TB['AREA']:v+=850 if _tb_stadium_id(obs)!=TB['AREA'] and any(p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']} for p in _tb_board(obs,True)) else -700
 elif cid==TB['G']:v+=700 if _tb_hand_count(obs,TB['G'])<2 and sum(1 for p in _tb_board(obs,True) if p.id==TB['TEAL'])>0 else 100
 elif cid==TB['CORNER']:v+=1800 if m in {'marnie','archaludon','alakazam'} and cnt==0 else -600 if cnt else 50
 elif cid==TB['WELLSPRING']:v+=1300 if m=='dragapult' and cnt==0 else -500 if cnt else 50
 elif cid in {TB['CHIYU'],TB['PECH']}:
  v+=1600 if m in {'crustle','single_prize'} and cnt==0 else -500 if cnt else 30
 elif cid==TB['PECH_EX']:v+=450 if m=='single_prize' and cnt==0 else -700
 # Search/cycle cards are role-completion tools, not standalone value.
 elif cid==TB['BUG']:v+=900 if turn<=5 else 120
 elif cid in {TB['ULTRA'],TB['TERA_ORB']}:v+=650 if not _tb_has(obs,TB['LATIAS']) or not _tb_has(obs,TB['TERAPAGOS']) else 100
 elif cid==TB['ESWITCH']:v+=350 if any(len(_tb_energy_types(p))>=2 for p in _tb_board(obs,True)) else -250
 elif cid==TB['CIPHER']:v+=400 if _v4_remaining_draws(obs)>0 else -800
 elif cid==TB['LILLIE']:v+=max(0,5-pl.handCount)*170
 # Duplicate in hand has diminishing immediate value.
 if for_search and hand>0 and cid not in {TB['G'],TB['ESWITCH']}:v-=650*hand
 return v

def _v4_remaining_draws(obs):
 pl=_v4_pl(obs,True);n=0
 active=pl.active[0] if pl.active else None
 if active and active.id==TB['KANGA'] and not _V4.get('kanga_used',False):n+=2
 grass=_tb_hand_count(obs,TB['G']);teals=sum(1 for p in _tb_board(obs,True) if p.id==TB['TEAL'])
 n+=min(grass,max(0,teals-int(_V4.get('teal_used',0))))
 return n

def _v4_cycle_gain(obs,draw_n):
 # Expected replacement value from the actual remaining deck versus exact hand pieces.
 hand=[x for x in _tb_hand(obs) if x.id!=TB['LILLIE']]
 keep=sum(max(0,_v4_role_score(obs,x,False)) for x in hand)
 counts=Counter(MY_DECK)
 pl=_v4_pl(obs,True)
 for x in list(pl.hand or [])+list(pl.discard or [])+list(pl.active or [])+list(pl.bench or [])+list(pl.prize or []):
  if counts[int(getattr(x,'id',0) or 0)]>0:counts[int(getattr(x,'id',0) or 0)]-=1
 vals=[]
 for cid,n in counts.items():
  if n<=0:continue
  vals += [max(0,_v4_role_score(obs,type('C',(),{'id':cid})(),False))]*n
 expected=draw_n*(sum(vals)/max(1,len(vals)))
 # Exact ready attack and unique infrastructure should not be shuffled away casually.
 if _v4_board_attack_value(obs)>0:keep+=700
 if _tb_hand_count(obs,TB['LATIAS']) and not _tb_has(obs,TB['LATIAS']):keep+=900
 if _tb_hand_count(obs,TB['AREA']) and _tb_stadium_id(obs)!=TB['AREA']:keep+=450
 if _tb_hand_count(obs,TB['BUG']):keep+=550
 return expected-keep

def _v4_transfer_plan(obs,active_only=False):
 pl=_v4_pl(obs,True);locs=_v4_locs(obs,True);active=pl.active[0] if pl.active else None
 targets=[x for x in locs if (not active_only or x[0]==int(AreaType.ACTIVE))]
 mobile=_tb_has(obs,TB['LATIAS']) or _tb_hand_count(obs,TB['SWITCH'])>0 or _tb_hand_count(obs,TB['PRIME'])>0
 best=None
 for sa,si,sp in locs:
  es=list(_tb_energy_types(sp))
  for ei,e in enumerate(es):
   after_src=es[:ei]+es[ei+1:]
   loss=max(0,_v4_attack_value(obs,sp)-_v4_attack_value(obs,sp,after_src))
   typed,_=_v4_energy_need(sp)
   if typed and e==typed and es.count(e)==1:loss+=700
   # Preserve engines that are one attachment from attacking.
   if sum(_v4_missing(sp,after_src))>sum(_v4_missing(sp,es)):loss+=250
   for ta,ti,tp in targets:
    if sa==ta and si==ti:continue
    before=_v4_attack_value(obs,tp);after_es=list(_tb_energy_types(tp))+[e];after=_v4_attack_value(obs,tp,after_es)
    gain=after-before
    if before<=0<after:gain+=1000
    if ta==int(AreaType.BENCH) and not mobile:gain-=900
    # A 20-damage Sob or zero-damage draw attack does not justify an Energy Switch.
    if after<60 and tp.id not in {TB['PECH']}:gain-=1000
    miss0=sum(_v4_missing(tp));miss1=sum(_v4_missing(tp,after_es));gain+=(miss0-miss1)*180
    val=gain-loss
    item=(val,sa,si,ei,e,ta,ti,tp.id,before,after)
    if best is None or item[0]>best[0]:best=item
 return best

def _v4_manual_attach(obs,o):
 e=_source(obs,o);t=_target(obs,o)
 if e is None or t is None:return -99999
 eid=int(e.id);es=list(_tb_energy_types(t));before=_v4_attack_value(obs,t);after=_v4_attack_value(obs,t,es+[eid]);typed,total=_v4_energy_need(t)
 v=(after-before)*6+(1300 if before<=0<after else 0)
 if typed and eid==typed and typed not in es:v+=1000
 if typed and eid!=typed and typed not in es:v-=500
 miss0=sum(_v4_missing(t,es));miss1=sum(_v4_missing(t,es+[eid]));v+=(miss0-miss1)*500
 if t.id==TB['TEAL'] and eid==TB['G']:
  # Manual Grass on Teal is useful as an energy bank, but not over opening an attacker.
  v+=350-120*len(es)
 if t.id in {TB['TERAPAGOS'],TB['KANGA']}:v+=250
 return v

def _v4_best_attacker(obs,include_bench=True):
 pl=_v4_pl(obs,True);locs=_v4_locs(obs,True);best=None
 for a,i,p in locs:
  if not include_bench and a!=int(AreaType.ACTIVE):continue
  v=_v4_attack_value(obs,p)
  if best is None or v>best[0]:best=(v,a,i,p.id)
 return best

def _v4_best_gust(obs):
 pl=_v4_pl(obs,False);att=_v4_best_attacker(obs,True);dmg=att[0] if att else 0;m=_tb_matchup(obs);best=None
 for i,p in enumerate(pl.bench or []):
  if not p:continue
  cd=CARDS.get(p.id);pr=2 if cd and cd.ex else 3 if cd and cd.megaEx else 1
  ko=dmg>=p.hp and dmg>0
  evo_bonus=900 if p.id in {119,120,169,646,647,741,742,344} else 0
  v=(1800+pr*500 if ko else 0)+evo_bonus+len(p.energyCards or [])*120-p.hp*.15
  if best is None or v>best[0]:best=(v,i,p.id,ko)
 return best

def _v4_search_score(obs,o,selected=None):
 c=_source(obs,o);v=_v4_role_score(obs,c,True);cid=int(getattr(c,'id',0) or 0);selected=selected or []
 role=_V4_ROLE.get(cid,'other')
 if cid in selected:v-=2200
 if role in {_V4_ROLE.get(x,'other') for x in selected}:v-=900
 return v

def _v4_discard_score(obs,o):
 c=_source(obs,o)
 if c is None:return -99999
 cid=int(c.id);v=-_v4_role_score(obs,c,False)
 # Keep unique engines, typed energy, and active attack pieces.
 if cid==TB['LATIAS'] and _tb_count(obs,cid)<=1:v-=1800
 if cid in {TB['CORNER'],TB['WELLSPRING'],TB['CHIYU'],TB['PECH']} and _tb_count(obs,cid)<=1:v-=1000
 if cid in {TB['R'],TB['W'],TB['F'],TB['D']} and MY_DECK.count(cid)<=1:v-=1300
 if cid==TB['TEAL'] and _tb_count(obs,cid)>=3:v+=800
 if cid==TB['TERAPAGOS'] and _tb_count(obs,cid)>=2:v+=700
 if cid==TB['KANGA'] and _tb_count(obs,cid)>=1:v+=500
 return v

def _v4_main(obs):
 sel=obs.select;opts=sel.option;pl=_v4_pl(obs,True);turn=int(obs.current.turn or 0);m=_tb_matchup(obs)
 # Exact KO attack takes precedence over optional deck manipulation.
 attacks=[]
 for i,o in enumerate(opts):
  if o.type==OptionType.ATTACK:
   a=pl.active[0] if pl.active else None;v=_v4_attack_value(obs,a)
   attacks.append((v,i))
 oa=_v4_pl(obs,False).active[0] if _v4_pl(obs,False).active else None
 if attacks and oa and max(attacks)[0]>=oa.hp+600:return [max(attacks)[1]]
 # User-requested Bug Catching Set priority. It also makes later probability estimates cleaner.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==TB['BUG']:return [i]
 # Draw engines before topdeck or shuffle cycles.
 for wanted in (TB['KANGA'],TB['TEAL']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.ABILITY and c and c.id==wanted:
    if wanted==TB['KANGA'] or _tb_hand_count(obs,TB['G'])>0:return [i]
 # Latias infrastructure and Area Zero.
 if not _tb_has(obs,TB['LATIAS']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['LATIAS']:return [i]
 tera=any(p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']} for p in _tb_board(obs,True))
 if tera and _tb_stadium_id(obs)!=TB['AREA']:
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['AREA']:return [i]
 # Deploy only missing board roles. Search comes before duplicate bodies.
 search=[]
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id in {TB['ULTRA'],TB['TERA_ORB'],TB['CYRANO']}:
   # Search is useful only while a required role is missing.
   missing=(not _tb_has(obs,TB['LATIAS']) or _tb_count(obs,TB['TEAL'])<2 or not _tb_has(obs,TB['TERAPAGOS']) or (m in {'crustle','single_prize'} and not any(_tb_has(obs,x) for x in {TB['CHIYU'],TB['PECH']})) or (m in {'marnie','archaludon','alakazam'} and not _tb_has(obs,TB['CORNER'])))
   if missing:search.append((900 if c.id==TB['TERA_ORB'] else 800,i))
 if search:return [max(search)[1]]
 # Play missing board role Pokémon.
 bodies=[]
 for i,o in enumerate(opts):
  if o.type!=OptionType.PLAY:continue
  c=_source(obs,o);cd=CARDS.get(int(getattr(c,'id',0) or 0)) if c else None
  if not c or not cd or cd.cardType!=CardType.POKEMON:continue
  cid=c.id;cap={TB['TEAL']:2,TB['TERAPAGOS']:1,TB['KANGA']:1,TB['LATIAS']:1,TB['CORNER']:1,TB['WELLSPRING']:1,TB['CHIYU']:1,TB['PECH']:1,TB['PECH_EX']:1}.get(cid,1)
  if _tb_count(obs,cid)>=cap:continue
  v=_v4_role_score(obs,c,False)
  # Do not expose irrelevant tech bodies.
  if cid==TB['CORNER'] and m not in {'marnie','archaludon','alakazam'}:v-=1000
  if cid in {TB['CHIYU'],TB['PECH']} and m not in {'crustle','single_prize'}:v-=900
  if cid==TB['WELLSPRING'] and m!='dragapult':v-=600
  bodies.append((v,i))
 if bodies and max(bodies)[0]>=900:return [max(bodies)[1]]
 # Manual attachment before movement; prefer actually opening an attack.
 at=[(_v4_manual_attach(obs,o),i) for i,o in enumerate(opts) if o.type==OptionType.ATTACH]
 if at and max(at)[0]>=650:return [max(at)[1]]
 # Energy movement only when post-transfer attack value is materially higher.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id in {TB['ESWITCH'],TB['NPLAN']}:
   plan=_v4_transfer_plan(obs,active_only=(c.id==TB['NPLAN']))
   threshold=900 if c.id==TB['ESWITCH'] else 700
   if plan and plan[0]>=threshold:
    _V4['transfer']=plan;return [i]
 # Cipher only if its two cards will actually be drawn before another shuffle.
 if _v4_remaining_draws(obs)>0 and not _tb_hand_count(obs,TB['BUG']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['CIPHER']:
    _V4['cipher']=True;return [i]
 # Lillie after consumable setup; compare exact hand to deck expectation.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==TB['LILLIE']:
   draw=8 if len(pl.prize or [])==6 else 6
   if _v4_cycle_gain(obs,draw)>900 and not _tb_hand_count(obs,TB['BUG']):return [i]
 # Prime/Boss only with a meaningful gust and a ready own attacker.
 gust=_v4_best_gust(obs);att=_v4_best_attacker(obs,True)
 if gust and att and gust[0]>=1200 and att[0]>0:
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id in {TB['PRIME'],TB['BOSS']}:
    if c.id==TB['PRIME']:
     # Prime must have a ready own switch target, not just a desirable opponent.
     if att[1]!=int(AreaType.ACTIVE):_V4['own_switch']=(att[1],att[2],att[3])
     elif _v4_attack_value(obs,pl.active[0])<=0:continue
    _V4['gust']=(int(AreaType.BENCH),gust[1],gust[2]);return [i]
 # Switch/retreat into the highest ready attacker, never into an unpowered tech.
 active=pl.active[0] if pl.active else None;best=_v4_best_attacker(obs,True)
 if best and best[0]>_v4_attack_value(obs,active)+100 and best[1]!=int(AreaType.ACTIVE):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['SWITCH']:
    _V4['own_switch']=(best[1],best[2],best[3]);return [i]
  for i,o in enumerate(opts):
   if o.type==OptionType.RETREAT:
    _V4['own_switch']=(best[1],best[2],best[3]);return [i]
 # If an attack is available, stop rotating cards and attack.
 if attacks and max(attacks)[0]>0:return [max(attacks)[1]]
 return None

def _v4_context(obs):
 sel=obs.select;ctx=sel.context;opts=sel.option;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
 # Resolve transfer by exact area/index, preventing donor/target drift across contexts.
 plan=_V4.get('transfer')
 if plan:
  _,sa,si,ei,e,ta,ti,tid,_,__=plan
  if ctx==SelectContext.SWITCH_ENERGY_CARD:
   for i,o in enumerate(opts):
    if int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-2))==si:return [i]
  if ctx in {SelectContext.ATTACH_FROM,SelectContext.DETACH_FROM}:
   for i,o in enumerate(opts):
    if int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-2))==si:return [i]
  if ctx==SelectContext.ATTACH_TO:
   for i,o in enumerate(opts):
    if int(getattr(o,'area',-1))==ta and int(getattr(o,'index',-2))==ti:
     _V4['transfer']=None;return [i]
 # Prime/Switch own target.
 own=_V4.get('own_switch')
 if own and ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
  a,idx,cid=own
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if int(getattr(o,'area',-1))==a and int(getattr(o,'index',-2))==idx:
    _V4['own_switch']=None;return [i]
  # Fallback by ID only if index changed after opponent gust.
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if c and c.id==cid:_V4['own_switch']=None;return [i]
 # Opponent gust target.
 gust=_V4.get('gust')
 if gust and ctx in {SelectContext.EFFECT_TARGET,SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
  _,idx,cid=gust
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if c and c.id==cid and int(getattr(o,'playerIndex',obs.current.yourIndex))!=obs.current.yourIndex:
    _V4['gust']=None;return [i]
 # Search and Cipher portfolio.
 if ctx in {SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH,SelectContext.SETUP_BENCH_POKEMON}:
  n=min(len(opts),int(sel.maxCount));selected=[];out=[];rem=list(range(len(opts)))
  for _ in range(n):
   if not rem:break
   best=max(rem,key=lambda i:_v4_search_score(obs,opts[i],selected));c=_source(obs,opts[best]);out.append(best);rem.remove(best);selected.append(int(getattr(c,'id',0) or 0))
  return out[:max(int(sel.minCount),min(n,len(out)))]
 if ctx==SelectContext.TO_DECK and eff==TB['CIPHER']:
  n=min(2,len(opts),int(sel.maxCount));rank=sorted(range(len(opts)),key=lambda i:_v4_search_score(obs,opts[i],[]),reverse=True);_V4['cipher']=None;return rank[:n]
 # Setup Active favors Kanga draw, then Latias mobility, then Teal/Terapagos.
 if ctx==SelectContext.SETUP_ACTIVE_POKEMON:
  rank={TB['KANGA']:4000,TB['LATIAS']:3200,TB['TEAL']:2600,TB['TERAPAGOS']:2200,TB['CHIYU']:900,TB['PECH']:700,TB['CORNER']:650,TB['WELLSPRING']:600,TB['PECH_EX']:400}
  return [max(range(len(opts)),key=lambda i:rank.get(int(getattr(_source(obs,opts[i]),'id',0) or 0),0))]
 # Select actual ready attacker after Switch/Retreat even without explicit intent.
 if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
  return [max(range(len(opts)),key=lambda i:_v4_attack_value(obs,_source(obs,opts[i]))+(_v4_role_score(obs,_source(obs,opts[i]),False)*.08))]
 # Discard/collapse/cost: discard lowest retained value. NOT_MOVE in this engine is
 # also a removal-selection context, so the same lowest-value ordering is used.
 if ctx in {SelectContext.DISCARD,SelectContext.NOT_MOVE,SelectContext.TO_DECK_BOTTOM,SelectContext.DISCARD_ENERGY_CARD,SelectContext.DISCARD_CARD_OR_ATTACHED_CARD}:
  n=max(int(sel.minCount),1 if int(sel.maxCount)>0 else 0);rank=sorted(range(len(opts)),key=lambda i:_v4_discard_score(obs,opts[i]),reverse=True);return rank[:min(n,len(rank))]
 # Glass Trumpet / generic attach target: make Terapagos or Kanga ready, not Latias.
 if ctx==SelectContext.ATTACH_TO:
  return [max(range(len(opts)),key=lambda i:_v4_attack_value(obs,_source(obs,opts[i]))+700*(int(getattr(_source(obs,opts[i]),'id',0) or 0) in {TB['TERAPAGOS'],TB['KANGA']}))]
 # Boss/target selection without an explicit intent.
 if ctx==SelectContext.EFFECT_TARGET:
  opp=_v4_pl(obs,False);dmg=(_v4_best_attacker(obs,True) or (0,))[0]
  def tv(i):
   c=_source(obs,opts[i]);cd=CARDS.get(c.id) if c else None
   return (1200 if c and dmg>=c.hp else 0)+(500 if cd and (cd.ex or cd.megaEx) else 0)+(700 if c and c.id in {119,120,169,646,647,741,742,344} else 0)
  return [max(range(len(opts)),key=tv)]
 return None

def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  if observation.get('current') is None:_V4.update({'turn':-1,'teal_used':0,'kanga_used':False,'transfer':None,'own_switch':None,'gust':None,'cipher':None,'played':set(),'last_sig':None,'repeat':0})
  return list(MY_DECK)
 try:obs=to_observation_class(observation)
 except Exception:return _TB_V3_AGENT(observation)
 if obs.select is None or not obs.select.option:return []
 turn=int(obs.current.turn or 0) if obs.current else -1
 if turn!=_V4.get('turn'):_V4.update({'turn':turn,'teal_used':0,'kanga_used':False,'transfer':None,'own_switch':None,'gust':None,'cipher':None,'played':set(),'last_sig':None,'repeat':0})
 out=_v4_main(obs) if obs.select.context==SelectContext.MAIN else _v4_context(obs)
 if out is None:out=_TB_V3_AGENT(observation)
 # Track abilities and repeated no-progress actions.
 try:
  if len(out)==1 and 0<=out[0]<len(obs.select.option):
   o=obs.select.option[out[0]];c=_source(obs,o)
   if o.type==OptionType.ABILITY and c:
    if c.id==TB['TEAL']:_V4['teal_used']+=1
    elif c.id==TB['KANGA']:_V4['kanga_used']=True
   if o.type==OptionType.PLAY and c:_V4['played'].add(c.id)
   sig=(int(obs.select.context),int(o.type),int(getattr(c,'id',0) or 0),int(getattr(o,'attackId',0) or 0))
   if sig==_V4.get('last_sig'):_V4['repeat']+=1
   else:_V4['repeat']=0;_V4['last_sig']=sig
 except Exception:pass
 return out

# === v4.1 context fixes ===
_V4_OLD_CONTEXT=_v4_context

def _v4_context(obs):
 sel=obs.select;ctx=sel.context;opts=sel.option;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
 plan=_V4.get('transfer')
 if plan and eff==TB['ESWITCH']:
  _,sa,si,ei,e,ta,ti,tid,_,__=plan
  if ctx==SelectContext.SWITCH_ENERGY_CARD:
   # Options identify the Pokémon/energy card carrying the selected Energy.
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if c is not None and int(getattr(c,'id',-1))==int(_v4_pl(obs,True).active[si].id if sa==int(AreaType.ACTIVE) and si<len(_v4_pl(obs,True).active) and _v4_pl(obs,True).active[si] else getattr(c,'id',-2)) and int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-2))==si:
     return [i]
   for i,o in enumerate(opts):
    if int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-2))==si:return [i]
  if ctx in {SelectContext.ATTACH_FROM,SelectContext.ATTACH_TO}:
   # In this engine Energy Switch uses ATTACH_FROM for the destination Pokémon.
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if int(getattr(o,'area',-1))==ta and int(getattr(o,'index',-2))==ti:
     _V4['transfer']=None;return [i]
    if c is not None and int(getattr(c,'id',-1))==tid:
     _V4['transfer']=None;return [i]
  # Do not let a stale transfer intent leak into unrelated contexts.
  if ctx not in {SelectContext.SWITCH_ENERGY_CARD,SelectContext.ATTACH_FROM,SelectContext.ATTACH_TO}:
   _V4['transfer']=None
 # Glass Trumpet: choose Colorless attackers that get closer to a real attack.
 if eff==TB['TRUMPET'] and ctx in {SelectContext.ATTACH_FROM,SelectContext.ATTACH_TO}:
  pokemon_opts=[i for i,o in enumerate(opts) if hasattr(_source(obs,o),'hp')]
  if pokemon_opts:
   return [max(pokemon_opts,key=lambda i:(_v4_attack_value(obs,_source(obs,opts[i]))+800*(int(getattr(_source(obs,opts[i]),'id',0) or 0) in {TB['TERAPAGOS'],TB['KANGA']})))]
  # Energy choice from discard: typed energy is not required for Colorless targets.
  return _TB_V3_AGENT({'current':None}) if False else None
 # Teal Dance and other energy-card selections must remain in the proven base policy.
 if ctx==SelectContext.ATTACH_TO and not any(hasattr(_source(obs,o),'hp') for o in opts):
  return None
 return _V4_OLD_CONTEXT(obs)

# Pecharunt's 20-damage attack is not a reason to spend Energy Switch unless it is
# the matchup's deliberate one-prize line and the moved Energy is Darkness.
_V4_OLD_TRANSFER=_v4_transfer_plan
def _v4_transfer_plan(obs,active_only=False):
 best=_V4_OLD_TRANSFER(obs,active_only)
 if best and best[7]==TB['PECH'] and best[4]!=TB['D']:
  # Recompute while making the illegal/meaningless line unattractive by rejecting it.
  return None
 return best

# === v4.2 safe fallback and strict attack-opening transfer ===
_V42_OLD_AGENT=agent

def _v4_transfer_plan(obs,active_only=False):
 pl=_v4_pl(obs,True);locs=_v4_locs(obs,True);targets=[x for x in locs if (not active_only or x[0]==int(AreaType.ACTIVE))]
 mobile=_tb_has(obs,TB['LATIAS']) or _tb_hand_count(obs,TB['SWITCH'])>0 or _tb_hand_count(obs,TB['PRIME'])>0
 best=None
 for sa,si,sp in locs:
  es=list(_tb_energy_types(sp))
  for ei,e in enumerate(es):
   src_after=es[:ei]+es[ei+1:]
   loss=max(0,_v4_attack_value(obs,sp)-_v4_attack_value(obs,sp,src_after))
   typed,_=_v4_energy_need(sp)
   if typed and e==typed and es.count(e)==1:loss+=900
   for ta,ti,tp in targets:
    if sa==ta and si==ti:continue
    before=_v4_attack_value(obs,tp);after=_v4_attack_value(obs,tp,list(_tb_energy_types(tp))+[e])
    # Energy movement is committed only if the target becomes a real attacker.
    if after<60 or after<=before:continue
    if ta==int(AreaType.BENCH) and not mobile:continue
    gain=(after-before)+(1100 if before<60<=after else 0)
    oa=_v4_pl(obs,False).active[0] if _v4_pl(obs,False).active else None
    if oa and before<oa.hp<=after:gain+=1000
    val=gain-loss
    z=(val,sa,si,ei,e,ta,ti,tp.id,before,after)
    if best is None or z[0]>best[0]:best=z
 return best

def _v4_safe_main(obs):
 sel=obs.select;pl=_v4_pl(obs,True);m=_tb_matchup(obs);vals=[]
 active=pl.active[0] if pl.active else None;current=_v4_attack_value(obs,active);bestbench=_v4_best_attacker(obs,True)
 gust=_v4_best_gust(obs)
 for i,o in enumerate(sel.option):
  c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);v=_TB_V3_MAIN(obs,o)
  if o.type==OptionType.ATTACK:
   v+=_v4_attack_value(obs,active)*8
  elif o.type==OptionType.ATTACH:
   v+=_v4_manual_attach(obs,o)
  elif o.type==OptionType.PLAY and cid in {TB['ESWITCH'],TB['NPLAN']}:
   p=_v4_transfer_plan(obs,active_only=(cid==TB['NPLAN']))
   if not p or p[0]<700:v=-1e8
   else:v+=p[0];_V4['transfer']=p
  elif o.type==OptionType.PLAY and cid==TB['CIPHER']:
   if _v4_remaining_draws(obs)<=0 or _tb_hand_count(obs,TB['BUG']):v=-1e8
  elif o.type==OptionType.PLAY and cid==TB['SWITCH']:
   if not bestbench or bestbench[1]==int(AreaType.ACTIVE) or bestbench[0]<=current+100:v=-1e8
   else:_V4['own_switch']=(bestbench[1],bestbench[2],bestbench[3]);v+=1000
  elif o.type==OptionType.RETREAT:
   if not bestbench or bestbench[1]==int(AreaType.ACTIVE) or bestbench[0]<=current+100:v=-1e8
   else:_V4['own_switch']=(bestbench[1],bestbench[2],bestbench[3]);v+=1000
  elif o.type==OptionType.PLAY and cid in {TB['BOSS'],TB['PRIME']}:
   if not gust or gust[0]<1200 or not bestbench or bestbench[0]<=0:v=-1e8
   else:
    _V4['gust']=(int(AreaType.BENCH),gust[1],gust[2]);v+=gust[0]
    if cid==TB['PRIME'] and bestbench[1]!=int(AreaType.ACTIVE):_V4['own_switch']=(bestbench[1],bestbench[2],bestbench[3])
  elif o.type==OptionType.ABILITY and cid==TB['PECH_EX']:
   # Subjugating Chains only if a Darkness attacker can attack immediately.
   ready=any(p.id==TB['PECH'] and _v4_attack_value(obs,p)>=20 for p in pl.bench if p)
   if not ready:v=-1e8
  elif o.type==OptionType.PLAY and c and CARDS.get(cid) and CARDS[cid].cardType==CardType.POKEMON:
   cap={TB['TEAL']:2,TB['TERAPAGOS']:1,TB['KANGA']:1,TB['LATIAS']:1,TB['CORNER']:1,TB['WELLSPRING']:1,TB['CHIYU']:1,TB['PECH']:1,TB['PECH_EX']:1}.get(cid,1)
   if _tb_count(obs,cid)>=cap:v-=4000
   if cid==TB['CORNER'] and m not in {'marnie','archaludon','alakazam'}:v-=2500
   if cid in {TB['CHIYU'],TB['PECH']} and m not in {'crustle','single_prize'}:v-=2200
   if cid==TB['WELLSPRING'] and m!='dragapult':v-=1800
  elif o.type==OptionType.END:
   v=0 if current<=0 else -500
  vals.append((v,-i,i))
 vals.sort(reverse=True)
 return [vals[0][2]] if vals else []

# Replace the final agent so a rejected action never leaks back from v3.
def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  if observation.get('current') is None:_V4.update({'turn':-1,'teal_used':0,'kanga_used':False,'transfer':None,'own_switch':None,'gust':None,'cipher':None,'played':set(),'last_sig':None,'repeat':0})
  return list(MY_DECK)
 try:obs=to_observation_class(observation)
 except Exception:return _TB_V3_AGENT(observation)
 if obs.select is None or not obs.select.option:return []
 turn=int(obs.current.turn or 0) if obs.current else -1
 if turn!=_V4.get('turn'):_V4.update({'turn':turn,'teal_used':0,'kanga_used':False,'transfer':None,'own_switch':None,'gust':None,'cipher':None,'played':set(),'last_sig':None,'repeat':0})
 if obs.select.context==SelectContext.MAIN:
  out=_v4_main(obs)
  if out is None:out=_v4_safe_main(obs)
 else:
  out=_v4_context(obs)
  if out is None:out=_TB_V3_AGENT(observation)
 try:
  if len(out)==1 and 0<=out[0]<len(obs.select.option):
   o=obs.select.option[out[0]];c=_source(obs,o)
   if o.type==OptionType.ABILITY and c:
    if c.id==TB['TEAL']:_V4['teal_used']+=1
    elif c.id==TB['KANGA']:_V4['kanga_used']=True
   if o.type==OptionType.PLAY and c:_V4['played'].add(c.id)
 except Exception:pass
 return out

# === v4.3 non-recursive safe fallback ===
def _v4_safe_main(obs):
 sel=obs.select;pl=_v4_pl(obs,True);m=_tb_matchup(obs);vals=[]
 active=pl.active[0] if pl.active else None;current=_v4_attack_value(obs,active);bestbench=_v4_best_attacker(obs,True);gust=_v4_best_gust(obs)
 for i,o in enumerate(sel.option):
  c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);v=-500.0
  if o.type==OptionType.ATTACK:v=2000+_v4_attack_value(obs,active)*12
  elif o.type==OptionType.ATTACH:v=800+_v4_manual_attach(obs,o)
  elif o.type==OptionType.ABILITY:
   v=1300
   if cid==TB['TEAL']:v+=800 if _tb_hand_count(obs,TB['G']) else -1800
   elif cid==TB['KANGA']:v+=850
   elif cid==TB['PECH_EX']:
    ready=any(p.id==TB['PECH'] and _v4_attack_value(obs,p)>=20 for p in pl.bench if p)
    v=600 if ready else -1e8
  elif o.type==OptionType.PLAY:
   v=400+_v4_role_score(obs,c,False)
   if cid==TB['BUG']:v=4000
   elif cid==TB['AREA']:
    tera=any(p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']} for p in _tb_board(obs,True));v=2300 if tera and _tb_stadium_id(obs)!=TB['AREA'] else -1e8
   elif cid in {TB['ULTRA'],TB['TERA_ORB'],TB['CYRANO']}:v=1200
   elif cid==TB['LILLIE']:
    draw=8 if len(pl.prize or [])==6 else 6;v=900+_v4_cycle_gain(obs,draw) if not _tb_hand_count(obs,TB['BUG']) else -1e8
   elif cid==TB['CIPHER']:v=1000+300*_v4_remaining_draws(obs) if _v4_remaining_draws(obs)>0 and not _tb_hand_count(obs,TB['BUG']) else -1e8
   elif cid in {TB['ESWITCH'],TB['NPLAN']}:
    p=_v4_transfer_plan(obs,active_only=(cid==TB['NPLAN']))
    if not p or p[0]<700:v=-1e8
    else:v=900+p[0];_V4['transfer']=p
   elif cid==TB['TRUMPET']:
    # Trumpet is useful only with a colorless target missing energy and energy in discard.
    disc=pl.discard or [];has_e=any(CARDS.get(x.id) and CARDS[x.id].cardType==int(CARDS[x.id].cardType)==5 for x in disc)
    tgt=any(p.id in {TB['TERAPAGOS'],TB['KANGA']} and sum(_v4_missing(p))>0 for p in pl.bench if p)
    v=1500 if has_e and tgt else -1e8
   elif cid==TB['SWITCH']:
    if not bestbench or bestbench[1]==int(AreaType.ACTIVE) or bestbench[0]<=current+100:v=-1e8
    else:v=1400+bestbench[0]-current;_V4['own_switch']=(bestbench[1],bestbench[2],bestbench[3])
   elif cid in {TB['BOSS'],TB['PRIME']}:
    if not gust or gust[0]<1200 or not bestbench or bestbench[0]<=0:v=-1e8
    else:
     v=1000+gust[0];_V4['gust']=(int(AreaType.BENCH),gust[1],gust[2])
     if cid==TB['PRIME'] and bestbench[1]!=int(AreaType.ACTIVE):_V4['own_switch']=(bestbench[1],bestbench[2],bestbench[3])
   elif c and CARDS.get(cid) and CARDS[cid].cardType==CardType.POKEMON:
    cap={TB['TEAL']:2,TB['TERAPAGOS']:1,TB['KANGA']:1,TB['LATIAS']:1,TB['CORNER']:1,TB['WELLSPRING']:1,TB['CHIYU']:1,TB['PECH']:1,TB['PECH_EX']:1}.get(cid,1)
    if _tb_count(obs,cid)>=cap:v-=5000
    if cid==TB['CORNER'] and m not in {'marnie','archaludon','alakazam'}:v-=2500
    if cid in {TB['CHIYU'],TB['PECH']} and m not in {'crustle','single_prize'}:v-=2200
    if cid==TB['WELLSPRING'] and m!='dragapult':v-=1800
  elif o.type==OptionType.RETREAT:
   if not bestbench or bestbench[1]==int(AreaType.ACTIVE) or bestbench[0]<=current+100:v=-1e8
   else:v=1200+bestbench[0]-current;_V4['own_switch']=(bestbench[1],bestbench[2],bestbench[3])
  elif o.type==OptionType.END:v=0 if current<=0 else -700
  vals.append((v,-i,i))
 vals.sort(reverse=True);return [vals[0][2]] if vals else []

# === v4.4 utility fixes ===
def _v4_missing_role(obs):
 m=_tb_matchup(obs)
 if not _tb_has(obs,TB['LATIAS']):return True
 if _tb_count(obs,TB['TEAL'])<2:return True
 if not _tb_has(obs,TB['TERAPAGOS']) and not _tb_has(obs,TB['KANGA']):return True
 if m in {'crustle','single_prize'} and not any(_tb_has(obs,x) for x in {TB['CHIYU'],TB['PECH']}):return True
 if m in {'marnie','archaludon','alakazam'} and not _tb_has(obs,TB['CORNER']):return True
 if m=='dragapult' and not _tb_has(obs,TB['WELLSPRING']):return True
 return False

_V44_PREV_SAFE=_v4_safe_main
def _v4_safe_main(obs):
 # Copy the previous result, but veto search cards when all required board roles exist,
 # and avoid Glass Trumpet when no basic Energy is in the discard.
 sel=obs.select;out=_V44_PREV_SAFE(obs)
 if not out:return out
 o=sel.option[out[0]];c=_source(obs,o);cid=int(getattr(c,'id',0) or 0)
 if o.type==OptionType.PLAY and cid in {TB['ULTRA'],TB['TERA_ORB'],TB['CYRANO']} and not _v4_missing_role(obs):
  # Re-rank after banning the redundant search action.
  saved=[]
  for i,x in enumerate(sel.option):
   cc=_source(obs,x);xx=int(getattr(cc,'id',0) or 0)
   if x.type==OptionType.PLAY and xx in {TB['ULTRA'],TB['TERA_ORB'],TB['CYRANO']}:continue
   saved.append(i)
  if saved:
   # Temporarily choose the highest explicit safe category.
   attacks=[i for i in saved if sel.option[i].type==OptionType.ATTACK]
   if attacks:return [max(attacks,key=lambda i:_v4_attack_value(obs,_v4_pl(obs,True).active[0] if _v4_pl(obs,True).active else None))]
   ends=[i for i in saved if sel.option[i].type==OptionType.END]
   # Let v4 main identify another useful deterministic action first.
   alt=_v4_main(obs)
   if alt and alt[0] in saved:return alt
   if ends:return [ends[0]]
 if o.type==OptionType.PLAY and cid==TB['TRUMPET']:
  pl=_v4_pl(obs,True);has_e=any(CARDS.get(x.id) and int(CARDS[x.id].cardType)==5 for x in (pl.discard or []))
  if not has_e:
   ends=[i for i,x in enumerate(sel.option) if x.type==OptionType.END]
   if ends:return [ends[0]]
 return out

# === Tera Box v6 matchup-specialist attack plan ===
_V6_OLD_MAIN=_v4_main
_V6_OLD_SEARCH=_v4_search_score
_V6_OLD_ATTACH=_v4_manual_attach
_V6_OLD_TRANSFER=_v4_transfer_plan

def _v6_primary(obs):
 m=_tb_matchup(obs)
 if m in {'marnie','archaludon','alakazam'}:return TB['CORNER']
 if m=='spidops':return TB['CORNER']
 if m=='crustle':return TB['CHIYU']
 if m=='dragapult':return TB['WELLSPRING']
 return TB['TERAPAGOS']

def _v6_secondary(obs):
 m=_tb_matchup(obs)
 if m=='crustle':return TB['PECH']
 if m in {'marnie','archaludon','alakazam','spidops'}:return TB['TERAPAGOS']
 return TB['KANGA']

def _v6_find(obs,cid):
 for a,i,p in _v4_locs(obs,True):
  if p.id==cid:return a,i,p
 return None

def _v6_progress(obs,p,add=None):
 es=list(_tb_energy_types(p));
 if add is not None:es.append(add)
 mt,ma=_v4_missing(p,es)
 # Lower is better; typed requirement dominates.
 return mt*4+ma

def _v4_search_score(obs,o,selected=None):
 v=_V6_OLD_SEARCH(obs,o,selected);c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);p=_v6_primary(obs);s=_v6_secondary(obs)
 if cid==p and not _tb_has(obs,p):v+=5000
 elif cid==s and not _tb_has(obs,s):v+=1700
 if cid==TB['LATIAS'] and not _tb_has(obs,TB['LATIAS']):v+=3500
 if cid==TB['TEAL'] and _tb_count(obs,TB['TEAL'])<2:v+=2600
 # Once the required role is present, do not search redundant attackers.
 if cid in {TB['TERAPAGOS'],TB['KANGA'],TB['CORNER'],TB['WELLSPRING'],TB['CHIYU'],TB['PECH']} and _tb_has(obs,cid):v-=2500
 return v

def _v4_manual_attach(obs,o):
 v=_V6_OLD_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o)
 if e is None or t is None:return v
 eid=int(e.id);p=_v6_primary(obs);sec=_v6_secondary(obs)
 before=_v6_progress(obs,t);after=_v6_progress(obs,t,eid)
 if t.id==p:v+=(before-after)*1800+1600
 elif t.id==sec:v+=(before-after)*700+250
 # Typed first attachment on the primary is critical.
 typed,_=_v4_energy_need(t)
 if t.id==p and typed and eid==typed and typed not in _tb_energy_types(t):v+=2400
 # Do not scatter a unique typed energy onto an unrelated colorless body.
 if eid in {TB['R'],TB['W'],TB['F'],TB['D']} and t.id not in {p,sec}:v-=3500
 return v

def _v4_transfer_plan(obs,active_only=False):
 p=_v6_primary(obs);target=_v6_find(obs,p)
 if target:
  ta,ti,tp=target
  if active_only and ta!=int(AreaType.ACTIVE):return None
  mobile=ta==int(AreaType.ACTIVE) or _tb_has(obs,TB['LATIAS']) or _tb_hand_count(obs,TB['SWITCH']) or _tb_hand_count(obs,TB['PRIME'])
  if mobile:
   best=None
   for sa,si,sp in _v4_locs(obs,True):
    if sp is tp:continue
    es=list(_tb_energy_types(sp))
    for ei,e in enumerate(es):
     before_prog=_v6_progress(obs,tp);after_prog=_v6_progress(obs,tp,e)
     if after_prog>=before_prog:continue
     src_after=es[:ei]+es[ei+1:]
     loss=max(0,_v4_attack_value(obs,sp)-_v4_attack_value(obs,sp,src_after))
     typed,_=_v4_energy_need(sp)
     if typed and e==typed and es.count(e)==1:loss+=1000
     before=_v4_attack_value(obs,tp);after=_v4_attack_value(obs,tp,list(_tb_energy_types(tp))+[e])
     gain=(before_prog-after_prog)*900+(after-before)*4+(1300 if before<60<=after else 0)
     val=gain-loss
     z=(val,sa,si,ei,e,ta,ti,tp.id,before,after)
     if best is None or z[0]>best[0]:best=z
   if best and best[0]>=500:return best
 return _V6_OLD_TRANSFER(obs,active_only)

def _v6_primary_ready(obs):
 x=_v6_find(obs,_v6_primary(obs));return x and _v4_attack_value(obs,x[2])>=60

def _v4_main(obs):
 sel=obs.select;opts=sel.option;pl=_v4_pl(obs,True);primary=_v6_primary(obs);match=_tb_matchup(obs)
 # Preserve Bug/engine/Latias/Area Zero ordering from v4.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==TB['BUG']:return [i]
 for wanted in (TB['KANGA'],TB['TEAL']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.ABILITY and c and c.id==wanted and (wanted==TB['KANGA'] or _tb_hand_count(obs,TB['G'])>0):return [i]
 if not _tb_has(obs,TB['LATIAS']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['LATIAS']:return [i]
 tera=any(p.id in {TB['TEAL'],TB['TERAPAGOS'],TB['CORNER'],TB['WELLSPRING']} for p in _tb_board(obs,True))
 if tera and _tb_stadium_id(obs)!=TB['AREA']:
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['AREA']:return [i]
 # Search specifically for the primary attacker, then Latias/Teal engine.
 if not _tb_has(obs,primary) or not _tb_has(obs,TB['LATIAS']) or _tb_count(obs,TB['TEAL'])<2:
  for preferred in (TB['TERA_ORB'],TB['ULTRA'],TB['CYRANO']):
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==preferred:return [i]
 # Deploy the primary immediately, irrelevant tech stays in hand.
 for cid in (primary,TB['TEAL'],_v6_secondary(obs)):
  if not _tb_has(obs,cid):
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==cid:return [i]
 # Typed/manual attachment must advance the current matchup plan.
 at=[(_v4_manual_attach(obs,o),i) for i,o in enumerate(opts) if o.type==OptionType.ATTACH]
 if at and max(at)[0]>=750:return [max(at)[1]]
 # Energy Switch / N's Plan completes the same primary attacker.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id in {TB['ESWITCH'],TB['NPLAN']}:
   plan=_v4_transfer_plan(obs,active_only=(c.id==TB['NPLAN']))
   if plan and plan[0]>=500:_V4['transfer']=plan;return [i]
 # If primary is ready on Bench, bring it Active before draw cycles.
 x=_v6_find(obs,primary);active=pl.active[0] if pl.active else None
 if x and x[0]==int(AreaType.BENCH) and _v4_attack_value(obs,x[2])>=60 and _v4_attack_value(obs,active)<_v4_attack_value(obs,x[2]):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id in {TB['SWITCH'],TB['PRIME']}:
    _V4['own_switch']=(x[0],x[1],x[2].id);return [i]
  for i,o in enumerate(opts):
   if o.type==OptionType.RETREAT:_V4['own_switch']=(x[0],x[1],x[2].id);return [i]
 # Attack immediately when the planned attacker is ready.
 if active and active.id==primary:
  ats=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
  if ats:return [max(ats,key=lambda i:_tb_dynamic_damage(obs,active,opts[i].attackId))]
 return _V6_OLD_MAIN(obs)

# === Tera Box v7 Crispin typed-energy planner ===
TB['CRISPIN']=1198
_V7_OLD_MAIN=_v4_main
_V7_OLD_CONTEXT=_v4_context

def _v7_needs_crispin(obs):
 x=_v6_find(obs,_v6_primary(obs))
 if not x:return False
 p=x[2];typed,total=_v4_energy_need(p);es=list(_tb_energy_types(p))
 return (typed and typed not in es) or len(es)<total

def _v4_main(obs):
 # Resolve universally strong engine actions first.
 sel=obs.select;opts=sel.option
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==TB['BUG']:return [i]
 for wanted in (TB['KANGA'],TB['TEAL']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.ABILITY and c and c.id==wanted and (wanted==TB['KANGA'] or _tb_hand_count(obs,TB['G'])>0):return [i]
 # Keep Latias/Area/search/deploy logic from the specialist layer.
 pre=_V7_OLD_MAIN(obs)
 if pre:
  try:
   o=opts[pre[0]];c=_source(obs,o);cid=int(getattr(c,'id',0) or 0)
   # Before generic Lillie/Cipher/attachment, Crispin opens the typed attacker.
   if (o.type==OptionType.PLAY and cid in {TB['LILLIE'],TB['CIPHER']}) or o.type==OptionType.ATTACH:
    if _v7_needs_crispin(obs) and not obs.current.supporterPlayed:
     for i,z in enumerate(opts):
      cc=_source(obs,z)
      if z.type==OptionType.PLAY and cc and cc.id==TB['CRISPIN']:return [i]
  except Exception:pass
 # If old layer did not produce a useful action, still use Crispin for the primary.
 if _v7_needs_crispin(obs) and not obs.current.supporterPlayed:
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==TB['CRISPIN']:return [i]
 return pre

def _v7_energy_rank(obs,c,selected=None):
 cid=int(getattr(c,'id',0) or 0);selected=selected or []
 x=_v6_find(obs,_v6_primary(obs));v=0
 if x:
  p=x[2];typed,total=_v4_energy_need(p);es=list(_tb_energy_types(p))
  if typed and cid==typed and typed not in es:v+=5000
  if cid==TB['G']:v+=1300 # future Teal Dance bank
  if cid not in selected:v+=500
  if cid in selected:v-=3000
 # Preserve different types as required by Crispin.
 return v

def _v4_context(obs):
 sel=obs.select;ctx=sel.context;opts=sel.option;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
 if eff==TB['CRISPIN']:
  if ctx==SelectContext.TO_HAND:
   n=min(int(sel.maxCount),len(opts));out=[];selids=[];rem=list(range(len(opts)))
   for _ in range(n):
    if not rem:break
    i=max(rem,key=lambda j:_v7_energy_rank(obs,_source(obs,opts[j]),selids));c=_source(obs,opts[i]);out.append(i);rem.remove(i);selids.append(int(getattr(c,'id',0) or 0))
   return out[:max(int(sel.minCount),len(out))]
  if ctx in {SelectContext.ATTACH_FROM,SelectContext.SWITCH_ENERGY_CARD}:
   # Attach the primary's missing typed Energy; put the other into hand.
   x=_v6_find(obs,_v6_primary(obs));typed=_v4_energy_need(x[2])[0] if x else None
   return [max(range(len(opts)),key=lambda i:(3000 if int(getattr(_source(obs,opts[i]),'id',0) or 0)==typed else 500 if int(getattr(_source(obs,opts[i]),'id',0) or 0)==TB['G'] else 0))]
  if ctx==SelectContext.ATTACH_TO:
   pokemon=[i for i,o in enumerate(opts) if hasattr(_source(obs,o),'hp')]
   x=_v6_find(obs,_v6_primary(obs))
   if x:
    for i in pokemon:
     c=_source(obs,opts[i])
     if c and c.id==x[2].id:return [i]
   if pokemon:return [max(pokemon,key=lambda i:_v4_attack_value(obs,_source(obs,opts[i])))]
 return _V7_OLD_CONTEXT(obs)

# Direct manual attachment: the primary's missing typed Energy always wins over
# partial colorless progress elsewhere.
_V7_PREV_ATTACH=_v4_manual_attach
def _v4_manual_attach(obs,o):
 v=_V7_PREV_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o)
 if e is None or t is None:return v
 p=_v6_primary(obs);typed,_=_v4_energy_need(t)
 if t.id==p and typed and int(e.id)==typed and typed not in _tb_energy_types(t):v+=7000
 if int(e.id) in {TB['R'],TB['W'],TB['F'],TB['D']} and t.id!=p:v-=5000
 return v

# === v7.1 corrected Crispin engine contexts ===
_V71_OLD_CONTEXT=_v4_context
_V71_OLD_ATTACH=_v4_manual_attach
_V71_OLD_TRANSFER=_v4_transfer_plan

def _v4_context(obs):
 sel=obs.select;ctx=sel.context;opts=sel.option;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
 if eff==TB['CRISPIN']:
  x=_v6_find(obs,_v6_primary(obs));primary=x[2] if x else None;typed=_v4_energy_need(primary)[0] if primary else None
  if ctx==SelectContext.TO_HAND:
   # Keep Grass for Teal Dance unless the primary's typed Energy cannot be attached
   # in the following step for some reason.
   return [max(range(len(opts)),key=lambda i:(2000 if int(getattr(_source(obs,opts[i]),'id',0) or 0)==TB['G'] else 400 if int(getattr(_source(obs,opts[i]),'id',0) or 0)!=typed else 0))]
  if ctx==SelectContext.ATTACH_TO:
   # Here options are the searched Energy cards; select the missing typed Energy.
   return [max(range(len(opts)),key=lambda i:(5000 if int(getattr(_source(obs,opts[i]),'id',0) or 0)==typed else 800 if int(getattr(_source(obs,opts[i]),'id',0) or 0)==TB['G'] else 0))]
  if ctx==SelectContext.ATTACH_FROM:
   # Here options are Pokémon receiving the selected Energy.
   if primary:
    for i,o in enumerate(opts):
     c=_source(obs,o)
     if c and c.id==primary.id:return [i]
   pokemon=[i for i,o in enumerate(opts) if hasattr(_source(obs,o),'hp')]
   if pokemon:return [max(pokemon,key=lambda i:_v4_attack_value(obs,_source(obs,opts[i])))]
 return _V71_OLD_CONTEXT(obs)

def _v4_manual_attach(obs,o):
 v=_V71_OLD_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o)
 if e is None or t is None:return v
 primary=_v6_primary(obs);typed,_=_v4_energy_need(t);eid=int(e.id);es=list(_tb_energy_types(t))
 if t.id==primary and typed and typed not in es:
  if eid==typed:v+=9000
  else:v-=7000
 return v

def _v4_transfer_plan(obs,active_only=False):
 x=_v6_find(obs,_v6_primary(obs))
 if x and _v4_attack_value(obs,x[2])>=60:
  # Never dismantle a completed matchup-specific attacker for generic energy spread.
  return None
 return _V71_OLD_TRANSFER(obs,active_only)

# === Exact uploaded "tera box" deck policy ==================================
# Source deck: decks(1).json / id 63698c2fd6e949d2.  This final layer replaces
# the experimental Terapagos/Crispin plan.  The exact deck uses 11 Grass +
# 4 Prism Energy and chooses one matchup tech to receive the scarce Prism cards.
_X_OLD_AGENT = agent
X={
 'TEAL':96,'WELL':108,'CLEF':272,'CORNER':117,'LATIAS':184,'MEOWTH':1071,
 'PECH':230,'MUNKI':112,'FEZ':140,'CHIYU':31,'KANGA':756,
 'BUG':1094,'AREA':1250,'TERA_ORB':1127,'STAMP':1080,'NIGHT':1097,
 'ULTRA':1121,'ESWITCH':1116,'NPLAN':1221,'CIPHER':1188,'BOSS':1182,
 'XERO':1197,'LILLIE':1227,'G':1,'PRISM':16,
}
_XMEM={'turn':-1,'teal_used':0,'kanga_used':False,'fez_used':False,'move':None,
      'played':set(),'supporter_fetch':False}

def _x_pl(obs,own=True):
 s=obs.current;return s.players[s.yourIndex if own else 1-s.yourIndex]
def _x_board(obs,own=True):
 p=_x_pl(obs,own);return [x for x in list(p.active or [])+list(p.bench or []) if x]
def _x_count(obs,cid,own=True):return sum(p.id==cid for p in _x_board(obs,own))
def _x_has(obs,cid,own=True):return _x_count(obs,cid,own)>0
def _x_hand(obs):return [x for x in (_x_pl(obs,True).hand or []) if x]
def _x_hcount(obs,cid):return sum(x.id==cid for x in _x_hand(obs))
def _x_stadium(obs):
 try:return int(obs.current.stadium[0].id) if obs.current.stadium else 0
 except Exception:return 0

def _x_matchup(obs):
 ids={p.id for p in _x_board(obs,False)}
 if ids & {344,345,58,756}:return 'crustle'
 if ids & {646,647,648,860,104,1259}:return 'marnie'
 if ids & {119,120,121,235}:return 'dragapult'
 if ids & {169,190,666,57}:return 'archaludon'
 if ids & {741,742,743,272}:return 'alakazam'
 if ids & {400,401,431,434}:return 'spidops'
 if ids & {96,1127,10,11,25}:return 'grass_ogerpon'
 opp=_x_board(obs,False)
 if len(opp)>=2 and not any(CARDS.get(p.id) and (CARDS[p.id].ex or CARDS[p.id].megaEx) for p in opp):return 'single_prize'
 return 'generic'

def _x_primary(obs):
 m=_x_matchup(obs)
 if m in {'archaludon','marnie','alakazam','spidops'}:return X['CORNER']
 if m=='dragapult':return X['CLEF']
 if m in {'crustle','single_prize'}:return X['CHIYU']
 return X['KANGA']
def _x_secondary(obs):
 m=_x_matchup(obs)
 if m in {'crustle','single_prize'}:return X['PECH']
 if m=='dragapult':return X['WELL']
 return X['TEAL']

def _x_energy_ids(p):return list(getattr(p,'energies',[]) or [])
def _x_can_pay(p,aid,energies=None):
 a=ATTACKS.get(int(aid or 0));pool=list(_x_energy_ids(p) if energies is None else energies)
 if not a:return False
 for req in a.energies:
  req=int(req)
  if req==0:
   if not pool:return False
   pool.pop(0);continue
  j=next((i for i,e in enumerate(pool) if int(e)==req or int(e)==X['PRISM']),None)
  if j is None:return False
  pool.pop(j)
 return True

def _x_damage(obs,p,aid):
 if p is None:return 0
 aid=int(aid or 0);opp=_x_pl(obs,False);oa=opp.active[0] if opp.active else None
 if aid==120:
  return 30+30*(len(_x_energy_ids(p))+(len(_x_energy_ids(oa)) if oa else 0))
 if aid==1092:return 250
 if aid==371:
  d=20+20*(len([x for x in _x_pl(obs,True).bench if x])+len([x for x in opp.bench if x]))
  if oa and oa.id in {119,120,121}:d*=2
  return d
 if aid==136:return 100
 if aid==148:return 140
 if aid==243:return 200
 if aid==183:return 100
 if aid==20:return 120 if _x_stadium(obs) else 60
 if aid==19:return 0
 if aid==315:return 70  # poison + retreat lock tactical value
 if aid==141:return 85  # damage + confusion tactical value
 a=ATTACKS.get(aid);return int(a.damage or 0) if a else 0

def _x_ready_damage(obs,p,energies=None):
 if p is None:return 0
 cd=CARDS.get(p.id);best=0
 if not cd:return 0
 es=_x_energy_ids(p) if energies is None else list(energies)
 for aid in cd.attacks:
  if _x_can_pay(p,aid,es):best=max(best,_x_damage(obs,p,aid))
 return best

def _x_energy_progress(p,energies=None):
 es=list(_x_energy_ids(p) if energies is None else energies);cid=p.id
 # (typed Prism count needed, total energy count).  Teal/Kanga/Fez use colorless/Grass pool.
 need={X['KANGA']:(0,3),X['TEAL']:(0,3),X['WELL']:(1,3),X['CORNER']:(1,3),
       X['CLEF']:(1,2),X['CHIYU']:(1,2),X['PECH']:(1,2),X['MUNKI']:(1,2),
       X['FEZ']:(0,3),X['LATIAS']:(2,3)}.get(cid,(0,99))
 prism=sum(int(e)==X['PRISM'] for e in es)
 return max(0,need[0]-prism)*5+max(0,need[1]-len(es))

def _x_role(obs,c,instance=False):
 if c is None:return -99999
 cid=int(getattr(c,'id',0) or 0);m=_x_matchup(obs);turn=int(obs.current.turn or 0)
 v=float(_TB_OLD_CARD_VALUE(obs,c))
 if cid==X['BUG']:v+=1500 if turn<=5 else 350
 elif cid==X['LATIAS']:v+=2600 if not _x_has(obs,cid) else -800
 elif cid==X['TEAL']:v+=1900 if _x_count(obs,cid)<2 else 500 if _x_count(obs,cid)<3 else -500
 elif cid==X['KANGA']:v+=1500 if not _x_has(obs,cid) else 250 if _x_count(obs,cid)<2 else -900
 elif cid==X['AREA']:v+=1200 if _x_stadium(obs)!=cid and any(p.id in {X['TEAL'],X['WELL'],X['CORNER']} for p in _x_board(obs,True)) else -700
 elif cid==X['G']:v+=650 if _x_hcount(obs,X['G'])<2 and _x_count(obs,X['TEAL'])>0 else 120
 elif cid==X['PRISM']:
  primary=_x_primary(obs);p=next((p for p in _x_board(obs,True) if p.id==primary),None)
  v+=1800 if p and _x_energy_progress(p)>0 else 650
 elif cid==X['CLEF']:v+=2600 if m=='dragapult' and not _x_has(obs,cid) else -500 if m!='dragapult' else 200
 elif cid==X['CORNER']:v+=2600 if m in {'archaludon','marnie','alakazam','spidops'} and not _x_has(obs,cid) else -600 if m not in {'archaludon','marnie','alakazam','spidops'} else 200
 elif cid==X['WELL']:v+=1800 if m=='dragapult' and not _x_has(obs,cid) else -450 if m!='dragapult' else 150
 elif cid==X['CHIYU']:v+=2600 if m in {'crustle','single_prize'} and not _x_has(obs,cid) else -650 if m not in {'crustle','single_prize'} else 150
 elif cid==X['PECH']:v+=2100 if m in {'crustle','single_prize'} and _x_count(obs,cid)<1 else -450 if m not in {'crustle','single_prize'} else 100
 elif cid==X['MUNKI']:
  damaged=any(p.hp<p.maxHp for p in _x_board(obs,True));v+=900 if damaged and _x_hcount(obs,X['PRISM']) else -250
 elif cid==X['FEZ']:
  v+=1000 if any(getattr(l,'type',None)==8 for l in (obs.logs or [])) else 100
 elif cid==X['MEOWTH']:
  v+=1300 if not _XMEM.get('supporter_fetch') and turn<=6 else -500
 elif cid==X['LILLIE']:v+=max(0,6-_x_pl(obs,True).handCount)*170
 elif cid==X['CIPHER']:v+=650 if _x_remaining_draws(obs)>0 else -1000
 elif cid==X['ESWITCH']:v+=650 if _x_transfer_plan(obs) else -700
 elif cid==X['NPLAN']:v+=800 if _x_nplan_gain(obs)>0 else -900
 elif cid==X['BOSS']:v+=500 if _x_best_gust(obs)[0]>0 else -150
 elif cid==X['XERO']:v+=900 if _x_pl(obs,False).handCount>=6 else -500
 elif cid==X['STAMP']:v+=1000 if _x_stamp_live(obs) else -900
 elif cid==X['NIGHT']:v+=700 if _x_night_targets(obs) else -400
 if instance and hasattr(c,'energyCards'):
  v+=200*len(getattr(c,'energyCards',[]) or [])+0.25*float(getattr(c,'hp',0) or 0)
  if cid==X['LATIAS']:v+=900
  if cid==_x_primary(obs):v+=900
 return v

def _x_remaining_draws(obs):
 pl=_x_pl(obs,True);n=0;active=pl.active[0] if pl.active else None
 if active and active.id==X['KANGA'] and not _XMEM.get('kanga_used'):n+=2
 n+=min(_x_hcount(obs,X['G']),max(0,_x_count(obs,X['TEAL'])-_XMEM.get('teal_used',0)))
 # Fez is deterministic only after a KO flag makes its ability legal; availability in options is checked elsewhere.
 return n

def _x_stamp_live(obs):
 # The engine exposes the Stamp play option only when legal, so the option itself is the final legality check.
 return _x_pl(obs,False).handCount>2

def _x_night_targets(obs):
 return [c for c in (_x_pl(obs,True).discard or []) if c.id in {X['PRISM'],X['G'],_x_primary(obs),X['LATIAS'],X['TEAL']}]

def _x_hand_ev(obs,exclude=None):
 exclude=exclude or set();return sum(max(0,_x_role(obs,c)) for c in _x_hand(obs) if c.id not in exclude)
def _x_deck_ev(obs):
 counts=Counter(MY_DECK);pl=_x_pl(obs,True)
 for c in list(pl.hand or [])+list(pl.discard or [])+list(pl.active or [])+list(pl.bench or [])+list(pl.prize or []):
  if c is not None and counts[c.id]>0:counts[c.id]-=1
 vals=[]
 for cid,n in counts.items():
  if n>0:vals.extend([max(0,_x_role(obs,type('C',(),{'id':cid})()))]*n)
 return sum(vals)/max(1,len(vals))

def _x_lillie_gain(obs):
 pl=_x_pl(obs,True);draw=8 if len(pl.prize or [])==6 else 6
 keep=_x_hand_ev(obs,{X['LILLIE']});expected=draw*_x_deck_ev(obs)
 # Preserve executable setup and a ready attack.
 if _x_hcount(obs,X['BUG']):keep+=900
 if _x_hcount(obs,X['LATIAS']) and not _x_has(obs,X['LATIAS']):keep+=1200
 if _x_ready_board(obs)>0:keep+=900
 return expected-keep

def _x_ready_board(obs):
 return max([_x_ready_damage(obs,p) for p in _x_board(obs,True)] or [0])

def _x_best_gust(obs):
 dmg=_x_ready_board(obs);best=(0,None)
 for i,p in enumerate(_x_pl(obs,False).bench or []):
  if not p:continue
  cd=CARDS.get(p.id);pr=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
  v=(1800+500*pr if dmg>=p.hp and dmg>0 else 0)
  if p.id in {119,120,169,646,647,741,742,344,400}:v+=500
  if v>best[0]:best=(v,i)
 return best

def _x_nplan_gain(obs):
 pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None
 if not active:return 0
 before=_x_ready_damage(obs,active);all_e=[]
 for p in pl.bench or []:
  if p:all_e.extend(_x_energy_ids(p))
 best=before
 for i,e in enumerate(all_e):
  best=max(best,_x_ready_damage(obs,active,_x_energy_ids(active)+[e]))
  for j in range(i+1,len(all_e)):
   best=max(best,_x_ready_damage(obs,active,_x_energy_ids(active)+[e,all_e[j]]))
 return best-before

def _x_transfer_plan(obs):
 pl=_x_pl(obs,True);board=[]
 for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
  for i,p in enumerate(xs or []):
   if p:board.append((int(area),i,p))
 best=None;primary=_x_primary(obs)
 for sa,si,sp in board:
  es=_x_energy_ids(sp)
  for ei,e in enumerate(es):
   src_after=es[:ei]+es[ei+1:];loss=max(0,_x_ready_damage(obs,sp)-_x_ready_damage(obs,sp,src_after))
   if int(e)==X['PRISM'] and _x_energy_progress(sp,es)<_x_energy_progress(sp,src_after):loss+=1000
   for ta,ti,tp in board:
    if sp is tp:continue
    before=_x_ready_damage(obs,tp);after_es=_x_energy_ids(tp)+[e];after=_x_ready_damage(obs,tp,after_es)
    prog=(_x_energy_progress(tp)-_x_energy_progress(tp,after_es))*450
    gain=(after-before)*6+prog+(1500 if before<60<=after else 0)+(900 if tp.id==primary else 0)
    val=gain-loss
    z=(val,sa,si,ei,e,ta,ti,tp.id)
    if best is None or z[0]>best[0]:best=z
 return best if best and best[0]>=500 else None

def _x_attach_score(obs,o):
 e=_source(obs,o);t=_target(obs,o)
 if not e or not t:return -99999
 eid=e.id;before=_x_ready_damage(obs,t);after_es=_x_energy_ids(t)+[eid];after=_x_ready_damage(obs,t,after_es)
 v=(after-before)*7+(_x_energy_progress(t)-_x_energy_progress(t,after_es))*650
 if before<60<=after:v+=1700
 if t.id==_x_primary(obs):v+=900
 if t.id==X['TEAL'] and eid==X['G']:v+=650-100*len(_x_energy_ids(t))
 if eid==X['PRISM'] and t.id not in {_x_primary(obs),_x_secondary(obs),X['MUNKI'],X['LATIAS']}:v-=1400
 return v

def _x_search_score(obs,c,selected=None):
 selected=selected or []
 if c is None:return -999999
 cid=c.id;v=_x_role(obs,c);m=_x_matchup(obs)
 if cid in selected:v-=1800
 roles={X['LATIAS']:'mob',X['TEAL']:'engine',X['KANGA']:'draw',X['CLEF']:'tech',X['CORNER']:'tech',X['WELL']:'tech',X['CHIYU']:'tech',X['PECH']:'tech',X['MEOWTH']:'support'}
 if roles.get(cid) in {roles.get(x) for x in selected}:v-=800
 if cid==_x_primary(obs) and not _x_has(obs,cid):v+=3200
 if cid==X['LATIAS'] and not _x_has(obs,cid):v+=3000
 if cid==X['TEAL'] and _x_count(obs,cid)<2:v+=2200
 if cid==X['KANGA'] and not _x_has(obs,cid):v+=1700
 if cid==X['PRISM'] and any(p.id==_x_primary(obs) and _x_energy_progress(p)>0 for p in _x_board(obs,True)):v+=1800
 return v

def _x_supporter_score(obs,cid):
 if cid==X['LILLIE']:return _x_lillie_gain(obs)
 if cid==X['BOSS']:return _x_best_gust(obs)[0]
 if cid==X['CIPHER']:return 1300 if _x_remaining_draws(obs)>=1 else -1200
 if cid==X['NPLAN']:return 7*_x_nplan_gain(obs)
 if cid==X['XERO']:return 550*max(0,_x_pl(obs,False).handCount-3)-300
 if cid==X['STAMP']:return 600*max(0,_x_pl(obs,False).handCount-2)
 return 0

def _x_setup_score(obs,c,active=False):
 if c.id==X['KANGA']:return 3000 if active else 1800
 if c.id==X['TEAL']:return 2600 if active else 2400
 if c.id==X['LATIAS']:return 1700 if active else 3200
 if c.id==X['PECH'] and _x_matchup(obs) in {'crustle','single_prize'}:return 2100 if active else 1200
 if c.id==_x_primary(obs):return 2000
 if c.id==X['MEOWTH']:return 500
 return _x_role(obs,c)

def _x_collapse_score(obs,p):
 # Higher score means discard first.
 unique={X['LATIAS'],_x_primary(obs)}
 v=0
 if p.id in unique:v-=3500
 if p.id==X['TEAL']:v-=1200 if _x_count(obs,X['TEAL'])<=2 else 100
 if p.id==X['KANGA']:v-=900 if _x_count(obs,X['KANGA'])<=1 else 200
 if p.id==X['MEOWTH']:v+=1300
 if p.id in {X['MUNKI'],X['FEZ'],X['PECH']} and p.id!=_x_primary(obs):v+=600
 v-=500*len(_x_energy_ids(p));v+=max(0,p.maxHp-p.hp)*2
 return v

def _x_context(obs):
 sel=obs.select;ctx=sel.context;opts=sel.option;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
 mn=max(0,int(sel.minCount));mx=min(len(opts),int(sel.maxCount));n=mn if ctx not in {SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH} else mx
 if n<=0:return []
 # Resolve Energy Switch plan.
 plan=_XMEM.get('move')
 if plan:
  _,sa,si,ei,e,ta,ti,tid=plan
  if ctx==SelectContext.SWITCH_ENERGY_CARD:
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if c and int(getattr(c,'id',-1))==int(e):return [i]
  if ctx in {SelectContext.ATTACH_FROM,SelectContext.DETACH_FROM}:
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if c and c.id==next((p.id for a,j,p in [(a,j,p) for a,j,p in []]),-999):pass
   # Area/index is more reliable than object identity.
   for i,o in enumerate(opts):
    if int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-1))==si:return [i]
  if ctx==SelectContext.ATTACH_TO:
   for i,o in enumerate(opts):
    if int(getattr(o,'area',-1))==ta and int(getattr(o,'index',-1))==ti:
     _XMEM['move']=None;return [i]
 # Setup.
 if ctx in {SelectContext.SETUP_ACTIVE_POKEMON,SelectContext.SETUP_BENCH_POKEMON}:
  vals=[(_x_setup_score(obs,_source(obs,o),ctx==SelectContext.SETUP_ACTIVE_POKEMON),-i,i) for i,o in enumerate(opts)]
  vals.sort(reverse=True);return [x[2] for x in vals[:n]]
 # Area Zero collapse / forced discard of in-play Pokémon.
 if ctx in {SelectContext.DISCARD,SelectContext.NOT_MOVE,SelectContext.TO_DECK_BOTTOM}:
  vals=[]
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if c and hasattr(c,'hp') and int(getattr(o,'area',-1)) in {int(AreaType.ACTIVE),int(AreaType.BENCH)}:v=_x_collapse_score(obs,c)
   else:v=-_x_role(obs,c)
   vals.append((v,-i,i))
  vals.sort(reverse=True);return [x[2] for x in vals[:n]]
 # Multi-card searches use role diversity.
 if ctx in {SelectContext.TO_HAND,SelectContext.TO_FIELD,SelectContext.TO_BENCH,SelectContext.TO_DECK}:
  out=[];ids=[];rem=list(range(len(opts)))
  for _ in range(min(n,len(rem))):
   i=max(rem,key=lambda j:_x_search_score(obs,_source(obs,opts[j]),ids));out.append(i);cc=_source(obs,opts[i]);ids.append(int(getattr(cc,'id',0) or 0));rem.remove(i)
  return out
 # Switch/active selection: choose attack value, with Kanga draw as early fallback.
 if ctx in {SelectContext.SWITCH,SelectContext.TO_ACTIVE}:
  vals=[]
  for i,o in enumerate(opts):
   c=_source(obs,o);v=_x_ready_damage(obs,c)*8+_x_role(obs,c,True)
   if c and c.id==X['KANGA'] and not _XMEM.get('kanga_used'):v+=900
   vals.append((v,-i,i))
  vals.sort(reverse=True);return [x[2] for x in vals[:n]]
 # Energy source/target choices outside an explicit transfer.
 if ctx in {SelectContext.ATTACH_FROM,SelectContext.DETACH_FROM}:
  vals=[]
  for i,o in enumerate(opts):
   c=_source(obs,o);v=-400*_x_ready_damage(obs,c)-700*len(_x_energy_ids(c)) if c and hasattr(c,'hp') else -_x_role(obs,c)
   vals.append((v,-i,i))
  vals.sort(reverse=True);return [x[2] for x in vals[:n]]
 if ctx==SelectContext.ATTACH_TO:
  vals=[]
  for i,o in enumerate(opts):
   c=_source(obs,o);v=_x_role(obs,c,True)+1200*(c.id==_x_primary(obs)) if c and hasattr(c,'hp') else _x_role(obs,c)
   vals.append((v,-i,i))
  vals.sort(reverse=True);return [x[2] for x in vals[:n]]
 # Damage/heal targets and every other context use exact role value plus original fallback.
 vals=[]
 for i,o in enumerate(opts):
  c=_source(obs,o)
  try:v=_TB_OLD_CONTEXT_SCORE(obs,o)+0.8*_x_role(obs,c,hasattr(c,'hp'))
  except Exception:v=_x_role(obs,c,hasattr(c,'hp'))
  vals.append((v,-i,i))
 vals.sort(reverse=True);return [x[2] for x in vals[:n]]

def _x_main(obs):
 sel=obs.select;opts=sel.option;pl=_x_pl(obs,True);m=_x_matchup(obs)
 # Hard priority only for universally productive, non-exclusive engine actions.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==X['BUG']:return [i]
 for wanted in (X['TEAL'],X['KANGA'],X['FEZ']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.ABILITY and c and c.id==wanted:
    if wanted==X['TEAL'] and _x_hcount(obs,X['G'])<=0:continue
    if wanted==X['KANGA'] and _XMEM.get('kanga_used'):continue
    if wanted==X['FEZ'] and _XMEM.get('fez_used'):continue
    return [i]
 # Latias and Area Zero infrastructure.
 if not _x_has(obs,X['LATIAS']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['LATIAS']:return [i]
 if _x_stadium(obs)!=X['AREA'] and any(p.id in {X['TEAL'],X['WELL'],X['CORNER']} for p in _x_board(obs,True)):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['AREA']:return [i]
 # Score all legal actions. Existing generic score remains a weak safety prior.
 vals=[];active=pl.active[0] if pl.active else None;ready_now=_x_ready_damage(obs,active)
 for i,o in enumerate(opts):
  c=_source(obs,o);cid=int(getattr(c,'id',0) or 0);v=0.0
  try:v=0.08*_TB_OLD_MAIN_SCORE(obs,o)
  except Exception:pass
  if o.type==OptionType.PLAY:
   if c and CARDS.get(cid) and CARDS[cid].cardType==CardType.POKEMON:
    cap={X['TEAL']:3,X['KANGA']:2,X['LATIAS']:1,X['MEOWTH']:1,X['PECH']:1,X['MUNKI']:1,X['FEZ']:1,X['CHIYU']:1,X['CLEF']:1,X['CORNER']:1,X['WELL']:1}.get(cid,1)
    v+=_x_role(obs,c)
    if _x_count(obs,cid)>=cap:v-=6000
    if cid==_x_primary(obs) and not _x_has(obs,cid):v+=2800
   elif cid in {X['ULTRA'],X['TERA_ORB']}:
    missing=not _x_has(obs,_x_primary(obs)) or not _x_has(obs,X['LATIAS']) or _x_count(obs,X['TEAL'])<2
    v+=1800 if missing else -1800
   elif cid==X['MEOWTH']:
    v+=_x_role(obs,c)
   elif cid==X['ESWITCH']:
    plan=_x_transfer_plan(obs);v+=plan[0] if plan else -5000
    if plan:_XMEM['candidate_move']=plan
   elif cid==X['NPLAN']:v+=7*_x_nplan_gain(obs)
   elif cid in {X['LILLIE'],X['BOSS'],X['CIPHER'],X['XERO'],X['STAMP']}:v+=_x_supporter_score(obs,cid)
   elif cid==X['NIGHT']:v+=1000 if _x_night_targets(obs) else -1500
   elif cid==X['AREA']:v-=2000 # already handled when productive
   else:v+=_x_role(obs,c)
  elif o.type==OptionType.ATTACH:v+=_x_attach_score(obs,o)
  elif o.type==OptionType.RETREAT:
   bench=max([_x_ready_damage(obs,p) for p in pl.bench if p] or [0]);v+=1800+8*(bench-ready_now) if bench>ready_now else -3000
  elif o.type==OptionType.ATTACK:
   dmg=_x_damage(obs,active,o.attackId);oa=_x_pl(obs,False).active[0] if _x_pl(obs,False).active else None
   v+=dmg*9+(2600 if oa and dmg>=oa.hp else 0)
   if active and active.id==_x_primary(obs):v+=500
  elif o.type==OptionType.END:v=-1000 if ready_now>0 else 0
  vals.append((v,-i,i))
 vals.sort(reverse=True)
 if not vals:return []
 out=[vals[0][2]]
 try:
  o=opts[out[0]];c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==X['ESWITCH']:_XMEM['move']=_XMEM.pop('candidate_move',None)
 except Exception:pass
 return out

def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  if observation.get('current') is None:_XMEM.update({'turn':-1,'teal_used':0,'kanga_used':False,'fez_used':False,'move':None,'played':set(),'supporter_fetch':False})
  return list(MY_DECK)
 try:obs=to_observation_class(observation)
 except Exception:return _X_OLD_AGENT(observation)
 if obs.select is None or not obs.select.option:return []
 turn=int(obs.current.turn or 0) if obs.current else -1
 if turn!=_XMEM.get('turn'):_XMEM.update({'turn':turn,'teal_used':0,'kanga_used':False,'fez_used':False,'move':None,'played':set(),'supporter_fetch':False})
 out=_x_main(obs) if obs.select.context==SelectContext.MAIN else _x_context(obs)
 try:
  if out and len(out)==1:
   o=obs.select.option[out[0]];c=_source(obs,o)
   if o.type==OptionType.ABILITY and c:
    if c.id==X['TEAL']:_XMEM['teal_used']+=1
    elif c.id==X['KANGA']:_XMEM['kanga_used']=True
    elif c.id==X['FEZ']:_XMEM['fez_used']=True
   if o.type==OptionType.PLAY and c and c.id==X['MEOWTH']:_XMEM['supporter_fetch']=True
 except Exception:pass
 return out

# === Exact Tera Box v2: engine-first, matchup-tech-second =====================
# Human replays consistently establish Kangaskhan + two Teal Ogerpon + Latias
# before investing scarce Prism Energy into the matchup tech.  v1 did the
# reverse and often spent 4-8 turns without an attack.
_X1_ROLE=_x_role
_X1_SEARCH=_x_search_score
_X1_ATTACH=_x_attach_score
_X1_TRANSFER=_x_transfer_plan
_X1_MAIN=_x_main
_X1_CONTEXT=_x_context

def _x_tech_primary(obs):
 m=_x_matchup(obs)
 if m in {'archaludon','marnie','alakazam','spidops'}:return X['CORNER']
 if m=='dragapult':return X['CLEF']
 if m in {'crustle','single_prize'}:return X['CHIYU']
 return X['KANGA']

def _x_primary(obs):
 tech=_x_tech_primary(obs);m=_x_matchup(obs);turn=int(obs.current.turn or 0)
 if m in {'crustle','single_prize'}:return tech
 k=next((p for p in _x_board(obs,True) if p.id==X['KANGA']),None)
 # Use Kangaskhan as the universal draw/first-prize attacker until it is ready or
 # the game has moved beyond the setup window.  Prism is never required here.
 if turn<=4 and (k is None or _x_ready_damage(obs,k)<60):return X['KANGA']
 return tech

def _x_role(obs,c,instance=False):
 v=_X1_ROLE(obs,c,instance);cid=int(getattr(c,'id',0) or 0);turn=int(obs.current.turn or 0);m=_x_matchup(obs)
 # Infrastructure dominates early search portfolios.
 if cid==X['KANGA'] and not _x_has(obs,cid):v+=2600
 if cid==X['LATIAS'] and not _x_has(obs,cid):v+=2300
 if cid==X['TEAL'] and _x_count(obs,cid)<2:v+=1900
 if cid==X['PRISM']:
  # Prism is a typed-tech resource, not generic colorless energy.
  tech=_x_tech_primary(obs);tp=next((p for p in _x_board(obs,True) if p.id==tech),None)
  v+=1600 if tp and _x_energy_progress(tp)>0 else 250
 if cid==X['FEZ'] and turn<=3:v-=1000
 if cid==X['MUNKI'] and not any(p.hp<p.maxHp for p in _x_board(obs,True)):v-=900
 # Do not expose unrelated tech Pokémon merely because Area Zero has room.
 relevant={_x_tech_primary(obs),_x_secondary(obs),X['KANGA'],X['LATIAS'],X['TEAL'],X['MEOWTH']}
 if cid in {X['CLEF'],X['CORNER'],X['WELL'],X['CHIYU'],X['PECH'],X['MUNKI'],X['FEZ']} and cid not in relevant:v-=1800
 return v

def _x_search_score(obs,c,selected=None):
 if c is None:return -999999
 v=_X1_SEARCH(obs,c,selected);cid=c.id;turn=int(obs.current.turn or 0)
 # Search hierarchy from the supplied winning replays.
 if cid==X['KANGA'] and not _x_has(obs,cid):v+=7000
 if cid==X['LATIAS'] and not _x_has(obs,cid):v+=6500
 if cid==X['TEAL'] and _x_count(obs,cid)<2:v+=5000
 tech=_x_tech_primary(obs)
 engine_ok=_x_has(obs,X['KANGA']) and _x_has(obs,X['LATIAS']) and _x_count(obs,X['TEAL'])>=1
 if cid==tech and not _x_has(obs,tech):v+=3500 if engine_ok or _x_matchup(obs) in {'crustle','single_prize'} else 500
 if cid==X['PRISM'] and _x_has(obs,tech):v+=2200
 if cid in {X['FEZ'],X['MUNKI']} and turn<=4:v-=2200
 return v

def _x_transfer_plan(obs):
 pl=_x_pl(obs,True);board=[]
 for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
  for i,p in enumerate(xs or []):
   if p:board.append((int(area),i,p))
 primary=_x_primary(obs);tech=_x_tech_primary(obs);best=None
 for sa,si,sp in board:
  es=_x_energy_ids(sp)
  for ei,e in enumerate(es):
   src_after=es[:ei]+es[ei+1:]
   loss=max(0,_x_ready_damage(obs,sp)-_x_ready_damage(obs,sp,src_after))
   # Keep the only Prism on a typed attacker/Adrena-Brain user.
   if int(e)==X['PRISM'] and sp.id in {tech,X['MUNKI'],X['LATIAS']} and es.count(e)<=1:loss+=2200
   for ta,ti,tp in board:
    if sp is tp:continue
    before=_x_ready_damage(obs,tp);after_es=_x_energy_ids(tp)+[e];after=_x_ready_damage(obs,tp,after_es)
    prog=(_x_energy_progress(tp)-_x_energy_progress(tp,after_es))*650
    gain=(after-before)*7+prog+(2000 if before<60<=after else 0)
    if tp.id==primary:gain+=1300
    # Grass is the preferred Kangaskhan fuel; save Prism for the later tech.
    if tp.id==X['KANGA']:
     gain+=900 if int(e)==X['G'] else -900
    if tp.id==tech and int(e)==X['PRISM']:gain+=1200
    val=gain-loss;z=(val,sa,si,ei,e,ta,ti,tp.id)
    if best is None or z[0]>best[0]:best=z
 return best if best and best[0]>=650 else None

def _x_attach_score(obs,o):
 v=_X1_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o)
 if not e or not t:return v
 eid=int(e.id);tech=_x_tech_primary(obs);primary=_x_primary(obs)
 if eid==X['PRISM']:
  if t.id==tech:v+=5000
  elif t.id in {X['MUNKI'],X['LATIAS']} and _x_matchup(obs) not in {'archaludon','marnie','dragapult','crustle'}:v+=300
  else:v-=5500
 if eid==X['G']:
  if t.id==X['KANGA'] and primary==X['KANGA']:v+=1800
  if t.id==X['TEAL']:v+=900
 return v

def _x_support_search_score(obs,c):
 if c is None:return -999999
 cid=c.id;pl=_x_pl(obs,True);v=_x_supporter_score(obs,cid)
 if cid==X['LILLIE']:
  v+=2600 if pl.handCount<=4 else 500
 if cid==X['BOSS'] and _x_best_gust(obs)[0]>0:v+=3000
 if cid==X['NPLAN'] and _x_nplan_gain(obs)>0:v+=2600
 if cid==X['CIPHER'] and _x_remaining_draws(obs)>=1:v+=1700
 if cid==X['XERO'] and _x_pl(obs,False).handCount>=6:v+=1400
 return v

def _x_context(obs):
 sel=obs.select;ctx=sel.context;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0);opts=sel.option
 # Meowth's Last-Ditch Catch: choose a supporter by immediate realized value.
 if eff==X['MEOWTH'] and ctx==SelectContext.TO_HAND:
  return [max(range(len(opts)),key=lambda i:_x_support_search_score(obs,_source(obs,opts[i])))]
 # Ciphermaniac top-deck plan: role completion first, and only cards that the
 # remaining deterministic draws can actually reach this turn.
 if eff==X['CIPHER'] and ctx==SelectContext.TO_DECK:
  n=min(len(opts),int(sel.maxCount));out=[];ids=[];rem=list(range(len(opts)))
  for _ in range(n):
   i=max(rem,key=lambda j:_x_search_score(obs,_source(obs,opts[j]),ids)+(1800 if _source(obs,opts[j]) and _source(obs,opts[j]).id in {_x_primary(obs),X['PRISM'],X['ESWITCH'],X['G']} else 0))
   out.append(i);c=_source(obs,opts[i]);ids.append(int(getattr(c,'id',0) or 0));rem.remove(i)
  return out
 return _X1_CONTEXT(obs)

def _x_main(obs):
 sel=obs.select;opts=sel.option;pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None
 # Always resolve the requested top-seven engine before any shuffle/topdeck plan.
 if X['CIPHER'] not in _XMEM.get('played',set()):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['BUG']:
    _XMEM['played'].add(X['BUG']);return [i]
 # Free draw/acceleration abilities.
 for wanted in (X['KANGA'],X['TEAL'],X['FEZ']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.ABILITY and c and c.id==wanted:
    if wanted==X['TEAL'] and _x_hcount(obs,X['G'])<=0:continue
    if wanted==X['KANGA'] and _XMEM.get('kanga_used'):continue
    if wanted==X['FEZ'] and _XMEM.get('fez_used'):continue
    return [i]
 # Infrastructure in the same order seen in the winning replays.
 if not _x_has(obs,X['LATIAS']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['LATIAS']:return [i]
 # Search missing engine pieces before the matchup tech.
 missing_k=not _x_has(obs,X['KANGA']);missing_l=not _x_has(obs,X['LATIAS']);missing_t=_x_count(obs,X['TEAL'])<1
 if missing_k or missing_l or missing_t:
  # Ultra Ball can find every missing engine piece; Tera Orb is only useful for Teal.
  preferred=[X['ULTRA']]+([X['TERA_ORB']] if missing_t else [])
  for cid in preferred:
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==cid:return [i]
 # Put Kangaskhan and Teal in play before filling Area Zero with tech bodies.
 for cid,cap in ((X['KANGA'],1),(X['TEAL'],2)):
  if _x_count(obs,cid)<cap:
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==cid:return [i]
 if _x_stadium(obs)!=X['AREA'] and any(p.id in {X['TEAL'],X['WELL'],X['CORNER']} for p in _x_board(obs,True)):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['AREA']:
    _XMEM['played'].add(X['AREA']);return [i]
 # Bring Kangaskhan Active for Run Errand and the first attack if no attacker is ready.
 k=next(((i,p) for i,p in enumerate(pl.bench or []) if p and p.id==X['KANGA']),None)
 if k and (active is None or _x_ready_damage(obs,active)==0) and not _XMEM.get('kanga_used') and _x_has(obs,X['LATIAS']):
  for i,o in enumerate(opts):
   if o.type==OptionType.RETREAT:return [i]
 # Open an attack with N's Plan / Energy Switch / manual attachment.
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and c.id==X['NPLAN'] and _x_nplan_gain(obs)>0:return [i]
 plan=_x_transfer_plan(obs)
 if plan:
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['ESWITCH']:
    _XMEM['move']=plan;return [i]
 attaches=[(_x_attach_score(obs,o),i) for i,o in enumerate(opts) if o.type==OptionType.ATTACH]
 if attaches and max(attaches)[0]>=600:return [max(attaches)[1]]
 # A ready attack is worth more than redundant search, cycling, or bench filling.
 ready=_x_ready_damage(obs,active)
 if ready>0:
  gust=_x_best_gust(obs)
  if gust[0]>=2300:
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==X['BOSS']:return [i]
  attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
  if attacks:return [max(attacks,key=lambda i:_x_damage(obs,active,opts[i].attackId))]
 # If a ready bench attacker exists, move it Active before further cycling.
 bench_ready=max([(_x_ready_damage(obs,p),i,p) for i,p in enumerate(pl.bench or []) if p] or [(0,-1,None)])
 if bench_ready[0]>ready and _x_has(obs,X['LATIAS']):
  for i,o in enumerate(opts):
   if o.type==OptionType.RETREAT:return [i]
 # Now build the matchup tech and only then consider hand cycling.
 tech=_x_tech_primary(obs)
 if not _x_has(obs,tech):
  for search in (X['TERA_ORB'] if tech in {X['TEAL'],X['WELL'],X['CORNER']} else X['ULTRA'],X['ULTRA']):
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==search:return [i]
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==tech:return [i]
 # Meowth is a supporter bridge only when a high-value supporter is missing.
 if not _XMEM.get('supporter_fetch') and pl.handCount<=4:
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['MEOWTH']:return [i]
 # Cipher before Lillie only when a deterministic draw is still available.
 if _x_remaining_draws(obs)>=1 and not _x_hcount(obs,X['BUG']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['CIPHER']:
    _XMEM['played'].add(X['CIPHER']);return [i]
 if _x_lillie_gain(obs)>500 and not _x_hcount(obs,X['BUG']):
  for i,o in enumerate(opts):
   c=_source(obs,o)
   if o.type==OptionType.PLAY and c and c.id==X['LILLIE']:return [i]
 # Xerosic/Stamp are disruption, not setup substitutes.
 for cid in (X['STAMP'],X['XERO']):
  if _x_supporter_score(obs,cid)>1200:
   for i,o in enumerate(opts):
    c=_source(obs,o)
    if o.type==OptionType.PLAY and c and c.id==cid:return [i]
 # Safe role-positive bench plays only.
 plays=[]
 for i,o in enumerate(opts):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and CARDS.get(c.id) and CARDS[c.id].cardType==CardType.POKEMON:
   v=_x_role(obs,c)
   if v>900:plays.append((v,i))
 if plays:return [max(plays)[1]]
 # Fall back to the exact v1 scorer for attacks/end and obscure legal contexts.
 return _X1_MAIN(obs)

# === Exact Tera Box v3: corrected Energy Switch / N's Plan sequencing =========
# The engine exposes Energy Switch as SOURCE_POKEMON -> TARGET_POKEMON, while
# N's Plan exposes individual bench energy cards and moves them to the Active.
# Earlier versions interpreted these contexts backwards.
_X2_PRIMARY = _x_primary
_X2_CONTEXT = _x_context
_X2_ATTACH = _x_attach_score
_X2_TRANSFER = _x_transfer_plan
_X2_MAIN = _x_main


def _x_primary(obs):
    m=_x_matchup(obs)
    # Kangaskhan is still the preferred Active draw engine, but these matchups
    # must charge the tech attacker from turn one rather than charging Kanga.
    if m in {'archaludon','marnie','alakazam','spidops'}:
        return X['CORNER']
    if m=='dragapult':
        # Supplied replay attacks with Kangaskhan first, Clefairy second.
        k=next((p for p in _x_board(obs,True) if p.id==X['KANGA']),None)
        if int(obs.current.turn or 0)<=4 and (k is None or _x_ready_damage(obs,k)<60):return X['KANGA']
        return X['CLEF']
    if m in {'crustle','single_prize'}:return X['CHIYU']
    return X['KANGA']


def _x_transfer_plan(obs):
    pl=_x_pl(obs,True);board=[]
    for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
        for i,p in enumerate(xs or []):
            if p:board.append((int(area),i,p))
    primary=_x_primary(obs);tech=_x_tech_primary(obs);best=None
    for sa,si,sp in board:
        es=_x_energy_ids(sp)
        for ei,e in enumerate(es):
            src_after=es[:ei]+es[ei+1:]
            loss=max(0,_x_ready_damage(obs,sp)-_x_ready_damage(obs,sp,src_after))*8
            # Do not strip a unique typed energy from a relevant tech attacker.
            if int(e)==X['PRISM'] and sp.id in {tech,X['MUNKI'],X['LATIAS']} and sum(int(x)==X['PRISM'] for x in es)<=1:
                loss+=5000
            for ta,ti,tp in board:
                if sp is tp:continue
                before=_x_ready_damage(obs,tp);after_es=_x_energy_ids(tp)+[e];after=_x_ready_damage(obs,tp,after_es)
                progress=(_x_energy_progress(tp)-_x_energy_progress(tp,after_es))*900
                gain=(after-before)*10+progress
                if before<60<=after:gain+=5000
                if tp.id==primary:gain+=2200
                if tp.id==tech and int(e)==X['PRISM']:gain+=1800
                # Kanga receives Grass only in the matchups where it is the first attacker.
                if tp.id==X['KANGA']:
                    gain += 900 if primary==X['KANGA'] and int(e)==X['G'] else -1800
                val=gain-loss
                z=(val,sa,si,ei,int(e),ta,ti,tp.id)
                if best is None or z[0]>best[0]:best=z
    # Energy Switch must materially improve the plan; mere deck compression is zero value.
    return best if best and best[0]>=1600 else None


def _x_nplan_option_score(obs,o):
    pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None
    src=_source(obs,o)
    if active is None or src is None or not hasattr(src,'hp'):return -999999
    ei=int(getattr(o,'energyIndex',-1))
    es=_x_energy_ids(src)
    if ei<0 or ei>=len(es):return -999999
    e=int(es[ei]);before=_x_ready_damage(obs,active);after=_x_ready_damage(obs,active,_x_energy_ids(active)+[e])
    donor_after=es[:ei]+es[ei+1:]
    loss=max(0,_x_ready_damage(obs,src)-_x_ready_damage(obs,src,donor_after))*8
    progress=(_x_energy_progress(active)-_x_energy_progress(active,_x_energy_ids(active)+[e]))*1000
    gain=(after-before)*10+progress+(5200 if before<60<=after else 0)
    if active.id==_x_primary(obs):gain+=1800
    if e==X['PRISM'] and active.id==_x_tech_primary(obs):gain+=1400
    if e==X['PRISM'] and src.id in {_x_tech_primary(obs),X['MUNKI'],X['LATIAS']} and sum(int(x)==X['PRISM'] for x in es)<=1:loss+=5000
    return gain-loss


def _x_context(obs):
    sel=obs.select;ctx=sel.context;opts=sel.option;eff=int(getattr(getattr(sel,'effect',None),'id',0) or 0)
    plan=_XMEM.get('move')
    if eff==X['ESWITCH'] and plan:
        _,sa,si,ei,e,ta,ti,tid=plan
        # First selection is the exact energy card on the source Pokémon.
        if ctx==SelectContext.SWITCH_ENERGY_CARD:
            for i,o in enumerate(opts):
                if int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-1))==si and int(getattr(o,'energyIndex',-1))==ei:
                    return [i]
            # Energy indices can shift after state serialization; retain source and prefer matching type.
            for i,o in enumerate(opts):
                if int(getattr(o,'area',-1))==sa and int(getattr(o,'index',-1))==si:
                    src=_source(obs,o);oe=int(getattr(o,'energyIndex',-1));es=_x_energy_ids(src) if src else []
                    if 0<=oe<len(es) and int(es[oe])==int(e):return [i]
        # Second selection is the destination Pokémon, despite the engine context name.
        if ctx==SelectContext.ATTACH_FROM:
            for i,o in enumerate(opts):
                if int(getattr(o,'area',-1))==ta and int(getattr(o,'index',-1))==ti:
                    _XMEM['move']=None;return [i]
    # N's Plan options represent individual bench energy cards. The Active target is implicit.
    if eff==X['NPLAN'] and ctx==SelectContext.SWITCH_ENERGY:
        vals=[(_x_nplan_option_score(obs,o),i) for i,o in enumerate(opts)]
        if not vals:return []
        v,i=max(vals)
        if int(sel.minCount)==0 and v<900:return []
        return [i]
    return _X2_CONTEXT(obs)


def _x_attach_score(obs,o):
    v=_X2_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o)
    if not e or not t:return v
    eid=int(e.id);m=_x_matchup(obs);primary=_x_primary(obs)
    # In tech matchups Grass follows the scarce Prism onto the tech attacker.
    if eid==X['G'] and t.id==primary and primary!=X['KANGA']:v+=3000
    if eid==X['G'] and t.id==X['KANGA'] and primary!=X['KANGA']:v-=2500
    if eid==X['PRISM'] and t.id!=_x_tech_primary(obs):v-=3000
    return v


def _x_main(obs):
    out=_X2_MAIN(obs)
    # Never launch Energy Switch without a fully recorded source/energy/target plan.
    try:
        if out and len(out)==1:
            o=obs.select.option[out[0]];c=_source(obs,o)
            if o.type==OptionType.PLAY and c and c.id==X['ESWITCH']:
                plan=_XMEM.get('move') or _x_transfer_plan(obs)
                if not plan:
                    # Re-score without Energy Switch by taking the best safe attack/attach/end fallback.
                    choices=[];pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None
                    for i,z in enumerate(obs.select.option):
                        cc=_source(obs,z)
                        if z.type==OptionType.ATTACK:choices.append((_x_damage(obs,active,z.attackId)*10+3000,i))
                        elif z.type==OptionType.ATTACH:choices.append((_x_attach_score(obs,z),i))
                        elif z.type==OptionType.END:choices.append((0,i))
                        elif not (z.type==OptionType.PLAY and cc and cc.id==X['ESWITCH']):choices.append((-1000,i))
                    return [max(choices)[1]] if choices else out
                _XMEM['move']=plan
    except Exception:pass
    return out

# Rebind the final public entry point to v3 functions.
_X3_INTERNAL_AGENT = agent
del agent
def agent(observation:dict)->list[int]:
    if observation.get('select') is None:
        if observation.get('current') is None:_XMEM.update({'turn':-1,'teal_used':0,'kanga_used':False,'fez_used':False,'move':None,'played':set(),'supporter_fetch':False})
        return list(MY_DECK)
    try:obs=to_observation_class(observation)
    except Exception:return _X_OLD_AGENT(observation)
    if obs.select is None or not obs.select.option:return []
    turn=int(obs.current.turn or 0) if obs.current else -1
    if turn!=_XMEM.get('turn'):_XMEM.update({'turn':turn,'teal_used':0,'kanga_used':False,'fez_used':False,'move':None,'played':set(),'supporter_fetch':False})
    out=_x_main(obs) if obs.select.context==SelectContext.MAIN else _x_context(obs)
    try:
        if out and len(out)==1:
            o=obs.select.option[out[0]];c=_source(obs,o)
            if o.type==OptionType.ABILITY and c:
                if c.id==X['TEAL']:_XMEM['teal_used']+=1
                elif c.id==X['KANGA']:_XMEM['kanga_used']=True
                elif c.id==X['FEZ']:_XMEM['fez_used']=True
            if o.type==OptionType.PLAY and c and c.id==X['MEOWTH']:_XMEM['supporter_fetch']=True
    except Exception:pass
    return out

# === Exact Tera Box v4: distinguish Prism card ID from attached energy type ===
# Card ID 16 becomes universal attached-energy type 10 in battle observations.
X['PRISM_CARD']=16
X['ANY_ENERGY']=10
_X3_TRANSFER=_x_transfer_plan
_X3_NPLAN_SCORE=_x_nplan_option_score


def _x_can_pay(p,aid,energies=None):
    a=ATTACKS.get(int(aid or 0));pool=list(_x_energy_ids(p) if energies is None else energies)
    if not a:return False
    for req in a.energies:
        req=int(req)
        if req==0:
            if not pool:return False
            pool.pop(0);continue
        j=next((i for i,e in enumerate(pool) if int(e)==req or int(e)==X['ANY_ENERGY']),None)
        if j is None:return False
        pool.pop(j)
    return True


def _x_energy_progress(p,energies=None):
    es=list(_x_energy_ids(p) if energies is None else energies);cid=p.id
    need={X['KANGA']:(0,3),X['TEAL']:(0,3),X['WELL']:(1,3),X['CORNER']:(1,3),
          X['CLEF']:(1,2),X['CHIYU']:(1,2),X['PECH']:(1,2),X['MUNKI']:(1,2),
          X['FEZ']:(0,3),X['LATIAS']:(2,3)}.get(cid,(0,99))
    any_n=sum(int(e)==X['ANY_ENERGY'] for e in es)
    return max(0,need[0]-any_n)*5+max(0,need[1]-len(es))


def _x_transfer_plan(obs):
    pl=_x_pl(obs,True);board=[]
    for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
        for i,p in enumerate(xs or []):
            if p:board.append((int(area),i,p))
    primary=_x_primary(obs);tech=_x_tech_primary(obs);best=None
    for sa,si,sp in board:
        es=_x_energy_ids(sp)
        for ei,e in enumerate(es):
            src_after=es[:ei]+es[ei+1:]
            loss=max(0,_x_ready_damage(obs,sp)-_x_ready_damage(obs,sp,src_after))*8
            if int(e)==X['ANY_ENERGY'] and sp.id in {tech,X['MUNKI'],X['LATIAS']} and sum(int(x)==X['ANY_ENERGY'] for x in es)<=1:
                loss+=5000
            for ta,ti,tp in board:
                if sp is tp:continue
                before=_x_ready_damage(obs,tp);after_es=_x_energy_ids(tp)+[e];after=_x_ready_damage(obs,tp,after_es)
                progress=(_x_energy_progress(tp)-_x_energy_progress(tp,after_es))*900
                gain=(after-before)*10+progress
                if before<60<=after:gain+=5000
                if tp.id==primary:gain+=2200
                if tp.id==tech and int(e)==X['ANY_ENERGY']:gain+=1800
                if tp.id==X['KANGA']:
                    gain += 900 if primary==X['KANGA'] and int(e)==X['G'] else -1800
                val=gain-loss;z=(val,sa,si,ei,int(e),ta,ti,tp.id)
                if best is None or z[0]>best[0]:best=z
    return best if best and best[0]>=1600 else None


def _x_nplan_option_score(obs,o):
    pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None;src=_source(obs,o)
    if active is None or src is None or not hasattr(src,'hp'):return -999999
    ei=int(getattr(o,'energyIndex',-1));es=_x_energy_ids(src)
    if ei<0 or ei>=len(es):return -999999
    e=int(es[ei]);before=_x_ready_damage(obs,active);after=_x_ready_damage(obs,active,_x_energy_ids(active)+[e])
    donor_after=es[:ei]+es[ei+1:]
    loss=max(0,_x_ready_damage(obs,src)-_x_ready_damage(obs,src,donor_after))*8
    progress=(_x_energy_progress(active)-_x_energy_progress(active,_x_energy_ids(active)+[e]))*1000
    gain=(after-before)*10+progress+(5200 if before<60<=after else 0)
    if active.id==_x_primary(obs):gain+=1800
    if e==X['ANY_ENERGY'] and active.id==_x_tech_primary(obs):gain+=1400
    if e==X['ANY_ENERGY'] and src.id in {_x_tech_primary(obs),X['MUNKI'],X['LATIAS']} and sum(int(x)==X['ANY_ENERGY'] for x in es)<=1:loss+=5000
    return gain-loss

# The final public agent from v3 dynamically resolves these global helpers.

# === Exact Tera Box v5: replay-faithful matchup engines =======================
_X4_PRIMARY=_x_primary
_X4_SECONDARY=_x_secondary
_X4_TECH_PRIMARY=_x_tech_primary
_X4_ROLE=_x_role
_X4_SEARCH=_x_search_score
_X4_BEST_GUST=_x_best_gust
_X4_MAIN=_x_main


def _x_tech_primary(obs):
    m=_x_matchup(obs)
    if m=='crustle':return X['PECH']
    if m=='dragapult':return X['CLEF']
    if m in {'archaludon','marnie','alakazam','spidops'}:return X['CORNER']
    return _X4_TECH_PRIMARY(obs)


def _x_primary(obs):
    m=_x_matchup(obs)
    if m=='crustle':return X['PECH']
    if m=='dragapult':return X['KANGA']
    if m in {'archaludon','marnie','alakazam','spidops'}:return X['CORNER']
    return _X4_PRIMARY(obs)


def _x_secondary(obs):
    m=_x_matchup(obs)
    if m=='crustle':return X['CHIYU']
    if m=='dragapult':return X['CLEF']
    if m=='archaludon':return X['WELL']
    return _X4_SECONDARY(obs)


def _x_role(obs,c,instance=False):
    v=_X4_ROLE(obs,c,instance);cid=int(getattr(c,'id',0) or 0);m=_x_matchup(obs)
    if m=='crustle':
        if cid==X['PECH']:v+=5200 if not _x_has(obs,cid) else 1200
        elif cid==X['CHIYU']:v+=3000 if not _x_has(obs,cid) else 500
        elif cid==X['PRISM_CARD']:
            p=next((p for p in _x_board(obs,True) if p.id==X['PECH']),None)
            v+=3500 if p and _x_energy_progress(p)>0 else 800
        elif cid==X['KANGA']:v+=900 if not _x_has(obs,cid) else -900
    elif m=='dragapult':
        if cid==X['KANGA']:v+=4000 if not _x_has(obs,cid) else 700
        elif cid==X['CLEF']:v+=2500 if not _x_has(obs,cid) else 300
        elif cid==X['WELL']:v+=1000 if not _x_has(obs,cid) else 0
    elif m=='archaludon':
        if cid==X['CORNER']:v+=4200 if not _x_has(obs,cid) else 900
        elif cid==X['WELL']:v+=1700 if not _x_has(obs,cid) else 250
    return v


def _x_search_score(obs,c,selected=None):
    v=_X4_SEARCH(obs,c,selected);cid=int(getattr(c,'id',0) or 0);m=_x_matchup(obs)
    if m=='crustle':
        if cid==X['PECH'] and not _x_has(obs,cid):v+=9000
        if cid==X['CHIYU'] and not _x_has(obs,cid):v+=5000
        if cid==X['PRISM_CARD'] and _x_has(obs,X['PECH']):v+=4500
    elif m=='dragapult':
        if cid==X['KANGA'] and not _x_has(obs,cid):v+=9000
        if cid==X['CLEF'] and not _x_has(obs,cid):v+=3500
    elif m=='archaludon':
        if cid==X['CORNER'] and not _x_has(obs,cid):v+=9000
        if cid==X['WELL'] and not _x_has(obs,cid):v+=2500
    return v


def _x_best_gust(obs):
    pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None
    dmg=_x_ready_damage(obs,active);m=_x_matchup(obs);best=(0,None)
    for i,p in enumerate(_x_pl(obs,False).bench or []):
        if not p:continue
        cd=CARDS.get(p.id);pr=3 if cd and cd.megaEx else 2 if cd and cd.ex else 1
        ko=dmg>0 and dmg>=p.hp
        v=(2500+700*pr if ko else 0)
        if m=='archaludon':
            # The supplied winning replay farms Duraludon/Cinderace/Relicanth
            # with 140-damage Demolish instead of three-hitting Archaludon ex.
            if p.id in {169,666,57} and ko:v+=7000
            elif p.id==190 and not ko:v-=2500
        elif m=='dragapult':
            if p.id in {119,120,235} and ko:v+=4500
        elif m=='marnie':
            if p.id in {646,647,860,104} and ko:v+=4200
        elif m=='crustle':
            if p.id in {344,343} and ko:v+=3200
        if v>best[0]:best=(v,i)
    return best


def _x_main(obs):
    pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None;m=_x_matchup(obs)
    # Once the replay-designated attacker is ready, do not leave it merely to use
    # Kangaskhan's draw ability or to perform another low-value cycle.
    if active is not None and _x_ready_damage(obs,active)>0:
        gust=_x_best_gust(obs)
        if gust[0]>=3500:
            for i,o in enumerate(obs.select.option):
                c=_source(obs,o)
                if o.type==OptionType.PLAY and c and c.id==X['BOSS']:return [i]
        attacks=[i for i,o in enumerate(obs.select.option) if o.type==OptionType.ATTACK]
        if attacks:return [max(attacks,key=lambda i:_x_damage(obs,active,obs.select.option[i].attackId))]
    return _X4_MAIN(obs)


# === Exact Tera Box v6: Crustle lock + Dragapult Kanga sequence ===============
_X5_DAMAGE=_x_damage
_X5_TRANSFER=_x_transfer_plan
_X5_ATTACH=_x_attach_score
_X5_MAIN=_x_main


def _x_damage(obs,p,aid):
    d=_X5_DAMAGE(obs,p,aid)
    try:
        oa=_x_pl(obs,False).active[0] if _x_pl(obs,False).active else None
        cd=CARDS.get(p.id) if p else None
        # Crustle prevents damage from Pokémon ex. Demolish explicitly ignores
        # effects on the opponent and remains live.
        if oa and oa.id==345 and cd and (cd.ex or cd.megaEx) and int(aid or 0)!=148:
            return 0
    except Exception:pass
    return d


def _x_transfer_plan(obs):
    m=_x_matchup(obs);pl=_x_pl(obs,True)
    if m=='dragapult':
        board=[]
        for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
            for i,p in enumerate(xs or []):
                if p:board.append((int(area),i,p))
        k=next(((a,i,p) for a,i,p in board if p.id==X['KANGA']),None)
        if k and len(_x_energy_ids(k[2]))<3:
            best=None
            for sa,si,sp in board:
                if sp is k[2]:continue
                for ei,e in enumerate(_x_energy_ids(sp)):
                    if int(e)!=X['G']:continue
                    # Keep a Teal attacker intact only after Kangaskhan is ready.
                    loss=max(0,_x_ready_damage(obs,sp)-_x_ready_damage(obs,sp,_x_energy_ids(sp)[:ei]+_x_energy_ids(sp)[ei+1:]))*4
                    val=7000+1800*(len(_x_energy_ids(k[2]))==2)-loss
                    z=(val,sa,si,ei,int(e),k[0],k[1],k[2].id)
                    if best is None or z[0]>best[0]:best=z
            if best:return best
    return _X5_TRANSFER(obs)


def _x_attach_score(obs,o):
    v=_X5_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o);m=_x_matchup(obs)
    if not e or not t:return v
    if m=='dragapult':
        if e.id==X['G'] and t.id==X['KANGA'] and len(_x_energy_ids(t))<3:v+=7500
        if e.id==X['G'] and t.id!=X['KANGA'] and any(p.id==X['KANGA'] and len(_x_energy_ids(p))<3 for p in _x_board(obs,True)):v-=5000
        if e.id==X['PRISM_CARD'] and t.id==X['CLEF']:v+=2500
    if m=='crustle':
        if e.id==X['PRISM_CARD'] and t.id in {X['PECH'],X['CHIYU'],X['CORNER']}:v+=6500
        if e.id==X['G'] and t.id in {X['PECH'],X['CHIYU'],X['CORNER']} and _x_energy_progress(t)>0:v+=4200
        if t.id in {X['TEAL'],X['KANGA']} and any(p.id in {X['PECH'],X['CHIYU'],X['CORNER']} and _x_energy_progress(p)>0 for p in _x_board(obs,True)):v-=4000
    return v


def _x_main(obs):
    pl=_x_pl(obs,True);active=pl.active[0] if pl.active else None;m=_x_matchup(obs);opts=obs.select.option
    if m=='dragapult':
        k=next((p for p in _x_board(obs,True) if p.id==X['KANGA']),None)
        if active and active.id==X['KANGA']:
            if _x_ready_damage(obs,active)>0:
                attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
                if attacks:return [max(attacks,key=lambda i:_x_damage(obs,active,opts[i].attackId))]
            # Continue charging Kangaskhan; never retreat it merely because a Teal
            # Ogerpon has become attack-ready.
            for wanted in (X['KANGA'],X['TEAL']):
                for i,o in enumerate(opts):
                    c=_source(obs,o)
                    if o.type==OptionType.ABILITY and c and c.id==wanted:
                        if wanted==X['TEAL'] and _x_hcount(obs,X['G'])<=0:continue
                        if wanted==X['KANGA'] and _XMEM.get('kanga_used'):continue
                        return [i]
            plan=_x_transfer_plan(obs)
            if plan:
                for i,o in enumerate(opts):
                    c=_source(obs,o)
                    if o.type==OptionType.PLAY and c and c.id==X['ESWITCH']:
                        _XMEM['move']=plan;return [i]
            attaches=[(_x_attach_score(obs,o),i) for i,o in enumerate(opts) if o.type==OptionType.ATTACH]
            if attaches and max(attaches)[0]>1000:return [max(attaches)[1]]
        if k and len(_x_energy_ids(k))>=3 and (not active or active.id!=X['KANGA']) and _x_has(obs,X['LATIAS']):
            for i,o in enumerate(opts):
                if o.type==OptionType.RETREAT:return [i]
    elif m=='crustle':
        oa=_x_pl(obs,False).active[0] if _x_pl(obs,False).active else None
        blocked=bool(oa and oa.id==345 and active and CARDS.get(active.id) and (CARDS[active.id].ex or CARDS[active.id].megaEx) and active.id!=X['CORNER'])
        ready_tech=max([(_x_ready_damage(obs,p),i,p) for i,p in enumerate(pl.bench or []) if p and p.id in {X['PECH'],X['CHIYU'],X['CORNER']}] or [(0,-1,None)])
        if blocked and ready_tech[0]>0 and _x_has(obs,X['LATIAS']):
            for i,o in enumerate(opts):
                if o.type==OptionType.RETREAT:return [i]
        # When locked and no bypass attacker exists, prioritize finding one or
        # cycling into one instead of repeatedly declaring a zero-damage attack.
        if blocked and not any(p.id in {X['PECH'],X['CHIYU'],X['CORNER']} for p in _x_board(obs,True)):
            for cid in (X['ULTRA'],X['LILLIE']):
                for i,o in enumerate(opts):
                    c=_source(obs,o)
                    if o.type==OptionType.PLAY and c and c.id==cid:return [i]
        if active and active.id in {X['PECH'],X['CHIYU'],X['CORNER']} and _x_ready_damage(obs,active)>0:
            attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
            if attacks:return [max(attacks,key=lambda i:_x_damage(obs,active,opts[i].attackId))]
    out=_X5_MAIN(obs)
    # Protect the Dragapult Kangaskhan charging line from a fallback retreat.
    if m=='dragapult' and active and active.id==X['KANGA'] and len(_x_energy_ids(active))<3 and out:
        try:
            if opts[out[0]].type==OptionType.RETREAT:
                ends=[i for i,o in enumerate(opts) if o.type==OptionType.END]
                return [ends[0]] if ends else out
        except Exception:pass
    return out



# === v10 explicit phase specialist ===
MATCH_NAME='crustle'
PRIMARY=X['CHIYU']
SECONDARY=X['PECH']
TEAL_CAP=2
SECONDARY_TURN=4
EX_PENALTY=1200
LILLIE_THRESHOLD=300
SEARCH_ORDER=[X['ULTRA']]

_V10_BASE_MAIN=_x_main;_V10_BASE_CONTEXT=_x_context;_V10_BASE_ROLE=_x_role;_V10_BASE_SEARCH=_x_search_score;_V10_BASE_ATTACH=_x_attach_score;_V10_BASE_TRANSFER=_x_transfer_plan

def _x_matchup(obs):return MATCH_NAME

def _x_primary(obs):return PRIMARY

def _x_secondary(obs):return SECONDARY

def _x_tech_primary(obs):return PRIMARY

def _v10_find_play(obs,cid):
 for i,o in enumerate(obs.select.option):
  c=_source(obs,o)
  if o.type==OptionType.PLAY and c and int(c.id)==cid:return i
 return None

def _v10_find_ability(obs,cid):
 for i,o in enumerate(obs.select.option):
  c=_source(obs,o)
  if o.type==OptionType.ABILITY and c and int(c.id)==cid:return i
 return None

def _v10_ready(obs,cid):
 return next((p for p in _x_board(obs,True) if p.id==cid and _x_ready_damage(obs,p)>0),None)

def _x_role(obs,c,instance=False):
 v=_V10_BASE_ROLE(obs,c,instance);cid=int(getattr(c,'id',0) or 0)
 if cid==PRIMARY:v+=10000 if not _x_has(obs,cid) else 1000
 elif cid==SECONDARY:v+=5500 if not _x_has(obs,cid) else 500
 elif cid==X['LATIAS']:v+=5000 if not _x_has(obs,cid) else -300
 elif cid==X['TEAL']:v+=3500 if _x_count(obs,cid)<TEAL_CAP else -1200
 elif cid in {X['KANGA'],X['FEZ'],X['MEOWTH']} and not _x_has(obs,cid):v-=EX_PENALTY
 return v

def _x_search_score(obs,c,selected=None):
 v=_V10_BASE_SEARCH(obs,c,selected);cid=int(getattr(c,'id',0) or 0)
 if cid==PRIMARY and not _x_has(obs,cid):v+=14000
 elif cid==SECONDARY and not _x_has(obs,cid):v+=7500
 elif cid==X['LATIAS'] and not _x_has(obs,cid):v+=6500
 elif cid==X['PRISM_CARD'] and any(p.id in {PRIMARY,SECONDARY} and _x_energy_progress(p)>0 for p in _x_board(obs,True)):v+=9000
 elif cid==X['G'] and any(p.id in {PRIMARY,SECONDARY} and _x_energy_progress(p)>0 for p in _x_board(obs,True)):v+=3000
 return v

def _x_attach_score(obs,o):
 v=_V10_BASE_ATTACH(obs,o);e=_source(obs,o);t=_target(obs,o)
 if not e or not t:return v
 if t.id==PRIMARY and _x_energy_progress(t)>0:v+=10000
 if t.id==SECONDARY and _x_energy_progress(t)>0:v+=5500
 if int(e.id)==X['PRISM_CARD'] and t.id==PRIMARY:v+=8000
 if t.id in {X['KANGA'],X['TEAL']} and any(p.id==PRIMARY and _x_energy_progress(p)>0 for p in _x_board(obs,True)):v-=7000
 return v

def _x_transfer_plan(obs):
 # Only move an energy when it advances the selected attacker's actual payment.
 pl=_x_pl(obs,True);board=[]
 for area,xs in ((AreaType.ACTIVE,pl.active),(AreaType.BENCH,pl.bench)):
  for i,p in enumerate(xs or []):
   if p:board.append((int(area),i,p))
 best=None
 for target_id,target_bonus in ((PRIMARY,5000),(SECONDARY,2200)):
  for ta,ti,tp in board:
   if tp.id!=target_id or _x_energy_progress(tp)<=0:continue
   for sa,si,sp in board:
    if sp is tp:continue
    es=_x_energy_ids(sp)
    for ei,e in enumerate(es):
     if sp.id in {PRIMARY,SECONDARY} and _x_ready_damage(obs,sp)>0:continue
     before=_x_ready_damage(obs,tp);after_es=_x_energy_ids(tp)+[e];after=_x_ready_damage(obs,tp,after_es)
     prog=(_x_energy_progress(tp)-_x_energy_progress(tp,after_es))*1800
     val=(after-before)*12+prog+target_bonus+(7000 if before<60<=after else 0)
     z=(val,sa,si,ei,int(e),ta,ti,tp.id)
     if best is None or z[0]>best[0]:best=z
 return best if best and best[0]>=3000 else None

def _x_main(obs):
 pl=_x_pl(obs,True);a=pl.active[0] if pl.active else None;opts=obs.select.option
 # Replay-tempo gate: once Cornerstone can Demolish, convert that state into
 # pressure before optional draw/search.  The previous order repeatedly spent
 # extra actions while the lock-breaking attacker was already complete.
 corner=next((p for p in _x_board(obs,True) if p.id==X['CORNER'] and _x_ready_damage(obs,p)>0),None)
 if corner is not None:
  if a is corner:
   attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
   if attacks:return [max(attacks,key=lambda i:_x_damage(obs,a,opts[i].attackId))]
  elif _x_has(obs,X['LATIAS']):
   for i,o in enumerate(opts):
    if o.type==OptionType.RETREAT:return [i]
 # Requested free engines first, but never a shuffle/cycle before an available attack.
 b=_v10_find_play(obs,X['BUG'])
 if b is not None:return [b]
 for cid in (X['KANGA'],X['TEAL'],X['FEZ']):
  i=_v10_find_ability(obs,cid)
  if i is not None:
   if cid==X['TEAL'] and _x_hcount(obs,X['G'])<=0:continue
   return [i]
 # Immediate attack with the matchup attacker.
 if a and a.id in {PRIMARY,SECONDARY,X['CORNER']} and _x_ready_damage(obs,a)>0:
  attacks=[i for i,o in enumerate(opts) if o.type==OptionType.ATTACK]
  if attacks:return [max(attacks,key=lambda i:_x_damage(obs,a,opts[i].attackId))]
 # Latias is infrastructure, then exactly one/two Teal engines and the primary attacker.
 if not _x_has(obs,X['LATIAS']):
  i=_v10_find_play(obs,X['LATIAS'])
  if i is not None:return [i]
 if _x_count(obs,X['TEAL'])<TEAL_CAP:
  i=_v10_find_play(obs,X['TEAL'])
  if i is not None:return [i]
 if not _x_has(obs,PRIMARY):
  i=_v10_find_play(obs,PRIMARY)
  if i is not None:return [i]
  for cid in SEARCH_ORDER:
   i=_v10_find_play(obs,cid)
   if i is not None:return [i]
 if not _x_has(obs,SECONDARY) and int(obs.current.turn or 0)>=SECONDARY_TURN:
  i=_v10_find_play(obs,SECONDARY)
  if i is not None:return [i]
 # Area Zero only after a Tera engine exists.
 if _x_stadium(obs)!=X['AREA'] and any(p.id in {X['TEAL'],X['CORNER'],X['WELL']} for p in _x_board(obs,True)):
  i=_v10_find_play(obs,X['AREA'])
  if i is not None:return [i]
 # Move and attach only toward the matchup attacker.
 plan=_x_transfer_plan(obs)
 if plan:
  i=_v10_find_play(obs,X['ESWITCH'])
  if i is not None:_XMEM['move']=plan;return [i]
 at=[(_x_attach_score(obs,o),i) for i,o in enumerate(opts) if o.type==OptionType.ATTACH]
 if at and max(at)[0]>=1500:return [max(at)[1]]
 # Put a prepared attacker Active before hand cycling.
 ready=next((p for p in pl.bench or [] if p and p.id in {PRIMARY,SECONDARY,X['CORNER']} and _x_ready_damage(obs,p)>0),None)
 if ready and (not a or _x_ready_damage(obs,a)==0) and _x_has(obs,X['LATIAS']):
  for i,o in enumerate(opts):
   if o.type==OptionType.RETREAT:return [i]
 # Use Cipher only when a deterministic draw remains; prioritize Prism through search scorer.
 if _x_remaining_draws(obs)>=1:
  i=_v10_find_play(obs,X['CIPHER'])
  if i is not None:return [i]
 # Lillie is a recovery action after infrastructure/attack progress, not before it.
 if _x_lillie_gain(obs)>LILLIE_THRESHOLD:
  i=_v10_find_play(obs,X['LILLIE'])
  if i is not None:return [i]
 return _V10_BASE_MAIN(obs)


# === v20 replay expected-value residual ===
V20={'cycle_penalty': 2200, 'es_ready': True, 'latias': 0}
_V20_LILLIE=_x_lillie_gain;_V20_SEARCH=_x_search_score;_V20_TRANSFER=_x_transfer_plan;_V20_AGENT=agent
_V20_STATE={'turn':-1,'search_value':0.0,'search_count':0}
_V20_SEARCH_IDS={X['BUG'],X['ULTRA'],X['TERA_ORB'],X['NIGHT'],X['MEOWTH'],X['CIPHER']}

def _x_lillie_gain(obs):
 base=_V20_LILLIE(obs)
 # The fetched card is only useful if it survives the same-turn redraw.  The
 # opportunity cost scales with the realized search actions, not a hard ban.
 return base-V20['cycle_penalty']*_V20_STATE.get('search_count',0)-0.18*_V20_STATE.get('search_value',0.0)

def _x_search_score(obs,c,selected=None):
 v=_V20_SEARCH(obs,c,selected)
 if c is not None and int(getattr(c,'id',0) or 0)==X['LATIAS'] and not _x_has(obs,X['LATIAS']):v+=V20.get('latias',0)
 return v

def _v20_target(obs,area,index):
 pl=_x_pl(obs,True);xs=pl.active if int(area)==int(AreaType.ACTIVE) else pl.bench if int(area)==int(AreaType.BENCH) else []
 try:return xs[int(index)]
 except Exception:return None

def _x_transfer_plan(obs):
 plan=_V20_TRANSFER(obs)
 if not plan or not V20.get('es_ready'):return plan
 try:
  val,sa,si,ei,e,ta,ti,tid=plan;tp=_v20_target(obs,ta,ti)
  if tp is None:return None
  before=_x_ready_damage(obs,tp);after=_x_ready_damage(obs,tp,_x_energy_ids(tp)+[e])
  # Probability proxy: the move must open a legal meaningful attack now, or
  # improve an already-ready attacker. Pure future compression is negative EV.
  if after<60 or after<=before:return None
 except Exception:return None
 return plan

def agent(observation:dict)->list[int]:
 if observation.get('select') is None:
  if observation.get('current') is None:_V20_STATE.update({'turn':-1,'search_value':0.0,'search_count':0})
  return _V20_AGENT(observation)
 cur=observation.get('current') or {};turn=int(cur.get('turn') or -1)
 if turn!=_V20_STATE.get('turn'):_V20_STATE.update({'turn':turn,'search_value':0.0,'search_count':0})
 out=_V20_AGENT(observation)
 try:
  obs=to_observation_class(observation)
  if out and len(out)==1 and obs.select and 0<=out[0]<len(obs.select.option):
   o=obs.select.option[out[0]];c=_source(obs,o)
   if o.type==OptionType.PLAY and c and int(c.id) in _V20_SEARCH_IDS:
    _V20_STATE['search_count']+=1
    # Expected retained role value. It is deliberately capped so a search can
    # still be followed by Lillie when the rest of the hand is truly poor.
    _V20_STATE['search_value']+=min(9000,max(0,_x_role(obs,c)))
 except Exception:pass
 return out

# === v21 Cornerstone-first lock breaker =====================================
# Demolish ignores the defending Crustle's protection and survives Superb
# Scissors.  Chi-Yu remains the faster backup when Prism is unavailable.
PRIMARY=X['CORNER']
SECONDARY=X['CHIYU']
SEARCH_ORDER=[X['TERA_ORB'],X['ULTRA']]
SECONDARY_TURN=3
EX_PENALTY=0
