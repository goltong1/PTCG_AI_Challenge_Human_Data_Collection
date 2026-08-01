const $ = (s) => document.querySelector(s);
const DB_NAME = 'cabt-local-replays-v1';
const STORE = 'replays';
let dbPromise = null;
let activeReplay = null;
let frameIndex = 0;
let playing = false;
let timer = null;
let viewSeat = 0;
let registeredCards = new Map();
let cardKey = 0;

function esc(v) {
  return String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(window.__cabtToast);
  window.__cabtToast = setTimeout(() => el.classList.add('hidden'), 3600);
}
function cardImage(id) { return `./assets/cards/${Number(id)}.webp`; }
function registerCard(card) { const key = `card-${++cardKey}`; registeredCards.set(key, card); return key; }
function cardImg(card, cls='') {
  if (!card?.id) return '<div class="empty-slot">EMPTY</div>';
  const key = registerCard(card);
  return `<img class="${cls} card-click" data-card-key="${key}" src="${cardImage(card.id)}" alt="${esc(card.name || `Card ${card.id}`)}" onerror="this.style.visibility='hidden'">`;
}

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}
async function dbPut(record) {
  const db = await openDb();
  await new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,'readwrite');
    tx.objectStore(STORE).put(record);
    tx.oncomplete=resolve; tx.onerror=()=>reject(tx.error);
  });
}
async function dbGet(id) {
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const req=db.transaction(STORE).objectStore(STORE).get(id);
    req.onsuccess=()=>resolve(req.result); req.onerror=()=>reject(req.error);
  });
}
async function dbAll() {
  const db=await openDb();
  return new Promise((resolve,reject)=>{
    const req=db.transaction(STORE).objectStore(STORE).getAll();
    req.onsuccess=()=>resolve(req.result||[]); req.onerror=()=>reject(req.error);
  });
}
async function dbDelete(id) {
  const db=await openDb();
  await new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,'readwrite'); tx.objectStore(STORE).delete(id);
    tx.oncomplete=resolve; tx.onerror=()=>reject(tx.error);
  });
}
async function dbClear() {
  const db=await openDb();
  await new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,'readwrite'); tx.objectStore(STORE).clear();
    tx.oncomplete=resolve; tx.onerror=()=>reject(tx.error);
  });
}

function deriveReleaseUrl() {
  const host = location.hostname;
  const path = location.pathname.split('/').filter(Boolean);
  if (host.endsWith('.github.io') && path.length) {
    const owner = host.split('.')[0];
    const repo = path[0];
    return `https://github.com/${owner}/${repo}/releases/latest/download/CABT_Local_AI_Arena_v6.zip`;
  }
  return './CABT_Local_AI_Arena_v6.zip';
}

function visualFromStep(step) {
  if (!Array.isArray(step)) return null;
  for (const seat of step) {
    const visuals = seat?.visualize;
    if (Array.isArray(visuals) && visuals.length) return visuals[visuals.length - 1];
  }
  for (const seat of step) {
    const observation = seat?.observation;
    if (observation?.current) {
      return {
        current: observation.current,
        select: observation.select || null,
        logs: observation.logs || [],
        selected: Array.isArray(seat.action) ? seat.action : [],
      };
    }
  }
  return null;
}
function normalizeReplay(payload) {
  let frames = [];
  let title = '';
  if (Array.isArray(payload)) {
    frames = payload.filter(x => x && x.current && Array.isArray(x.current.players));
  } else if (payload && Array.isArray(payload.steps)) {
    title = payload.title || payload.name || '';
    frames = payload.steps.map(visualFromStep).filter(x => x?.current && Array.isArray(x.current.players));
  } else if (payload?.frames && Array.isArray(payload.frames)) {
    frames = payload.frames.filter(x => x?.current);
    title = payload.title || '';
  }
  if (!frames.length) throw new Error('지원되는 CABT 리플레이 장면을 찾지 못했습니다.');
  return { frames, title };
}
async function readReplayFile(file) {
  const lower = file.name.toLowerCase();
  if (lower.endsWith('.zip')) {
    const zip = await JSZip.loadAsync(await file.arrayBuffer());
    const names = Object.keys(zip.files).filter(n => !zip.files[n].dir);
    const preferred = [
      names.find(n => n.toLowerCase().endsWith('replay_visualize.json')),
      names.find(n => n.toLowerCase().endsWith('official_replay.json')),
      names.find(n => n.toLowerCase().endsWith('.json')),
    ].filter(Boolean)[0];
    if (!preferred) throw new Error('ZIP 안에서 CABT JSON을 찾지 못했습니다.');
    return JSON.parse(await zip.file(preferred).async('string'));
  }
  return JSON.parse(await file.text());
}
function makeReplayId(file, frames) {
  const first = frames[0]?.current || {};
  const last = frames[frames.length-1]?.current || {};
  return `${file.name}:${file.size}:${first.turn || 0}:${last.turn || 0}:${frames.length}`;
}
async function loadFile(file) {
  if (!file) return;
  try {
    toast('리플레이를 브라우저에서 분석하는 중입니다.');
    const raw = await readReplayFile(file);
    const normalized = normalizeReplay(raw);
    const record = {
      id: makeReplayId(file, normalized.frames),
      name: normalized.title || file.name,
      sourceName: file.name,
      loadedAt: Date.now(),
      frames: normalized.frames,
    };
    await dbPut(record);
    await renderHistory();
    openReplay(record);
    toast(`${record.name} · ${record.frames.length}개 장면을 로컬에 저장했습니다.`);
  } catch (e) { toast(e.message || String(e)); }
  $('#fileInput').value='';
}

function resultLabel(result) {
  if (result === -1 || result == null) return '진행 중';
  if (result === 0) return 'Player 0 승리';
  if (result === 1) return 'Player 1 승리';
  return '무승부';
}
function pokemonCard(card) {
  if (!card) return '<div class="empty-slot">EMPTY</div>';
  const energies = Array.isArray(card.energyCards) ? card.energyCards : [];
  const energyCount = Array.isArray(card.energies) ? card.energies.length : energies.length;
  return `<article class="pokemon-card">
    ${cardImg(card)}
    <div class="pokemon-name">${esc(card.name || `#${card.id}`)}</div>
    <div class="hp-line"><span>HP ${Number(card.hp ?? 0)}</span><span>${Number(card.maxHp ?? card.hp ?? 0)}</span></div>
    <div class="energy-strip">
      ${energies.slice(0,5).map(e => cardImg(e)).join('')}
      ${energyCount ? `<span class="energy-count">⚡ ${energyCount}</span>` : '<span style="color:#60736b;font-size:10px">NO ENERGY</span>'}
    </div>
  </article>`;
}
function prizeHtml(prize) {
  const count = Array.isArray(prize) ? prize.length : Number(prize || 0);
  return `<div class="prize-cards">${Array.from({length:Math.min(6,count)},()=>'<i class="card-back"></i>').join('')}</div>`;
}
function pileCard(title, count, extra='') {
  return `<div class="pile"><small>${title}</small><strong>${count}</strong>${extra}</div>`;
}
function renderPlayer(player, index, hiddenHand=false) {
  const active = Array.isArray(player?.active) ? player.active[0] : null;
  const bench = Array.isArray(player?.bench) ? player.bench : [];
  const discard = Array.isArray(player?.discard) ? player.discard : [];
  const hand = Array.isArray(player?.hand) ? player.hand : [];
  const prize = Array.isArray(player?.prize) ? player.prize : [];
  const handCount = Number(player?.handCount ?? hand.length);
  const trashPreview = discard.slice(-4).reverse().map(c => cardImg(c)).join('');
  const handPreview = !hiddenHand && hand.length ? hand.slice(0,8).map(c => cardImg(c)).join('') : '';
  return `<div class="player-head"><div><strong>Player ${index}</strong><small>${active?.name ? `Active · ${esc(active.name)}` : 'Active 없음'}</small></div><small>Bench ${bench.length}/${player?.benchMax ?? 5}</small></div>
    <div class="zone-grid">
      <div class="active-slot"><div class="slot-label">ACTIVE</div>${pokemonCard(active)}</div>
      <div class="bench-row">${Array.from({length:Math.max(5,bench.length)},(_,i)=>pokemonCard(bench[i])).join('')}</div>
      <div class="pile-column">
        ${pileCard('DECK', Number(player?.deckCount ?? 0))}
        ${pileCard('PRIZE', prize.length, prizeHtml(prize))}
        ${pileCard('HAND', handCount, `<div class="hand-strip">${handPreview}</div>`)}
        ${pileCard('TRASH', discard.length, `<div class="trash-preview">${trashPreview}</div>`)}
      </div>
    </div>`;
}
function describeLog(log) {
  if (!log || typeof log !== 'object') return String(log ?? '');
  const p = log.playerIndex != null ? `P${log.playerIndex}` : '';
  const card = log.cardId != null ? `Card #${log.cardId}` : '';
  const type = log.type != null ? `Log ${log.type}` : 'Action';
  return [p,type,card].filter(Boolean).join(' · ');
}
function renderFrame() {
  if (!activeReplay) return;
  registeredCards = new Map(); cardKey=0;
  const frames=activeReplay.frames;
  frameIndex=Math.max(0,Math.min(frameIndex,frames.length-1));
  const frame=frames[frameIndex];
  const current=frame.current;
  const players=current.players || [{},{}];
  const bottom=viewSeat; const top=1-viewSeat;
  $('#topPlayer').innerHTML=renderPlayer(players[top]||{},top,true);
  $('#bottomPlayer').innerHTML=renderPlayer(players[bottom]||{},bottom,false);
  const stadium=Array.isArray(current.stadium)?current.stadium[0]:null;
  $('#stadium').innerHTML=stadium?cardImg(stadium):'<span>비어 있음</span>';
  $('#turnBadge').textContent=`TURN ${Number(current.turn||0)}`;
  $('#frameMeta').textContent=`${frameIndex+1} / ${frames.length}`;
  $('#actionText').textContent=`Action ${Number(current.turnActionCount||0)} · First P${current.firstPlayer ?? '-'}`;
  $('#resultText').textContent=resultLabel(current.result);
  $('#scrubber').max=String(frames.length-1); $('#scrubber').value=String(frameIndex);
  const logs=Array.isArray(frame.logs)?frame.logs:[];
  $('#logList').innerHTML=logs.length?logs.slice(-8).reverse().map(l=>`<div class="log-item">${esc(describeLog(l))}</div>`).join(''):'<div class="empty">표시할 로그가 없습니다.</div>';
  $('#prevButton').disabled=frameIndex<=0; $('#nextButton').disabled=frameIndex>=frames.length-1;
  $('#playButton').textContent=playing?'Ⅱ':'▶';
  document.querySelectorAll('.card-click').forEach(el=>el.addEventListener('click',()=>showCard(registeredCards.get(el.dataset.cardKey))));
}
function showCard(card) {
  if (!card) return;
  $('#cardDialogBody').innerHTML=`<div class="dialog-card">${cardImg(card)}<div><span class="eyebrow">CARD #${Number(card.id)}</span><h2>${esc(card.name||'Unknown')}</h2><p>HP ${Number(card.hp??0)} / ${Number(card.maxHp??card.hp??0)}</p><p>부착 에너지 ${Array.isArray(card.energies)?card.energies.length:0}개 · 도구 ${Array.isArray(card.tools)?card.tools.length:0}개</p></div></div>`;
  $('#cardDialog').showModal();
}
function stopPlay(){playing=false;clearTimeout(timer);timer=null;renderFrame();}
function schedule(){clearTimeout(timer);if(!playing||!activeReplay)return;if(frameIndex>=activeReplay.frames.length-1){stopPlay();return;}timer=setTimeout(()=>{frameIndex++;renderFrame();schedule();},Number($('#speedSelect').value||750));}
function togglePlay(){if(playing){stopPlay();return;}playing=true;if(frameIndex>=activeReplay.frames.length-1)frameIndex=0;renderFrame();schedule();}
function openReplay(record){activeReplay=record;frameIndex=0;viewSeat=0;stopPlay();$('#landing').classList.add('hidden');$('#viewer').classList.remove('hidden');$('#replayName').textContent=record.name;$('#viewSeat').value='0';renderFrame();scrollTo({top:0,behavior:'smooth'});}
function closeReplay(){stopPlay();activeReplay=null;$('#viewer').classList.add('hidden');$('#landing').classList.remove('hidden');renderHistory();}

async function renderHistory(){
  const rows=(await dbAll()).sort((a,b)=>b.loadedAt-a.loadedAt);
  $('#historyList').innerHTML=rows.length?rows.map(r=>`<button class="history-card" type="button" data-id="${esc(r.id)}"><strong>${esc(r.name)}</strong><small>${r.frames.length} frames · ${new Date(r.loadedAt).toLocaleString('ko-KR')}</small></button>`).join(''):'<div class="empty">저장된 리플레이가 없습니다.</div>';
  document.querySelectorAll('.history-card').forEach(btn=>btn.addEventListener('click',async()=>{const r=await dbGet(btn.dataset.id);if(r)openReplay(r);}));
}

$('#downloadButton').href=deriveReleaseUrl();
$('#pickButton').addEventListener('click',()=>$('#fileInput').click());
$('#fileInput').addEventListener('change',e=>loadFile(e.target.files[0]));
$('#dropzone').addEventListener('dragover',e=>{e.preventDefault();$('#dropzone').classList.add('dragging');});
$('#dropzone').addEventListener('dragleave',()=>$('#dropzone').classList.remove('dragging'));
$('#dropzone').addEventListener('drop',e=>{e.preventDefault();$('#dropzone').classList.remove('dragging');loadFile(e.dataTransfer.files[0]);});
$('#homeButton').addEventListener('click',()=>activeReplay?closeReplay():scrollTo({top:0,behavior:'smooth'}));
$('#closeViewer').addEventListener('click',closeReplay);
$('#prevButton').addEventListener('click',()=>{stopPlay();frameIndex--;renderFrame();});
$('#nextButton').addEventListener('click',()=>{stopPlay();frameIndex++;renderFrame();});
$('#playButton').addEventListener('click',togglePlay);
$('#speedSelect').addEventListener('change',()=>{if(playing)schedule();});
$('#scrubber').addEventListener('input',e=>{stopPlay();frameIndex=Number(e.target.value);renderFrame();});
$('#viewSeat').addEventListener('change',e=>{viewSeat=Number(e.target.value);renderFrame();});
$('#closeCardDialog').addEventListener('click',()=>$('#cardDialog').close());
$('#cardDialog').addEventListener('click',e=>{if(e.target===$('#cardDialog'))$('#cardDialog').close();});
$('#clearHistory').addEventListener('click',async()=>{if(confirm('이 브라우저에 저장된 리플레이를 모두 삭제할까요?')){await dbClear();renderHistory();}});
$('#forgetReplay').addEventListener('click',async()=>{if(!activeReplay)return;if(confirm('현재 리플레이를 이 브라우저에서 삭제할까요?')){await dbDelete(activeReplay.id);closeReplay();}});

renderHistory().catch(e=>toast(`로컬 저장소를 열 수 없습니다: ${e.message}`));
