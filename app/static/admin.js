const $ = (selector) => document.querySelector(selector);
let adminToken = sessionStorage.getItem('cabtAdminToken') || '';
let submissions = [];

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  }[char]));
}

function toast(message, tone = 'error') {
  const element = $('#toast');
  element.textContent = message;
  element.dataset.tone = tone;
  element.classList.remove('hidden');
  clearTimeout(window.__adminToast);
  window.__adminToast = setTimeout(() => element.classList.add('hidden'), 4500);
}

async function adminFetch(url, options = {}) {
  const token = $('#admin-token').value.trim();
  if (!token) throw new Error('관리자 토큰을 입력하세요.');
  adminToken = token;
  sessionStorage.setItem('cabtAdminToken', token);
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response;
}

function resultLabel(value) {
  if (value === 'human_win') return '<span class="result-chip human-win">사람 승리</span>';
  if (value === 'ai_win') return '<span class="result-chip ai-win">AI 승리</span>';
  return '<span class="result-chip draw">무승부</span>';
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value || '-') : date.toLocaleString('ko-KR');
}

function render() {
  $('#metric-total').textContent = submissions.length;
  $('#metric-human').textContent = submissions.filter((item) => item.result_text === 'human_win').length;
  $('#metric-ai').textContent = submissions.filter((item) => item.result_text === 'ai_win').length;
  $('#metric-draw').textContent = submissions.filter((item) => item.result_text === 'draw').length;
  const rows = $('#submission-rows');
  if (!submissions.length) {
    rows.innerHTML = '<tr><td colspan="7" class="admin-empty">전송된 결과가 없습니다.</td></tr>';
    return;
  }
  rows.innerHTML = submissions.map((item) => `<tr>
    <td><time>${esc(formatDate(item.submitted_at))}</time><small>${esc(item.submission_id)}</small></td>
    <td>${esc(item.player_name || 'Anonymous')}</td>
    <td>${esc(item.agent_name || item.agent_id || 'AI')}</td>
    <td>${esc(item.human_deck_label || '-')}</td>
    <td>${resultLabel(item.result_text)}</td>
    <td>${Number(item.decision_count || 0)}</td>
    <td><button class="button quiet download-submission" data-id="${esc(item.submission_id)}" type="button" ${item.zip_available ? '' : 'disabled'}>ZIP</button></td>
  </tr>`).join('');
  rows.querySelectorAll('.download-submission').forEach((button) => {
    button.addEventListener('click', () => downloadSubmission(button.dataset.id));
  });
}

async function loadSubmissions() {
  $('#admin-status').textContent = '결과를 불러오는 중…';
  try {
    const response = await adminFetch('/api/admin/submissions?limit=1000');
    const payload = await response.json();
    submissions = payload.submissions || [];
    render();
    $('#admin-status').textContent = `최근 ${submissions.length}개 결과를 표시합니다.`;
  } catch (error) {
    $('#admin-status').textContent = error.message;
    toast(error.message);
  }
}

async function downloadSubmission(submissionId) {
  try {
    const response = await adminFetch(`/api/admin/submissions/${encodeURIComponent(submissionId)}/zip`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${submissionId}.zip`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    toast(error.message);
  }
}

$('#admin-token').value = adminToken;
$('#load-submissions').addEventListener('click', loadSubmissions);
$('#refresh-submissions').addEventListener('click', loadSubmissions);
$('#admin-token').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') loadSubmissions();
});
