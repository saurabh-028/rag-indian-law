/* =========================================================
   ShastraShaw — Chat Interface JS
   ========================================================= */

// ── Configure marked.js ────────────────────────────────────
marked.use({
  breaks: true,
  gfm: true,
});

// ── Markdown renderer with sanitization ───────────────────
function renderMarkdown(text) {
  const rawHtml = marked.parse(text || '');
  const clean   = DOMPurify.sanitize(rawHtml, {
    ADD_ATTR: ['target', 'rel'],
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'del', 's',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'code', 'pre',
      'blockquote', 'hr',
      'a',
      'table', 'thead', 'tbody', 'tr', 'th', 'td',
    ],
  });

  // Make all links open in new tab (safe since we sanitized)
  const tmp = document.createElement('div');
  tmp.innerHTML = clean;
  tmp.querySelectorAll('a').forEach(a => {
    a.target = '_blank';
    a.rel    = 'noopener noreferrer';
  });
  return tmp.innerHTML;
}

// ── State ──────────────────────────────────────────────────
let isLoading         = false;
let isRecording       = false;
let isSpeaking        = false;
let lastInputWasVoice = false;
let currentSector     = null;
let selectedLang      = 'auto';   // 'auto' | 'en' | 'hi'
let currentSessionId  = null;     // set from the server's response; reused so follow-ups get memory context

// ── Elements ───────────────────────────────────────────────
const messagesEl      = document.getElementById('messages');
const emptyState      = document.getElementById('emptyState');
const questionInput   = document.getElementById('questionInput');
const sendBtn         = document.getElementById('sendBtn');
const micBtn          = document.getElementById('micBtn');
const stopSpeechBtn   = document.getElementById('stopSpeechBtn');
const sidebar         = document.getElementById('sidebar');
const sidebarOverlay  = document.getElementById('sidebarOverlay');
const sidebarToggleBtn = document.getElementById('sidebarToggle');
const topbarSector    = document.getElementById('topbarSector');
const topbarSectorLabel = document.getElementById('topbarSectorLabel');

// ── Auto-resize textarea ───────────────────────────────────
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
});

// ── Mobile sidebar ─────────────────────────────────────────
sidebarToggleBtn?.addEventListener('click', () => {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('open');
});

sidebarOverlay?.addEventListener('click', () => {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('open');
});

// ── Sector filter ──────────────────────────────────────────
document.querySelectorAll('.sector-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.sector-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSector = btn.dataset.sector || null;

    // Update topbar sector indicator
    if (currentSector) {
      topbarSectorLabel.textContent = currentSector.replace(/_/g, ' ').toUpperCase();
      topbarSector.style.display = 'block';
    } else {
      topbarSector.style.display = 'none';
    }

    // Close sidebar on mobile after selection
    if (window.innerWidth <= 800) {
      sidebar.classList.remove('open');
      sidebarOverlay.classList.remove('open');
    }
  });
});

// ── New chat ───────────────────────────────────────────────
document.getElementById('newChatBtn')?.addEventListener('click', () => {
  messagesEl.querySelectorAll('.message-wrap, .typing-wrap').forEach(el => el.remove());
  showEmpty();
  questionInput.value = '';
  questionInput.style.height = 'auto';
  currentSessionId = null;   // start a fresh conversation — don't carry over prior context
  questionInput.focus();
});

// ── Language selector buttons ──────────────────────────────
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedLang = btn.dataset.lang;
  });
});

// ── Speech Recognition (STT) ───────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous    = false;
  recognition.interimResults = false;
  // lang is set dynamically at start time based on selectedLang

  recognition.onstart = () => {
    isRecording = true;
    micBtn.classList.add('recording');
    micBtn.title = 'Listening… tap to stop';
  };

  recognition.onend = () => {
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.title = 'Voice input';
  };

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript.trim();
    if (transcript) {
      questionInput.value = transcript;
      questionInput.style.height = 'auto';
      questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
      lastInputWasVoice = true;
      sendMessage();
    }
  };

  recognition.onerror = (e) => {
    console.warn('Speech recognition error:', e.error);
    isRecording = false;
    micBtn.classList.remove('recording');
  };
} else {
  micBtn.disabled = true;
  micBtn.title    = 'Voice not supported — use Chrome';
}

// ── Strip markdown to plain text for TTS ──────────────────
function toPlainText(markdown) {
  const tmp = document.createElement('div');
  tmp.innerHTML = marked.parse(markdown || '');
  return tmp.innerText;
}

// ── Text-to-Speech (TTS) ───────────────────────────────────
function speak(text, lang) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();

  const utt  = new SpeechSynthesisUtterance(toPlainText(text));
  utt.lang   = lang === 'hi' ? 'hi-IN' : 'en-IN';
  utt.rate   = 0.92;
  utt.pitch  = 1;

  // Show stop button immediately — Chrome's onstart is unreliable
  isSpeaking = true;
  stopSpeechBtn.style.display = 'flex';

  utt.onend = utt.onerror = () => {
    isSpeaking = false;
    stopSpeechBtn.style.display = 'none';
  };

  window.speechSynthesis.speak(utt);
}

// ── Stop speech button ─────────────────────────────────────
stopSpeechBtn?.addEventListener('click', () => {
  window.speechSynthesis.cancel();
  isSpeaking = false;
  stopSpeechBtn.style.display = 'none';
});

// ── Mic button ─────────────────────────────────────────────
micBtn.addEventListener('click', () => {
  if (!recognition) return;
  if (isRecording) recognition.stop();
  else {
    // Set recognition language based on the selected response language
    recognition.lang = selectedLang === 'hi' ? 'hi-IN' : 'en-IN';
    try { recognition.start(); }
    catch (e) { console.warn('recognition.start error:', e); }
  }
});

// ── Send on Enter / button ─────────────────────────────────
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', () => { if (!isLoading) sendMessage(); });

// ── Suggestion chips ────────────────────────────────────────
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    questionInput.value = chip.textContent.trim();
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
    sendMessage();
  });
});

// ── Helpers ────────────────────────────────────────────────
function setLoading(state) {
  isLoading               = state;
  questionInput.disabled  = state;
  sendBtn.disabled        = state;
  if (recognition) micBtn.disabled = state;
}

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

function hideEmpty() {
  if (emptyState) emptyState.style.display = 'none';
}

function showEmpty() {
  if (emptyState) emptyState.style.display = 'flex';
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Render: user message ───────────────────────────────────
function addUserMessage(text) {
  hideEmpty();
  const wrap       = document.createElement('div');
  wrap.className   = 'message-wrap user';
  wrap.innerHTML   = `<div class="bubble">${esc(text)}</div>`;
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

// ── Render: typing indicator ───────────────────────────────
function addTyping() {
  const wrap     = document.createElement('div');
  wrap.className = 'typing-wrap';
  wrap.id        = 'typingIndicator';
  wrap.innerHTML = `
    <div class="typing-bubble">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function removeTyping() {
  document.getElementById('typingIndicator')?.remove();
}

// ── Render: bot message (Markdown) ────────────────────────
function addBotMessage(data, wasVoice) {
  removeTyping();

  const safeHtml = renderMarkdown(data.answer || '');

  // Badges
  let badgesHtml = '';
  if (data.sector_used) {
    badgesHtml += `<span class="badge badge-sector">${esc(data.sector_used.replace(/_/g, ' '))}</span>`;
  }
  // Use the explicitly selected language for the badge; fall back to backend detection
  const displayLang = selectedLang !== 'auto' ? selectedLang : (data.detected_lang || 'en');
  if (displayLang === 'hi') {
    badgesHtml += `<span class="badge badge-lang-hi">HI · हिंदी</span>`;
  } else {
    badgesHtml += `<span class="badge badge-lang-en">EN</span>`;
  }

  // Sources
  let sourcesHtml = '';
  if (data.sources && data.sources.length) {
    const items = data.sources
      .map(s => `<div class="source-item">${esc(s)}</div>`)
      .join('');
    sourcesHtml = `
      <div class="sources-wrap">
        <button class="sources-toggle" onclick="toggleSources(this)">
          <svg class="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
          ${data.sources.length} source${data.sources.length > 1 ? 's' : ''}
        </button>
        <div class="sources-list">${items}</div>
      </div>`;
  }

  const wrap     = document.createElement('div');
  wrap.className = 'message-wrap bot';
  wrap.innerHTML = `
    <div class="bubble">${safeHtml}</div>
    ${badgesHtml ? `<div class="meta-row">${badgesHtml}</div>` : ''}
    ${sourcesHtml}`;
  messagesEl.appendChild(wrap);
  scrollToBottom();

  if (wasVoice) speak(data.answer, data.detected_lang || 'en');
}

// ── Render: error ──────────────────────────────────────────
function addErrorMessage(msg) {
  removeTyping();
  const wrap     = document.createElement('div');
  wrap.className = 'message-wrap bot';
  wrap.innerHTML = `<div class="bubble error-bubble">${esc(msg)}</div>`;
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

// ── Sources toggle ─────────────────────────────────────────
function toggleSources(btn) {
  const list = btn.nextElementSibling;
  const open = list.classList.toggle('open');
  btn.classList.toggle('open', open);
}

// ── Main send ──────────────────────────────────────────────
async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || isLoading) return;

  const wasVoice    = lastInputWasVoice;
  lastInputWasVoice = false;
  questionInput.value       = '';
  questionInput.style.height = 'auto';

  setLoading(true);
  addUserMessage(question);
  addTyping();

  try {
    const body = { question, top_k: 5 };
    if (currentSector) body.sector_filter = currentSector;
    if (selectedLang !== 'auto') body.lang = selectedLang;
    if (currentSessionId) body.session_id = currentSessionId;

    const res = await fetch('/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    currentSessionId = data.session_id || currentSessionId;
    addBotMessage(data, wasVoice);

  } catch (err) {
    addErrorMessage(`Something went wrong: ${err.message}`);
  } finally {
    setLoading(false);
    questionInput.focus();
  }
}
