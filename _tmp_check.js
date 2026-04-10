




const API_MODE = (new URLSearchParams(window.location.search).get('api_mode') || '').toLowerCase();
const USE_PHP_API = API_MODE === 'php' || window.location.pathname.endsWith('.php');
function apiPath(path) {
  if (!USE_PHP_API) return path;
  if (path.startsWith('/stats/')) {
    return `/stats.php?session_id=${encodeURIComponent(path.split('/').pop() || '')}`;
  }
  return `${path}.php`;
}
// =====================================================================
// STATE
// =====================================================================
const state = {
  sessionId: null,
  topic: '',
  explanationMode: 'in_depth',
  slides: [],
  currentIndex: 0,
  timerInterval: null,
  timerSeconds: 0,
  slideStartTime: 0,
  performance: [],            // local mirror of server-side records
  understandingChoice: null,  // 'yes' | 'somewhat' | 'no'
  quizQuestions: [],
  quizAnswers: [],            // chosen option index per question
  quizCurrentQ: 0,
  quizDone: false,
  adaptiveProfile: null,
  visualLoaded: false,        // has the visual been fetched for current slide?
  visualOpen: false,          // is the visual panel currently visible?
  xp: 0,
  streakSlides: 0,
  chatHistory: [],            // [{role:'user'|'tutor', text}] per session
  notesBySlide: {},           // { slideIndex: "text" }
  notesOpen: false,
};

// =====================================================================
// SCREEN MANAGEMENT
// =====================================================================
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// =====================================================================
// TOAST
// =====================================================================
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}

// =====================================================================
// START SCREEN
// =====================================================================
const topicInput = document.getElementById('topic-input');
const charCount  = document.getElementById('char-count');
const modeOptions = Array.from(document.querySelectorAll('input[name="explain-mode"]'));
const modeLabels = {
  brief: document.getElementById('mode-brief-label'),
  in_depth: document.getElementById('mode-depth-label')
};
const hudExplanationMode = document.getElementById('hud-explanation-mode');

function setExplanationMode(mode) {
  const next = (mode === 'brief') ? 'brief' : 'in_depth';
  state.explanationMode = next;
  modeOptions.forEach(opt => { opt.checked = (opt.value === next); });
  Object.entries(modeLabels).forEach(([key, el]) => {
    if (!el) return;
    el.classList.toggle('active', key === next);
  });
  if (hudExplanationMode) hudExplanationMode.value = next;
}

function getExplanationMode() {
  const selected = modeOptions.find(opt => opt.checked);
  return selected ? selected.value : 'in_depth';
}

topicInput.addEventListener('input', () => {
  charCount.textContent = topicInput.value.length;
});
modeOptions.forEach(opt => {
  opt.addEventListener('change', () => setExplanationMode(opt.value));
});
if (hudExplanationMode) {
  hudExplanationMode.addEventListener('change', () => setExplanationMode(hudExplanationMode.value));
}
setExplanationMode('in_depth');

// =====================================================================
// LEARNING CONTROLS — hint bar (show when all 3 dropdowns are set)
// =====================================================================
(function () {
  const lcSubject = document.getElementById('lc-subject');
  const lcMode    = document.getElementById('lc-mode');
  const lcDepth   = document.getElementById('lc-depth');
  const lcHint    = document.getElementById('lc-hint');
  const lcHintTxt = document.getElementById('lc-hint-text');

  function updateHint() {
    const s = lcSubject.value;
    const m = lcMode.value;
    const d = lcDepth.value;
    if (s && m && d) {
      lcHintTxt.textContent = `${s} · ${m} · ${d}`;
      lcHint.classList.add('show');
    } else {
      lcHint.classList.remove('show');
    }
  }

  [lcSubject, lcMode, lcDepth].forEach(el => el.addEventListener('change', updateHint));
})();

document.getElementById('start-btn').addEventListener('click', startSession);
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startSession(); }
});

async function startSession() {
  const topic = topicInput.value.trim();
  if (!topic) { showToast('Please enter a topic first.'); return; }
  const explanationMode = getExplanationMode();

  // Learning controls
  const subject = document.getElementById('lc-subject').value;
  const mode    = document.getElementById('lc-mode').value;
  const depth   = document.getElementById('lc-depth').value;

  state.topic = topic;
  state.explanationMode = explanationMode;
  setLoading('Generating your lecture…', 'Crafting 10–12 slides on: ' + topic);
  showScreen('screen-loading');

  try {
    const res = await fetch(apiPath('/generate'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, explanation_mode: explanationMode, subject, mode, depth }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Failed to generate slides.');

    state.sessionId = data.session_id;
    state.slides    = data.slides;
    state.currentIndex = 0;
    state.performance  = [];
    state.adaptiveProfile = null;
    state.xp = 0;
    state.streakSlides = 0;
    state.chatHistory = [];
    state.notesBySlide = {};

    document.getElementById('hud-topic').textContent = topic;
    showScreen('screen-learning');
    loadSlide(0);

  } catch (err) {
    showScreen('screen-start');
    showToast(err.message);
  }
}

function setLoading(title, sub) {
  document.getElementById('loading-text').textContent = title;
  document.getElementById('loading-sub').textContent  = sub;
}

// =====================================================================
// HUD INLINE SEARCH — search a new topic without leaving the page
// =====================================================================
(function () {
  const hudInput = document.getElementById('hud-search-input');
  const hudBtn   = document.getElementById('hud-search-btn');

  async function hudSearch() {
    const topic = hudInput.value.trim();
    if (!topic) { showToast('Enter a topic to search.'); return; }
    const explanationMode = hudExplanationMode ? hudExplanationMode.value : state.explanationMode;

    // Carry over current learning control selections
    const subject = document.getElementById('lc-subject').value;
    const mode    = document.getElementById('lc-mode').value;
    const depth   = document.getElementById('lc-depth').value;

    // Reset state fully
    stopTimer();
    state.sessionId = null; state.slides = []; state.currentIndex = 0;
    state.performance = []; state.adaptiveProfile = null;
    state.visualLoaded = false; state.visualOpen = false;
    state.xp = 0; state.streakSlides = 0;
    state.chatHistory = []; state.notesBySlide = {};

    // Clear hud input
    hudInput.value = '';

    // Also sync the start-screen input so it stays consistent
    topicInput.value = topic;
    charCount.textContent = topic.length;

    state.topic = topic;
    state.explanationMode = explanationMode;
    setLoading('Generating your lecture…', 'Crafting 10–12 slides on: ' + topic);
    showScreen('screen-loading');
    hudBtn.disabled = true;

    try {
      const res = await fetch(apiPath('/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, explanation_mode: explanationMode, subject, mode, depth }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to generate slides.');

      state.sessionId = data.session_id;
      state.slides    = data.slides;
      state.currentIndex = 0;
      state.performance  = [];
      state.adaptiveProfile = null;
      state.xp = 0;
      state.streakSlides = 0;
      state.chatHistory = [];
      state.notesBySlide = {};

      document.getElementById('hud-topic').textContent = topic;
      showScreen('screen-learning');
      loadSlide(0);

    } catch (err) {
      showScreen('screen-learning');   // stay on learning screen, don't go back to start
      showToast(err.message);
    } finally {
      hudBtn.disabled = false;
    }
  }

  hudBtn.addEventListener('click', hudSearch);
  hudInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); hudSearch(); }
  });
})();

// =====================================================================
// SLIDE RENDERING
// =====================================================================
function loadSlide(index, overrideSlide = null) {
  const slide = overrideSlide || state.slides[index];
  const isReteach = !!slide.reteach;

  // Reset quiz/understanding state
  state.understandingChoice = null;
  state.quizQuestions = [];
  state.quizAnswers   = [];
  state.quizCurrentQ  = 0;
  state.quizDone      = false;

  // Update HUD
  const total = state.slides.length;
  document.getElementById('slide-counter').textContent = `${index + 1} / ${total}`;
  document.getElementById('progress-bar').style.width = `${((index) / total) * 100}%`;

  // Adaptive banner
  const banner = document.getElementById('adaptive-banner');
  if (isReteach) {
    banner.style.display = 'flex';
    banner.className = 'adaptive-banner reteach';
    banner.innerHTML = '<span>🔁</span> Simplified re-explanation — easier language and more examples.';
  } else if (state.adaptiveProfile) {
    const hint = state.adaptiveProfile.difficulty_hint;
    if (hint === 'simplify' && index > 0) {
      banner.style.display = 'flex';
      banner.className = 'adaptive-banner simplify';
      banner.innerHTML = '<span>📉</span> Adjusted for your pace — more examples and clearer language ahead.';
    } else if (hint === 'advanced' && index > 0) {
      banner.style.display = 'flex';
      banner.className = 'adaptive-banner advanced';
      banner.innerHTML = '<span>🚀</span> You\'re doing great — content has been elevated to match your performance.';
    } else {
      banner.style.display = 'none';
    }
  } else {
    banner.style.display = 'none';
  }

  // Build slide card
  const card = document.getElementById('slide-card');

  // Helper: render one rich point (object with text/inline_latex/sub_steps)
  function renderPoint(p, idx, isReteach) {
    const isObj = (typeof p === 'object' && p !== null && !Array.isArray(p));
    const text = isObj ? (p.text || '') : String(p);
    const sourceTitle = isObj ? String(p.source_title || '').trim() : '';
    const sourceUrl = isObj ? String(p.source_url || '').trim() : '';
    const latex = isObj ? (p.inline_latex || '') : '';
    const label = isObj ? (p.inline_label || 'Formula') : '';
    const steps = isObj ? (p.sub_steps || []) : [];

    let mathHtml = '';
    if (latex) {
      mathHtml = `
        <div class="math-block">
          <div class="math-label">📐 ${escHtml(label)}</div>
          <div class="math-formula">$$${latex}$$</div>
        </div>`;
    }

    let stepsHtml = '';
    if (steps.length) {
      const stepsInner = steps.map((s, si) => {
        // Try to split "Step N — Name: detail" nicely
        const stepText = String(s);
        return `<div class="sub-step">
          <span class="sub-step-num">${si + 1}</span>${escHtml(stepText)}
        </div>`;
      }).join('');
      stepsHtml = `<div class="sub-steps">${stepsInner}</div>`;
    }

    const sourceHtml = sourceTitle
      ? `<div class="point-source">Source: ${
          sourceUrl
            ? `<a href="${escHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escHtml(sourceTitle)}</a>`
            : escHtml(sourceTitle)
        }</div>`
      : '';

    return `
      <li class="slide-point ${isReteach ? 'reteach-point' : ''}"
          style="animation-delay:${idx * 0.07}s; flex-direction:column; gap:.5rem;">
        <div style="display:flex;gap:.85rem;align-items:flex-start;">
          <span class="point-bullet" style="margin-top:.55rem;flex-shrink:0;"></span>
          <div class="point-main-text">
            <span>${escHtml(text)}</span>
            ${sourceHtml}
          </div>
        </div>
        ${mathHtml}
        ${stepsHtml}
      </li>`;
  }

  // Helper: render worked example
  function renderWorkedExample(we) {
    if (!we || !we.problem) return '';
    const stepRows = (we.steps || []).map((s, i) => {
      const txt = String(s);
      // Split on first colon after the label for nicer display
      return `<div class="worked-step">
        <span class="worked-step-num">${i + 1}</span>
        <span>${escHtml(txt)}</span>
      </div>`;
    }).join('');
    return `
      <div class="worked-example">
        <div class="worked-example-header">✏️ Worked Example</div>
        <div class="worked-problem">${escHtml(we.problem)}</div>
        <div class="worked-steps">${stepRows}</div>
        ${we.answer ? `<div class="worked-answer">
          <span class="worked-answer-label">Answer</span>
          <span>${escHtml(we.answer)}</span>
        </div>` : ''}
      </div>`;
  }

  const pointsHtml = slide.points.map((p, i) => renderPoint(p, i, isReteach)).join('');
  const workedHtml = slide.worked_example ? renderWorkedExample(slide.worked_example) : '';

  card.innerHTML = `
    <div class="slide-number-tag ${isReteach ? 'reteach-tag' : ''}">
      ${isReteach ? '🔁 Re-Teaching — ' : ''}Slide ${index + 1}
    </div>
    <div class="slide-title">${escHtml(slide.title)}</div>
    <ul class="slide-points">${pointsHtml}</ul>
    ${workedHtml}
    <div style="margin-top:1.25rem;">
      <button class="visual-toggle-btn" id="visual-toggle-btn" onclick="toggleVisual()">
        <span class="vt-icon">📊</span> Show Visual Diagram
      </button>
    </div>
  `;

  // Re-render KaTeX after injecting HTML (auto-render won't catch dynamic content)
  if (window.renderMathInElement) {
    renderMathInElement(card, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$',  right: '$',  display: false}
      ],
      throwOnError: false
    });
  }

  // Reset visual panel
  const vp = document.getElementById('visual-panel');
  vp.style.display = 'none';
  vp.innerHTML = '';
  state.visualLoaded = false;
  state.visualOpen = false;

  // Reset understanding section
  const us = document.getElementById('understanding-section');
  us.style.display = 'block';
  us.querySelectorAll('.u-btn').forEach(b => b.classList.remove('selected'));
  us.querySelectorAll('.u-btn').forEach(b => b.disabled = false);

  // Hide quiz
  const qs = document.getElementById('quiz-section');
  qs.style.display = 'none';
  qs.innerHTML = '';

  // Show chat section (reset messages for new slide)
  const chatSection = document.getElementById('chat-section');
  chatSection.style.display = 'block';
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('chat-input').value = '';

  // Show HUD inline chat bar
  const hudChatBar = document.getElementById('hud-chat-bar');
  if (hudChatBar) { hudChatBar.classList.add('visible'); }
  const hudChatInput = document.getElementById('hud-chat-input');
  if (hudChatInput) hudChatInput.value = '';
  // Clear the inline answer from the previous slide
  const hudChatAnswer = document.getElementById('hud-chat-answer');
  if (hudChatAnswer) { hudChatAnswer.classList.remove('visible', 'loading-state'); }
  const hudChatAnswerText = document.getElementById('hud-chat-answer-text');
  if (hudChatAnswerText) hudChatAnswerText.textContent = '';

  // Show notes section and load saved note for this slide
  const notesSection = document.getElementById('notes-section');
  notesSection.style.display = 'block';
  const savedNote = state.notesBySlide[index] || '';
  document.getElementById('notes-textarea').value = savedNote;
  // Collapse notes body by default on new slide
  document.getElementById('notes-body').style.display = 'none';
  document.getElementById('notes-toggle-icon').textContent = '▼';
  state.notesOpen = false;

  // Reset sources panel for new slide
  resetSources();

  // Scroll to top
  document.getElementById('learning-body').scrollTop = 0;

  // Update nav arrow states
  updateNavButtons();

  // Start timer
  startTimer();
}

// =====================================================================
// TIMER
// =====================================================================
function startTimer() {
  clearInterval(state.timerInterval);
  state.timerSeconds = 0;
  state.slideStartTime = Date.now();

  const badge = document.getElementById('timer-badge');
  badge.className = 'timer-badge';

  state.timerInterval = setInterval(() => {
    state.timerSeconds++;
    const s = state.timerSeconds;
    const display = s < 60 ? `${s}s` : `${Math.floor(s/60)}m ${s%60}s`;
    badge.innerHTML = `⏱ ${display}`;
    if (s > 90)  badge.className = 'timer-badge warning';
    if (s > 180) badge.className = 'timer-badge long';
  }, 1000);
}

function stopTimer() {
  clearInterval(state.timerInterval);
  return Math.round((Date.now() - state.slideStartTime) / 1000);
}

// =====================================================================
// UNDERSTANDING BUTTONS
// =====================================================================
document.querySelectorAll('.u-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (state.understandingChoice) return; // already chosen
    const val = btn.dataset.val;
    state.understandingChoice = val;

    // Visual feedback
    document.querySelectorAll('.u-btn').forEach(b => b.disabled = true);
    btn.classList.add('selected');

    // Fetch quiz
    await loadQuiz();
  });
});

// =====================================================================
// QUIZ
// =====================================================================
async function loadQuiz() {
  const qs = document.getElementById('quiz-section');
  qs.style.display = 'block';
  qs.innerHTML = `
    <div class="quiz-header">
      <div class="quiz-label">Quick Quiz</div>
      <div class="quiz-progress-dots" id="q-dots"></div>
    </div>
    <div style="text-align:center; padding:1rem; color:var(--text-3); font-size:.85rem;">
      <div class="loading-ring" style="width:32px;height:32px;margin:0 auto .5rem;"></div>
      Generating adaptive quiz…
    </div>
  `;
  qs.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const res = await fetch(apiPath('/quiz'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, slide_index: state.currentIndex }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Failed to generate quiz.');

    state.quizQuestions = data.questions;
    state.quizAnswers   = new Array(data.questions.length).fill(null);
    state.quizCurrentQ  = 0;
    renderQuestion(0);

  } catch (err) {
    qs.innerHTML = `<div style="color:var(--text-3);font-size:.85rem;padding:.5rem;">${err.message}</div>`;
    showNavigationButtons(0);
  }
}

function renderQuestion(qIndex) {
  const qs = document.getElementById('quiz-section');
  const questions = state.quizQuestions;
  const q = questions[qIndex];

  const dotsHtml = questions.map((_, i) => {
    let cls = '';
    if (i < qIndex) {
      cls = state.quizAnswers[i] === questions[i].correct ? 'correct' : 'wrong';
    } else if (i === qIndex) {
      cls = 'active';
    }
    return `<div class="q-dot ${cls}"></div>`;
  }).join('');

  qs.innerHTML = `
    <div class="quiz-header">
      <div class="quiz-label">Quick Quiz</div>
      <div class="quiz-progress-dots">${dotsHtml}</div>
    </div>
    <div class="quiz-question-wrap active" id="q-wrap">
      <div class="quiz-q-index">Question ${qIndex + 1} of ${questions.length}</div>
      <div class="quiz-q-text">${escHtml(q.question)}</div>
      <div class="quiz-options" id="q-options">
        ${q.options.map((opt, i) => `
          <button class="quiz-option" data-idx="${i}">${escHtml(opt)}</button>
        `).join('')}
      </div>
    </div>
  `;

  // Option click
  qs.querySelectorAll('.quiz-option').forEach(btn => {
    btn.addEventListener('click', () => handleAnswer(qIndex, parseInt(btn.dataset.idx)));
  });
}

function handleAnswer(qIndex, chosen) {
  const q = state.quizQuestions[qIndex];
  state.quizAnswers[qIndex] = chosen;

  // Disable all options and highlight
  const opts = document.querySelectorAll('.quiz-option');
  opts.forEach(b => { b.disabled = true; });
  opts[chosen].classList.add(chosen === q.correct ? 'correct' : 'wrong');
  if (chosen !== q.correct) opts[q.correct].classList.add('correct');

  // Explanation
  const optWrap = document.getElementById('q-options');
  const expDiv = document.createElement('div');
  expDiv.className = 'quiz-explanation';
  expDiv.innerHTML = `<strong>${chosen === q.correct ? '✅ Correct!' : '❌ Incorrect.'}</strong> ${escHtml(q.explanation)}`;
  optWrap.after(expDiv);

  // Next button
  const nav = document.createElement('div');
  nav.className = 'quiz-nav';

  const isLast = qIndex === state.quizQuestions.length - 1;
  const btn = document.createElement('button');
  btn.className = 'btn-quiz-next';
  btn.textContent = isLast ? 'See Results →' : 'Next Question →';
  btn.addEventListener('click', () => {
    if (isLast) {
      showQuizResult();
    } else {
      renderQuestion(qIndex + 1);
    }
  });
  nav.appendChild(btn);
  document.getElementById('q-wrap').appendChild(nav);

  document.getElementById('quiz-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showQuizResult() {
  const questions = state.quizQuestions;
  const correct = state.quizAnswers.filter((a, i) => a === questions[i].correct).length;
  const score = correct / questions.length;
  const pct = Math.round(score * 100);

  const colorClass = score >= 0.8 ? 'good' : score >= 0.5 ? 'ok' : 'bad';
  const understood = state.understandingChoice === 'yes' ||
                     (state.understandingChoice === 'somewhat' && score >= 0.5);

  // Send feedback to server
  sendFeedback(score, understood);

  const qs = document.getElementById('quiz-section');

  const dots = questions.map((q, i) => {
    const cls = state.quizAnswers[i] === q.correct ? 'correct' : 'wrong';
    return `<div class="q-dot ${cls}"></div>`;
  }).join('');

  let actionsHtml = '';
  const hasNext = state.currentIndex < state.slides.length - 1;

  if (score < 0.5 || state.understandingChoice === 'no') {
    actionsHtml += `<button class="btn-reteach" onclick="triggerReteach()">🔁 Simplify & Re-teach</button>`;
  }
  if (hasNext) {
    actionsHtml += `<button class="btn-next-slide" onclick="nextSlide()">Next Slide →</button>`;
  } else {
    actionsHtml += `<button class="btn-next-slide" onclick="showStats()">📊 View My Results</button>`;
  }

  qs.innerHTML = `
    <div class="quiz-header">
      <div class="quiz-label">Quiz Results</div>
      <div class="quiz-progress-dots">${dots}</div>
    </div>
    <div class="quiz-result">
      <div class="quiz-score-display ${colorClass}">${pct}%</div>
      <div class="quiz-score-label">${correct} / ${questions.length} correct</div>
      <div class="quiz-result-actions">${actionsHtml}</div>
    </div>
  `;

  qs.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// =====================================================================
// FEEDBACK
// =====================================================================
async function sendFeedback(quizScore, understood) {
  const timeSpent = stopTimer();
  const slide = state.slides[state.currentIndex];

  const record = {
    session_id: state.sessionId,
    slide_index: state.currentIndex,
    slide_title: slide.title,
    time_spent: timeSpent,
    understood,
    quiz_score: quizScore,
    quiz_answers: state.quizAnswers,
  };

  try {
    const res = await fetch(apiPath('/feedback'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(record),
    });
    const data = await res.json();
    if (res.ok) {
      state.adaptiveProfile = data.profile;
      state.performance.push(record);
      // Update XP
      if (data.xp) {
        state.xp = data.xp.total_xp;
        state.streakSlides = data.xp.streak_slides;
        updateXPDisplay(data.xp);
      }
    }
  } catch (_) {}
}

// =====================================================================
// XP DISPLAY
// =====================================================================
function updateXPDisplay(xpData) {
  const xpEl = document.getElementById('xp-display');
  const streakEl = document.getElementById('streak-display');
  if (xpEl) xpEl.textContent = `⚡ ${xpData.total_xp} XP`;
  if (streakEl) streakEl.textContent = xpData.streak_slides >= 2 ? `🔥 ${xpData.streak_slides}` : '';

  // Pop toast
  if (xpData.xp_earned > 0) {
    const t = document.getElementById('xp-toast');
    t.textContent = `+${xpData.xp_earned} XP${xpData.streak_slides >= 5 && xpData.streak_slides % 5 === 0 ? ' 🔥 Streak Bonus!' : ''}`;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2500);
  }
}

// =====================================================================
// AI TUTOR CHAT
// =====================================================================
function closeChatSection() {
  document.getElementById('chat-section').style.display = 'none';
}

// Core chat send function — works for both HUD input and the slide-body chat
async function _doSendChat(question, inputEl, sendBtnEl) {
  if (!question) return;
  sendBtnEl.disabled = true;
  inputEl.value = '';

  // Ensure the slide-body chat section is visible to show replies
  const chatSection = document.getElementById('chat-section');
  chatSection.style.display = 'block';

  const msgs = document.getElementById('chat-messages');

  // Add user bubble
  msgs.innerHTML += `<div class="chat-bubble user">${escHtml(question)}</div>`;

  // Add loading bubble
  const loadId = 'chat-loading-' + Date.now();
  msgs.innerHTML += `<div class="chat-bubble loading" id="${loadId}">Tutor is thinking…</div>`;
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch(apiPath('/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        slide_index: state.currentIndex,
        question,
      }),
    });
    const data = await res.json();
    const loadEl = document.getElementById(loadId);
    if (loadEl) loadEl.remove();

    if (!res.ok) throw new Error(data.error || 'Tutor unavailable.');

    msgs.innerHTML += `<div class="chat-bubble tutor">${escHtml(data.answer)}</div>`;
  } catch (err) {
    const loadEl = document.getElementById(loadId);
    if (loadEl) loadEl.remove();
    msgs.innerHTML += `<div class="chat-bubble tutor" style="color:var(--red);">⚠️ ${escHtml(err.message)}</div>`;
  }

  msgs.scrollTop = msgs.scrollHeight;
  sendBtnEl.disabled = false;
  inputEl.focus();

  // Scroll replies into view within the learning-body scroll container
  const learningBody = document.getElementById('learning-body');
  if (learningBody) {
    setTimeout(() => {
      learningBody.scrollTo({ top: learningBody.scrollHeight, behavior: 'smooth' });
    }, 50);
  }
}

async function sendHudChatMessage() {
  const input      = document.getElementById('hud-chat-input');
  const btn        = document.getElementById('hud-chat-send-btn');
  const answerBox  = document.getElementById('hud-chat-answer');
  const answerText = document.getElementById('hud-chat-answer-text');
  const question   = input.value.trim();
  if (!question) return;

  // Show loading state inline in HUD
  answerBox.classList.remove('loading-state');
  answerBox.classList.add('visible', 'loading-state');
  answerText.textContent = '🤖 Tutor is thinking…';

  btn.disabled = true;
  input.value  = '';

  try {
    const res  = await fetch(apiPath('/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:  state.sessionId,
        slide_index: state.currentIndex,
        question,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Tutor unavailable.');
    answerBox.classList.remove('loading-state');
    answerText.textContent = data.answer;

    // Also append to the full chat-section below for history
    const chatSection = document.getElementById('chat-section');
    chatSection.style.display = 'block';
    const msgs = document.getElementById('chat-messages');
    msgs.innerHTML += `<div class="chat-bubble user">${escHtml(question)}</div>`;
    msgs.innerHTML += `<div class="chat-bubble tutor">${escHtml(data.answer)}</div>`;
    msgs.scrollTop = msgs.scrollHeight;
  } catch (err) {
    answerBox.classList.remove('loading-state');
    answerText.textContent = '⚠️ ' + err.message;
  }

  btn.disabled = false;
  input.focus();
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const btn   = document.getElementById('chat-send-btn');
  await _doSendChat(input.value.trim(), input, btn);
}

// Allow Enter to send from either input (Shift+Enter for newline)
document.addEventListener('keydown', e => {
  if (e.key !== 'Enter' || e.shiftKey) return;
  const active = document.activeElement;
  if (active && active.id === 'hud-chat-input') { e.preventDefault(); sendHudChatMessage(); }
  if (active && active.id === 'chat-input')     { e.preventDefault(); sendChatMessage(); }
});

// =====================================================================
// NOTES
// =====================================================================
function toggleNotes() {
  const body = document.getElementById('notes-body');
  const icon = document.getElementById('notes-toggle-icon');
  state.notesOpen = !state.notesOpen;
  body.style.display = state.notesOpen ? 'block' : 'none';
  icon.textContent = state.notesOpen ? '▲' : '▼';
}

async function saveNote() {
  const text = document.getElementById('notes-textarea').value;
  state.notesBySlide[state.currentIndex] = text;

  const savedMsg = document.getElementById('notes-saved-msg');
  savedMsg.classList.add('show');
  setTimeout(() => savedMsg.classList.remove('show'), 2000);

  try {
    await fetch(apiPath('/note'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        slide_index: state.currentIndex,
        text,
      }),
    });
  } catch (_) {}
}

// =====================================================================
// SESSION SUMMARY
// =====================================================================
async function generateSummary() {
  const btn = document.getElementById('btn-gen-summary');
  const body = document.getElementById('summary-body');
  btn.disabled = true;
  btn.textContent = 'Generating…';
  body.style.display = 'block';
  body.textContent = 'Generating your personalised summary…';

  try {
    const res = await fetch(apiPath('/summary'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to generate summary.');
    body.textContent = data.summary;
    btn.textContent = 'Regenerate';
    btn.disabled = false;
  } catch (err) {
    body.textContent = '⚠️ ' + err.message;
    btn.textContent = 'Try Again';
    btn.disabled = false;
  }
}

// =====================================================================
// NAVIGATION
// =====================================================================
function nextSlide() {
  state.currentIndex++;
  loadSlide(state.currentIndex);
  document.getElementById('learning-body').scrollTop = 0;
}

async function triggerReteach() {
  const qs = document.getElementById('quiz-section');
  qs.innerHTML = `
    <div style="text-align:center;padding:1.5rem;color:var(--text-3);">
      <div class="loading-ring" style="width:32px;height:32px;margin:0 auto .75rem;"></div>
      Generating simplified explanation…
    </div>
  `;
  qs.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const res = await fetch(apiPath('/reteach'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, slide_index: state.currentIndex }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Reteach failed.');

    qs.innerHTML = '';
    loadSlide(state.currentIndex, data.slide);

  } catch (err) {
    showToast(err.message);
    qs.innerHTML = '';
    // Show next slide button anyway
    const hasNext = state.currentIndex < state.slides.length - 1;
    qs.innerHTML = `
      <div style="text-align:center;padding:1rem;">
        ${hasNext
          ? `<button class="btn-next-slide" onclick="nextSlide()">Next Slide →</button>`
          : `<button class="btn-next-slide" onclick="showStats()">📊 View My Results</button>`
        }
      </div>
    `;
  }
}

// =====================================================================
// STATS SCREEN
// =====================================================================
async function showStats() {
  stopTimer();

  setLoading('Analysing your session…', 'Crunching the numbers');
  showScreen('screen-loading');

  try {
    const res = await fetch(apiPath(`/stats/${state.sessionId}`));
    const data = await res.json();
    renderStats(data);
    showScreen('screen-stats');
  } catch (_) {
    // fallback with local data
    renderStatsLocal();
    showScreen('screen-stats');
  }
}

function renderStats(data) {
  const profile  = data.profile;
  const perf     = data.performance;

  document.getElementById('stats-topic-label').textContent = `Topic: ${data.topic}`;

  // Stat cards
  const avgQuiz  = profile.avg_quiz_score !== null ? Math.round(profile.avg_quiz_score * 100) + '%' : '—';
  const understood = profile.understood_rate !== null ? Math.round(profile.understood_rate * 100) + '%' : '—';
  const avgTime  = profile.avg_time_seconds !== null ? fmtTime(profile.avg_time_seconds) : '—';
  const completed = perf.length + ' / ' + data.total_slides;

  const quizColor = profile.avg_quiz_score >= .8 ? 'green' : profile.avg_quiz_score >= .5 ? 'gold' : 'red';
  const undColor  = profile.understood_rate >= .8 ? 'green' : profile.understood_rate >= .5 ? 'gold' : 'red';

  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-value gold">${completed}</div>
      <div class="stat-key">Slides Completed</div>
    </div>
    <div class="stat-card">
      <div class="stat-value ${quizColor}">${avgQuiz}</div>
      <div class="stat-key">Avg Quiz Score</div>
    </div>
    <div class="stat-card">
      <div class="stat-value ${undColor}">${understood}</div>
      <div class="stat-key">Slides Understood</div>
    </div>
    <div class="stat-card">
      <div class="stat-value teal">${avgTime}</div>
      <div class="stat-key">Avg Time / Slide</div>
    </div>
  `;

  // XP & Streak cards
  const xpGrid = document.getElementById('stats-xp-grid');
  if (profile.xp !== undefined) {
    xpGrid.style.display = 'grid';
    xpGrid.innerHTML = `
      <div class="stat-card">
        <div class="stat-value gold">⚡ ${profile.xp}</div>
        <div class="stat-key">Total XP Earned</div>
      </div>
      <div class="stat-card">
        <div class="stat-value teal">🔥 ${profile.streak_slides}</div>
        <div class="stat-key">Best Streak</div>
      </div>
    `;
  }

  // Adaptive insights
  const note = document.getElementById('adaptive-note');
  const noteBody = document.getElementById('adaptive-note-body');
  if (profile.summary && profile.summary !== 'new learner – no data yet') {
    note.style.display = 'block';
    let html = `<p><strong>AI Diagnosis:</strong> ${escHtml(profile.summary)}.</p>`;
    if (profile.weak_slides && profile.weak_slides.length > 0) {
      html += `<p style="margin-top:.5rem;">Topics to review: `;
      html += profile.weak_slides.map(t => `<span class="weak-tag">${escHtml(t)}</span>`).join(' ');
      html += '</p>';
    }
    if (profile.difficulty_hint === 'simplify') {
      html += `<p style="margin-top:.5rem;color:var(--text-3);">💡 Next time, try breaking this topic into smaller sub-topics for better retention.</p>`;
    } else if (profile.difficulty_hint === 'advanced') {
      html += `<p style="margin-top:.5rem;color:var(--text-3);">🚀 You're ready for advanced material — try a more specific sub-topic next.</p>`;
    }
    noteBody.innerHTML = html;
  } else {
    note.style.display = 'none';
  }

  // Needs review (spaced repetition)
  const reviewNote = document.getElementById('needs-review-note');
  const reviewBody = document.getElementById('needs-review-body');
  if (profile.needs_review && profile.needs_review.length > 0) {
    reviewNote.style.display = 'block';
    reviewBody.innerHTML = `<p style="margin-bottom:.5rem;color:var(--text-3);font-size:.82rem;">These slides scored below 60% — revisit them before your next session:</p>` +
      profile.needs_review.map(r => `<span class="review-tag">${escHtml(r.slide_title)}</span>`).join(' ');
  } else {
    reviewNote.style.display = 'none';
  }

  // Reset summary panel
  const summaryBody = document.getElementById('summary-body');
  summaryBody.style.display = 'none';
  summaryBody.textContent = '';
  const summaryBtn = document.getElementById('btn-gen-summary');
  summaryBtn.disabled = false;
  summaryBtn.textContent = 'Generate Summary';

  // Performance table
  if (perf.length > 0) {
    const tableWrap = document.getElementById('perf-table-wrap');
    tableWrap.style.display = 'block';

    tableWrap.innerHTML = `
      <div class="perf-table-title">Slide-by-Slide Breakdown</div>
      <div style="overflow-x:auto;">
        <table id="perf-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Slide</th>
              <th>Time</th>
              <th>Quiz</th>
              <th>Understood</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            ${perf.map((p, i) => {
              const qPct = Math.round(p.quiz_score * 100);
              const qClass = p.quiz_score >= .8 ? 'good' : p.quiz_score >= .5 ? 'ok' : 'bad';
              const uClass = p.understood ? 'good' : 'bad';
              const barColor = p.quiz_score >= .8 ? 'var(--green)' : p.quiz_score >= .5 ? 'var(--gold)' : 'var(--red)';
              return `
                <tr>
                  <td style="color:var(--text-3);font-family:var(--mono);">${i+1}</td>
                  <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(p.slide_title)}</td>
                  <td style="font-family:var(--mono);">${fmtTime(p.time_spent)}</td>
                  <td>
                    <div class="mini-bar">
                      <div class="mini-bar-fill" style="width:${qPct}%;background:${barColor};"></div>
                    </div>
                  </td>
                  <td><span class="badge ${uClass}">${p.understood ? 'Yes' : 'No'}</span></td>
                  <td><span class="badge ${qClass}">${qPct}%</span></td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      </div>
    `;
  }
}

function renderStatsLocal() {
  document.getElementById('stats-topic-label').textContent = `Topic: ${state.topic}`;
  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-value gold">${state.performance.length}</div>
      <div class="stat-key">Slides Reviewed</div>
    </div>
  `;
}

function restartSession() {
  // Reset all state
  state.sessionId = null; state.slides = []; state.currentIndex = 0;
  state.performance = []; state.adaptiveProfile = null;
  state.xp = 0; state.streakSlides = 0;
  state.chatHistory = []; state.notesBySlide = {};
  topicInput.value = '';
  charCount.textContent = '0';
  showScreen('screen-start');
}

// =====================================================================
// VISUAL DIAGRAM ENGINE
// =====================================================================

async function toggleVisual() {
  const btn = document.getElementById('visual-toggle-btn');
  const vp  = document.getElementById('visual-panel');

  if (state.visualOpen) {
    // Collapse
    vp.style.display = 'none';
    state.visualOpen = false;
    if (btn) { btn.classList.remove('active'); btn.innerHTML = '<span class="vt-icon">📊</span> Show Visual Diagram'; }
    return;
  }

  // Expand
  state.visualOpen = true;
  vp.style.display = 'block';
  if (btn) { btn.classList.add('active'); btn.innerHTML = '<span class="vt-icon">📊</span> Hide Visual Diagram'; }

  if (state.visualLoaded) return; // already fetched

  // Show loading state
  vp.innerHTML = `
    <div class="visual-panel">
      <div class="visual-panel-header">
        <div class="visual-panel-label">Visual Diagram</div>
      </div>
      <div class="visual-panel-body">
        <div class="visual-loading">
          <div class="loading-ring" style="width:32px;height:32px;"></div>
          <span>Generating visual…</span>
        </div>
      </div>
    </div>
  `;
  vp.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const res = await fetch(apiPath('/visual'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, slide_index: state.currentIndex }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Visual generation failed.');

    state.visualLoaded = true;
    renderVisualPanel(vp, data.visual);

  } catch (err) {
    vp.innerHTML = `
      <div class="visual-panel">
        <div class="visual-panel-header">
          <div class="visual-panel-label">Visual Diagram</div>
        </div>
        <div class="visual-panel-body">
          <div class="visual-error">
            ⚠️ ${escHtml(err.message)}
            <br>
            <button onclick="state.visualLoaded=false; toggleVisual(); toggleVisual();">Retry</button>
          </div>
        </div>
      </div>
    `;
  }
}

function renderVisualPanel(container, visual) {
  const typeLabel = visual.type.charAt(0).toUpperCase() + visual.type.slice(1);
  const svgHtml   = buildDiagramSVG(visual);
  const wrapClass = `viz-scroll-wrap${visual.type === 'mindmap' ? ' mindmap-wrap' : visual.type === 'flowchart' ? ' flowchart-wrap' : ''}`;
  const diagramTitle = visual.title
    ? `<div class="visual-diagram-title">${escHtml(visual.title)}</div>`
    : '';

  container.innerHTML = `
    <div class="visual-panel">
      <div class="visual-panel-header">
        <div class="visual-panel-label">📊 Visual Diagram</div>
        <span class="visual-type-badge">${typeLabel}</span>
      </div>
      ${diagramTitle}
      <div class="visual-panel-body">
        <div class="${wrapClass}">
          ${svgHtml}
        </div>
      </div>
    </div>
  `;
}

// ---- Diagram dispatcher ----
function buildDiagramSVG(visual) {
  const { type, data } = visual;
  switch (type) {
    case 'flowchart':   return buildFlowchart(data);
    case 'cycle':       return buildCycle(data);
    case 'comparison':  return buildComparison(data);
    case 'timeline':    return buildTimeline(data);
    case 'pyramid':     return buildPyramid(data);
    case 'mindmap':     return buildMindmap(data);
    default:            return `<p style="color:var(--text-3)">Unknown diagram type: ${escHtml(type)}</p>`;
  }
}

/* ============================================================
   SVG TEXT WRAP HELPER
   Splits text into lines of maxChars, returns <tspan> elements
   ============================================================ */
function svgTextLines(text, x, startY, lineHeight, maxChars, attrs) {
  const words = String(text).split(' ');
  const lines = [];
  let current = '';
  words.forEach(w => {
    if ((current + ' ' + w).trim().length <= maxChars) {
      current = (current + ' ' + w).trim();
    } else {
      if (current) lines.push(current);
      current = w.length > maxChars ? w.substring(0, maxChars) : w;
    }
  });
  if (current) lines.push(current);
  return lines.map((line, i) =>
    `<tspan x="${x}" dy="${i === 0 ? 0 : lineHeight}" ${attrs || ''}>${escHtml(line)}</tspan>`
  ).join('');
}

/* ============================================================
   FLOWCHART  — collision-free, dynamic node heights
   Each node is as tall as its content needs. GAP_Y is fixed
   padding between nodes, never compressed.
   ============================================================ */
function buildFlowchart(data) {
  const nodes = (data.nodes || []).slice(0, 9);
  const edges = data.edges || [];

  const W       = 780;
  const NODE_W  = 330;
  const GAP_Y   = 60;    // fixed gap between nodes
  const MIN_H   = 68;    // minimum node height
  const LINE_H  = 20;    // px per label line
  const SUB_H   = 15;    // px per sub line
  const PAD_V   = 18;    // vertical internal padding
  const BADGE_W = 28;
  const CX      = W / 2;
  const PAD_TOP = 30;

  // ── text helper ──────────────────────────────────────────
  function wrapText(text, maxCh) {
    const words = String(text).split(' ');
    const lines = []; let cur = '';
    words.forEach(w => {
      const test = cur ? cur + ' ' + w : w;
      if (test.length <= maxCh) { cur = test; }
      else { if (cur) lines.push(cur); cur = w.length > maxCh ? w.substring(0, maxCh) : w; }
    });
    if (cur) lines.push(cur);
    return lines;
  }

  // Measure each node's needed height before layout
  const measured = nodes.map(n => {
    const isStartEnd = n.type === 'start' || n.type === 'end';
    const maxCh = isStartEnd ? 36 : 28;
    const lblLines = wrapText(n.label || '', maxCh).slice(0, 4);
    const subLines = n.sub ? wrapText(n.sub, isStartEnd ? 38 : 34).slice(0, 2) : [];
    const contentH = lblLines.length * LINE_H + (subLines.length ? subLines.length * SUB_H + 4 : 0);
    const nodeH = Math.max(MIN_H, contentH + PAD_V * 2);
    return { ...n, lblLines, subLines, nodeH };
  });

  // Topological sort
  const adjMap = {};
  nodes.forEach(n => { adjMap[n.id] = []; });
  edges.forEach(e => { if (adjMap[e.from]) adjMap[e.from].push(e.to); });
  const visited = new Set(), order = [];
  function topo(id) {
    if (visited.has(id)) return; visited.add(id);
    (adjMap[id] || []).forEach(topo); order.unshift(id);
  }
  nodes.forEach(n => topo(n.id));

  // Assign y positions using actual node heights + fixed gap
  const posY = {}, nodeById = {};
  measured.forEach(n => { nodeById[n.id] = n; });
  let curY = PAD_TOP;
  order.forEach(id => {
    posY[id] = curY;
    const m = nodeById[id];
    curY += (m ? m.nodeH : MIN_H) + GAP_Y;
  });
  // Fallback for nodes not in topo order
  measured.forEach(n => {
    if (posY[n.id] === undefined) { posY[n.id] = curY; curY += n.nodeH + GAP_Y; }
  });

  const H = curY + 20;

  const defs = `<defs>
    <marker id="fc-arrow" markerWidth="12" markerHeight="12" refX="9" refY="5" orient="auto">
      <path d="M0,1 L0,9 L11,5 z" fill="#3a4560"/>
    </marker>
    <filter id="fc-glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;

  // Edges
  let edgeSvg = '';
  edges.forEach(e => {
    const fn = nodeById[e.from], tn = nodeById[e.to];
    if (!fn || !tn) return;
    const y1 = (posY[e.from] ?? 0) + fn.nodeH;
    const y2 = (posY[e.to]   ?? 0);
    const my = (y1 + y2) / 2;
    const path = `M${CX},${y1} C${CX},${my+8} ${CX},${my-8} ${CX},${y2-8}`;
    edgeSvg += `<path d="${path}" fill="none" stroke="rgba(232,160,32,.15)" stroke-width="7"/>`;
    edgeSvg += `<path d="${path}" fill="none" stroke="#3a4560" stroke-width="2.5" marker-end="url(#fc-arrow)"/>`;
    if (e.label) {
      edgeSvg += `<rect x="${CX+6}" y="${my-11}" width="${e.label.length*7+12}" height="20" rx="5"
        style="fill:#0f1420;stroke:#2a3347;stroke-width:1;"/>`;
      edgeSvg += `<text x="${CX+12}" y="${my+1}"
        style="fill:#6b7a99;font-size:11px;font-weight:600;font-family:Syne,sans-serif;dominant-baseline:middle;">${escHtml(e.label)}</text>`;
    }
  });

  // Nodes
  let nodeSvg = '';
  measured.forEach((n, stepNum) => {
    const y  = posY[n.id] ?? (PAD_TOP + stepNum * (MIN_H + GAP_Y));
    const x  = CX - NODE_W / 2;
    const NH = n.nodeH;
    const type = n.type || 'step';
    const isStart    = type === 'start';
    const isEnd      = type === 'end';
    const isDecision = type === 'decision';

    let borderColor, bgColor, textColor, glowColor, accentColor;
    if (isStart) {
      borderColor='#20c5b0'; bgColor='rgba(32,197,176,.12)'; textColor='#20c5b0';
      glowColor='rgba(32,197,176,.18)'; accentColor='#20c5b0';
    } else if (isEnd) {
      borderColor='#3ec87a'; bgColor='rgba(62,200,122,.12)'; textColor='#3ec87a';
      glowColor='rgba(62,200,122,.18)'; accentColor='#3ec87a';
    } else if (isDecision) {
      borderColor='#e8a020'; bgColor='rgba(232,160,32,.09)'; textColor='#e8a020';
      glowColor='rgba(232,160,32,.16)'; accentColor='#e8a020';
    } else {
      borderColor='#2a3860'; bgColor='#141926'; textColor='#dde2f0';
      glowColor='rgba(100,130,255,.07)'; accentColor='#5a7aff';
    }

    const rx = (isStart || isEnd) ? NH/2 : isDecision ? 14 : 12;

    // Glow halo
    nodeSvg += `<rect x="${x-4}" y="${y-4}" width="${NODE_W+8}" height="${NH+8}" rx="${rx+3}"
      style="fill:${glowColor};"/>`;
    // Main box
    nodeSvg += `<rect x="${x}" y="${y}" width="${NODE_W}" height="${NH}" rx="${rx}"
      style="fill:${bgColor};stroke:${borderColor};stroke-width:2;"/>`;

    // Left accent stripe (step nodes only)
    if (!isStart && !isEnd) {
      nodeSvg += `<rect x="${x}" y="${y+10}" width="4" height="${NH-20}" rx="2"
        style="fill:${accentColor};opacity:.7;"/>`;
    }

    // Step badge
    const hasBadge = !isStart && !isEnd;
    if (hasBadge) {
      nodeSvg += `<rect x="${x+12}" y="${y+NH/2-14}" width="${BADGE_W}" height="${BADGE_W}" rx="7"
        style="fill:rgba(255,255,255,.06);stroke:${borderColor};stroke-width:1.2;"/>`;
      nodeSvg += `<text x="${x+12+BADGE_W/2}" y="${y+NH/2+1}" text-anchor="middle" dominant-baseline="middle"
        style="fill:${accentColor};font-size:11px;font-weight:800;font-family:'JetBrains Mono',monospace;">${stepNum}</text>`;
    }

    // Text positioning
    const textX   = (isStart || isEnd) ? CX : x + 54;
    const anchor  = (isStart || isEnd) ? 'middle' : 'start';
    const hasSub  = n.subLines.length > 0;

    // Calculate vertical centre of label block
    const lblBlockH = n.lblLines.length * LINE_H;
    const subBlockH = hasSub ? n.subLines.length * SUB_H + 6 : 0;
    const totalContentH = lblBlockH + subBlockH;
    const contentTopY = y + (NH - totalContentH) / 2;

    n.lblLines.forEach((line, li) => {
      nodeSvg += `<text x="${textX}" y="${contentTopY + li*LINE_H + LINE_H/2}"
        text-anchor="${anchor}" dominant-baseline="middle"
        style="fill:${textColor};font-size:14px;font-weight:700;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
    });

    if (hasSub) {
      const subTopY = contentTopY + lblBlockH + 6;
      n.subLines.forEach((line, li) => {
        nodeSvg += `<text x="${textX}" y="${subTopY + li*SUB_H + SUB_H/2}"
          text-anchor="${anchor}" dominant-baseline="middle"
          style="fill:#8b93a8;font-size:11px;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
      });
    }
  });

  return `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    ${defs}${edgeSvg}${nodeSvg}
  </svg>`;
}

/* ============================================================
   CYCLE  — improved rebuild with full label support
   ============================================================ */
function buildCycle(data) {
  const steps = (data.steps || []).slice(0, 7);
  const n = steps.length;
  if (!n) return '<p style="color:var(--text-3)">No cycle data.</p>';

  const W = 720, H = 720, CX = W / 2, CY = H / 2;
  const R  = 240;   // spoke radius
  const NR = 58;    // node circle radius
  const colors = ['#e8a020','#20c5b0','#3ec87a','#8b5cf6','#f97316','#ec4899','#06b6d4'];

  let arcs = '', nodes = '';

  steps.forEach((step, i) => {
    const angle     = (2 * Math.PI * i / n) - Math.PI / 2;
    const nextAngle = (2 * Math.PI * (i + 1) / n) - Math.PI / 2;
    const x  = CX + R * Math.cos(angle);
    const y  = CY + R * Math.sin(angle);
    const color = colors[i % colors.length];

    // Arc between nodes
    const largeArc = (1 / n) > 0.5 ? 1 : 0;
    const arcR = R - 26;
    arcs += `<path d="M ${CX + arcR*Math.cos(angle)} ${CY + arcR*Math.sin(angle)}
      A ${arcR} ${arcR} 0 ${largeArc} 1 ${CX + arcR*Math.cos(nextAngle)} ${CY + arcR*Math.sin(nextAngle)}"
      fill="none" stroke="${color}" stroke-width="2.5" opacity="0.25"/>`;

    // Arrowhead dot at 85% of arc
    const midA = angle + (Math.PI * 2 / n) * 0.82;
    const ax = CX + (arcR + 2) * Math.cos(midA);
    const ay = CY + (arcR + 2) * Math.sin(midA);
    arcs += `<circle cx="${ax}" cy="${ay}" r="5" fill="${color}" opacity="0.55"/>`;

    // Glow halo
    nodes += `<circle cx="${x}" cy="${y}" r="${NR + 10}"
      style="fill:${color};opacity:.07;"/>`;
    // Node circle
    nodes += `<circle cx="${x}" cy="${y}" r="${NR}"
      style="fill:#0f1420;stroke:${color};stroke-width:2.5;"/>`;

    // Step number
    nodes += `<text x="${x}" y="${y - NR + 18}" text-anchor="middle" dominant-baseline="middle"
      style="fill:${color};font-size:11px;font-weight:800;font-family:'JetBrains Mono',monospace;opacity:.7;">${i + 1}</text>`;

    // Label — wrap inside circle
    const label = String(step.label || '');
    const sub   = String(step.sub   || '');
    const labelWords = label.split(' ');
    const maxChars = 11; // chars per line inside the circle
    const labelLines = [];
    let cur = '';
    labelWords.forEach(w => {
      if ((cur + ' ' + w).trim().length <= maxChars) cur = (cur + ' ' + w).trim();
      else { if (cur) labelLines.push(cur); cur = w.substring(0, maxChars); }
    });
    if (cur) labelLines.push(cur);
    const lSlice = labelLines.slice(0, 3);

    const hasSub = sub.length > 0;
    const blockH = (lSlice.length - 1) * 16;
    // If sub exists push label up
    const labelCY = hasSub ? y - blockH/2 - 8 : y - blockH/2 + 3;

    lSlice.forEach((line, li) => {
      nodes += `<text x="${x}" y="${labelCY + li * 16}" text-anchor="middle" dominant-baseline="middle"
        style="fill:${color};font-size:12px;font-weight:800;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
    });

    if (hasSub) {
      // sub — one short line
      const subTrimmed = sub.length > 16 ? sub.substring(0, 15) + '…' : sub;
      nodes += `<text x="${x}" y="${y + blockH/2 + 14}" text-anchor="middle" dominant-baseline="middle"
        style="fill:#8b93a8;font-size:10px;font-family:Syne,sans-serif;">${escHtml(subTrimmed)}</text>`;
    }
  });

  // Center label
  const centerLabel = `
    <ellipse cx="${CX}" cy="${CY}" rx="56" ry="28" style="fill:rgba(255,255,255,.03);stroke:#2a3347;stroke-width:1.5;"/>
    <text x="${CX}" y="${CY}" text-anchor="middle" dominant-baseline="middle"
      style="fill:#4a5268;font-size:12px;font-weight:700;font-family:Syne,sans-serif;letter-spacing:.1em;">CYCLE</text>`;

  return `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    ${arcs}${nodes}${centerLabel}
  </svg>`;
}



/* ============================================================
   COMPARISON TABLE  — improved with full value display
   ============================================================ */
function buildComparison(data) {
  const headers = data.headers || ['A', 'B'];
  const items   = (data.items || []).slice(0, 6);
  const W       = 760, CRIT_W = 190, PAD = 18;
  const colW    = (W - CRIT_W) / headers.length;
  const HEADER_H = 60;
  // Dynamic row height based on longest value
  const maxValLen = Math.max(...items.map(it => {
    const vals = headers.map((_, ci) => ci === 0 ? (it.a||'') : (it.b||(it.points&&it.points[ci])||''));
    return Math.max(...vals.map(v => String(v).length));
  }), 1);
  const ROW_H = maxValLen > 30 ? 68 : maxValLen > 18 ? 58 : 50;
  const H = HEADER_H + items.length * ROW_H + PAD;
  const colors  = ['#e8a020', '#20c5b0', '#3ec87a', '#8b5cf6'];

  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;

  // Header bg
  svg += `<rect x="0" y="0" width="${W}" height="${HEADER_H}" rx="10"
    style="fill:#141926;stroke:#1e2535;stroke-width:1;"/>`;

  // Criterion column header
  svg += `<text x="${PAD}" y="${HEADER_H/2 + 1}" dominant-baseline="middle"
    style="fill:#4a5268;font-size:11px;font-weight:700;font-family:Syne,sans-serif;letter-spacing:.1em;text-transform:uppercase;">Criterion</text>`;

  headers.forEach((h, ci) => {
    const x     = CRIT_W + ci * colW + colW / 2;
    const color = colors[ci % colors.length];
    svg += `<rect x="${CRIT_W + ci * colW + 8}" y="10" width="${colW - 16}" height="${HEADER_H - 20}" rx="8"
      style="fill:${color};opacity:.1;"/>`;
    // Header text wrap
    const hStr = String(h);
    svg += `<text x="${x}" y="${HEADER_H/2 + 1}" text-anchor="middle" dominant-baseline="middle"
      style="fill:${color};font-size:15px;font-weight:800;font-family:Syne,sans-serif;">${escHtml(hStr.substring(0, 22))}</text>`;
  });

  // Rows
  items.forEach((item, ri) => {
    const y  = HEADER_H + ri * ROW_H;
    const bg = ri % 2 === 0 ? 'rgba(255,255,255,.018)' : 'transparent';
    svg += `<rect x="0" y="${y}" width="${W}" height="${ROW_H}" style="fill:${bg};"/>`;
    svg += `<line x1="0" y1="${y + ROW_H}" x2="${W}" y2="${y + ROW_H}" stroke="#1e2535" stroke-width="0.8"/>`;

    // Criterion label — wrap up to 2 lines
    const critStr = String(item.label || '');
    const critWords = critStr.split(' ');
    const critLines = [];
    let cCur = '';
    critWords.forEach(w => {
      if ((cCur + ' ' + w).trim().length <= 20) cCur = (cCur + ' ' + w).trim();
      else { if (cCur) critLines.push(cCur); cCur = w; }
    });
    if (cCur) critLines.push(cCur);
    critLines.slice(0, 2).forEach((line, li) => {
      const blockH = (Math.min(critLines.length, 2) - 1) * 14;
      svg += `<text x="${PAD}" y="${y + ROW_H/2 - blockH/2 + li * 14 + 1}" dominant-baseline="middle"
        style="fill:#8b93a8;font-size:12px;font-weight:700;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
    });

    // Vertical divider
    svg += `<line x1="${CRIT_W - 2}" y1="${HEADER_H}" x2="${CRIT_W - 2}" y2="${H}" stroke="#1e2535" stroke-width="1"/>`;

    // Values
    headers.forEach((h, ci) => {
      const val   = ci === 0 ? String(item.a || '') : String(item.b || (item.points && item.points[ci]) || '');
      const x     = CRIT_W + ci * colW + colW / 2;

      // Wrap value text
      const vWords = val.split(' ');
      const vLines = [];
      let vCur = '';
      const maxVC = Math.floor(colW / 8) - 2;
      vWords.forEach(w => {
        if ((vCur + ' ' + w).trim().length <= Math.max(maxVC, 14)) vCur = (vCur + ' ' + w).trim();
        else { if (vCur) vLines.push(vCur); vCur = w; }
      });
      if (vCur) vLines.push(vCur);
      const vSlice = vLines.slice(0, 2);
      const blockH = (vSlice.length - 1) * 15;
      vSlice.forEach((line, li) => {
        svg += `<text x="${x}" y="${y + ROW_H/2 - blockH/2 + li * 15 + 1}" text-anchor="middle" dominant-baseline="middle"
          style="fill:#eae6df;font-size:12px;font-family:Syne,sans-serif;font-weight:500;">${escHtml(line)}</text>`;
      });

      if (ci < headers.length - 1) {
        svg += `<line x1="${CRIT_W + (ci+1)*colW}" y1="${HEADER_H}" x2="${CRIT_W + (ci+1)*colW}" y2="${H}"
          stroke="#1e2535" stroke-width="0.6"/>`;
      }
    });
  });

  svg += `</svg>`;
  return svg;
}

/* ============================================================
   TIMELINE  — improved with full label support
   ============================================================ */
function buildTimeline(data) {
  const events = (data.events || []).slice(0, 8);
  if (!events.length) return '<p style="color:var(--text-3)">No timeline data.</p>';

  const W = 760, ROW_H = 96, PAD_TOP = 28, LINE_X = 130;
  const H = PAD_TOP + events.length * ROW_H + 28;
  const colors = ['#e8a020','#20c5b0','#3ec87a','#8b5cf6','#f97316','#ec4899','#e8a020','#20c5b0'];

  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;

  // Vertical timeline rail
  svg += `<line x1="${LINE_X}" y1="${PAD_TOP}" x2="${LINE_X}" y2="${H - 28}"
    stroke="#1e2535" stroke-width="3"/>`;

  events.forEach((ev, i) => {
    const y     = PAD_TOP + i * ROW_H + ROW_H / 2;
    const color = colors[i % colors.length];

    // Glow pulse ring
    svg += `<circle cx="${LINE_X}" cy="${y}" r="22" style="fill:${color};opacity:.07;"/>`;
    svg += `<circle cx="${LINE_X}" cy="${y}" r="14" style="fill:${color};opacity:.14;"/>`;
    // Main dot
    svg += `<circle cx="${LINE_X}" cy="${y}" r="9" style="fill:${color};"/>`;
    // Inner dot
    svg += `<circle cx="${LINE_X}" cy="${y}" r="4" style="fill:#080b12;"/>`;

    // Year badge (left of line)
    const yr = String(ev.year || '').substring(0, 10);
    const yrW = Math.max(56, yr.length * 9 + 16);
    svg += `<rect x="${LINE_X - yrW - 12}" y="${y - 15}" width="${yrW}" height="30" rx="8"
      style="fill:rgba(255,255,255,.04);stroke:${color};stroke-width:1.3;"/>`;
    svg += `<text x="${LINE_X - 12 - yrW/2}" y="${y + 1}" text-anchor="middle" dominant-baseline="middle"
      style="fill:${color};font-size:11px;font-weight:800;font-family:'JetBrains Mono',monospace;">${escHtml(yr)}</text>`;

    // Connector tick
    svg += `<line x1="${LINE_X + 9}" y1="${y}" x2="${LINE_X + 24}" y2="${y}"
      stroke="${color}" stroke-width="1.5" opacity="0.5"/>`;

    // Event label — full text, wrap up to 2 lines
    const label = String(ev.label || '');
    const sub   = String(ev.sub   || '');
    const maxCharsLabel = 42;
    const lWords = label.split(' ');
    const lLines = [];
    let lCur = '';
    lWords.forEach(w => {
      if ((lCur + ' ' + w).trim().length <= maxCharsLabel) lCur = (lCur + ' ' + w).trim();
      else { if (lCur) lLines.push(lCur); lCur = w; }
    });
    if (lCur) lLines.push(lCur);
    const lSlice = lLines.slice(0, 2);

    const hasSub = sub.length > 0;
    const blockH = (lSlice.length - 1) * 18;
    const labelY = hasSub ? y - blockH/2 - 11 : y - blockH/2 + 1;

    lSlice.forEach((line, li) => {
      svg += `<text x="${LINE_X + 30}" y="${labelY + li * 18}" dominant-baseline="middle"
        style="fill:#eae6df;font-size:14px;font-weight:700;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
    });

    if (hasSub) {
      // sub — wrap up to 2 lines
      const sWords = sub.split(' ');
      const sLines = [];
      let sCur = '';
      sWords.forEach(w => {
        if ((sCur + ' ' + w).trim().length <= 54) sCur = (sCur + ' ' + w).trim();
        else { if (sCur) sLines.push(sCur); sCur = w; }
      });
      if (sCur) sLines.push(sCur);
      sLines.slice(0, 2).forEach((line, li) => {
        svg += `<text x="${LINE_X + 30}" y="${y + blockH/2 + 12 + li * 14}" dominant-baseline="middle"
          style="fill:#8b93a8;font-size:11px;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
      });
    }
  });

  svg += `</svg>`;
  return svg;
}

/* ============================================================
   PYRAMID  — improved with full label support and taller rows
   ============================================================ */
function buildPyramid(data) {
  const levels = (data.levels || []).slice(0, 7);
  if (!levels.length) return '<p style="color:var(--text-3)">No pyramid data.</p>';

  const W = 720, BASE_W = 640, TOP_W = 60, ROW_H = 76, PAD_TOP = 20;
  const H = PAD_TOP + levels.length * ROW_H + 30;
  const CX = W / 2;
  const colors = [
    ['rgba(232,160,32,.88)','#e8a020'],
    ['rgba(32,197,176,.78)','#20c5b0'],
    ['rgba(62,200,122,.68)','#3ec87a'],
    ['rgba(139,92,246,.62)','#8b5cf6'],
    ['rgba(249,115,22,.62)','#f97316'],
    ['rgba(236,72,153,.58)','#ec4899'],
    ['rgba(6,182,212,.52)','#06b6d4'],
  ];

  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;

  levels.forEach((lv, i) => {
    const widthPct = lv.width_pct || ((i + 1) / levels.length * 90 + 10);
    const rowW  = TOP_W + (BASE_W - TOP_W) * (widthPct / 100);
    const x     = CX - rowW / 2;
    const y     = PAD_TOP + i * ROW_H;
    const [fill, stroke] = colors[i % colors.length];

    // Glow shadow
    svg += `<rect x="${x - 2}" y="${y - 2}" width="${rowW + 4}" height="${ROW_H - 1}" rx="6"
      style="fill:${stroke};opacity:.1;"/>`;
    // Main bar
    svg += `<rect x="${x}" y="${y}" width="${rowW}" height="${ROW_H - 4}" rx="5"
      style="fill:${fill};stroke:${stroke};stroke-width:1.5;"/>`;

    // Level number badge
    svg += `<rect x="${x + 10}" y="${y + ROW_H/2 - 15}" width="30" height="30" rx="8"
      style="fill:rgba(0,0,0,.25);"/>`;
    svg += `<text x="${x + 25}" y="${y + ROW_H/2 + 1}" text-anchor="middle" dominant-baseline="middle"
      style="fill:rgba(255,255,255,.9);font-size:12px;font-weight:800;font-family:'JetBrains Mono',monospace;">${i + 1}</text>`;

    // Label — wrapped, up to 2 lines
    const label = String(lv.label || '');
    const sub   = String(lv.sub   || '');
    const maxChars = Math.floor(rowW / 9) - 8;
    const lWords = label.split(' ');
    const lLines = [];
    let lCur = '';
    lWords.forEach(w => {
      if ((lCur + ' ' + w).trim().length <= Math.max(maxChars, 16)) lCur = (lCur + ' ' + w).trim();
      else { if (lCur) lLines.push(lCur); lCur = w; }
    });
    if (lCur) lLines.push(lCur);
    const lSlice = lLines.slice(0, 2);

    const hasSub = sub.length > 0;
    const blockH = (lSlice.length - 1) * 17;
    const labelCY = hasSub ? y + ROW_H/2 - blockH/2 - 9 : y + ROW_H/2 - blockH/2 + 1;

    lSlice.forEach((line, li) => {
      svg += `<text x="${CX + 14}" y="${labelCY + li * 17}" text-anchor="middle" dominant-baseline="middle"
        style="fill:#fff;font-size:14px;font-weight:800;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
    });

    if (hasSub) {
      const subTrimmed = sub.length > 50 ? sub.substring(0, 48) + '…' : sub;
      svg += `<text x="${CX + 14}" y="${y + ROW_H/2 + blockH/2 + 12}" text-anchor="middle" dominant-baseline="middle"
        style="fill:rgba(255,255,255,.75);font-size:11px;font-family:Syne,sans-serif;">${escHtml(subTrimmed)}</text>`;
    }
  });

  svg += `</svg>`;
  return svg;
}

/* ============================================================
   MINDMAP  — collision-free layout
   Branches split into LEFT / RIGHT columns, equally spaced
   vertically. Sub-items extend further outward in their own
   column. No angular spoke placement = no collisions.
   ============================================================ */
function buildMindmap(data) {
  const center   = String(data.center || 'Topic');
  const branches = (data.branches || []).slice(0, 8);
  const n        = branches.length;
  if (!n) return '<p style="color:var(--text-3)">No branches.</p>';

  const colors = ['#e8a020','#20c5b0','#3ec87a','#8b5cf6','#f97316','#ec4899','#06b6d4','#f43f5e'];

  // ── text wrapper helper ───────────────────────────────────
  function wrapText(text, maxCh) {
    const words = String(text).split(' ');
    const lines = []; let cur = '';
    words.forEach(w => {
      const test = cur ? cur + ' ' + w : w;
      if (test.length <= maxCh) { cur = test; }
      else { if (cur) lines.push(cur); cur = w.length > maxCh ? w.substring(0, maxCh) : w; }
    });
    if (cur) lines.push(cur);
    return lines;
  }

  // ── Layout constants ──────────────────────────────────────
  const ITEM_H      = 30;   // pill height (1 line)
  const ITEM_LINE_H = 13;   // extra height per extra item line
  const ITEM_GAP    = 12;   // gap between pills
  const BR_PAD_V    = 36;   // guaranteed vertical gap between branch slots
  const BR_H_BASE   = 42;   // branch box height for 1-line label
  const BR_LINE_H   = 18;   // extra height per extra branch label line

  // measure how tall a branch's slot needs to be
  function measureBranch(br) {
    const lblLines = wrapText(br.label || '', 16).slice(0, 2);
    const brH = BR_H_BASE + (lblLines.length - 1) * BR_LINE_H;
    const items = (br.items || []).slice(0, 5);
    let itemsH = 0;
    items.forEach(item => {
      const iLines = wrapText(String(item), 20).slice(0, 2);
      itemsH += ITEM_H + (iLines.length - 1) * ITEM_LINE_H + ITEM_GAP;
    });
    if (items.length) itemsH -= ITEM_GAP; // remove last gap
    return { lblLines, brH, items, itemsH };
  }

  // Split branches: even-indexed → right, odd-indexed → left
  const rightList = [], leftList = [];
  branches.forEach((br, i) => {
    (i % 2 === 0 ? rightList : leftList).push({ br, i });
  });

  // Total height needed per side (max of both sides drives canvas height)
  function sideH(list) {
    return list.reduce((acc, { br }) => {
      const m = measureBranch(br);
      return acc + Math.max(m.brH, m.itemsH) + BR_PAD_V;
    }, BR_PAD_V);
  }

  const neededH = Math.max(sideH(rightList), sideH(leftList), 380);
  const W = 1200, H = neededH, CY = H / 2, CX = W / 2;

  // Column x positions
  const BR_R_X   = CX + 200;   // right branch col centre-x
  const BR_L_X   = CX - 200;   // left branch col centre-x
  const IT_R_X   = CX + 430;   // right items col centre-x
  const IT_L_X   = CX - 430;   // left items col centre-x
  const CRX = 84, CRY = 40;    // centre ellipse radii

  let svg = `<svg class="viz-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<defs>
    <radialGradient id="mm-bg" cx="50%" cy="50%" r="45%">
      <stop offset="0%" stop-color="rgba(232,160,32,.07)"/>
      <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
    </radialGradient>
  </defs>`;
  svg += `<ellipse cx="${CX}" cy="${CY}" rx="${W*0.44}" ry="${H*0.44}" fill="url(#mm-bg)"/>`;

  // ── Centre node ───────────────────────────────────────────
  const cLines = wrapText(center, 14).slice(0, 3);
  svg += `<ellipse cx="${CX}" cy="${CY}" rx="${CRX+7}" ry="${CRY+7}"
    style="fill:rgba(232,160,32,.06);stroke:rgba(232,160,32,.18);stroke-width:1;"/>`;
  svg += `<ellipse cx="${CX}" cy="${CY}" rx="${CRX}" ry="${CRY}"
    style="fill:rgba(232,160,32,.15);stroke:#e8a020;stroke-width:2.5;"/>`;
  cLines.forEach((line, li) => {
    const totalH = (cLines.length - 1) * 17;
    svg += `<text x="${CX}" y="${CY - totalH/2 + li*17}"
      text-anchor="middle" dominant-baseline="middle"
      style="fill:#e8a020;font-size:14px;font-weight:800;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
  });

  // ── Draw one side ─────────────────────────────────────────
  function drawSide(list, isRight) {
    const BRX  = isRight ? BR_R_X : BR_L_X;
    const ITX  = isRight ? IT_R_X : IT_L_X;
    const SPKX = isRight ? CX + CRX : CX - CRX; // spoke origin on ellipse

    // Compute total slot heights to centre the group on CY
    const heights = list.map(({ br }) => {
      const m = measureBranch(br);
      return Math.max(m.brH, m.itemsH) + BR_PAD_V;
    });
    const totalH = heights.reduce((a, b) => a + b, 0);
    let curY = CY - totalH / 2;

    list.forEach(({ br, i }, idx) => {
      const color = colors[i % colors.length];
      const m     = measureBranch(br);
      const slotH = heights[idx];
      const by    = curY + slotH / 2;
      curY += slotH;

      // Branch box dimensions
      const BW = Math.max(148, Math.max(...m.lblLines.map(l => l.length)) * 8.5 + 30);
      const BH = m.brH;

      // Bezier spoke: centre ellipse edge → branch box
      const cpx = (SPKX + BRX) / 2;
      svg += `<path d="M${SPKX},${CY} C${cpx},${CY} ${cpx},${by} ${BRX},${by}"
        fill="none" stroke="${color}" stroke-width="7" opacity="0.08"/>`;
      svg += `<path d="M${SPKX},${CY} C${cpx},${CY} ${cpx},${by} ${BRX},${by}"
        fill="none" stroke="${color}" stroke-width="2" opacity="0.65"/>`;

      // Branch node
      svg += `<rect x="${BRX-BW/2-3}" y="${by-BH/2-3}" width="${BW+6}" height="${BH+6}" rx="13"
        style="fill:${color};opacity:.1;"/>`;
      svg += `<rect x="${BRX-BW/2}" y="${by-BH/2}" width="${BW}" height="${BH}" rx="11"
        style="fill:#0f1420;stroke:${color};stroke-width:2;"/>`;
      m.lblLines.forEach((line, li) => {
        const totalLH = (m.lblLines.length - 1) * BR_LINE_H;
        svg += `<text x="${BRX}" y="${by - totalLH/2 + li*BR_LINE_H}"
          text-anchor="middle" dominant-baseline="middle"
          style="fill:${color};font-size:13px;font-weight:800;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
      });

      // Sub-items — vertical column at ITX, centred on slot
      if (!m.items.length) return;

      // Compute total items block height
      let totalItemsH = 0;
      const itemMeasures = m.items.map(item => {
        const iLines = wrapText(String(item), 20).slice(0, 2);
        const IH = ITEM_H + (iLines.length - 1) * ITEM_LINE_H;
        const IW = Math.max(108, Math.max(...iLines.map(l => l.length)) * 7 + 26);
        totalItemsH += IH;
        return { iLines, IH, IW };
      });
      totalItemsH += (m.items.length - 1) * ITEM_GAP;

      let iy = by - totalItemsH / 2;

      itemMeasures.forEach(({ iLines, IH, IW }, ii) => {
        const itemCY = iy + IH / 2;
        iy += IH + ITEM_GAP;

        // Elbow connector: branch box edge → item pill edge
        const brEdgeX  = isRight ? BRX + BW/2 : BRX - BW/2;
        const itEdgeX  = isRight ? ITX - IW/2  : ITX + IW/2;
        const elbowX   = (brEdgeX + itEdgeX) / 2;
        svg += `<path d="M${brEdgeX},${by} C${elbowX},${by} ${elbowX},${itemCY} ${itEdgeX},${itemCY}"
          fill="none" stroke="${color}" stroke-width="1.3" stroke-dasharray="5 3" opacity="0.42"/>`;

        // Pill
        svg += `<rect x="${ITX-IW/2}" y="${itemCY-IH/2}" width="${IW}" height="${IH}" rx="12"
          style="fill:rgba(255,255,255,.04);stroke:${color};stroke-width:1.2;"/>`;
        iLines.forEach((line, li) => {
          const totalLH = (iLines.length - 1) * ITEM_LINE_H;
          svg += `<text x="${ITX}" y="${itemCY - totalLH/2 + li*ITEM_LINE_H}"
            text-anchor="middle" dominant-baseline="middle"
            style="fill:#dde2f0;font-size:11px;font-weight:600;font-family:Syne,sans-serif;">${escHtml(line)}</text>`;
        });
      });
    });
  }

  drawSide(rightList, true);
  drawSide(leftList,  false);

  svg += `</svg>`;
  return svg;
}


function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function fmtTime(s) {
  s = Math.round(s);
  if (s < 60) return s + 's';
  return `${Math.floor(s/60)}m ${s%60}s`;
}

// =====================================================================
// SOURCES / REFERENCES
// =====================================================================
const SOURCE_ICONS = {
  book: '📖',
  paper: '📄',
  website: '🌐',
  course: '🎓',
  encyclopedia: '📕',
};

let _sourcesOpen = false;
let _sourcesLoaded = false;   // per-slide flag
let _sourcesLoading = false;

function resetSources() {
  _sourcesOpen = false;
  _sourcesLoaded = false;
  // Bug 3 fix: explicitly clear loading flag so a stale in-flight request
  // from the previous slide doesn't block fetching on the new slide.
  _sourcesLoading = false;

  const section = document.getElementById('sources-section');
  const body    = document.getElementById('sources-body');
  const icon    = document.getElementById('sources-toggle-icon');
  if (!section) return;

  section.style.display = 'block';
  body.innerHTML = '';
  body.classList.remove('open');
  icon.classList.remove('open');
}

function toggleSources() {
  _sourcesOpen = !_sourcesOpen;

  const body = document.getElementById('sources-body');
  const icon = document.getElementById('sources-toggle-icon');

  if (_sourcesOpen) {
    body.classList.add('open');
    icon.classList.add('open');
    if (!_sourcesLoaded && !_sourcesLoading) {
      fetchSources();
    }
  } else {
    body.classList.remove('open');
    icon.classList.remove('open');
  }
}

async function fetchSources() {
  if (_sourcesLoading) return;

  // Single body reference used throughout the entire function
  const body = document.getElementById('sources-body');

  // Bug 1 fix: guard against missing sessionId before any network call
  if (!state.sessionId) {
    body.innerHTML = `<div class="sources-error">⚠ No active session. Please generate slides first.</div>`;
    return;
  }

  _sourcesLoading = true;

  body.innerHTML = `
    <div class="sources-loading">
      <div class="loading-ring" style="width:22px;height:22px;border-width:2px;flex-shrink:0;"></div>
      Finding credible sources…
    </div>`;

  try {
    const res = await fetch(apiPath('/sources'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, slide_index: state.currentIndex }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Failed to load sources.');

    renderSources(data.sources || []);
    _sourcesLoaded = true;

  } catch (err) {
    body.innerHTML = `<div class="sources-error">⚠ ${escHtml(err.message)}
      <br><button class="btn-load-sources" style="margin-top:.5rem;" onclick="fetchSources()">Retry</button>
    </div>`;
  } finally {
    _sourcesLoading = false;
  }
}

function renderSources(sources) {
  const body = document.getElementById('sources-body');

  if (!sources || sources.length === 0) {
    // Bug 2 fix: include retry button just like failed fetches do
    body.innerHTML = `<div class="sources-error">No sources found for this slide.
      <br><button class="btn-load-sources" style="margin-top:.5rem;" onclick="_sourcesLoaded=false;fetchSources()">Try Again</button>
    </div>`;
    return;
  }

  const items = sources.map((s, i) => {
    const icon = SOURCE_ICONS[s.type] || '🔗';
    const typeClass = (s.type || 'website').replace(/\s+/g, '');
    const badge = `<span class="source-type-badge ${typeClass}">${escHtml(s.type || 'source')}</span>`;

    const titleHtml = s.url_hint
      ? `<a href="${escHtml(s.url_hint)}" target="_blank" rel="noopener noreferrer">${escHtml(s.title)}</a>`
      : escHtml(s.title);

    const meta = [s.authors, s.year].filter(Boolean).join(' · ');

    return `
      <div class="source-item" style="animation-delay:${i * 0.06}s">
        <div class="source-icon">${icon}</div>
        <div class="source-content">
          <div class="source-title">${titleHtml}${badge}</div>
          ${meta ? `<div class="source-meta">${escHtml(meta)}</div>` : ''}
          ${s.description ? `<div class="source-desc">${escHtml(s.description)}</div>` : ''}
        </div>
      </div>`;
  }).join('');

  body.innerHTML = items;
}

// =====================================================================
// VISIBLE NAV BUTTONS + KEYBOARD NAVIGATION
// =====================================================================
function updateNavButtons() {
  const prev = document.getElementById('btn-prev');
  const next = document.getElementById('btn-next');
  if (!prev || !next) return;
  prev.disabled = state.currentIndex <= 0;
  next.disabled = state.currentIndex >= state.slides.length - 1;
}

function navPrev() {
  if (state.currentIndex <= 0) return;
  stopTimer();
  state.currentIndex--;
  loadSlide(state.currentIndex);
  document.getElementById('learning-body').scrollTop = 0;
  updateNavButtons();
}

function navNext() {
  if (state.currentIndex >= state.slides.length - 1) {
    showStats();
    return;
  }
  stopTimer();
  state.quizDone = true;
  state.currentIndex++;
  loadSlide(state.currentIndex);
  document.getElementById('learning-body').scrollTop = 0;
  updateNavButtons();
}

document.addEventListener('keydown', e => {
  const learningScreen = document.getElementById('screen-learning');
  if (!learningScreen || !learningScreen.classList.contains('active')) return;
  const tag = document.activeElement ? document.activeElement.tagName : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  if (e.key === 'ArrowRight') { e.preventDefault(); navNext(); }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); navPrev(); }
});

// =====================================================================
// PPT EXPORT
// =====================================================================

let _pptTheme = 'dark';

function selectPptTheme(theme) {
  _pptTheme = theme;
  document.getElementById('ppt-theme-dark').classList.toggle('selected', theme === 'dark');
  document.getElementById('ppt-theme-light').classList.toggle('selected', theme === 'light');
}

function openPptModal() {
  if (!state.sessionId) {
    showToast('Start a session first before exporting.');
    return;
  }
  const confirmBtn  = document.getElementById('ppt-confirm-btn');
  const cancelBtn   = document.getElementById('ppt-cancel-btn');
  const progressWrap= document.getElementById('ppt-progress-wrap');
  const fill        = document.getElementById('ppt-progress-fill');
  const status      = document.getElementById('ppt-status-text');

  confirmBtn.disabled = false;
  confirmBtn.textContent = '⬇ Download PPT';
  cancelBtn.disabled = false;
  progressWrap.classList.remove('show');
  fill.style.width = '0%';
  fill.style.background = '';
  status.textContent = 'Generating slides…';

  // Reset theme to dark each time modal opens
  selectPptTheme('dark');

  document.getElementById('ppt-modal-topic').textContent = state.topic || 'Current Session';
  document.getElementById('ppt-modal').classList.add('show');
}

function closePptModal() {
  document.getElementById('ppt-modal').classList.remove('show');
}

document.getElementById('ppt-modal').addEventListener('click', function(e) {
  if (e.target === this) closePptModal();
});

async function confirmPptDownload() {
  const confirmBtn  = document.getElementById('ppt-confirm-btn');
  const cancelBtn   = document.getElementById('ppt-cancel-btn');
  const progressWrap= document.getElementById('ppt-progress-wrap');
  const fill        = document.getElementById('ppt-progress-fill');
  const status      = document.getElementById('ppt-status-text');

  confirmBtn.disabled = true;
  cancelBtn.disabled  = true;
  confirmBtn.textContent = 'Building…';
  progressWrap.classList.add('show');
  fill.style.background = '';

  const steps = [
    { pct: 12, msg: 'Laying out title slide…' },
    { pct: 30, msg: 'Building content slides…' },
    { pct: 55, msg: `Applying ${_pptTheme} theme & colours…` },
    { pct: 75, msg: 'Adding bullet cards & accents…' },
    { pct: 90, msg: 'Finalising closing slide…' },
  ];
  let stepIdx = 0;
  const ticker = setInterval(() => {
    if (stepIdx < steps.length) {
      fill.style.width   = steps[stepIdx].pct + '%';
      status.textContent = steps[stepIdx].msg;
      stepIdx++;
    }
  }, 520);

  try {
    const res = await fetch(apiPath('/generate_ppt'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, theme: _pptTheme }),
    });

    clearInterval(ticker);

    if (!res.ok) {
      let errMsg = 'Failed to generate PPT.';
      try { const d = await res.json(); errMsg = d.error || errMsg; } catch (_) {}
      throw new Error(errMsg);
    }

    fill.style.width   = '100%';
    status.textContent = 'Done! Starting download…';

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');

    let fname = (state.topic || 'slides').replace(/[^a-zA-Z0-9 _-]/g, '').trim().replace(/\s+/g, '_').slice(0, 40)
                + `_${_pptTheme === 'light' ? 'Light' : 'Dark'}_AdaptiveLearn.pptx`;
    const cd  = res.headers.get('Content-Disposition');
    if (cd) {
      const m = cd.match(/filename[^;=\n]*=['"]?([^'";\n]+)/i);
      if (m) fname = m[1];
    }
    a.href     = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);

    setTimeout(() => {
      closePptModal();
      const themeLabel = _pptTheme === 'light' ? '☀️ Light' : '🌙 Dark';
      showPptSuccessToast(`🎞️  ${themeLabel} PPT downloaded — ${state.slides.length} slides ready!`);
    }, 800);

  } catch (err) {
    clearInterval(ticker);
    fill.style.width      = '40%';
    fill.style.background = 'var(--red)';
    status.textContent    = '✗  ' + err.message;
    confirmBtn.disabled   = false;
    confirmBtn.textContent= '⬇ Retry Download';
    cancelBtn.disabled    = false;
  }
}

function showPptSuccessToast(msg) {
  const t = document.getElementById('ppt-success-toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4500);
}