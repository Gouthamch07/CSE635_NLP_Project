const $ = (sel) => document.querySelector(sel);

const welcome      = $("#welcome");
const thread       = $("#thread");
const composer     = $("#composer");
const inputEl      = $("#input");
const sendBtn      = $("#send");
const clearBtn     = $("#clear-btn");
const themeBtn     = $("#theme-toggle");
const traceBtn     = $("#trace-toggle");
const traceClose   = $("#trace-close");
const traceDrawer  = $("#trace-drawer");
const traceScrim   = $("#trace-scrim");
const traceBody    = $("#trace-body");

const userId = localStorage.getItem("ubcse_uid") || (() => {
  const id = "u_" + Math.random().toString(36).slice(2, 10);
  localStorage.setItem("ubcse_uid", id);
  return id;
})();

const savedTheme = localStorage.getItem("ubcse_theme") || "dark";
document.documentElement.dataset.theme = savedTheme;
themeBtn.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("ubcse_theme", next);
});

if (window.marked) {
  marked.setOptions({ breaks: true, gfm: true });
}
function renderMarkdown(text) {
  if (!window.marked || !window.DOMPurify) return escapeHtml(text);
  const raw = marked.parse(text || "");
  return DOMPurify.sanitize(raw);
}
function escapeHtml(s) {
  return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + "px";
}
inputEl.addEventListener("input", autosize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); composer.requestSubmit(); }
});

document.querySelectorAll(".prompt-card").forEach((card) => {
  card.addEventListener("click", () => {
    inputEl.value = card.dataset.prompt;
    autosize();
    composer.requestSubmit();
  });
});

clearBtn.addEventListener("click", async () => {
  await fetch("/clear", { method: "POST" }).catch(() => {});
  thread.innerHTML = "";
  thread.classList.add("hidden");
  welcome.classList.remove("hidden");
  inputEl.value = "";
  localStorage.removeItem("ubcse_personalize_asked");
  localStorage.removeItem("ubcse_personalize_choice");
  inputEl.focus();
});

function appendPersonalizePrompt(onChoice) {
  const wrap = document.createElement("div");
  wrap.className = "msg bot personalize-prompt";
  wrap.innerHTML = `
    <div class="bubble">
      <p><strong>Before I answer — should I personalize this chat?</strong></p>
      <p>If yes, I'll remember your program, interests, and prior questions across this session and future visits — so answers get more tailored over time. If no, every question starts fresh and nothing about you is stored.</p>
      <div class="personalize-actions">
        <button class="personalize-btn yes" data-choice="yes">Yes, personalize</button>
        <button class="personalize-btn no"  data-choice="no">No, keep it generic</button>
      </div>
    </div>
  `;
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;

  wrap.querySelectorAll(".personalize-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const choice = btn.dataset.choice;
      const enabled = choice === "yes";
      localStorage.setItem("ubcse_personalize_asked", "1");
      localStorage.setItem("ubcse_personalize_choice", choice);
      wrap.querySelectorAll(".personalize-btn").forEach((b) => (b.disabled = true));
      try {
        await fetch("/personalize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, enabled }),
        });
      } catch (_) { /* ignore network errors */ }
      const note = document.createElement("p");
      note.className = "personalize-confirm";
      note.textContent = enabled
        ? "Personalization is on. I'll tailor future answers based on what you tell me."
        : "Personalization is off. Every question is treated as fresh.";
      wrap.querySelector(".bubble").appendChild(note);
      if (typeof onChoice === "function") onChoice(enabled);
    });
  });
}

function openTrace() {
  traceDrawer.classList.add("show");
  traceScrim.classList.add("show");
}
function closeTrace() {
  traceDrawer.classList.remove("show");
  traceScrim.classList.remove("show");
}
traceBtn.addEventListener("click", openTrace);
traceClose.addEventListener("click", closeTrace);
traceScrim.addEventListener("click", closeTrace);

function ensureThreadShown() {
  if (!welcome.classList.contains("hidden")) {
    welcome.classList.add("hidden");
    thread.classList.remove("hidden");
  }
}

function appendUser(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  wrap.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
}

function appendBot(text, meta) {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  let html = `<div class="bubble">${renderMarkdown(text)}</div>`;
  if (meta) {
    const pills = [];
    if (meta.scope) {
      pills.push(`<span class="meta-pill scope-${meta.scope}">${meta.scope.replace(/_/g, " ")}</span>`);
    }
    if (Number.isFinite(meta.ttft_ms))  pills.push(`<span class="meta-pill">ttft ${Math.round(meta.ttft_ms)}ms</span>`);
    if (Number.isFinite(meta.total_ms)) pills.push(`<span class="meta-pill">${Math.round(meta.total_ms)}ms total</span>`);
    if (pills.length) html += `<div class="meta-row">${pills.join("")}</div>`;

    if (meta.sources && meta.sources.length) {
      const chips = meta.sources.slice(0, 5).map((s) => {
        const label = escapeHtml(s.title || s.section || s.url || "source");
        const url = s.url ? escapeHtml(s.url) : "";
        const inner = `<span class="source-num">${s.index}</span><span>${label}</span>`;
        return url
          ? `<a class="source-chip" href="${url}" target="_blank" rel="noopener">${inner}</a>`
          : `<span class="source-chip">${inner}</span>`;
      }).join("");
      html += `<div class="sources-row">${chips}</div>`;
    }
  }
  wrap.innerHTML = html;
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
}

function appendBotShell() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.innerHTML = `<div class="bubble"></div>`;
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return wrap;
}

function updateBotShell(wrap, text) {
  const bubble = wrap.querySelector(".bubble");
  if (bubble) bubble.innerHTML = renderMarkdown(text);
  thread.scrollTop = thread.scrollHeight;
}

function finalizeBotShell(wrap, meta) {
  if (!meta) return;
  wrap.querySelectorAll(".meta-row,.sources-row").forEach((el) => el.remove());

  const pills = [];
  if (meta.scope) {
    pills.push(`<span class="meta-pill scope-${meta.scope}">${meta.scope.replace(/_/g, " ")}</span>`);
  }
  if (Number.isFinite(meta.ttft_ms))  pills.push(`<span class="meta-pill">ttft ${Math.round(meta.ttft_ms)}ms</span>`);
  if (Number.isFinite(meta.total_ms)) pills.push(`<span class="meta-pill">${Math.round(meta.total_ms)}ms total</span>`);
  if (pills.length) {
    const row = document.createElement("div");
    row.className = "meta-row";
    row.innerHTML = pills.join("");
    wrap.appendChild(row);
  }

  if (meta.sources && meta.sources.length) {
    const row = document.createElement("div");
    row.className = "sources-row";
    row.innerHTML = meta.sources.slice(0, 5).map((s) => {
      const label = escapeHtml(s.title || s.section || s.url || "source");
      const url = s.url ? escapeHtml(s.url) : "";
      const inner = `<span class="source-num">${s.index}</span><span>${label}</span>`;
      return url
        ? `<a class="source-chip" href="${url}" target="_blank" rel="noopener">${inner}</a>`
        : `<span class="source-chip">${inner}</span>`;
    }).join("");
    wrap.appendChild(row);
  }
  thread.scrollTop = thread.scrollHeight;
}

function appendTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.id = "typing-indicator";
  wrap.innerHTML = `<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
}
function removeTyping() {
  const t = document.getElementById("typing-indicator");
  if (t) t.remove();
}

function renderTrace(trace) {
  if (!trace || !trace.stages || !trace.stages.length) {
    traceBody.innerHTML = `<div class="empty-trace"><p>No retrieval trace for this turn.</p></div>`;
    return;
  }
  const stageLabels = {
    "bm25": "BM25 sparse",
    "sparse": "BM25 sparse",
    "dense": "Dense (Pinecone)",
    "pinecone": "Dense (Pinecone)",
    "rrf": "RRF fusion",
    "rrf_fusion": "RRF fusion",
    "lexical": "Lexical rerank",
    "lexical_rerank": "Lexical rerank",
    "cross_encoder": "Cross-encoder rerank",
    "cross_encoder_rerank": "Cross-encoder rerank",
    "ce": "Cross-encoder rerank",
  };

  // Build id -> {title, section, url} map from final hits
  const idMeta = {};
  (trace.hits || []).forEach((h) => {
    idMeta[h.id] = {
      title: h.title || h.section || h.id,
      section: h.section || "",
      url: h.url || "",
    };
  });

  const blocks = trace.stages.map((stage) => {
    const name = stageLabels[stage.stage] || stage.stage || "stage";
    const scores = stage.scores || [];
    const items = scores.slice(0, 8).map((entry) => {
      const [id, score] = Array.isArray(entry) ? entry : [entry.id, entry.score];
      const meta = idMeta[id] || { title: id, section: "" };
      const titleHtml = escapeHtml(meta.title || id);
      const section = meta.section ? `<div class="hit-section">${escapeHtml(meta.section)}</div>` : "";
      return `
        <div class="hit">
          <div>
            <div class="hit-title">${titleHtml}</div>
            ${section}
          </div>
          <div class="hit-score">${Number(score).toFixed(3)}</div>
        </div>
      `;
    }).join("");
    return `
      <div class="trace-stage">
        <div class="trace-stage-head">
          <span class="stage-name">${escapeHtml(name)}</span>
          <span class="stage-count">${scores.length} ${scores.length === 1 ? "hit" : "hits"}</span>
        </div>
        <div class="trace-hits">${items || `<div class="hit"><div class="hit-title" style="color:var(--text-3)">no hits</div></div>`}</div>
      </div>
    `;
  }).join("");
  traceBody.innerHTML = blocks;
}

async function sendMessage(text) {
  sendBtn.disabled = true;
  appendTyping();
  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, user_id: userId }),
    });
    if (!res.ok) {
      removeTyping();
      const err = await res.text();
      appendBot(`**Error ${res.status}**\n\n${err.slice(0, 600)}`);
      return;
    }
    if (!res.body) throw new Error("Streaming is not supported by this browser.");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let botWrap = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "start") {
          removeTyping();
          botWrap = appendBotShell();
          renderTrace(event.retrieval_trace);
        } else if (event.type === "token") {
          if (!botWrap) {
            removeTyping();
            botWrap = appendBotShell();
          }
          answer += event.text || "";
          updateBotShell(botWrap, answer);
        } else if (event.type === "done") {
          if (!botWrap) {
            removeTyping();
            botWrap = appendBotShell();
          }
          answer = event.text || answer;
          updateBotShell(botWrap, answer);
          finalizeBotShell(botWrap, {
            scope: event.scope,
            ttft_ms: event.ttft_ms,
            total_ms: event.total_ms,
            sources: event.sources,
          });
          renderTrace(event.retrieval_trace);
        } else if (event.type === "error") {
          removeTyping();
          appendBot(`**Error**\n\n${escapeHtml(event.message || "Unknown stream error")}`);
        }
      }
    }
  } catch (err) {
    removeTyping();
    appendBot(`**Network error**\n\n${err.message}`);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  ensureThreadShown();
  appendUser(text);
  inputEl.value = "";
  autosize();

  if (!localStorage.getItem("ubcse_personalize_asked")) {
    sendBtn.disabled = true;
    appendPersonalizePrompt(() => {
      sendMessage(text);
    });
    return;
  }

  await sendMessage(text);
});

inputEl.focus();
