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
  inputEl.focus();
});

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

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  ensureThreadShown();
  appendUser(text);
  inputEl.value = "";
  autosize();
  sendBtn.disabled = true;
  appendTyping();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, user_id: userId }),
    });
    removeTyping();
    if (!res.ok) {
      const err = await res.text();
      appendBot(`**Error ${res.status}**\n\n${err.slice(0, 600)}`);
      return;
    }
    const data = await res.json();
    appendBot(data.text, {
      scope: data.scope,
      ttft_ms: data.ttft_ms,
      total_ms: data.total_ms,
      sources: data.sources,
    });
    renderTrace(data.retrieval_trace);
  } catch (err) {
    removeTyping();
    appendBot(`**Network error**\n\n${err.message}`);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
});

inputEl.focus();
