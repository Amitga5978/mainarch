const state = {
  messages: [],
  status: null,
  scenario: "open prompt",
};

const $ = (id) => document.getElementById(id);
const els = {
  backendMode: $("backendMode"),
  heroBackend: $("heroBackend"),
  heroModel: $("heroModel"),
  heroProof: $("heroProof"),
  deployHint: $("deployHint"),
  perfHeadline: $("perfHeadline"),
  modelName: $("modelName"),
  statusDot: $("statusDot"),
  perfGrid: $("perfGrid"),
  winNote: $("winNote"),
  transcript: $("transcript"),
  prompt: $("prompt"),
  send: $("send"),
  form: $("chatForm"),
  runState: $("runState"),
  runTimer: $("runTimer"),
  tokenCount: $("tokenCount"),
  receipt: $("receipt"),
  flightLog: $("flightLog"),
  copyCurl: $("copyCurl"),
  curlBlock: $("curlBlock"),
  examples: document.querySelectorAll("[data-example]"),
};

let runTicker = null;
let runStarted = 0;
let streamedChunks = 0;
let firstChunkAt = null;

function escapeHtml(input) {
  return String(input)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value >= 1000) return value.toLocaleString();
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function fmtMs(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 100) return `${value.toFixed(0)} ms`;
  if (value >= 10) return `${value.toFixed(1)} ms`;
  return `${value.toFixed(2)} ms`;
}

function modeBoundary(mode) {
  if (mode === "openai") return "upstream-backed text";
  if (mode === "command") return "local command backend";
  return "synthetic preview text";
}

function setStatus(status) {
  state.status = status;
  const label = status.mode_label || status.mode || "unknown";
  const model = status.model || "mainarch-preview";
  const measurements = status.perf?.measurements || [];
  const publicUrl = status.public_url || "http://HOST:8080";

  els.backendMode.textContent = label;
  els.heroBackend.textContent = status.backend_ready ? label : `${label} not configured`;
  els.heroModel.textContent = model;
  els.heroProof.textContent = measurements.length ? `${measurements.length} proof cards` : "no export";
  els.modelName.textContent = model;
  els.statusDot.dataset.mode = status.mode || "scripted";
  els.perfHeadline.textContent = status.perf?.headline || "What the sandbox can prove today";
  els.winNote.textContent = status.perf?.win || "Perf note will appear here when a bench export is attached.";
  els.deployHint.textContent = status.public_url
    ? `Public sandbox: ${status.public_url}`
    : "Container default is scripted preview; set MAINARCH_DEMO_MODE=openai or command to attach a real backend.";
  els.curlBlock.textContent = `curl ${publicUrl}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{"model":"${model}","stream":true,"messages":[{"role":"user","content":"why is this fast?"}]}'`;
  renderPerf(measurements);
}

function renderPerf(items) {
  els.perfGrid.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("article");
    empty.className = "metric";
    empty.innerHTML = `<div class="metric-top"><span>No perf export</span><strong>pending</strong></div><div class="metric-values"><b>-</b><span>Attach MAINARCH_DEMO_PERF_JSON</span></div>`;
    els.perfGrid.appendChild(empty);
    return;
  }

  for (const item of items) {
    const mainarch = Number(item.mainarch);
    const baseline = Number(item.baseline);
    const lowerIsBetter = mainarch > 0 && baseline > 0 && mainarch < baseline;
    const ratio = mainarch > 0 && baseline > 0 ? baseline / mainarch : 0;
    const pct = ratio > 0 ? Math.min(100, Math.max(8, (1 / ratio) * 100)) : 100;
    const label = ratio ? (lowerIsBetter ? `${ratio.toFixed(2)}x lower` : `${ratio.toFixed(2)}x ratio`) : "live";

    const card = document.createElement("article");
    card.className = "metric";
    card.innerHTML = `
      <div class="metric-top">
        <span>${escapeHtml(item.name || "Measurement")}</span>
        <strong>${escapeHtml(label)}</strong>
      </div>
      <div class="metric-values">
        <b>${fmt(mainarch)} ${escapeHtml(item.unit || "")}</b>
        <span>vs ${fmt(baseline)} ${escapeHtml(item.unit || "")}</span>
      </div>
      <div class="bar" aria-hidden="true"><i style="width:${pct}%"></i></div>
      <small>${escapeHtml(item.source || "local")}</small>
    `;
    els.perfGrid.appendChild(card);
  }
}

function addFlight(label, detail) {
  const item = document.createElement("li");
  item.innerHTML = `<b>${escapeHtml(label)}</b><span>${escapeHtml(detail)}</span>`;
  els.flightLog.prepend(item);
  while (els.flightLog.children.length > 7) {
    els.flightLog.lastElementChild.remove();
  }
}

function resetFlight() {
  els.flightLog.innerHTML = "";
  addFlight("route", "/v1/chat/completions stream=true");
}

function setReceipt(title, body, items, variant = "") {
  els.receipt.classList.toggle("done", variant === "done");
  els.receipt.classList.toggle("error", variant === "error");
  els.receipt.textContent = "";

  const label = document.createElement("span");
  label.textContent = "demo receipt";
  const heading = document.createElement("b");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = body;
  const list = document.createElement("dl");

  for (const [key, value] of items) {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    row.appendChild(dt);
    row.appendChild(dd);
    list.appendChild(row);
  }

  els.receipt.appendChild(label);
  els.receipt.appendChild(heading);
  els.receipt.appendChild(copy);
  els.receipt.appendChild(list);
}

function appendMessage(role, text = "") {
  const node = document.createElement("section");
  node.className = `message ${role}`;
  node.innerHTML = `<div class="role">${escapeHtml(role)}</div><p></p>`;
  node.querySelector("p").textContent = text;
  els.transcript.appendChild(node);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return node.querySelector("p");
}

function appendToken(target, text) {
  target.textContent += text;
  streamedChunks += 1;
  els.tokenCount.textContent = `${streamedChunks} chunk${streamedChunks === 1 ? "" : "s"}`;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function setRunState(label) {
  els.runState.textContent = label;
}

function startRunClock() {
  runStarted = performance.now();
  firstChunkAt = null;
  streamedChunks = 0;
  els.tokenCount.textContent = "0 chunks";
  els.runTimer.textContent = "0.00s";
  setRunState("streaming");
  if (runTicker) clearInterval(runTicker);
  runTicker = setInterval(() => {
    const seconds = (performance.now() - runStarted) / 1000;
    els.runTimer.textContent = `${seconds.toFixed(2)}s`;
  }, 80);
}

function stopRunClock(label = "idle") {
  if (runTicker) {
    clearInterval(runTicker);
    runTicker = null;
  }
  if (runStarted) {
    const seconds = (performance.now() - runStarted) / 1000;
    els.runTimer.textContent = `${seconds.toFixed(2)}s`;
  }
  setRunState(label);
}

async function loadStatus() {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error(`status ${res.status}`);
  setStatus(await res.json());
}

function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const rawLine of block.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    const idx = rawLine.indexOf(":");
    const field = idx >= 0 ? rawLine.slice(0, idx) : rawLine;
    const value = idx >= 0 ? rawLine.slice(idx + 1).trimStart() : "";
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  }
  return { event, data: data.join("\n") };
}

function handlePayload(packet, answer) {
  if (!packet.data || packet.data === "[DONE]") return;
  const payload = JSON.parse(packet.data);
  if (packet.event === "meta") {
    addFlight("backend", `${payload.mode || "mode"} ${payload.model || "model"}`);
    setRunState(`streaming ${payload.model || "backend"}`);
  }

  const choice = payload.choices && payload.choices[0];
  const delta = choice && choice.delta;
  const message = choice && choice.message;
  const text = (delta && delta.content) || (message && message.content) || payload.text || "";

  if (text) {
    if (firstChunkAt === null) {
      firstChunkAt = performance.now();
      addFlight("first chunk", fmtMs(firstChunkAt - runStarted));
    }
    appendToken(answer, text);
  }

  if (payload.mainarch) {
    const mode = payload.mainarch.demo_mode || payload.mainarch.mode || "mainarch metadata";
    addFlight("metadata", `${mode}, proof=${payload.mainarch.proof || "attached"}`);
    setRunState(`streaming ${mode}`);
  }

  if (payload.error || packet.event === "error") {
    const detail = payload.error?.message || payload.error || "Backend error";
    appendToken(answer, `\n\n${detail}`);
    addFlight("error", String(detail).slice(0, 120));
  }

  if (choice?.finish_reason) {
    addFlight("finish", choice.finish_reason);
    setRunState(choice.finish_reason === "stop" ? "complete" : choice.finish_reason);
  }
}

async function sendPrompt(prompt) {
  const trimmed = prompt.trim();
  if (!trimmed) return;

  els.prompt.value = "";
  els.send.disabled = true;
  appendMessage("user", trimmed);
  const answer = appendMessage("assistant", "");
  startRunClock();
  resetFlight();

  const mode = state.status?.mode || "scripted";
  const model = state.status?.model || "mainarch-preview";
  setReceipt(
    "Prompt entered the serving seam.",
    "The browser is calling the OpenAI-shaped route that the real Qwen/Kimi backend will keep.",
    [
      ["scenario", state.scenario],
      ["route", "/v1/chat/completions"],
      ["stream", "true"],
      ["backend", state.status?.mode_label || mode],
      ["model", model],
      ["boundary", modeBoundary(mode)]
    ]
  );

  state.messages.push({ role: "user", content: trimmed });
  addFlight("dispatch", state.scenario);

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        messages: state.messages,
      }),
    });

    if (!res.ok || !res.body) throw new Error(`chat ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(/\n\n/);
      buffer = parts.pop() || "";
      for (const part of parts) {
        handlePayload(parseSseBlock(part), answer);
      }
    }

    state.messages.push({ role: "assistant", content: answer.textContent });
    const elapsed = performance.now() - runStarted;
    const ttfb = firstChunkAt === null ? null : firstChunkAt - runStarted;
    setReceipt(
      "Sandbox stream completed.",
      mode === "scripted"
        ? "The product surface, stream parser, receipt, and proof rail ran. The answer text is synthetic in this safe preview mode."
        : "The product surface streamed through the configured backend without changing the route contract.",
      [
        ["scenario", state.scenario],
        ["route", "/v1/chat/completions"],
        ["backend", state.status?.mode_label || mode],
        ["client first chunk", fmtMs(ttfb)],
        ["client total", fmtMs(elapsed)],
        ["chunks", String(streamedChunks)],
        ["claim", modeBoundary(mode)]
      ],
      "done"
    );
    addFlight("receipt", "completed proof summary");
    stopRunClock("complete");
  } catch (err) {
    appendToken(answer, `The demo backend returned an error: ${err.message}`);
    setReceipt(
      "Sandbox route returned an error.",
      "The page exposes backend failure plainly so the demo does not overclaim.",
      [
        ["route", "/v1/chat/completions"],
        ["scenario", state.scenario],
        ["error", err.message.slice(0, 160)]
      ],
      "error"
    );
    addFlight("error", err.message.slice(0, 120));
    stopRunClock("error");
  } finally {
    els.send.disabled = false;
    els.prompt.focus();
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendPrompt(els.prompt.value);
});

for (const button of els.examples) {
  button.addEventListener("click", () => {
    state.scenario = button.dataset.scenario || "example prompt";
    els.prompt.value = button.dataset.example || "";
    els.prompt.focus();
    addFlight("scenario", state.scenario);
  });
}

els.copyCurl.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(els.curlBlock.textContent);
    els.copyCurl.textContent = "Copied";
    setTimeout(() => { els.copyCurl.textContent = "Copy curl"; }, 1200);
  } catch (_err) {
    els.copyCurl.textContent = "Select curl";
    setTimeout(() => { els.copyCurl.textContent = "Copy curl"; }, 1200);
  }
});

appendMessage(
  "assistant",
  "This is the mainarch sandbox shell. Pick a scenario or ask directly. The safest default is scripted preview; the important part is that every run uses the same /v1/chat/completions route that a real Qwen/Kimi backend will keep."
);

loadStatus().catch((err) => {
  els.backendMode.textContent = "status error";
  els.winNote.textContent = err.message;
  addFlight("status error", err.message);
});
