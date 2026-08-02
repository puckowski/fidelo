const state = { jobs: [], activeJobId: null, jobSockets: new Map(), waveformToken: 0 };
const byId = (id) => document.getElementById(id);
const form = byId("generation-form");
const promptInput = byId("prompt");
const canvas = byId("waveform");
const context = canvas.getContext("2d");

if (window.lucide) window.lucide.createIcons();

function setRangeOutput(inputId, outputId, format) {
  const input = byId(inputId);
  const update = () => { byId(outputId).textContent = format(Number(input.value)); };
  input.addEventListener("input", update);
  update();
}

setRangeOutput("duration", "duration-value", (value) => `${value}s`);
setRangeOutput("source-strength", "source-strength-value", (value) => `${Math.round(value * 100)}%`);
setRangeOutput("temperature", "temperature-value", (value) => value.toFixed(2));
setRangeOutput("creativity", "creativity-value", (value) => `${Math.round(value * 100)}%`);
byId("duration").addEventListener("input", (event) => { byId("button-duration").textContent = `${event.target.value} SEC`; });
promptInput.addEventListener("input", () => { byId("char-count").textContent = promptInput.value.length; });
document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.textContent;
    promptInput.dispatchEvent(new Event("input"));
    promptInput.focus();
  });
});

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function payloadFromForm() {
  const seed = byId("seed").value.trim();
  return {
    prompt: promptInput.value.trim(),
    duration_seconds: Number(byId("duration").value),
    seed: seed ? Number(seed) : null,
    source_strength: Number(byId("source-strength").value),
    temperature: Number(byId("temperature").value),
    top_k: Number(byId("top-k").value),
    top_p: Number(byId("top-p").value),
    creativity: Number(byId("creativity").value),
    theme_seconds: Number(byId("theme-seconds").value),
    transition_seconds: Number(byId("transition-seconds").value),
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("generate-button");
  byId("form-error").textContent = "";
  button.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    state.activeJobId = job.id;
    state.jobs.unshift(job);
    renderJobs();
    showJob(job);
    watchJob(job.id);
  } catch (error) {
    byId("form-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function loadHealth() {
  try {
    const health = await api("/api/health");
    byId("status-dot").className = "status-dot online";
    byId("system-label").textContent = `${health.queue} · ${health.storage}`;
    byId("mode-tag").textContent = health.mode.toUpperCase();
    const config = await api("/api/config");
    byId("duration").max = config.max_duration_seconds;
  } catch (_) {
    byId("status-dot").className = "status-dot offline";
    byId("system-label").textContent = "API offline";
  }
}

async function loadJobs() {
  try {
    state.jobs = await api("/api/jobs");
    renderJobs();
    if (!state.activeJobId && state.jobs.length) {
      state.activeJobId = state.jobs[0].id;
      showJob(state.jobs[0]);
    }
    state.jobs
      .filter((job) => ["queued", "running"].includes(job.status))
      .forEach((job) => watchJob(job.id));
  } catch (error) {
    byId("job-list").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderJobs() {
  const list = byId("job-list");
  if (!state.jobs.length) {
    list.innerHTML = '<div class="empty-state">No generations yet</div>';
    return;
  }
  list.innerHTML = "";
  state.jobs.forEach((job, index) => {
    const row = byId("job-template").content.firstElementChild.cloneNode(true);
    row.classList.toggle("active", job.id === state.activeJobId);
    row.querySelector(".job-number").textContent = String(index + 1).padStart(2, "0");
    row.querySelector("strong").textContent = job.prompt;
    row.querySelector("small").textContent = `${job.duration_seconds}s · seed ${job.seed}`;
    row.querySelector(".job-status").className = `job-status ${job.status}`;
    row.querySelector(".job-status").title = job.status;
    row.addEventListener("click", () => {
      state.activeJobId = job.id;
      renderJobs();
      showJob(job);
    });
    list.appendChild(row);
  });
}

function showJob(job) {
  const activeIndex = Math.max(0, state.jobs.findIndex((item) => item.id === job.id));
  byId("track-index").textContent = `F / ${String(activeIndex + 1).padStart(3, "0")}`;
  byId("active-prompt").textContent = job.prompt;
  byId("active-meta").textContent = `${job.duration_seconds} SEC · SEED ${job.seed}`;
  const stateLabel = byId("render-state");
  stateLabel.textContent = job.status.toUpperCase();
  stateLabel.className = `render-state ${job.status === "queued" ? "idle" : job.status}`;
  byId("artwork").classList.toggle("rendering", ["queued", "running"].includes(job.status));
  byId("job-error").classList.toggle("hidden", job.status !== "failed");
  byId("job-error").textContent = job.error || "";

  const player = byId("audio-player");
  const download = byId("download-button");
  if (job.status === "completed" && job.audio_url) {
    byId("render-message").textContent = "Track complete";
    byId("render-detail").textContent = "WAV · READY TO PLAY";
    if (player.dataset.jobId !== job.id) {
      player.src = job.audio_url;
      player.dataset.jobId = job.id;
      drawAudioWaveform(job.audio_url);
    }
    download.href = job.download_url;
    download.classList.remove("hidden");
  } else {
    player.removeAttribute("src");
    player.dataset.jobId = "";
    player.load();
    download.classList.add("hidden");
    byId("render-message").textContent = job.status === "failed" ? "Generation failed" : job.status === "running" ? "Building your track" : "Waiting for a GPU";
    byId("render-detail").textContent = job.status === "running" ? "MODEL INFERENCE IN PROGRESS" : job.status.toUpperCase();
    drawIdleWaveform(job.seed);
  }
}

function watchJob(jobId) {
  if (state.jobSockets.has(jobId)) return;
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/api/jobs/${encodeURIComponent(jobId)}/events`);
  state.jobSockets.set(jobId, socket);

  socket.addEventListener("message", (event) => {
    const update = JSON.parse(event.data);
    const index = state.jobs.findIndex((job) => job.id === update.id);
    if (index < 0) return;
    state.jobs[index] = { ...state.jobs[index], ...update };
    renderJobs();
    if (update.id === state.activeJobId) showJob(state.jobs[index]);
  });
  socket.addEventListener("close", () => state.jobSockets.delete(jobId));
  socket.addEventListener("error", () => socket.close());
}

function fitCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

function drawBars(values) {
  const rect = fitCanvas();
  context.clearRect(0, 0, rect.width, rect.height);
  context.fillStyle = "#151713";
  const gap = 3;
  const width = Math.max(2, (rect.width - gap * values.length) / values.length);
  values.forEach((value, index) => {
    const height = Math.max(2, value * rect.height * .63);
    const x = index * (width + gap);
    context.fillRect(x, (rect.height - height) / 2, width, height);
  });
}

function drawIdleWaveform(seed = 17) {
  let value = Number(seed) || 17;
  const bars = Array.from({ length: 76 }, (_, index) => {
    value = (value * 16807) % 2147483647;
    return .08 + ((value / 2147483647) * .58) * (.65 + .35 * Math.sin(index * .22) ** 2);
  });
  drawBars(bars);
}

async function drawAudioWaveform(url) {
  const token = ++state.waveformToken;
  try {
    const response = await fetch(url);
    const data = await response.arrayBuffer();
    const audioContext = new AudioContext();
    const buffer = await audioContext.decodeAudioData(data);
    await audioContext.close();
    if (token !== state.waveformToken) return;
    const samples = buffer.getChannelData(0);
    const count = 88;
    const block = Math.max(1, Math.floor(samples.length / count));
    const bars = Array.from({ length: count }, (_, index) => {
      let peak = 0;
      for (let sample = index * block; sample < Math.min(samples.length, (index + 1) * block); sample += 16) peak = Math.max(peak, Math.abs(samples[sample]));
      return Math.max(.03, peak);
    });
    drawBars(bars);
  } catch (_) {
    drawIdleWaveform();
  }
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

byId("refresh-button").addEventListener("click", loadJobs);
window.addEventListener("resize", () => {
  const job = state.jobs.find((item) => item.id === state.activeJobId);
  drawIdleWaveform(job?.seed);
  if (job?.audio_url) drawAudioWaveform(job.audio_url);
});

drawIdleWaveform();
loadHealth();
loadJobs();
