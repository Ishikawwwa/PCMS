"use strict";

const ACTIVE_ESTIMATE = new Set(["running", "resuming", "starting"]);

function eventTime(event) { return Date.parse(event.at); }

function activeTime(start, end, events) {
  let pausedAt = null;
  let pausedFor = 0;
  for (const event of events) {
    const time = eventTime(event);
    if (time < start || time > end || event.kind !== "control") continue;
    if (event.title === "Paused at checkpoint") pausedAt ??= time;
    if (event.title === "Continued from checkpoint" && pausedAt !== null) {
      pausedFor += time - pausedAt;
      pausedAt = null;
    }
  }
  return pausedAt === null ? Math.max(0, end - start - pausedFor) : null;
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function remainingForCurrent(typical, elapsed) {
  // One slow item should not move the estimate into the past.
  return Math.max(typical - (elapsed ?? 0), typical * 0.15);
}

function pairedEstimate(manifest, events, now) {
  const arms = manifest.conditions || ["mage", "mage_plus_concepts"];
  const progress = new Map(arms.map((arm) => [arm, {started: null, next: null, completed: new Set(), samples: []}]));
  let current = null;
  for (const event of events) {
    const arm = progress.get(event.condition);
    if (!arm) continue;
    const time = eventTime(event);
    if (event.title === "Comparison arm started") {
      arm.started = arm.next = time;
      current = arm;
    } else if (["Traveler turn ended", "PCMS traveler turn ended"].includes(event.title) && !arm.completed.has(event.turn)) {
      if (arm.next !== null) {
        const duration = activeTime(arm.next, time, events);
        if (duration > 0) arm.samples.push(duration);
      }
      arm.next = time;
      arm.completed.add(event.turn);
    }
  }
  const sharedPace = mean([...progress.values()].flatMap((arm) => arm.samples));
  if (sharedPace === null) return {state: "estimating"};
  let remaining = 0;
  let rough = false;
  for (const arm of progress.values()) {
    const turns = Math.max(0, manifest.turns - arm.completed.size);
    if (!turns) continue;
    const pace = mean(arm.samples);
    if (pace === null) rough = true;
    const typical = pace ?? sharedPace;
    remaining += turns * typical;
    if (arm === current && arm.next !== null) {
      remaining += remainingForCurrent(typical, activeTime(arm.next, now, events)) - typical;
    }
  }
  return remaining > 0 ? {state: "estimated", finish: now + remaining, remaining, rough} : {state: "finishing"};
}

function diagnosticEstimate(manifest, events, now) {
  const expected = manifest.planned_model_calls || {shopping: {write: 5, read: 1, act: 0},
    travel: {write: 5, read: 2, act: 8}, all: {write: 10, read: 3, act: 8}}[manifest.case];
  if (!expected) return {state: "estimating"};
  const samples = {write: [], read: [], act: []};
  const completed = {write: 0, read: 0, act: 0};
  let pending = null;
  for (const event of events) {
    const operation = event.scope?.turn;
    if (!Object.hasOwn(expected, operation)) continue;
    if (event.kind === "prompt") pending = event;
    if (event.kind !== "response") continue;
    completed[operation] += 1;
    if (pending?.scope.turn === operation) {
      const duration = activeTime(eventTime(pending), eventTime(event), events);
      if (duration > 0) samples[operation].push(duration);
      pending = null;
    }
  }
  const sharedPace = mean(Object.values(samples).flat());
  if (sharedPace === null) return {state: "estimating"};
  let remaining = 0;
  let rough = false;
  for (const operation of Object.keys(expected)) {
    const calls = Math.max(0, expected[operation] - completed[operation]);
    if (!calls) continue;
    const pace = mean(samples[operation]);
    if (pace === null) rough = true;
    const typical = pace ?? sharedPace;
    remaining += calls * typical;
    if (pending?.scope.turn === operation) {
      remaining += remainingForCurrent(typical, activeTime(eventTime(pending), now, events)) - typical;
    }
  }
  return remaining > 0 ? {state: "estimated", finish: now + remaining, remaining, rough} : {state: "finishing"};
}

function estimateFinish(run, events, now = Date.now(), benchmark = null) {
  if (!ACTIVE_ESTIMATE.has(run.status)) return null;
  const manifest = run.manifest;
  if (manifest.benchmark) {
    const arms = benchmark?.splits?.all?.arms;
    const arm = manifest.conditions?.find((condition) => arms?.[condition]);
    const scored = arms?.[arm]?.travelers_scored ?? 0;
    const elapsed = now - Date.parse(manifest.started);
    const pilotPace = manifest.pilot_seconds_per_traveler * 1000 / manifest.workers;
    const pace = scored >= manifest.workers * 7 ? elapsed / scored : pilotPace;
    const remaining = (manifest.travelers - scored) * pace;
    if (!Number.isFinite(remaining)) return {state: "estimating"};
    return remaining > 0 ? {state: "estimated", finish: now + remaining, remaining, rough: true} : {state: "finishing"};
  }
  if (manifest.turns) return pairedEstimate(manifest, events, now);
  return diagnosticEstimate(manifest, events, now);
}

if (typeof module !== "undefined") module.exports = {estimateFinish, summarizeUsage};

const $ = (id) => document.getElementById(id);
const terminal = new Set(["complete", "failed", "stopped", "terminated"]);
const state = {
  config: null, runs: [], selected: null, tab: "timeline", events: [], cursor: 0,
  detail: null, detailSequence: null, raw: false, files: [], file: null, page: null,
  latestPage: true, polling: false, busy: false, generation: 0, runListKey: "", fileListKey: "",
  resultData: null, resultLoading: false, benchmark: null, workers: new Map(),
};
const views = ["timeline", "results", "files", "manifest"];
const statusNames = {
  complete: "Complete", failed: "Failed", running: "Running", stopped: "Stopped", paused: "Paused",
  terminated: "Terminated", starting: "Starting", pause_requested: "Pause requested",
  stop_requested: "Stop requested", executing_step: "Stepping", resuming: "Resuming", archived: "Archived",
};
const labels = {
  shopping: "Task 24 · Shopping retrieval", travel: "Travel · Confirmation and revision", paired: "Travel · MAGE vs. PCMS",
};
const breakpoints = {model: "model request", memory: "memory commit", retrieval: "context packing",
  tool: "tool action", score: "scoring", arm: "comparison arm"};

function node(tag, text = "", className = "") {
  const value = document.createElement(tag);
  value.textContent = text;
  if (className) value.className = className;
  return value;
}

function button(text, handler, className = "") {
  const value = node("button", text, className);
  value.type = "button";
  value.addEventListener("click", handler);
  return value;
}

function pretty(value) { return JSON.stringify(value, null, 2); }
function info() { return state.runs.find((run) => run.id === state.selected); }
function displayStatus(run) { return statusNames[run.status] || run.status; }

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function get(path, data) { return api(`/api/${path}?${new URLSearchParams(data)}`); }
function post(path, data) {
  return api(`/api/${path}`, {
    method: "POST", headers: {"Content-Type": "application/json", "X-Lab-Control": state.config.token},
    body: JSON.stringify(data),
  });
}

function notice(text) {
  $("notice").textContent = text || "";
  $("notice").hidden = !text;
}

function name(run) {
  const manifest = run.manifest;
  if (manifest.benchmark || run.id.includes("-lab-benchmark-")) return manifest.conditions?.length === 1 ?
    "Travel benchmark · PCMS" : "Travel benchmark · MAGE vs. PCMS";
  if (manifest.group) return manifest.comparison === "official_mage_vs_standalone_pcms" ?
    `MAGE vs. PCMS · Group ${manifest.group}` : `MAGE augmentation pilot · Group ${manifest.group}`;
  if (manifest.case === "all") return "Shopping + travel";
  if (manifest.case) return labels[manifest.case] || manifest.case;
  const experiment = run.id.match(/-lab-(shopping|travel|paired)-/)?.[1];
  return labels[experiment] || run.id.replace("concept-subspaces-", "");
}

function renderRuns() {
  const key = JSON.stringify([state.selected, state.runs.map((run) => [run.id, run.status, run.updated, name(run)])]);
  if (key === state.runListKey) return;
  state.runListKey = key;
  const fragment = document.createDocumentFragment();
  for (const run of state.runs) {
    const selected = run.id === state.selected;
    const item = button("", () => selectRun(run.id), `run-item${selected ? " active" : ""}`);
    item.title = run.id;
    item.setAttribute("aria-current", selected ? "true" : "false");
    item.append(node("strong", name(run)));
    const metadata = node("div", "", "run-meta");
    const date = run.updated ? new Date(run.updated * 1000) : null;
    metadata.append(node("span", date ? date.toLocaleDateString([], {month: "short", day: "numeric"}) + " " +
      date.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "Launching"), node("span", displayStatus(run)));
    item.append(metadata);
    fragment.append(item);
  }
  if (!state.runs.length) fragment.append(node("p", "No experiment runs yet.", "empty-list"));
  $("run-list").replaceChildren(fragment);
}

function renderHeader() {
  const run = info();
  if (!run) return;
  const m = run.manifest;
  $("run-title").textContent = name(run);
  $("run-title").title = run.id;
  $("run-status").textContent = displayStatus(run);
  $("run-status").dataset.status = run.status;
  const parts = [];
  if (m.memory_tokens) parts.push(`${m.memory_tokens} memory tokens`);
  if (m.turns) parts.push(`${m.turns} of ${m.full_turns} travelers`);
  if (m.benchmark) {
    const progress = state.benchmark?.splits?.all;
    const groups = state.benchmark?.groups;
    parts.push(`${progress?.paired ?? 0} / ${m.groups} groups scored`);
    if (groups && !terminal.has(run.status)) {
      const active = groups.filter((group) => group.status === "running").length;
      parts.push(`${active} / ${m.workers} workers active`);
      const checks = m.repair_check_groups || [];
      const completed = groups.filter((group) => checks.includes(group.group) && group.status === "complete").length;
      if (completed < checks.length) {
        parts.push(`Repair checks ${completed} / ${checks.length} complete · remaining groups start after the checks pass`);
      }
    }
    if (progress?.failed) parts.push(`${progress.failed} failed`);
  }
  if (m.route?.only?.length) parts.push(m.route.only.join(", "));
  if (run.status === "paused") parts.push(`Next: ${run.checkpoint.operation}`);
  if (run.command?.mode === "run_to" && run.status !== "paused") parts.push(`Run to ${breakpoints[run.command.target]}`);
  if (run.status === "pause_requested" || run.status === "stop_requested") parts.push("Waiting for the next operation boundary");
  if (!run.managed) parts.push("Read-only · launched outside this console");
  if (terminal.has(run.status)) parts.push("Run ended");
  $("run-summary").textContent = parts.join("  ·  ") || run.id;
  renderEta(run);
  const canControl = run.managed && !terminal.has(run.status) && run.status !== "stop_requested";
  $("control-bar").hidden = false;
  $("control-bar").title = !run.managed ? "This run was launched outside the current console." :
    terminal.has(run.status) ? "This run has ended." : "";
  $("resume").disabled = !canControl || state.busy || !["paused", "pause_requested", "executing_step"].includes(run.status);
  $("pause").disabled = !canControl || state.busy || !["running", "resuming", "executing_step"].includes(run.status);
  $("step").disabled = !canControl || state.busy || m.benchmark || run.status !== "paused";
  $("run-to").disabled = !canControl || state.busy || m.benchmark || !["paused", "running"].includes(run.status);
  $("breakpoint").disabled = !canControl || Boolean(m.benchmark);
  const paired = Boolean(m.group) || run.id.includes("-lab-paired-");
  for (const option of $("breakpoint").querySelectorAll("[data-paired]")) option.disabled = !paired;
  if ($("breakpoint").selectedOptions[0]?.disabled) $("breakpoint").value = "model";
  $("stop").disabled = !canControl || state.busy;
  $("terminate").disabled = !run.managed || terminal.has(run.status) || state.busy;
  if (state.tab === "manifest") renderManifest();
}

function renderEta(run) {
  const eta = estimateFinish(run, state.events, Date.now(), state.benchmark);
  const label = $("run-eta");
  label.hidden = eta === null;
  if (!eta) return;
  if (eta.state === "estimating") {
    label.textContent = "Expected finish · Waiting for enough progress to estimate";
  } else if (eta.state === "finishing") {
    label.textContent = "Expected finish · Finalizing";
  } else {
    const finish = new Date(eta.finish);
    const time = finish.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
    const date = finish.toDateString() !== new Date().toDateString() ?
      `${finish.toLocaleDateString([], {month: "short", day: "numeric"})}, ` : "";
    const minutes = Math.ceil(eta.remaining / 60000);
    const duration = minutes >= 120 ? `about ${Math.round(minutes / 60)} hr` :
      eta.remaining < 60000 ? "under 1 min" : `about ${minutes} min`;
    label.textContent = `Expected finish · ${date}${time} (${duration}${eta.rough ? ", rough estimate" : ""})`;
  }
}

function showTab(tab) {
  state.tab = tab;
  $("metrics").hidden = tab === "results";
  for (const id of views) {
    $(id).hidden = id !== tab;
    $(`${id}-tab`).classList.toggle("selected", id === tab);
    $(`${id}-tab`).setAttribute("aria-pressed", String(id === tab));
  }
  if (tab === "manifest") renderManifest();
  if (tab === "files") void refreshFiles().catch((error) => notice(error.message));
  if (tab === "timeline") renderEvents();
  if (tab === "results") {
    renderResults();
    void refreshResults();
  }
}

function selectRun(run) {
  if (state.selected === run) return;
  state.generation += 1;
  state.selected = run;
  state.cursor = 0;
  state.events = [];
  state.files = [];
  state.file = null;
  state.page = null;
  state.detail = null;
  state.detailSequence = null;
  state.raw = false;
  state.fileListKey = "";
  state.resultData = null;
  state.resultLoading = false;
  state.benchmark = null;
  state.workers.clear();
  $("file-list").replaceChildren();
  $("follow").checked = true;
  $("search").value = "";
  $("kind").value = "";
  renderDetail();
  renderFile();
  renderHeader();
  renderRuns();
  const params = new URLSearchParams({run});
  history.replaceState(null, "", `?${params}`);
  const selected = info();
  showTab(selected?.manifest.benchmark || terminal.has(selected?.status) ? "results" :
    selected?.timeline || selected?.managed ? "timeline" : "files");
  if (!state.polling) void sync();
}

function eventLabel(event) {
  const scope = event.scope || {};
  return [event.condition || scope.group, event.mode || scope.operation,
    typeof event.turn === "number" ? `turn ${event.turn + 1}` : scope.turn, event.evidence_id].filter(Boolean).join(" · ");
}

function visibleEvents() {
  const query = $("search").value.trim().toLowerCase();
  const kind = $("kind").value;
  return state.events.filter((event) => (!kind || event.kind === kind) &&
    `${event.seq} ${event.title} ${event.kind} ${eventLabel(event)}`.toLowerCase().includes(query));
}

function renderEvents() {
  const list = $("events");
  const scroll = list.scrollTop;
  const visible = visibleEvents();
  const fragment = document.createDocumentFragment();
  for (const event of visible) {
    const item = button("", () => { $("follow").checked = false; void selectEvent(event); }, "event");
    item.dataset.seq = event.seq;
    item.classList.toggle("active", event.seq === state.detailSequence);
    const title = node("span", "", "event-line");
    title.append(node("span", `#${event.seq}`, "sequence"), node("strong", event.title));
    const meta = node("span", "", "event-meta");
    const tag = node("span", event.kind, "event-tag");
    tag.dataset.kind = event.kind;
    meta.append(tag, node("span", eventLabel(event)), node("span", event.at ? new Date(event.at).toLocaleTimeString() : "", "event-time"));
    item.append(title, meta);
    fragment.append(item);
  }
  if (!visible.length) {
    const message = state.events.length ? "No events match the filters." : info()?.managed ?
      "Waiting for the worker’s first checkpoint." : "This run predates the timeline. Inspect its recorded artifacts.";
    fragment.append(node("p", message, "empty-list"));
  }
  list.replaceChildren(fragment);
  list.scrollTop = $("follow").checked ? list.scrollHeight : scroll;
  $("event-count").textContent = `${visible.length} of ${state.events.length} events`;
  const usage = state.events.filter((event) => event.kind === "usage");
  const tokens = usage.reduce((total, event) => total + (event.tokens || 0), 0);
  const billed = usage.filter((event) => typeof event.cost === "number");
  const cost = billed.reduce((total, event) => total + event.cost, 0);
  const billing = billed.length ? `${cost.toFixed(4)} RUB${billed.length === usage.length ? "" : " known"}` : "";
  $("metrics").textContent = usage.length ? [`${usage.length} request${usage.length === 1 ? "" : "s"}`, `${tokens.toLocaleString()} tokens`, billing].filter(Boolean).join(" · ") : "";
}

async function selectEvent(event) {
  const generation = state.generation;
  state.detailSequence = event.seq;
  for (const item of $("events").querySelectorAll(".event")) item.classList.toggle("active", Number(item.dataset.seq) === event.seq);
  try {
    const detail = await get("event", {run: state.selected, offset: event.offset});
    if (generation !== state.generation || state.detailSequence !== event.seq) return;
    state.detail = detail;
    renderDetail();
  } catch (error) { notice(error.message); }
}

function block(label, value, collapsible = false) {
  const section = node("section", "", "detail-block");
  const body = node("pre", typeof value === "string" ? value : pretty(value));
  if (collapsible) {
    const disclosure = node("details");
    disclosure.append(node("summary", label), body);
    section.append(disclosure);
  } else section.append(node("h4", label), body);
  return section;
}

function extractionCards(memories) {
  const section = node("section", "", "detail-block");
  section.append(node("h4", `Extracted memories · ${memories.length}`));
  for (const memory of memories) {
    const card = node("div", "", "memory-card");
    const status = node("span", memory.status, "pill");
    status.dataset.status = memory.status === "confirmed" ? "complete" : "paused";
    const metadata = node("div", "", "memory-meta");
    metadata.append(status, node("code", memory.id));
    card.append(metadata, node("p", memory.text));
    const scope = [memory.concepts?.join(" · "), memory.table && `Row: ${memory.table} / ${memory.row}`, `Source: ${memory.source}`].filter(Boolean);
    card.append(node("div", scope.join("  |  "), "subtle"));
    section.append(card);
  }
  return section;
}

function renderDetail() {
  const target = $("detail");
  const entry = state.detail;
  $("raw").hidden = $("copy-event").hidden = !entry;
  $("detail-title").textContent = entry ? `#${entry.seq} · ${entry.title}` : "Event inspector";
  target.classList.toggle("empty", !entry);
  if (!entry) {
    target.textContent = "Select an event to inspect its full contents.";
    return;
  }
  $("raw").textContent = state.raw ? "Structured view" : "Raw JSON";
  $("raw").setAttribute("aria-pressed", String(state.raw));
  target.replaceChildren(node("p", `${entry.kind} · ${new Date(entry.at).toLocaleString()}`, "detail-meta"));
  if (state.raw) {
    target.append(block("Event", entry));
    return;
  }
  const data = entry.data;
  const links = node("div", "", "linked");
  for (const key of ["request", "action", "candidate", "checkpoint"]) {
    if (typeof data[key] !== "number") continue;
    const event = state.events.find((row) => row.seq === data[key]);
    if (event) links.append(button(`${key} #${data[key]}`, () => { $("follow").checked = false; void selectEvent(event); }));
  }
  if (links.childElementCount) target.append(links);
  if (entry.kind === "prompt" && Array.isArray(data.payload?.messages)) {
    const {messages, ...settings} = data.payload;
    for (const message of messages) target.append(block(message.role, message.content));
    target.append(block("Model request parameters", settings), block("Scope", data.scope));
  } else if (entry.kind === "response" && data.response?.choices?.[0]?.message) {
    const response = data.response;
    const message = response.choices[0].message;
    if (message.content) target.append(block("Model output", message.content));
    if (message.reasoning_content || message.reasoning) target.append(block("Provider-returned reasoning", message.reasoning_content || message.reasoning, true));
    target.append(block("Full response", response, true));
  } else {
    const emphasized = ["observation", "context", "error", "memories", "aliases", "scope", "evidence", "environment", "catalog", "current_rows",
      "ranked_candidates", "packing_decisions", "nodes", "summaries", "raw_context", "stored_context"];
    const remainder = {...data};
    for (const key of emphasized) {
      if (!(key in remainder)) continue;
      const value = remainder[key];
      if (key === "memories" && Array.isArray(value)) target.append(extractionCards(value));
      else if (value !== null) target.append(block(key.replaceAll("_", " "), value,
        ["environment", "catalog", "current_rows", "ranked_candidates", "packing_decisions", "nodes", "summaries", "raw_context", "stored_context"].includes(key)));
      delete remainder[key];
    }
    if (Object.keys(remainder).length) target.append(block("Details", remainder));
  }
}

async function refreshEvents() {
  if (!state.selected || !info()?.timeline) return;
  const generation = state.generation;
  const run = state.selected;
  const completed = terminal.has(info().status);
  let batch;
  let added = 0;
  do {
    batch = await get("events", {run, after: state.cursor});
    if (generation !== state.generation) return;
    state.cursor = batch.next;
    state.events.push(...batch.events);
    added += batch.events.length;
  } while (completed && batch.events.length === 200);
  if (!added) return;
  renderEta(info());
  if (state.tab === "results" && state.resultData) renderResults();
  renderEvents();
  if ($("follow").checked && state.tab === "timeline") {
    const current = visibleEvents().at(-1);
    if (current) await selectEvent(current);
  }
}

function formatBytes(bytes) {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}

function renderFiles() {
  const key = JSON.stringify([state.file, state.files]);
  if (key === state.fileListKey) return;
  state.fileListKey = key;
  const fragment = document.createDocumentFragment();
  for (const file of state.files) {
    const item = button("", () => { void selectFile(file.path); }, `artifact${file.path === state.file ? " active" : ""}`);
    item.title = file.path;
    item.append(node("strong", file.path === "@worker.log" ? "Worker console log" : file.path),
                node("span", formatBytes(file.bytes), "artifact-size"));
    fragment.append(item);
  }
  if (!state.files.length) fragment.append(node("p", "No recorded files yet.", "empty-list"));
  $("file-list").replaceChildren(fragment);
}

function renderFile() {
  const page = state.page;
  const detail = $("file-detail");
  $("file-title").textContent = state.file || "Artifact inspector";
  $("older").hidden = !page || page.start === 0;
  $("latest").hidden = !page || state.latestPage;
  $("copy-file").hidden = !page;
  detail.classList.toggle("empty", !page);
  if (!page) { detail.textContent = "Select an artifact to inspect its contents."; return; }
  let content = page.text;
  if (page.start === 0 && page.end === page.size && state.file.endsWith(".json")) {
    try { content = pretty(JSON.parse(content)); } catch { /* The writer may still be finishing. */ }
  }
  detail.replaceChildren(node("p", `Bytes ${page.start.toLocaleString()}–${page.end.toLocaleString()} of ${page.size.toLocaleString()}`, "detail-meta"), block("Recorded text", content));
}

async function selectFile(path, end) {
  const generation = state.generation;
  state.file = path;
  state.latestPage = end === undefined;
  renderFiles();
  try {
    const parameters = {run: state.selected, path};
    if (end !== undefined) parameters.end = end;
    const page = await get("file", parameters);
    if (generation !== state.generation || path !== state.file) return;
    state.page = page;
    renderFile();
  } catch (error) { notice(error.message); }
}

async function refreshFiles() {
  if (!state.selected) return;
  const generation = state.generation;
  const files = await get("files", {run: state.selected});
  if (generation !== state.generation) return;
  state.files = files;
  renderFiles();
  if (!state.file) {
    const first = ["results.jsonl", "model_calls.jsonl", "manifest.json", "@worker.log"].find((path) => files.some((f) => f.path === path));
    if (first) await selectFile(first);
  } else if (state.latestPage && files.find((file) => file.path === state.file)?.bytes !== state.page?.size) {
    await selectFile(state.file);
  }
}

function renderManifest() {
  const run = info();
  if (!run) return;
  $("manifest").replaceChildren(block("Run manifest", run.manifest), block("Console state", {
    id: run.id, managed: run.managed, status: run.status, checkpoint: run.checkpoint, command: run.command, exit_code: run.exit_code,
  }));
}

function summarizeUsage(records) {
  const totals = {};
  for (const key of ["input_tokens", "output_tokens", "total_tokens", "cost_rub"]) {
    const known = records.map((record) => record[key]).filter(Number.isFinite);
    totals[key] = {sum: known.reduce((sum, value) => sum + value, 0), known: known.length};
  }
  return {requests: records.length, failures: records.filter((record) => record.request_status === "failed").length,
    memoryCalls: records.filter((record) => record.operation === "extraction").length, totals};
}

async function artifactText(run, path) {
  const pages = [];
  let end;
  do {
    const query = {run, path};
    if (end !== undefined) query.end = end;
    const page = await get("file", query);
    pages.push(page.text);
    end = page.start;
  } while (end > 0);
  return pages.reverse().join("");
}

function recordedRows(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const rows = [];
  let incomplete = 0;
  for (const line of lines) {
    try { rows.push(JSON.parse(line)); } catch { incomplete += 1; }
  }
  return {rows, incomplete};
}

async function refreshResults() {
  const selected = info();
  if (selected?.manifest.benchmark) return;
  if (!selected || !terminal.has(selected.status) || state.resultData || state.resultLoading) return;
  const run = selected.id;
  const generation = state.generation;
  state.resultLoading = true;
  try {
    const artifacts = new Set((await get("files", {run})).map((file) => file.path));
    const load = async (path) => artifacts.has(path) ? artifactText(run, path) : "";
    const arms = selected.manifest.conditions || ["mage", "mage_plus_concepts", "pcms"].filter((arm) =>
      artifacts.has(`${arm}/result.json`) || artifacts.has(`${arm}/usage.jsonl`));
    const armData = await Promise.all(arms.map(async (arm) => {
      const [result, usage] = await Promise.all([load(`${arm}/result.json`), load(`${arm}/usage.jsonl`)]);
      return {arm, result: result ? JSON.parse(result) : null, usage: recordedRows(usage)};
    }));
    const [diagnostics, usage] = arms.length ? ["", ""] : await Promise.all([load("results.jsonl"), load("usage.jsonl")]);
    if (generation !== state.generation) return;
    state.resultData = {arms: armData, diagnostics: recordedRows(diagnostics), usage: recordedRows(usage)};
    if (state.tab === "results") renderResults();
  } catch (error) {
    if (generation === state.generation && state.tab === "results") {
      $("results").replaceChildren(node("p", `Could not load results. ${error.message}`, "error"),
        button("Try again", () => { void refreshResults(); }));
    }
  } finally {
    if (generation === state.generation) state.resultLoading = false;
  }
}

function resultTable(headers, rows) {
  const wrapper = node("div", "", "result-table");
  const table = node("table");
  const heading = node("tr");
  for (const label of headers) {
    const cell = node("th", label);
    cell.scope = "col";
    heading.append(cell);
  }
  table.createTHead().append(heading);
  const body = table.createTBody();
  for (const [label, ...values] of rows) {
    const row = body.insertRow();
    const cell = node("th", label);
    cell.scope = "row";
    row.append(cell);
    for (const value of values) row.insertCell().append(value);
  }
  wrapper.append(table);
  return wrapper;
}

function displayTotal(usage, key) {
  const {sum, known} = usage.totals[key];
  if (!known) return "—";
  const value = key === "cost_rub" ? `${sum.toFixed(2)} RUB` : sum.toLocaleString();
  return known < usage.requests ? `${value} known · ${usage.requests - known} unknown` : value;
}

function usageRows(summaries) {
  return [
    ["Model requests", ...summaries.map((row) => row.requests.toLocaleString())],
    ["Memory model requests", ...summaries.map((row) => row.memoryCalls.toLocaleString())],
    ["Failed requests", ...summaries.map((row) => row.failures.toLocaleString())],
    ...[["Input tokens", "input_tokens"], ["Output tokens", "output_tokens"],
      ["Total tokens", "total_tokens"], ["Billed cost", "cost_rub"]].map(([label, key]) =>
      [label, ...summaries.map((row) => displayTotal(row, key))]),
  ];
}

function armElapsed(arm) {
  const events = state.events.filter((event) => event.condition === arm);
  const start = events.find((event) => event.title === "Comparison arm started");
  const end = events.find((event) => event.title === "Comparison arm scored");
  if (!start || !end) return "—";
  const seconds = Math.round((eventTime(end) - eventTime(start)) / 1000);
  const hours = Math.floor(seconds / 3600);
  return `${hours ? `${hours}h ` : ""}${Math.floor(seconds / 60) % 60}m ${seconds % 60}s`;
}

function updateWorker(worker, events) {
  for (const event of events) {
    if (event.seq <= worker.sequence) continue;
    worker.sequence = event.seq;
    worker.at = event.at;
    const data = event.data || {};
    if (event.title === "Comparison arm started") worker.arm = data.condition;
    if (event.title === "Traveler turn ended" || event.title === "PCMS traveler turn ended") {
      worker.completed.add(`${data.condition}:${data.turn}`);
    }
    if (event.title === "Paused at checkpoint") worker.activity = "Paused";
    else if (event.title === "Continued from checkpoint") worker.activity = data.operation || "Resuming";
    else if (event.kind === "transport" && data.status === "streaming") worker.activity = "Receiving model response";
    else if (event.kind === "checkpoint") {
      worker.activity = event.title === "Send model request" && data.scope?.operation === "extraction" ?
        "Waiting for memory model" : event.title === "Send model request" ? "Waiting for model" : event.title;
    }
  }
}

async function refreshWorkers(run, groups) {
  const active = groups.filter((group) => group.status === "running" && group.path);
  const previous = state.workers;
  const workers = await Promise.all(active.map(async (group) => {
    const worker = previous.get(group.path) || {offset: 0, sequence: 0, completed: new Set()};
    const pages = [];
    let end;
    let size;
    try {
      do {
        const query = {run, path: `${group.path}/timeline.jsonl`};
        if (end !== undefined) query.end = end;
        const page = await get("file", query);
        size ??= page.size;
        pages.push(page.text);
        end = page.start;
      } while (end > 0 && end >= worker.offset);
      const content = pages.reverse().join("");
      const completeLines = content.slice(0, content.lastIndexOf("\n") + 1);
      updateWorker(worker, recordedRows(completeLines).rows);
      // Revisit the last page next poll if the final line was still being written.
      if (content.endsWith("\n")) worker.offset = size;
    } catch {
      // A worker can be listed before it creates its timeline.
    }
    return [group.path, worker];
  }));
  if (state.selected === run) state.workers = new Map(workers);
}

function renderWorkerProgress(run, groups) {
  const view = $("worker-progress");
  if (!view) return;
  const active = groups.filter((group) => group.status === "running");
  view.hidden = !active.length;
  if (!active.length) return;
  const cards = active.map((group) => {
    const worker = state.workers.get(group.path);
    const completed = worker?.completed.size ?? 0;
    const max = group.travelers * (run.manifest.conditions?.length || 1);
    const card = node("article", "", "worker-card");
    const header = node("div", "", "worker-header");
    const label = `Group ${group.group}`;
    header.append(node("strong", label), node("span", `${completed} / ${max} travelers`));
    const bar = node("progress");
    bar.max = max;
    bar.value = completed;
    bar.setAttribute("aria-label", `${label} travelers completed`);
    const age = worker?.at ? Math.max(0, Math.floor((Date.now() - Date.parse(worker.at)) / 1000)) : null;
    const elapsed = age === null ? "" : age < 60 ? ` · ${age}s ago` : ` · ${Math.floor(age / 60)}m ago`;
    const arm = run.manifest.conditions?.length > 1 && worker?.arm ? `${worker.arm.toUpperCase()} · ` : "";
    card.append(header, bar, node("p", `${arm}${worker?.activity || "Starting"}${elapsed}`, "subtle"));
    return card;
  });
  view.replaceChildren(node("h3", "Active workers"), ...cards);
}

function renderBenchmarkResults(view, run) {
  if (run.manifest.error) view.append(node("p", run.manifest.error, "error"));
  const data = state.benchmark;
  if (!data) {
    view.append(node("p", "Loading benchmark progress…", "subtle"));
    return;
  }
  const workers = node("section", "", "worker-grid");
  workers.id = "worker-progress";
  view.append(workers);
  renderWorkerProgress(run, data.groups);
  const arms = run.manifest.conditions;
  if (arms.length === 1) {
    const reference = node("p", "Published MAGE on all 270 groups · 15.19% group success · 53.51% average group progress · 1.978M tokens per group");
    const link = node("a", "MAGE paper, Table 3");
    link.href = "https://aclanthology.org/2026.emnlp-main.828.pdf#page=8";
    view.append(reference, link);
  }
  const fraction = (numerator, denominator) => denominator ?
    `${numerator} / ${denominator} (${(100 * numerator / denominator).toFixed(1)}%)` : "—";
  for (const [key, title] of Object.entries({all: "Full release", holdout: "Held-out groups", validation: "Validation", dev: "Development"})) {
    const split = data.splits?.[key];
    if (!split) continue;
    const values = (callback) => arms.map((arm) => split.arms?.[arm] ? callback(split.arms[arm]) : "—");
    view.append(node("h3", title), node("p", `${split.paired} / ${split.planned} groups scored · ${split.failed} failed`, "subtle"),
      resultTable(["Metric", ...arms.map((arm) => arm.toUpperCase())], [
        ["Fully solved groups", ...values((arm) => fraction(arm.groups_passed, split.paired))],
        ["Travelers solved", ...values((arm) => fraction(arm.travelers_passed, arm.travelers_scored))],
        ["Average group progress", ...values((arm) => arm.average_group_progress == null ? "—" :
          `${(100 * arm.average_group_progress).toFixed(2)}%`)],
        ["Model requests", ...values((arm) => arm.requests.toLocaleString())],
        ["Failed requests", ...values((arm) => arm.failed.toLocaleString())],
        ["Known tokens", ...values((arm) => arm.tokens.toLocaleString())],
        ["Known tokens per group", ...values((arm) => split.paired === split.planned && split.paired ?
          Math.round(arm.tokens / split.paired).toLocaleString() : "—")],
        ["Known billed cost", ...values((arm) => `₽${arm.cost_rub.toFixed(2)}`)],
        ["Requests with unknown cost", ...values((arm) => arm.unknown_cost.toLocaleString())],
      ]));
  }
  const attempted = data.groups.filter((group) => group.path);
  if (attempted.length) {
    const rows = attempted.map((group) => {
      const inspect = node("button", "Open trace");
      inspect.addEventListener("click", async () => {
        showTab("files");
        await refreshFiles();
        const path = `${group.path}/timeline.jsonl`;
        if (state.files.some((file) => file.path === path)) await selectFile(path);
      });
      return [String(group.group), group.partition, group.status, group.error || "—", inspect];
    });
    view.append(node("h3", "Group runs"), resultTable(["Group", "Split", "Status", "Error", "Trace"], rows));
  }
}

function renderResults() {
  const view = $("results");
  const run = info();
  view.replaceChildren(node("h3", "Run results"));
  if (run?.manifest.benchmark) {
    renderBenchmarkResults(view, run);
    return;
  }
  if (!run || !terminal.has(run.status)) {
    view.append(node("p", "Results will appear when the run ends.", "subtle"));
    return;
  }
  if (!state.resultData) {
    view.append(node("p", "Loading recorded results…", "subtle"));
    return;
  }
  const {arms, diagnostics, usage} = state.resultData;
  if (run.status !== "complete") view.append(node("p", `${displayStatus(run)} · Showing recorded results only.`, "subtle"));
  if (run.manifest.error) view.append(node("p", run.manifest.error, "error"));
  if (arms.length) {
    const names = {mage: "MAGE", pcms: "PCMS", mage_plus_concepts: "MAGE + PCMS"};
    const headers = arms.map(({arm}) => names[arm] || arm);
    const counts = arms.map(({result}) => result?.sub_successes);
    const prefix = run.manifest.turns < run.manifest.full_turns;
    view.append(resultTable(["Metric", ...headers], [
      ["Traveler success", ...counts.map((scores) => scores?.length ?
        `${scores.filter(Boolean).length}/${scores.length} (${Math.round(100 * scores.filter(Boolean).length / scores.length)}%)` : "Unscored")],
      [prefix ? "Prefix success" : "Full-group success", ...arms.map(({result}) => {
        const passed = result?.[prefix ? "prefix_success" : "task_success"] ??
          (result?.sub_successes?.length === run.manifest.turns ? result.sub_successes.every(Boolean) : null);
        return typeof passed === "boolean" ? (passed ? "Pass" : "Fail") : "Unscored";
      })],
      ...usageRows(arms.map((arm) => summarizeUsage(arm.usage.rows))),
      ["Elapsed wall time", ...arms.map(({arm}) => armElapsed(arm))],
    ]));
    const total = Math.max(0, ...counts.map((scores) => scores?.length ?? 0));
    if (total) view.append(node("h3", "Traveler outcomes"), resultTable(["Traveler", ...headers],
      Array.from({length: total}, (_, index) => [`${index + 1}`, ...counts.map((scores) =>
        scores?.[index] === undefined ? "—" : scores[index] ? "Pass" : "Fail")])));
  } else {
    if (diagnostics.rows.length) view.append(resultTable(["Case", "Mode", "Checkpoint", "Outcome", "Memory tokens"],
      diagnostics.rows.map((row) => [row.case, row.mode, row.checkpoint,
        typeof row.passed === "boolean" ? (row.passed ? "Correct decision" : "Incorrect decision") :
          `Predecessor ${row.predecessor_present ? "found" : "missing"} · Balance ${row.remaining_present ? "found" : "missing"}`,
        String(row.tokens ?? "—")])));
    else view.append(node("p", "No score was recorded.", "subtle"));
    if (usage.rows.length) view.append(node("h3", "Run usage"),
      resultTable(["Metric", "Total"], usageRows([summarizeUsage(usage.rows)])));
  }
  const incomplete = diagnostics.incomplete + usage.incomplete + arms.reduce((sum, arm) => sum + arm.usage.incomplete, 0);
  if (incomplete) view.append(node("p", `${incomplete} incomplete log row(s) omitted.`, "subtle"));
}

async function sync() {
  if (state.polling) return;
  state.polling = true;
  try {
    const previous = info();
    state.runs = await api("/api/runs");
    if (!state.selected && state.runs.length) {
      const target = new URLSearchParams(location.search).get("run");
      selectRun(state.runs.find((run) => run.id === target)?.id || state.runs[0].id);
    }
    renderRuns();
    renderHeader();
    const current = info();
    if (current?.manifest.benchmark) {
      const benchmark = await get("benchmark", {run: current.id});
      if (state.selected === current.id) {
        const changed = JSON.stringify(benchmark) !== JSON.stringify(state.benchmark);
        state.benchmark = benchmark;
        renderHeader();
        if (state.tab === "results" && changed) renderResults();
        if (state.tab === "results" && !terminal.has(current.status)) {
          await refreshWorkers(current.id, benchmark.groups);
          if (state.selected === current.id && state.tab === "results") renderWorkerProgress(current, benchmark.groups);
        }
      }
    }
    if (previous?.id === current?.id && !terminal.has(previous?.status) && terminal.has(current?.status)) showTab("results");
    await refreshEvents();
    if (state.tab === "files") await refreshFiles();
  } catch (error) { notice(error.message); }
  finally { state.polling = false; }
}

async function control(action) {
  if (state.busy || !state.selected) return;
  if (action === "terminate" && !confirm("Terminate the worker now? A provider request already sent may still incur a charge, and its final response may be missing.")) return;
  state.busy = true;
  notice("");
  renderHeader();
  try {
    const data = {run: state.selected, action};
    if (action === "run_to") data.target = $("breakpoint").value;
    const result = await post("control", data);
    state.runs = state.runs.map((run) => run.id === result.id ? {...run, ...result} : run);
    await sync();
  } catch (error) { notice(error.message); }
  finally { state.busy = false; renderHeader(); renderRuns(); }
}

function showExperiment() {
  const choice = $("experiment").value;
  const benchmark = choice === "benchmark";
  const paired = choice === "paired" || benchmark;
  $("paired-fields").hidden = !paired;
  $("group-fields").hidden = benchmark;
  $("worker-field").hidden = !benchmark;
  $("pcms-only-field").hidden = !benchmark;
  if (benchmark) {
    const form = $("experiment").form;
    form.elements.max_steps.value = 12;
    form.elements.memory_tokens.value = 256;
    form.elements.thinking.checked = false;
  }
  $("experiment-description").textContent = {
    shopping: "Use the released Product 4 instruction and the published purchase prefix. Inspect retrieval and budget reconstruction.",
    travel: "Inspect proposed and confirmed updates, then compare live travel decisions across memory reads.",
    paired: "Run official MAGE and standalone PCMS against the same travel tools, catalog, scorer, and Polza route.",
    benchmark: "Run all 270 Travel Planning groups. Use the published MAGE score or run a matched MAGE arm.",
  }[choice];
  const missing = (choice === "shopping" && !state.config.shopping_ready) || (paired && !state.config.paired_ready);
  $("launch").disabled = missing;
  $("launch-error").hidden = !missing;
  $("launch-error").textContent = missing ? "Required local benchmark files are missing. See the console setup instructions." : "";
}

async function launch(event) {
  event.preventDefault();
  const fields = new FormData(event.target);
  const data = {experiment: fields.get("experiment"), memory_tokens: Number(fields.get("memory_tokens"))};
  if (data.experiment === "paired" || data.experiment === "benchmark") {
    data.max_steps = Number(fields.get("max_steps"));
    data.thinking = fields.has("thinking");
    if (data.experiment === "benchmark") {
      data.workers = Number(fields.get("workers"));
      data.pcms_only = fields.has("pcms_only");
    } else {
      data.group = Number(fields.get("group"));
      data.turns = fields.get("turns") ? Number(fields.get("turns")) : null;
    }
  }
  $("launch").disabled = true;
  try {
    const result = await post("start", data);
    $("setup").close();
    state.runs = await api("/api/runs");
    selectRun(result.run);
  } catch (error) {
    $("launch-error").textContent = error.message;
    $("launch-error").hidden = false;
  } finally { $("launch").disabled = false; }
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); }
  catch (error) { notice(`Clipboard unavailable: ${error.message}`); }
}

async function boot() {
  try {
    state.config = await api("/api/config");
    for (const group of state.config.groups) {
      const option = node("option", `Group ${group}`);
      option.value = group;
      $("groups").append(option);
    }
    await sync();
    setInterval(sync, 1600);
  } catch (error) { notice(error.message); }
}

if (typeof document !== "undefined") {
  for (const tab of views) $(`${tab}-tab`).addEventListener("click", () => showTab(tab));
  for (const [id, action] of Object.entries({resume: "run", pause: "pause", step: "step", stop: "stop", terminate: "terminate"})) {
    $(id).addEventListener("click", () => { void control(action); });
  }
  $("run-to").addEventListener("click", () => { void control("run_to"); });
  $("search").addEventListener("input", renderEvents);
  $("kind").addEventListener("change", renderEvents);
  $("follow").addEventListener("change", () => {
    renderEvents();
    const event = visibleEvents().at(-1);
    if ($("follow").checked && event) void selectEvent(event);
  });
  $("raw").addEventListener("click", () => { state.raw = !state.raw; renderDetail(); });
  $("copy-event").addEventListener("click", () => { if (state.detail) void copy(pretty(state.detail)); });
  $("copy-file").addEventListener("click", () => { if (state.page) void copy(state.page.text); });
  $("older").addEventListener("click", () => { if (state.page) void selectFile(state.file, state.page.start); });
  $("latest").addEventListener("click", () => { if (state.file) void selectFile(state.file); });
  $("new-run").addEventListener("click", () => { if (state.config) { showExperiment(); $("setup").showModal(); } });
  $("close-setup").addEventListener("click", () => $("setup").close());
  $("cancel-setup").addEventListener("click", () => $("setup").close());
  $("experiment").addEventListener("change", showExperiment);
  $("launch-form").addEventListener("submit", (event) => { void launch(event); });

  void boot();
}
