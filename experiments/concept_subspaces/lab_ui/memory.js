"use strict";

function memorySnapshot(events, step) {
  const records = new Map();
  const sources = new Map();
  let rows = {};
  let read = null;
  let selection = null;
  let turn = null;
  let readTurn = null;
  for (const event of events.slice(0, step)) {
    const data = event.data;
    if (event.title === "PCMS traveler turn ended") {
      turn = data.turn + 1;
    } else if (event.title === "PCMS traveler task prepared") {
      turn = data.turn;
    } else if (event.title === "Incoming evidence" && data.evidence) {
      sources.set(data.evidence.id, data.evidence);
    } else if (event.title === "Memory write committed") {
      for (const record of data.memories) records.set(record.id, record);
      rows = data.current_rows || {};
    } else if (["Resolved evidence references", "Selected concept evidence", "Resolved retrieval scope"].includes(event.title)) {
      selection = data;
      read = null;
    } else if (event.title === "Packed actor context") {
      read = data;
      readTurn = turn;
    }
  }
  return {records: [...records.values()], sources, rows, read, selection, readTurn};
}

function rowLabel(field, record) {
  return record.subject ? JSON.parse(field)[1] : field;
}

function constraintTables(rows) {
  const tables = new Map();
  for (const [owner, fields] of Object.entries(rows)) {
    const families = new Map();
    for (const [field, record] of Object.entries(fields)) {
      const family = record.constraint_family;
      if (!family || record.status !== "confirmed") continue;
      if (!families.has(family)) families.set(family, []);
      const prefix = `constraint / ${family} / `;
      const label = rowLabel(field, record);
      families.get(family).push([label.startsWith(prefix) ? label.slice(prefix.length) : label, record]);
    }
    const groups = [...families].filter(([, entries]) => entries.length >= 2);
    const legacy = Object.entries(fields).filter(([, record]) => !Object.hasOwn(record, "constraint_family"));
    if (legacy.length) groups.push(["Recorded rows", legacy]);
    const observations = Object.entries(fields)
      .filter(([, record]) => record.subject && !record.constraint_family)
      .map(([field, record]) => [rowLabel(field, record), record]);
    if (observations.length) groups.push(["Other linked rows", observations]);
    if (groups.length) tables.set(owner, groups);
  }
  return tables;
}

if (typeof module !== "undefined") module.exports = {memorySnapshot, constraintTables};

const memoryReplay = (() => {
  let run, activeCase, generation = 0, busy = false, loaded = false;
  let events = [], offset = 0, step = 0, error = "", concept = "", search = "", limit = 40;
  let focusedRecord = null;
  let tablesOpen = false, tableOwner = "";
  const make = (tag, text, cls) => node(tag, text, cls);

  function detail(title, value) {
    const section = make("details");
    section.append(make("summary", title), make("pre", typeof value === "string" ? value : pretty(value)));
    return section;
  }

  function render() {
    const event = events[step - 1];
    const body = $("memory-body");
    const select = $("memory-case");
    $("memory-slider").max = events.length;
    $("memory-slider").value = step;
    $("memory-prev").disabled = step === 0;
    $("memory-next").disabled = step >= events.length;
    $("memory-status").textContent = error || (event
      ? `Step ${step} of ${events.length} · Event #${event.seq} · ${event.title}`
      : loaded ? (events.length ? "Initial state · before the first recorded event" : "No memory events recorded for this case yet.")
        : (select.options.length ? "Loading memory history…" : "Waiting for a case trace…"));
    body.replaceChildren();
    if (!events.length) return;

    if (event?.title === "MAGE execution state") {
      const snapshot = make("section", "", "memory-card");
      snapshot.append(make("h3", `MAGE · active node ${event.data.active_node ?? "—"}`));
      const button = make("button", "Load recorded tree and context");
      const selectedRun = run, selectedCase = activeCase;
      button.onclick = async () => {
        button.disabled = true;
        try {
          const row = await get("memory-event", {run: selectedRun, case: selectedCase, offset: event.offset});
          if (!snapshot.isConnected) return;
          for (const [label, value] of Object.entries({
            "Tree nodes": row.data.nodes, "Summaries": row.data.summaries,
            "Stored context": row.data.stored_context, "Recent context": row.data.raw_context,
          })) snapshot.append(detail(label, value));
          button.remove();
        } catch (e) { button.textContent = e.message; button.disabled = false; }
      };
      snapshot.append(button);
      body.append(snapshot);
      return;
    }

    const {records, sources, rows, read, selection, readTurn} = memorySnapshot(events, step);
    const concepts = new Map();
    for (const record of records) for (const name of new Set(record.concepts)) {
      if (!concepts.has(name)) concepts.set(name, []);
      concepts.get(name).push(record);
    }
    const selected = new Set([...(selection?.required_ids || []), ...(read?.initial_ids || [])]);
    const delivered = new Set(read?.delivered_ids || []);
    const added = new Set(event?.title === "Memory write committed" ? event.data.memories.map(r => r.id) : []);
    const overview = make("div", "", "memory-overview");
    const metrics = [
      ["Records", records.length], ["Concepts", concepts.size],
      ["Selected", selection || read ? selected.size : "—"],
      ["Delivered", read ? delivered.size : "—"],
      ["Actor memory tokens", read ? `${read.tokens} / ${read.token_limit}` : "—"],
    ];
    for (const [label, value] of metrics) {
      const card = make("div", "", "memory-card");
      card.append(make("strong", String(value)), make("span", label, "subtle"));
      overview.append(card);
    }
    body.append(overview);
    if (read?.token_limit > 0) {
      const meter = make("progress", "", "memory-budget");
      meter.max = read.token_limit;
      meter.value = read.tokens;
      meter.setAttribute("aria-label", "Actor memory budget used");
      body.append(meter);
    }

    const graph = make("section", "", "memory-graph");
    const map = make("div", "", "memory-card memory-map");
    const mapHeader = make("div", "", "memory-map-header");
    mapHeader.append(make("h3", "Concept map"));
    const tablePanel = make("section", "", "memory-card memory-table-view");
    tablePanel.hidden = !tablesOpen;
    const tableButton = make("button", "Concept tables");
    tableButton.setAttribute("aria-expanded", String(tablesOpen));
    tableButton.onclick = () => {
      tablesOpen = !tablesOpen;
      tablePanel.hidden = !tablesOpen;
      tableButton.setAttribute("aria-expanded", String(tablesOpen));
      if (tablesOpen) tablePanel.scrollIntoView({block: "nearest"});
    };
    mapHeader.append(tableButton);
    const mapSearch = make("input");
    mapSearch.type = "search";
    mapSearch.placeholder = "Find a concept or record";
    mapSearch.setAttribute("aria-label", "Find a concept or record on the map");
    mapHeader.append(mapSearch);
    const legend = make("div", "", "memory-legend");
    const colors = {stored: "Stored", added: "New write", selected: "Selected",
      delivered: "Delivered", omitted: "Omitted", superseded: "Superseded"};
    for (const [key, label] of Object.entries(colors)) legend.append(make("span", label, `memory-mark ${key}`));
    const mapRows = make("div", "", "memory-map-rows");
    const card = make("aside", "", "memory-card memory-node-detail");
    const routed = new Set(selection?.concepts || read?.concepts || []);

    function recordState(record) {
      if (delivered.has(record.id)) return "delivered";
      if (selected.has(record.id)) return read ? "omitted" : "selected";
      if (added.has(record.id)) return "added";
      const current = rows[record.table]?.[record.row];
      return current && current.id !== record.id && ["confirmed", "withdrawn"].includes(record.status)
        ? "superseded" : "stored";
    }

    function showRecord(record) {
      focusedRecord = record?.id ?? null;
      card.replaceChildren(make("h3", "Record inspector"));
      for (const marker of mapRows.querySelectorAll("button[data-record]")) {
        marker.setAttribute("aria-pressed", String(marker.dataset.record === focusedRecord));
      }
      if (!record) {
        card.append(make("p", "Select a record on the map to see its content and concept links.", "subtle"));
        return;
      }
      card.append(make("span", `${record.id} · ${record.status}`, "subtle"),
        make("span", colors[recordState(record)], `memory-mark ${recordState(record)}`),
        make("pre", record.text));
      if (record.subject) {
        card.append(make("p", `Subject: ${record.subject} · ${record.table} · ${rowLabel(record.row, record)}`, "subtle"));
      }
      const links = make("div", "", "memory-concept-links");
      for (const name of record.concepts) links.append(make("span", name));
      card.append(links);
      const score = selection?.scores?.[record.id];
      if (typeof score === "number") card.append(make("p", `Jev relevance ${score.toFixed(2)}`, "subtle"));
      if (sources.has(record.source)) card.append(detail("Original evidence", sources.get(record.source).text));
    }

    function drawMap() {
      const term = mapSearch.value.trim().toLowerCase();
      mapRows.replaceChildren();
      const ordered = [...concepts].sort(([a], [b]) => Number(routed.has(b)) - Number(routed.has(a)) ||
        a.localeCompare(b));
      for (const [name, members] of ordered) {
        const visible = !term || name.toLowerCase().includes(term) ? members :
          members.filter(record => `${record.id} ${record.text}`.toLowerCase().includes(term));
        if (!visible.length) continue;
        const row = make("div", "", `memory-map-row${routed.has(name) ? " routed" : ""}`);
        const label = make("div", "", "memory-concept-node");
        label.append(make("strong", name), make("small",
          routed.has(name) ? `${members.length} records · routed` : `${members.length} records`, "subtle"));
        const dots = make("div", "", "memory-map-nodes");
        for (const record of visible) {
          const status = recordState(record);
          const marker = make("button", "", `memory-node ${status}`);
          marker.dataset.record = record.id;
          marker.title = `${colors[status]} · ${record.id}\n${record.text.slice(0, 180)}`;
          marker.setAttribute("aria-label", `${colors[status]} record ${record.id}`);
          marker.setAttribute("aria-pressed", String(focusedRecord === record.id));
          marker.onclick = () => showRecord(record);
          dots.append(marker);
        }
        row.append(label, dots);
        mapRows.append(row);
      }
      if (!mapRows.childElementCount) mapRows.append(make("p",
        records.length ? "No matching concepts or records." : "No records have been written at this step.", "subtle"));
    }
    mapSearch.oninput = drawMap;
    drawMap();
    showRecord(records.find(record => record.id === focusedRecord));
    map.append(mapHeader, legend, mapRows);
    graph.append(map, card);
    body.append(graph);
    const tables = constraintTables(rows);
    const owners = [...tables.keys()].sort((a, b) => a.localeCompare(b));
    tablePanel.append(make("h3", "Concept tables"));
    if (!owners.length) tablePanel.append(make("p", "No grouped constraints or subject-linked rows at this step.", "subtle"));
    else {
      const picker = make("select");
      picker.setAttribute("aria-label", "Table owner");
      picker.append(...owners.map(owner => new Option(owner, owner)));
      if (owners.includes(tableOwner)) picker.value = tableOwner;
      const table = make("table", "", "memory-state-table");
      const head = make("thead");
      const headings = make("tr");
      for (const name of ["Subject", "Scope or field", "Source evidence", "Status"]) {
        const heading = make("th", name);
        heading.scope = "col";
        headings.append(heading);
      }
      head.append(headings);
      table.append(head);
      const renderTable = () => {
        table.replaceChildren(head);
        for (const [family, entries] of tables.get(picker.value)) {
          const group = make("tbody");
          const headingRow = make("tr");
          const heading = make("th", family);
          heading.scope = "rowgroup";
          heading.colSpan = 4;
          headingRow.append(heading);
          group.append(headingRow);
          for (const [field, record] of entries) {
            const row = make("tr");
            const subject = make("td", record.subject || "—");
            if (record.subject) {
              const link = make("button", record.subject);
              link.type = "button";
              link.onclick = () => {
                mapSearch.value = record.subject;
                drawMap();
                showRecord(record);
                map.scrollIntoView({block: "nearest"});
              };
              subject.replaceChildren(link);
            }
            const cell = make("td");
            cell.append(detail(record.text, `${record.id} · ${record.source}`));
            row.append(subject, make("td", field), cell, make("td", record.status));
            group.append(row);
          }
          table.append(group);
        }
      };
      picker.onchange = () => { tableOwner = picker.value; renderTable(); };
      renderTable();
      tablePanel.append(picker, table);
    }
    body.append(tablePanel);

    const columns = make("div", "", "memory-columns");
    const inventory = make("section", "", "memory-card memory-inventory");
    const header = make("h3", "Memory records");
    const filters = make("div", "", "memory-filters");
    const selector = make("select");
    selector.setAttribute("aria-label", "Filter by concept");
    selector.append(new Option("All concepts", ""));
    for (const [name, members] of [...concepts].sort(([a], [b]) => a.localeCompare(b))) {
      selector.append(new Option(`${name} (${members.length})`, name));
    }
    selector.value = concepts.has(concept) ? concept : "";
    const input = make("input");
    input.type = "search";
    input.placeholder = "Search memory records";
    input.setAttribute("aria-label", "Search memory records");
    input.value = search;
    const stage = make("select");
    stage.setAttribute("aria-label", "Filter by retrieval outcome");
    stage.append(new Option("All records", ""));
    if (added.size) stage.append(new Option("Added at this step", "added"));
    if (selected.size) stage.append(new Option("Selected", "selected"));
    if (read) stage.append(new Option("Delivered", "delivered"), new Option("Selected and omitted", "omitted"));
    filters.append(selector, stage, input);
    const list = make("div", "", "memory-records");
    function renderRecords() {
      const needle = search.trim().toLowerCase();
      const stageMatches = {"": () => true, added: id => added.has(id), selected: id => selected.has(id),
        delivered: id => delivered.has(id), omitted: id => selected.has(id) && !delivered.has(id)};
      const matching = records.filter(r => stageMatches[stage.value](r.id) &&
        (!selector.value || r.concepts.includes(selector.value)) &&
        (!needle || [r.id, r.source, r.text, ...r.concepts].join(" ").toLowerCase().includes(needle)));
      matching.sort((a,b) => Number(added.has(b.id)) - Number(added.has(a.id)) ||
        Number(selected.has(b.id)) - Number(selected.has(a.id)) || b.step - a.step);
      list.replaceChildren();
      if (!matching.length) list.append(make("p", "No matching records.", "subtle"));
      for (const r of matching.slice(0, limit)) {
        const item = make("details", "", "memory-record");
        const summary = make("summary");
        const text = r.text.replace(/\s+/g, " ");
        summary.append(make("span", text.length > 150 ? `${text.slice(0, 147)}…` : text));
        const tags = [r.status];
        const current = rows[r.table]?.[r.row];
        if (current && current.id !== r.id && ["confirmed", "withdrawn"].includes(r.status)) tags.push("superseded");
        if (added.has(r.id)) tags.push("added");
        if (delivered.has(r.id)) tags.push("delivered");
        else if (selected.has(r.id)) tags.push(read ? "selected · omitted" : "selected");
        summary.append(make("small", tags.join(" · "), delivered.has(r.id) ? "memory-delivered" : "subtle"));
        item.append(summary, make("pre", r.text),
          make("p", `${r.id} · ${r.concepts.join(", ")}`, "subtle"));
        const source = sources.get(r.source);
        if (source) item.append(detail(`Source · ${r.source}`, source.text));
        list.append(item);
      }
      if (matching.length > limit) {
        const more = make("button", `Show more (${matching.length - limit} remaining)`);
        more.onclick = () => { limit += 40; renderRecords(); };
        list.append(more);
      }
    }
    selector.onchange = () => { concept = selector.value; limit = 40; renderRecords(); };
    stage.onchange = () => { limit = 40; renderRecords(); };
    input.oninput = () => { search = input.value; limit = 40; renderRecords(); };
    renderRecords();
    inventory.append(header, filters, list);

    const why = make("aside", "", "memory-card memory-step");
    why.append(make("h3", "Step details"));
    if (event?.data.query) why.append(detail("Request", event.data.query));
    if (read) {
      why.append(make("h3", readTurn === null ? "What reached the actor" : `What reached traveler ${readTurn + 1}`),
        make("pre", read.context || "No recalled evidence was delivered."));
      const omitted = [...selected].filter(id => !delivered.has(id));
      if (omitted.length) why.append(detail(`Selected and omitted (${omitted.length})`, omitted));
      if (read.packing_decisions) why.append(detail("Packing decisions", read.packing_decisions));
    } else if (selection) {
      why.append(make("p", "Selection recorded. Continue to the packing step to see what reached the actor.", "subtle"));
    }
    if (event) why.append(detail("Recorded event", event.data));
    if (sources.size) why.append(detail(`Source evidence (${sources.size})`, [...sources.values()]));
    if (Object.keys(rows).length) why.append(detail("Current table rows", rows));
    columns.append(inventory, why);
    const recordedDetails = make("details", "", "memory-raw-details");
    recordedDetails.append(make("summary", "Actor context, records, and raw events"), columns);
    body.append(recordedDetails);
  }

  function reset(caseId) {
    activeCase = caseId;
    events = [];
    offset = step = 0;
    loaded = false;
    error = concept = search = "";
    focusedRecord = null;
    tableOwner = "";
    limit = 40;
    generation++;
    $("memory-follow").checked = true;
    $("memory-jump").replaceChildren(new Option("Jump to…", ""));
  }

  async function refresh(info) {
    if (!info) return;
    if (run !== info.id) {
      run = info.id;
      reset(undefined);
      $("memory-case").replaceChildren();
      render();
    }
    if (busy || (loaded && terminal.has(info.status))) return;
    busy = true;
    let version = generation;
    try {
      const cases = info.manifest.benchmark ? await get("memory-cases", {run}) : [""];
      if (version !== generation) return;
      const selector = $("memory-case");
      if (JSON.stringify(cases) !== JSON.stringify([...selector.options].map(o => o.value))) {
        selector.replaceChildren(...cases.map(id => {
          const match = id.match(/group-(\d+)\/attempt-(\d+)/);
          return new Option(match ? `Group ${Number(match[1])} · Attempt ${match[2]}` : "Run timeline", id);
        }));
      }
      if (!cases.length) { render(); return; }
      if (!cases.includes(activeCase)) {
        reset(cases[0]);
        version = generation;
      }
      selector.value = activeCase;
      let changed = false;
      for (let page = 0; page < 12; page++) {
        const data = await get("memory", {run, case: activeCase, after: offset});
        if (version !== generation) return;
        offset = data.next;
        const start = events.length;
        events.push(...data.events);
        data.events.forEach((event, i) => {
          if (["Memory write committed", "Packed actor context", "PCMS traveler turn ended"].includes(event.title)
              || event.kind === "error") {
            $("memory-jump").append(new Option(`#${event.seq} · ${event.title}`, start + i + 1));
          }
        });
        changed ||= data.events.length > 0;
        loaded = data.caught_up;
        if (loaded) break;
      }
      if ($("memory-follow").checked) step = events.length;
      if (changed && !$("memory-follow").checked && !error) {
        $("memory-slider").max = events.length;
        $("memory-next").disabled = step >= events.length;
      } else if (changed || error || !$("memory-status").textContent.includes("Step")) {
        error = "";
        render();
      }
    } catch (e) {
      if (version === generation) { error = e.message; render(); }
    } finally { busy = false; }
  }

  function seek(value) {
    $("memory-follow").checked = false;
    step = Math.max(0, Math.min(events.length, value));
    render();
  }

  function bindControls() {
    $("memory-case").onchange = () => { reset($("memory-case").value); render(); void refresh(info()); };
    $("memory-prev").onclick = () => seek(step - 1);
    $("memory-next").onclick = () => seek(step + 1);
    $("memory-slider").oninput = event => seek(Number(event.target.value));
    $("memory-jump").onchange = event => {
      if (event.target.value) seek(Number(event.target.value));
      event.target.value = "";
    };
    $("memory-follow").onchange = () => {
      if ($("memory-follow").checked) { step = events.length; render(); }
    };
  }

  return {refresh, bindControls};
})();
