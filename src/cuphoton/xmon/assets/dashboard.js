// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";
(() => {
  const byId = id => document.getElementById(id);
  const element = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const valueText = value => value === null || value === undefined ? "Unavailable" :
    typeof value === "number" ? Number(value.toPrecision(7)).toString() : String(value);
  const activeStatus = status => ["queued", "pending", "running"].includes(status);
  const clockText = value => {
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? "" : date.toLocaleTimeString();
  };
  let state = JSON.parse(byId("state").textContent);
  let ui = state.ui || {live: false, can_ask: false};
  let selectedId = state.records[0]?.sample.id || null;
  let selectedMetrics = new Set();
  let metricNames = [];
  let refreshTimer;
  let refreshing = false;
  let connected = true;
  let submitting = false;
  let reportSignature = "";
  let displaySignature = "";
  let displayedDomain = null;
  const cancelling = new Set();
  const domainViews = new Map();
  document.querySelectorAll(".domain").forEach(panel => {
    const record = state.records[Number(panel.dataset.index)];
    if (record) domainViews.set(record.sample.id, {digest: record.digest, html: panel.innerHTML});
  });
  try {
    const requested = decodeURIComponent(location.hash.slice(1));
    if (state.records.some(record => record.sample.id === requested)) selectedId = requested;
  } catch (_) { /* An invalid bookmark does not prevent inspection. */ }

  function selectedRecord() {
    return state.records.find(record => record.sample.id === selectedId);
  }

  function selectSample(id) {
    if (!state.records.some(record => record.sample.id === id)) return;
    selectedId = id;
    history.replaceState(null, "", "#" + encodeURIComponent(id));
    renderSelection();
    renderReports(true);
    if (ui.live) refresh();
  }

  function renderTimeline() {
    const filter = byId("filter").value.toLowerCase();
    const category = byId("category").value;
    const nav = byId("timeline");
    const scroll = nav.scrollTop;
    nav.replaceChildren();
    for (const record of state.records) {
      const sample = record.sample;
      const text = sample.id + " " + sample.findings.map(f => f.message + " " + f.code).join(" ");
      if (!text.toLowerCase().includes(filter)) continue;
      if (category && !sample.findings.some(f => f.category === category)) continue;
      const button = element("button", sample.id,
        (sample.id === selectedId ? "selected " : "") + (sample.findings.length ? "flagged" : ""));
      button.type = "button";
      button.setAttribute("aria-pressed", String(sample.id === selectedId));
      button.append(element("span", "Sequence " + sample.sequence + " · revision " + sample.revision));
      button.append(element("span", sample.findings.length ? sample.findings.length + " finding(s)" : "No findings"));
      button.onclick = () => selectSample(sample.id);
      nav.append(button);
    }
    if (!nav.childElementCount) nav.append(element("p", "No samples match this filter.", "muted"));
    nav.scrollTop = scroll;
  }

  function renderDomain() {
    const record = selectedRecord();
    const container = byId("domain-panels");
    if (!record) { container.replaceChildren(); displayedDomain = null; return; }
    const cached = domainViews.get(record.sample.id);
    const current = cached && cached.digest === record.digest ? cached.html : null;
    if (displayedDomain && displayedDomain.id === record.sample.id &&
        displayedDomain.digest === record.digest && displayedDomain.html === current &&
        displayedDomain.connected === connected) return;
    displayedDomain = {id: record.sample.id, digest: record.digest, html: current, connected};
    const panel = element("section", undefined, "domain");
    panel.dataset.sampleId = record.sample.id;
    if (current !== null) {
      // Only trusted, application-authored HTML from this server is inserted.
      panel.innerHTML = current;
    } else {
      const message = !ui.live ? "This sample's application view is unavailable in this snapshot." :
        connected ? "Loading the current application view…" :
        "Connection interrupted. This sample's application view will load when the monitor reconnects.";
      panel.append(element("p", message, "muted"));
    }
    container.replaceChildren(panel);
  }

  function renderSelection() {
    renderTimeline();
    const record = selectedRecord();
    if (!record) {
      byId("sample-title").textContent = "Waiting for samples";
      for (const id of ["revision", "times", "provenance"]) byId(id).textContent = "";
      byId("measurements").replaceChildren();
      byId("findings").replaceChildren();
      renderDomain();
      renderPlots();
      updateControls();
      return;
    }
    const sample = record.sample;
    byId("sample-title").textContent = sample.id;
    byId("revision").textContent = "Revision " + sample.revision;
    const observation = sample.observed_start === null ? "not supplied" :
      valueText(sample.observed_start) + (sample.observed_end === null ? "" : "–" + valueText(sample.observed_end)) + " " + sample.time_unit;
    byId("times").textContent = "Sequence " + sample.sequence + " · Observation " + observation + " · Processed " + record.processed_at;
    byId("findings").replaceChildren(...sample.findings.map(f =>
      element("p", f.category + " / " + f.severity + " · " + f.message, "finding " + f.category)));
    if (!sample.findings.length) byId("findings").append(element("p", "No deterministic findings for this sample.", "muted"));
    byId("measurements").replaceChildren(...Object.entries(sample.measurements).map(([name, measurement]) => {
      const row = element("tr");
      [name, valueText(measurement.value), measurement.unit, measurement.status,
        measurement.coverage === null ? "—" : (100 * measurement.coverage).toFixed(1) + "%"]
        .forEach(value => row.append(element("td", value)));
      if (measurement.note) row.title = measurement.note;
      return row;
    }));
    byId("provenance").textContent = JSON.stringify({metadata: sample.metadata,
      artifacts: sample.artifacts, evidence_digest: record.digest}, null, 2);
    renderDomain();
    renderPlots();
    updateControls();
  }

  function updateMetricChoices() {
    const names = [...new Set(state.records.flatMap(record => Object.entries(record.sample.measurements)
      .filter(([, m]) => typeof m.value === "number").map(([name]) => name)))];
    if (JSON.stringify(names) === JSON.stringify(metricNames)) return;
    const first = metricNames.length === 0;
    metricNames = names;
    selectedMetrics = new Set([...selectedMetrics].filter(name => names.includes(name)));
    if (first) selectedMetrics = new Set(names.slice(0, 3));
    const choices = byId("metrics");
    choices.replaceChildren(element("legend", "Show up to six measurements"));
    names.forEach(name => {
      const label = element("label");
      const checkbox = element("input");
      checkbox.type = "checkbox";
      checkbox.value = name;
      checkbox.checked = selectedMetrics.has(name);
      checkbox.onchange = () => {
        if (checkbox.checked && selectedMetrics.size >= 6) { checkbox.checked = false; return; }
        if (checkbox.checked) selectedMetrics.add(name); else selectedMetrics.delete(name);
        updateMetricLimit();
        renderPlots();
      };
      label.append(checkbox, element("span", name));
      choices.append(label);
    });
    updateMetricLimit();
  }

  function updateMetricLimit() {
    byId("metric-count").textContent = selectedMetrics.size + " of " + metricNames.length + " measurements";
    byId("metrics").querySelectorAll("input").forEach(input => {
      input.disabled = !input.checked && selectedMetrics.size >= 6;
    });
  }

  function renderPlots() {
    const container = byId("trends");
    container.replaceChildren();
    if (!selectedMetrics.size) container.append(element("p", "Choose a measurement to display its history.", "muted"));
    for (const name of selectedMetrics) container.append(makePlot(name));
  }

  function makePlot(name) {
    const figure = element("figure", undefined, "trend-panel");
    const timed = byId("axis").value === "observed";
    const logarithmic = byId("scale").value === "log";
    const records = state.records.filter(record => Object.hasOwn(record.sample.measurements, name));
    const firstValid = records.find(record => record.sample.measurements[name].status === "valid" &&
      typeof record.sample.measurements[name].value === "number");
    const unit = firstValid?.sample.measurements[name].unit || "";
    const timeUnit = records.find(record => record.sample.observed_start !== null)?.sample.time_unit || "";
    figure.append(element("figcaption", name + (unit ? " · " + unit : " · unitless")));
    const points = state.records.map(record => ({record, m: record.sample.measurements[name],
      x: timed ? record.sample.observed_start : record.sample.sequence}));
    if (timed) points.sort((a, b) => (a.x ?? Infinity) - (b.x ?? Infinity));
    const usable = point => point.x !== null && Number.isFinite(point.x) &&
      (!timed || point.record.sample.time_unit === timeUnit) && point.m?.status === "valid" &&
      typeof point.m.value === "number" && Number.isFinite(point.m.value) && point.m.unit === unit &&
      (!logarithmic || point.m.value > 0);
    const valid = points.filter(usable);
    if (!valid.length) {
      figure.append(element("p", logarithmic ? "No positive valid measurements for this scale and time axis." :
        "No valid measurements for this time axis.", "muted"));
      return figure;
    }
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    const width = Math.max(320, Math.min(800, byId("trends").clientWidth));
    const left = 70, right = width - 30;
    svg.setAttribute("viewBox", "0 0 " + width + " 200");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", name + " by " + (timed ? "observation time" : "sample sequence"));
    const add = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      if (text !== undefined) node.textContent = text;
      svg.append(node);
      return node;
    };
    const xs = points.filter(point => point.x !== null && Number.isFinite(point.x) &&
      (!timed || point.record.sample.time_unit === timeUnit)).map(point => point.x);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const transformed = value => logarithmic ? Math.log10(value) : value;
    const values = valid.map(point => transformed(point.m.value));
    const low = Math.min(...values), high = Math.max(...values);
    const margin = (high - low || Math.abs(high) * .1 || 1) * .08;
    const ymin = low - margin, ymax = high + margin;
    const x = value => left + (value - xmin) / (xmax - xmin || 1) * (right - left);
    const y = value => 155 - (transformed(value) - ymin) / (ymax - ymin) * 125;
    const labels = logarithmic ? [Math.pow(10, low), Math.pow(10, high)] : [low, high];
    for (const value of [...new Set(labels)]) {
      add("line", {x1: left, y1: y(value), x2: right, y2: y(value), stroke: "var(--line)"});
      add("text", {x: left - 10, y: y(value) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11}, valueText(value));
    }
    add("line", {x1: left, y1: 160, x2: right, y2: 160, stroke: "var(--line-strong)"});
    add("text", {x: left, y: 180, fill: "var(--muted)", "font-size": 11}, valueText(xmin));
    add("text", {x: right, y: 180, fill: "var(--muted)", "font-size": 11, "text-anchor": "end"}, valueText(xmax));
    add("text", {x: (left + right) / 2, y: 197, fill: "var(--muted)", "font-size": 11, "text-anchor": "middle"},
      (timed ? "Observation start" + (timeUnit ? " (" + timeUnit + ")" : "") : "Sample sequence") + (logarithmic ? " · logarithmic values" : ""));
    let previous = null;
    for (const point of points) {
      if (!usable(point)) { previous = null; continue; }
      if (previous) add("line", {x1: x(previous.x), y1: y(previous.m.value), x2: x(point.x), y2: y(point.m.value), stroke: "var(--accent)", "stroke-width": 2});
      const sample = point.record.sample;
      const flagged = sample.findings.length > 0;
      const selected = sample.id === selectedId;
      const color = sample.findings.some(f => f.category === "processing") ? "#ff9a9a" :
        sample.findings.some(f => f.category === "data_quality") ? "#83c9e9" : "#edbc76";
      if (flagged) add("line", {x1: x(point.x), y1: 24, x2: x(point.x), y2: 160, stroke: color, "stroke-dasharray": "3 5", opacity: .3});
      const circle = add("circle", {cx: x(point.x), cy: y(point.m.value), r: selected ? 6 : 4,
        fill: selected ? "#edbc76" : flagged ? "var(--panel)" : "var(--accent)",
        stroke: flagged ? color : "var(--accent)", "stroke-width": flagged ? 2.5 : 1,
        class: "trend-point", tabindex: 0, role: "button", "aria-label": sample.id + ": " + name + " " + valueText(point.m.value) + " " + unit});
      const title = document.createElementNS(ns, "title");
      title.textContent = sample.id + " · " + valueText(point.m.value) + " " + unit +
        (flagged ? " · " + sample.findings.map(f => f.category + ": " + f.message).join("; ") : "");
      circle.append(title);
      circle.onclick = () => selectSample(sample.id);
      circle.onkeydown = event => {
        if (["Enter", " "].includes(event.key)) { event.preventDefault(); selectSample(sample.id); }
      };
      previous = point;
    }
    figure.append(svg);
    const excluded = points.length - valid.length;
    if (excluded) figure.append(element("p", excluded + " measurement(s) omitted: missing, invalid, incompatible units/time, or nonpositive on a logarithmic scale.", "muted"));
    return figure;
  }

  function canInteract() {
    return ui.live && ui.can_ask && (!ui.authentication_required || byId("access-token").value.length > 0);
  }

  function updateControls() {
    const home = byId("home-link");
    home.hidden = !ui.home_url;
    if (ui.home_url) {
      home.href = ui.home_url;
      home.textContent = "← " + (ui.home_label || "Back to site");
    }
    if (ui.description) byId("description").textContent = ui.description;
    byId("backend").textContent = ui.can_ask && ui.backend ? ui.backend : "";
    byId("access-control").hidden = !ui.live || !ui.can_ask || !ui.authentication_required;
    byId("interaction-status").textContent = !ui.live ?
      "Offline snapshot. Questions and cancellation require a connected monitor; saved reports remain available below." :
      !ui.can_ask ? ui.reason || "Read-only monitor. Investigation is not configured." :
      ui.authentication_required && !canInteract() ? "Enter the monitor access token to enable investigation." :
      "Questions use recorded evidence. New results and investigator actions appear automatically.";
    const disabled = !canInteract();
    byId("question").disabled = disabled;
    byId("scope").disabled = disabled;
    byId("question-submit").disabled = disabled || submitting || !state.records.length;
    byId("question-submit").textContent = submitting ? "Submitting…" : "Ask";
    byId("scope-note").textContent = byId("scope").value === "selected" ?
      "Evidence: " + (selectedId || "no sample selected") :
      "The investigator can inspect this run within its configured evidence limits.";
    byId("footer-mode").textContent = ui.live ? "Connected viewer · saved state refreshes automatically." : "Offline snapshot · saved measurements and reports.";
    document.querySelectorAll("button[data-cancel]").forEach(button => {
      button.disabled = disabled || cancelling.has(button.dataset.cancel);
    });
  }

  function renderReports(force = false) {
    const reports = state.reports || [];
    const related = byId("related-reports").checked;
    const signature = JSON.stringify([reports, related, related ? selectedId : null, canInteract()]);
    if (!force && signature === reportSignature) return;
    reportSignature = signature;
    const container = byId("reports");
    const opened = new Set([...container.querySelectorAll("details[open]")].map(node => node.dataset.detail));
    container.replaceChildren();
    const visible = reports.filter(report => !related || Object.hasOwn(report.evidence || {}, selectedId));
    if (!visible.length) container.append(element("p", related ? "No reports reference this sample yet." :
      "No saved questions or investigations yet. Deterministic measurements remain available.", "muted"));
    for (const report of [...visible].reverse()) {
      const box = element("div", undefined, "report " + report.status);
      box.dataset.reportId = report.id;
      const header = element("div", undefined, "report-head");
      const identity = element("div");
      identity.append(element("strong", report.question || "Investigation"));
      identity.append(element("p", String(report.status).toUpperCase() +
        (report.started_at ? " · " + new Date(report.started_at).toLocaleString() : ""), "report-meta"));
      header.append(identity);
      if (activeStatus(report.status) && ui.live && ui.can_ask) {
        const cancel = element("button", cancelling.has(report.id) ? "Cancelling…" : "Cancel");
        cancel.type = "button";
        cancel.dataset.cancel = report.id;
        cancel.disabled = !canInteract() || cancelling.has(report.id);
        cancel.onclick = () => cancelReport(report.id);
        header.append(cancel);
      }
      box.append(header);
      if (report.narrative) box.append(element("p", report.narrative, "report-narrative"));
      if (report.diagnostic) box.append(element("p", report.diagnostic, "error"));
      const events = report.events || [];
      if (events.length) {
        const actions = element("details");
        actions.dataset.detail = report.id + ":actions";
        actions.open = activeStatus(report.status) || opened.has(actions.dataset.detail);
        actions.append(element("summary", "Actions and progress · " + events.length));
        const list = element("ol", undefined, "report-events");
        events.forEach(event => {
          const row = element("li");
          if (event.at) row.append(element("time", clockText(event.at)));
          row.append(element("span", (event.tool ? event.tool + " · " : "") + (event.message || event.kind || "Update")));
          list.append(row);
        });
        actions.append(list);
        box.append(actions);
      }
      if ((report.citations || []).length) {
        const citations = element("details");
        citations.dataset.detail = report.id + ":citations";
        citations.open = opened.has(citations.dataset.detail);
        citations.append(element("summary", "Verified measurement citations · " + report.citations.length));
        report.citations.forEach(citation => {
          const button = element("button", citation.sample_id + " · revision " + citation.revision + " · " + citation.measurement + " = " + valueText(citation.value) + " " + citation.unit, "citation");
          button.type = "button";
          button.disabled = !state.records.some(record => record.sample.id === citation.sample_id);
          button.onclick = () => { selectSample(citation.sample_id); byId("sample-title").scrollIntoView({block: "start", behavior: "smooth"}); };
          citations.append(button);
        });
        box.append(citations);
      }
      const evidence = element("details");
      evidence.dataset.detail = report.id + ":evidence";
      evidence.open = opened.has(evidence.dataset.detail);
      evidence.append(element("summary", "Evidence revisions"), element("pre", JSON.stringify(report.evidence || {}, null, 2)));
      box.append(evidence);
      container.append(box);
    }
  }

  function renderState() {
    byId("run").textContent = state.run_id;
    byId("count").textContent = state.total_samples;
    byId("health").textContent = state.storage_error ? "Storage failure" : state.collector;
    byId("history").textContent = state.history_count;
    const observationAvailable = state.records.some(record => record.sample.observed_start !== null);
    const timeOption = byId("axis").querySelector('[value="observed"]');
    timeOption.disabled = !observationAvailable;
    timeOption.textContent = observationAvailable ? "Observation time" : "Observation time (unavailable)";
    if (!observationAvailable) byId("axis").value = "sequence";
    const notices = [];
    if (state.total_samples > state.records.length) notices.push("Showing the most recent " + state.records.length + " samples. Full history remains in saved state.");
    if (state.storage_error || ["failed", "interrupted"].includes(state.collector)) notices.push("Collection needs attention; displayed evidence may be incomplete.");
    byId("notice").textContent = notices.join(" ");
    byId("errors").replaceChildren(...state.errors.map(error =>
      element("p", error.source_id + " · " + error.error_type + " · " + error.processed_at, "error")));
    if (!state.errors.length) byId("errors").append(element("p", "No ingestion failures recorded.", "muted"));
    updateMetricChoices();
    renderSelection();
    renderReports();
  }

  async function post(path, payload) {
    const headers = {"Content-Type": "application/json"};
    if (ui.authentication_required) headers["X-XMon-Token"] = byId("access-token").value;
    const response = await fetch(path, {method: "POST", headers, body: JSON.stringify(payload), credentials: "same-origin", redirect: "error"});
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.error === "string" ? body.error : "Request was not accepted.");
    return body;
  }

  async function cancelReport(id) {
    if (!canInteract() || cancelling.has(id)) return;
    cancelling.add(id);
    renderReports(true);
    try {
      await post("api/cancel", {report_id: id});
      byId("question-status").textContent = "Cancellation requested.";
      await refresh();
    } catch (error) {
      byId("question-status").textContent = error.message || "Cancellation could not be requested.";
    } finally {
      cancelling.delete(id);
      renderReports(true);
    }
  }

  byId("question-form").onsubmit = async event => {
    event.preventDefault();
    const question = byId("question").value.trim();
    if (!canInteract() || !question || submitting) return;
    submitting = true;
    updateControls();
    byId("question-status").textContent = "Submitting question…";
    try {
      const scope = byId("scope").value;
      await post("api/ask", {question, scope, sample_ids: scope === "selected" ? [selectedId] : []});
      byId("question").value = "";
      byId("question-status").textContent = "Question accepted. Progress appears in the history below.";
      byId("related-reports").checked = false;
      await refresh();
    } catch (error) {
      byId("question-status").textContent = error.message || "The question could not be submitted.";
    } finally {
      submitting = false;
      updateControls();
    }
  };

  async function refresh() {
    if (!ui.live || refreshing) return;
    refreshing = true;
    clearTimeout(refreshTimer);
    let delay = Math.max(1, Number(ui.refresh_seconds) || 1) * 1000;
    const requestedId = selectedId;
    try {
      const query = requestedId ? "?sample_id=" + encodeURIComponent(requestedId) : "";
      const response = await fetch("api/dashboard" + query, {cache: "no-store", credentials: "same-origin", redirect: "error"});
      if (!response.ok) throw new Error("Saved monitor state is unavailable.");
      const next = await response.json();
      if (!Array.isArray(next.records) || !Array.isArray(next.reports)) throw new Error("The monitor returned an incomplete snapshot.");
      state = next;
      ui = state.ui || ui;
      connected = true;
      if (!state.records.some(record => record.sample.id === selectedId)) selectedId = state.records[0]?.sample.id || null;
      const view = state.domain_view;
      if (view && typeof view.html === "string") {
        const record = state.records.find(item => item.sample.id === view.sample_id);
        if (record && record.sample.revision === view.revision) domainViews.set(view.sample_id, {digest: record.digest, html: view.html});
      }
      const currentIds = new Set(state.records.map(record => record.sample.id));
      for (const id of domainViews.keys()) if (!currentIds.has(id)) domainViews.delete(id);
      const signature = JSON.stringify([state.records.map(record => [record.sample.id, record.digest]), state.collector, state.storage_error, state.errors, state.history_count, state.total_samples]);
      if (signature !== displaySignature) { displaySignature = signature; renderState(); }
      else { renderDomain(); renderReports(); updateControls(); }
      byId("connection").textContent = "Connected · updated " + new Date().toLocaleTimeString();
      byId("connection").classList.remove("error");
      if (requestedId !== selectedId) delay = 0;
    } catch (_) {
      connected = false;
      renderDomain();
      byId("connection").textContent = "Connection interrupted · showing last saved view";
      byId("connection").classList.add("error");
      delay = 5000;
    } finally {
      refreshing = false;
      if (ui.live) refreshTimer = setTimeout(refresh, delay);
    }
  }

  byId("filter").oninput = renderTimeline;
  byId("category").onchange = renderTimeline;
  byId("axis").onchange = renderPlots;
  byId("scale").onchange = renderPlots;
  byId("scope").onchange = updateControls;
  byId("access-token").oninput = () => { updateControls(); renderReports(); };
  byId("related-reports").onchange = () => renderReports(true);
  byId("skip-link").onclick = event => {
    event.preventDefault();
    byId("workspace").focus();
  };
  window.addEventListener("resize", renderPlots);
  window.addEventListener("hashchange", () => {
    try { selectSample(decodeURIComponent(location.hash.slice(1))); } catch (_) { /* Invalid bookmark. */ }
  });
  window.addEventListener("pagehide", () => { clearTimeout(refreshTimer); byId("access-token").value = ""; });
  renderState();
  if (ui.live) refresh();
})();
