// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";
const state = JSON.parse(document.getElementById("state").textContent);
const byId = id => document.getElementById(id);
const element = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
const valueText = value => value === null ? "Unavailable" : typeof value === "number" ? Number(value.toPrecision(7)).toString() : String(value);
let selected = 0;
byId("run").textContent = state.run_id;
byId("count").textContent = state.total_samples;
byId("health").textContent = state.storage_error ? "Storage failure" : state.collector;
byId("history").textContent = state.history_count;
if (state.total_samples > state.records.length) byId("notice").textContent = "Showing the most recent 256 samples. Full history remains in saved state.";
if (state.storage_error || ["failed", "interrupted"].includes(state.collector)) byId("notice").textContent += " Collection needs attention; displayed evidence may be incomplete.";
function timeline() {
  const filter = byId("filter").value.toLowerCase(), category = byId("category").value;
  const nav = byId("timeline"); nav.replaceChildren();
  state.records.forEach((record, index) => {
    const sample = record.sample;
    if (!(sample.id + " " + sample.findings.map(x => x.message + " " + x.code).join(" ")).toLowerCase().includes(filter)) return;
    if (category && !sample.findings.some(f => f.category === category)) return;
    const button = element("button", sample.id, (index === selected ? "selected " : "") + (sample.findings.length ? "flagged" : ""));
    button.append(element("span", "Sequence " + sample.sequence + " · revision " + sample.revision));
    button.append(element("span", sample.findings.length ? sample.findings.length + " finding(s)" : "No findings"));
    button.onclick = () => { selected = index; show(); }; nav.append(button);
  });
}
function show() {
  timeline(); const record = state.records[selected]; if (!record) return;
  const sample = record.sample;
  byId("sample-title").textContent = sample.id;
  byId("revision").textContent = "Revision " + sample.revision;
  byId("times").textContent = "Sequence " + sample.sequence + " · Observation " + (sample.observed_start === null ? "not supplied" : sample.observed_start + (sample.observed_end === null ? "" : "–" + sample.observed_end) + " " + sample.time_unit) + " · Processed " + record.processed_at;
  byId("findings").replaceChildren(...sample.findings.map(f => element("p", f.category + " / " + f.severity + " · " + f.message, "finding " + f.category)));
  if (!sample.findings.length) byId("findings").append(element("p", "No deterministic findings for this sample.", "muted"));
  const rows = Object.entries(sample.measurements).map(([name, m]) => {const row = element("tr"); [name, valueText(m.value), m.unit, m.status, m.coverage === null ? "—" : (100*m.coverage).toFixed(1)+"%"].forEach(v => row.append(element("td", v))); if (m.note) row.title=m.note; return row;});
  byId("measurements").replaceChildren(...rows);
  document.querySelectorAll(".domain").forEach(panel => panel.hidden = Number(panel.dataset.index) !== selected);
  byId("provenance").textContent = JSON.stringify({metadata:sample.metadata, artifacts:sample.artifacts, evidence_digest:record.digest}, null, 2);
  const reports = state.reports.filter(report => Object.hasOwn(report.evidence, sample.id));
  byId("reports").replaceChildren();
  if (!reports.length) byId("reports").append(element("p", "No model report. Deterministic measurements remain available.", "muted"));
  reports.forEach(report => {const box=element("div", undefined, "report "+report.status); box.append(element("strong", report.status.toUpperCase()+" · "+report.question)); if(report.narrative)box.append(element("p", report.narrative)); if(report.diagnostic)box.append(element("p",report.diagnostic,"error")); box.append(element("pre",JSON.stringify({evidence:report.evidence,citations:report.citations || []},null,2)));byId("reports").append(box);});
  plot();
}
const names = [...new Set(state.records.flatMap(r => Object.keys(r.sample.measurements)))];
names.forEach(name => {const option=element("option", name); option.value=name; byId("metric").append(option);});
function plot() {
  const svg=byId("trend"), name=byId("metric").value; svg.replaceChildren();
  const points=state.records.map((r,i)=>({x:r.sample.sequence,m:r.sample.measurements[name],i}));
  const valid=points.filter(p=>p.m && p.m.status==="valid" && typeof p.m.value==="number"); if(!valid.length)return;
  const unit=valid[0].m.unit; const values=valid.filter(p=>p.m.unit===unit);
  const xmin=Math.min(...points.map(p=>p.x)), xmax=Math.max(...points.map(p=>p.x));
  const ymin=Math.min(...values.map(p=>p.m.value)), ymax=Math.max(...values.map(p=>p.m.value));
  const x=p=>50+(p.x-xmin)/Math.max(1,xmax-xmin)*720, y=p=>140-(p.m.value-ymin)/(ymax-ymin || 1)*110;
  const ns="http://www.w3.org/2000/svg"; const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;svg.append(n);};
  add("line",{x1:50,y1:145,x2:770,y2:145,stroke:"#526379"});
  let previous=null; points.forEach(p=>{if(!p.m || p.m.status!=="valid" || typeof p.m.value!=="number" || p.m.unit!==unit){previous=null;return;}if(previous)add("line",{x1:x(previous),y1:y(previous),x2:x(p),y2:y(p),stroke:"#8fd39b","stroke-width":2});add("circle",{cx:x(p),cy:y(p),r:p.i===selected?6:3,fill:p.i===selected?"#edbc76":"#8fd39b"});previous=p;});
  add("text",{x:50,y:18,fill:"#a0b2c5","font-size":12},valueText(ymax)+" "+unit);add("text",{x:50,y:170,fill:"#a0b2c5","font-size":12},"Sequence "+xmin+" … "+xmax+" · minimum "+valueText(ymin));
}
byId("errors").replaceChildren(...state.errors.map(error=>element("p",error.source_id+" · "+error.error_type+" · "+error.processed_at,"error")));
if(!state.errors.length)byId("errors").append(element("p","No ingestion failures recorded.","muted"));
byId("filter").oninput=timeline;byId("category").onchange=timeline;byId("metric").onchange=plot;show();
