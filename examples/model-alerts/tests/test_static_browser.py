# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Chromium regression for offline Bokeh image updates."""

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from views import candidate_explorer


def test_static_callback_keeps_raw_pixels_and_handles_empty_filter(tmp_path):
    chromium = os.environ.get("CUPHOTON_MODEL_ALERTS_CHROMIUM")
    if not chromium:
        pytest.skip("set CUPHOTON_MODEL_ALERTS_CHROMIUM to test real browser")
    from bokeh.embed import file_html
    from bokeh.resources import INLINE

    rows = [
        dict(candidate_id=i, snr=8.0, family="appearance") for i in range(4)
    ]
    layout, _ = candidate_explorer(
        np.random.default_rng(13).normal(size=(4, 3, 31, 31)),
        rows,
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.6, 0.3]),
        0.5,
        evaluation=dict(
            labels=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            scores=np.array([0.1, 0.9, 0.6, 0.3, 0.1, 0.9, 0.6, 0.3]),
            snr=np.full(8, 8.0),
            candidate_id=np.arange(8).astype(str),
            family=["appearance"] * 8,
        ),
    )
    page = file_html(layout, INLINE, "Model Alerts browser regression")
    capture = """<script>
window._errors=[]; window._network=0;
window.addEventListener('unhandledrejection',e=>_errors.push(String(e.reason)));
window.addEventListener('error',e=>_errors.push(e.message));
window.WebSocket=function(){_network++;throw Error('unexpected WebSocket')};
window.fetch=function(){_network++;throw Error('unexpected fetch')};
window.XMLHttpRequest=function(){_network++;throw Error('unexpected XHR')};
</script>"""
    exercise = """<script>
setTimeout(async()=>{
const report={};
try {
 const models=Array.from(Bokeh.documents[0]._all_models.values());
 const get=title=>models.find(m=>m.title===title);
 const sources=models.filter(m=>m.data && m.data.raw_flat);
 const before=sources.map(s=>Array.from(s.data.raw_flat[0]));
 const callback=models.find(m=>m.code?.includes('const t=threshold.value'));
 const limits=callback.args.mappers.map(m=>[m.low,m.high]);
 get('Display units').value='robust_z';
 await new Promise(r=>setTimeout(r,100));
 const after=sources.map(s=>Array.from(s.data.raw_flat[0]));
 report.unchanged=JSON.stringify(before)===JSON.stringify(after);
 report.transformed=sources.every(s=>s.data.transform[0]==='robust_z');
 report.scales_unchanged=callback.args.mappers.every((m,i)=>{
  const sigma=callback.args.sources[i].data.robust_sigma[0];
  const center=callback.args.sources[i].data.robust_center[0];
  return Math.abs(m.low*sigma+center-limits[i][0])<1e-10 &&
         Math.abs(m.high*sigma+center-limits[i][1])<1e-10;
 });
 get('Exploratory score threshold').value=1;
 get('Gallery outcome at current threshold').value='TP';
 await new Promise(r=>setTimeout(r,100));
 report.empty=sources.every(s=>s.data.image.length===0);
 report.population=callback.args.queue.text.includes('FN 4, TN 4');
} catch(e){report.error=String(e);}
report.errors=window._errors;report.network=window._network;
const pre=document.createElement('pre');pre.id='browser-report';
pre.textContent=JSON.stringify(report);document.body.appendChild(pre);
},1500);
</script>"""
    page = page.replace("<head>", "<head>" + capture, 1)
    before, after = page.rsplit("</body>", 1)
    path = tmp_path / "offline.html"
    path.write_text(before + exercise + "</body>" + after)
    result = subprocess.run(
        [
            chromium,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--dump-dom",
            "--virtual-time-budget=5000",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    matches = re.findall(
        r'<pre id="browser-report">(.*?)</pre>', result.stdout, re.S
    )
    assert matches, result.stderr
    report = json.loads(html.unescape(matches[-1]))
    assert report == dict(
        unchanged=True,
        transformed=True,
        scales_unchanged=True,
        population=True,
        empty=True,
        errors=[],
        network=0,
    )
