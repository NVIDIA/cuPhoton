# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Browser adapters over data-only primitives in the public xScan source."""

import numpy as np
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    LinearColorMapper,
    Range1d,
    Select,
)
from bokeh.palettes import Greys256, RdBu11

from cuphoton.xscan.raw_compare_review import (
    _image_figure,
    _image_payload,
    _pixel_inspector_callback,
    _pixel_label_source,
)


def linked_panels(cases, labels, *, signed=(), units="electrons / exposure"):
    """Select saved experiments, retaining shared zoom, taps and raw values.

    ``cases`` maps each explicit precomputed experiment name to a sequence
    of equally shaped numeric images. Display bounds stay fixed across cases.
    No paths, fixtures, server callbacks, or implicit data loading are used.
    """
    names = list(cases)
    images = list(cases.values())
    shape = np.asarray(images[0][0]).shape
    if any(np.asarray(a).shape != shape for group in images for a in group):
        raise ValueError("Linked panels need identical array shapes")
    height, width = shape
    xr, yr = Range1d(0, width), Range1d(0, height)
    selection = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
    sources, mappers, annotations, plots, banks = [], [], [], [], []
    unsigned_values = np.concatenate(
        [
            np.asarray(group[i]).ravel()
            for group in images
            for i in range(len(labels))
            if i not in signed
        ]
    )
    shared_low, shared_high = (
        float(unsigned_values.min()),
        float(unsigned_values.max()),
    )
    for index, label in enumerate(labels):
        payloads = [_image_payload(group[index]) for group in images]
        source = ColumnDataSource(payloads[0])
        bank = ColumnDataSource(
            {
                key: [payload[key][0] for payload in payloads]
                for key in payloads[0]
            }
        )
        values = np.concatenate(
            [np.asarray(group[index]).ravel() for group in images]
        )
        if index in signed:
            bound = float(np.max(np.abs(values))) or 1
            mapper = LinearColorMapper(palette=RdBu11, low=-bound, high=bound)
        else:
            mapper = LinearColorMapper(
                palette=Greys256,
                low=shared_low,
                high=shared_high,
            )
        annotation = _pixel_label_source()
        plot = _image_figure(
            f"{label} [{units}]",
            source,
            mapper,
            selection,
            annotation,
            shared_x_range=xr,
            shared_y_range=yr,
            stamp_width=width,
            stamp_height=height,
        )
        plot.name = f"model-scene-panel-{index}"
        sources.append(source)
        banks.append(bank)
        mappers.append(mapper)
        annotations.append(annotation)
        plots.append(plot)
    inspector = _pixel_inspector_callback(
        labels=labels,
        sources=sources,
        mappers=mappers,
        label_sources=annotations,
        selection_source=selection,
        stamp_width=width,
        stamp_height=height,
    )
    for plot in plots:
        plot.js_on_event(Tap, inspector)
    selector = Select(
        title="Precomputed experiment", value=names[0], options=names
    )
    selector.js_on_change(
        "value",
        CustomJS(
            args=dict(
                names=names,
                sources=sources,
                banks=banks,
                annotations=annotations,
                selection=selection,
            ),
            code="""
selection.data = {x:[],y:[],width:[],height:[]};
const i = names.indexOf(cb_obj.value);
for (let j=0; j<sources.length; ++j) {
  const data = {};
  for (const key of Object.keys(banks[j].data))
    data[key] = [banks[j].data[key][i]];
  sources[j].data = data;
  sources[j].change.emit();
  annotations[j].data = {x:[],y:[],x_offset:[],y_offset:[],text:[]};
}
""",
        ),
    )
    display_range = Select(
        title="Display range (shared intensity scale)",
        value="Full",
        options=["Full", "1–99 percentile"],
    )
    clipped_low, clipped_high = np.percentile(unsigned_values, [1, 99])
    display_range.js_on_change(
        "value",
        CustomJS(
            args=dict(
                mappers=[m for i, m in enumerate(mappers) if i not in signed],
                low=shared_low,
                high=shared_high,
                clipped_low=float(clipped_low),
                clipped_high=float(clipped_high),
            ),
            code="""
for (const mapper of mappers) {
  mapper.low = cb_obj.value === 'Full' ? low : clipped_low;
  mapper.high = cb_obj.value === 'Full' ? high : clipped_high;
}
""",
        ),
    )
    exact = Div(
        text="<output>Tap a pixel for full-precision values.</output>",
        styles={"font-family": "monospace", "white-space": "pre-wrap"},
    )
    selection.js_on_change(
        "data",
        CustomJS(
            args=dict(
                sources=sources,
                labels=labels,
                exact=exact,
                selection=selection,
                width=width,
            ),
            code="""
if (!selection.data.x.length) {
  exact.text = '<output>Tap a pixel for full-precision values.</output>';
} else {
  const col=Math.floor(selection.data.x[0]);
  const row=Math.floor(selection.data.y[0]);
  const index=row*width+col;
  const values=sources.map((source,i) => {
    const flat=source.data.flat[0];
    const value=typeof flat.get === 'function'
      ? flat.get(index) : flat[index];
    return labels[i]+': '+Number(value).toString();
  });
  const lines=['Pixel ('+col+', '+row+')', ...values];
  exact.text='<output aria-live="polite">'+
    lines.join(String.fromCharCode(10))+'</output>';
}
""",
        ),
    )
    return column(
        Div(
            text="<p>Pan and zoom are linked. Tap a pixel for exact saved "
            "values. Selection changes show saved experiments.</p>"
        ),
        row(selector, display_range),
        row(*plots, sizing_mode="stretch_width"),
        exact,
    )
