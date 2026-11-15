# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scientific tables; full machine-readable inventories stay in receipts."""

from __future__ import annotations

import html
import json


def table(records, *, caption, collapsed=False):
    """Display explicit records without Pandas or implicit data reduction."""
    from IPython.display import HTML, display

    if isinstance(records, dict):
        records = [dict(quantity=k, value=v) for k, v in records.items()]
    records = list(records)
    if not records:
        display(HTML(f"<p>{html.escape(caption)}: no entries.</p>"))
        return
    columns = list(dict.fromkeys(k for row in records for k in row))

    def cell(value):
        if value is None:
            return "undefined"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (dict, list)):
            return html.escape(json.dumps(value, ensure_ascii=False))
        return html.escape(str(value))

    text = (
        '<div class="model-alerts-table"><table><caption>'
        + html.escape(caption)
        + "</caption><thead><tr>"
    )
    text += (
        "".join(
            '<th scope="col">' + html.escape(k) + "</th>" for k in columns
        )
        + "</tr></thead><tbody>"
    )
    for row in records:
        text += (
            "<tr>"
            + "".join("<td>" + cell(row.get(k)) + "</td>" for k in columns)
            + "</tr>"
        )
    text += "</tbody></table></div>"
    if collapsed:
        text = (
            "<details><summary>"
            + html.escape(caption)
            + "</summary>"
            + text
            + "</details>"
        )
    display(HTML(text))


def model_rows(outcomes):
    """Flatten model outcomes with explicit confusion denominators."""
    rows = []
    for name, result in outcomes.items():
        rows.append(
            dict(
                model=name,
                **{k: v for k, v in result.items() if k != "injections"},
                **result.get("injections", {}),
            )
        )
    return rows
