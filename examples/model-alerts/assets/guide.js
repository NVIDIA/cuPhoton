// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
  "use strict";
  const root = document.body;
  const original = new WeakMap();
  const sourceColors = new WeakMap();
  const styled = new WeakSet();
  const discovered = new WeakSet();
  const uiTypes = new Set(["Div", "Button", "CheckboxGroup", "RadioButtonGroup", "Slider", "Select", "Toolbar"]);

  // Variables cross shadow boundaries and follow dynamic readouts without
  // rewriting scientific callbacks or their text.
  const uiCSS = `
    :host { color: var(--model-alerts-plot-text) !important; color-scheme: inherit;
      --bokeh-icon-color: var(--model-alerts-plot-muted);
      --bokeh-icon-color-disabled: var(--model-alerts-plot-border); }
    .bk-btn, .bk-input { color: var(--model-alerts-plot-text); background: var(--model-alerts-panel-raised);
      border-color: var(--model-alerts-plot-border); }
    .bk-btn:hover { background: var(--model-alerts-panel); }
    .bk-btn.bk-active, .bk-btn-primary,
    .bk-btn.bk-active:hover, .bk-btn-primary:hover { color: var(--model-alerts-background);
      background: var(--model-alerts-accent); border-color: var(--model-alerts-accent); }
    .bk-slider-title, .bk-input-group { color: var(--model-alerts-plot-text); }
    .bk-toolbar-button { background: var(--model-alerts-panel-raised); }
    input { accent-color: var(--model-alerts-accent); }
    [style*="color:#f0f4eb"], [style*="color: #f0f4eb"],
    [style*="color:#172b45"], [style*="color: #172b45"] { color: var(--model-alerts-plot-text) !important; }
    [style*="color:#b7c9a9"], [style*="color: #b7c9a9"],
    [style*="color:#52647c"], [style*="color: #52647c"] { color: var(--model-alerts-plot-muted) !important; }
    [style*="background:#182310"], [style*="background:#f0f6fa"] {
      background: var(--model-alerts-panel-raised) !important; }
  `;

  function remember(model, property) {
    if (!original.has(model)) original.set(model, new Map());
    const values = original.get(model);
    if (!values.has(property)) values.set(property, model[property]);
    return values.get(property);
  }
  function palette() {
    const css = getComputedStyle(root);
    const color = (name, fallback) => css.getPropertyValue(`--model-alerts-plot-${name}`).trim() || fallback;
    return {
      background: color("background", "#ffffff"), text: color("text", "#20291b"),
      muted: color("muted", "#4f6043"), border: color("border", "#ccd7c2"),
      cyan: color("cyan", "#007f99"), pink: color("pink", "#ab3d75"),
    };
  }

  function styleDocument(doc, light, colors) {
    const replacements = new Map([
      ["#67d5e5", colors.cyan], ["#007f99", colors.cyan], ["#00bed6", colors.cyan],
      ["#ed91bd", colors.pink], ["#ab3d75", colors.pink],
      ["#82938a", colors.muted], ["#93a6b8", colors.muted], ["#d97a00", "#a35600"],
    ]);
    const replace = (value) => light && typeof value === "string"
      ? (replacements.get(value.toLowerCase()) || value) : value;
    for (const model of doc._all_models.values()) {
      const type = model.type;
      // Never theme image pixels, color maps, limits or palettes.
      if (["Image", "ImageRGBA", "LinearColorMapper"].includes(type)) continue;
      const set = (property, value, restoreDark = false) => {
        if (!(property in model.properties)) return;
        const before = remember(model, property);
        const after = restoreDark && !light ? before : value;
        if (model[property] !== after) model[property] = after;
      };
      if (type === "Figure" || type === "Plot") {
        set("background_fill_color", colors.background);
        set("border_fill_color", colors.background);
        set("outline_line_color", colors.border);
      } else if (type.endsWith("Axis")) {
        set("axis_label_text_color", colors.muted);
        set("major_label_text_color", colors.muted);
        set("axis_line_color", colors.border);
        set("major_tick_line_color", colors.border);
      } else if (type === "Grid") {
        set("grid_line_color", colors.border);
      } else if (type === "Title") {
        set("text_color", colors.text);
      } else if (type === "ColorBar") {
        set("background_fill_color", colors.background);
        set("major_label_text_color", colors.muted);
        set("title_text_color", colors.text);
        set("major_tick_line_color", colors.muted);
      } else if (type === "Legend") {
        set("background_fill_color", colors.background);
        set("label_text_color", colors.text);
        set("title_text_color", colors.text);
        set("border_line_color", colors.border);
      }
      // Change only known presentation colors, leaving field transforms intact.
      for (const property of ["line_color", "fill_color"]) {
        if (!(property in model.properties)) continue;
        const before = remember(model, property);
        if (typeof before === "string") {
          set(property, replace(before), true);
        } else if (before && typeof before.value === "string") {
          const value = replace(before.value);
          set(property, value === before.value ? before : {...before, value}, true);
        }
      }
      if (type === "ColumnDataSource" && Array.isArray(model.data.color)
          && model.data.color.every((value) => typeof value === "string")) {
        if (!sourceColors.has(model)) sourceColors.set(model, model.data.color.slice());
        const colors = sourceColors.get(model).map(replace);
        if (colors.some((value, index) => value !== model.data.color[index])) {
          // Numerical arrays and selections retain their current objects.
          model.data = {...model.data, color: colors};
        }
      }
      if (uiTypes.has(type) && !styled.has(model)) {
        model.stylesheets = [...model.stylesheets, uiCSS];
        styled.add(model);
      }
    }
  }

  function apply(theme) {
    root.dataset.modelAlertsTheme = theme;
    const light = theme === "light";
    const colors = palette();
    for (const doc of window.Bokeh?.documents || []) {
      styleDocument(doc, light, colors);
      discovered.add(doc);
    }
  }

  const sync = () => apply(document.body.classList.contains("quarto-dark") ? "dark" : "light");
  sync();
  new MutationObserver(sync).observe(document.body, {attributes: true, attributeFilter: ["class"]});

  // Bokeh initializes asynchronously. Discover each document once, stop after
  // ten seconds, and let subsequent theme clicks cover any later documents.
  let attempts = 80;
  function discoverDocuments() {
    const colors = palette();
    for (const doc of window.Bokeh?.documents || []) {
      if (!discovered.has(doc)) {
        styleDocument(doc, root.dataset.modelAlertsTheme === "light", colors);
        discovered.add(doc);
      }
    }
    if (--attempts > 0) window.setTimeout(discoverDocuments, 125);
  }
  discoverDocuments();
})();

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Definitions remain ordinary document content. Clicking, touching, printing
// and disabled JavaScript all retain the same internal definition links.
(() => {
  const links = document.querySelectorAll('a.model-alerts-glossary-term');
  if (!links.length) return;
  const tip = document.createElement('div');
  tip.id = 'model-alerts-glossary-tip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  document.body.append(tip);
  let active = null;
  let hideTimer;

  function hide() {
    window.clearTimeout(hideTimer);
    active?.removeAttribute('aria-describedby');
    active = null;
    tip.hidden = true;
  }

  function position() {
    if (!active || tip.hidden) return;
    const anchor = active.getBoundingClientRect();
    const box = tip.getBoundingClientRect();
    const margin = 12;
    const left = Math.max(margin, Math.min(
      anchor.left + anchor.width / 2 - box.width / 2,
      window.innerWidth - box.width - margin,
    ));
    const below = anchor.bottom + 9;
    const above = anchor.top - box.height - 9;
    const top = below + box.height <= window.innerHeight - margin ? below : above;
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(margin, Math.min(top, window.innerHeight - box.height - margin))}px`;
  }

  function show(link) {
    window.clearTimeout(hideTimer);
    const term = document.getElementById(link.hash.slice(1));
    const definition = term?.nextElementSibling;
    if (definition?.tagName !== 'DD') return;
    active?.removeAttribute('aria-describedby');
    active = link;
    const heading = document.createElement('strong');
    heading.textContent = term.textContent;
    const body = document.createElement('span');
    body.textContent = definition.textContent;
    tip.replaceChildren(heading, body);
    tip.hidden = false;
    link.setAttribute('aria-describedby', tip.id);
    position();
  }

  function later() {
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(hide, 180);
  }

  for (const link of links) {
    // Avoid a second native title popup only once this enhancement exists.
    link.removeAttribute('title');
    link.addEventListener('pointerenter', (event) => {
      if (event.pointerType !== 'touch') show(link);
    });
    link.addEventListener('pointerleave', () => {
      if (document.activeElement !== link) later();
    });
    link.addEventListener('focus', () => show(link));
    link.addEventListener('blur', later);
    link.addEventListener('click', () => {
      hide();
      const term = document.getElementById(link.hash.slice(1));
      // Preserve native anchor navigation/history; Back restores the reader.
      window.setTimeout(() => term?.focus({ preventScroll: true }), 0);
    });
  }
  tip.addEventListener('pointerenter', () => window.clearTimeout(hideTimer));
  tip.addEventListener('pointerleave', later);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hide();
  });
  document.addEventListener('pointerdown', (event) => {
    if (active && !active.contains(event.target) && !tip.contains(event.target)) hide();
  });
  document.addEventListener('scroll', position, true);
  window.addEventListener('resize', position);
})();
