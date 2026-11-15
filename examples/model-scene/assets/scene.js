// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(() => {
  function start() {
    function themePlots() {
      const dark = document.body.classList.contains('quarto-dark');
      const background = dark ? '#181b16' : '#ffffff';
      const text = dark ? '#e6eddf' : '#172b45';
      for (const doc of window.Bokeh?.documents ?? []) {
        for (const model of doc.all_models.values()) {
          if (model.type === 'Figure') {
            model.background_fill_color = background;
            model.border_fill_color = background;
          }
          if (model.type === 'Title') model.text_color = text;
          if (model.type.endsWith('Axis')) {
            model.axis_label_text_color = text;
            model.major_label_text_color = text;
          }
          if (model.type === 'Legend') {
            model.background_fill_color = background;
            model.label_text_color = text;
          }
          if (model.type === 'ColorBar') {
            model.background_fill_color = background;
            model.major_label_text_color = text;
            model.title_text_color = text;
          }
        }
      }
    }
    new MutationObserver(themePlots).observe(document.body,
      {attributes:true, attributeFilter:['class']});
    // Bokeh embeds initialize after DOM parsing; the load event follows them.
    addEventListener('load', themePlots);
    themePlots();
    const popup = document.createElement('div');
    popup.id = 'scene-definition'; popup.role = 'tooltip'; popup.hidden = true;
    document.body.appendChild(popup);
    let active = null;
    function close() {
      popup.hidden = true;
      if (active) active.removeAttribute('aria-describedby');
      active = null;
    }
    function open(term) {
      close(); active = term;
      popup.textContent = term.dataset.definition;
      term.setAttribute('aria-describedby', popup.id); popup.hidden = false;
      const rect = term.getBoundingClientRect();
      popup.style.left = Math.max(8, Math.min(rect.left,
        innerWidth - popup.offsetWidth - 8)) + 'px';
      popup.style.top = Math.max(8, Math.min(rect.bottom + 6,
        innerHeight - popup.offsetHeight - 8)) + 'px';
    }
    document.querySelectorAll('.scene-term').forEach(term => {
      let pointerWasOpen;
      term.addEventListener('pointerdown', () => { pointerWasOpen = active === term; });
      term.addEventListener('mouseenter', () => open(term));
      term.addEventListener('focus', () => open(term));
      term.addEventListener('mouseleave', close);
      term.addEventListener('blur', close);
      term.addEventListener('click', event => {
        event.stopPropagation();
        if (pointerWasOpen ?? active === term) close(); else open(term);
        pointerWasOpen = undefined;
      });
    });
    document.addEventListener('keydown', e => { if(e.key === 'Escape') close(); });
    document.addEventListener('click', close);
    addEventListener('scroll', close, {passive:true});
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', start); else start();
})();
