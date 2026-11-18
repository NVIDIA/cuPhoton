// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(() => {
  "use strict";
  const script = document.currentScript;
  const select = document.getElementById("cuphoton-version-select");
  const notice = document.getElementById("cuphoton-version-notice");
  const catalogURL = new URL(script.dataset.catalog, document.baseURI);
  const root = new URL("./", catalogURL);
  const current = script.dataset.version;
  const page = script.dataset.page;
  fetch(catalogURL, {cache: "no-cache"})
    .then(response => {
      if (!response.ok) throw new Error("Catalog unavailable");
      return response.json();
    })
    .then(catalog => {
      if (catalog.schema !== 1 || !Array.isArray(catalog.versions)) return;
      const entries = catalog.versions;
      if (!entries.some(entry => entry.version === current)) return;
      select.replaceChildren();
      for (const entry of entries) {
        const label = entry.version === "main" ? "main (development)" : entry.version;
        const option = new Option(label + (entry.version === catalog.stable ? " (stable)" : ""), entry.version);
        option.selected = entry.version === current;
        select.add(option);
      }
      select.disabled = false;
      select.addEventListener("change", () => {
        const entry = entries.find(item => item.version === select.value);
        const samePage = Object.hasOwn(entry.pages, page);
        const targetPage = samePage ? page : "index.html";
        const target = new URL(entry.path + targetPage, root);
        target.search = location.search;
        let anchor;
        try { anchor = decodeURIComponent(location.hash.slice(1)); } catch { anchor = ""; }
        if (samePage && entry.pages[page].includes(anchor)) target.hash = location.hash;
        location.assign(target);
      });
      if (catalog.stable && current !== catalog.stable) {
        const link = document.createElement("a");
        link.href = new URL(catalog.stable + "/index.html", root);
        link.textContent = "Read stable " + catalog.stable + ".";
        notice.replaceChildren(document.createTextNode(current === "main" ? "Development documentation. " : "An earlier release. "), link);
      }
    })
    .catch(() => { /* The server-rendered version and Versions link remain. */ });
})();
