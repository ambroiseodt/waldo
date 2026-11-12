// Explorer of the layer x module grid and BibTeX copy.
(() => {
  const D = window.WALDO_DATA;
  const SVG = "http://www.w3.org/2000/svg";
  const RULES = {
    vision: { name: "Act@100", module: 5, depth: 1 },
    language: { name: "LN2@60", module: 3, depth: 0.6 },
  };
  const DEFAULT_MODEL = { vision: "ViT L", language: "Qwen3 8B" };
  const MODULE_NAMES = ["first normalization", "attention", "residual stream after attention", "second normalization",
    "first FFN layer", "FFN activation", "FFN output", "residual stream"];
  const state = { mode: "vision", model: "ViT L", dataset: "Food-101", pickDepth: 0.5, pickModule: 7 };
  const $ = (id) => document.getElementById(id);
  const chart = $("explorer-chart");
  const tip = $("explorer-tip");
  const style = getComputedStyle(document.documentElement);
  const token = (name) => style.getPropertyValue(name).trim();

  const ramp = D.ramp.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  function color(t) {
    const x = Math.min(Math.max(t, 0), 1) * (ramp.length - 1);
    const i = Math.min(Math.floor(x), ramp.length - 2);
    const f = x - i;
    return "rgb(" + ramp[i].map((c, k) => Math.round(c + f * (ramp[i + 1][k] - c))).join(",") + ")";
  }
  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  function starPoints(cx, cy, r) {
    const points = [];
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      const rr = i % 2 ? r * 0.45 : r;
      points.push((cx + rr * Math.cos(a)).toFixed(1) + "," + (cy + rr * Math.sin(a)).toFixed(1));
    }
    return points.join(" ");
  }
  const pct = (v) => v.toFixed(1) + "%";
  const pp = (d) => (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(1) + " pp";

  function currentGrid() {
    return state.mode === "vision" ? D.vision.grids[state.model + "|" + state.dataset] : D.language.grids[state.model];
  }

  function fillControls() {
    const source = D[state.mode];
    $("explorer-model").innerHTML = source.models
      .map((m) => `<option${m === state.model ? " selected" : ""}>${m}</option>`)
      .join("");
    $("explorer-dataset-field").hidden = state.mode !== "vision";
    if (state.mode === "vision") {
      $("explorer-dataset").innerHTML = source.datasets
        .map((d) => `<option${d === state.dataset ? " selected" : ""}>${d}</option>`)
        .join("");
    }
    $("explorer-module").innerHTML = D.modules
      .map((m, i) => `<option value="${i}"${i === state.pickModule ? " selected" : ""}>${m}</option>`)
      .join("");
  }

  function render() {
    const grid = currentGrid();
    const L = grid.length;
    const M = D.modules.length;
    const pickLayer = Math.round(state.pickDepth * (L - 1));
    const slider = $("explorer-layer");
    slider.max = L - 1;
    slider.value = pickLayer;
    $("explorer-module").value = state.pickModule;
    const depth = (l) => Math.round((100 * l) / (L - 1));
    $("explorer-layer-label").textContent = `Layer ${pickLayer} of ${L - 1} (${depth(pickLayer)}% depth)`;

    const width = chart.clientWidth;
    const left = 46, right = 14, top = 14, cellH = 26, gap = 2;
    const cw = (width - left - right) / L;
    const height = top + M * cellH + 46;
    const values = grid.flat();
    const hi = Math.max(...values);
    const base = grid[L - 1][M - 1];
    let best = [0, 0];
    grid.forEach((row, l) => row.forEach((v, m) => { if (v > grid[best[0]][best[1]]) best = [l, m]; }));
    const rule = RULES[state.mode];
    const ruleLayer = Math.floor(rule.depth * (L - 1));
    const cx = (l) => left + (l + 0.5) * cw;
    const cy = (m) => top + m * cellH + (cellH - gap) / 2;

    chart.querySelector("svg")?.remove();
    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
      "aria-label": `Probing accuracy of ${state.model} at each layer and module. Cells above standard probing are purple.`,
    });
    D.modules.forEach((name, m) => {
      el("text", { x: left - 8, y: cy(m) + 4, "text-anchor": "end", class: "mod" }, svg).textContent = name;
      grid.forEach((row, l) => {
        const v = row[m];
        const rect = el("rect", {
          x: left + l * cw, y: top + m * cellH, width: Math.max(cw - (cw > 6 ? 1 : 0), 1), height: cellH - gap,
          fill: v > base ? color((v - base) / (hi - base)) : token("--waldo-below"),
        }, svg);
        rect.addEventListener("pointerenter", () => {
          tip.textContent = `${name}, ${MODULE_NAMES[m]} · layer ${l} (${depth(l)}%) · ${pct(v)} (${pp(v - base)})`;
          tip.style.left = Math.min(Math.max(cx(l), 120), width - 120) + "px";
          tip.style.top = top + m * cellH + "px";
          tip.style.display = "block";
        });
        rect.addEventListener("click", () => {
          state.pickDepth = L > 1 ? l / (L - 1) : 0;
          state.pickModule = m;
          render();
        });
      });
    });
    svg.addEventListener("pointerleave", () => { tip.style.display = "none"; });
    [0, 50, 100].forEach((p) => {
      el("text", { x: cx(Math.round((p / 100) * (L - 1))), y: top + M * cellH + 16, "text-anchor": "middle", class: "tick" }, svg)
        .textContent = p;
    });
    el("text", { x: left + (width - left - right) / 2, y: top + M * cellH + 36, "text-anchor": "middle", class: "tick" }, svg)
      .textContent = `Layer depth (%) · ${L} layers`;

    const ink = token("--waldo-ink");
    const none = { "pointer-events": "none" };
    el("rect", { x: left + pickLayer * cw - 1.5, y: top + state.pickModule * cellH - 1.5, width: Math.max(cw, 4) + 2, height: cellH - gap + 3,
      fill: "none", stroke: ink, "stroke-width": 2.5, rx: 2, ...none }, svg);
    const rx = cx(ruleLayer), ry = cy(rule.module);
    el("polygon", { points: `${rx},${ry - 8} ${rx + 8},${ry} ${rx},${ry + 8} ${rx - 8},${ry}`, fill: "none", stroke: ink, "stroke-width": 1.8, ...none }, svg);
    el("circle", { cx: cx(L - 1), cy: cy(M - 1), r: 6.5, fill: "#ffffff", stroke: ink, "stroke-width": 2, ...none }, svg);
    el("polygon", { points: starPoints(cx(best[0]), cy(best[1]), 9.5), fill: token("--waldo-star"), stroke: ink, "stroke-width": 0.8, ...none }, svg);
    chart.appendChild(svg);
    renderDepth(grid, pickLayer, width, left, right);

    const bestValue = grid[best[0]][best[1]];
    const oracle = bestValue - base;
    const share = (v) => (oracle > 0 ? `, ${Math.round((100 * Math.max(v - base, 0)) / oracle)}% of the oracle gain` : "");
    const pick = grid[pickLayer][state.pickModule];
    const ruleValue = grid[ruleLayer][rule.module];
    $("explorer-readout").innerHTML = `
      <div class="pick"><dt>▢ Your pick: ${D.modules[state.pickModule]} (${MODULE_NAMES[state.pickModule]}) at ${depth(pickLayer)}% depth</dt><dd>${pct(pick)}<small>${pp(pick - base)}${share(pick)}</small></dd></div>
      <div><dt>○ Standard probing, RC2@100</dt><dd>${pct(base)}<small>residual stream, last layer</small></dd></div>
      <div><dt>◇ ${rule.name}, fixed rule</dt><dd>${pct(ruleValue)}<small>${pp(ruleValue - base)}${share(ruleValue)}</small></dd></div>
      <div><dt>★ Best location, oracle</dt><dd>${pct(bestValue)}<small>${pp(oracle)}, ${D.modules[best[1]]} (${MODULE_NAMES[best[1]]}) at ${depth(best[0])}% depth</small></dd></div>`;
    $("explorer-legend").innerHTML = `
      <span><span class="swatch"></span>not better than standard probing</span>
      <span>better: +0<span class="ramp" style="background:linear-gradient(90deg,${D.ramp.join(",")})"></span>${pp(oracle)}</span>
      <span>${state.mode === "vision" ? "Top-1 test accuracy of linear probing" : "Average accuracy over the 24 classification tasks of MTEB(Multilingual)"}</span>`;
  }


  // Accuracy along depth: the residual stream, the module of the fixed rule and the picked module
  const MODULE_COLOR = { 5: "--waldo-act", 3: "--waldo-ln2", 7: "--waldo-rc2" };
  function renderDepth(grid, pickLayer, width, left, right) {
    const box = $("explorer-depth");
    box.querySelector("svg")?.remove();
    const L = grid.length, M = D.modules.length, ruleModule = RULES[state.mode].module;
    const shown = [...new Set([M - 1, ruleModule, state.pickModule])];
    const series = shown.map((m) => grid.map((row) => row[m]));
    const values = series.flat();
    const lo = Math.floor(Math.min(...values)), hi = Math.ceil(Math.max(...values));
    const top = 26, h = 130, height = top + h + 16;
    const x = (l) => left + ((l + 0.5) / L) * (width - left - right);
    const y = (v) => top + h - ((v - lo) / (hi - lo || 1)) * h;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
      "aria-label": `Accuracy along depth of ${shown.map((m) => D.modules[m]).join(", ")}.` });
    el("text", { x: left, y: 12 }, svg).textContent = "Accuracy (%) along depth";
    [lo, (lo + hi) / 2, hi].forEach((v) => {
      el("line", { x1: left, x2: width - right, y1: y(v), y2: y(v), stroke: "#ededed" }, svg);
      el("text", { x: left - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = Math.round(v);
    });
    const ink = token("--waldo-ink");
    let legendX = left + 170;
    shown.forEach((m, i) => {
      const color = MODULE_COLOR[m] ? token(MODULE_COLOR[m]) : ink;
      const dash = m === M - 1 ? { "stroke-dasharray": "5 4" } : {};
      const d = series[i].map((v, l) => `${l ? "L" : "M"}${x(l).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: color, "stroke-width": 2.2, ...dash }, svg);
      el("line", { x1: legendX, x2: legendX + 22, y1: 8, y2: 8, stroke: color, "stroke-width": 2.2, ...dash }, svg);
      el("text", { x: legendX + 28, y: 12 }, svg).textContent = D.modules[m] + (m === state.pickModule ? " (your pick)" : "");
      legendX += m === state.pickModule ? 140 : 70;
    });
    const rc2 = series[0];
    el("circle", { cx: x(L - 1), cy: y(rc2[L - 1]), r: 5, fill: "#ffffff", stroke: ink, "stroke-width": 2 }, svg);
    const pick = series[shown.indexOf(state.pickModule)];
    el("rect", { x: x(pickLayer) - 5, y: y(pick[pickLayer]) - 5, width: 10, height: 10, fill: "#ffffff", stroke: ink, "stroke-width": 2 }, svg);
    box.appendChild(svg);
  }

  document.querySelectorAll("#explorer .segmented button").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      state.model = DEFAULT_MODEL[state.mode];
      document.querySelectorAll("#explorer .segmented button")
        .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      fillControls();
      render();
    });
  });
  $("explorer-model").addEventListener("change", (e) => { state.model = e.target.value; render(); });
  $("explorer-dataset").addEventListener("change", (e) => { state.dataset = e.target.value; render(); });
  $("explorer-module").addEventListener("change", (e) => { state.pickModule = Number(e.target.value); render(); });
  $("explorer-layer").addEventListener("input", (e) => {
    const L = currentGrid().length;
    state.pickDepth = L > 1 ? Number(e.target.value) / (L - 1) : 0;
    render();
  });
  let frame;
  window.addEventListener("resize", () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(render); });
  fillControls();
  render();

  // Copy the BibTeX
  const copyButton = document.querySelector(".copy-bibtex-btn");
  copyButton.addEventListener("click", async () => {
    const label = copyButton.querySelector(".copy-text");
    try {
      await navigator.clipboard.writeText($("bibtex-code").textContent);
      label.textContent = "Copied";
    } catch (error) {
      label.textContent = "Select the text to copy";
    }
    setTimeout(() => { label.textContent = "Copy"; }, 2000);
  });
})();
