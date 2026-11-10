import { Chart, registerables, type ChartConfiguration } from "chart.js";
import "./style.css";

Chart.register(...registerables);

// ── Data types (written by export_web_data.py) ──────────────────────────────

interface Season {
  year: number;
  era: string;
  races: number;
  real_champion: string;
  model_champion: string;
  perfect_champion: string;
  baseline_correct: boolean;
  title_changes: boolean;
  differs_from_real: boolean;
  avg_points_gain: number;
}

interface Standing {
  driver: string;
  races: number;
  model_points: number;
  perfect_points: number;
  gain: number;
  model_rank: number;
  perfect_rank: number;
}

interface Era {
  name: string;
  start: number;
  end: number;
  rows: number;
  strategy: number;
  grid: number;
  ratio: number;
  features: Record<string, number>;
}

interface Summary {
  seasons: number;
  driver_races: number;
  test_r2: number;
  ratio_first: number;
  ratio_last: number;
  strategy_changes: number[];
  differs_from_real: number[];
}

type Mode = "strategy" | "real";
type Tab = "eras" | "seasons";

// ── Embedding ────────────────────────────────────────────────────────────────

// Inside an iframe the window fills the frame with no desktop around it
const embedded = window.self !== window.top;
if (embedded) document.documentElement.classList.add("embedded");

// Tell the parent page how tall the content is, so it can resize the iframe.
// Parent pages that don't listen just get a scrollable frame.
function reportHeight() {
  if (!embedded) return;
  const height = Math.ceil(document.documentElement.scrollHeight);
  window.parent.postMessage({ type: "boxbox:height", height }, "*");
}
new ResizeObserver(reportHeight).observe(document.body);

// ── Chart look: flat colours, no animation, square points ───────────────────

const C = {
  navy: "#000080",
  maroon: "#800000",
  green: "#008000",
  grey: "#808080",
  black: "#000000",
};

Chart.defaults.animation = false;
Chart.defaults.locale = "en-US";
Chart.defaults.color = C.black;
Chart.defaults.font.family = "Tahoma, Verdana, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.borderColor = "rgba(0, 0, 0, 0.25)";
Chart.defaults.elements.bar.borderWidth = 1;
Chart.defaults.elements.bar.borderColor = C.black;
Chart.defaults.elements.point.pointStyle = "rect";
Chart.defaults.elements.line.tension = 0;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.boxHeight = 12;
Chart.defaults.plugins.legend.labels.useBorderRadius = false;

const tooltip = {
  backgroundColor: "#ffffe1",
  borderColor: C.black,
  borderWidth: 1,
  titleColor: C.black,
  bodyColor: C.black,
  cornerRadius: 0,
  caretSize: 0,
  padding: 6,
  displayColors: true,
  boxWidth: 10,
  boxHeight: 10,
};

const gridLines = { color: "rgba(0, 0, 0, 0.25)", lineWidth: 1 };
const axisBorder = { color: C.black, width: 1 };

// ── Helpers ──────────────────────────────────────────────────────────────────

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

async function load<T>(name: string): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!res.ok) throw new Error(`Could not load ${name}`);
  return res.json() as Promise<T>;
}

const surname = (name: string) => name.split(" ").slice(-1)[0];
const status = (text: string) => ($("statusText").textContent = text);

// ── State ────────────────────────────────────────────────────────────────────

let seasons: Season[] = [];
let standings: Record<string, Standing[]> = {};
let eras: Era[] = [];
let mode: Mode = "strategy";
let year = 1994;
let eraIndex = 0;
let eraChart: Chart | null = null;
let standingsChart: Chart | null = null;

const isHighlighted = (s: Season) =>
  mode === "strategy" ? s.baseline_correct && s.title_changes : s.differs_from_real;

// ── Tabs ─────────────────────────────────────────────────────────────────────

function showTab(tab: Tab) {
  (["eras", "seasons"] as Tab[]).forEach((t) => {
    const on = t === tab;
    $(`tab-${t}`).setAttribute("aria-selected", String(on));
    $(`tab-${t}`).tabIndex = on ? 0 : -1;
    $(`panel-${t}`).classList.toggle("active", on);
  });
  // Charts in a hidden panel have no size, so resize them when the panel opens
  if (tab === "eras") eraChart?.resize();
  if (tab === "seasons") standingsChart?.resize();
  status(tab === "eras" ? `${eras[eraIndex]?.name ?? ""} era selected` : seasonStatus());
  reportHeight();
}

// ── Summary ──────────────────────────────────────────────────────────────────

function renderKpis(sum: Summary) {
  const drop = Math.round((1 - sum.ratio_last / sum.ratio_first) * 100);
  const items = [
    { v: String(sum.seasons), l: "Seasons", c: "" },
    { v: sum.driver_races.toLocaleString("en-US"), l: "Driver-races", c: "" },
    { v: sum.test_r2.toFixed(2), l: "Test R² on unseen seasons", c: "" },
    { v: `-${drop}%`, l: "Strategy vs grid ratio, 1994 to 2024", c: "" },
    { v: `${sum.strategy_changes.length}/${sum.seasons}`, l: "Titles changed by strategy alone", c: "select" },
  ];
  $("kpis").innerHTML = items
    .map((k) => `<div class="kpi"><div class="kpi-v ${k.c}">${k.v}</div><div class="kpi-l">${k.l}</div></div>`)
    .join("");
  $("countStrategy").textContent = String(sum.strategy_changes.length);
  $("countReal").textContent = String(sum.differs_from_real.length);
}

// ── Era chart ────────────────────────────────────────────────────────────────

function renderEraChart() {
  const config: ChartConfiguration<"bar" | "line"> = {
    type: "bar",
    data: {
      labels: eras.map((e) => [e.name, `${e.start}-${e.end}`]),
      datasets: [
        { type: "bar", label: "Strategy importance", data: eras.map((e) => e.strategy), backgroundColor: C.maroon, yAxisID: "y" },
        { type: "bar", label: "Grid position importance", data: eras.map((e) => e.grid), backgroundColor: C.navy, yAxisID: "y" },
        {
          type: "line",
          label: "Strategy / grid ratio",
          data: eras.map((e) => e.ratio),
          borderColor: C.green,
          backgroundColor: C.green,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          yAxisID: "ratio",
          order: -1, // draw the line on top of the bars
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      onClick: (_evt, els) => {
        if (els.length) selectEra(els[0].index);
      },
      onHover: (evt, els) => {
        const t = evt.native?.target as HTMLElement | undefined;
        if (t) t.style.cursor = els.length ? "pointer" : "default";
      },
      plugins: {
        legend: { position: "bottom" },
        tooltip: { ...tooltip, callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.parsed.y as number).toFixed(3)}` } },
      },
      scales: {
        x: {
          grid: { display: false },
          border: axisBorder,
          ticks: {
            autoSkip: false,
            maxRotation: 0,
            font: (ctx) => ({ weight: ctx.index === eraIndex ? "bold" : "normal" }),
            color: (ctx) => (ctx.index === eraIndex ? C.navy : C.black),
          },
        },
        y: { beginAtZero: true, max: 0.8, grid: gridLines, border: axisBorder, title: { display: true, text: "Feature importance" } },
        ratio: {
          position: "right",
          beginAtZero: true,
          max: 1,
          grid: { display: false },
          border: axisBorder,
          title: { display: true, text: "Ratio", color: C.green },
          ticks: { color: C.green },
        },
      },
    },
  };
  eraChart = new Chart($<HTMLCanvasElement>("eraChart"), config as ChartConfiguration);
}

function selectEra(i: number) {
  eraIndex = i;
  eraChart?.update();
  const e = eras[i];
  // Features that never vary inside one era (era itself, safety car before 2018) have zero importance
  const feats = Object.entries(e.features)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  const max = feats[0][1] || 1;
  const strategyFeats = new Set(["Pit window error", "Window penalty", "Execution penalty"]);
  const verdict =
    e.ratio < 0.5
      ? "Car and driver quality clearly decide more than strategy."
      : "Strategy still matters almost as much as the starting position.";
  $("eraDetail").innerHTML = `
    <legend>${esc(e.name)}</legend>
    <div class="era-head"><b>${e.start}-${e.end}</b><span>${e.rows.toLocaleString("en-US")} driver-races</span></div>
    <div class="ratio-field"><div class="kpi-v">${e.ratio.toFixed(3)}</div></div>
    <div class="kpi-l">Strategy / grid ratio</div>
    <p class="hint">${verdict}</p>
    <div class="feat-list">
      ${feats
        .map(
          ([name, v]) => `
        <div class="feat">
          <span>${esc(name)}</span>
          <div class="feat-bar"><div style="width:${(v / max) * 100}%;background:${
            strategyFeats.has(name) ? C.maroon : name === "Grid position" ? C.navy : C.grey
          }"></div></div>
          <span class="num">${v.toFixed(3)}</span>
        </div>`
        )
        .join("")}
    </div>
    <div class="era-nav">
      ${eras.map((x, j) => `<button class="${j === i ? "active" : ""}" data-era="${j}">${esc(x.name)}</button>`).join("")}
    </div>`;
  $("eraDetail")
    .querySelectorAll<HTMLButtonElement>("[data-era]")
    .forEach((b) => b.addEventListener("click", () => selectEra(Number(b.dataset.era))));
  status(`${e.name} era selected`);
  reportHeight();
}

// ── Season explorer ──────────────────────────────────────────────────────────

function renderMode() {
  $("modeNote").textContent =
    mode === "strategy"
      ? "Counts a season only when the model picks the real champion with real strategy, and perfect strategy alone gives a different one."
      : "Counts every season where the perfect-strategy champion isn't the real one. This also includes seasons where the model already picked the wrong champion with real strategy.";
  renderTimeline();
  renderSelect();
}

function renderSelect() {
  const sel = $<HTMLSelectElement>("seasonSelect");
  sel.innerHTML = seasons
    .map((s) => `<option value="${s.year}">${s.year}${isHighlighted(s) ? " *" : ""} - ${esc(s.real_champion)}</option>`)
    .join("");
  sel.value = String(year);
}

function renderTimeline() {
  $("timeline").innerHTML = seasons
    .map((s) => {
      const cls = [isHighlighted(s) ? "hl" : "", s.year === year ? "active" : ""].join(" ");
      const label = `${s.year}${isHighlighted(s) ? ": champion changes" : ""}`;
      return `<button class="${cls}" data-year="${s.year}" title="${label}" aria-label="${label}" aria-pressed="${s.year === year}">${String(s.year).slice(2)}</button>`;
    })
    .join("");
  $("timeline")
    .querySelectorAll<HTMLButtonElement>("[data-year]")
    .forEach((b) => b.addEventListener("click", () => selectSeason(Number(b.dataset.year))));
}

function verdictFor(s: Season): { tone: string; icon: string; text: string } {
  if (s.baseline_correct && s.title_changes)
    return {
      tone: "changed",
      icon: "!",
      text: `With perfect strategy for everyone, ${esc(s.perfect_champion)} wins instead of ${esc(s.real_champion)}. Strategy errors made the difference in this title.`,
    };
  if (s.baseline_correct)
    return { tone: "ok", icon: "i", text: `${esc(s.real_champion)} wins either way. Strategy errors didn't decide this title.` };
  if (s.title_changes)
    return {
      tone: "",
      icon: "i",
      text: `The model already picks ${esc(s.model_champion)} instead of the real champion with real strategy, and perfect strategy changes it again to ${esc(s.perfect_champion)}. That says more about the model than about strategy, so this season isn't counted as a strategy change.`,
    };
  return {
    tone: "",
    icon: "i",
    text: `The model picks ${esc(s.model_champion)} instead of the real champion even with real strategy. Perfect strategy doesn't change its pick, so strategy didn't decide this one.`,
  };
}

function seasonStatus(): string {
  const s = seasons.find((x) => x.year === year);
  if (!s) return "Ready";
  return `${s.year}: ${s.baseline_correct && s.title_changes ? "champion changes with perfect strategy" : "same champion with perfect strategy"}`;
}

function selectSeason(y: number) {
  year = y;
  if (location.hash !== `#${y}`) history.replaceState(null, "", `#${y}`);
  const s = seasons.find((x) => x.year === y)!;
  const table = standings[String(y)] ?? [];
  const v = verdictFor(s);

  $<HTMLSelectElement>("seasonSelect").value = String(y);
  $<HTMLButtonElement>("prevSeason").disabled = y <= seasons[0].year;
  $<HTMLButtonElement>("nextSeason").disabled = y >= seasons[seasons.length - 1].year;

  $("seasonCard").innerHTML = `
    <legend>${s.year} season</legend>
    <p class="hint" style="margin:0 0 8px">${esc(s.era)} era, ${s.races} races with pit stop data</p>
    <div class="champs">
      <span>Real champion</span><span class="field">${esc(s.real_champion)}</span>
      <span>Model, real strategy</span><span class="field ${s.baseline_correct ? "" : "off"}">${esc(s.model_champion)}</span>
      <span>Model, perfect strategy</span><span class="field ${s.title_changes ? "changed" : ""}">${esc(s.perfect_champion)}</span>
    </div>
    <div class="verdict ${v.tone}"><span class="verdict-icon" aria-hidden="true">${v.icon}</span><span>${v.text}</span></div>
    <div class="season-stat">Average gain from perfect strategy: <b>${s.avg_points_gain.toFixed(2)}</b> points per driver per race</div>`;

  $("standingsBody").innerHTML = table
    .map((r, i) => {
      const move = r.model_rank - r.perfect_rank;
      const arrow =
        move > 0 ? `<span class="up">&#9650; ${move}</span>` : move < 0 ? `<span class="down">&#9660; ${-move}</span>` : `<span class="dim">-</span>`;
      const champ = r.driver === s.real_champion ? '<span class="tag">real champion</span>' : "";
      return `<tr class="${i === 0 ? "lead" : ""}">
        <td class="num">${r.model_rank}</td>
        <td>${esc(r.driver)}${champ}</td>
        <td class="num">${r.races}</td>
        <td class="num">${r.model_points.toFixed(1)}</td>
        <td class="num">${r.perfect_points.toFixed(1)}</td>
        <td class="num ${r.gain >= 0 ? "pos" : "neg"}">${r.gain >= 0 ? "+" : ""}${r.gain.toFixed(1)}</td>
        <td class="num">${arrow}</td>
      </tr>`;
    })
    .join("");

  renderStandingsChart(table.slice(0, 10));
  renderTimeline();
  status(seasonStatus());
  reportHeight();
}

function renderStandingsChart(top: Standing[]) {
  const labels = top.map((r) => surname(r.driver));
  if (standingsChart) {
    standingsChart.data.labels = labels;
    standingsChart.data.datasets[0].data = top.map((r) => r.model_points);
    standingsChart.data.datasets[1].data = top.map((r) => r.perfect_points);
    standingsChart.update();
    return;
  }
  standingsChart = new Chart($<HTMLCanvasElement>("standingsChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Real strategy", data: top.map((r) => r.model_points), backgroundColor: C.navy },
        { label: "Perfect strategy", data: top.map((r) => r.perfect_points), backgroundColor: C.maroon },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        title: { display: true, text: "Top 10, model points", align: "start", font: { weight: "bold" } },
        tooltip: { ...tooltip, callbacks: { label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.parsed.x as number).toFixed(1)} pts` } },
      },
      scales: {
        x: { beginAtZero: true, grid: gridLines, border: axisBorder, title: { display: true, text: "Model points" } },
        y: { grid: { display: false }, border: axisBorder },
      },
    },
  });
}

// ── Start ────────────────────────────────────────────────────────────────────

async function main() {
  const [sum, s, st, e] = await Promise.all([
    load<Summary>("summary.json"),
    load<Season[]>("seasons.json"),
    load<Record<string, Standing[]>>("standings.json"),
    load<Era[]>("eras.json"),
  ]);
  seasons = s;
  standings = st;
  eras = e;

  renderKpis(sum);
  renderEraChart();
  selectEra(0);

  document.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((r) =>
    r.addEventListener("change", () => {
      mode = r.value as Mode;
      renderMode();
    })
  );
  $<HTMLSelectElement>("seasonSelect").addEventListener("change", (ev) =>
    selectSeason(Number((ev.target as HTMLSelectElement).value))
  );
  $("prevSeason").addEventListener("click", () => selectSeason(year - 1));
  $("nextSeason").addEventListener("click", () => selectSeason(year + 1));

  document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) =>
    b.addEventListener("click", () => showTab(b.dataset.tab as Tab))
  );
  $("tab-eras").parentElement!.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      const next: Tab = $("tab-eras").getAttribute("aria-selected") === "true" ? "seasons" : "eras";
      showTab(next);
      $(`tab-${next}`).focus();
      ev.preventDefault();
    }
  });

  const fromHash = Number(location.hash.slice(1));
  const linked = seasons.some((x) => x.year === fromHash);
  year = linked ? fromHash : sum.strategy_changes[0] ?? seasons[0].year;
  renderMode();
  selectSeason(year);
  showTab(linked ? "seasons" : "eras");

  document.addEventListener("keydown", (ev) => {
    if (!$("panel-seasons").classList.contains("active")) return;
    if ((ev.target as HTMLElement).closest("input, select, textarea, [role=tab]")) return;
    if (ev.key === "ArrowLeft" && year > seasons[0].year) selectSeason(year - 1);
    if (ev.key === "ArrowRight" && year < seasons[seasons.length - 1].year) selectSeason(year + 1);
  });
}

main().catch((err) => {
  document.querySelector(".window-body")!.insertAdjacentHTML(
    "afterbegin",
    `<div class="error">Could not load the results data. ${esc(String(err))}</div>`
  );
  status("Error");
});
