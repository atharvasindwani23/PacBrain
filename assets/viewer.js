/* The dashboard reads checked-in evaluation artifacts; it never calls a model. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { data: {}, seed: 2000, variant: "tuned", still: matchMedia("(prefers-reduced-motion: reduce)").matches };
  const number = (value, digits = 0) => new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
  const signed = (value, digits = 0) => `${value > 0 ? "+" : value < 0 ? "−" : ""}${number(Math.abs(value), digits)}`;
  const percent = (value, digits = 0) => `${number(value * 100, digits)}%`;
  const name = () => state.variant === "tuned" ? "Fine-tuned" : "Fine-tuned · t04";
  const text = (id, value) => { $(id).textContent = value; };

  function validate(data, label) {
    if (data.label !== label || !Number.isInteger(data.games) || data.games <= 0 || !Array.isArray(data.per_game) || data.per_game.length !== data.games) throw new Error(`Invalid ${label} result structure`);
    for (const key of ["avg_score", "win_rate", "illegal_move_rate"]) if (!Number.isFinite(data[key])) throw new Error(`Invalid ${label} summary`);
    const seeds = new Set();
    for (const game of data.per_game) {
      if (!Number.isInteger(game.seed) || seeds.has(game.seed) || !Number.isFinite(game.score) || typeof game.win !== "boolean" || !Number.isInteger(game.moves) || game.moves < 0 || !Number.isInteger(game.illegal) || game.illegal < 0 || game.illegal > game.moves) throw new Error(`Invalid ${label} episode`);
      seeds.add(game.seed);
    }
    const scores = data.per_game.reduce((sum, game) => sum + game.score, 0) / data.games;
    const wins = data.per_game.filter((game) => game.win).length / data.games;
    const moves = data.per_game.reduce((sum, game) => sum + game.moves, 0);
    const illegal = data.per_game.reduce((sum, game) => sum + game.illegal, 0) / Math.max(moves, 1);
    if (Math.abs(scores - data.avg_score) > .001 || Math.abs(wins - data.win_rate) > .001 || Math.abs(illegal - data.illegal_move_rate) > .001) throw new Error(`Inconsistent ${label} summary`);
    return data;
  }

  function metricSummary() {
    const base = state.data.base, tuned = state.data[state.variant];
    const wins = (data) => data.per_game.filter((game) => game.win).length;
    text("hero-base-wins", `${wins(base)}/${base.games}`);
    text("hero-tuned-wins", `${wins(tuned)}/${tuned.games}`);
    text("avg-score", signed(tuned.avg_score, 1));
    text("score-baseline", `Base ${signed(base.avg_score, 1)}`);
    text("score-change", `${signed(tuned.avg_score - base.avg_score, 1)} points`);
    text("win-rate", percent(tuned.win_rate));
    text("wins-baseline", `Base ${percent(base.win_rate)}`);
    text("wins-count", `${wins(tuned)} of ${tuned.games} games`);
    text("invalid-rate", percent(tuned.illegal_move_rate, 1));
    text("invalid-baseline", `Base ${percent(base.illegal_move_rate, 1)}`);
    text("invalid-change", `${signed((tuned.illegal_move_rate - base.illegal_move_rate) * 100, 1)} pp`);
    text("sample-count", `N = ${base.games} + ${tuned.games}`);
  }

  function setStillFrame(which) {
    const img = $(`${which}-replay`), canvas = $(`${which}-still`);
    const absent = which === "tuned" && state.variant !== "tuned";
    if (state.still && img.complete && img.naturalWidth && !absent) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      canvas.hidden = false;
      img.hidden = true;
    } else {
      canvas.hidden = true;
      img.hidden = absent;
    }
  }

  function replay(which, data, restart = false) {
    const game = data.per_game.find((item) => item.seed === state.seed);
    if (!game) throw new Error("Selected replay seed is absent from results");
    const outcome = $(`${which}-outcome`);
    outcome.textContent = game.win ? "Board cleared ↗" : "Not cleared";
    outcome.classList.toggle("won", game.win);
    text(`${which}-seed-score`, signed(game.score));
    text(`${which}-seed-moves`, number(game.moves));
    text(`${which}-seed-invalid`, `${game.illegal}/${game.moves}`);
    text(`${which}-run-id`, `RUN / ${game.seed}`);
    const image = $(`${which}-replay`);
    const absent = which === "tuned" && state.variant !== "tuned";
    if (which === "tuned") $("no-tuned-replay").hidden = !absent;
    $(`${which}-image-error`).hidden = true;
    image.onload = () => setStillFrame(which);
    image.onerror = () => { $(`${which}-image-error`).hidden = false; };
    image.alt = `Recorded ${which === "base" ? "base" : "fine-tuned"} model playing Pac-Man, seed ${state.seed}`;
    if (!absent) {
      const path = `results/${which}_seed${state.seed}.gif`;
      if (restart || image.dataset.path !== path) {
        image.dataset.path = path;
        image.src = path + (restart ? `?replay=${Date.now()}` : "");
      }
    }
    setStillFrame(which);
  }

  function chart() {
    const base = state.data.base, tuned = state.data[state.variant];
    const paired = base.per_game.map((game) => ({ base: game, tuned: tuned.per_game.find((item) => item.seed === game.seed) }));
    if (paired.some((pair) => !pair.tuned)) throw new Error("Evaluation seeds do not match");
    const width = 1000, height = 262, left = 52, right = 15, top = 18, bottom = 31;
    const values = paired.flatMap((pair) => [pair.base.score, pair.tuned.score]);
    const lower = Math.min(-500, Math.floor(Math.min(...values) / 500) * 500);
    const upper = Math.max(1000, Math.ceil(Math.max(...values) / 500) * 500);
    const y = (value) => top + (upper - value) / (upper - lower) * (height - top - bottom);
    const zero = y(0), step = (width - left - right) / paired.length, barWidth = Math.min(22, step * .26);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("aria-hidden", "true");
    const node = (tag, attrs, content) => {
      const item = document.createElementNS(ns, tag);
      for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, String(value));
      if (content !== undefined) item.textContent = content;
      return item;
    };
    for (let value = lower; value <= upper; value += 500) {
      svg.append(node("line", { x1: left, x2: width - right, y1: y(value), y2: y(value), class: value === 0 ? "chart-grid chart-zero" : "chart-grid" }));
      svg.append(node("text", { x: left - 12, y: y(value) + 3, "text-anchor": "end", class: "chart-axis" }, number(value)));
    }
    paired.forEach((pair, i) => {
      const center = left + step * (i + .5);
      ["base", "tuned"].forEach((variant, index) => {
        const game = pair[variant], py = y(game.score);
        const rect = node("rect", { x: center + (index ? 3 : -barWidth - 3), y: Math.min(py, zero), width: barWidth, height: Math.max(Math.abs(py - zero), 1), rx: 2, fill: index ? "#f6d853" : "#71849c", class: "chart-bar" });
        rect.append(node("title", {}, `Seed ${game.seed} · ${variant === "base" ? "Base" : name()}: ${signed(game.score)} points; ${game.win ? "board cleared" : "not cleared"}`));
        svg.append(rect);
      });
      svg.append(node("text", { x: center, y: height - 9, "text-anchor": "middle", class: "chart-axis" }, pair.base.seed));
    });
    $("score-chart").replaceChildren(svg);
    const improved = paired.filter((pair) => pair.tuned.score > pair.base.score).length;
    $("score-chart").setAttribute("aria-label", `Scores for ${paired.length} matched seeds. Base average ${base.avg_score}; ${name()} average ${tuned.avg_score}. Full per-seed data is in the expandable table below.`);
    text("chart-variant-name", name());
    text("chart-insight", `${name()} scored higher on ${improved} of ${paired.length} seeds; ${tuned.per_game.filter((game) => game.win).length} boards cleared.`);
    const rows = paired.map((pair) => {
      const row = document.createElement("tr");
      const values = [pair.base.seed, signed(pair.base.score), signed(pair.tuned.score), signed(pair.tuned.score - pair.base.score), pair.base.win ? "Cleared" : "Not cleared", pair.tuned.win ? "Cleared" : "Not cleared"];
      values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
      return row;
    });
    $("data-table").replaceChildren(...rows);
    $("variant-source").href = `results/${state.variant}.json`;
    text("variant-source", `${state.variant}.json ↗`);
    $("variant-note").hidden = state.variant === "tuned";
  }

  function render(restart = false) {
    metricSummary();
    replay("base", state.data.base, restart);
    replay("tuned", state.data[state.variant], restart);
    chart();
    document.querySelectorAll("[data-seed]").forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.seed) === state.seed)));
    $("still-frames").setAttribute("aria-pressed", String(state.still));
  }

  function fail() {
    const status = $("load-status");
    status.hidden = false;
    status.classList.add("error");
    status.replaceChildren();
    const message = document.createElement("span");
    message.textContent = "The recorded data could not be loaded or validated. Serve the repository folder with ";
    const code = document.createElement("code");
    code.textContent = "python3 -m http.server 8000";
    const ending = document.createElement("span");
    ending.textContent = ", then open http://localhost:8000/viewer.html. The raw results remain available in results/.";
    status.append(message, code, ending);
    document.querySelectorAll("button,select").forEach((element) => { element.disabled = true; });
  }

  async function start() {
    try {
      const labels = ["base", "tuned", "tuned_t04"];
      const results = await Promise.all(labels.map(async (label) => {
        const response = await fetch(`results/${label}.json`);
        if (!response.ok) throw new Error("Missing evaluation artifact");
        return validate(await response.json(), label);
      }));
      results.forEach((result) => { state.data[result.label] = result; });
      render();
      $("load-status").hidden = true;
      document.querySelectorAll("[data-seed]").forEach((button) => button.addEventListener("click", () => { state.seed = Number(button.dataset.seed); render(); }));
      $("variant").addEventListener("change", (event) => { state.variant = event.target.value; render(); });
      $("restart").addEventListener("click", () => render(true));
      $("still-frames").addEventListener("click", () => { state.still = !state.still; render(!state.still); });
    } catch (error) { fail(); }
  }
  start();
})();
