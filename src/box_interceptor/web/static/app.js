// Пульт камер: получает по WebSocket цифры (JSON) и кадры (JPEG) и рисует их.
// Управление (какая сцена, какая модель) — обычными POST-запросами.

const $ = (sel) => document.querySelector(sel);
const CUSTOM = "__custom";

const state = {
  tiles: new Map(), // ключ камеры ("cam0") -> {img, count, extra}
  urls: new Map(), // ключ -> текущий blob: URL картинки (старые освобождаем)
  wallKeys: "",
  autostarted: false,
  synced: false,
};

// 1 человек, 2 человека, 5 человек
function people(n) {
  const m10 = n % 10;
  const m100 = n % 100;
  const few = m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14);
  return `${n} ${few ? "человека" : "человек"}`;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Сервер ответил ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------- запуск

async function init() {
  const opts = await (await fetch("/api/options")).json();

  const scene = $("#scene");
  for (const s of opts.scenes) {
    scene.add(new Option(s.description.split(":")[0] || s.name, s.name));
  }
  scene.add(new Option("Своё видео или камера…", CUSTOM));
  scene.addEventListener("change", () => {
    $("#source-field").hidden = scene.value !== CUSTOM;
  });

  const model = $("#model");
  for (const m of opts.models) model.add(new Option(m.label, m.id));

  $("#trace").checked = opts.overlays.trace;
  $("#heatmap").checked = opts.overlays.heatmap;
  for (const id of ["trace", "heatmap"]) {
    $(`#${id}`).addEventListener("change", (e) =>
      postJSON("/api/overlays", { [id]: e.target.checked }),
    );
  }

  $("#controls").addEventListener("submit", (e) => {
    e.preventDefault();
    start();
  });

  connect();
}

async function start() {
  const body = { model: $("#model").value };
  if ($("#scene").value === CUSTOM) {
    body.source = $("#source").value.trim();
    if (!body.source) {
      showNotice("Укажите источник: путь к видеофайлу, 0 для камеры Mac или адрес rtsp://…", "error");
      $("#source").focus();
      return;
    }
  } else {
    body.scene = $("#scene").value;
  }
  try {
    await postJSON("/api/start", body);
  } catch (err) {
    showNotice(`Не удалось запустить: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------- поток

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) =>
    typeof e.data === "string" ? onStats(JSON.parse(e.data)) : onFrame(e.data);
  ws.onclose = () => {
    showNotice("Нет связи с сервером. Переподключаюсь…", "error");
    setTimeout(connect, 1000);
  };
}

// Бинарное сообщение: [1 байт длина ключа][ключ][JPEG]
function onFrame(buffer) {
  const bytes = new Uint8Array(buffer);
  const n = bytes[0];
  const key = new TextDecoder().decode(bytes.subarray(1, 1 + n));
  const img = key === "top" ? $("#top") : state.tiles.get(key)?.img;
  if (!img) return;
  const url = URL.createObjectURL(new Blob([bytes.subarray(1 + n)], { type: "image/jpeg" }));
  const old = state.urls.get(key);
  state.urls.set(key, url);
  img.src = url;
  if (old) URL.revokeObjectURL(old);
}

// Страницу открыли, когда сервер уже что-то показывает — выставляем это в списках.
function syncControls(s) {
  if (state.synced || !s.scene) return;
  state.synced = true;
  if (s.scene === "custom") {
    $("#scene").value = CUSTOM;
    $("#source").value = s.description || "";
    $("#source-field").hidden = false;
  } else {
    $("#scene").value = s.scene;
  }
  if ([...$("#model").options].some((o) => o.value === s.model)) $("#model").value = s.model;
}

function onStats(s) {
  if (s.status === "idle") {
    if (!state.autostarted) {
      state.autostarted = true;
      start(); // открыли страницу — сразу показываем первую сцену
    }
    return;
  }
  syncControls(s);
  buildWall(s.cameras || []);

  if (s.status === "loading") showNotice(s.message || "Загрузка…");
  else if (s.status === "error") showNotice(s.message, "error");
  else hideNotice();

  if (s.status !== "running") return;
  renderSceneLine(s);
  renderCameras(s.cameras);
  renderFloor(s.floor);
  renderSpeed(s);
}

// ---------------------------------------------------------------- отрисовка

function buildWall(cameras) {
  const keys = cameras.map((c) => c.key).join(",");
  if (keys === state.wallKeys) return;
  state.wallKeys = keys;
  state.tiles.clear();
  const wall = $("#wall");
  wall.replaceChildren();
  wall.dataset.count = cameras.length;
  for (const c of cameras) {
    const fig = document.createElement("figure");
    fig.className = "cam";
    fig.innerHTML = `
      <img alt="Камера ${c.id}">
      <figcaption>
        <div class="cam-head"><span>Камера ${c.id}</span><span class="cam-count"></span></div>
        <p class="cam-extra"></p>
      </figcaption>`;
    wall.append(fig);
    state.tiles.set(c.key, {
      img: fig.querySelector("img"),
      count: fig.querySelector(".cam-count"),
      extra: fig.querySelector(".cam-extra"),
    });
  }
}

function renderSceneLine(s) {
  const where = s.total
    ? `кадр ${s.frame} из ${s.total}, проход ${s.loop}`
    : "живой поток";
  $("#scene-line").innerHTML = `<strong></strong> <span></span>`;
  $("#scene-line strong").textContent = s.description || s.scene;
  $("#scene-line span").textContent = `(${where})`;
}

function renderCameras(cameras) {
  for (const c of cameras) {
    const tile = state.tiles.get(c.key);
    if (!tile) continue;
    tile.count.textContent = people(c.people);
    const parts = [];
    for (const [name, n] of Object.entries(c.zones || {})) parts.push(`зона ${name}: ${n}`);
    for (const [name, l] of Object.entries(c.lines || {}))
      parts.push(`линия ${name}: вошли ${l.in}, вышли ${l.out}`);
    tile.extra.textContent = parts.join("; ");
  }
}

function renderFloor(floor) {
  const top = $("#top");
  top.hidden = !floor;
  $("#floor-none").hidden = Boolean(floor);
  const dl = $("#floor-figures");
  dl.replaceChildren();
  if (!floor) return;
  const rows = [["Людей на полу", floor.people]];
  for (const [name, n] of Object.entries(floor.zones || {})) rows.push([`В зоне ${name}`, n]);
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    dl.append(dt, dd);
  }
}

function renderSpeed(s) {
  const ai = Math.min(s.ai_fps, s.video_fps);
  if (s.live) {
    $("#speed-text").textContent = `ИИ обрабатывает ${ai.toFixed(0)} кадров в секунду.`;
  } else {
    $("#speed-text").textContent =
      `ИИ успевает ${ai.toFixed(0)} из ${s.video_fps.toFixed(0)} кадров в секунду.`;
  }
  $("#speed-bar").style.width = `${Math.min(100, (100 * ai) / s.video_fps)}%`;
  const cams = s.cameras.length;
  const dropped = s.live
    ? ""
    : s.dropped_pct > 1
      ? ` ${s.dropped_pct.toFixed(0)}% кадров пропущено, чтобы картинка не отставала от реального времени.`
      : " Кадры не пропускаются.";
  $("#speed-detail").textContent =
    `Один момент на ${cams === 1 ? "камере" : `всех ${cams} камерах`}: ${s.latency_ms.toFixed(0)} мс.` +
    dropped;
}

function showNotice(text, kind = "info") {
  const el = $("#notice");
  el.textContent = text;
  el.dataset.kind = kind;
  el.hidden = false;
}

function hideNotice() {
  $("#notice").hidden = true;
}

init().catch((err) => showNotice(`Не удалось загрузить пульт: ${err.message}`, "error"));
