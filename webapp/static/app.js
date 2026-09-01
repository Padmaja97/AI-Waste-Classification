/* ═══════════════════════════════════════════════════════════════
   EcoSort AI — inspection line
   ═══════════════════════════════════════════════════════════════ */
(() => {
'use strict';

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

const C = {
  organic: '#3FBF6E',
  recycle: '#2E9FE0',
  signal:  '#E8A33D',
  line:    '#1E2B28',
  muted:   '#7C918B',
  faint:   '#4E605B',
};

/* ─────────────────────────────────────────────────────────────
   Toast
   ───────────────────────────────────────────────────────────── */
let toastTimer;
function toast(msg, kind = 'info') {
  const el = $('#toast');
  el.textContent = msg;
  el.dataset.kind = kind;
  el.classList.add('is-up');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('is-up'), 3600);
}

/* ─────────────────────────────────────────────────────────────
   1 · Hero canvas — the sorter
   Items drift in from the left, hit the inspection line, and are
   deflected up (organic) or down (recyclable). This is the thesis.
   ───────────────────────────────────────────────────────────── */
function initSorter() {
  const cv = $('#sorterCanvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  let W, H, items = [], raf, gateX;

  const DPR = Math.min(devicePixelRatio || 1, 2);

  function size() {
    const r = cv.getBoundingClientRect();
    W = r.width; H = r.height;
    cv.width = W * DPR; cv.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    gateX = W * 0.60;              // gate sits in the open right half
  }

  function spawn() {
    const organic = Math.random() < 0.56;   // dataset is 56% organic
    return {
      x: -20,
      y: H * (0.34 + Math.random() * 0.32), // wider intake band
      vx: 0.9 + Math.random() * 1.1,
      vy: 0,
      r: 1.8 + Math.random() * 2.8,
      organic,
      sorted: false,
      life: 1,
    };
  }

  function step() {
    ctx.clearRect(0, 0, W, H);

    // the inspection line
    ctx.strokeStyle = 'rgba(124,145,139,.16)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 7]);
    ctx.beginPath(); ctx.moveTo(gateX, H * 0.14); ctx.lineTo(gateX, H * 0.86); ctx.stroke();
    ctx.setLineDash([]);

    if (items.length < 150 && Math.random() < 0.85) items.push(spawn());

    for (let i = items.length - 1; i >= 0; i--) {
      const p = items[i];

      if (!p.sorted && p.x >= gateX) {
        p.sorted = true;
        p.vy = (p.organic ? -1 : 1) * (0.42 + Math.random() * 0.72);
      }
      p.x += p.vx;
      p.y += p.vy;
      if (p.sorted) p.life -= 0.006;

      if (p.x > W + 30 || p.life <= 0) { items.splice(i, 1); continue; }

      const col = p.sorted ? (p.organic ? C.organic : C.recycle) : C.faint;
      const a = p.sorted ? Math.max(0, p.life) * 0.95 : 0.62;

      // trail
      ctx.strokeStyle = col;
      ctx.globalAlpha = a * 0.28;
      ctx.lineWidth = p.r * 0.9;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(p.x - p.vx * 13, p.y - p.vy * 13);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();

      // head
      ctx.globalAlpha = a;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(step);
  }

  size();
  addEventListener('resize', size, { passive: true });

  if (REDUCED) {
    // one still frame, no motion
    for (let i = 0; i < 220; i++) {
      const p = spawn();
      p.x = Math.random() * W;
      if (p.x > gateX) { p.sorted = true; p.y += (p.organic ? -1 : 1) * (p.x - gateX) * 0.42; }
      items.push(p);
    }
    ctx.clearRect(0, 0, W, H);
    items.forEach(p => {
      ctx.fillStyle = p.sorted ? (p.organic ? C.organic : C.recycle) : C.faint;
      ctx.globalAlpha = 0.62;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
    return;
  }

  // pause when off-screen — no wasted frames
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) { if (!raf) raf = requestAnimationFrame(step); }
    else { cancelAnimationFrame(raf); raf = null; }
  }, { threshold: 0 });
  io.observe(cv);
}

/* ─────────────────────────────────────────────────────────────
   2 · Counters + scroll spy + reveal
   ───────────────────────────────────────────────────────────── */
function initCounters() {
  const nums = $$('.num[data-count]');
  const run = el => {
    const target = parseFloat(el.dataset.count);
    const dp = parseInt(el.dataset.dp, 10) || 0;
    if (REDUCED) { el.textContent = target.toLocaleString('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp }); return; }
    const t0 = performance.now(), dur = 1150;
    const tick = now => {
      const k = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * eased).toLocaleString('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp });
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  const io = new IntersectionObserver((es, o) => {
    es.forEach(e => { if (e.isIntersecting) { run(e.target); o.unobserve(e.target); } });
  }, { threshold: 0.6 });
  nums.forEach(n => io.observe(n));
}

function initSpy() {
  const links = $$('.nav__links a[data-spy]');
  const secs  = links.map(a => $('#' + a.dataset.spy)).filter(Boolean);
  const io = new IntersectionObserver(es => {
    es.forEach(e => {
      if (!e.isIntersecting) return;
      links.forEach(a => a.classList.toggle('is-on', a.dataset.spy === e.target.id));
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach(s => io.observe(s));
}

function initReveal() {
  if (REDUCED) return;
  const targets = $$('.sec__head, .bay, .tablewrap, .panel, .net, .gaps, .bins, .api, .readout');
  targets.forEach(t => t.classList.add('will-lift'));
  const io = new IntersectionObserver((es, o) => {
    es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('is-in'); o.unobserve(e.target); } });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
  targets.forEach(t => io.observe(t));
}

/* ─────────────────────────────────────────────────────────────
   3 · Validation-accuracy chart (real numbers from the run)
   ───────────────────────────────────────────────────────────── */
const RUN = {
  mobilenet: [
    { ep: 1, phase: 'head', trainAcc: .8220, valAcc: .8838, trainLoss: .3893, valLoss: .3083 },
    { ep: 2, phase: 'head', trainAcc: .8880, valAcc: .8237, trainLoss: .2847, valLoss: .3904 },
    { ep: 3, phase: 'ft',   trainAcc: .8905, valAcc: .8925, trainLoss: .2724, valLoss: .2565 },
  ],
  resnet: [
    { ep: 1, phase: 'head', trainAcc: .8205, valAcc: .8500, trainLoss: .3970, valLoss: .3506 },
    { ep: 2, phase: 'head', trainAcc: .8650, valAcc: .8512, trainLoss: .3159, valLoss: .3311 },
    { ep: 3, phase: 'ft',   trainAcc: .8715, valAcc: .8788, trainLoss: .3219, valLoss: .3113 },
  ],
};

function drawCurves() {
  const host = $('#curvePlot');
  if (!host) return;

  const W = 720, H = 268;
  const m = { t: 18, r: 54, b: 40, l: 46 };          // room for the outermost labels
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const yMin = .80, yMax = .90;                       // the band the data actually occupies
  const X = ep => m.l + ((ep - 1) / 2) * iw;
  const Y = v  => m.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

  const path = rows => rows.map((d, i) => `${i ? 'L' : 'M'}${X(d.ep).toFixed(1)},${Y(d.valAcc).toFixed(1)}`).join(' ');

  const ticks = [.80, .82, .84, .86, .88, .90];

  const svg = `
<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation accuracy per epoch for MobileNetV2 and ResNet18">
  <title>Validation accuracy per epoch</title>

  ${ticks.map(t => `
  <line x1="${m.l}" y1="${Y(t).toFixed(1)}" x2="${m.l + iw}" y2="${Y(t).toFixed(1)}" stroke="${C.line}" stroke-width="1"/>
  <text x="${m.l - 10}" y="${(Y(t) + 4).toFixed(1)}" fill="${C.muted}" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">${(t * 100).toFixed(0)}%</text>`).join('')}

  <!-- phase change marker: head-only → fine-tune -->
  <line x1="${X(2.5).toFixed(1)}" y1="${m.t}" x2="${X(2.5).toFixed(1)}" y2="${m.t + ih}"
        stroke="${C.signal}" stroke-width="1" stroke-dasharray="3 5" opacity=".55"/>
  <text x="${(X(2.5) + 7).toFixed(1)}" y="${m.t + 13}" fill="${C.signal}"
        font-family="JetBrains Mono, monospace" font-size="10">unfreeze</text>

  ${[1, 2, 3].map(ep => `
  <text x="${X(ep).toFixed(1)}" y="${H - 14}" fill="${C.muted}" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="middle">epoch ${ep}</text>`).join('')}

  <path d="${path(RUN.resnet)}"    fill="none" stroke="${C.recycle}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  <path d="${path(RUN.mobilenet)}" fill="none" stroke="${C.organic}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>

  ${RUN.resnet.map(d => `<circle cx="${X(d.ep).toFixed(1)}" cy="${Y(d.valAcc).toFixed(1)}" r="3.4" fill="${C.recycle}"/>`).join('')}
  ${RUN.mobilenet.map(d => `<circle cx="${X(d.ep).toFixed(1)}" cy="${Y(d.valAcc).toFixed(1)}" r="3.8" fill="${C.organic}"/>`).join('')}

  <!-- endpoint callouts -->
  <text x="${(X(3) + 10).toFixed(1)}" y="${(Y(RUN.mobilenet[2].valAcc) + 4).toFixed(1)}" fill="${C.organic}"
        font-family="JetBrains Mono, monospace" font-size="11" font-weight="500">89.3%</text>
  <text x="${(X(3) + 10).toFixed(1)}" y="${(Y(RUN.resnet[2].valAcc) + 4).toFixed(1)}" fill="${C.recycle}"
        font-family="JetBrains Mono, monospace" font-size="11" font-weight="500">87.9%</text>
</svg>`;

  host.innerHTML = svg;
}

/* ─────────────────────────────────────────────────────────────
   4 · Network probe
   ───────────────────────────────────────────────────────────── */
function initNet() {
  const blocks = $$('.blk');
  const shape = $('#probeShape'), note = $('#probeNote');
  if (!blocks.length) return;

  const show = b => {
    blocks.forEach(x => x.classList.toggle('is-on', x === b));
    shape.textContent = b.dataset.shape;
    note.textContent  = b.dataset.note;
  };
  blocks.forEach(b => {
    b.addEventListener('mouseenter', () => show(b));
    b.addEventListener('focus', () => show(b));
    b.addEventListener('click', () => show(b));
  });
}

/* ─────────────────────────────────────────────────────────────
   5 · Research gaps
   ───────────────────────────────────────────────────────────── */
function initGaps() {
  $$('.gap__bar').forEach(bar => {
    bar.addEventListener('click', () => {
      const li = bar.closest('.gap');
      const open = li.classList.toggle('is-open');
      bar.setAttribute('aria-expanded', String(open));
    });
  });
}

/* ─────────────────────────────────────────────────────────────
   6 · The inspection bay
   ───────────────────────────────────────────────────────────── */
const bay = {
  file: null,
  dataUrl: null,
  stream: null,
};

function setPaneState(state) {
  const pane = $('#verdictPane');
  pane.dataset.state = state;
  $('#verdictIdle').hidden = state !== 'idle';
  $('#verdictWork').hidden = state !== 'work';
  $('#verdictOut').hidden  = state !== 'out';
  $('#scanBeam').hidden    = state !== 'work';
}

function showPreview(url) {
  const img = $('#previewImg');
  img.src = url;
  img.hidden = false;
  $('#dropEmpty').hidden = true;
  $('#clearBtn').hidden = false;
}

function clearBay() {
  stopCam();
  bay.file = null; bay.dataUrl = null;
  const img = $('#previewImg');
  img.hidden = true; img.removeAttribute('src');
  $('#dropEmpty').hidden = false;
  $('#clearBtn').hidden = true;
  setPaneState('idle');
}

async function readFile(file) {
  if (!file.type.startsWith('image/')) { toast('That file is not an image.', 'warn'); return; }
  if (file.size > 12 * 1024 * 1024)   { toast('Image is over 12 MB — use a smaller one.', 'warn'); return; }
  bay.file = file;
  bay.dataUrl = await new Promise(res => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.readAsDataURL(file);
  });
  showPreview(bay.dataUrl);
  classify();
}

const WORK_STEPS = [
  'Decoding image',
  'Resize → 128 × 128',
  'Normalising · ImageNet stats',
  'Forward pass · 5 conv blocks',
  'Softmax over 2 classes',
  'Grad-CAM · block 5 activations',
];

async function classify() {
  if (!bay.file && !bay.dataUrl) return;
  setPaneState('work');

  // step ticker while the request is in flight
  let si = 0;
  const stepEl = $('#workStep');
  stepEl.textContent = WORK_STEPS[0];
  const ticker = setInterval(() => {
    si = (si + 1) % WORK_STEPS.length;
    stepEl.textContent = WORK_STEPS[si];
  }, 420);

  const t0 = performance.now();
  let result = null;

  try {
    const fd = new FormData();
    if (bay.file) fd.append('image', bay.file, bay.file.name || 'upload.png');
    else          fd.append('image_b64', bay.dataUrl);

    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    result = normalise(await res.json());
  } catch (err) {
    clearInterval(ticker);
    setPaneState('idle');
    toast(`Prediction failed — ${err.message}. Is /api/predict wired up?`, 'warn');
    return;
  }

  clearInterval(ticker);
  if (result.inferMs == null) result.inferMs = Math.round(performance.now() - t0);
  render(result);
}

/* Accept a few reasonable response shapes so this survives backend tweaks. */
function normalise(j) {
  const probs = j.probabilities || j.probs || {};
  let pO = probs.Organic ?? probs.organic ?? probs.O;
  let pR = probs.Recyclable ?? probs.recyclable ?? probs.R;

  if (pO == null || pR == null) {
    const conf = j.confidence ?? j.score ?? 0;
    const isO = (j.label ?? j.class ?? j.prediction ?? '').toString().toLowerCase().startsWith('o')
             || j.class_index === 0;
    pO = isO ? conf : 1 - conf;
    pR = 1 - pO;
  }
  if (pO > 1 || pR > 1) { pO /= 100; pR /= 100; }

  return {
    pO, pR,
    label:   pO >= pR ? 'Organic' : 'Recyclable',
    inferMs: j.inference_ms ?? j.inferMs ?? j.latency_ms ?? null,
    gradcam: j.gradcam ?? j.grad_cam ?? j.heatmap ?? null,
    image:   j.image ?? bay.dataUrl,
  };
}

function render(r) {
  setPaneState('out');

  const isO = r.label === 'Organic';
  const conf = Math.max(r.pO, r.pR);
  const accent = isO ? C.organic : C.recycle;

  $('#outClass').textContent = r.label;
  $('#outClass').style.color = accent;
  $('#outBin').innerHTML = isO
    ? 'Route to the <strong style="color:' + C.organic + '">green bin</strong> — wet, biodegradable.'
    : 'Route to the <strong style="color:' + C.recycle + '">blue bin</strong> — dry, recyclable.';

  // ring
  const ring = $('#ringFill');
  ring.style.stroke = accent;
  ring.style.strokeDashoffset = String(327 - 327 * conf);

  // ring number counts up
  const valEl = $('#ringVal');
  if (REDUCED) { valEl.textContent = (conf * 100).toFixed(1); }
  else {
    const t0 = performance.now(), dur = 850;
    const tick = now => {
      const k = Math.min(1, (now - t0) / dur);
      valEl.textContent = (conf * 100 * (1 - Math.pow(1 - k, 3))).toFixed(1);
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // probability bars
  $('#pOrganic').textContent = (r.pO * 100).toFixed(1) + '%';
  $('#pRecycle').textContent = (r.pR * 100).toFixed(1) + '%';
  requestAnimationFrame(() => {
    $('#barOrganic').style.width = (r.pO * 100) + '%';
    $('#barRecycle').style.width = (r.pR * 100) + '%';
  });

  $('#inferMs').textContent = r.inferMs != null ? `${r.inferMs} ms` : '— ms';

  // Grad-CAM
  const base = $('#camBase'), heat = $('#camHeat'), none = $('#camNone');
  base.src = r.image || '';
  if (r.gradcam) {
    heat.src = r.gradcam.startsWith('data:') ? r.gradcam : `data:image/png;base64,${r.gradcam}`;
    heat.hidden = false; none.hidden = true;
  } else {
    heat.hidden = true; none.hidden = false;
  }
}

function initBay() {
  const zone = $('#dropZone');
  if (!zone) return;

  $('#browseBtn').addEventListener('click', e => { e.stopPropagation(); $('#fileInput').click(); });
  zone.addEventListener('click', () => { if (!bay.stream) $('#fileInput').click(); });
  zone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#fileInput').click(); }
  });

  $('#fileInput').addEventListener('change', e => {
    if (e.target.files[0]) readFile(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('is-drag'); }));
  ['dragleave', 'drop'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('is-drag'); }));
  zone.addEventListener('drop', e => {
    const f = e.dataTransfer?.files?.[0];
    if (f) readFile(f);
  });

  addEventListener('paste', e => {
    const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
    if (item) { readFile(item.getAsFile()); toast('Image pasted from clipboard.'); }
  });

  $('#clearBtn').addEventListener('click', e => { e.stopPropagation(); clearBay(); });
  $('#camBtn').addEventListener('click', e => { e.stopPropagation(); bay.stream ? stopCam() : startCam(); });
  $('#shotBtn').addEventListener('click', e => { e.stopPropagation(); capture(); });

  // Grad-CAM overlay slider
  const mix = $('#camMix');
  mix.addEventListener('input', () => {
    $('#camHeat').style.opacity = mix.value / 100;
    $('#camMixOut').textContent = mix.value + '%';
  });

  // sample chips — no bundled images, so these explain rather than fake a result
  $$('.sample').forEach(btn => {
    btn.addEventListener('click', () => {
      toast(`Drop a photo of a ${btn.dataset.name.toLowerCase()} to test the ${btn.dataset.kind === 'organic' ? 'green' : 'blue'} stream.`);
      $('#fileInput').click();
    });
  });
}

async function startCam() {
  try {
    bay.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    const v = $('#camVideo');
    v.srcObject = bay.stream;
    v.hidden = false;
    await v.play();
    $('#dropEmpty').hidden = true;
    $('#previewImg').hidden = true;
    $('#dropZone').classList.add('is-live');
    $('#shotBtn').hidden = false;
    $('#camBtn').textContent = 'Stop camera';
  } catch {
    toast('Camera unavailable — check browser permissions.', 'warn');
  }
}

function stopCam() {
  if (!bay.stream) return;
  bay.stream.getTracks().forEach(t => t.stop());
  bay.stream = null;
  const v = $('#camVideo');
  v.hidden = true; v.srcObject = null;
  $('#dropZone').classList.remove('is-live');
  $('#shotBtn').hidden = true;
  $('#camBtn').innerHTML = '<svg viewBox="0 0 16 16"><circle cx="8" cy="9" r="3.2" stroke="currentColor" stroke-width="1.4" fill="none"/><path d="M2 5.5h2.2L5.4 3.6h5.2l1.2 1.9H14v7.5H2z" stroke="currentColor" stroke-width="1.4" fill="none" stroke-linejoin="round"/></svg> Camera';
  if (!bay.dataUrl) $('#dropEmpty').hidden = false;
}

function capture() {
  const v = $('#camVideo'), cv = $('#camCanvas');
  cv.width = v.videoWidth; cv.height = v.videoHeight;
  cv.getContext('2d').drawImage(v, 0, 0);
  cv.toBlob(blob => {
    stopCam();
    readFile(new File([blob], 'capture.png', { type: 'image/png' }));
  }, 'image/png');
}

/* ─────────────────────────────────────────────────────────────
   7 · API console
   ───────────────────────────────────────────────────────────── */
function paint(json) {
  const s = JSON.stringify(json, null, 2);
  return s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/"([^"]+)":/g, '<span class="k">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="s">"$1"</span>')
    .replace(/: (-?\d+\.?\d*)/g, ': <span class="n">$1</span>');
}

function initApi() {
  const eps = $$('.ep');
  if (!eps.length) return;
  let cur = eps[0];

  const select = ep => {
    cur = ep;
    eps.forEach(e => {
      const on = e === ep;
      e.classList.toggle('is-on', on);
      e.setAttribute('aria-selected', String(on));
    });
    $('#conPath').textContent = `${ep.dataset.method} ${ep.dataset.path}`;
    $('#conNote').textContent = ep.dataset.method === 'POST'
      ? 'POST /api/predict needs an image — load one in the inspection bay first.'
      : 'Sends a live request to your Flask server.';
  };

  eps.forEach(ep => ep.addEventListener('click', () => select(ep)));
  select(eps[0]);

  $('#sendBtn').addEventListener('click', async () => {
    const out = $('#conOut');
    const btn = $('#sendBtn');
    btn.disabled = true;
    out.innerHTML = '<code>…requesting</code>';

    try {
      let res;
      if (cur.dataset.method === 'POST') {
        if (!bay.file && !bay.dataUrl) {
          out.innerHTML = '<code class="err">No image loaded. Drop one in the inspection bay above, then send.</code>';
          btn.disabled = false; return;
        }
        const fd = new FormData();
        if (bay.file) fd.append('image', bay.file, bay.file.name || 'upload.png');
        else          fd.append('image_b64', bay.dataUrl);
        res = await fetch(cur.dataset.path, { method: 'POST', body: fd });
      } else {
        res = await fetch(cur.dataset.path);
      }

      const body = await res.json();
      // a base64 heat map would flood the console — show its size instead
      if (body && typeof body === 'object') {
        for (const k of ['gradcam', 'grad_cam', 'heatmap', 'image']) {
          if (typeof body[k] === 'string' && body[k].length > 120) {
            body[k] = `‹base64 · ${(body[k].length / 1024).toFixed(1)} KB›`;
          }
        }
      }
      out.innerHTML = `<code>HTTP ${res.status}\n\n${paint(body)}</code>`;
    } catch (err) {
      out.innerHTML = `<code class="err">Request failed — ${err.message}</code>`;
    }
    btn.disabled = false;
  });
}

/* ─────────────────────────────────────────────────────────────
   8 · Model health badge
   ───────────────────────────────────────────────────────────── */
async function initHealth() {
  const el = $('#navStatus');
  const txt = $('.nav__statusText', el);
  try {
    const r = await fetch('/api/health');
    if (!r.ok) throw new Error();
    const j = await r.json();
    const loaded = j.model_loaded ?? j.loaded ?? j.ok ?? true;
    const dev = (j.device || 'cpu').toString().toUpperCase();
    el.dataset.state = loaded ? 'live' : 'down';
    txt.textContent = loaded ? `model live · ${dev}` : 'model not loaded';
  } catch {
    el.dataset.state = 'down';
    txt.textContent = 'server unreachable';
  }
}

/* ─────────────────────────────────────────────────────────────
   boot
   ───────────────────────────────────────────────────────────── */
function boot() {
  initSorter();
  initCounters();
  initSpy();
  initReveal();
  drawCurves();
  initNet();
  initGaps();
  initBay();
  initApi();
  initHealth();
}

if (document.readyState === 'loading') addEventListener('DOMContentLoaded', boot);
else boot();

})();
