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
  hazard:  '#E8734D',
  nonrec:  '#9B8CA3',
  signal:  '#E8A33D',
  line:    '#1E2B28',
  muted:   '#7C918B',
  faint:   '#4E605B',
};

const CLASS_META = {
  'Organic':          { color: C.organic, key: 'o', bin: 'Route to the <strong style="color:' + C.organic + '">green bin</strong> — wet, biodegradable.' },
  'Recyclable':       { color: C.recycle, key: 'r', bin: 'Route to the <strong style="color:' + C.recycle + '">blue bin</strong> — dry, recyclable.' },
  'Hazardous':        { color: C.hazard,  key: 'h', bin: 'Separate collection required — <strong style="color:' + C.hazard + '">batteries, electronics, chemicals</strong>.' },
  'Non-Recyclable':   { color: C.nonrec,  key: 'n', bin: 'Route to the <strong style="color:' + C.nonrec + '">black bin</strong> — non-recyclable, non-biodegradable.' },
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
    gateX = W * 0.54;              // gate sits in the open right half
  }

  function spawn() {
    const organic = Math.random() < 0.56;   // dataset is 56% organic
    return {
      x: -20,
      y: H * (0.34 + Math.random() * 0.32), // wider intake band
      vx: 1.7 + Math.random() * 1.6,
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
        p.vy = (p.organic ? -1 : 1) * (0.34 + Math.random() * 0.6);
      }
      p.x += p.vx;
      p.y += p.vy;
      if (p.sorted) p.life -= 0.0032;

      if (p.x > W + 30 || p.life <= 0) { items.splice(i, 1); continue; }

      const col = p.sorted ? (p.organic ? C.organic : C.recycle) : C.muted;
      const a = p.sorted ? Math.max(0, p.life) * 0.95 : 0.55;

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

  function seed(n) {
    for (let i = 0; i < n; i++) {
      const p = spawn();
      p.x = Math.random() * W;
      if (p.x > gateX) {
        p.sorted = true;
        p.vy = (p.organic ? -1 : 1) * (0.34 + Math.random() * 0.6);
        p.y += p.vy * ((p.x - gateX) / p.vx);
        p.life = Math.max(0.15, 1 - ((p.x - gateX) / p.vx) * 0.0032);
      }
      items.push(p);
    }
  }

  size();
  addEventListener('resize', size, { passive: true });

  if (REDUCED) {
    // one still frame, no motion
    seed(220);
    ctx.clearRect(0, 0, W, H);
    items.forEach(p => {
      ctx.fillStyle = p.sorted ? (p.organic ? C.organic : C.recycle) : C.muted;
      ctx.globalAlpha = 0.62;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
    return;
  }

  // pause when off-screen — no wasted frames
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) { if (!raf) { if (!items.length) seed(130); raf = requestAnimationFrame(step); } }
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
    if (REDUCED) { el.textContent = target.toLocaleString('en-GB', { minimumFractionDigits: dp, maximumFractionDigits: dp }); return; }
    const t0 = performance.now(), dur = 1150;
    const tick = now => {
      const k = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * eased).toLocaleString('en-GB', { minimumFractionDigits: dp, maximumFractionDigits: dp });
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
   3 · Measurements
   Every figure on this page is read from /api/metrics, which serves
   outputs/eval_report.json. Nothing here is hardcoded — if a number
   has not been measured, the page says so instead of inventing one.
   ───────────────────────────────────────────────────────────── */
const M = { report: null, history: null };

const pct  = v => (v * 100).toFixed(2);
// thousands grouping, not lakh/crore — parameter counts are read against papers
const num  = v => v.toLocaleString('en-US');
const DASH = '—';

async function loadMetrics() {
  try {
    const r = await fetch('/api/metrics');
    const j = await r.json();
    M.report = j && j.available ? j : null;
  } catch { M.report = null; }

  try {
    const r = await fetch('/api/training-history');
    if (r.ok) {
      const j = await r.json();
      M.history = j && j.available !== false ? j : null;
    }
  } catch { M.history = null; }

  if (!M.report) { showNoData(); return; }
  paintReadout();
  paintProvenance();
  paintComparison();
  paintConfusion();
  paintPerClass();
  paintSpecs();
  paintGap3();
  drawCurves();
}

function showNoData() {
  const msg = 'No evaluation report found. Run: python -m src.evaluate';
  $('#readoutSrc').textContent = msg;
  $('#readout').dataset.state = 'empty';
  $('#cmpBody').innerHTML = `<tr class="empty"><td colspan="7">${msg}</td></tr>`;
  $('#curvePlot').innerHTML = `<p class="plot__empty">${msg}</p>`;
  $('#cmFoot').textContent = msg;
  $('#pcm').innerHTML = `<p class="plot__empty">${msg}</p>`;
  const prov = $('#prov');
  prov.hidden = false;
  prov.dataset.kind = 'none';
  $('#provFlag').textContent = 'no data';
  $('#provText').textContent = 'This page shows measurements only. None have been recorded yet — run the evaluation to populate it.';
}

function paintReadout() {
  const rep = M.report;
  const best = rep.models.find(m => m.name === rep.best) || rep.models[0];
  const total = rep.dataset ? (rep.dataset.train.total + rep.dataset.test.total) : null;

  $('#mAcc').textContent  = pct(best.accuracy);
  $('#mAccBar').style.setProperty('--w', pct(best.accuracy) + '%');

  $('#mImgs').textContent = total != null ? num(total) : DASH;
  $('#heroCount').textContent = total != null ? num(total) : 'countless';
  $('#refCount').textContent  = total != null ? num(total) : DASH;

  $('#mLat').textContent  = best.latency.median_ms.toFixed(1);
  $('#mLatBar').style.setProperty('--w', Math.min(100, best.latency.median_ms / 2) + '%');

  $('#mSize').textContent = best.size_mb.toFixed(2);
  $('#mSizeBar').style.setProperty('--w', Math.min(100, best.size_mb) + '%');

  $('#scopeN').textContent = rep.num_classes;

  $('#readout').dataset.state = 'ok';
  $('#readoutSrc').textContent =
    `${best.name} · median of ${best.latency.runs} timed forward passes on ${best.latency.device} · ` +
    `${rep.evaluation.test_images_used} test images`;
}

function paintProvenance() {
  const rep = M.report, run = rep.run, prov = $('#prov');
  prov.hidden = false;
  prov.dataset.kind = run.quick_mode ? 'quick' : 'full';
  $('#provFlag').textContent = run.quick_mode ? 'QUICK mode' : 'full run';
  $('#provText').textContent = run.quick_mode
    ? `Subset run — ${num(run.train_subset)} training and ${num(run.test_subset)} test images at `
      + `${run.transfer_input_size}px, ${run.epochs_head}+${run.epochs_finetune} epochs on ${run.device}. `
      + `These are not full-dataset results and should be labelled as such in any report.`
    : `Full training set, full test set, ${run.transfer_input_size}px, `
      + `${run.epochs_head}+${run.epochs_finetune} epochs on ${run.device}.`;
  $('#provWhen').textContent = new Date(rep.generated_at).toLocaleString('en-GB', {
    dateStyle: 'medium', timeStyle: 'short',
  });
}

function paintComparison() {
  const rep = M.report;
  const top = Math.max(...rep.models.map(m => m.accuracy));
  const swatch = { baseline: 'n', mobilenet: 'o', resnet: 'r' };

  $('#cmpBody').innerHTML = rep.models.map(m => `
    <tr class="${m.name === rep.best ? 'is-pick' : ''}">
      <th scope="row">${m.name}<span class="tnote">${m.input_size}px input</span></th>
      <td class="mono">${num(m.params)}</td>
      <td class="mono">${m.size_mb.toFixed(2)} MB</td>
      <td class="mono">${pct(m.accuracy)}%</td>
      <td class="mono">${m.f1.toFixed(4)}</td>
      <td class="mono">${m.latency.median_ms.toFixed(1)} ms</td>
      <td><div class="tbar"><i style="--w:${(m.accuracy / top * 100).toFixed(1)}%" data-s="${swatch[m.key] || 'n'}"></i></div></td>
    </tr>`).join('');
}

function paintConfusion() {
  const rep = M.report;
  const best = rep.models.find(m => m.name === rep.best) || rep.models[0];
  const cm = best.confusion;                       // [[TL, TR], [BL, BR]]
  const rowO = cm[0][0] + cm[0][1], rowR = cm[1][0] + cm[1][1];

  const put = (id, v, denom) => {
    const el = $('#' + id);
    const frac = denom ? v / denom : 0;
    el.style.setProperty('--i', frac.toFixed(3));
    el.querySelector('b').textContent = v;
    el.querySelector('small').textContent = denom ? (frac * 100).toFixed(0) + '%' : '';
  };
  put('cmTL', cm[0][0], rowO); put('cmTR', cm[0][1], rowO);
  put('cmBL', cm[1][0], rowR); put('cmBR', cm[1][1], rowR);

  $('#cmSub').textContent = `${best.name} · n=${rowO + rowR}`;

  const worst = cm[1][0] >= cm[0][1]
    ? `${cm[1][0]} recyclable items were sent to the wet stream`
    : `${cm[0][1]} organic items were sent to the dry stream`;
  $('#cmFoot').textContent = `${worst} — the model's largest error mode, and where the next round of training should go.`;
}

function paintPerClass() {
  const rep = M.report;
  const best = rep.models.find(m => m.name === rep.best) || rep.models[0];
  const keyMap = { Organic: 'o', Recyclable: 'r', Hazardous: 'h', 'Non-Recyclable': 'n' };

  $('#pcm').innerHTML = Object.entries(best.per_class).map(([name, v]) => `
    <div class="pcm__group" data-s="${keyMap[name] || 'n'}">
      <p class="pcm__name">${name} <span class="mono">n=${v.support}</span></p>
      ${[['P', v.precision], ['R', v.recall], ['F1', v.f1]].map(([lbl, val]) => `
      <div class="pcm__row">
        <span>${lbl}</span>
        <div class="pcm__track"><i style="--w:${(val * 100).toFixed(2)}%"></i></div>
        <b class="mono">${val.toFixed(4)}</b>
      </div>`).join('')}
    </div>`).join('');

  const entries = Object.entries(best.per_class);
  if (entries.length >= 2) {
    const sorted = entries.sort((a, b) => b[1].recall - a[1].recall);
    $('#pcmFoot').textContent =
      `Highest recall: ${sorted[0][0]} (${sorted[0][1].recall.toFixed(2)}), `
      + `lowest: ${sorted[sorted.length - 1][0]} (${sorted[sorted.length - 1][1].recall.toFixed(2)}).`;
  }
}

function paintSpecs() {
  const rep = M.report;
  const base = rep.models.find(m => m.key === 'baseline');
  $('#specBatch').textContent = rep.run.batch_size ?? DASH;
  $('#specParams').textContent = base ? (base.params / 1e6).toFixed(2) + ' M' : DASH;
}

function paintGap3() {
  const rep = M.report;
  const mob = rep.models.find(m => m.key === 'mobilenet');
  const base = rep.models.find(m => m.key === 'baseline');
  if (!mob) return;

  const smaller = base ? Math.round((1 - mob.params / base.params) * 100) : null;
  $('#g3Text').innerHTML =
    `fine-tuned MobileNetV2 to <span class="mono">${pct(mob.accuracy)}%</span> at `
    + `<span class="mono">${(mob.params / 1e6).toFixed(2)} M</span> parameters and `
    + `<span class="mono">${mob.size_mb.toFixed(2)} MB</span>`
    + (smaller != null ? ` — ${smaller}% fewer parameters than the baseline CNN` : '')
    + `, at a measured <span class="mono">${mob.latency.median_ms.toFixed(1)} ms</span> per image on `
    + `${mob.latency.device}.`;

  $('#g3Caveat').textContent = rep.run.quick_mode
    ? `these figures come from a ${num(rep.run.train_subset)}-image subset run, not full-dataset training.`
    : `measured on this machine only; other hardware will differ.`;
}

/* ── validation-accuracy chart, drawn from the recorded history ── */
function drawCurves() {
  const host = $('#curvePlot');
  if (!host) return;

  const hist = M.history;
  if (!hist) {
    host.innerHTML = '<p class="plot__empty">No training history found. Run: python -m src.train</p>';
    return;
  }

  const series = Object.entries(hist)
    .filter(([, v]) => v && Array.isArray(v.val_acc) && v.val_acc.length)
    .map(([k, v]) => ({
      key: k,
      label: k === 'mobilenet' ? 'MobileNetV2' : k === 'resnet' ? 'ResNet18' : k,
      colour: k === 'mobilenet' ? C.organic : k === 'resnet' ? C.recycle : C.muted,
      vals: v.val_acc,
    }));

  if (!series.length) {
    host.innerHTML = '<p class="plot__empty">Training history is empty.</p>';
    return;
  }

  $('#curveLegend').innerHTML = series
    .map(s => `<span class="legend__k" style="--lc:${s.colour}">${s.label}</span>`).join('');

  const nEp = Math.max(...series.map(s => s.vals.length));
  const all = series.flatMap(s => s.vals);
  // pad the band so the extremes are not glued to the frame
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.02) * 50) / 50);
  const hi = Math.min(1, Math.ceil((Math.max(...all) + 0.02) * 50) / 50);

  const W = 720, H = 268;
  const m = { t: 18, r: 62, b: 40, l: 50 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const X = i => m.l + (nEp === 1 ? iw / 2 : (i / (nEp - 1)) * iw);
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * ih;

  const ticks = [];
  const stepCount = 5;
  for (let i = 0; i <= stepCount; i++) ticks.push(lo + (hi - lo) * (i / stepCount));

  // where the fine-tune phase begins, if the run recorded one
  const headEp = M.report?.run?.epochs_head ?? null;
  const marker = headEp && headEp < nEp
    ? `<line x1="${X(headEp - 0.5).toFixed(1)}" y1="${m.t}" x2="${X(headEp - 0.5).toFixed(1)}" y2="${m.t + ih}"
             stroke="${C.signal}" stroke-width="1" stroke-dasharray="3 5" opacity=".55"/>
       <text x="${(X(headEp - 0.5) + 7).toFixed(1)}" y="${m.t + 13}" fill="${C.signal}"
             font-family="JetBrains Mono, monospace" font-size="10">unfreeze</text>`
    : '';

  host.innerHTML = `
<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation accuracy per epoch">
  <title>Validation accuracy per epoch</title>
  ${ticks.map(t => `
  <line x1="${m.l}" y1="${Y(t).toFixed(1)}" x2="${m.l + iw}" y2="${Y(t).toFixed(1)}" stroke="${C.line}" stroke-width="1"/>
  <text x="${m.l - 10}" y="${(Y(t) + 4).toFixed(1)}" fill="${C.muted}" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">${(t * 100).toFixed(0)}%</text>`).join('')}
  ${marker}
  ${Array.from({ length: nEp }, (_, i) => `
  <text x="${X(i).toFixed(1)}" y="${H - 14}" fill="${C.muted}" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="middle">epoch ${i + 1}</text>`).join('')}
  ${series.map(s => `
  <path d="${s.vals.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}"
        fill="none" stroke="${s.colour}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
  ${s.vals.map((v, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="3.6" fill="${s.colour}"/>`).join('')}
  <text x="${(X(s.vals.length - 1) + 10).toFixed(1)}" y="${(Y(s.vals[s.vals.length - 1]) + 4).toFixed(1)}"
        fill="${s.colour}" font-family="JetBrains Mono, monospace" font-size="11" font-weight="500">${(s.vals[s.vals.length - 1] * 100).toFixed(1)}%</text>`).join('')}
</svg>`;

  $('#curveFoot').textContent = headEp && headEp < nEp
    ? `Epochs 1–${headEp} train the classifier head only; the rest unfreeze the last blocks at a lower learning rate.`
    : '';
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

/* Accept a few reasonable response shapes so this survives backend tweaks.
   Works for 2-class (Organic/Recyclable) or N-class responses. */
function normalise(j) {
  const raw = j.probabilities || j.probs || {};
  const aliases = {
    'Organic':        ['Organic', 'organic', 'O'],
    'Recyclable':     ['Recyclable', 'recyclable', 'R'],
    'Hazardous':      ['Hazardous', 'hazardous', 'H'],
    'Non-Recyclable': ['Non-Recyclable', 'non_recyclable', 'N'],
  };

  const probs = {};
  for (const [name, keys] of Object.entries(aliases)) {
    for (const k of keys) {
      if (raw[k] != null) { probs[name] = raw[k]; break; }
    }
  }

  if (Object.keys(probs).length < 2) {
    const conf = j.confidence ?? j.score ?? 0;
    const lbl = (j.label ?? j.class ?? j.prediction ?? '').toString();
    const matched = Object.keys(CLASS_META).find(n => lbl.toLowerCase().startsWith(n.toLowerCase().slice(0, 3)));
    if (matched) {
      probs[matched] = conf;
      const rest = Object.keys(CLASS_META).filter(n => n !== matched);
      const share = (1 - conf) / rest.length;
      rest.forEach(n => { probs[n] = share; });
    } else {
      probs['Organic'] = conf;
      probs['Recyclable'] = 1 - conf;
    }
  }

  const maxV = Math.max(...Object.values(probs));
  if (maxV > 1) for (const k of Object.keys(probs)) probs[k] /= 100;

  const label = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];

  return {
    probs,
    pO: probs['Organic'] ?? 0,
    pR: probs['Recyclable'] ?? 0,
    label,
    confidence: Math.max(...Object.values(probs)),
    inferMs: j.inference_ms ?? j.inferMs ?? j.latency_ms ?? null,
    gradcam: j.gradcam ?? j.grad_cam ?? j.heatmap ?? null,
    image:   j.image ?? bay.dataUrl,
  };
}

function render(r) {
  setPaneState('out');

  const meta = CLASS_META[r.label] || { color: C.muted, key: 'n', bin: '' };
  const conf = r.confidence;
  const accent = meta.color;

  $('#outClass').textContent = r.label;
  $('#outClass').style.color = accent;
  $('#outBin').innerHTML = meta.bin;

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

  // dynamic probability bars for N classes
  const probsEl = $('#probs');
  probsEl.innerHTML = Object.entries(r.probs)
    .sort((a, b) => b[1] - a[1])
    .map(([name, prob]) => {
      const m = CLASS_META[name] || { color: C.muted, key: 'n' };
      return `<div class="prob" data-s="${m.key}">
        <div class="prob__head"><span>${name}</span><b style="color:${m.color}">${(prob * 100).toFixed(1)}%</b></div>
        <div class="prob__track"><i style="width:${(prob * 100)}%;background:${m.color}"></i></div>
      </div>`;
    }).join('');

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

  addToHistory(r);
}

/* ─────────────────────────────────────────────────────────────
   Prediction History
   ───────────────────────────────────────────────────────────── */
const historyItems = [];

function addToHistory(r) {
  historyItems.unshift({
    label: r.label,
    confidence: r.confidence,
    image: r.image,
    time: new Date(),
  });
  if (historyItems.length > 10) historyItems.pop();
  renderHistory();
}

function renderHistory() {
  const list = $('#historyList');
  const clearBtn = $('#historyClear');
  if (!list) return;

  if (historyItems.length === 0) {
    list.innerHTML = '<p class="history__empty">No items classified yet.</p>';
    if (clearBtn) clearBtn.hidden = true;
    return;
  }

  if (clearBtn) clearBtn.hidden = false;
  list.innerHTML = historyItems.map(item => {
    const meta = CLASS_META[item.label] || { color: C.muted };
    const ago = timeAgo(item.time);
    return `<div class="history__item">
      <img class="history__thumb" src="${item.image || ''}" alt="${item.label}">
      <div class="history__info">
        <span class="history__class" style="color:${meta.color}">${item.label}</span>
        <span class="history__time">${ago}</span>
      </div>
      <span class="history__conf" style="color:${meta.color}">${(item.confidence * 100).toFixed(1)}%</span>
    </div>`;
  }).join('');
}

function timeAgo(date) {
  const s = Math.floor((Date.now() - date.getTime()) / 1000);
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
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
  $('#liveBtn').addEventListener('click', e => { e.stopPropagation(); liveActive ? stopLiveScan() : startLiveScan(); });

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
   6b · Live Scan — continuous real-time classification
   ───────────────────────────────────────────────────────────── */
let liveInterval = null;
let liveActive = false;
let liveBusy = false;

async function startLiveScan() {
  if (!bay.stream) await startCam();
  if (!bay.stream) return;

  liveActive = true;
  $('#liveOverlay').hidden = false;
  $('#liveBtn').innerHTML = '<svg viewBox="0 0 16 16"><rect x="4" y="4" width="8" height="8" rx="1" fill="currentColor"/></svg> Stop Live';
  $('#liveBtn').classList.add('is-live');
  $('#shotBtn').hidden = true;
  toast('Live scan active — point camera at waste items.', 'info');

  liveClassify();
  liveInterval = setInterval(liveClassify, 1500);
}

function stopLiveScan() {
  liveActive = false;
  clearInterval(liveInterval);
  liveInterval = null;
  $('#liveOverlay').hidden = true;
  $('#liveBtn').innerHTML = '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="4" fill="#dc2626"/></svg> Live Scan';
  $('#liveBtn').classList.remove('is-live');
  stopCam();
}

async function liveClassify() {
  if (!bay.stream || !liveActive || liveBusy) return;
  liveBusy = true;

  const v = $('#camVideo'), cv = $('#camCanvas');
  cv.width = v.videoWidth; cv.height = v.videoHeight;
  cv.getContext('2d').drawImage(v, 0, 0);

  try {
    const blob = await new Promise(r => cv.toBlob(r, 'image/jpeg', 0.8));
    const fd = new FormData();
    fd.append('image', new File([blob], 'live.jpg', { type: 'image/jpeg' }));

    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error();
    const result = normalise(await res.json());

    if (!liveActive) { liveBusy = false; return; }

    const meta = CLASS_META[result.label] || { color: C.muted };
    $('#liveClass').textContent = result.label;
    $('#liveClass').style.color = meta.color;
    $('#liveConf').textContent = (result.confidence * 100).toFixed(1) + '%';
    $('#liveConf').style.color = meta.color;
  } catch { }

  liveBusy = false;
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
  initSpy();
  initReveal();
  initNet();
  initGaps();
  initBay();
  initApi();
  initHealth();

  const hc = $('#historyClear');
  if (hc) hc.addEventListener('click', () => { historyItems.length = 0; renderHistory(); });
  loadMetrics();     // fills every figure on the page
}

if (document.readyState === 'loading') addEventListener('DOMContentLoaded', boot);
else boot();

})();
