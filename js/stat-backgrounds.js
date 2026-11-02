/* ============================================================
   STAT BACKGROUNDS — full-page animated code-character backgrounds
   Psychology and statistics scenes drawn as 0s, 1s and math symbols
   coloured by the scene underneath. Same renderer as the Moonlit
   Social Labs code backgrounds (jellyfish, mycelium, constellation).

   USAGE in HTML (anywhere in <body>):
     <canvas data-stat-bg="galton"></canvas>
     <script src="/js/stat-backgrounds.js"></script>

   The canvas is pinned to the viewport behind everything
   (position:fixed, z-index:-1, no pointer events).

   SCENES (data-stat-bg). Each one runs the real thing:
     "galton"     — beads through a Galton board; binomial bins converge on the normal curve
     "neurons"    — leaky integrate-and-fire network; spikes excite or inhibit their targets
     "network"    — psychometric network (qgraph style); activation spreads over +/− edges
     "regression" — data arrives one case at a time; OLS line and its 95% band tighten
     "ridgeline"  — posterior densities narrowing as evidence accumulates

   OPTIONS:
     data-palette="indigo"  — indigo | synapse | aurora | moonlit
     data-intensity="0.75"  — scene brightness, 0..1
     data-dim="0.3"         — veil of page colour over the scene, 0..1 (for text-heavy pages)
     data-cell="7"          — character cell size in px

   Page-level preview overrides: ?bg=network&palette=synapse

   Scenes paint light straight into a per-cell colour grid instead of
   reading pixels back from a canvas (getImageData), so browsers that
   randomise read-back (Firefox resistFingerprinting, Brave) render the
   same frames as Chrome. The character layer is drawn once; each frame
   only uploads a small colour+alpha image and stretches it over it.
   Pauses in background tabs. Holds a still frame under
   prefers-reduced-motion.
   ============================================================ */
(function () {
  'use strict';

  const PALETTES = {
    indigo:  { void: '#0B0F1A', c1: [6, 182, 212],  c2: [99, 102, 241], c3: [245, 158, 11],  c4: [244, 63, 94] },
    synapse: { void: '#0D0A17', c1: [217, 70, 239], c2: [124, 58, 237], c3: [251, 191, 36],  c4: [34, 211, 238] },
    aurora:  { void: '#08120F', c1: [16, 185, 129], c2: [13, 148, 136], c3: [163, 230, 53],  c4: [251, 113, 133] },
    moonlit: { void: '#07070D', c1: [0, 212, 170],  c2: [139, 92, 246], c3: [212, 160, 32],  c4: [196, 74, 138] },
  };

  const BINARY = '01', SYMBOLS = '=+-*/|<>≤≥≠≈±∑∫√∞∆∂μσβχρ';
  const THEME = { galton: 'μσ²∑·∘x̄', neurons: 'ψΨ~^∿·', network: 'ρ+−○●·', regression: 'βεr²ŷ·', ridgeline: 'θπ∝∫∼·' };
  const GLYPHS = [...new Set([...BINARY, ...SYMBOLS, ...[...Object.values(THEME).join('')].filter(ch => ch.codePointAt(0) < 0x300 || ch.codePointAt(0) > 0x36F)])];
  const GLYPH_INDEX = new Map(GLYPHS.map((g, i) => [g, i]));
  const rnd = (a, b) => a + Math.random() * (b - a);
  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  // Smooth, saturating brightness curve (a stepped boost makes dim gradients blotchy).
  const tone = br => 255 * (1 - Math.exp(-br / 95));
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const gauss = () => { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * Math.random()); };

  // ---------- light field ----------
  class Field {
    constructor(cols, rows, cell) { this.cols = cols; this.rows = rows; this.cell = cell; this.d = new Float32Array(cols * rows * 3); }
    clear() { this.d.fill(0); }
    add(o, a) { const d = this.d, e = o.d; for (let k = 0; k < d.length; k++) d[k] += e[k] * a; }
    // soft radial light, (1-t²)² falloff; additive like canvas 'lighter'
    glow(x, y, r, c, a) {
      const s = this.cell, cx = x / s - .5, cy = y / s - .5, R = Math.max(1.5, r / s), R2 = R * R, { cols, rows, d } = this;
      const i0 = Math.max(0, Math.ceil(cx - R)), i1 = Math.min(cols - 1, Math.floor(cx + R));
      const j0 = Math.max(0, Math.ceil(cy - R)), j1 = Math.min(rows - 1, Math.floor(cy + R));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const q = ((i - cx) ** 2 + (j - cy) ** 2) / R2; if (q >= 1) continue;
        const w = a * (1 - q) * (1 - q), k = (j * cols + i) * 3; d[k] += c[0] * w; d[k + 1] += c[1] * w; d[k + 2] += c[2] * w;
      }
    }
    // disc with an antialiased rim so moving strokes fade cells in/out instead of popping
    disc(cx, cy, R, c, a) {
      const { cols, rows, d } = this, E = R + .5;
      const i0 = Math.max(0, Math.ceil(cx - E)), i1 = Math.min(cols - 1, Math.floor(cx + E));
      const j0 = Math.max(0, Math.ceil(cy - E)), j1 = Math.min(rows - 1, Math.floor(cy + E));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const cov = Math.min(1, E - Math.hypot(i - cx, j - cy)); if (cov <= 0) continue;
        const w = a * cov, k = (j * cols + i) * 3; d[k] += c[0] * w; d[k + 1] += c[1] * w; d[k + 2] += c[2] * w;
      }
    }
    // stroke of width w: a row of discs whose overlap sums to `a` on the centre line
    line(x1, y1, x2, y2, w, c, a) {
      const s = this.cell, r = Math.max(.5, w / 2 / s), len = Math.hypot(x2 - x1, y2 - y1) / s;
      const n = Math.max(1, Math.ceil(len / .5)), aa = a * Math.min(1, (len / n) / (2 * r + 1));
      for (let k = 0; k < n; k++) { const t = (k + .5) / n; this.disc((x1 + (x2 - x1) * t) / s - .5, (y1 + (y2 - y1) * t) / s - .5, r, c, aa); }
    }
    // axis-aligned block of light (histogram bars, confidence bands)
    rect(x0, y0, x1, y1, c, a) {
      const s = this.cell, { cols, rows, d } = this;
      const i0 = Math.max(0, Math.round(x0 / s)), i1 = Math.min(cols, Math.round(x1 / s));
      const j0 = Math.max(0, Math.round(y0 / s)), j1 = Math.min(rows, Math.round(y1 / s));
      for (let j = j0; j < j1; j++) for (let i = i0; i < i1; i++) { const k = (j * cols + i) * 3; d[k] += c[0] * a; d[k + 1] += c[1] * a; d[k + 2] += c[2] * a; }
    }
  }

  // ---------- scenes ----------
  // Binomial bins fill bead by bead; the normal curve scales with the running total.
  class Galton {
    init(W, H, cell, cols, rows) {
      this.n = W < 640 ? 14 : 20;
      this.layer = new Field(cols, rows, cell);
      const n = this.n; this.dx = W * .94 / (n + 1); this.dy = H * .5 / n; this.x0 = W / 2; this.y0 = H * .05;
      this.yb = H * .99; this.target = 48;
      const pmax = this.pmf(Math.floor(n / 2)); this.N = Math.round(this.target / pmax); this.unit = H * .4 / this.target;
      this.pegs = []; for (let r = 0; r < n; r++) for (let k = 0; k <= r; k++) this.pegs.push(this.at(r, k));
      this.lit = false; this.reset(.45);
    }
    pmf(k) { const n = this.n; let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); return c / 2 ** n; }
    at(r, k) { return [this.x0 + (k - r / 2) * this.dx, this.y0 + r * this.dy]; }
    reset(frac) {
      this.count = new Array(this.n + 1).fill(0); this.beads = []; this.spawn = 0; this.phase = 'fill'; this.timer = 0; this.alpha = 1; this.total = 0;
      for (let b = 0, m = Math.round(this.N * frac); b < m; b++) { let k = 0; for (let r = 0; r < this.n; r++) k += Math.random() < .5 ? 1 : 0; this.count[k]++; this.total++; }
    }
    prefill() { this.reset(.95); }
    step(dt) {
      if (this.phase === 'fill') {
        this.spawn -= dt;
        while (this.spawn <= 0 && this.beads.length < 90) { this.beads.push({ r: 0, k: 0, u: 0, dir: Math.random() < .5 ? 0 : 1, v: rnd(5, 7), fall: 0 }); this.spawn += .085; }
        if (Math.max(...this.count) >= this.target) { this.phase = 'hold'; this.timer = 6; }
      } else if (this.phase === 'hold') { this.timer -= dt; if (this.timer <= 0) this.phase = 'fade'; }
      else { this.alpha -= dt / 2.2; if (this.alpha <= 0) { this.reset(0); return; } }
      for (const b of this.beads) {
        if (b.r >= this.n) { b.fall += dt * 2.2; if (b.fall >= 1) { b.dead = true; this.count[b.k]++; this.total++; } continue; }
        b.u += dt * b.v;
        if (b.u >= 1) { b.u -= 1; b.k += b.dir; b.r++; b.dir = Math.random() < .5 ? 0 : 1; if (b.r >= this.n) b.u = 0; }
      }
      this.beads = this.beads.filter(b => !b.dead);
    }
    draw(F, W, H, p, t, cell) {
      const { n, dx, dy, x0, y0, yb, unit, alpha } = this;
      if (!this.lit) { this.lit = true; for (const [x, y] of this.pegs) this.layer.glow(x, y, cell * 1.7, p.c2, .75); }
      F.add(this.layer, alpha);
      for (let k = 0; k <= n; k++) {
        const x = x0 + (k - n / 2) * dx, h = this.count[k] * unit, w = dx * .72;
        if (h <= 0) continue;
        F.rect(x - w / 2, yb - h, x + w / 2, yb, p.c1, .3 * alpha);
        F.rect(x - w / 2, yb - h, x + w / 2, yb - h + cell * 2, p.c1, .5 * alpha);
      }
      // normal approximation: N(n/2, n/4), scaled to the beads landed so far
      const sd = Math.sqrt(n) / 2; let px = null;
      for (let i = 0; i <= 90; i++) {
        const z = (i / 90 - .5) * (n + 1), x = x0 + z * dx, y = yb - this.total * unit * Math.exp(-z * z / (2 * sd * sd)) / (sd * 2.5066);
        if (px) F.line(px[0], px[1], x, y, cell * 1.2, p.c3, .85 * alpha);
        px = [x, y];
      }
      for (const b of this.beads) {
        let x, y;
        if (b.r >= n) { [x, y] = this.at(n, b.k); y += (yb - this.count[b.k] * unit - y) * b.fall * b.fall; }
        else { [x, y] = [x0 + (b.k + b.dir * b.u - (b.r + b.u) / 2) * dx, y0 + (b.r + b.u) * dy - Math.sin(Math.PI * b.u) * dy * .4]; }
        F.glow(x, y, cell * 2.4, p.c3, .9 * alpha);
      }
    }
  }

  // Leaky integrate-and-fire: inputs charge the membrane, it fires at threshold, then rests.
  class Neurons {
    init(W, H, cell, cols, rows) {
      const n = Math.round(Math.min(38, Math.max(12, W * H / 52000))), D = Math.sqrt(W * H / n) * .72;
      this.pts = [];
      for (let tries = 0; this.pts.length < n && tries < 4000; tries++) {
        const x = rnd(W * .02, W * .98), y = rnd(H * .02, H * .98);
        if (this.pts.every(q => Math.hypot(q[0] - x, q[1] - y) > D)) this.pts.push([x, y]);
      }
      this.layer = new Field(cols, rows, cell); this.lit = false;
      this.v = this.pts.map(() => rnd(0, .9)); this.drive = this.pts.map(() => rnd(.09, .15)); this.rest = this.pts.map(() => 0); this.flash = this.pts.map(() => 0);
      this.syn = this.pts.map((a, i) => this.pts.map((b, j) => [Math.hypot(a[0] - b[0], a[1] - b[1]), j]).filter(e => e[1] !== i).sort((x, y) => x[0] - y[0]).slice(0, 3).map(([d, j]) => {
        const b = this.pts[j], bend = rnd(-.28, .28), mx = (a[0] + b[0]) / 2 - (b[1] - a[1]) * bend, my = (a[1] + b[1]) / 2 + (b[0] - a[0]) * bend, path = [];
        for (let u = 0; u <= 1.0001; u += 1 / 14) path.push([(1 - u) ** 2 * a[0] + 2 * u * (1 - u) * mx + u * u * b[0], (1 - u) ** 2 * a[1] + 2 * u * (1 - u) * my + u * u * b[1]]);
        return { to: j, path, len: d, w: Math.random() < .2 ? -.3 : .42 };
      }));
      this.spikes = [];
    }
    branch(x, y, ang, len, depth, p, cell) {
      const nx = x + Math.cos(ang) * len, ny = y + Math.sin(ang) * len;
      this.layer.line(x, y, nx, ny, cell * .9, p.c2, .42 + .16 * depth);
      if (depth > 0) for (let k = 0; k < 2; k++) this.branch(nx, ny, ang + rnd(-.8, .8), len * rnd(.55, .8), depth - 1, p, cell);
    }
    step(dt) {
      const { v, rest, flash } = this;
      for (let i = 0; i < v.length; i++) {
        flash[i] = Math.max(0, flash[i] - dt * .55);
        if (rest[i] > 0) { rest[i] -= dt; continue; }
        v[i] += dt * (this.drive[i] + .1 * gauss() - .08 * v[i]);
        if (v[i] < -.3) v[i] = -.3;
        if (v[i] >= 1) { v[i] = 0; rest[i] = 3; flash[i] = 1; if (this.spikes.length < 110) for (const s of this.syn[i]) this.spikes.push({ s, u: 0 }); }
      }
      for (const q of this.spikes) {
        q.u += dt * 52 / Math.max(60, q.s.len);
        if (q.u >= 1) { q.dead = true; if (rest[q.s.to] <= 0) v[q.s.to] += q.s.w; }
      }
      this.spikes = this.spikes.filter(q => !q.dead);
    }
    draw(F, W, H, p, t, cell) {
      if (!this.lit) {
        this.lit = true;
        for (const row of this.syn) for (const s of row) for (let k = 1; k < s.path.length; k++) this.layer.line(s.path[k - 1][0], s.path[k - 1][1], s.path[k][0], s.path[k][1], cell * .9, p.c2, .55);
        for (const [x, y] of this.pts) { const m = Math.floor(rnd(4, 7)), a0 = rnd(0, 6.28); for (let k = 0; k < m; k++) this.branch(x, y, a0 + k * 6.28 / m + rnd(-.3, .3), rnd(16, 34), 2, p, cell); }
      }
      F.add(this.layer, .8 + .2 * Math.sin(t * .12));
      const at = (s, u) => { const f = Math.min(s.path.length - 1.001, u * (s.path.length - 1)), i = Math.floor(f), e = f - i, a = s.path[i], b = s.path[i + 1]; return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e]; };
      for (const q of this.spikes) {
        const c = q.s.w < 0 ? p.c4 : p.c3, [x, y] = at(q.s, q.u), [bx, by] = at(q.s, Math.max(0, q.u - .14));
        F.line(bx, by, x, y, cell, c, .5); F.glow(x, y, cell * 3, c, .95);
      }
      this.pts.forEach(([x, y], i) => {
        const v = Math.max(0, this.v[i]), f = this.flash[i];
        F.glow(x, y, cell * (3.2 + 2.4 * f), p.c1, .8 + .4 * v + .4 * f);
        if (f > 0) F.glow(x, y, cell * (5 + 9 * (1 - f)), p.c3, .28 * f);
      });
    }
  }

  // Communities of symptoms joined by partial correlations: width = |weight|, colour = sign.
  class Network {
    init(W, H) {
      const spots = W > 900 ? [[.17, .24], [.8, .2], [.86, .76], [.24, .78], [.5, .5]] : [[.3, .17], [.72, .5], [.28, .84]];
      const per = W > 900 ? 7 : 6, Rc = Math.min(W, H) * (W > 900 ? .2 : .15);
      this.nodes = []; this.edges = [];
      spots.forEach(([cx, cy], c) => {
        for (let k = 0; k < per; k++) {
          const a = k / per * 6.283 + rnd(-.25, .25), r = Rc * rnd(.55, 1);
          this.nodes.push({ c, x0: cx * W + Math.cos(a) * r, y0: cy * H + Math.sin(a) * r, ax: rnd(5, 16), ay: rnd(5, 16), f: rnd(.05, .14), ph: rnd(0, 6.28), m: rnd(.9, 1.4) });
        }
      });
      for (let i = 0; i < this.nodes.length; i++) for (let j = i + 1; j < this.nodes.length; j++) {
        const same = this.nodes[i].c === this.nodes[j].c;
        if (Math.random() < (same ? .72 : .045)) {
          const neg = Math.random() < (same ? .1 : .42);
          this.edges.push({ i, j, w: (neg ? -1 : 1) * (same ? rnd(.25, .75) : rnd(.15, .4)), ph: rnd(0, 6.28), f: rnd(.03, .09) });
        }
      }
      this.a = this.nodes.map(() => 0); this.seed = 0;
    }
    step(dt, t) {
      this.P = this.nodes.map(n => [n.x0 + Math.sin(t * n.f + n.ph) * n.ax, n.y0 + Math.cos(t * n.f * .8 + n.ph) * n.ay]);
      this.seed -= dt;
      if (this.seed <= 0) { this.seed = rnd(1.4, 2.8); this.a[Math.floor(rnd(0, this.a.length))] = 1; }
      const inp = this.a.map(() => 0);
      for (const e of this.edges) { inp[e.i] += e.w * this.a[e.j]; inp[e.j] += e.w * this.a[e.i]; }
      for (let i = 0; i < this.a.length; i++) this.a[i] = Math.min(1, Math.max(0, this.a[i] + dt * (1.5 * inp[i] - 1.55 * this.a[i])));
    }
    draw(F, W, H, p, t, cell) {
      const P = this.P, A = this.a;
      for (const e of this.edges) {
        const w = Math.abs(e.w) * (1 + .15 * Math.sin(t * e.f * 6.28 + e.ph)), act = Math.max(A[e.i], A[e.j]);
        F.line(P[e.i][0], P[e.i][1], P[e.j][0], P[e.j][1], cell * (.7 + 2.2 * w), e.w < 0 ? p.c4 : p.c1, .16 + .42 * w + .35 * act);
      }
      this.nodes.forEach((n, i) => {
        const [x, y] = P[i], r = cell * 2.6 * n.m, a = A[i];
        F.disc(x / cell - .5, y / cell - .5, r / cell, mix(p.c2, p.c3, a), .75 + .25 * a);
        F.glow(x, y, r * 2.2, p.c2, .3); if (a > .05) F.glow(x, y, r * 4, p.c3, .5 * a);
      });
    }
  }

  // Cases arrive one at a time; running OLS and its 95% confidence band for the mean.
  class Regression {
    init(W, H) { this.W = W; this.H = H; this.maxN = Math.round(Math.min(150, Math.max(55, W / 9))); this.reset(.4); }
    reset(frac) {
      const { W, H } = this;
      this.slope = -rnd(.28, .55) * H / W; this.icpt = H * rnd(.72, .8); this.sd = H * .13;
      this.pts = []; this.sx = this.sy = this.sxx = this.sxy = this.syy = 0; this.spawn = 0; this.phase = 'fill'; this.timer = 0; this.alpha = 1;
      for (let i = 0, m = Math.round(this.maxN * frac); i < m; i++) this.addPoint(-9);
    }
    prefill() { this.reset(.97); }
    addPoint(age) {
      const x = rnd(this.W * .02, this.W * .98), y = this.icpt + this.slope * x + gauss() * this.sd;
      this.pts.push({ x, y, born: age }); this.sx += x; this.sy += y; this.sxx += x * x; this.sxy += x * y; this.syy += y * y;
    }
    fit() {
      const n = this.pts.length; if (n < 3) return null;
      const Sxx = this.sxx - this.sx ** 2 / n, Sxy = this.sxy - this.sx * this.sy / n, Syy = this.syy - this.sy ** 2 / n;
      const b = Sxy / Sxx, a = (this.sy - b * this.sx) / n, s = Math.sqrt(Math.max(0, (Syy - b * Sxy) / (n - 2)));
      return { n, a, b, s, Sxx, xbar: this.sx / n };
    }
    step(dt, t) {
      this.now = t;
      if (this.phase === 'fill') { this.spawn -= dt; if (this.spawn <= 0) { this.spawn = .16; this.addPoint(t); } if (this.pts.length >= this.maxN) { this.phase = 'hold'; this.timer = 6; } }
      else if (this.phase === 'hold') { this.timer -= dt; if (this.timer <= 0) this.phase = 'fade'; }
      else { this.alpha -= dt / 2; if (this.alpha <= 0) this.reset(0); }
    }
    draw(F, W, H, p, t, cell) {
      const f = this.fit(), al = this.alpha, now = this.now || 0;
      if (f) {
        const q = 1.96 + 2.4 / f.n, step = cell * 2;
        for (let x = 0; x < W; x += step) {
          const y = f.a + f.b * (x + step / 2), h = q * f.s * Math.sqrt(1 / f.n + (x + step / 2 - f.xbar) ** 2 / f.Sxx);
          F.rect(x, y - h, x + step, y + h, p.c2, .34 * al);
        }
        F.line(0, f.a, W, f.a + f.b * W, cell * 2, p.c3, 1.1 * al);
      }
      for (const q of this.pts) {
        const age = now - q.born, fresh = Math.max(0, 1 - age / 1.4);
        F.glow(q.x, q.y, cell * (3 + 1.8 * fresh), fresh > .02 ? mix(p.c1, p.c3, fresh) : p.c1, (1.1 + .3 * fresh) * al);
        if (f && fresh > 0) F.line(q.x, q.y, q.x, f.a + f.b * q.x, cell * .8, p.c4, .55 * fresh * al);
      }
    }
  }

  // Normal–normal updating: each row is one effect; evidence accumulates, its posterior
  // tightens around the true value, then a new effect starts from the prior.
  class Ridgeline {
    init(W, H) {
      this.K = Math.round(Math.min(13, Math.max(7, H / 78)));
      this.rows = Array.from({ length: this.K }, (_, r) => ({ theta: rnd(-2.7, 2.7), T: rnd(18, 30), ph: (r * .618) % 1, cycle: 0, wob: rnd(0, 6.28) }));
    }
    step(dt, t) {
      for (const q of this.rows) {
        const c = t / q.T + q.ph, u = c % 1, n = 26 * u * u, cycle = Math.floor(c);
        if (cycle !== q.cycle) { q.cycle = cycle; q.theta = rnd(-2.7, 2.7); }
        q.env = smooth(0, .08, u) * (1 - smooth(.93, 1, u)); q.n = n;
        q.obs = q.theta + Math.sin(t * .5 + q.wob) * .7 / Math.sqrt(Math.max(n, 1));
      }
    }
    draw(F, W, H, p, t, cell) {
      const K = this.K, gap = H * .92 / K, span = W / 8.4, cols = 110, prior = 1.5;
      this.rows.forEach((q, r) => {
        const base = H * .05 + (r + 1) * gap, post = 1 / (1 / (prior * prior) + q.n), sd = Math.sqrt(post), mean = post * q.n * q.obs;
        const tight = 1 - sd / prior, c = mix(p.c2, p.c1, tight), amp = gap * (.8 + 1.5 * tight), al = q.env;
        const hOf = x => { const z = ((x - W / 2) / span - mean) / sd; return amp * Math.exp(-z * z / 2); };
        let px = 0, ph = hOf(0);
        for (let i = 1; i <= cols; i++) {
          const x = W * i / cols, h = hOf(x);
          if (Math.max(h, ph) > cell * .9) {            // draw the density, not its empty tails
            F.line(px, base - ph, x, base - h, cell * 1.2, c, .95 * al);
            F.rect(px, base - Math.max(h, ph), x, base, c, .14 * al);
          }
          px = x; ph = h;
        }
        F.glow(W / 2 + mean * span, base - amp, cell * 3.4, p.c3, al);
      });
    }
  }

  const SCENES = { galton: Galton, neurons: Neurons, network: Network, regression: Regression, ridgeline: Ridgeline };

  // ---------- renderer ----------
  class StatBackground {
    constructor(canvas, opts = {}) {
      this.canvas = canvas; this.ctx = canvas.getContext('2d');
      this.name = SCENES[opts.scene] ? opts.scene : 'network';
      this.cell = opts.cell || 7;
      this.intensity = opts.intensity != null ? opts.intensity : .75;
      this.dim = opts.dim || 0;
      const P = PALETTES[opts.palette] || PALETTES.indigo;
      this.void = P.void;
      this.pal = { c1: P.c1, c2: P.c2, c3: P.c3, c4: P.c4, faint: mix(hex(P.void), P.c2, .2) };
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.t = 0; this.running = false;
      this.atlas = document.createElement('canvas');
      this.mask = document.createElement('canvas'); this.mctx = this.mask.getContext('2d');
      this.glyphs = document.createElement('canvas'); this.gctx = this.glyphs.getContext('2d');
      this.tint = document.createElement('canvas'); this.tctx = this.tint.getContext('2d');
      Object.assign(canvas.style, { position: 'fixed', inset: '0', width: '100vw', height: '100vh', zIndex: '-1', pointerEvents: 'none', display: 'block' });
      canvas.setAttribute('aria-hidden', 'true');
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      this.resize();
      document.addEventListener('visibilitychange', () => document.hidden ? this.stop() : this.start());
      reduceMotion.addEventListener?.('change', () => reduceMotion.matches ? this.stop() : this.start());
      document.fonts?.ready.then(() => { if (!this.W) return; this.buildAtlas(); this.drawMask(); this.still(); });
      this.start();
    }
    buildAtlas() {
      const g = Math.round(this.cell * this.dpr), a = this.atlas, c = a.getContext('2d');
      a.width = g * GLYPHS.length; a.height = g; this.g = g;
      c.clearRect(0, 0, a.width, a.height);
      c.font = `${(this.cell - 1) * this.dpr}px monospace`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#fff';
      GLYPHS.forEach((ch, i) => c.fillText(ch, i * g + g / 2, g / 2));
    }
    resize() {
      const r = this.canvas.getBoundingClientRect(); if (!r.width || !r.height) return;
      // ignore the small height jitter from mobile toolbars showing/hiding
      if (this.W === r.width && Math.abs(this.H - r.height) < 120 && this.F) return;
      this.W = r.width; this.H = r.height; this.cols = Math.ceil(this.W / this.cell); this.rows = Math.ceil(this.H / this.cell);
      this.canvas.width = Math.floor(r.width * this.dpr); this.canvas.height = Math.floor(r.height * this.dpr);
      this.buildAtlas();
      this.mask.width = this.glyphs.width = this.cols * this.g; this.mask.height = this.glyphs.height = this.rows * this.g;
      this.tint.width = this.cols; this.tint.height = this.rows; this.img = this.tctx.createImageData(this.cols, this.rows);
      this.F = new Field(this.cols, this.rows, this.cell);
      this.scene = new SCENES[this.name]();
      this.scene.init(this.W, this.H, this.cell, this.cols, this.rows);
      if (reduceMotion.matches) this.scene.prefill?.();
      for (let i = 0; i < 90; i++) { this.t += 1 / 60; this.scene.step(1 / 60, this.t, this.W, this.H, this.pal); }
      const n = this.cols * this.rows; this.grid = new Uint8Array(n); this.faint = new Uint8Array(n);
      for (let i = 0; i < n; i++) { this.grid[i] = this.pick(); this.faint[i] = Math.random() < .12 ? 1 : 0; }
      this.drawMask(); this.still();
    }
    pick() {
      const r = Math.random();
      if (r < .5) return GLYPH_INDEX.get(r < .25 ? '0' : '1');
      if (r < .75) return GLYPH_INDEX.get(SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)]);
      const th = this.theme || (this.theme = [...THEME[this.name]].filter(ch => GLYPH_INDEX.has(ch)));
      return GLYPH_INDEX.get(th[Math.floor(Math.random() * th.length)]);
    }
    drawMask() { this.mctx.clearRect(0, 0, this.mask.width, this.mask.height); for (let i = 0; i < this.grid.length; i++) this.drawGlyph(i); }
    drawGlyph(i) {
      const g = this.g, c = i % this.cols, r = (i / this.cols) | 0;
      this.mctx.clearRect(c * g, r * g, g, g);
      this.mctx.drawImage(this.atlas, this.grid[i] * g, 0, g, g, c * g, r * g, g, g);
    }
    start() {
      if (this.running || reduceMotion.matches || document.hidden) return;
      this.running = true; this.last = performance.now();
      const loop = now => { if (!this.running) return; this.frame(Math.min(.05, (now - this.last) / 1000)); this.last = now; this.raf = requestAnimationFrame(loop); };
      this.raf = requestAnimationFrame(loop);
    }
    stop() { this.running = false; cancelAnimationFrame(this.raf); }
    still() { if (!this.running && this.W) this.frame(0); }
    // dim ambience: a vertical wash plus a few huge, slow light pools
    atmosphere(F, p, t) {
      const { cols, rows, d } = F;
      for (let j = 0; j < rows; j++) {
        const v = j / rows, c = mix(p.c2, p.c1, v), a = .06 - .035 * v;
        for (let i = 0; i < cols; i++) { const k = (j * cols + i) * 3; d[k] += c[0] * a; d[k + 1] += c[1] * a; d[k + 2] += c[2] * a; }
      }
      const { W, H } = this, R = Math.max(W, H) * .5;
      F.glow(W * (.3 + .2 * Math.sin(t * .031)), H * (.1 + .1 * Math.sin(t * .047)), R, p.c1, .07);
      F.glow(W * (.75 + .15 * Math.sin(t * .023 + 2)), H * (.35 + .15 * Math.sin(t * .037 + 1)), R * .8, p.c2, .06);
      F.glow(W * (.5 + .3 * Math.sin(t * .017 + 4)), H * (.95 + .05 * Math.sin(t * .029)), R * .9, p.c2, .05);
    }
    frame(dt) {
      this.t += dt;
      const { F, W, H, pal } = this;
      this.scene.step(dt, this.t, W, H, pal);
      F.clear();
      this.scene.draw(F, W, H, pal, this.t, this.cell);
      if (this.intensity !== 1) for (let k = 0; k < F.d.length; k++) F.d[k] *= this.intensity;
      this.atmosphere(F, pal, this.t);
      // swap characters a trickle at a time (~10%/s, no bursts)
      const swaps = Math.round(this.grid.length * .1 * dt);
      for (let s = 0; s < swaps; s++) { const i = Math.floor(Math.random() * this.grid.length); this.grid[i] = this.pick(); this.drawGlyph(i); }
      this.render();
    }
    render() {
      const { cols, rows, pal, g, gctx, img } = this, d = this.F.d, px = img.data, fr = pal.faint;
      for (let i = 0, n = cols * rows; i < n; i++) {
        const k = i * 3, o = i * 4, R = d[k], G = d[k + 1], B = d[k + 2], br = R * .299 + G * .587 + B * .114;
        if (br < 1) { px[o] = fr[0]; px[o + 1] = fr[1]; px[o + 2] = fr[2]; px[o + 3] = this.faint[i] ? 200 * (1 - br) : 0; continue; }
        const s = tone(br) / br;
        px[o] = R * s; px[o + 1] = G * s; px[o + 2] = B * s;
        px[o + 3] = 255 * Math.min(.95, .2 + br / 50) * smooth(0, 3, br);
      }
      this.tctx.putImageData(img, 0, 0);
      gctx.globalCompositeOperation = 'copy'; gctx.drawImage(this.mask, 0, 0);
      gctx.globalCompositeOperation = 'source-in'; gctx.imageSmoothingEnabled = true; gctx.imageSmoothingQuality = 'high';
      gctx.drawImage(this.tint, 0, 0, cols * g, rows * g);
      const ctx = this.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1; ctx.fillStyle = this.void; ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.drawImage(this.glyphs, 0, 0, cols * g, rows * g, 0, 0, cols * this.cell * this.dpr, rows * this.cell * this.dpr);
      if (this.dim) { ctx.globalAlpha = this.dim; ctx.fillRect(0, 0, this.canvas.width, this.canvas.height); ctx.globalAlpha = 1; }
    }
  }

  function hex(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }

  // ---------- auto-init ----------
  function init() {
    const q = new URLSearchParams(location.search);
    document.querySelectorAll('canvas[data-stat-bg]').forEach(c => {
      if (c._statBg) return;
      c._statBg = new StatBackground(c, {
        scene: q.get('bg') || c.dataset.statBg,
        palette: q.get('palette') || c.dataset.palette,
        cell: c.dataset.cell ? +c.dataset.cell : undefined,
        intensity: c.dataset.intensity ? +c.dataset.intensity : undefined,
        dim: c.dataset.dim ? +c.dataset.dim : undefined,
      });
    });
  }

  window.StatBackgrounds = { init, StatBackground, scenes: SCENES, palettes: PALETTES };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
