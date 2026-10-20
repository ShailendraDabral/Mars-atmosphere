/* sim-core.js — numerical engine of the Mars PINN gap-filling simulator.
 *
 * Pure JavaScript, no dependencies. Runs in the browser (window.SimCore) and in Node
 * (module.exports) so it can be unit-tested. It mirrors the Python package `marspinn`:
 * synthetic data, quality control, physics audit, splits, gaps, baselines (linear, PCHIP,
 * hydrostatic integration, Gaussian process, EOF optimal interpolation), MLP / PINN with
 * hand-written forward-mode tangents and reverse-mode gradients, deep ensembles, L2-SP
 * fine-tuning, metrics, bootstrap CIs, Wilcoxon + Holm tests and split-conformal calibration.
 */
(function (root, factory) {
  const lib = factory();
  if (typeof module === "object" && module.exports) module.exports = lib;
  else root.SimCore = lib;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------ constants
  const K_B = 1.380649e-23, N_A = 6.02214076e23, G0 = 3.71, T_REF = 200.0;
  const phys = { m: 43.49e-3 / N_A, g0: G0, Tref: T_REF, floor: NaN };
  const Href = () => K_B * phys.Tref / (phys.m * phys.g0) / 1e3;
  const mGmol = () => phys.m * N_A * 1e3;

  // ------------------------------------------------------------------ random numbers
  function RNG(seed) {
    let s = (seed >>> 0) || 1;
    const next = () => {             // mulberry32
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    return {
      u: next,
      uniform: (a, b) => a + (b - a) * next(),
      int: (n) => Math.floor(next() * n),
      normal(mu = 0, sd = 1) {
        if (spare !== null) { const v = spare; spare = null; return mu + sd * v; }
        let u, v, r;
        do { u = 2 * next() - 1; v = 2 * next() - 1; r = u * u + v * v; } while (r >= 1 || r === 0);
        const f = Math.sqrt(-2 * Math.log(r) / r);
        spare = v * f; return mu + sd * u * f;
      },
      permutation(arr) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
        return a;
      },
    };
  }
  function hashSeed(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  // ------------------------------------------------------------------ small array helpers
  const mean = (a) => { let s = 0, n = 0; for (const v of a) if (Number.isFinite(v)) { s += v; n++; } return n ? s / n : NaN; };
  const median = (a) => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); const n = b.length; if (!n) return NaN; return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]); };
  const quantile = (a, q) => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); if (!b.length) return NaN; const p = (b.length - 1) * q, i = Math.floor(p), f = p - i; return i + 1 < b.length ? b[i] * (1 - f) + b[i + 1] * f : b[i]; };
  const std = (a) => { const m = mean(a); let s = 0, n = 0; for (const v of a) if (Number.isFinite(v)) { s += (v - m) ** 2; n++; } return n > 1 ? Math.sqrt(s / (n - 1)) : 0; };
  function interp(x, xp, fp) {            // numpy.interp for sorted xp
    const n = xp.length;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
    const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] + t * (fp[hi] - fp[lo]);
  }
  function polyfit1(x, y) {               // least-squares line, returns [slope, intercept]
    const n = x.length, mx = mean(x), my = mean(y);
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
    const b = sxx > 0 ? sxy / sxx : 0;
    return [b, my - b * mx];
  }
  function gradient(f, x) {               // numpy.gradient, second-order interior, non-uniform x
    const n = f.length, g = new Array(n);
    if (n < 2) return g.fill(NaN);
    g[0] = (f[1] - f[0]) / (x[1] - x[0]);
    g[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
    for (let i = 1; i < n - 1; i++) {
      const hl = x[i] - x[i - 1], hr = x[i + 1] - x[i];
      g[i] = (hl * hl * f[i + 1] - hr * hr * f[i - 1] + (hr * hr - hl * hl) * f[i]) / (hl * hr * (hl + hr));
    }
    return g;
  }
  function cholesky(A, n) {               // A: Float64Array n*n (SPD). Returns lower L or null
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) { if (s <= 0) return null; L[i * n + i] = Math.sqrt(s); }
        else L[i * n + j] = s / L[j * n + j];
      }
    }
    return L;
  }
  function cholSolve(L, n, b) {          // solve (L L^T) x = b
    const y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
    return x;
  }
  function forwardSub(L, n, b) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    return y;
  }
  function invSym(A, n) {                 // inverse of SPD matrix via Cholesky
    const L = cholesky(A, n); if (!L) return null;
    const inv = new Float64Array(n * n);
    for (let j = 0; j < n; j++) { const e = new Float64Array(n); e[j] = 1; const c = cholSolve(L, n, e); for (let i = 0; i < n; i++) inv[i * n + j] = c[i]; }
    return inv;
  }
  function jacobiEigen(A, n) {            // symmetric eigen-decomposition, eigenvalues descending
    const a = Float64Array.from(A), V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
      if (off < 1e-20) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
        const app = a[p * n + p], aqq = a[q * n + q];
        const th = (aqq - app) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
    const idx = [...Array(n).keys()].sort((i, j) => a[j * n + j] - a[i * n + i]);
    return { values: idx.map((i) => a[i * n + i]), vectors: idx.map((i) => Array.from({ length: n }, (_, k) => V[k * n + i])) };
  }

  // ------------------------------------------------------------------ physics
  function integrateLnP(z, T, lnP0) {
    const H = Href(), out = new Array(z.length); out[0] = lnP0;
    for (let i = 1; i < z.length; i++) out[i] = out[i - 1] - 0.5 * (phys.Tref / (H * T[i]) + phys.Tref / (H * T[i - 1])) * (z[i] - z[i - 1]);
    return out;
  }
  function hydroResidual(z, T, lnP) {
    const o = [...z.keys()].sort((a, b) => z[a] - z[b]);
    const zs = o.map((i) => z[i]), Ts = o.map((i) => T[i]), Ps = o.map((i) => lnP[i]);
    const g = gradient(Ps, zs), H = Href();
    return g.map((d, i) => H * d + phys.Tref / Ts[i]);
  }
  const rms = (a) => Math.sqrt(mean(a.map((v) => v * v)));

  // ------------------------------------------------------------------ synthetic data
  const YEAR_TRACKS = { 29: { ls: [106.8, 142.5], lat: [-12.5, 14.5] }, 30: { ls: [21.0, 45.5], lat: [69.3, 48.8] }, 31: { ls: [37.4, 75.1], lat: [65.8, 47.5] } };

  function correlatedNoise(rng, z, corr) {
    const n = z.length, C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) C[i * n + j] = Math.exp(-0.5 * ((z[i] - z[j]) / corr) ** 2) + (i === j ? 1e-6 : 0);
    const L = cholesky(C, n), e = Array.from({ length: n }, () => rng.normal());
    return Array.from({ length: n }, (_, i) => { let s = 0; for (let k = 0; k <= i; k++) s += L[i * n + k] * e[k]; return s; });
  }
  function trueTemperature(z, lat, ls, my, rng) {
    const phi = lat * Math.PI / 180, Ls = ls * Math.PI / 180, z0 = Math.min(...z);
    const Tinv = 203 + 9 * Math.cos(phi) + 4 * Math.sin(Ls) * Math.sign(lat + 1e-9) + rng.normal(0, 3);
    const invDepth = rng.uniform(1, 3), invStr = rng.uniform(6, 22) * (0.6 + 0.4 * Math.abs(Math.sin(phi)));
    const gamma = rng.uniform(1.6, 2.4), Tmid = 150 + 22 * Math.cos(phi) ** 2 + (my === 29 ? 8 : 0) + rng.normal(0, 3);
    const zi = z0 + invDepth, k = 0.35;
    const T = z.map((zz) => { const lin = Tinv - gamma * (zz - zi); const x = k * (lin - Tmid); return Tmid + (x > 30 ? x : Math.log1p(Math.exp(x))) / k; });
    z.forEach((zz, i) => { if (zz < zi) T[i] -= invStr * ((zi - zz) / invDepth) ** 2; });
    for (let w = 0; w < 2; w++) {
      const lam = rng.uniform(8, 20), amp = rng.uniform(0.5, 2.0), ph = rng.uniform(0, 2 * Math.PI);
      z.forEach((zz, i) => { T[i] += amp * (1 + (zz - z0) / 40) * Math.sin(2 * Math.PI * zz / lam + ph); });
    }
    return T;
  }
  function generateDataset(perYear = [65, 46, 73], seed = 7, noiseScale = 1) {
    const rng = RNG(seed), profiles = [];
    Object.entries(YEAR_TRACKS).forEach(([myS, tr], yi) => {
      const my = +myS, n = perYear[yi] || 0;
      for (let i = 0; i < n; i++) {
        const f = i / Math.max(n - 1, 1);
        const ls = tr.ls[0] + f * (tr.ls[1] - tr.ls[0]);
        const lat = tr.lat[0] + f * (tr.lat[1] - tr.lat[0]) + rng.normal(0, 0.3);
        const lon = rng.uniform(0, 360);
        const zs = rng.uniform(-4.8, 0.5), span = rng.uniform(38, 46), dz = rng.uniform(0.4, 0.75);
        const z = []; for (let v = zs; v < zs + span; v += dz) z.push(v);
        const Tt = trueTemperature(z, lat, ls, my, rng), ztop = z[z.length - 1];
        const sigT = z.map((zz) => 0.15 + 9.8 * Math.exp(-(ztop - zz) / 7));
        const e = correlatedNoise(rng, z, 2.5);
        const T = Tt.map((t, k) => Math.min(260, Math.max(110, t + 0.5 * noiseScale * sigT[k] * e[k])));
        const P0 = 610 * (1 + 0.08 * Math.sin((ls - 250) * Math.PI / 180)) * Math.exp(rng.normal(0, 0.03));
        const lnP = integrateLnP(z, T, Math.log(P0) - zs / 11);
        const sigRel = z.map((zz) => 0.0015 + 0.03 * Math.exp(-(ztop - zz) / 7));
        const dlat = rng.uniform(-0.6, 0.6), dlon = rng.uniform(-0.6, 0.6), m = z.length;
        profiles.push({
          id: `SYN${my}_${String(i).padStart(3, "0")}`, my, ls, z, T, Ttrue: Tt, lnP, sigT,
          sigLnP: sigRel, n: T.map((t, k) => Math.exp(lnP[k]) / (K_B * t)),
          latArr: z.map((_, k) => lat + dlat * k / (m - 1)), lonArr: z.map((_, k) => (lon + dlon * k / (m - 1) + 360) % 360),
        });
      }
    });
    return profiles;
  }

  // ------------------------------------------------------------------ real CSV
  function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
    const head = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
    const need = ["Profile_ID", "Mars_Year", "Solar_Long", "Radius", "Latitude", "Longitude", "Geopotential", "Pressure", "Sigma_P", "Temperature", "Sigma_T", "Number_Density"];
    const miss = need.filter((c) => !head.includes(c));
    if (miss.length) throw new Error("The CSV is missing these columns: " + miss.join(", "));
    const col = Object.fromEntries(head.map((h, i) => [h, i]));
    const groups = new Map();
    for (let r = 1; r < lines.length; r++) {
      const c = lines[r].split(",");
      const id = c[col.Profile_ID].replace(/^"|"$/g, "");
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(c);
    }
    const profiles = [];
    for (const [id, rows] of groups) {
      rows.sort((a, b) => +a[col.Radius] - +b[col.Radius]);
      const g = (k) => rows.map((c) => +c[col[k]]);
      const P = g("Pressure");
      profiles.push({
        id, my: +rows[0][col.Mars_Year], ls: +rows[0][col.Solar_Long], z: g("Geopotential").map((v) => v / G0 / 1e3),
        T: g("Temperature"), lnP: P.map(Math.log), sigT: g("Sigma_T"), sigLnP: g("Sigma_P").map((s, i) => s / P[i]),
        n: g("Number_Density"), latArr: g("Latitude"), lonArr: g("Longitude"),
      });
    }
    return profiles;
  }
  function toCSV(profiles) {
    const out = ["Profile_ID,Mars_Year,Solar_Long,Radius,Latitude,Longitude,Geopotential,Pressure,Sigma_P,Temperature,Sigma_T,Number_Density,Sigma_n"];
    for (const p of profiles) p.z.forEach((z, i) => {
      const P = Math.exp(p.lnP[i]);
      out.push([p.id, p.my, p.ls.toFixed(3), (3389.5e3 + z * 1e3).toFixed(1), p.latArr[i].toFixed(4), p.lonArr[i].toFixed(4), (z * 1e3 * G0).toFixed(3),
        P.toPrecision(7), (p.sigLnP[i] * P).toPrecision(4), p.T[i].toFixed(3), p.sigT[i].toFixed(3), p.n[i].toPrecision(7), (p.sigLnP[i] * p.n[i]).toPrecision(4)].join(","));
    });
    return out.join("\n");
  }

  // ------------------------------------------------------------------ QC, audit, features
  function finalize(p) {
    p.zMin = p.z[0]; p.zMax = p.z[p.z.length - 1];
    p.lat = mean(p.latArr); p.lon = mean(p.lonArr);
    const L = p.ls * Math.PI / 180;
    p.sinls = Math.sin(L); p.cosls = Math.cos(L);
    p.sinlon = p.lonArr.map((l) => Math.sin(l * Math.PI / 180)); p.coslon = p.lonArr.map((l) => Math.cos(l * Math.PI / 180));
    return p;
  }
  function qualityControl(profiles, sigTMax = 5, minLevels = 40) {
    const keys = ["z", "T", "lnP", "sigT", "sigLnP", "n", "latArr", "lonArr", "Ttrue"];
    let trims = [], levelsRaw = 0;
    const kept = [];
    for (const p of profiles) {
      levelsRaw += p.z.length;
      const keep = p.sigT.map((s) => s <= sigTMax);
      const q = { id: p.id, my: p.my, ls: p.ls };
      for (const k of keys) if (p[k]) q[k] = p[k].filter((_, i) => keep[i]);
      if (q.z.length) trims.push(p.z[p.z.length - 1] - q.z[q.z.length - 1]);
      let mono = true; for (let i = 1; i < q.z.length; i++) if (!(q.z[i] > q.z[i - 1])) mono = false;
      if (mono && q.z.length >= minLevels) kept.push(finalize(q));
    }
    const levelsKept = kept.reduce((s, p) => s + p.z.length, 0);
    return { profiles: kept, qc: { profiles_raw: profiles.length, levels_raw: levelsRaw, profiles_kept: kept.length, levels_kept: levelsKept,
      median_top_trim_km: median(trims), median_span_km: median(kept.map((p) => p.zMax - p.zMin)), median_levels: median(kept.map((p) => p.z.length)) } };
  }
  function physicsAudit(profiles) {
    const ms = profiles.map((p) => {
      const m = [];
      for (let i = 1; i < p.z.length; i++) {
        const dPhi = (p.z[i] - p.z[i - 1]) * 1e3 * G0;
        if (dPhi > 0) m.push(K_B * 0.5 * (p.T[i] + p.T[i - 1]) * (p.lnP[i - 1] - p.lnP[i]) / dPhi);
      }
      return median(m);
    });
    phys.m = median(ms);
    let maxDev = 0;
    for (const p of profiles) p.T.forEach((t, i) => { const d = Math.abs(Math.exp(p.lnP[i]) / (p.n[i] * K_B * t) - 1); if (d > maxDev) maxDev = d; });
    phys.floor = median(profiles.map((p) => rms(hydroResidual(p.z, p.T, p.lnP))));
    return { mean_molecular_mass_gmol: mGmol(), molecular_mass_IQR_gmol: [quantile(ms, 0.25) * N_A * 1e3, quantile(ms, 0.75) * N_A * 1e3],
      H_ref_km: Href(), ideal_gas_max_rel_dev: maxDev, reference_hydro_residual_RMS_median: phys.floor };
  }

  // ------------------------------------------------------------------ splits and gaps
  function stratifiedSplit(profiles, fracs, seed) {
    const rng = RNG(seed), byYear = new Map();
    for (const p of profiles) { if (!byYear.has(p.my)) byYear.set(p.my, []); byYear.get(p.my).push(p.id); }
    const tr = [], va = [], te = [];
    for (const my of [...byYear.keys()].sort()) {
      const ids = rng.permutation(byYear.get(my).sort()), n = ids.length;
      if (n < 3) { tr.push(...ids); continue; }
      const nte = Math.max(1, Math.round(fracs[2] * n)), nva = Math.max(1, Math.round(fracs[1] * n));
      te.push(...ids.slice(0, nte)); va.push(...ids.slice(nte, nte + nva)); tr.push(...ids.slice(nte + nva));
    }
    return { train: tr.sort(), val: va.sort(), test: te.sort() };
  }
  function makeGap(z, size, loc, minObs = 5) {
    const z0 = z[0], z1 = z[z.length - 1];
    if (z1 - z0 - size < minObs) return null;
    let lo, hi;
    if (loc === "lower") { lo = z0 - 1e-9; hi = z0 + size; }
    else if (loc === "upper") { lo = z1 - size; hi = z1 + 1e-9; }
    else { const c = 0.5 * (z0 + z1); lo = c - size / 2; hi = c + size / 2; }
    const gap = z.map((v) => v >= lo && v <= hi), ng = gap.filter(Boolean).length;
    if (ng < 3 || z.length - ng < 5) return null;
    return gap;
  }
  function buildCases(byId, ids, configs, minObs) {
    const out = [];
    for (const id of ids) for (const [s, loc] of configs) if (makeGap(byId[id].z, s, loc, minObs)) out.push([id, s, loc]);
    return out;
  }

  // ------------------------------------------------------------------ baselines
  function pchipSlopes(x, y) {
    const n = x.length, h = [], d = [], m = new Array(n).fill(0);
    for (let k = 0; k < n - 1; k++) { h.push(x[k + 1] - x[k]); d.push((y[k + 1] - y[k]) / h[k]); }
    if (n === 2) return [d[0], d[0]];
    for (let k = 1; k < n - 1; k++) {
      if (d[k - 1] * d[k] <= 0) m[k] = 0;
      else { const w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1]; m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]); }
    }
    const end = (h0, h1, d0, d1) => {
      let v = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
      if (Math.sign(v) !== Math.sign(d0)) v = 0;
      else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(v) > Math.abs(3 * d0)) v = 3 * d0;
      return v;
    };
    m[0] = end(h[0], h[1], d[0], d[1]);
    m[n - 1] = end(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
    return m;
  }
  function pchipEval(x, y, m, xq) {
    let lo = 0, hi = x.length - 1;
    if (xq <= x[0]) return y[0];
    if (xq >= x[hi]) return y[hi];
    while (hi - lo > 1) { const k = (lo + hi) >> 1; if (x[k] <= xq) lo = k; else hi = k; }
    const hh = x[hi] - x[lo], t = (xq - x[lo]) / hh, t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * y[lo] + (t3 - 2 * t2 + t) * hh * m[lo] + (-2 * t3 + 3 * t2) * y[hi] + (t3 - t2) * hh * m[hi];
  }
  function interpExtrap(zo, yo, zn, kind = "linear", win = 3) {
    const o = [...zo.keys()].sort((a, b) => zo[a] - zo[b]);
    const z = o.map((i) => zo[i]), y = o.map((i) => yo[i]), n = z.length;
    const m = kind === "pchip" ? pchipSlopes(z, y) : null;
    let below = null, above = null;
    return zn.map((q) => {
      if (q >= z[0] && q <= z[n - 1]) return kind === "pchip" ? pchipEval(z, y, m, q) : interp(q, z, y);
      if (q < z[0]) {
        if (!below) { let s = [...z.keys()].filter((i) => z[i] <= z[0] + win); if (s.length < 2) s = [0, 1]; below = polyfit1(s.map((i) => z[i]), s.map((i) => y[i])); }
        return below[0] * q + below[1];
      }
      if (!above) { let s = [...z.keys()].filter((i) => z[i] >= z[n - 1] - win); if (s.length < 2) s = [n - 2, n - 1]; above = polyfit1(s.map((i) => z[i]), s.map((i) => y[i])); }
      return above[0] * q + above[1];
    });
  }
  function hydrostaticIntegration(zo, To, Po, zg) {
    const Tg = interpExtrap(zo, To, zg, "pchip");
    const pts = zo.map((z, i) => [z, To[i]]).concat(zg.map((z, i) => [z, Tg[i]])).sort((a, b) => a[0] - b[0]);
    const za = pts.map((p) => p[0]), Ta = pts.map((p) => p[1]);
    const nf = Math.max(2000, 20 * za.length), zf = [], If = [0];
    for (let i = 0; i < nf; i++) zf.push(za[0] + (za[za.length - 1] - za[0]) * i / (nf - 1));
    const H = Href(), integ = zf.map((z) => -phys.Tref / (H * interp(z, za, Ta)));
    for (let i = 1; i < nf; i++) If.push(If[i - 1] + 0.5 * (integ[i] + integ[i - 1]) * (zf[i] - zf[i - 1]));
    const Ig = zg.map((z) => interp(z, zf, If));
    const gmin = Math.min(...zg), gmax = Math.max(...zg);
    const bel = zo.map((z, i) => i).filter((i) => zo[i] < gmin), abv = zo.map((z, i) => i).filter((i) => zo[i] > gmax);
    const preds = [];
    let zb, zaa;
    if (bel.length) { const ib = bel.reduce((a, b) => (zo[b] > zo[a] ? b : a)); zb = zo[ib]; const c = Po[ib] - interp(zb, zf, If); preds.push(Ig.map((v) => c + v)); }
    if (abv.length) { const ia = abv.reduce((a, b) => (zo[b] < zo[a] ? b : a)); zaa = zo[ia]; const c = Po[ia] - interp(zaa, zf, If); preds.push(Ig.map((v) => c + v)); }
    const Pg = preds.length === 2 ? zg.map((z, i) => { const w = (z - zb) / (zaa - zb); return (1 - w) * preds[0][i] + w * preds[1][i]; }) : preds[0];
    return [Tg, Pg];
  }
  function gpPredict(zo, yo, zn) {
    const [b, a] = polyfit1(zo, yo), n = zo.length;
    const r = zo.map((z, i) => yo[i] - (b * z + a)), sd = std(r) || 1, mu = mean(r);
    const y = r.map((v) => (v - mu) / sd);
    let best = null;
    for (const ell of [0.4, 0.7, 1, 1.5, 2.5, 4, 6, 9, 14, 22, 35]) for (const sf2 of [0.3, 1, 3]) for (const sn2 of [1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.6]) {
      const K = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = sf2 * Math.exp(-0.5 * ((zo[i] - zo[j]) / ell) ** 2) + (i === j ? sn2 : 0);
      const L = cholesky(K, n); if (!L) continue;
      const alpha = cholSolve(L, n, y);
      let lml = 0; for (let i = 0; i < n; i++) lml -= 0.5 * y[i] * alpha[i] + Math.log(L[i * n + i]);
      if (!best || lml > best.lml) best = { lml, ell, sf2, sn2, L, alpha };
    }
    const { ell, sf2, sn2, L, alpha } = best;
    const mean_ = [], sdv = [];
    for (const q of zn) {
      const k = zo.map((z) => sf2 * Math.exp(-0.5 * ((q - z) / ell) ** 2));
      let m = 0; for (let i = 0; i < n; i++) m += k[i] * alpha[i];
      const v = forwardSub(L, n, k); let vv = 0; for (const t of v) vv += t * t;
      mean_.push(m * sd + mu + b * q + a); sdv.push(Math.sqrt(Math.max(sf2 + sn2 - vv, 1e-12)) * sd);
    }
    return { mean: mean_, sd: sdv, hyper: { ell, sf2, sn2 } };
  }
  function fitEOF(profiles, key, opts = {}) {
    const dz = opts.dz || 0.5, varFrac = opts.varFrac || 0.99, maxK = opts.maxK || 15, minCover = opts.minCover || 0.2;
    const zmin = Math.floor(Math.min(...profiles.map((p) => p.zMin))), zmax = Math.ceil(Math.max(...profiles.map((p) => p.zMax)));
    let grid = []; for (let z = zmin; z <= zmax + 1e-9; z += dz) grid.push(z);
    let X = profiles.map((p) => grid.map((g) => (g >= p.zMin && g <= p.zMax ? interp(g, p.z, p[key]) : NaN)));
    const cover = grid.map((_, j) => X.filter((r) => Number.isFinite(r[j])).length / X.length);
    const keepJ = grid.map((_, j) => j).filter((j) => cover[j] >= minCover);
    grid = keepJ.map((j) => grid[j]); X = X.map((r) => keepJ.map((j) => r[j]));
    const nP = X.length, G = grid.length;
    const mask = X.map((r) => r.map(Number.isFinite));
    const colMean = grid.map((_, j) => mean(X.map((r) => r[j])));
    let Xf = X.map((r) => r.map((v, j) => (Number.isFinite(v) ? v : colMean[j])));
    let k = null, eig, mu;
    const decompose = () => {
      mu = grid.map((_, j) => mean(Xf.map((r) => r[j])));
      const C = new Float64Array(G * G);
      for (const r of Xf) { const d = r.map((v, j) => v - mu[j]); for (let i = 0; i < G; i++) for (let j = i; j < G; j++) C[i * G + j] += d[i] * d[j]; }
      for (let i = 0; i < G; i++) for (let j = i; j < G; j++) { C[i * G + j] /= Math.max(nP - 1, 1); C[j * G + i] = C[i * G + j]; }
      eig = jacobiEigen(C, G);
    };
    for (let it = 0; it < 15; it++) {
      decompose();
      if (k === null) {
        const tot = eig.values.reduce((s, v) => s + Math.max(v, 0), 0); let c = 0; k = 0;
        while (k < eig.values.length && c / tot < varFrac) { c += Math.max(eig.values[k], 0); k++; }
        k = Math.max(1, Math.min(maxK, k));
      }
      let delta = 0;
      Xf = Xf.map((r, ri) => {
        const d = r.map((v, j) => v - mu[j]);
        const coef = eig.vectors.slice(0, k).map((v) => v.reduce((s, vj, j) => s + vj * d[j], 0));
        return r.map((v, j) => {
          if (mask[ri][j]) return v;
          let rec = mu[j]; for (let q = 0; q < k; q++) rec += coef[q] * eig.vectors[q][j];
          delta = Math.max(delta, Math.abs(rec - v)); return rec;
        });
      });
      if (delta < 1e-4) break;
    }
    decompose();
    let se = 0, cnt = 0;
    Xf.forEach((r, ri) => {
      const d = r.map((v, j) => v - mu[j]);
      const coef = eig.vectors.slice(0, k).map((v) => v.reduce((s, vj, j) => s + vj * d[j], 0));
      r.forEach((v, j) => { if (!mask[ri][j]) return; let rec = mu[j]; for (let q = 0; q < k; q++) rec += coef[q] * eig.vectors[q][j]; se += (v - rec) ** 2; cnt++; });
    });
    return { grid, mu, V: eig.vectors.slice(0, k), lam: eig.values.slice(0, k).map((v) => Math.max(v, 1e-10)), sig2: se / Math.max(cnt, 1), k, key };
  }
  function eofPredict(E, zo, yo, sigo, zn) {
    const g = E.grid, k = E.k, cl = (z) => Math.min(Math.max(z, g[0]), g[g.length - 1]);
    const Uo = zo.map((z) => E.V.map((v) => interp(cl(z), g, v))), Un = zn.map((z) => E.V.map((v) => interp(cl(z), g, v)));
    const mo = zo.map((z) => interp(cl(z), g, E.mu)), mn = zn.map((z) => interp(cl(z), g, E.mu));
    const rinv = sigo.map((s) => 1 / (E.sig2 + s * s));
    const A = new Float64Array(k * k), rhs = new Float64Array(k);
    for (let a = 0; a < k; a++) {
      A[a * k + a] += 1 / E.lam[a];
      for (let i = 0; i < zo.length; i++) { rhs[a] += Uo[i][a] * rinv[i] * (yo[i] - mo[i]); for (let b = 0; b < k; b++) A[a * k + b] += Uo[i][a] * rinv[i] * Uo[i][b]; }
    }
    const Sa = invSym(A, k);
    const ma = new Array(k).fill(0); for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) ma[a] += Sa[a * k + b] * rhs[b];
    return {
      mean: Un.map((u, i) => mn[i] + u.reduce((s, v, a) => s + v * ma[a], 0)),
      sd: Un.map((u) => { let v = E.sig2; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) v += u[a] * Sa[a * k + b] * u[b]; return Math.sqrt(v); }),
    };
  }

  // ------------------------------------------------------------------ neural network
  // Inputs x = (Z, lat, sin lon, cos lon, sin Ls, cos Ls); outputs (T, lnP). Forward pass carries
  // the tangent dX/dZ alongside every activation so d lnP/dZ is exact; gradients are by hand.
  function computeNorm(profiles) {
    const cols = [[], [], [], [], [], []], T = [], P = [];
    for (const p of profiles) p.z.forEach((z, i) => { cols[0].push(z); cols[1].push(p.latArr[i]); cols[2].push(p.sinlon[i]); cols[3].push(p.coslon[i]); cols[4].push(p.sinls); cols[5].push(p.cosls); T.push(p.T[i]); P.push(p.lnP[i]); });
    return { inMu: cols.map(mean), inSd: cols.map((c) => Math.max(std(c), 1e-6)), outMu: [mean(T), mean(P)], outSd: [std(T), std(P)] };
  }
  function makeNet(norm, hidden, depth, seed) {
    const rng = RNG(seed * 7919 + 13), layers = [];
    let nin = 6;
    for (let l = 0; l <= depth; l++) {
      const nout = l === depth ? 2 : hidden, sd = Math.sqrt(2 / (nin + nout));
      layers.push({ nin, nout, W: Float64Array.from({ length: nout * nin }, () => rng.normal(0, sd)), b: new Float64Array(nout) });
      nin = nout;
    }
    return { norm, layers, hidden, depth };
  }
  const cloneNet = (net) => ({ norm: net.norm, hidden: net.hidden, depth: net.depth, layers: net.layers.map((L) => ({ nin: L.nin, nout: L.nout, W: Float64Array.from(L.W), b: Float64Array.from(L.b) })) });
  const zerosLike = (net) => net.layers.map((L) => ({ W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) }));
  function featuresOf(p, i, norm) {
    const raw = [p.z[i], p.latArr[i], p.sinlon[i], p.coslon[i], p.sinls, p.cosls];
    return Float64Array.from(raw, (v, j) => (v - norm.inMu[j]) / norm.inSd[j]);
  }
  function featuresAt(p, z, norm) {        // collocation coordinates (no atmospheric values)
    const lat = interp(z, p.z, p.latArr), sl = interp(z, p.z, p.sinlon), cl = interp(z, p.z, p.coslon), r = Math.hypot(sl, cl) || 1;
    const raw = [z, lat, sl / r, cl / r, p.sinls, p.cosls];
    return Float64Array.from(raw, (v, j) => (v - norm.inMu[j]) / norm.inSd[j]);
  }
  function makeCache(net) { return { a: net.layers.map((L) => new Float64Array(L.nout)), t: net.layers.map((L) => new Float64Array(L.nout)) }; }
  function forward(net, x, cache, tangent) {
    const L = net.layers, nl = L.length, sd0 = net.norm.inSd[0];
    let a = x, t = null;
    if (tangent) { t = new Float64Array(6); t[0] = 1 / sd0; }
    for (let l = 0; l < nl; l++) {
      const { W, b, nin, nout } = L[l], ao = cache.a[l], to = cache.t[l], last = l === nl - 1;
      for (let o = 0; o < nout; o++) {
        let h = b[o], dh = 0; const row = o * nin;
        for (let i = 0; i < nin; i++) h += W[row + i] * a[i];
        if (tangent) for (let i = 0; i < nin; i++) dh += W[row + i] * t[i];
        if (last) { ao[o] = h; to[o] = dh; }
        else { const th = Math.tanh(h); ao[o] = th; to[o] = (1 - th * th) * dh; }
      }
      a = ao; t = to;
    }
    return cache;
  }
  // gy: dLoss/dy (2), gty: dLoss/d(dy/dZ) (2)
  function backward(net, x, cache, gy, gty, grads, tangent) {
    const L = net.layers, nl = L.length;
    let ga = Float64Array.from(gy), gt = tangent ? Float64Array.from(gty) : null;
    const t0 = new Float64Array(6); t0[0] = 1 / net.norm.inSd[0];
    for (let l = nl - 1; l >= 0; l--) {
      const { W, nin, nout } = L[l], G = grads[l];
      const aPrev = l === 0 ? x : cache.a[l - 1], tPrev = l === 0 ? t0 : cache.t[l - 1];
      let gh, gdh = null;
      if (l === nl - 1) { gh = ga; gdh = gt; }
      else {
        const a = cache.a[l], t = cache.t[l];
        gh = new Float64Array(nout);
        if (tangent) gdh = new Float64Array(nout);
        for (let o = 0; o < nout; o++) {
          const s = 1 - a[o] * a[o];
          let gA = ga[o];
          if (tangent) { const dh = s > 1e-300 ? t[o] / s : 0; gdh[o] = s * gt[o]; gA += -2 * a[o] * dh * gt[o]; }
          gh[o] = s * gA;
        }
      }
      const gaPrev = new Float64Array(nin), gtPrev = tangent ? new Float64Array(nin) : null;
      for (let o = 0; o < nout; o++) {
        const row = o * nin, g1 = gh[o], g2 = tangent ? gdh[o] : 0;
        G.b[o] += g1;
        for (let i = 0; i < nin; i++) {
          G.W[row + i] += g1 * aPrev[i] + (tangent ? g2 * tPrev[i] : 0);
          if (l > 0) { gaPrev[i] += W[row + i] * g1; if (tangent) gtPrev[i] += W[row + i] * g2; }
        }
      }
      ga = gaPrev; gt = gtPrev;
    }
  }
  function predictNet(net, X) {
    const c = makeCache(net), { outMu, outSd } = net.norm;
    return X.map((x) => { forward(net, x, c, false); const y = c.a[c.a.length - 1]; return [y[0] * outSd[0] + outMu[0], y[1] * outSd[1] + outMu[1]]; });
  }
  // Points: {x, y:[normT, normlnP], w:[wT,wP]}; colloc: [x...]
  function lossAndGrad(net, pts, colloc, lam, grads, ref) {
    const c = makeCache(net), { outMu, outSd } = net.norm, H = Href(), Tref = phys.Tref;
    let ld = 0, lp = 0;
    const N = pts.length, Nc = colloc.length;
    for (const p of pts) {
      forward(net, p.x, c, false);
      const y = c.a[c.a.length - 1], r0 = y[0] - p.y[0], r1 = y[1] - p.y[1];
      ld += (p.w[0] * r0 * r0 + p.w[1] * r1 * r1) / N;
      if (grads) backward(net, p.x, c, [2 * p.w[0] * r0 / N, 2 * p.w[1] * r1 / N], [0, 0], grads, false);
    }
    if (lam > 0) for (const x of colloc) {
      forward(net, x, c, true);
      const y = c.a[c.a.length - 1], ty = c.t[c.t.length - 1];
      const T = y[0] * outSd[0] + outMu[0], Tc = Math.max(T, 50), dlnP = ty[1] * outSd[1];
      const r = H * dlnP + Tref / Tc;
      lp += r * r / Nc;
      if (grads) {
        const k = lam * 2 * r / Nc;
        backward(net, x, c, [T > 50 ? k * (-Tref / (Tc * Tc)) * outSd[0] : 0, 0], [0, k * H * outSd[1]], grads, true);
      }
    }
    let lr = 0;
    if (ref && ref.alpha > 0) net.layers.forEach((Ly, l) => {
      for (const key of ["W", "b"]) { const A = Ly[key], A0 = ref.net.layers[l][key], G = grads ? grads[l][key] : null; for (let i = 0; i < A.length; i++) { const d = A[i] - A0[i]; lr += ref.alpha * d * d; if (G) G[i] += 2 * ref.alpha * d; } }
    });
    return { loss: ld + lam * lp + lr, ld, lp };
  }
  function makeAdam(net) { return { t: 0, m: zerosLike(net), v: zerosLike(net) }; }
  function adamStep(net, grads, opt, lr, clip = 1) {
    let nrm = 0; for (const G of grads) for (const k of ["W", "b"]) for (const g of G[k]) nrm += g * g;
    nrm = Math.sqrt(nrm); const sc = nrm > clip ? clip / nrm : 1;
    opt.t++; const b1 = 0.9, b2 = 0.999, eps = 1e-8, c1 = 1 - b1 ** opt.t, c2 = 1 - b2 ** opt.t;
    net.layers.forEach((L, l) => { for (const k of ["W", "b"]) { const P = L[k], G = grads[l][k], M = opt.m[l][k], V = opt.v[l][k];
      for (let i = 0; i < P.length; i++) { const g = G[i] * sc; M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g; P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + eps); G[i] = 0; } } });
  }
  function dataPoints(profiles, norm, sigFloorT = 0.5, sigFloorP = 0.003, wNorm = null) {
    const pts = [];
    for (const p of profiles) p.z.forEach((_, i) => {
      const wT = 1 / Math.max(p.sigT[i], sigFloorT) ** 2, wP = 1 / Math.max(p.sigLnP[i], sigFloorP) ** 2;
      pts.push({ x: featuresOf(p, i, norm), y: [(p.T[i] - norm.outMu[0]) / norm.outSd[0], (p.lnP[i] - norm.outMu[1]) / norm.outSd[1]], w: [wT, wP], z: p.z[i] });
    });
    if (!wNorm) wNorm = [mean(pts.map((q) => q.w[0])), mean(pts.map((q) => q.w[1]))];
    for (const q of pts) { q.w = [q.w[0] / wNorm[0], q.w[1] / wNorm[1]]; }
    return { pts, wNorm };
  }
  const sleep0 = () => new Promise((r) => setTimeout(r, 0));

  // Global pre-training (async, yields to keep the page responsive)
  async function pretrain(opts) {
    const { trainP, valP, norm, lam, seed, hidden, depth, steps, lr, batch, collocBatch, onProgress, shouldStop } = opts;
    const net = makeNet(norm, hidden, depth, seed), rng = RNG(seed + 101);
    const { pts, wNorm } = dataPoints(trainP, norm);
    const val = dataPoints(valP, norm, 0.5, 0.003, wNorm).pts;
    const pool = [];
    for (const p of trainP) for (let k = 0; k < 40; k++) pool.push(featuresAt(p, rng.uniform(p.zMin, p.zMax), norm));
    const grads = zerosLike(net), opt = makeAdam(net), hist = [];
    let best = Infinity, bestNet = cloneNet(net), t0 = performance.now();
    for (let s = 1; s <= steps; s++) {
      const bp = Array.from({ length: Math.min(batch, pts.length) }, () => pts[rng.int(pts.length)]);
      const bc = lam > 0 ? Array.from({ length: collocBatch }, () => pool[rng.int(pool.length)]) : [];
      const { ld, lp } = lossAndGrad(net, bp, bc, lam, grads);
      const lrNow = 1e-4 + 0.5 * (lr - 1e-4) * (1 + Math.cos(Math.PI * s / steps));
      adamStep(net, grads, opt, lrNow);
      if (s % 25 === 0 || s === steps) {
        const vl = lossAndGrad(net, val, [], 0, null).ld;
        hist.push({ step: s, train: ld, phys: lp, val: vl });
        if (vl < best) { best = vl; bestNet = cloneNet(net); }
        if (onProgress) onProgress(s, steps, hist);
      }
      if (performance.now() - t0 > 30) { await sleep0(); t0 = performance.now(); if (shouldStop && shouldStop()) break; }
    }
    bestNet.wNorm = wNorm;
    return { net: bestNet, hist };
  }

  // Per-profile fine-tuning: L2-SP towards pre-trained weights (alpha) and physics weight lam
  async function finetune(base, prof, obsIdx, gapLo, gapHi, o) {
    const net = cloneNet(base), norm = base.norm, rng = RNG(o.seed || 1);
    const sub = { ...prof, z: obsIdx.map((i) => prof.z[i]), T: obsIdx.map((i) => o.Tobs ? o.Tobs[i] : prof.T[i]), lnP: obsIdx.map((i) => o.Pobs ? o.Pobs[i] : prof.lnP[i]),
      sigT: obsIdx.map((i) => prof.sigT[i]), sigLnP: obsIdx.map((i) => prof.sigLnP[i]), latArr: obsIdx.map((i) => prof.latArr[i]),
      sinlon: obsIdx.map((i) => prof.sinlon[i]), coslon: obsIdx.map((i) => prof.coslon[i]) };
    const { pts } = dataPoints([sub], norm, 0.5, 0.003, base.wNorm);
    const colloc = [];
    for (let k = 0; k < (o.nColloc || 64); k++) colloc.push(featuresAt(prof, rng.uniform(gapLo, gapHi), norm));
    for (let k = 0; k < (o.nColloc || 64) / 2; k++) colloc.push(featuresAt(prof, rng.uniform(prof.zMin, prof.zMax), norm));
    const grads = zerosLike(net), opt = makeAdam(net), ref = o.alpha > 0 ? { net: base, alpha: o.alpha } : null;
    let t0 = performance.now();
    for (let s = 0; s < o.steps; s++) {
      lossAndGrad(net, pts, colloc, o.lam, grads, ref);
      adamStep(net, grads, opt, o.lr);
      if (performance.now() - t0 > 40) { await sleep0(); t0 = performance.now(); }
    }
    return net;
  }

  // ------------------------------------------------------------------ one reconstruction case
  const METHODS = {
    lin: "Linear", pchip: "PCHIP", hydro: "Hydrostatic int.", gp: "Gaussian process", eof: "EOF (opt. interp.)",
    mlp_zs: "MLP (zero-shot)", pinn_zs: "PINN (zero-shot)", mlp_rft: "MLP (reg. fine-tuned)", pinn_rft: "PINN (reg. fine-tuned)",
  };
  const OURS = METHODS.pinn_rft;
  async function runCase(prof, gap, ctx, o) {
    const rng = RNG(o.seed || 1), sigFloor = [0.5, 0.003];
    const obsIdx = [], hidIdx = [];
    gap.forEach((g, i) => (g ? hidIdx : obsIdx).push(i));
    const Tobs = prof.T.slice(), Pobs = prof.lnP.slice();
    if (o.noise > 0) obsIdx.forEach((i) => { Tobs[i] += rng.normal(0, o.noise * Math.max(prof.sigT[i], sigFloor[0])); Pobs[i] += rng.normal(0, o.noise * Math.max(prof.sigLnP[i], sigFloor[1])); });
    const zo = obsIdx.map((i) => prof.z[i]), To = obsIdx.map((i) => Tobs[i]), Po = obsIdx.map((i) => Pobs[i]), zh = hidIdx.map((i) => prof.z[i]);
    const out = {}, m = o.methods;
    if (m.includes("lin")) out.lin = { T: interpExtrap(zo, To, zh), P: interpExtrap(zo, Po, zh) };
    if (m.includes("pchip")) out.pchip = { T: interpExtrap(zo, To, zh, "pchip"), P: interpExtrap(zo, Po, zh, "pchip") };
    if (m.includes("hydro")) { const [T, P] = hydrostaticIntegration(zo, To, Po, zh); out.hydro = { T, P }; }
    if (m.includes("gp")) { const a = gpPredict(zo, To, zh), b = gpPredict(zo, Po, zh); out.gp = { T: a.mean, P: b.mean, sT: a.sd, sP: b.sd }; }
    if (m.includes("eof") && ctx.eof) {
      const a = eofPredict(ctx.eof.T, zo, To, obsIdx.map((i) => prof.sigT[i]), zh), b = eofPredict(ctx.eof.P, zo, Po, obsIdx.map((i) => prof.sigLnP[i]), zh);
      out.eof = { T: a.mean, P: b.mean, sT: a.sd, sP: b.sd };
    }
    const Xh = hidIdx.map((i) => featuresOf(prof, i, ctx.norm));
    const gLo = Math.min(...zh), gHi = Math.max(...zh);
    for (const fam of ["mlp", "pinn"]) {
      const members = (ctx.nets && ctx.nets[fam]) || [];
      if (!members.length) continue;
      const lim = members.slice(0, o.members || members.length);
      const ens = async (fn) => {
        const preds = [];
        for (const net of lim) preds.push(predictNet(await fn(net), Xh));
        const T = zh.map((_, k) => mean(preds.map((p) => p[k][0]))), P = zh.map((_, k) => mean(preds.map((p) => p[k][1])));
        const sd = (j) => zh.map((_, k) => (preds.length > 1 ? std(preds.map((p) => p[k][j])) : NaN));
        return { T, P, sT: sd(0), sP: sd(1) };
      };
      if (m.includes(fam + "_zs")) out[fam + "_zs"] = await ens(async (n) => n);
      if (m.includes(fam + "_rft")) {
        const hp = o.hp[fam];
        out[fam + "_rft"] = await ens((n) => finetune(n, prof, obsIdx, gLo, gHi, { ...hp, lam: fam === "mlp" ? 0 : (o.lamOverride ?? hp.lam), steps: o.ftSteps, lr: o.ftLr, seed: o.seed, Tobs, Pobs }));
      }
    }
    const ref = { z: zh, T: hidIdx.map((i) => prof.T[i]), P: hidIdx.map((i) => prof.lnP[i]), sigT: hidIdx.map((i) => prof.sigT[i]), sigP: hidIdx.map((i) => prof.sigLnP[i]) };
    return { ref, preds: out, obs: { z: zo, T: To, P: Po } };
  }

  // ------------------------------------------------------------------ metrics and statistics
  const normPPF = (p) => {                 // Acklam's inverse normal
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) return -normPPF(1 - p);
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  };
  const normCDF = (x) => { const t = 1 / (1 + 0.2316419 * Math.abs(x)); const d = 0.3989423 * Math.exp(-x * x / 2); const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return x > 0 ? 1 - p : p; };
  const COV = [0.5, 0.8, 0.9, 0.95];
  function caseMetrics(ref, pr) {
    const n = ref.z.length, eT = pr.T.map((v, i) => v - ref.T[i]), eP = pr.P.map((v, i) => v - ref.P[i]);
    const d = {
      T_MAE: mean(eT.map(Math.abs)), T_RMSE: Math.sqrt(mean(eT.map((v) => v * v))), T_bias: mean(eT),
      T_within_2sig: mean(eT.map((v, i) => (Math.abs(v) <= 2 * ref.sigT[i] ? 1 : 0))),
      P_relMAE: 100 * mean(eP.map((v) => Math.abs(Math.exp(v) - 1))),
      n_relMAE: 100 * mean(eP.map((v, i) => Math.abs(Math.exp(v) * ref.T[i] / pr.T[i] - 1))),
      Hydro_RMS: n >= 3 ? rms(hydroResidual(ref.z, pr.T, pr.P)) : NaN,
    };
    if (pr.sT && pr.sT.every(Number.isFinite)) {
      const sT = pr.sT.map((s, i) => Math.hypot(s, ref.sigT[i])), sP = pr.sP.map((s, i) => Math.hypot(s, ref.sigP[i]));
      for (const q of COV) { const zq = normPPF(0.5 + q / 2); d["T_cov" + q * 100] = mean(eT.map((e, i) => (Math.abs(e) <= zq * sT[i] ? 1 : 0))); d["P_cov" + q * 100] = mean(eP.map((e, i) => (Math.abs(e) <= zq * sP[i] ? 1 : 0))); }
      d.scoresT = eT.map((e, i) => Math.abs(e) / sT[i]); d.scoresP = eP.map((e, i) => Math.abs(e) / sP[i]);
    }
    return d;
  }
  function bootCI(x, n = 2000, seed = 0) {
    x = x.filter(Number.isFinite); if (x.length < 2) return [NaN, NaN];
    const rng = RNG(seed + 1), b = [];
    for (let k = 0; k < n; k++) { let s = 0; for (let i = 0; i < x.length; i++) s += x[rng.int(x.length)]; b.push(s / x.length); }
    return [quantile(b, 0.025), quantile(b, 0.975)];
  }
  // rows: [{method, profile, case, ...metrics}] -> per method: profile-level means, mean + CI
  function summarize(rows, metrics, by = ["method"], nBoot = 2000) {
    const groups = new Map();
    for (const r of rows) { const key = by.map((k) => r[k]).join("\u0001"); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(r); }
    const out = [];
    for (const [key, rs] of groups) {
      const perProf = new Map();
      for (const r of rs) { if (!perProf.has(r.profile)) perProf.set(r.profile, []); perProf.get(r.profile).push(r); }
      const o = Object.fromEntries(by.map((k, i) => [k, rs[0][k]])); o.n_profiles = perProf.size;
      for (const m of metrics) {
        const vals = [...perProf.values()].map((list) => mean(list.map((r) => r[m])));
        o[m] = mean(vals); [o[m + "_lo"], o[m + "_hi"]] = bootCI(vals, nBoot);
      }
      out.push(o);
    }
    return out;
  }
  function wilcoxon(d) {                    // two-sided signed-rank test; exact for small n without ties
    d = d.filter((v) => v !== 0 && Number.isFinite(v)); const n = d.length;
    if (n === 0) return 1;
    const abs = d.map(Math.abs), order = [...abs.keys()].sort((a, b) => abs[a] - abs[b]), ranks = new Array(n);
    let ties = false, tieCorr = 0;
    for (let i = 0; i < n;) { let j = i; while (j + 1 < n && abs[order[j + 1]] === abs[order[i]]) j++; const r = (i + j + 2) / 2; for (let k = i; k <= j; k++) ranks[order[k]] = r; if (j > i) { ties = true; const t = j - i + 1; tieCorr += t ** 3 - t; } i = j + 1; }
    const Wp = d.reduce((s, v, i) => s + (v > 0 ? ranks[i] : 0), 0), Wm = n * (n + 1) / 2 - Wp, W = Math.min(Wp, Wm);
    if (!ties && n <= 30) {
      const maxS = n * (n + 1) / 2; let dist = new Float64Array(maxS + 1); dist[0] = 1;
      for (let k = 1; k <= n; k++) { const nd = new Float64Array(maxS + 1); for (let s = 0; s <= maxS; s++) { if (!dist[s]) continue; nd[s] += dist[s]; if (s + k <= maxS) nd[s + k] += dist[s]; } dist = nd; }
      const tot = 2 ** n; let c = 0; for (let s = 0; s <= W; s++) c += dist[s];
      return Math.min(1, 2 * c / tot);
    }
    const mu = n * (n + 1) / 4, sd = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tieCorr / 48);
    return Math.min(1, 2 * normCDF(-(Math.abs(W - mu) - 0.5) / sd));
  }
  function pairedTests(rows, metric, ours = OURS, alpha = 0.05, nBoot = 2000) {
    const pp = new Map();
    for (const r of rows) { const k = r.method + "\u0001" + r.profile; if (!pp.has(k)) pp.set(k, []); pp.get(k).push(r[metric]); }
    const val = (m, p) => { const v = pp.get(m + "\u0001" + p); return v ? mean(v) : NaN; };
    const profs = [...new Set(rows.map((r) => r.profile))], methods = [...new Set(rows.map((r) => r.method))].filter((m) => m !== ours);
    const out = [];
    for (const m of methods) {
      const pairs = profs.map((p) => [val(m, p), val(ours, p)]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
      if (pairs.length < 3) continue;
      const d = pairs.map(([a, b]) => a - b), ci = bootCI(d, nBoot);
      out.push({ metric, competitor: m, n_profiles: d.length, mean_diff: mean(d), ci_lo: ci[0], ci_hi: ci[1],
        rel_improvement_pct: 100 * mean(d) / mean(pairs.map((p) => p[0])), ours_better_frac: mean(d.map((v) => (v > 0 ? 1 : 0))),
        p_wilcoxon: d.every((v) => Math.abs(v) < 1e-12) ? 1 : wilcoxon(d) });
    }
    const o = [...out.keys()].sort((a, b) => out[a].p_wilcoxon - out[b].p_wilcoxon); let run = 0;
    o.forEach((idx, k) => { run = Math.max(run, Math.min(1, (out.length - k) * out[idx].p_wilcoxon)); out[idx].p_holm = run; out[idx].significant = run < alpha; });
    return out;
  }
  // split conformal: per (method, location, var) the ceil((n+1)q)/n quantile of normalised scores
  function fitConformal(rows) {
    const Q = {}, groups = new Map();
    for (const r of rows) { if (!r.scoresT) continue; const k = r.method + "|" + r.location; if (!groups.has(k)) groups.set(k, { T: [], P: [] }); groups.get(k).T.push(...r.scoresT); groups.get(k).P.push(...r.scoresP); }
    for (const [k, g] of groups) for (const v of ["T", "P"]) {
      const s = g[v].slice().sort((a, b) => a - b), n = s.length;
      for (const q of COV) { const lvl = Math.min(1, Math.ceil((n + 1) * q) / n); Q[`${k}|${v}|${q}`] = quantile(s, lvl); }
    }
    return Q;
  }
  function conformalCoverage(rows, Q) {
    for (const r of rows) {
      if (!r.scoresT) continue;
      for (const q of COV) for (const v of ["T", "P"]) {
        const c = Q[`${r.method}|${r.location}|${v}|${q}`];
        if (c === undefined) continue;
        r[`${v}_ccov${q * 100}`] = mean(r[v === "T" ? "scoresT" : "scoresP"].map((s) => (s <= c ? 1 : 0)));
      }
    }
    return rows;
  }

  return {
    phys, Href, mGmol, K_B, N_A, G0, RNG, hashSeed, mean, median, std, quantile, interp, gradient, jacobiEigen, cholesky,
    integrateLnP, hydroResidual, rms, generateDataset, parseCSV, toCSV, qualityControl, physicsAudit, stratifiedSplit, makeGap,
    buildCases, interpExtrap, hydrostaticIntegration, gpPredict, fitEOF, eofPredict, computeNorm, makeNet, cloneNet, zerosLike,
    featuresOf, featuresAt, makeCache, forward, backward, predictNet, lossAndGrad, pretrain, finetune, dataPoints, METHODS, OURS,
    runCase, caseMetrics, bootCI, summarize, wilcoxon, pairedTests, fitConformal, conformalCoverage, normPPF, COV,
  };
});
