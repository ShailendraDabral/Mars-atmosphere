// Node tests for docs/sim-core.js:  node tests/test_sim_core.mjs
import { createRequire } from "module";
import assert from "assert";
const require = createRequire(import.meta.url);
globalThis.performance ??= { now: () => Date.now() };
const S = require("../docs/sim-core.js");

let t = Date.now();
const raw = S.generateDataset([30, 22, 34], 7, 1);
const { profiles, qc } = S.qualityControl(raw, 5, 40);
const audit = S.physicsAudit(profiles);
console.log("qc", qc, "\naudit", audit);
assert(Math.abs(audit.mean_molecular_mass_gmol - 43.49) < 0.05, "molecular mass recovered");
assert(audit.ideal_gas_max_rel_dev < 1e-9, "ideal gas exact");
assert(audit.reference_hydro_residual_RMS_median < 0.01, "hydrostatic floor small");

// gradient check (data + physics + L2-SP)
const norm = S.computeNorm(profiles);
const net = S.makeNet(norm, 8, 2, 3), base = S.cloneNet(net);
base.layers[0].W[0] += 0.3;
const { pts } = S.dataPoints(profiles.slice(0, 2), norm);
const coll = profiles.slice(0, 2).flatMap((p) => [0.2, 0.5, 0.8].map((f) => S.featuresAt(p, p.zMin + f * (p.zMax - p.zMin), norm)));
const sub = pts.slice(0, 12), ref = { net: base, alpha: 0.7 }, lam = 0.9;
const g = S.zerosLike(net);
S.lossAndGrad(net, sub, coll, lam, g, ref);
let maxRel = 0;
for (const [l, key, i] of [[0, "W", 0], [0, "W", 5], [0, "b", 2], [1, "W", 7], [1, "b", 1], [2, "W", 3], [2, "b", 1], [2, "b", 0]]) {
  const P = net.layers[l][key], h = 1e-6, v = P[i];
  P[i] = v + h; const lp = S.lossAndGrad(net, sub, coll, lam, null, ref).loss;
  P[i] = v - h; const lm = S.lossAndGrad(net, sub, coll, lam, null, ref).loss;
  P[i] = v;
  const num = (lp - lm) / (2 * h), an = g[l][key][i];
  const rel = Math.abs(num - an) / Math.max(1e-8, Math.abs(num) + Math.abs(an));
  maxRel = Math.max(maxRel, rel);
}
console.log("gradient check max rel err", maxRel.toExponential(2));
assert(maxRel < 1e-5, "analytic gradients match finite differences");

// tangent check: d lnP/dZ from forward-mode vs finite difference in Z
const p0 = profiles[0], c = S.makeCache(net);
const x = S.featuresAt(p0, 10, norm); S.forward(net, x, c, true);
const dAn = c.t.at(-1)[1] * norm.outSd[1];
const xp = Float64Array.from(x); xp[0] += 1e-5 / norm.inSd[0];
const yA = S.predictNet(net, [x])[0][1], yB = S.predictNet(net, [xp])[0][1];
assert(Math.abs((yB - yA) / 1e-5 - dAn) < 1e-4 * Math.max(1, Math.abs(dAn)), "tangent = dlnP/dZ");

// hydrostatic integration reproduces a hydrostatic reference closely in a middle gap
const gap = S.makeGap(p0.z, 5, "middle");
const zo = p0.z.filter((_, i) => !gap[i]), zh = p0.z.filter((_, i) => gap[i]);
const [Tg, Pg] = S.hydrostaticIntegration(zo, p0.T.filter((_, i) => !gap[i]), p0.lnP.filter((_, i) => !gap[i]), zh);
const Pref = p0.lnP.filter((_, i) => gap[i]);
const perr = 100 * S.mean(Pg.map((v, i) => Math.abs(Math.exp(v - Pref[i]) - 1)));
console.log("hydro int 5 km middle P error %", perr.toFixed(3));
assert(perr < 1, "hydrostatic integration accurate for small interior gap");

// wilcoxon sanity: exact small-sample p, scipy gives 0.0078125 for 8 positive diffs
assert(Math.abs(S.wilcoxon([1, 2, 3, 4, 5, 6, 7, 8]) - 0.0078125) < 1e-9);

// short training + reconstruction
const sp = S.stratifiedSplit(profiles, [0.7, 0.15, 0.15], 2024);
const byId = Object.fromEntries(profiles.map((p) => [p.id, p]));
const tr = sp.train.map((i) => byId[i]), va = sp.val.map((i) => byId[i]);
const n2 = S.computeNorm(tr);
const nets = { mlp: [], pinn: [] };
for (const [fam, l] of [["mlp", 0], ["pinn", 1]]) {
  const r = await S.pretrain({ trainP: tr, valP: va, norm: n2, lam: l, seed: 0, hidden: 24, depth: 3, steps: 400, lr: 3e-3, batch: 128, collocBatch: 128 });
  console.log(fam, "val loss start/end", r.hist[0].val.toFixed(4), r.hist.at(-1).val.toFixed(4), "phys", r.hist.at(-1).phys.toExponential(2));
  nets[fam].push(r.net);
}
const eof = { T: S.fitEOF(tr, "T"), P: S.fitEOF(tr, "lnP") };
console.log("EOF modes", eof.T.k, eof.P.k);
const tp = byId[sp.test[0]], g2 = S.makeGap(tp.z, 10, "upper");
const res = await S.runCase(tp, g2, { norm: n2, nets, eof }, { methods: Object.keys(S.METHODS), hp: { mlp: { alpha: 1, lam: 0 }, pinn: { alpha: 1, lam: 0.1 } }, ftSteps: 100, ftLr: 1e-3, noise: 0, seed: 1 });
for (const [k, pr] of Object.entries(res.preds)) { const m = S.caseMetrics(res.ref, pr); console.log(S.METHODS[k].padEnd(24), m.T_MAE.toFixed(2), m.P_relMAE.toFixed(2), m.Hydro_RMS.toFixed(4)); }
console.log("all tests passed in", ((Date.now() - t) / 1000).toFixed(1), "s");
