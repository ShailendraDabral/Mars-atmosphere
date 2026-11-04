# Martian Profile Gap Lab — interactive simulator

A browser version of the full workflow, published at
<https://shailendradabral.github.io/Mars-atmosphere/>.

| File | Content |
|---|---|
| `index.html` | User interface: an overview with the reported real-data results, the *Gap challenge* (draw your own reconstruction and compete against the methods), and one panel per workflow step (data, QC, splits, gaps, training, reconstruction, benchmark, experiments, export) |
| `sim-core.js` | Computational engine: synthetic data, physics audit, baselines, MLP/PINN with hand-written derivatives, statistics |
| `img/` | Screenshots and the architecture diagram used in the main README |

Everything runs client-side in plain JavaScript with no build step and no dependencies. The network gradients are
checked against finite differences in [`tests/test_sim_core.mjs`](../tests/test_sim_core.mjs).

**Run locally:** `scripts/serve_simulator.sh`, then open <http://localhost:8000>, or open `index.html` directly.

**Deployment:** `.github/workflows/pages.yml` publishes this folder to GitHub Pages on every push to `main` that
changes `docs/` (repository *Settings → Pages → Source: GitHub Actions*).
