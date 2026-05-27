# VRP Signal Project — Handoff

## Context

University neural-network group project, 2 people. Predicting **Variance Risk Premium (VRP)** signals on SPY using a hierarchical deep generative model + downstream classifier.

### Part 1 (collaborator, done)
**Triple-layer Deep Markov Model (DMM)** in Pyro. Three latent states:
- `sm` (3-dim, monthly): macro state, pure Markov on previous macro
- `sc` (1-dim, monthly): corporate state, conditioned on prev corp + prev macro
- `f` (6-dim, daily): fast state, conditioned on prev fast + current corp + current macro
- K=20 fast steps per slow step. Strict t-1 causal guide (no future leakage).

DMM trained on 1993-03-30 → 2015-06-25 (5600 days / 280 months). Outputs saved to `dmm_artifacts.npz`:
- `f_tr`, `f_te`, `f_std_tr`, `f_std_te` — fast latents + uncertainty
- `sc_tr`, `sc_te`, `sc_std_tr`, `sc_std_te` — corp latents
- `sm_tr`, `sm_te`, `sm_std_tr`, `sm_std_te` — macro latents
- `fast_ep_*_st_mu/sig`, `_ln_mu/sig`, `_n_mu/sig`, `_df` — DMM's daily emission distribution params
- `log_ret_tr/te`, `fast_train_dates`, `fast_test_dates`

### Part 2 (mine — in progress)
Downstream classifier consuming DMM outputs to predict VRP. **Foundation built, several extensions remaining.**

## Files (in project root)

**From collaborator (don't modify):**
- `dmm_triple.py`, `dmm_triple_diagnostics.ipynb`, `dmm_artifacts.npz`
- `utils.py`, `fetch_data.py`
- `gru_signals.py`, `gru_signals.ipynb` — collaborator's initial 2-head model (VRP + MR). MR was dropped per collaborator's verdict ("MR — fuck that"). VRP works, kept and refactored.
- Data CSVs: `spy_ohlcv.csv`, `fred_*.csv`, `shiller_earnings.xlsx`, `sp500_more_data.xlsx`

**My Part 2 deliverables (current state):**
- `vrp_signals.py` — VRP-only refactor. Contains: `VRPDataset`, `VRPGRU` (2-layer GRU, 937 params), `VRPCNN` (2-layer dilated 1D conv, 769 params), `build_vrp_targets`, `vrp_loss` (asymmetric weighted BCE, α=5 on spike class)
- `vrp_features.py` — Loads `dmm_artifacts.npz` + raw data, builds 20-dim feature matrix, computes VRP targets. Exactly reproduces collaborator's feature pipeline (verified: positive rates match 0.693/0.747).
- `vrp_pipeline.py` — Training + eval + backtest as script. Multi-seed, chronological train/val split, early stopping.
- `vrp_pipeline.ipynb` — Notebook wrapper with all plots. Pre-executed.

## Key project constraints (from collaborator, treat as hard rules)

1. **No overlapping windows in training.** stride = horizon = 10. Overlapping windows → severe overfitting.
2. **Tiny models only.** Hidden sizes ≤ 8. Collaborator empirically found anything larger overfits the ~500-window training set.
3. **Train/val/test split:** TRAIN = 1993-03-30 → 2015-06-25 (5600 days), TEST = 2015-06-26 → 2025-10-27 (2600 days). The DMM artifacts already define this split — never re-split. Val = last 15% of train chronologically, used only for early stopping.
4. **Test set touched once.** All hyperparameter selection on train+val. Test gets a single end-of-pipeline evaluation. No tuning against test metrics.

## 20-dim feature vector (per day)

| Group | Dims | Source |
|-------|------|--------|
| A | 6 | Fast latent posterior means `f` |
| B | 6 | Log fast latent posterior stds `log(f_std)` |
| C | 4 | Emission residuals: actual - DMM emission_μ for [ret, log_rv, log_vix, log_rsv] |
| D | 1 | Baa-10Y credit spread (raw) |
| E | 1 | 20-day rolling correlation of SPY return with -Δyield |
| F | 1 | Log volume ratio (current / 20d MA) |
| G | 1 | Price z-score: (close - MA50) / rolling_std(close - MA50, 50d) |

All normalized z-score with **train-only** mean/std.

## VRP target

```
rs_minus[t] = min(log_ret[t], 0) ** 2
y_vrp[t] = 1  if  max(rs_minus[t+1 : t+11]) - median_train  <=  jump_threshold_train
           0  otherwise
```

where `median_train` and `jump_threshold_train = np.percentile(rs_minus_train, 95)` are fit on train only.

Last 10 targets are NaN. Positive rate: 0.693 train, 0.747 test. Baseline accuracy (predict-always-1) = 0.747 on test — accuracy is NOT a meaningful metric here, use AUC / AP / precision-at-threshold.

## Loss

Asymmetric BCE: `y=0` (spike incoming, dangerous to short vol) is weighted α=5×. This makes models conservative: they rarely predict p > 0.65, which is desired but means raw probabilities are under-confident (visible in calibration plot — fixing this is Task 1 below).

## Current results (5-seed ensemble, test set)

| Arch | AUC | AP | Notes |
|------|-----|-----|-------|
| GRU | 0.697 | 0.868 | Per-seed AUC: 0.479-0.700 (high variance) |
| CNN | 0.669 | 0.838 | Per-seed AUC: 0.474-0.671 (similar variance) |

**Threshold sweep:** GRU precision climbs from 0.75 base rate to ~0.90 at τ=0.6 → genuine skill at high confidence.
**Calibration:** Both models systematically under-confident. When they predict 0.4 the actual rate is 0.7. Asymmetric loss is the cause.
**Backtest (toy):** Naive short-vol gated at τ=0.5 — got hit by Feb 2018 volmageddon + Mar 2020. Final cum log return +0.30 (GRU) vs +1.36 buy-and-hold, Sharpe 0.10 vs 0.75. This profile (steady small wins + rare big losses) is realistic for short-vol but the backtest itself uses an oversimplified PnL model.

## Outstanding tasks (priority order)

### Priority 1 — must add for a defensible Part 2

**Task 1: Platt scaling for probability calibration**
- Fit `LogisticRegression(C=1.0).fit(p_val_raw.reshape(-1,1), y_val)` on the validation slice only
- Apply transform to test predictions
- Re-plot calibration (should be near-diagonal after fix)
- Re-run threshold sweep and backtest on calibrated probabilities
- **Critical**: fit Platt scaling on val, NOT test. Easy mistake to make.

**Task 2: Feature group ablation**
- Train 4 GRU variants using only feature groups:
  - DMM latents only (A+B, 12 dims)
  - Emission residuals only (C, 4 dims)
  - Macro/microstructure only (D+E+F+G, 4 dims)
  - All 20 dims (current baseline)
- Same 5 seeds, same hyperparameters, same train/val/test split
- Report test AUC mean±std per variant in a bar chart
- **This is the key chart that answers "did the DMM help?"** — the most important question for the writeup
- **Critical**: when ablating, you must rebuild `mu_feat`/`sig_feat` for the subsetted features, not reuse the 20-dim ones. See `vrp_features.py` for the pattern.

**Task 3: Add next-day realized volatility as a second target**
- Build `y_rv[t] = log(rs_vol[t+1])` (continuous regression target, train-normalized)
- Modify `VRPGRU`/`VRPCNN` to have two heads: VRP sigmoid head + RV linear head
- Multi-task loss: `vrp_loss(α=5) + λ * MSE(rv)`, start with λ=1.0
- Eval RV head: RMSE on test vs naive baseline (predict yesterday's RV); plot predicted vs actual scatter
- Hypothesis worth checking: does the multi-task objective improve VRP test AUC by regularizing the shared trunk? If yes, that's the headline finding.

### Priority 2 — nice to have

**Task 4: MLP-on-flattened-window baseline**
- Flatten 30×20 window to 600-dim, feed to 2-layer MLP with same total params (~900-1000)
- If it matches GRU/CNN, the temporal structure isn't actually being exploited → important negative finding for the writeup
- Drop into the existing `run_arch` loop

**Task 5: Walk-forward evaluation**
- Retrain annually on expanding window: train through 2016 → predict 2017, train through 2017 → predict 2018, etc.
- Stitch out-of-sample predictions across the 2015-2025 test period
- Re-compute AUC/AP on stitched OOS predictions
- Heavier compute but answers "is the model stable across regimes?"

### Priority 3 — only if time allows

**Task 6: Real backtest**
- Replace toy PnL with actual VIX futures front-month roll (CBOE has historical data) or VXX ETF
- Apply p_vrp gate properly; add transaction costs based on bid-ask
- Compute Sharpe, max drawdown, Calmar properly
- Only worth it if you find time at the end

## How to run

```bash
# Files needed in working directory:
# - dmm_artifacts.npz, spy_ohlcv.csv, fred_vix.csv, fred_dgs10.csv, fred_baa10y.csv
# - utils.py (from collaborator)
# - vrp_signals.py, vrp_features.py, vrp_pipeline.py
python vrp_pipeline.py        # full run, ~30s on CPU
# or open vrp_pipeline.ipynb in Jupyter / PyCharm
```

## Code style notes

- All randomness seeded via `seed` argument; don't add ad-hoc `np.random.*` calls
- Train/test split comes from `dmm_artifacts.npz` (T_tr=5600, T_te=2600). Don't override.
- Train normalization constants (`mu_feat`, `sig_feat`) computed from `feat_tr` only, applied to both
- VRP target threshold constants (`median_train`, `jump_threshold`) fit on train, frozen for test
- Use `VRPDataset.anchor_indices` to map dataset indices back to date indices for time-series plotting

## Two specific gotchas

1. **The DMM's emission params on the test set are causal** (RNN uses `x[:-1]`). They're safe to use as features. Don't second-guess this.

2. **Volume from `spy_ohlcv.csv` reads as strings by default.** The pipeline already casts via `spy = spy.astype(float)` in `load_vrp_data`. If you re-implement, watch for this.

## Writeup framing

- Part 1 (collaborator): DMM as hierarchical feature extractor for financial time series. Three timescales, three latent processes. Causal posterior inference.
- Part 2 (you): downstream signal modeling. Architecture comparison (GRU vs CNN vs MLP), feature ablation showing DMM features contribute beyond raw macro, probability calibration, multi-task learning with auxiliary RV head, backtest with honest caveats.
- MR head: brief paragraph on attempted second signal, why it failed (low base rate, magnitude weighting unstable), why it was dropped.
