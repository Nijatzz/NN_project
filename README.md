# Hierarchical Deep Markov Model + VRP Signal Network

University neural network group project, 2025.

## Overview

Two-stage architecture for predicting the Variance Risk Premium (VRP) signal on SPY:

**Part 1 — Triple-layer Deep Markov Model (DMM)**
Hierarchical latent variable model with macro / corporate / fast state layers.
Trained on 1993–2015 SPY data. Extracts a 6-dim daily latent representation
plus monthly slow-state summaries.
*Author: [Michael Haitin]*

**Part 2 — VRP Signal Network**
Downstream neural network consuming DMM outputs to predict whether the next
10 days will be free of downside volatility spikes. Includes GRU vs CNN
architecture comparison, feature ablation, calibration analysis, and multi-task
learning with an auxiliary realized volatility prediction head.
*Author: [Nijat Ahmadov]*

## Headline results (test set 2015–2025, 5-seed ensemble)

| Model | VRP AUC | Notes |
|---|---|---|
| GRU single-task | 0.697 | Baseline |
| CNN single-task | 0.669 | Architecture comparison |
| GRU multi-task λ=10 | **0.725** | Best result |

**Key finding 1 — Feature ablation:**
DMM latents alone → AUC 0.690. Raw macro data alone → AUC 0.484 (random).
The DMM is responsible for essentially all predictive signal.

**Key finding 2 — Multi-task learning:**
Adding an auxiliary next-day realized volatility regression head improves
VRP AUC from 0.697 to 0.725 and achieves R²=0.476 on log-vol prediction
vs R²=0.238 for a persistence baseline.

## How to reproduce

```bash
pip install torch scikit-learn pandas numpy matplotlib
python vrp_pipeline.py
```

Runs in ~5 minutes on CPU. Regenerates all plots and prints all results.

## File structure

| File | Description |
|---|---|
| `dmm_triple.py` | Part 1: DMM model definition |
| `dmm_triple_diagnostics.ipynb` | Part 1: training notebook |
| `dmm_artifacts.npz` | Part 1 output: latent states + emission params |
| `utils.py` | Shared utilities (log returns, RS vol, etc.) |
| `fetch_data.py` | Data download helpers |
| `vrp_signals.py` | Part 2: model architectures + dataset + loss |
| `vrp_features.py` | Part 2: feature engineering pipeline |
| `vrp_pipeline.py` | Part 2: training, calibration, multi-task, evaluation |
| `gru_signals.py` | Initial two-head signal model (collaborator) |
| `handoff.md` | Project state and task documentation |
| `*.csv / *.xlsx` | Raw market and macro data |
| `*.png` | Generated evaluation plots |

## Notes

- The toy short-vol **backtest was removed**: VRP is harvested through options, so a
  spot-only short-vol PnL cannot even approximately measure how much the signal network
  helps. Signal-level metrics (AUC / AP / precision-at-threshold) are the relevant
  evaluation for this representation-learning question — prediction quality is not a
  tradeable strategy.
- **Temperature scaling was removed** from the calibration analysis: it is rank-preserving
  (a monotone reparametrisation), so it cannot change any threshold-based decision and
  produced results indistinguishable from raw. Platt scaling + ECE are kept as the
  documented (negative) calibration result.

## Authors

- [Michael Haitin] - Part 1 (DMM architecture and training)
- [Nijat Ahmadov] - Part 2 (VRP signal network, ablations, multi-task extension)