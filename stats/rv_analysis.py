"""
stats/rv_analysis.py

Statistical rigor for the auxiliary realized-volatility (RV) head used in the
multi-task models. Recomputes the RV R-squared and RMSE (in log-vol space) for
each lambda with bootstrap 95% CIs, alongside the persistence baseline that sees
the same information. Same splits / windowing / train-only normalization as the
rest of the project. Bootstrap resampling uses a fixed master seed.

Run:  python stats/rv_analysis.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vrp_pipeline as vp
from vrp_signals import VRPGRU_MT

N_SEEDS     = 20
SEEDS       = list(range(N_SEEDS))
N_BOOT      = 4000
MASTER_SEED = 12345
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))


def r2_score(actual, pred):
    actual = np.asarray(actual); pred = np.asarray(pred)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def boot_ci(actual, pred, fn, n_boot=N_BOOT, seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    n = len(actual)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(actual[idx], pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_boot_diff(actual, pred_a, pred_b, fn, n_boot=N_BOOT, seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    n = len(actual)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(actual[idx], pred_a[idx]) - fn(actual[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return float(diffs.mean()), (float(lo), float(hi)), p


def main():
    print(f'Loading data... (N_SEEDS={N_SEEDS}, N_BOOT={N_BOOT}, MASTER_SEED={MASTER_SEED})')
    data = vp.load_vrp_data('.')
    data = vp.augment_data_mt(data)
    mt_loaders = vp.make_mt_loaders(data, vp.W, vp.STRIDE, vp.BATCH, vp.VAL_FRAC)
    test_loader = mt_loaders[2]
    anchors = mt_loaders[5].anchor_indices

    # actual next-day log realized vol, and persistence (today's log vol)
    actual = np.log(np.clip(data['rs_vol_te'][anchors + 1], 1e-8, None))
    persist = np.log(np.clip(data['rs_vol_te'][anchors], 1e-8, None))

    results = {}
    for lam in [0.1, 1.0, 10.0]:
        per_seed_mu = []
        for s in SEEDS:
            model, _, _ = vp.train_mt_one(lambda: VRPGRU_MT(input_dim=20),
                                          mt_loaders, seed=s, lambda_rv=lam)
            _, _, mu_te, _ = vp.collect_preds_mt(model, test_loader)
            per_seed_mu.append(mu_te)
        mu_ens = np.vstack(per_seed_mu).mean(axis=0)
        pred_log = mu_ens * data['sig_rv'] + data['mu_rv']      # inverse transform
        r2 = r2_score(actual, pred_log)
        rm = rmse(actual, pred_log)
        r2_ci = boot_ci(actual, pred_log, r2_score)
        rm_ci = boot_ci(actual, pred_log, rmse)
        dmean, dci, dp = paired_boot_diff(actual, pred_log, persist, r2_score)
        results[f'lambda={lam}'] = {
            'r2': float(r2), 'r2_ci': r2_ci, 'rmse': rm, 'rmse_ci': rm_ci,
            'r2_minus_persist': dmean, 'r2_diff_ci': dci, 'r2_diff_p': dp,
        }
        print(f'  MT lambda={lam}: R2={r2:.3f} CI[{r2_ci[0]:.3f},{r2_ci[1]:.3f}]  '
              f'RMSE={rm:.4f} CI[{rm_ci[0]:.4f},{rm_ci[1]:.4f}]  '
              f'R2-persist diff={dmean:+.3f} CI[{dci[0]:.3f},{dci[1]:.3f}] p={dp:.3f}')

    p_r2 = r2_score(actual, persist)
    p_rm = rmse(actual, persist)
    p_r2_ci = boot_ci(actual, persist, r2_score)
    p_rm_ci = boot_ci(actual, persist, rmse)
    results['persistence'] = {'r2': float(p_r2), 'r2_ci': p_r2_ci,
                              'rmse': p_rm, 'rmse_ci': p_rm_ci}
    print(f'  Persistence: R2={p_r2:.3f} CI[{p_r2_ci[0]:.3f},{p_r2_ci[1]:.3f}]  '
          f'RMSE={p_rm:.4f} CI[{p_rm_ci[0]:.4f},{p_rm_ci[1]:.4f}]')

    with open(os.path.join(OUT_DIR, 'rv_results.json'), 'w') as f:
        json.dump({'n_seeds': N_SEEDS, 'n_boot': N_BOOT, 'master_seed': MASTER_SEED,
                   'n_test': int(len(actual)), 'results': results}, f, indent=2)
    print(f'\nSaved {OUT_DIR}/rv_results.json')


if __name__ == '__main__':
    main()
