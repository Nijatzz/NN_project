"""
stats/analysis.py

Statistical-rigor analysis for the VRP signal project. Regenerates every
headline test-set number with uncertainty and significance testing.

Reuses the exact pipeline from vrp_pipeline.py: same chronological splits,
same non-overlapping windowing (W=30, stride=10), same train-only
normalization. Nothing is re-fit on the test set. Each model is trained with
its own seed via train_one / train_mt_one (real MAX_EPOCHS / PATIENCE).

What it computes, on the 257 test windows:
  1. Per-seed test AUC / AP (N_SEEDS seeds), reported as mean with 95% CI
     (t-interval) and standard error, not a raw range.
  2. Ensemble (mean of per-seed predicted probabilities) test AUC / AP with
     bootstrap 95% CIs (label-stratified resampling of the 257 windows).
  3. Significance of the headline comparisons:
       - GRU multi-task (lambda=10) vs GRU VRP-only
       - GRU vs CNN
       - each ablation group (residuals, macro, all-20) vs DMM-latents-only
     reported two ways:
       (a) DeLong test on the paired ensemble AUCs (same windows / labels)
       (b) paired comparison across the matched seeds (Wilcoxon signed-rank)
     plus a paired bootstrap CI on the ensemble AP difference.
  4. ECE (10 quantile bins) for the GRU ensemble with a bootstrap 95% CI.

Run:  python stats/analysis.py
All randomness used for resampling is controlled by MASTER_SEED.
"""
import os, sys, json, time
import numpy as np

# import the project pipeline (run from repo root or stats/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vrp_pipeline as vp
from vrp_signals import VRPGRU, VRPCNN, VRPGRU_MT
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import norm, wilcoxon

# ── Settings ────────────────────────────────────────────────────────────────
N_SEEDS     = 20
SEEDS       = list(range(N_SEEDS))
N_BOOT      = 4000          # >= 2000 as requested
ECE_BINS    = 10
MASTER_SEED = 12345
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))


# ── DeLong test for two correlated ROC AUCs ──────────────────────────────────
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) array, first m columns are the positive class."""
    k = preds_sorted.shape[0]
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, p1, p2):
    """Two-sided DeLong test for AUC(p1) - AUC(p2) on the same labels."""
    y_true = np.asarray(y_true).astype(int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    order = np.concatenate([pos_idx, neg_idx])
    m = len(pos_idx)
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, m)
    cov = np.atleast_2d(cov)
    L = np.array([[1.0, -1.0]])
    var = float((L @ cov @ L.T).ravel()[0])
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    p = float(2 * norm.sf(abs(z)))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return {'auc1': float(aucs[0]), 'auc2': float(aucs[1]), 'diff': diff,
            'se': se, 'z': z, 'p': p, 'ci': ci}


# ── Bootstrap helpers (label-stratified resampling of test windows) ───────────
def _strat_indices(y, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])


def bootstrap_ci(y, p, metric_fn, n_boot=N_BOOT, seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _strat_indices(y, rng)
        vals[b] = metric_fn(y[idx], p[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), vals


def paired_bootstrap_diff(y, pa, pb, metric_fn, n_boot=N_BOOT, seed=MASTER_SEED):
    """Paired bootstrap CI + two-sided p for metric(pa) - metric(pb)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _strat_indices(y, rng)
        diffs[b] = metric_fn(y[idx], pa[idx]) - metric_fn(y[idx], pb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac_le = float(np.mean(diffs <= 0))
    frac_ge = float(np.mean(diffs >= 0))
    p = float(min(1.0, 2 * min(frac_le, frac_ge)))
    return {'diff_mean': float(diffs.mean()), 'ci': (float(lo), float(hi)), 'p': p}


def mean_ci(vals):
    """Mean with 95% t-CI and standard error across seeds."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else float('nan')
    se = sd / np.sqrt(n) if n > 1 else float('nan')
    # 95% t critical value
    from scipy.stats import t as tdist
    tcrit = float(tdist.ppf(0.975, n - 1)) if n > 1 else float('nan')
    return {'mean': mean, 'sd': sd, 'se': se,
            'ci': (mean - tcrit * se, mean + tcrit * se) if n > 1 else (float('nan'), float('nan'))}


# ── Train one configuration across seeds ──────────────────────────────────────
def train_config(name, model_fn, loaders, seeds, mt=False, lambda_rv=None):
    is_mt = mt
    test_loader = loaders[2]
    per_seed_preds = []
    per_seed_auc = []
    per_seed_ap = []
    y_ref = None
    anchors = loaders[5].anchor_indices
    t0 = time.time()
    for s in seeds:
        if is_mt:
            model, _, _ = vp.train_mt_one(model_fn, loaders, seed=s, lambda_rv=lambda_rv)
            p, y, _, _ = vp.collect_preds_mt(model, test_loader)
        else:
            model, _, _ = vp.train_one(model_fn, loaders, seed=s)
            p, y = vp.collect_preds(model, test_loader)
        per_seed_preds.append(p)
        per_seed_auc.append(roc_auc_score(y, p))
        per_seed_ap.append(average_precision_score(y, p))
        if y_ref is None:
            y_ref = y
    P = np.vstack(per_seed_preds)                 # (n_seeds, n_test)
    ens = P.mean(axis=0)
    dt = time.time() - t0
    print(f'  [{name}] {len(seeds)} seeds in {dt:.0f}s  '
          f'ensemble AUC={roc_auc_score(y_ref, ens):.4f} AP={average_precision_score(y_ref, ens):.4f}')
    return {
        'name': name,
        'y': np.asarray(y_ref),
        'anchors': np.asarray(anchors),
        'per_seed_preds': P,
        'ens': ens,
        'per_seed_auc': np.asarray(per_seed_auc),
        'per_seed_ap': np.asarray(per_seed_ap),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Loading data...  (N_SEEDS={N_SEEDS}, N_BOOT={N_BOOT}, ECE_BINS={ECE_BINS}, MASTER_SEED={MASTER_SEED})')
    data = vp.load_vrp_data('.')
    loaders = vp.make_loaders(data, vp.W, vp.STRIDE, vp.BATCH, vp.VAL_FRAC)

    # multi-task loaders
    data = vp.augment_data_mt(data)
    mt_loaders = vp.make_mt_loaders(data, vp.W, vp.STRIDE, vp.BATCH, vp.VAL_FRAC)

    # ablation feature-subset loaders (same windows, sliced feature columns)
    abl_loaders = {}
    for gname, idx in vp.FEATURE_GROUPS.items():
        sub = vp.subset_features(data, idx)
        abl_loaders[gname] = (vp.make_loaders(sub, vp.W, vp.STRIDE, vp.BATCH, vp.VAL_FRAC), len(idx))

    print('\nTraining configurations:')
    R = {}
    R['GRU']    = train_config('GRU VRP-only (20d)', lambda: VRPGRU(input_dim=20), loaders, SEEDS)
    R['CNN']    = train_config('CNN VRP-only (20d)', lambda: VRPCNN(input_dim=20), loaders, SEEDS)
    R['MT0.1']  = train_config('GRU MT lambda=0.1', lambda: VRPGRU_MT(input_dim=20), mt_loaders, SEEDS, mt=True, lambda_rv=0.1)
    R['MT1.0']  = train_config('GRU MT lambda=1.0', lambda: VRPGRU_MT(input_dim=20), mt_loaders, SEEDS, mt=True, lambda_rv=1.0)
    R['MT10']   = train_config('GRU MT lambda=10',  lambda: VRPGRU_MT(input_dim=20), mt_loaders, SEEDS, mt=True, lambda_rv=10.0)

    abl = {}
    for gname, (ldrs, ndim) in abl_loaders.items():
        abl[gname] = train_config(f'ABL [{gname}]',
                                  (lambda n=ndim: VRPGRU(input_dim=n)), ldrs, SEEDS)

    # ── Consistency checks: shared windows/labels across configs ──────────────
    y_gru = R['GRU']['y']
    for key in ['CNN', 'MT0.1', 'MT1.0', 'MT10']:
        assert np.array_equal(R[key]['anchors'], R['GRU']['anchors']), f'{key} anchors differ'
        assert np.array_equal(R[key]['y'], y_gru), f'{key} labels differ'
    for gname, r in abl.items():
        assert np.array_equal(r['anchors'], R['GRU']['anchors']), f'ablation {gname} anchors differ'
    print(f'\nShared test windows: {len(y_gru)}  positive rate={y_gru.mean():.4f}')

    auc = roc_auc_score
    ap  = average_precision_score

    # ── 1+5.1/5.3: per-seed mean CI and ensemble bootstrap CI ─────────────────
    def summarize(r, tag):
        ps_auc = mean_ci(r['per_seed_auc'])
        ps_ap  = mean_ci(r['per_seed_ap'])
        ens_auc = auc(r['y'], r['ens']); ens_ap = ap(r['y'], r['ens'])
        bl_auc = bootstrap_ci(r['y'], r['ens'], auc)
        bl_ap  = bootstrap_ci(r['y'], r['ens'], ap)
        ens5 = r['per_seed_preds'][:5].mean(axis=0)
        return {
            'tag': tag,
            'per_seed_auc': ps_auc, 'per_seed_ap': ps_ap,
            'ens_auc': ens_auc, 'ens_auc_ci': (bl_auc[0], bl_auc[1]),
            'ens_ap': ens_ap,  'ens_ap_ci': (bl_ap[0], bl_ap[1]),
            'ens5_auc': auc(r['y'], ens5), 'ens5_ap': ap(r['y'], ens5),
            'mean5_auc': float(r['per_seed_auc'][:5].mean()),
            'mean5_ap': float(r['per_seed_ap'][:5].mean()),
        }

    summ = {}
    for k in ['GRU', 'CNN', 'MT0.1', 'MT1.0', 'MT10']:
        summ[k] = summarize(R[k], R[k]['name'])
    for gname, r in abl.items():
        summ['ABL::' + gname] = summarize(r, r['name'])

    # ── 2: significance of comparisons ────────────────────────────────────────
    def compare(rA, rB, label):
        dl = delong_test(rA['y'], rA['ens'], rB['ens'])
        ap_diff = paired_bootstrap_diff(rA['y'], rA['ens'], rB['ens'], ap)
        # across-seed paired (matched seeds)
        dauc = rA['per_seed_auc'] - rB['per_seed_auc']
        try:
            w_stat, w_p = wilcoxon(rA['per_seed_auc'], rB['per_seed_auc'])
        except Exception:
            w_stat, w_p = float('nan'), float('nan')
        return {
            'label': label,
            'delong': dl,
            'ap_diff_boot': ap_diff,
            'seed_auc_mean_diff': float(dauc.mean()),
            'seed_auc_diff_ci': tuple(mean_ci(dauc)['ci']),
            'wilcoxon_p': float(w_p),
        }

    comparisons = {}
    comparisons['MT10_vs_GRU'] = compare(R['MT10'], R['GRU'], 'GRU MT(lambda=10) vs GRU VRP-only')
    comparisons['GRU_vs_CNN']  = compare(R['GRU'], R['CNN'], 'GRU vs CNN')
    dmm = abl['DMM latents (A+B, 12d)']
    for gname, r in abl.items():
        if 'DMM latents' in gname:
            continue
        comparisons['ABL_' + gname] = compare(r, dmm, f'[{gname}] vs DMM-latents-only')

    # ── 4: ECE with bootstrap CI (GRU ensemble) ───────────────────────────────
    ece_fn = lambda y, p: vp.compute_ece(p, y, n_bins=ECE_BINS)
    ece_results = {}
    for k in ['GRU', 'CNN']:
        e = ece_fn(R[k]['y'], R[k]['ens'])
        lo, hi, _ = bootstrap_ci(R[k]['y'], R[k]['ens'], ece_fn)
        ece_results[k] = {'ece': float(e), 'ci': (lo, hi), 'bins': ECE_BINS}

    # ── Print report ──────────────────────────────────────────────────────────
    def fmt_ci(c):
        return f'[{c[0]:.3f}, {c[1]:.3f}]'

    print('\n' + '=' * 92)
    print('PER-CONFIG TEST METRICS  (per-seed mean +/- 95% CI ; ensemble with bootstrap 95% CI)')
    print('=' * 92)
    print(f'{"Config":<26}{"seed AUC mean":>14}{"95% CI":>16}{"ens AUC":>9}{"ens AUC CI":>16}{"ens AP":>8}{"ens AP CI":>16}')
    order = ['GRU', 'CNN', 'MT0.1', 'MT1.0', 'MT10',
             'ABL::DMM latents (A+B, 12d)', 'ABL::Emission residuals (C, 4d)',
             'ABL::Macro/microstr (D-G, 4d)', 'ABL::All features (20d)']
    for k in order:
        s = summ[k]
        print(f'{s["tag"][:26]:<26}{s["per_seed_auc"]["mean"]:>14.3f}'
              f'{fmt_ci(s["per_seed_auc"]["ci"]):>16}{s["ens_auc"]:>9.3f}'
              f'{fmt_ci(s["ens_auc_ci"]):>16}{s["ens_ap"]:>8.3f}{fmt_ci(s["ens_ap_ci"]):>16}')

    print('\n' + '-' * 92)
    print('REPRODUCTION CHECK: first-5-seed numbers (compare to originally reported 5-seed run)')
    print('-' * 92)
    print(f'{"Config":<26}{"mean5 AUC":>11}{"ens5 AUC":>10}{"mean5 AP":>10}{"ens5 AP":>10}')
    for k in order:
        s = summ[k]
        print(f'{s["tag"][:26]:<26}{s["mean5_auc"]:>11.3f}{s["ens5_auc"]:>10.3f}{s["mean5_ap"]:>10.3f}{s["ens5_ap"]:>10.3f}')

    print('\n' + '=' * 92)
    print('SIGNIFICANCE OF KEY COMPARISONS')
    print('=' * 92)
    for key, c in comparisons.items():
        dl = c['delong']
        print(f'\n{c["label"]}')
        print(f'  ensemble AUC: {dl["auc1"]:.3f} vs {dl["auc2"]:.3f}  diff={dl["diff"]:+.3f}  '
              f'DeLong 95% CI {fmt_ci(dl["ci"])}  z={dl["z"]:+.2f}  p={dl["p"]:.3f}')
        apd = c['ap_diff_boot']
        print(f'  ensemble AP diff (paired bootstrap): {apd["diff_mean"]:+.3f}  '
              f'95% CI {fmt_ci(apd["ci"])}  p={apd["p"]:.3f}')
        print(f'  across-seed AUC diff: {c["seed_auc_mean_diff"]:+.3f}  '
              f'95% CI {fmt_ci(c["seed_auc_diff_ci"])}  Wilcoxon p={c["wilcoxon_p"]:.4f}')
        sig = 'SIGNIFICANT' if dl['p'] < 0.05 else 'NOT significant'
        print(f'  -> DeLong: {sig} at alpha=0.05')

    print('\n' + '=' * 92)
    print(f'CALIBRATION ECE  ({ECE_BINS} quantile bins, ensemble, bootstrap 95% CI)')
    print('=' * 92)
    for k, e in ece_results.items():
        print(f'  {k}: ECE={e["ece"]:.3f}  95% CI {fmt_ci(e["ci"])}')

    # ── Save ──────────────────────────────────────────────────────────────────
    def jsonify(o):
        if isinstance(o, dict):
            return {k: jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonify(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    out = {
        'settings': {'n_seeds': N_SEEDS, 'n_boot': N_BOOT, 'ece_bins': ECE_BINS,
                     'master_seed': MASTER_SEED, 'n_test_windows': int(len(y_gru)),
                     'test_pos_rate': float(y_gru.mean())},
        'summary': summ,
        'comparisons': comparisons,
        'ece': ece_results,
    }
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(jsonify(out), f, indent=2)
    np.savez(os.path.join(OUT_DIR, 'analysis_predictions.npz'),
             y=y_gru, anchors=R['GRU']['anchors'],
             **{f'{k}_ens': R[k]['ens'] for k in ['GRU', 'CNN', 'MT0.1', 'MT1.0', 'MT10']},
             **{f'{k}_perseed_auc': R[k]['per_seed_auc'] for k in ['GRU', 'CNN', 'MT0.1', 'MT1.0', 'MT10']})
    print(f'\nSaved {OUT_DIR}/results.json and analysis_predictions.npz')
    return out


if __name__ == '__main__':
    main()
