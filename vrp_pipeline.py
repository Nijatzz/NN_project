
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vrp_features import load_vrp_data
from vrp_signals import VRPDataset, VRPGRU, VRPCNN, vrp_loss, count_params


# ── Config ────────────────────────────────────────────────────────────────────
W          = 30
STRIDE     = 10
BATCH      = 128
VRP_ALPHA  = 5.0
LR         = 1e-3
MAX_EPOCHS = 400
PATIENCE   = 40
VAL_FRAC   = 0.15
SEEDS      = [0, 1, 2, 3, 4]

# Feature group definitions (column indices into the 20-dim feature matrix)
FEATURE_GROUPS = {
    'DMM latents (A+B, 12d)':     list(range(0, 12)),   # fast means + log-stds
    'Emission residuals (C, 4d)':  list(range(12, 16)),  # actual - DMM emission_mu
    'Macro/microstr (D-G, 4d)':    list(range(16, 20)),  # Baa, corr, vol-ratio, pz
    'All features (20d)':          list(range(0, 20)),
}


# ── Data loaders ─────────────────────────────────────────────────────────────

def make_loaders(data, W, stride, batch, val_frac):
    full_train = VRPDataset(data['feat_tr'], data['y_vrp_tr'], W=W, stride=stride)
    n_val      = int(len(full_train) * val_frac)
    n_tr       = len(full_train) - n_val
    train_ds   = VRPDataset(data['feat_tr'], data['y_vrp_tr'], W=W, stride=stride)
    train_ds.starts = full_train.starts[:n_tr]
    val_ds     = VRPDataset(data['feat_tr'], data['y_vrp_tr'], W=W, stride=stride)
    val_ds.starts = full_train.starts[n_tr:]
    test_ds    = VRPDataset(data['feat_te'], data['y_vrp_te'], W=W, stride=stride)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True),
            DataLoader(val_ds,   batch_size=batch, shuffle=False),
            DataLoader(test_ds,  batch_size=batch, shuffle=False),
            train_ds, val_ds, test_ds)


def subset_features(data, indices):

    feat_tr_sub = data['feat_tr'][:, indices]
    feat_te_sub = data['feat_te'][:, indices]
    # Sanity check: sliced columns should still be ~zero-mean (train z-scoring)
    assert np.allclose(feat_tr_sub.mean(0), 0, atol=0.1), (
        f"Subset features not zero-mean! indices={indices}, "
        f"means={feat_tr_sub.mean(0).round(3)}")
    sub = dict(data)          # shallow copy -- shares all non-feature fields
    sub['feat_tr'] = feat_tr_sub
    sub['feat_te'] = feat_te_sub
    return sub


def augment_data_mt(data):
    def make_rv(rs_vol):
        y = np.full_like(rs_vol, np.nan)
        y[:-1] = np.log(np.clip(rs_vol[1:], 1e-8, None))
        return y
        
    y_rv_tr = make_rv(data['rs_vol_tr'])
    y_rv_te = make_rv(data['rs_vol_te'])
    
    # Normalize with train stats
    mu_rv = float(np.nanmean(y_rv_tr))
    sig_rv = float(np.nanstd(y_rv_tr))
    
    y_rv_tr_norm = (y_rv_tr - mu_rv) / sig_rv
    y_rv_te_norm = (y_rv_te - mu_rv) / sig_rv
    
    data['y_rv_tr_norm'] = y_rv_tr_norm
    data['y_rv_te_norm'] = y_rv_te_norm
    data['y_rv_tr_raw'] = y_rv_tr
    data['y_rv_te_raw'] = y_rv_te
    data['mu_rv'] = mu_rv
    data['sig_rv'] = sig_rv
    return data


def make_mt_loaders(data, W, stride, batch, val_frac):
    from vrp_signals import VRPMTDataset
    full_train = VRPMTDataset(data['feat_tr'], data['y_vrp_tr'], data['y_rv_tr_norm'], W=W, stride=stride)
    n_val      = int(len(full_train) * val_frac)
    n_tr       = len(full_train) - n_val
    train_ds   = VRPMTDataset(data['feat_tr'], data['y_vrp_tr'], data['y_rv_tr_norm'], W=W, stride=stride)
    train_ds.starts = full_train.starts[:n_tr]
    val_ds     = VRPMTDataset(data['feat_tr'], data['y_vrp_tr'], data['y_rv_tr_norm'], W=W, stride=stride)
    val_ds.starts = full_train.starts[n_tr:]
    test_ds    = VRPMTDataset(data['feat_te'], data['y_vrp_te'], data['y_rv_te_norm'], W=W, stride=stride)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True),
            DataLoader(val_ds,   batch_size=batch, shuffle=False),
            DataLoader(test_ds,  batch_size=batch, shuffle=False),
            train_ds, val_ds, test_ds)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(model_fn, loaders, seed, alpha=VRP_ALPHA, lr=LR,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_fn()
    opt   = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader, _, _, _, _ = loaders

    best_val, best_state, bad = float('inf'), None, 0
    history = {'train': [], 'val': []}
    for ep in range(max_epochs):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            p    = model(xb)
            loss = vrp_loss(p, yb, alpha=alpha)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        train_loss = tl / max(n, 1)

        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                p   = model(xb)
                vl += vrp_loss(p, yb, alpha=alpha).item() * len(xb); nv += len(xb)
        val_loss = vl / max(nv, 1)

        history['train'].append(train_loss); history['val'].append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f'    ep {ep:3d}] train={train_loss:.4f}  val={val_loss:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def train_mt_one(model_fn, loaders, seed, lambda_rv=1.0, alpha=VRP_ALPHA, lr=LR,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_fn()
    opt   = optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader, _, _, _, _ = loaders

    best_val, best_state, bad = float('inf'), None, 0
    history = {'train_tot': [], 'val_tot': [], 'train_vrp': [], 'train_rv': [], 'val_vrp': [], 'val_rv': []}
    
    for ep in range(max_epochs):
        model.train()
        tl_tot, tl_vrp, tl_rv, n = 0.0, 0.0, 0.0, 0
        for xb, yb_vrp, yb_rv in train_loader:
            opt.zero_grad()
            p_vrp, mu_rv = model(xb)
            loss_vrp = vrp_loss(p_vrp, yb_vrp, alpha=alpha)
            loss_rv = torch.nn.functional.mse_loss(mu_rv, yb_rv)
            loss = loss_vrp + lambda_rv * loss_rv
            loss.backward()
            opt.step()
            tl_tot += loss.item() * len(xb)
            tl_vrp += loss_vrp.item() * len(xb)
            tl_rv  += loss_rv.item() * len(xb)
            n += len(xb)
            
        train_loss = tl_tot / max(n, 1)
        history['train_tot'].append(train_loss)
        history['train_vrp'].append(tl_vrp / max(n, 1))
        history['train_rv'].append(tl_rv / max(n, 1))

        model.eval()
        vl_tot, vl_vrp, vl_rv, nv = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb_vrp, yb_rv in val_loader:
                p_vrp, mu_rv = model(xb)
                loss_vrp = vrp_loss(p_vrp, yb_vrp, alpha=alpha).item()
                loss_rv = torch.nn.functional.mse_loss(mu_rv, yb_rv).item()
                loss = loss_vrp + lambda_rv * loss_rv
                vl_tot += loss * len(xb)
                vl_vrp += loss_vrp * len(xb)
                vl_rv  += loss_rv * len(xb)
                nv += len(xb)
                
        val_loss = vl_tot / max(nv, 1)
        history['val_tot'].append(val_loss)
        history['val_vrp'].append(vl_vrp / max(nv, 1))
        history['val_rv'].append(vl_rv / max(nv, 1))
        
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f'    ep {ep:3d}] train_tot={train_loss:.4f}  val_tot={val_loss:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def collect_preds(model, loader):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(model(xb).numpy()); ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def collect_preds_mt(model, loader):
    model.eval()
    ps_vrp, ys_vrp, mus_rv, ys_rv = [], [], [], []
    with torch.no_grad():
        for xb, yb_vrp, yb_rv in loader:
            p_vrp, mu_rv = model(xb)
            ps_vrp.append(p_vrp.numpy())
            ys_vrp.append(yb_vrp.numpy())
            mus_rv.append(mu_rv.numpy())
            ys_rv.append(yb_rv.numpy())
    return np.concatenate(ps_vrp), np.concatenate(ys_vrp), np.concatenate(mus_rv), np.concatenate(ys_rv)

def eval_split(p, y, label):
    auc  = roc_auc_score(y, p)
    ap   = average_precision_score(y, p)
    acc  = ((p > 0.5).astype(int) == y.astype(int)).mean()
    base = max(y.mean(), 1 - y.mean())
    return {'split': label, 'AUC': auc, 'AP': ap, 'acc@0.5': acc, 'baseline_acc': base}


def run_arch(name, model_fn, loaders, seeds=SEEDS):
    print(f'\n=== {name} ({count_params(model_fn()):,} params) ===')
    _, _, test_loader, _, _, test_ds = loaders
    train_loader, val_loader, _, _, _, _ = loaders

    per_seed_test_preds = []
    per_seed_val_preds  = []
    per_seed_metrics    = []
    histories           = []
    y_va_ref            = None
    for s in seeds:
        model, hist, bv = train_one(model_fn, loaders, seed=s)
        histories.append(hist)
        p_tr, y_tr = collect_preds(model, train_loader)
        p_va, y_va = collect_preds(model, val_loader)
        p_te, y_te = collect_preds(model, test_loader)
        per_seed_test_preds.append(p_te)
        per_seed_val_preds.append(p_va)
        if y_va_ref is None:
            y_va_ref = y_va
        per_seed_metrics.append({
            'seed': s, 'best_val_loss': bv,
            **{f'tr_{k}': v for k, v in eval_split(p_tr, y_tr, 'tr').items() if k != 'split'},
            **{f'va_{k}': v for k, v in eval_split(p_va, y_va, 'va').items() if k != 'split'},
            **{f'te_{k}': v for k, v in eval_split(p_te, y_te, 'te').items() if k != 'split'},
        })
        print(f"  seed {s}: val={bv:.4f}  test_AUC={per_seed_metrics[-1]['te_AUC']:.3f}  "
              f"test_AP={per_seed_metrics[-1]['te_AP']:.3f}  "
              f"acc={per_seed_metrics[-1]['te_acc@0.5']:.3f}  "
              f"(baseline {per_seed_metrics[-1]['te_baseline_acc']:.3f})")

    p_te_ens = np.mean(per_seed_test_preds, axis=0)
    p_va_ens = np.mean(per_seed_val_preds,  axis=0)
    _, y_te  = collect_preds(model, test_loader)
    ens_metrics = eval_split(p_te_ens, y_te, 'test_ensemble')
    print(f"  ENSEMBLE: AUC={ens_metrics['AUC']:.3f}  AP={ens_metrics['AP']:.3f}  "
          f"acc@0.5={ens_metrics['acc@0.5']:.3f}")

    return {
        'name':             name,
        'per_seed_metrics': pd.DataFrame(per_seed_metrics),
        'p_te_ensemble':    p_te_ens,
        'p_va_ensemble':    p_va_ens,
        'y_va':             y_va_ref,
        'y_te':             y_te,
        'test_anchors':     test_ds.anchor_indices,
        'histories':        histories,
        'ensemble_metrics': ens_metrics,
    }


def run_mt_arch(name, model_fn, loaders, lambda_rv=1.0, seeds=SEEDS):
    print(f'\n=== {name} MT (lambda_rv={lambda_rv}, {count_params(model_fn()):,} params) ===')
    _, _, test_loader, _, _, test_ds = loaders
    train_loader, val_loader, _, _, _, _ = loaders

    per_seed_test_preds_vrp = []
    per_seed_test_preds_rv  = []
    per_seed_metrics        = []
    histories               = []
    for s in seeds:
        model, hist, bv = train_mt_one(model_fn, loaders, seed=s, lambda_rv=lambda_rv)
        histories.append(hist)
        p_tr, y_tr, mu_tr, y_rv_tr = collect_preds_mt(model, train_loader)
        p_va, y_va, mu_va, y_rv_va = collect_preds_mt(model, val_loader)
        p_te, y_te, mu_te, y_rv_te = collect_preds_mt(model, test_loader)
        
        per_seed_test_preds_vrp.append(p_te)
        per_seed_test_preds_rv.append(mu_te)
        
        rmse_tr = np.sqrt(np.mean((mu_tr - y_rv_tr)**2))
        rmse_va = np.sqrt(np.mean((mu_va - y_rv_va)**2))
        rmse_te = np.sqrt(np.mean((mu_te - y_rv_te)**2))
        
        m_te = eval_split(p_te, y_te, 'te')
        per_seed_metrics.append({
            'seed': s, 'best_val_loss': bv,
            'te_AUC': m_te['AUC'], 'te_AP': m_te['AP'],
            'rmse_tr': rmse_tr, 'rmse_va': rmse_va, 'rmse_te': rmse_te,
        })
        print(f"  seed {s}: val_tot={bv:.4f}  te_AUC={m_te['AUC']:.3f}  te_AP={m_te['AP']:.3f}  "
              f"te_RV_RMSE(norm)={rmse_te:.4f}")

    p_te_ens_vrp = np.mean(per_seed_test_preds_vrp, axis=0)
    mu_te_ens_rv = np.mean(per_seed_test_preds_rv,  axis=0)
    
    ens_metrics = eval_split(p_te_ens_vrp, y_te, 'test_ensemble')
    ens_rmse_te = np.sqrt(np.mean((mu_te_ens_rv - y_rv_te)**2))
    print(f"  ENSEMBLE: AUC={ens_metrics['AUC']:.3f}  AP={ens_metrics['AP']:.3f}  "
          f"RV_RMSE(norm)={ens_rmse_te:.4f}")

    return {
        'name':             name,
        'lambda_rv':        lambda_rv,
        'per_seed_metrics': pd.DataFrame(per_seed_metrics),
        'p_te_ensemble':    p_te_ens_vrp,
        'mu_te_ensemble':   mu_te_ens_rv,
        'y_te_vrp':         y_te,
        'y_te_rv':          y_rv_te,
        'test_anchors':     test_ds.anchor_indices,
        'histories':        histories,
        'ensemble_metrics': ens_metrics,
        'ensemble_rmse_te': ens_rmse_te,
    }


# ── Calibration helpers ────────────────────────────────────────────────────────

def fit_platt(p_val, y_val):
    """Platt scaling on val only. NEVER pass test data."""
    scaler = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    scaler.fit(p_val.reshape(-1, 1), y_val.astype(int))
    return scaler


def calibrate(p, scaler):
    return scaler.predict_proba(p.reshape(-1, 1))[:, 1].astype(np.float64)


def compute_ece(p, y, n_bins=10):
    """ECE with quantile (equal-frequency) bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(p)
    bin_edges   = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bin_indices = np.digitize(p, bin_edges[:-1]) - 1
    ece = 0.0
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


# ── Calibration plot ───────────────────────────────────────────────────────────

def plot_calibration_before_after(results_gru,
                                  save_path='vrp_calibration_before_after.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Probability Calibration -- Before vs After Platt Scaling  (test set)',
                 fontsize=14, fontweight='bold')

    res     = results_gru
    y_te    = res['y_te']
    p_raw   = res['p_te_ensemble']
    p_platt = res['p_te_platt']

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration', zorder=1)

    try:
        fr, mr = calibration_curve(y_te, p_raw, n_bins=10, strategy='quantile')
        ax.plot(mr, fr, 'o-', color='#e74c3c', lw=2, ms=6,
                label=f'Raw (ECE={res["ece_raw"]:.4f})', zorder=3)
    except Exception:
        pass

    p_uniq = np.unique(np.round(p_platt, 3))
    if len(p_uniq) <= 2:
        cm = p_platt.mean()
        ax.scatter([cm], [y_te.mean()], s=180, marker='s', color='#2ecc71', zorder=5,
                   label=f'Platt (ECE={res["ece_platt"]:.4f}, degenerate: p->{cm:.3f})')
        ax.annotate('Platt\n(constant)', xy=(cm, y_te.mean()),
                    xytext=(cm - 0.18, y_te.mean() - 0.08), fontsize=8,
                    color='#2ecc71', arrowprops=dict(arrowstyle='->', color='#2ecc71'))
    else:
        try:
            fp, mp = calibration_curve(y_te, p_platt, n_bins=10, strategy='quantile')
            ax.plot(mp, fp, 's-', color='#2ecc71', lw=2, ms=6,
                    label=f'Platt (ECE={res["ece_platt"]:.4f})', zorder=4)
        except Exception:
            pass

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('GRU: Calibration Curve', fontsize=13)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.hist(p_raw, bins=25, alpha=0.12, color='#e74c3c')
    ax2.set_ylabel('Count', fontsize=8, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


# ── Feature Ablation ───────────────────────────────────────────────────────────

def run_feature_ablation(data, seeds=SEEDS):
    """Train VRPGRU on each of 4 feature subsets, same config as full baseline.

    Returns a list of result dicts, one per feature group.
    Groups defined in FEATURE_GROUPS at top of file.
    """
    print('\n' + '='*70)
    print('=== Feature Group Ablation (VRPGRU only) ===')
    print('=== 4 variants x 5 seeds = 20 total training runs  ===')
    print('='*70)

    ablation_results = []

    for group_name, indices in FEATURE_GROUPS.items():
        n_dims = len(indices)
        print(f'\n--- [{group_name}] indices={indices} ---')

        # Slice features and verify z-scoring is preserved
        sub_data = subset_features(data, indices)

        # Build loaders for this subset
        loaders = make_loaders(sub_data, W, STRIDE, BATCH, VAL_FRAC)
        _, _, test_loader, _, _, test_ds = loaders

        # VRPGRU with input_dim matched to feature subset
        def model_fn(n=n_dims):
            return VRPGRU(input_dim=n, hidden1=8, hidden2=4,
                          head_hidden=8, dropout=0.2)

        print(f'  Model: VRPGRU(input_dim={n_dims}) = {count_params(model_fn()):,} params')

        per_seed_test_preds = []
        per_seed_aucs       = []
        per_seed_aps        = []

        for s in seeds:
            model, _, bv = train_one(model_fn, loaders, seed=s)
            p_te, y_te   = collect_preds(model, test_loader)
            per_seed_test_preds.append(p_te)
            m = eval_split(p_te, y_te, 'te')
            per_seed_aucs.append(m['AUC'])
            per_seed_aps.append(m['AP'])
            print(f'    seed {s}: val_loss={bv:.4f}  AUC={m["AUC"]:.3f}  AP={m["AP"]:.3f}')

        # Ensemble (mean predictions across seeds)
        p_ens = np.mean(per_seed_test_preds, axis=0)
        _, y_te_final = collect_preds(model, test_loader)
        ens_auc = roc_auc_score(y_te_final, p_ens)
        ens_ap  = average_precision_score(y_te_final, p_ens)
        ens_ece = compute_ece(p_ens, y_te_final, n_bins=10)

        auc_mean = float(np.mean(per_seed_aucs))
        auc_std  = float(np.std(per_seed_aucs, ddof=1))
        ap_mean  = float(np.mean(per_seed_aps))
        ap_std   = float(np.std(per_seed_aps, ddof=1))

        print(f'  ENSEMBLE: AUC={ens_auc:.3f}  AP={ens_ap:.3f}  ECE={ens_ece:.4f}')
        print(f'  Per-seed  AUC={auc_mean:.3f}+/-{auc_std:.3f}  '
              f'AP={ap_mean:.3f}+/-{ap_std:.3f}')

        ablation_results.append({
            'group':       group_name,
            'indices':     indices,
            'n_dims':      n_dims,
            'auc_mean':    auc_mean,
            'auc_std':     auc_std,
            'ap_mean':     ap_mean,
            'ap_std':      ap_std,
            'ens_auc':     ens_auc,
            'ens_ap':      ens_ap,
            'ens_ece':     ens_ece,
            'p_te_ens':    p_ens,
            'y_te':        y_te_final,
        })

    return ablation_results


def plot_feature_ablation(ablation_results,
                          save_path='vrp_feature_ablation.png'):
    """Bar chart: test AUC mean +/- std for each feature group.
    Horizontal lines at AUC=0.5 (random) and AUC=0.747 (base rate).
    Ensemble AUC shown as scatter marker on each bar.
    """
    groups      = [r['group'] for r in ablation_results]
    auc_means   = [r['auc_mean'] for r in ablation_results]
    auc_stds    = [r['auc_std']  for r in ablation_results]
    ens_aucs    = [r['ens_auc']  for r in ablation_results]

    # Short labels for x-axis
    short_labels = ['DMM\nlatents\n(A+B, 12d)',
                    'Emission\nresiduals\n(C, 4d)',
                    'Macro/\nmicrostr\n(D-G, 4d)',
                    'All\nfeatures\n(20d)']

    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Feature Group Ablation: VRPGRU Test AUC\n'
                 '(5-seed mean +/- std, test set 2015-2025)',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(groups))
    bars = ax.bar(x, auc_means, yerr=auc_stds, capsize=6, width=0.55,
                  color=colors, alpha=0.80, edgecolor='black', lw=0.8,
                  error_kw={'elinewidth': 2, 'ecolor': 'black', 'capthick': 2})

    # Ensemble AUC markers
    ax.scatter(x, ens_aucs, s=100, zorder=5, color='white',
               edgecolors='black', linewidths=2, label='Ensemble AUC')

    # Reference lines
    ax.axhline(0.5,   color='#c0392b', ls='--', lw=1.5, label='AUC=0.500 (random)')
    ax.axhline(0.747, color='#8e44ad', ls=':',  lw=1.5, label='AUC=0.747 (base rate acc)')

    # Value labels on bars
    for i, (mean, std, ens) in enumerate(zip(auc_means, auc_stds, ens_aucs)):
        ax.text(i, mean + std + 0.012, f'{mean:.3f}\n+/-{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i, ens - 0.015, f'ens={ens:.3f}',
                ha='center', va='top', fontsize=8, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel('Test AUC (ROC)', fontsize=12)
    ax.set_ylim(0.40, max(auc_means) + max(auc_stds) + 0.10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def print_ablation_table(ablation_results):
    """Print summary table matching handoff format."""
    print('\n' + '='*80)
    print('=== Feature Ablation Summary Table ===')
    print(f'  {"Group":<30}  {"Dims":>4}  {"AUC mean":>9}  {"AUC std":>8}  '
          f'{"AP mean":>8}  {"AP std":>7}  {"Ens AUC":>8}  {"Ens ECE":>8}')
    print('  ' + '-'*78)
    for r in ablation_results:
        print(f'  {r["group"]:<30}  {r["n_dims"]:>4}  '
              f'{r["auc_mean"]:>9.3f}  {r["auc_std"]:>8.3f}  '
              f'{r["ap_mean"]:>8.3f}  {r["ap_std"]:>7.3f}  '
              f'{r["ens_auc"]:>8.3f}  {r["ens_ece"]:>8.4f}')
    print()

    # DMM contribution summary
    dmm_ens   = next(r['ens_auc'] for r in ablation_results if 'DMM' in r['group'])
    macro_ens = next(r['ens_auc'] for r in ablation_results if 'Macro' in r['group'])
    all_ens   = next(r['ens_auc'] for r in ablation_results if 'All' in r['group'])
    dmm_mean  = next(r['auc_mean'] for r in ablation_results if 'DMM' in r['group'])
    macro_mean= next(r['auc_mean'] for r in ablation_results if 'Macro' in r['group'])
    all_mean  = next(r['auc_mean'] for r in ablation_results if 'All' in r['group'])

    print(f'  DMM latents (A+B) vs Macro/microstr: '
          f'ensemble AUC delta = {dmm_ens - macro_ens:+.3f}  '
          f'(mean delta = {dmm_mean - macro_mean:+.3f})')
    print(f'  Full model vs DMM-only: '
          f'ensemble AUC delta = {all_ens - dmm_ens:+.3f}  '
          f'(mean delta = {all_mean - dmm_mean:+.3f})')
    print()

    # One-sentence answer
    delta = dmm_ens - macro_ens
    delta_mean = dmm_mean - macro_mean
    if abs(delta) > 0.01:
        direction = "contribute beyond" if delta > 0 else "do NOT outperform"
        print(f'  ANSWER: {"Yes" if delta > 0 else "No"}, the DMM features '
              f'{direction} raw macro by {abs(delta):.3f} ensemble AUC points '
              f'({abs(delta_mean):.3f} per-seed mean).')
    else:
        print(f'  ANSWER: Marginal difference ({delta:+.3f} ens AUC); '
              f'DMM features add limited discriminatory power over macro alone.')


def plot_mt_comparison(baseline_res, mt_results_list, save_path='vrp_multitask_comparison.png'):
    labels = ['Baseline\n(VRP only)'] + [f"MT (lam={r['lambda_rv']})" for r in mt_results_list]
    
    auc_means = [baseline_res['per_seed_metrics']['te_AUC'].mean()] + [r['per_seed_metrics']['te_AUC'].mean() for r in mt_results_list]
    auc_stds  = [baseline_res['per_seed_metrics']['te_AUC'].std(ddof=1)] + [r['per_seed_metrics']['te_AUC'].std(ddof=1) for r in mt_results_list]
    ens_aucs  = [baseline_res['ensemble_metrics']['AUC']] + [r['ensemble_metrics']['AUC'] for r in mt_results_list]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('VRP Test AUC: Single-Task vs Multi-Task (RV head)', fontsize=14, fontweight='bold')
    
    x = np.arange(len(labels))
    colors = ['#3498db'] + ['#9b59b6']*len(mt_results_list)
    
    bars = ax.bar(x, auc_means, yerr=auc_stds, capsize=6, width=0.5,
                  color=colors, alpha=0.80, edgecolor='black', lw=0.8,
                  error_kw={'elinewidth': 2, 'ecolor': 'black', 'capthick': 2})
                  
    ax.scatter(x, ens_aucs, s=100, zorder=5, color='white',
               edgecolors='black', linewidths=2, label='Ensemble AUC')
               
    for i, (mean, std, ens) in enumerate(zip(auc_means, auc_stds, ens_aucs)):
        ax.text(i, mean + std + 0.012, f'{mean:.3f}\n+/-{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i, ens - 0.015, f'ens={ens:.3f}',
                ha='center', va='top', fontsize=8, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Test AUC (ROC)', fontsize=12)
    ax.set_ylim(0.40, max(auc_means) + max(auc_stds) + 0.10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_rv_scatter(actual, pred, rmse, r2, title_prefix, save_path='vrp_rv_scatter.png'):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, pred, alpha=0.3, color='#e74c3c', s=15)
    
    # Diagonal line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Perfect prediction')
    
    ax.set_xlabel('Actual Log-Vol [log(RS Vol_{t+1})]', fontsize=11)
    ax.set_ylabel('Predicted Log-Vol', fontsize=11)
    ax.set_title(f'{title_prefix}: RV Prediction\nRMSE={rmse:.4f}  R²={r2:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_mt_loss_curves(mt_results_list, save_path='vrp_multitask_loss_curves.png'):
    fig, axes = plt.subplots(1, len(mt_results_list), figsize=(15, 4), sharey=False)
    if len(mt_results_list) == 1: axes = [axes]
    fig.suptitle('Multi-Task Training Loss Components (Seed 0)', fontsize=14, fontweight='bold')
    
    for ax, res in zip(axes, mt_results_list):
        hist = res['histories'][0] # Seed 0
        epochs = range(len(hist['train_tot']))
        
        ax.plot(epochs, hist['train_vrp'], color='#3498db', ls='-', label='Train VRP')
        ax.plot(epochs, hist['val_vrp'],   color='#3498db', ls='--', label='Val VRP')
        
        ax.plot(epochs, hist['train_rv'], color='#e74c3c', ls='-', label='Train RV')
        ax.plot(epochs, hist['val_rv'],   color='#e74c3c', ls='--', label='Val RV')
        
        ax.set_title(f'lam_rv = {res["lambda_rv"]}', fontsize=12)
        ax.set_xlabel('Epoch')
        if ax == axes[0]: ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...')
    data = load_vrp_data('.')

    loaders = make_loaders(data, W, STRIDE, BATCH, VAL_FRAC)
    train_ds, val_ds, test_ds = loaders[3], loaders[4], loaders[5]
    print(f'  train windows: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}')

    # Data augmentation for MT
    data = augment_data_mt(data)
    mt_loaders = make_mt_loaders(data, W, STRIDE, BATCH, VAL_FRAC)
    mt_train_ds, mt_val_ds, mt_test_ds = mt_loaders[3], mt_loaders[4], mt_loaders[5]
    print(f'  MT train windows: {len(mt_train_ds)}  val: {len(mt_val_ds)}  test: {len(mt_test_ds)}')

    # ── Baseline: full 20-dim GRU ────────────────────────────────────────────
    results = {}
    results['GRU'] = run_arch('VRPGRU', lambda: VRPGRU(input_dim=20), loaders)
    
    print('\n=== Multi-Task Sweep: lam_rv in [0.1, 1.0, 10.0] ===')
    from vrp_signals import VRPGRU_MT
    
    mt_results = []
    for lam in [0.1, 1.0, 10.0]:
        res = run_mt_arch('VRPGRU', lambda: VRPGRU_MT(input_dim=20), mt_loaders, lambda_rv=lam, seeds=SEEDS)
        mt_results.append(res)
        
    print('\n=== Persistence Baseline for RV Head ===')
    y_rv_te_actual = np.log(np.clip(data['rs_vol_te'][results['GRU']['test_anchors'] + 1], 1e-8, None))
    y_rv_te_persist = np.log(np.clip(data['rs_vol_te'][results['GRU']['test_anchors']], 1e-8, None))
    
    persist_rmse = np.sqrt(np.mean((y_rv_te_persist - y_rv_te_actual)**2))
    persist_r2   = 1 - np.sum((y_rv_te_actual - y_rv_te_persist)**2) / np.sum((y_rv_te_actual - y_rv_te_actual.mean())**2)
    print(f'  Persistence RMSE = {persist_rmse:.4f}  R² = {persist_r2:.4f}')
    
    print('\n=== RV Head Inverse Transform Performance ===')
    for res in mt_results:
        # Inverse transform the standardized predictions back to log-vol space
        pred_log_vol = res['mu_te_ensemble'] * data['sig_rv'] + data['mu_rv']
        
        rmse = np.sqrt(np.mean((pred_log_vol - y_rv_te_actual)**2))
        r2   = 1 - np.sum((y_rv_te_actual - pred_log_vol)**2) / np.sum((y_rv_te_actual - y_rv_te_actual.mean())**2)
        res['real_rmse'] = rmse
        res['real_r2'] = r2
        print(f"  MT (lam={res['lambda_rv']}): RMSE = {rmse:.4f}  R2 = {r2:.4f}")

    print('\n=== Generating MT Plots ===')
    plot_mt_comparison(results['GRU'], mt_results, save_path='vrp_multitask_comparison.png')
    plot_mt_loss_curves(mt_results, save_path='vrp_multitask_loss_curves.png')
    
    # Scatter plot for best lambda (lowest RMSE)
    best_res = min(mt_results, key=lambda x: x['real_rmse'])
    best_pred_log_vol = best_res['mu_te_ensemble'] * data['sig_rv'] + data['mu_rv']
    plot_rv_scatter(y_rv_te_actual, best_pred_log_vol, best_res['real_rmse'], best_res['real_r2'],
                    title_prefix=f"MT (lam={best_res['lambda_rv']})", save_path='vrp_rv_scatter.png')

    # ── Calibration: Platt scaling + ECE (documented negative result) ─────────
    for arch in ['GRU']:
        res   = results[arch]
        p_val = res['p_va_ensemble']
        y_val = res['y_va']
        y_te  = res['y_te']
        p_raw = res['p_te_ensemble']

        platt_scaler       = fit_platt(p_val, y_val)
        res['p_te_platt']  = calibrate(p_raw, platt_scaler)
        res['ece_raw']     = compute_ece(p_raw,             y_te, n_bins=10)
        res['ece_platt']   = compute_ece(res['p_te_platt'], y_te, n_bins=10)
        res['platt_coef']  = platt_scaler.coef_[0][0]

    print('\n=== ECE Summary (10 quantile bins, test set) ===')
    print(f'  {"":6s}  {"Before":>10}  {"After Platt":>12}')
    for arch in ['GRU']:
        print(f'  {arch:<6}  {results[arch]["ece_raw"]:>10.4f}  '
              f'{results[arch]["ece_platt"]:>12.4f}')

    plot_calibration_before_after(results['GRU'])

    # ── Feature group ablation ────────────────────────────────────────────────
    ablation_results = run_feature_ablation(data, seeds=SEEDS)
    print_ablation_table(ablation_results)

    print('\n=== Generating ablation plot ===')
    plot_feature_ablation(ablation_results, save_path='vrp_feature_ablation.png')

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez('vrp_pipeline_results.npz',
             gru_p_te       = results['GRU']['p_te_ensemble'],
             gru_p_te_platt = results['GRU']['p_te_platt'],
             y_te           = results['GRU']['y_te'],
             test_anchors   = results['GRU']['test_anchors'])
    print('Saved vrp_pipeline_results.npz')

    return results, ablation_results, data


if __name__ == '__main__':
    main()
