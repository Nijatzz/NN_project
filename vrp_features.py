"""
vrp_features.py
End-to-end feature + target construction for the VRP signal model.

Public API:
    load_vrp_data(data_dir='.') -> dict with:
        feat_tr, feat_te        : (T, 20) normalized feature matrices
        y_vrp_tr, y_vrp_te      : (T,) VRP binary targets (NaN tail = horizon)
        log_ret_tr, log_ret_te  : (T,) raw log returns
        rs_vol_tr,  rs_vol_te   : (T,) raw daily Rogers-Satchell vol (decimal)
        vix_pct_tr, vix_pct_te  : (T,) raw VIX percent
        dates_tr, dates_te      : pd.DatetimeIndex
        mu_feat, sig_feat       : (20,) train normalization constants
        vrp_threshold_constants : (median, jump_thresh) fit on train
"""
import numpy as np
import pandas as pd
from utils import (
    log_returns, rogers_satchell_rv, realized_semivariance, volume_ratio,
)
from vrp_signals import build_vrp_targets


def _emission_residuals(ret_r, rv_log_r, log_vix_r, rsv_r,
                        ep_st_mu, ep_ln_mu, mu_f_st, sig_f_st, mu_f_ln):
    """4 residuals = actual_normalized - DMM emission mu, for ret/RV/VIX/RSV."""
    r0 = (ret_r     - mu_f_st[0]) / sig_f_st[0] - ep_st_mu[:, 0]
    r1 = (rv_log_r  - mu_f_st[1]) / sig_f_st[1] - ep_st_mu[:, 1]
    r2 = (log_vix_r - mu_f_st[2]) / sig_f_st[2] - ep_st_mu[:, 2]
    r3 = np.log(np.maximum(rsv_r / mu_f_ln, 1e-12)) - ep_ln_mu[:, 0]
    return np.column_stack([r0, r1, r2, r3]).astype(np.float32)


def _assemble(f_means, f_stds, resid, baa, corr, vol_rat, pz):
    return np.column_stack([
        f_means,
        np.log(np.maximum(f_stds, 1e-8)),
        resid,
        baa.reshape(-1, 1),
        corr.reshape(-1, 1),
        np.log(np.maximum(vol_rat, 1e-8)).reshape(-1, 1),
        pz.reshape(-1, 1),
    ]).astype(np.float32)


def load_vrp_data(data_dir='.', vrp_horizon=10):
    d = np.load(f'{data_dir}/dmm_artifacts.npz', allow_pickle=True)

    f_tr,  f_te  = d['f_tr'],  d['f_te']
    fs_tr, fs_te = d['f_std_tr'], d['f_std_te']
    ep_st_mu_tr, ep_st_mu_te = d['fast_ep_tr_st_mu'], d['fast_ep_te_st_mu']
    ep_ln_mu_tr, ep_ln_mu_te = d['fast_ep_tr_ln_mu'], d['fast_ep_te_ln_mu']
    log_ret_tr = d['log_ret_tr'].astype(np.float64)
    log_ret_te = d['log_ret_te'].astype(np.float64)
    dates_tr = pd.to_datetime(d['fast_train_dates'])
    dates_te = pd.to_datetime(d['fast_test_dates'])
    T_tr, T_te = len(f_tr), len(f_te)
    all_dates = pd.DatetimeIndex(np.concatenate([dates_tr.values, dates_te.values]))

    spy    = pd.read_csv(f'{data_dir}/spy_ohlcv.csv', index_col=0, parse_dates=True,
                         skiprows=[1, 2])
    vix    = pd.read_csv(f'{data_dir}/fred_vix.csv',    index_col=0, parse_dates=True).squeeze()
    dgs10  = pd.read_csv(f'{data_dir}/fred_dgs10.csv',  index_col=0, parse_dates=True).squeeze()
    baa10y = pd.read_csv(f'{data_dir}/fred_baa10y.csv', index_col=0, parse_dates=True).squeeze()

    # Daily series — full history, then aligned to DMM date index
    spy = spy.astype(float)
    ret_full     = log_returns(spy['Close'])
    rs_vol_full  = rogers_satchell_rv(spy)
    rv_log_full  = np.log(rs_vol_full.clip(lower=1e-8))
    log_vix_full = np.log(vix.ffill().clip(lower=1e-8))
    rsv_full     = realized_semivariance(ret_full, window=5).clip(lower=1e-8)
    gap_full     = np.log(spy['Open'] / spy['Close'].shift(1))
    log_vol_full = np.log(spy['Volume'].clip(lower=1))
    rel_vol_full = log_vol_full - log_vol_full.ewm(span=63, adjust=False).mean()

    # Price z-score: rolling std of (close - MA50) over 50d
    close = spy['Close']
    ma50  = close.rolling(50).mean()
    dev   = close - ma50
    std50 = dev.rolling(50).std().clip(lower=1e-6)
    pz_full = dev / std50

    # Equity-bond rolling 20d correlation
    dgs10_al    = dgs10.ffill().reindex(all_dates, method='ffill')
    delta_y     = dgs10_al.diff()
    ret_all_ser = pd.Series(np.concatenate([log_ret_tr, log_ret_te]), index=all_dates)
    roll_corr   = ret_all_ser.rolling(20).corr(-delta_y).fillna(0.0)

    def align(s):
        return s.reindex(all_dates, method='ffill').fillna(0.0).values.astype(np.float32)

    ret_a     = align(ret_full)
    rv_log_a  = align(rv_log_full)
    log_vix_a = align(log_vix_full)
    rsv_a     = align(rsv_full)
    rel_vol_a = align(rel_vol_full)        # unused in features but kept available
    pz_a      = align(pz_full)
    vol_rat_a = align(volume_ratio(spy['Volume'], window=20).clip(lower=1e-3))
    baa_a     = align(baa10y.ffill())
    rs_vol_a  = align(rs_vol_full)
    vix_pct_a = align(vix.ffill())
    corr_a    = roll_corr.values.astype(np.float32)

    # split
    def split(arr): return arr[:T_tr], arr[T_tr:]
    ret_tr,     ret_te     = split(ret_a)
    rv_log_tr,  rv_log_te  = split(rv_log_a)
    log_vix_tr, log_vix_te = split(log_vix_a)
    rsv_tr,     rsv_te     = split(rsv_a)
    pz_tr,      pz_te      = split(pz_a)
    vol_rat_tr, vol_rat_te = split(vol_rat_a)
    baa_tr,     baa_te     = split(baa_a)
    corr_tr,    corr_te    = split(corr_a)
    rs_vol_tr,  rs_vol_te  = split(rs_vol_a)
    vix_pct_tr, vix_pct_te = split(vix_pct_a)

    # DMM normalization constants (fit on train, needed for residuals)
    fast_raw_tr = np.column_stack([ret_tr, rv_log_tr, log_vix_tr, np.zeros_like(ret_tr), rsv_tr])
    # gap not actually used in residuals so we skip its mu/sig; only need [ret, rv, vix, rsv]
    mu_f_st  = np.array([ret_tr.mean(), rv_log_tr.mean(), log_vix_tr.mean(), 0.0])
    sig_f_st = np.array([ret_tr.std(),  rv_log_tr.std(),  log_vix_tr.std(),  1.0])
    mu_f_ln  = rsv_tr.mean()

    resid_tr = _emission_residuals(ret_tr, rv_log_tr, log_vix_tr, rsv_tr,
                                   ep_st_mu_tr, ep_ln_mu_tr,
                                   mu_f_st, sig_f_st, mu_f_ln)
    resid_te = _emission_residuals(ret_te, rv_log_te, log_vix_te, rsv_te,
                                   ep_st_mu_te, ep_ln_mu_te,
                                   mu_f_st, sig_f_st, mu_f_ln)

    feat_tr_raw = _assemble(f_tr, fs_tr, resid_tr, baa_tr, corr_tr, vol_rat_tr, pz_tr)
    feat_te_raw = _assemble(f_te, fs_te, resid_te, baa_te, corr_te, vol_rat_te, pz_te)
    assert feat_tr_raw.shape == (T_tr, 20)
    assert feat_te_raw.shape == (T_te, 20)

    # z-score normalization, train statistics only
    mu_feat  = feat_tr_raw.mean(0)
    sig_feat = np.maximum(feat_tr_raw.std(0), 1e-8)
    feat_tr  = (feat_tr_raw - mu_feat) / sig_feat
    feat_te  = (feat_te_raw - mu_feat) / sig_feat

    # VRP targets, threshold fit on train only
    rs_minus_tr = np.minimum(log_ret_tr, 0.0) ** 2
    rs_minus_te = np.minimum(log_ret_te, 0.0) ** 2
    rs_med_tr   = float(np.median(rs_minus_tr))
    jump_tr     = float(np.percentile(rs_minus_tr, 95))
    y_vrp_tr = build_vrp_targets(rs_minus_tr, vrp_horizon, rs_med_tr, jump_tr)
    y_vrp_te = build_vrp_targets(rs_minus_te, vrp_horizon, rs_med_tr, jump_tr)

    return {
        'feat_tr': feat_tr, 'feat_te': feat_te,
        'y_vrp_tr': y_vrp_tr, 'y_vrp_te': y_vrp_te,
        'log_ret_tr': log_ret_tr, 'log_ret_te': log_ret_te,
        'rs_vol_tr': rs_vol_tr, 'rs_vol_te': rs_vol_te,
        'vix_pct_tr': vix_pct_tr, 'vix_pct_te': vix_pct_te,
        'dates_tr': dates_tr, 'dates_te': dates_te,
        'mu_feat': mu_feat, 'sig_feat': sig_feat,
        'vrp_threshold_constants': (rs_med_tr, jump_tr),
    }
