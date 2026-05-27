"""
gru_signals.py
Two-layer GRU for VRP (Variance Risk Premium) and Mean-Reversion signal detection.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


# ── Model ─────────────────────────────────────────────────────────────────────

class VRPMRNet(nn.Module):
    """
    Shared two-layer GRU trunk with separate VRP and MR binary output heads.

    Input : (batch, W, input_dim)
    Output: (p_vrp, p_mr) each shape (batch,), values in (0, 1)
    """
    def __init__(self, input_dim=20, hidden1=64, hidden2=32, head_hidden=16):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1,   hidden2, batch_first=True)
        self.vrp_head = nn.Sequential(
            nn.Linear(hidden2, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),       nn.Sigmoid(),
        )
        self.mr_head = nn.Sequential(
            nn.Linear(hidden2, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),       nn.Sigmoid(),
        )

    def forward(self, x):
        h, _ = self.gru1(x)    # (B, W, hidden1)
        _, h = self.gru2(h)    # (1, B, hidden2)
        z = h.squeeze(0)       # (B, hidden2)
        return self.vrp_head(z).squeeze(-1), self.mr_head(z).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VRPMRDataset(Dataset):
    """
    Sliding-window dataset (stride = 1).
    Anchor day for all targets is the last day of each window (index t = start + W - 1).
    Windows where any target is NaN are excluded automatically.
    """
    def __init__(self, feat, y_vrp, y_mr, mr_mask, mr_weight, W=30, stride=1):
        T = len(feat)
        starts = [i for i in range(0, T - W + 1, stride)
                  if not (np.isnan(y_vrp[i + W - 1]) or
                          np.isnan(y_mr[i + W - 1]) or
                          np.isnan(mr_weight[i + W - 1]))]
        self.starts    = starts
        self.feat      = torch.tensor(feat,      dtype=torch.float32)
        self.y_vrp     = torch.tensor(y_vrp,     dtype=torch.float32)
        self.y_mr      = torch.tensor(y_mr,      dtype=torch.float32)
        self.mr_mask   = torch.tensor(mr_mask,   dtype=torch.float32)
        self.mr_weight = torch.tensor(mr_weight, dtype=torch.float32)
        self.W         = W

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        t = s + self.W - 1
        return (self.feat[s : s + self.W],
                self.y_vrp[t],
                self.y_mr[t],
                self.mr_mask[t],
                self.mr_weight[t])

    @property
    def anchor_indices(self):
        return [s + self.W - 1 for s in self.starts]


# ── Target builders ───────────────────────────────────────────────────────────

def build_vrp_targets(rs_minus_daily, horizon=10, median_train=None, jump_threshold=None):
    """
    rs_minus_daily  : (T,) per-day downside semivariance = min(log_ret[t], 0)**2
    median_train    : median of rs_minus_daily over the training set (fit once)
    jump_threshold  : np.percentile(rs_minus_train, 95) — fit on training only

    daily_jump_proxy[t] = max(rs_minus[t+1 : t+1+horizon]) - median_train
    y_vrp[t] = 0  if  daily_jump_proxy[t] > jump_threshold  (jump-active; don't sell vol)
               1  otherwise
    Last `horizon` entries are NaN (no future window available).
    """
    T       = len(rs_minus_daily)
    n_valid = T - horizon
    if n_valid <= 0:
        return np.full(T, np.nan, dtype=np.float32)

    rs     = rs_minus_daily.astype(np.float64)
    future = sliding_window_view(rs, horizon)[1 : n_valid + 1]   # (n_valid, horizon)
    jump_proxy = future.max(axis=1) - float(median_train)

    y = (jump_proxy <= float(jump_threshold)).astype(np.float32)
    return np.concatenate([y, np.full(horizon, np.nan, dtype=np.float32)])


def build_mr_targets(pzscore, log_ret, horizon=5, zscore_thresh=2.0):
    """
    pzscore    : (T,) float32 – (price - MA50) / ATR20
    log_ret    : (T,) float32 – daily log returns

    trigger[t] = |pzscore[t]| > zscore_thresh
    y_mr[t]    = 1  if trigger[t] AND sum(log_ret[t+1:t+1+horizon]) has opposite sign to pzscore[t]
                 0  if trigger[t] AND no reversion
                 0  if not trigger[t]   (zeroed by mr_mask in the loss)
    mr_mask    = trigger (float32; non-trigger days contribute 0 to loss)
    mr_weight  = |sum(log_ret[t+1:t+1+horizon])|  (magnitude of realised move)

    Last `horizon` entries are NaN.
    """
    T       = len(log_ret)
    trigger = np.abs(pzscore) > zscore_thresh

    n_valid = T - horizon
    if n_valid <= 0:
        nan = np.full(T, np.nan, dtype=np.float32)
        return nan, nan, nan

    future_windows = sliding_window_view(log_ret.astype(np.float64), horizon)[1 : n_valid + 1]
    future_5d      = future_windows.sum(axis=1)                # (n_valid,)
    sign_pz        = np.sign(pzscore[:n_valid]).astype(np.float64)
    reverts        = (sign_pz * future_5d < 0)                 # opposite sign → reversion

    trig_sl  = trigger[:n_valid]
    y_mr     = np.where(trig_sl, reverts.astype(np.float32), 0.0).astype(np.float32)
    mr_mask  = trig_sl.astype(np.float32)
    mr_weight = np.abs(future_5d).astype(np.float32)

    nan_tail = np.full(horizon, np.nan, dtype=np.float32)
    return (np.concatenate([y_mr,       nan_tail]),
            np.concatenate([mr_mask,    nan_tail]),
            np.concatenate([mr_weight,  nan_tail]))


# ── Loss functions ────────────────────────────────────────────────────────────

def vrp_loss(p_vrp, y_vrp, alpha=5.0):
    """
    Weighted BCE for VRP head.
    y=0 (spike incoming — dangerous to sell vol) gets weight alpha.
    y=1 (VRP realised — selling vol was profitable) gets weight 1.
    """
    w = torch.where(y_vrp == 0,
                    p_vrp.new_full(p_vrp.shape, alpha),
                    torch.ones_like(p_vrp))
    return F.binary_cross_entropy(p_vrp, y_vrp, weight=w)


def mr_loss(p_mr, y_mr, mask, weight):
    """
    Magnitude-weighted, trigger-masked BCE for MR head.
    Non-trigger days (mask=0) contribute zero to the loss.
    """
    bce = F.binary_cross_entropy(p_mr, y_mr, reduction='none')
    return (bce * weight * mask).sum() / (mask.sum() + 1e-8)
