"""
vrp_signals.py
VRP-only signal pipeline (the mean-reversion head was dropped as it did not work).

Two trunk variants share a common interface:
  - VRPGRU : 2-layer GRU
  - VRPCNN : 2-layer dilated 1D-CNN (for the architecture comparison)

Both consume (B, W, input_dim) windows and emit p_vrp in (0, 1) per window
using the same VRP head, so they are drop-in interchangeable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


# ── Dataset ───────────────────────────────────────────────────────────────────

class VRPDataset(Dataset):
    """Sliding-window dataset, stride configurable. Anchor day = last day of window.
    Drops windows whose anchor target is NaN (tail of series)."""

    def __init__(self, feat, y_vrp, W=30, stride=10):
        T = len(feat)
        starts = [i for i in range(0, T - W + 1, stride)
                  if not np.isnan(y_vrp[i + W - 1])]
        self.starts = starts
        self.feat   = torch.tensor(feat,  dtype=torch.float32)
        self.y_vrp  = torch.tensor(y_vrp, dtype=torch.float32)
        self.W      = W

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        t = s + self.W - 1
        return self.feat[s : s + self.W], self.y_vrp[t]

    @property
    def anchor_indices(self):
        return np.array([s + self.W - 1 for s in self.starts])


class VRPMTDataset(Dataset):
    """Sliding-window dataset for Multi-Task learning. 
    Drops windows whose anchor target (vrp or rv) is NaN."""

    def __init__(self, feat, y_vrp, y_rv, W=30, stride=10):
        T = len(feat)
        starts = [i for i in range(0, T - W + 1, stride)
                  if not np.isnan(y_vrp[i + W - 1]) and not np.isnan(y_rv[i + W - 1])]
        self.starts = starts
        self.feat   = torch.tensor(feat,  dtype=torch.float32)
        self.y_vrp  = torch.tensor(y_vrp, dtype=torch.float32)
        self.y_rv   = torch.tensor(y_rv,  dtype=torch.float32)
        self.W      = W

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        t = s + self.W - 1
        return self.feat[s : s + self.W], self.y_vrp[t], self.y_rv[t]

    @property
    def anchor_indices(self):
        return np.array([s + self.W - 1 for s in self.starts])


# ── Models ────────────────────────────────────────────────────────────────────

class _VRPHead(nn.Module):
    """Shared classification head: trunk_dim -> p_vrp in (0, 1)."""
    def __init__(self, trunk_dim, head_hidden=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class VRPGRU(nn.Module):
    """2-layer GRU trunk. Hidden sizes kept <=8 to avoid overfitting the small window count."""
    def __init__(self, input_dim=20, hidden1=8, hidden2=4, head_hidden=8, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1,   hidden2, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = _VRPHead(hidden2, head_hidden, dropout)

    def forward(self, x):
        h, _ = self.gru1(x)
        _, h = self.gru2(h)
        z = self.drop(h.squeeze(0))
        return self.head(z)


class VRPCNN(nn.Module):
    """2-layer dilated 1D-CNN trunk + global avg pool. Same capacity envelope as VRPGRU."""
    def __init__(self, input_dim=20, channels1=8, channels2=8,
                 kernel=3, head_hidden=8, dropout=0.2):
        super().__init__()
        pad1 = (kernel - 1) // 2 * 1   # dilation 1
        pad2 = (kernel - 1) // 2 * 2   # dilation 2
        self.conv1 = nn.Conv1d(input_dim, channels1, kernel,
                               padding=pad1, dilation=1)
        self.conv2 = nn.Conv1d(channels1, channels2, kernel,
                               padding=pad2, dilation=2)
        self.drop = nn.Dropout(dropout)
        self.head = _VRPHead(channels2, head_hidden, dropout)

    def forward(self, x):
        # x: (B, W, C) -> (B, C, W) for conv1d
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        z = h.mean(dim=-1)          # global average pool
        z = self.drop(z)
        return self.head(z)


class VRPGRU_MT(nn.Module):
    """2-layer GRU trunk with Multi-Task heads (VRP + RV)."""
    def __init__(self, input_dim=20, hidden1=8, hidden2=4, head_hidden=8, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1,   hidden2, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.vrp_head = _VRPHead(hidden2, head_hidden, dropout)
        self.rv_head = nn.Sequential(
            nn.Linear(hidden2, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        h, _ = self.gru1(x)
        _, h = self.gru2(h)
        z = self.drop(h.squeeze(0))
        return self.vrp_head(z), self.rv_head(z).squeeze(-1)


class VRPCNN_MT(nn.Module):
    """2-layer dilated 1D-CNN trunk with Multi-Task heads (VRP + RV)."""
    def __init__(self, input_dim=20, channels1=8, channels2=8,
                 kernel=3, head_hidden=8, dropout=0.2):
        super().__init__()
        pad1 = (kernel - 1) // 2 * 1
        pad2 = (kernel - 1) // 2 * 2
        self.conv1 = nn.Conv1d(input_dim, channels1, kernel,
                               padding=pad1, dilation=1)
        self.conv2 = nn.Conv1d(channels1, channels2, kernel,
                               padding=pad2, dilation=2)
        self.drop = nn.Dropout(dropout)
        self.vrp_head = _VRPHead(channels2, head_hidden, dropout)
        self.rv_head = nn.Sequential(
            nn.Linear(channels2, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        z = h.mean(dim=-1)
        z = self.drop(z)
        return self.vrp_head(z), self.rv_head(z).squeeze(-1)


# ── VRP target builder (unchanged from gru_signals.py) ────────────────────────

def build_vrp_targets(rs_minus_daily, horizon=10, median_train=None, jump_threshold=None):
    """y[t] = 1 if max(RS_minus[t+1 : t+1+horizon]) - median_train <= jump_threshold
            else 0. Last `horizon` rows are NaN. Constants must be fit on train only."""
    T = len(rs_minus_daily)
    n_valid = T - horizon
    if n_valid <= 0:
        return np.full(T, np.nan, dtype=np.float32)

    rs = rs_minus_daily.astype(np.float64)
    future = sliding_window_view(rs, horizon)[1 : n_valid + 1]
    jump_proxy = future.max(axis=1) - float(median_train)
    y = (jump_proxy <= float(jump_threshold)).astype(np.float32)
    return np.concatenate([y, np.full(horizon, np.nan, dtype=np.float32)])


# ── Loss ──────────────────────────────────────────────────────────────────────

def vrp_loss(p_vrp, y_vrp, alpha=5.0):
    """Asymmetric BCE: y=0 (spike incoming, dangerous to sell vol) up-weighted by alpha."""
    w = torch.where(y_vrp == 0,
                    p_vrp.new_full(p_vrp.shape, alpha),
                    torch.ones_like(p_vrp))
    return F.binary_cross_entropy(p_vrp, y_vrp, weight=w)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
