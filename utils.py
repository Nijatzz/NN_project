import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute log returns: ln(P_t / P_{t-1}). First row is NaN."""
    return np.log(prices / prices.shift(1))


def rogers_satchell_rv(ohlcv: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell realized volatility (per-bar, not annualized).

    RS = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
    Returns the square root (volatility), one value per row.
    Requires columns: Open, High, Low, Close.
    """
    h, l, o, c = ohlcv['High'], ohlcv['Low'], ohlcv['Open'], ohlcv['Close']
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(rs.clip(lower=0))


def rolling_realized_vol(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """Rolling average of Rogers-Satchell realized volatility over a lookback window."""
    rs = rogers_satchell_rv(ohlcv)
    return rs.rolling(window).mean()


def plot_hmm_probs(
    model,
    X: torch.Tensor,
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    state_labels: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    prob_type: str = 'filtered',
) -> None:
    """Plot HMM state probabilities alongside price.

    prob_type: 'filtered' uses the forward pass; 'smoothed' uses forward-backward.
    """
    if prob_type == 'filtered':
        f = model.forward(X)
        log_probs = f - torch.logsumexp(f, dim=-1, keepdim=True)
        probs = torch.exp(log_probs).squeeze(0)
    else:
        probs = model.predict_proba(X).squeeze(0)

    probs = probs.detach()

    mask = pd.Series(True, index=dates)
    if start:
        mask &= dates >= start
    if end:
        mask &= dates <= end
    mask = mask.values

    order       = np.argsort(dates[mask])
    dates_plot  = dates[mask][order]
    probs_plot  = probs[mask][order].numpy()
    prices_plot = prices.reindex(dates_plot, method='ffill')

    n_states = probs_plot.shape[1]
    labels = state_labels if state_labels else [f'state {i}' for i in range(n_states)]

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(dates_plot, probs_plot, label=labels)
    ax1.set_ylabel('State probability')

    ax2 = ax1.twinx()
    ax2.plot(dates_plot, prices_plot.values, color='black', alpha=0.4, linewidth=0.8, label='price')
    ax2.set_ylabel('Price')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    title = f'HMM state probabilities ({prob_type})'
    if start or end:
        title += f'  [{start or ""}–{end or ""}]'
    plt.title(title)
    plt.tight_layout()
    plt.show()


def hmm_summary(model) -> None:
    print("Transition matrix:")
    print(model.edges.exp())

    for i, dist in enumerate(model.distributions):
        means = dist.means.detach().numpy().flatten()
        covs  = dist.covs.detach().numpy()
        stds  = np.sqrt(np.diag(covs) if covs.ndim == 2 else covs.flatten())
        print(f"\nState {i}:")
        for j, (m, s) in enumerate(zip(means, stds)):
            cv = s / m if m != 0 else float('nan')
            print(f"  feature {j}: mean={m:.6f}  std={s:.6f}  CV={cv:.4f}")

    trans = model.edges.exp().detach().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(trans.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi  = np.real(eigenvectors[:, idx])
    pi  = pi / pi.sum()
    print(f"\nStationary probabilities: {pi}")


def yoy_log_change(s: pd.Series, periods: int = 52) -> pd.Series:
    """Year-over-year log change: log(s_t / s_{t-periods}). First `periods` rows are NaN."""
    return np.log(s / s.shift(periods))


def realized_semivariance(log_ret: pd.Series, window: int = 5) -> pd.Series:
    """Rolling downside realized semivariance: mean squared negative log returns over `window` days.

    Only negative returns contribute; positive days count as zero variance.
    Result is non-negative — suitable for LogNormal emission after clipping.
    """
    neg_sq = log_ret.clip(upper=0) ** 2
    return neg_sq.rolling(window, min_periods=1).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (EWM with alpha=1/window)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def drawdown_from_rolling_high(prices: pd.Series, window: int) -> pd.Series:
    """Log drawdown from the rolling high over the past `window` days.

    Returns log(price / rolling_max), always <= 0.
    """
    rolling_max = prices.rolling(window, min_periods=1).max()
    return np.log(prices / rolling_max)


def recovery_from_rolling_low(prices: pd.Series, window: int) -> pd.Series:
    """Log recovery from the rolling low over the past `window` days.

    Returns log(price / rolling_min), always >= 0.
    """
    rolling_min = prices.rolling(window, min_periods=1).min()
    return np.log(prices / rolling_min)


def tail_day_count(log_ret: pd.Series, window: int, z: float = 2.567) -> pd.Series:
    """Count of days in the rolling window where |return| > z * rolling std.

    z=2.567 corresponds to the 1% two-tailed threshold under normality.
    Rolling std is computed over the same window, excluding the current day
    (shift(1)) to avoid using today's return in its own threshold.
    """
    rolling_std = log_ret.shift(1).rolling(window, min_periods=2).std()
    is_tail = (log_ret.abs() > z * rolling_std).astype(float)
    return is_tail.rolling(window, min_periods=1).sum()


def consecutive_up_days(log_ret: pd.Series, window: int) -> pd.Series:
    """Count of consecutive positive-return days ending at each date,
    capped at `window`.
    """
    is_up = (log_ret > 0).astype(int).values
    result = np.zeros(len(is_up), dtype=np.float32)
    streak = 0
    for i in range(len(is_up)):
        if is_up[i] == 1:
            streak = min(streak + 1, window)
        else:
            streak = 0
        result[i] = streak
    return pd.Series(result, index=log_ret.index)


def consecutive_down_days(log_ret: pd.Series, window: int) -> pd.Series:
    """Count of consecutive negative-return days ending at each date,
    capped at `window`.
    """
    is_down = (log_ret < 0).astype(int).values
    result = np.zeros(len(is_down), dtype=np.float32)
    streak = 0
    for i in range(len(is_down)):
        if is_down[i] == 1:
            streak = min(streak + 1, window)
        else:
            streak = 0
        result[i] = streak
    return pd.Series(result, index=log_ret.index)


def candle_range(ohlcv: pd.DataFrame) -> pd.Series:
    """Absolute high-low range per bar: H - L."""
    return ohlcv['High'] - ohlcv['Low']


def upper_shadow(ohlcv: pd.DataFrame) -> pd.Series:
    """Upper wick as a fraction of the H-L range: (H - max(O,C)) / (H-L).
    Returns 0 on doji bars where H == L.
    """
    hl = (ohlcv['High'] - ohlcv['Low']).replace(0, np.nan)
    return ((ohlcv['High'] - ohlcv[['Open', 'Close']].max(axis=1)) / hl).fillna(0.0)


def lower_shadow(ohlcv: pd.DataFrame) -> pd.Series:
    """Lower wick as a fraction of the H-L range: (min(O,C) - L) / (H-L).
    Returns 0 on doji bars where H == L.
    """
    hl = (ohlcv['High'] - ohlcv['Low']).replace(0, np.nan)
    return ((ohlcv[['Open', 'Close']].min(axis=1) - ohlcv['Low']) / hl).fillna(0.0)


def overnight_gap(ohlcv: pd.DataFrame) -> pd.Series:
    """Simple return from previous close to current open: (O - C_prev) / C_prev."""
    return (ohlcv['Open'] - ohlcv['Close'].shift(1)) / ohlcv['Close'].shift(1)


def intraday_return(ohlcv: pd.DataFrame) -> pd.Series:
    """Simple return from open to close within the bar: (C - O) / O."""
    return (ohlcv['Close'] - ohlcv['Open']) / ohlcv['Open']


def volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """Current volume as a multiple of its rolling mean: V / mean(V, window)."""
    return volume / volume.rolling(window).mean()


def ret_per_volume(log_ret: pd.Series, volume: pd.Series) -> pd.Series:
    """Amihud-style illiquidity: |log_ret| / volume. Zero when volume is zero."""
    return (log_ret.abs() / volume.replace(0, np.nan)).fillna(0.0)
