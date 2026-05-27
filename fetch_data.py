import yfinance as yf
import pandas as pd
import requests

# ── SPY OHLCV ────────────────────────────────────────────────────────────────

ticker = "SPY"
print(f"Fetching all available daily OHLCV data for {ticker}...")

spy = yf.download(ticker, period="max", interval="1d", auto_adjust=True)

if spy.empty:
    print("No data returned.")
else:
    spy.index = pd.to_datetime(spy.index)
    spy.to_csv("spy_ohlcv.csv")
    print(f"Saved {len(spy)} rows ({spy.index[0].date()} to {spy.index[-1].date()}) to spy_ohlcv.csv")
    print(spy.tail())

# ── FRED ─────────────────────────────────────────────────────────────────────

FRED_API_KEY = "e63271d323ba5ebd30238b4238aa6240"

FRED_SERIES = {
    "T10Y3M":   "fred_t10y3m.csv",
    "BAA10Y":   "fred_baa10y.csv",
    "VIXCLS":   "fred_vix.csv",
    "DGS2":     "fred_dgs2.csv",
    "DGS10":    "fred_dgs10.csv",
    "ICSA":     "fred_icsa.csv",
    "CPIAUCSL": "fred_cpi.csv",
    "UMCSENT":  "fred_umcsent.csv",
}

def fetch_fred(series_id: str, api_key: str) -> pd.Series:
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    s = pd.Series(
        {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
        name=series_id,
    )
    s.index = pd.to_datetime(s.index)
    return s

for series_id, out_file in FRED_SERIES.items():
    print(f"Fetching FRED series {series_id}...")
    s = fetch_fred(series_id, FRED_API_KEY)
    s.to_csv(out_file, header=True)
    print(f"Saved {len(s)} rows ({s.index[0].date()} to {s.index[-1].date()}) to {out_file}")
