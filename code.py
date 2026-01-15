# ============================================================
# REGIME-ROBUST MARKET-NEUTRAL HEDGE DISCOVERY (FINAL STABLE)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import pywt

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# ============================================================
# CONFIG
# ============================================================

STOCKS = [
    "NTPC.NS", "ADANIPOWER.NS", "POWERGRID.NS", "TATAPOWER.NS",
    "JSWENERGY.NS", "ADANIGREEN.NS", "NHPC.NS",
    "TORNTPOWER.NS", "ADANIENSOL.NS"
]

MARKET = "^NSEI"
START = "2022-01-01"
END   = "2026-01-01"

MIN_SAMPLE = 60   # << FIX: aligned with your data length

# ============================================================
# PHASE 1 — DATA INGESTION
# ============================================================

def fetch_prices(tickers):
    px = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False
    )["Close"]
    return px.ffill().dropna(subset=[MARKET])

prices = fetch_prices(STOCKS + [MARKET])

# ============================================================
# PHASE 2 — LOG RETURNS
# ============================================================

returns = np.log(prices / prices.shift(1)).dropna()

# ============================================================
# PHASE 3 — WAVELET MID-BAND FILTER (DAILY-SAFE)
# ============================================================

def wavelet_midband(x, wavelet="db4"):
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    level = min(1, max_level)
    if level < 1:
        return x - np.mean(x)

    coeffs = pywt.wavedec(x, wavelet, level=level)
    filt = [coeffs[i] if i == 1 else np.zeros_like(coeffs[i])
            for i in range(len(coeffs))]
    return pywt.waverec(filt, wavelet)[:len(x)]

filtered_returns = pd.DataFrame(
    {c: wavelet_midband(returns[c].values) for c in returns.columns},
    index=returns.index
)

# ============================================================
# PHASE 4 — MARKET NEUTRALIZATION
# ============================================================

def market_neutral(asset, market):
    beta = np.cov(asset, market)[0,1] / np.var(market)
    return asset - beta * market

market_ret = filtered_returns[MARKET]

idio_returns = pd.DataFrame({
    c: market_neutral(filtered_returns[c].values, market_ret.values)
    for c in STOCKS
}, index=filtered_returns.index)

# ============================================================
# PHASE 5 — REGIME DETECTION (VALIDATION ONLY)
# ============================================================

def detect_regimes(mkt_ret, n_states=3):
    vol = mkt_ret.ewm(span=21, adjust=False).std()
    df = pd.DataFrame({"ret": mkt_ret, "vol": vol}).dropna()

    X = StandardScaler().fit_transform(df.values)
    X += 1e-6 * np.random.randn(*X.shape)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1200,
        random_state=42
    )
    hmm.fit(X)
    return pd.Series(hmm.predict(X), index=df.index)

regimes = detect_regimes(returns[MARKET])

print("\nREGIME DISTRIBUTION:")
print(regimes.value_counts())
print("Total observations:", len(regimes))

# ============================================================
# PHASE 6 — BAYESIAN HEDGE AVERAGING
# ============================================================

def hedge_states(beta_hat, width=0.30, n=15):
    return np.linspace(beta_hat*(1-width), beta_hat*(1+width), n)

def log_likelihood(spread):
    mu = np.mean(spread)
    sig = np.std(spread) + 1e-6
    return np.sum(norm.logpdf(spread, mu, sig))

def bayesian_spread(x, y):
    beta_hat = np.cov(x, y)[0,1] / np.var(x)
    betas = hedge_states(beta_hat)

    spreads, ll = [], []
    for b in betas:
        s = y - b * x
        spreads.append(s)
        ll.append(log_likelihood(s))

    ll = np.array(ll)
    w = np.exp(ll - ll.max())
    w /= w.sum()

    return np.sum(np.array(spreads).T * w, axis=1)

# ============================================================
# PHASE 7 — REGIME OPPOSITION (WEAK BONUS)
# ============================================================

def regime_exposure(series, regimes):
    out = []
    for r in np.unique(regimes):
        s = series[regimes == r]
        if len(s) > 20:
            out.append(np.mean(s) / (np.std(s) + 1e-6))
    return np.array(out)

def regime_opposition(x, y, regimes):
    ex_x = regime_exposure(x, regimes)
    ex_y = regime_exposure(y, regimes)
    if len(ex_x) < 2 or len(ex_y) < 2:
        return 0.0
    return -np.corrcoef(ex_x, ex_y)[0,1]

# ============================================================
# PHASE 8 — ECONOMIC METRICS
# ============================================================

def max_drawdown(x):
    c = np.cumsum(x)
    peak = np.maximum.accumulate(c)
    return np.min(c - peak)

# ============================================================
# PHASE 9 — FINAL HEDGE SCORING
# ============================================================

results = []

# Hedge-centric weights
W_DD  = 1.0
W_RET = 0.5
W_VOL = 0.3
W_OPP = 0.2

for i in range(len(STOCKS)):
    for j in range(i+1, len(STOCKS)):

        x = idio_returns[STOCKS[i]].loc[regimes.index].values
        y = idio_returns[STOCKS[j]].loc[regimes.index].values

        spread = pd.Series(
            bayesian_spread(x, y),
            index=regimes.index
        )

        if len(spread) < MIN_SAMPLE:
            continue

        cum_ret = spread.sum()
        dd = max_drawdown(spread)
        vol = spread.std()
        opp = regime_opposition(x, y, regimes)

        hedge_score = (
            -W_DD  * abs(dd)
            -W_RET * abs(cum_ret)
            -W_VOL * vol
            +W_OPP * opp
        )

        results.append({
            "pair": f"{STOCKS[i]}-{STOCKS[j]}",
            "hedge_score": hedge_score,
            "cumulative_return": cum_ret,
            "max_drawdown": dd,
            "spread_vol": vol,
            "opposition": opp
        })

# ============================================================
# FINAL OUTPUT (SAFE)
# ============================================================

if len(results) == 0:
    raise RuntimeError(
        "No valid hedge pairs found. "
        "Extend date range or reduce MIN_SAMPLE."
    )

df = (
    pd.DataFrame(results)
    .sort_values("hedge_score", ascending=False)
)

print("\nBEST REGIME-ROBUST STRUCTURAL HEDGES:\n")
print(
    df.head(15)
      .round(6)
      .to_string(index=False)
)

