
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    sharpe: float
    max_drawdown: float

def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0)

def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    sr = np.sqrt(252) * (mu / (sigma + 1e-9))
    return sr

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min()

def momentum_signal(prices: pd.Series, lookback: int = 126) -> pd.Series:
    return (prices / prices.shift(lookback) - 1.0).fillna(0.0)

def mean_reversion_signal(returns: pd.Series, lookback: int = 5) -> pd.Series:
    z = (returns - returns.rolling(lookback).mean()) / (returns.rolling(lookback).std() + 1e-9)
    return -z.fillna(0.0)

def position_from_signal(sig: pd.Series, thresh: float = 0.0, cap: float = 1.0) -> pd.Series:
    pos = sig.copy()
    pos = pos.clip(lower=-cap, upper=cap)
    pos[np.abs(pos) < thresh] = 0.0
    return pos.shift(1).fillna(0.0)  # trade next day

def backtest(prices: pd.Series, pos: pd.Series, fees_bps: float = 1.0) -> BacktestResult:
    rets = compute_returns(prices)
    strat_rets = pos * rets
    # transaction costs when position changes
    tc = (pos.diff().abs() * fees_bps / 1e4).fillna(0.0)
    strat_rets = strat_rets - tc
    equity = (1.0 + strat_rets).cumprod()
    trades = pd.DataFrame({"position": pos, "returns": strat_rets})
    sharpe = (np.sqrt(252) * strat_rets.mean() / (strat_rets.std() + 1e-9))
    mdd = max_drawdown(equity)
    return BacktestResult(equity, trades, float(sharpe), float(mdd))

def run(ticker: str, start: str, end: str, mode: str):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"].dropna()
    rets = compute_returns(prices)

    if mode == "momentum":
        sig = momentum_signal(prices, lookback=126)
        pos = position_from_signal(sig, thresh=0.0, cap=1.0)
    elif mode == "meanrev":
        sig = mean_reversion_signal(rets, lookback=5)
        pos = position_from_signal(sig, thresh=0.5, cap=1.0)
    else:
        raise ValueError("mode must be momentum or meanrev")

    result = backtest(prices, pos)
    print(f"Sharpe: {result.sharpe:.2f} | Max Drawdown: {result.max_drawdown:.1%}")
    print(result.equity.tail())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--mode", default="momentum", choices=["momentum", "meanrev"])
    args = ap.parse_args()
    run(args.ticker, args.start, args.end, args.mode)
