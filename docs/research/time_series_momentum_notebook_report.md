# Time Series Momentum Notebook Report

- Paper reference: Moskowitz, Ooi, and Pedersen (2012), "Time Series Momentum."
- Proxy universe used here: SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, SHY, LQD, HYG, GLD, SLV, DBC, USO, UNG, FXE, FXY
- Missing configured symbols: none
- Signal: `(1 + px__ret_252) / (1 + px__ret_21) - 1`, then `sign(signal)`.
- Rebalance frequency: monthly (first trading day of each month, lagged into next-day execution).
- Evaluation window requested: 2020-01-01 to 2026-03-19
- Evaluation window: 2020-01-02 to 2026-03-19
- Strategy Sharpe: 0.251
- Strategy total return: 12.57%
- Strategy max drawdown: -19.34%
- Benchmark total return: 53.17%
- Excess return vs equal-weight buy-and-hold benchmark: -40.60%

Differences from the paper:
- This notebook uses local ETF proxies rather than the paper's 58 futures/forwards.
- It uses total-return ETF prices instead of the paper's excess-return futures construction.
- It applies equal gross normalization to signed signals instead of the paper's volatility-scaled portfolio construction.
