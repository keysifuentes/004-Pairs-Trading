# main.py
import pandas as pd
from itertools import combinations
from pathlib import Path

from src.data import load_prices, train_test_val_split
from src.pairs import (
    screen_pairs, engle_granger_test, johansen_vecm_fit,
    save_rolling_corr, save_pairs_table, rolling_johansen_eigenvector
)
from src.kalman import HedgeKalman, SignalKalman
from src.strategy import VECMZScoreStrategy
from src.backtest import Backtester
from src.performance import (
    summarize_performance, plot_all, trade_stats,
    plot_hedge_beta, plot_spread_evolution, plot_johansen_eigenvector,
    plot_zscore_and_signals, plot_equity_by_phase
)

# =============================
# CONFIGURACIÓN
# =============================
TICKERS = ["AAPL", "MSFT", "NVDA"]   # puedes agregar más
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
INITIAL_CASH = 1_000_000.0

FORCE_PAIR = False
FORCED_X = "AAPL"
FORCED_Y = "MSFT"

CORR_THRESHOLD = 0.7
# =============================


def select_pair(train_df: pd.DataFrame):
    candidates = screen_pairs(train_df, threshold=CORR_THRESHOLD)

    if len(candidates) == 0:
        print(f"No hubo pares con correlación >= {CORR_THRESHOLD}. Evaluando todas las combinaciones…")
        cols = train_df.columns.tolist()
        candidates = list(combinations(cols, 2))

    ranked = []
    for (x, y) in candidates:
        eg = engle_granger_test(train_df[[x, y]])
        # Descomenta para filtrar pares no cointegrados por EG:
        # if eg["adf_pvalue"] >= 0.05:
        #     continue
        joh = johansen_vecm_fit(train_df[[x, y]])
        ranked.append((x, y, eg["adf_pvalue"], joh["trace_stat"], joh["eigvec"]))

    if len(ranked) == 0:
        raise ValueError("No fue posible evaluar pares. Revisa los datos o cambia TICKERS/fechas.")

    ranked = sorted(ranked, key=lambda t: (t[2], -t[3]))  # p-valor EG bajo y trace alto
    x, y, pval, trace, _ = ranked[0]
    print(f"Par seleccionado: {x}/{y} | EG p-value={pval:.4f} | Johansen trace={trace:.3f}")
    return x, y


def main():
    print("=== CARGANDO DATOS ===")
    prices = load_prices(TICKERS, start=START_DATE, end=END_DATE, save=True, data_dir="data")
    print(f"Datos cargados: {prices.shape[0]} días x {prices.shape[1]} activos\n")

    # Split
    train, test, val = train_test_val_split(prices, 0.6, 0.2, 0.2)
    print("Split en TRAIN/TEST/VAL completado.")

    # Crear carpeta de reportes
    Path("reports").mkdir(exist_ok=True)

    # Guardar correlación rolling y tabla de pares (para el reporte)
    save_rolling_corr(train, window=252, out_csv="reports/rolling_corr_last.csv")
    save_pairs_table(train, out_csv="reports/pair_stats.csv")

    # Selección de par
    print("\n=== SELECCIONANDO PARES ===")
    if FORCE_PAIR:
        x, y = FORCED_X, FORCED_Y
        if x not in train.columns or y not in train.columns:
            raise ValueError(f"Los tickers forzados {x}/{y} no están en los datos. Columnas: {list(train.columns)}")
        print(f"Par seleccionado (forzado): {x}/{y}")
    else:
        x, y = select_pair(train)

    # Kalman (hedge y señal)
    hedge = HedgeKalman()
    _ = hedge.fit(train[x], train[y])
    signal_kf = SignalKalman()

    # Estrategia VECM/ECT z-score
    strat = VECMZScoreStrategy(entry_z=1.5, exit_z=0.3, stop_z=3.5)

    # Backtests
    print("\n=== BACKTEST TRAIN ===")
    bt_train = Backtester(cash=INITIAL_CASH)
    res_train = bt_train.run(train[[x, y]], hedge, signal_kf, strat, phase="train", pair=(x, y))

    print("\n=== BACKTEST TEST ===")
    bt_test = Backtester(cash=res_train["equity"]["portfolio_value"].iloc[-1])
    res_test = bt_test.run(test[[x, y]], hedge, signal_kf, strat, phase="test", pair=(x, y))

    print("\n=== BACKTEST VALIDATION ===")
    bt_val = Backtester(cash=res_test["equity"]["portfolio_value"].iloc[-1])
    res_val = bt_val.run(val[[x, y]], hedge, signal_kf, strat, phase="validation", pair=(x, y))

    # Gráficos extra: β_t y spread (TRAIN)
    beta_series = None
    alpha_series = None
    for attr in ["beta_", "beta_series", "beta"]:
        if hasattr(hedge, attr):
            beta_series = getattr(hedge, attr)
            break
    for attr in ["alpha_", "alpha_series", "alpha"]:
        if hasattr(hedge, attr):
            alpha_series = getattr(hedge, attr)
            break

    if beta_series is None:
        import numpy as np
        bx = np.polyfit(train[x].values, train[y].values, 1)  # y ≈ b1*x + b0
        beta_series = pd.Series(bx[0], index=train.index)
        if alpha_series is None:
            alpha_series = pd.Series(bx[1], index=train.index)
    if alpha_series is None:
        alpha_series = pd.Series(0.0, index=train.index)

    spread_series = train[y] - (alpha_series + beta_series * train[x])
    plot_hedge_beta(beta_series, path="hedge_beta.png")
    plot_spread_evolution(spread_series, path="spread_evolution.png")

    # Eigenvector de Johansen en ventana móvil (TRAIN)
    eig_series = rolling_johansen_eigenvector(train, x, y, window=252, step=5)
    plot_johansen_eigenvector(eig_series, path="johansen_eigenvector.png")

    # ======= NUEVAS GRÁFICAS =======
    plot_zscore_and_signals(
        res_train, res_test, res_val,
        entry_z=strat.entry_z, exit_z=strat.exit_z, stop_z=strat.stop_z,
        path="zscore_signals.png"
    )
    plot_equity_by_phase(
        res_train, res_test, res_val,
        paths=("equity_train.png", "equity_test.png", "equity_val.png")
    )
    # ================================

    # Resultados estándar
    print("\n=== RESULTADOS ===")
    plot_all(res_train, res_test, res_val)
    summarize_performance(res_train, res_test, res_val, save_csv=True)

    all_trades = pd.concat([res_train["trades"], res_test["trades"], res_val["trades"]], ignore_index=True)
    ts = trade_stats(all_trades)
    print("Global Trade Stats:", ts)

    total_comm = (
        res_train["costs"]["commissions_total"]
        + res_test["costs"]["commissions_total"]
        + res_val["costs"]["commissions_total"]
    )
    total_borrow = (
        res_train["costs"]["borrow_total"]
        + res_test["costs"]["borrow_total"]
        + res_val["costs"]["borrow_total"]
    )
    print(f"Total commissions: {total_comm:.2f} | Total borrow: {total_borrow:.2f}")
    print("\nListo. Datos en /data y reportes/CSV/PNG en la raíz y /reports/.")


if __name__ == "__main__":
    main()
