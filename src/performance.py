# src/performance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


# =========================
# Helpers de métricas
# =========================
def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return float((r.mean() * 252 - rf) / (r.std() * np.sqrt(252)))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    downside = r[r < 0].std()
    if len(r) == 0 or downside == 0 or np.isnan(downside):
        return np.nan
    return float((r.mean() * 252 - rf) / (downside * np.sqrt(252)))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return np.nan
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1.0
    mdd = max_drawdown(equity)
    if mdd is None or mdd >= 0:
        return np.nan
    return float(cagr / abs(mdd))


# =========================
# Resumen y plots estándar
# =========================
def summarize_performance(train_res: dict, test_res: dict, val_res: dict, save_csv: bool = False):
    """
    Recibe los dicts que devuelve Backtester.run y muestra/guarda métricas por fase.
    """
    sections = {
        "TRAIN": train_res,
        "TEST": test_res,
        "VALIDATION": val_res
    }
    rows = []
    for name, res in sections.items():
        eq = res["equity"]["portfolio_value"]
        r = _returns(eq)
        rows.append([
            name,
            float(eq.iloc[0]),
            float(eq.iloc[-1]),
            sharpe(r),
            sortino(r),
            calmar(eq),
            max_drawdown(eq),
            res["costs"]["commissions_total"],
            res["costs"]["borrow_total"],
            len(res["trades"])
        ])

    headers = ["Phase", "Start", "End", "Sharpe", "Sortino", "Calmar", "MaxDD", "Commissions", "Borrow", "N_Trades"]
    print(tabulate(rows, headers=headers, floatfmt=".4f"))

    if save_csv:
        out = pd.DataFrame(rows, columns=headers)
        out.to_csv("performance_summary.csv", index=False)


def trade_stats(trades_df: pd.DataFrame):
    """
    Retorna estadísticas agregadas de trades para impresión en consola.
    """
    if trades_df is None or len(trades_df) == 0:
        return {"n": 0, "win_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan, "profit_factor": np.nan}

    rets = trades_df["return"].astype(float) if "return" in trades_df.columns else trades_df["PnL"].astype(float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / len(rets) if len(rets) > 0 else np.nan
    avg_win = wins.mean() if len(wins) > 0 else np.nan
    avg_loss = losses.mean() if len(losses) > 0 else np.nan
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf

    return {
        "n": int(len(rets)),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
        "avg_win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "profit_factor": float(pf) if np.isfinite(pf) else np.inf
    }


def plot_all(train_res: dict, test_res: dict, val_res: dict):
    """
    Genera gráficos estándar con los dicts de Backtester.run:
      - equity_curve.png (curva unificada de TRAIN/TEST/VAL)
      - trade_distribution.png (histograma de retornos por trade)
    """
    # 1) Equity curve concatenada
    eq_all = pd.concat([
        train_res["equity"]["portfolio_value"],
        test_res["equity"]["portfolio_value"],
        val_res["equity"]["portfolio_value"]
    ])
    plt.figure()
    eq_all.plot()
    plt.title("Equity Curve (Train/Test/Validation)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.close()

    # 2) Distribución de retornos por trade (con manejo robusto de NaN/vacío)
    all_trades = pd.concat(
        [train_res["trades"], test_res["trades"], val_res["trades"]],
        ignore_index=True
    )
    plt.figure()
    col = None
    if len(all_trades) > 0:
        if "return" in all_trades.columns:
            col = "return"
        elif "PnL" in all_trades.columns:
            col = "PnL"

    if col is not None:
        vals = pd.to_numeric(all_trades[col], errors="coerce").dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=20)
        else:
            plt.hist([], bins=20)  # vacío, pero no crashea
    else:
        plt.hist([], bins=20)

    plt.title("Trade Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("trade_distribution.png")
    plt.close()


# =========================
# Gráficos EXTRA
# =========================
def plot_hedge_beta(beta_series: pd.Series, path: str = "hedge_beta.png"):
    """
    Grafica la serie temporal del hedge ratio (beta_t).
    """
    if beta_series is None or len(beta_series) == 0:
        print("plot_hedge_beta: beta_series vacío; no se genera gráfico.")
        return
    plt.figure()
    beta_series.plot()
    plt.title("Hedge Ratio (β_t) over time")
    plt.xlabel("Date")
    plt.ylabel("β_t")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


def plot_spread_evolution(spread_series: pd.Series, path: str = "spread_evolution.png"):
    """
    Grafica la evolución del spread y su media móvil (para visualizar mean reversion).
    """
    if spread_series is None or len(spread_series) == 0:
        print("plot_spread_evolution: spread vacío; no se genera gráfico.")
        return
    ma = spread_series.rolling(60).mean()
    plt.figure()
    spread_series.plot(label="Spread")
    ma.plot(label="MA(60)")
    plt.title("Spread Evolution")
    plt.xlabel("Date")
    plt.ylabel("y - (α_t + β_t x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


def plot_johansen_eigenvector(eigvec_series: pd.Series | None, path: str = "johansen_eigenvector.png"):
    """
    (Opcional) Si tienes la 1ª componente del eigenvector de Johansen a través del tiempo,
    grafícalo. Si no, solo avisa y omite.
    """
    if eigvec_series is None or len(eigvec_series) == 0:
        print("plot_johansen_eigenvector: no hay eigenvector en el tiempo; se omite.")
        return
    plt.figure()
    eigvec_series.plot()
    plt.title("First Johansen Eigenvector over time")
    plt.xlabel("Date")
    plt.ylabel("Eigenvector component")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


# ======== Z-SCORE + SEÑALES y EQUITY POR FASES ========
def _concat_series_if_present(res_train: dict, res_test: dict, res_val: dict, key: str):
    """Concatena res[key] de train/test/val si existen; si no, devuelve None."""
    parts = []
    for res in (res_train, res_test, res_val):
        if key in res and res[key] is not None and len(res[key]) > 0:
            s = res[key].copy()
            try:
                s = s.squeeze()
            except Exception:
                pass
            parts.append(s)
    if len(parts) == 0:
        return None
    return pd.concat(parts).sort_index()


def plot_zscore_and_signals(res_train: dict, res_test: dict, res_val: dict,
                            entry_z: float, exit_z: float, stop_z: float,
                            path: str = "zscore_signals.png"):
    """
    Dibuja el z-score (preferentemente res['z']) y marca las señales (res['signals']).
    Requiere que el backtester esté guardando 'z' y 'signals'. Si no existen, avisa y no genera.
    """
    z = _concat_series_if_present(res_train, res_test, res_val, "z")
    sig = _concat_series_if_present(res_train, res_test, res_val, "signals")

    if z is None or sig is None:
        print("plot_zscore_and_signals: No se encontraron 'z' o 'signals' en los resultados; se omite.")
        return

    z = z.astype(float)
    sig = sig.reindex(z.index).astype(float).fillna(0.0)

    # Puntos de entrada (cambios 0 -> ±1)
    sig_shift = sig.shift(1).fillna(0.0)
    entries = (sig != 0) & (sig_shift == 0)
    long_entries = entries & (sig > 0)
    short_entries = entries & (sig < 0)

    plt.figure(figsize=(10, 5))
    z.plot(label="Z-score")
    plt.axhline(entry_z, linestyle="--")
    plt.axhline(-entry_z, linestyle="--")
    plt.axhline(exit_z, linestyle=":")
    plt.axhline(-exit_z, linestyle=":")
    plt.axhline(stop_z, linestyle="-.")
    plt.axhline(-stop_z, linestyle="-.")

    # Marcadores de señales
    plt.scatter(z.index[long_entries], z[long_entries], marker="^", s=40, label="Long entry")
    plt.scatter(z.index[short_entries], z[short_entries], marker="v", s=40, label="Short entry")

    plt.title("Z-score (VECM/ECT) y Señales")
    plt.xlabel("Date"); plt.ylabel("Z")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Guardado: {path}")


def plot_equity_by_phase(res_train: dict, res_test: dict, res_val: dict,
                         paths: tuple[str, str, str] = ("equity_train.png", "equity_test.png", "equity_val.png")):
    """
    Genera una imagen de equity curve por cada fase.
    """
    names = ("TRAIN", "TEST", "VALIDATION")
    ress = (res_train, res_test, res_val)

    for name, res, path in zip(names, ress, paths):
        if "equity" not in res or "portfolio_value" not in res["equity"]:
            print(f"plot_equity_by_phase: no hay equity en {name}; se omite.")
            continue
        series = res["equity"]["portfolio_value"]
        plt.figure(figsize=(9, 4))
        series.plot()
        plt.title(f"Equity Curve - {name}")
        plt.xlabel("Date"); plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Guardado: {path}")
