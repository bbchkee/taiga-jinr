
"""
Unfolding utilities (Tikhonov regularization) for IACT energy spectra.

We solve for true counts N (per *true-energy* bin) from measured counts M (per *reco-energy* bin):
    M ≈ R @ N
where R is the response matrix built from MC: R[i,j] = P(reco_bin i | true_bin j).

We use a Tikhonov-regularized least squares:
    argmin_N  ||W^(1/2) (R N - M)||^2 + tau * ||L N||^2
where W is a diagonal weight matrix (1/σ_i^2, σ_i ~ sqrt(max(M_i,1))), and L is the 2nd-difference operator.

Outputs can be converted to a differential flux using effective area, time, and bin widths.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt, os

def build_response_matrix(
    model_df: pd.DataFrame,
    true_energy_col: str = "energy",
    reco_energy_col: str = "reconstructed_energy",
    bin_edges_true: np.ndarray | None = None,
    bin_edges_reco: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build response matrix R[i,j] = P(reco_bin i | true_bin j) from MC.
    If bin_edges_* not provided, use the same edges for true and reco based on the union of energies.
    Returns: (R, edges_reco, edges_true)
    """
    E_true = model_df[true_energy_col].to_numpy(dtype=float)
    E_reco = model_df[reco_energy_col].to_numpy(dtype=float)

    if bin_edges_true is None or bin_edges_reco is None:
        Emin = np.nanmin([np.nanmin(E_true), np.nanmin(E_reco)])
        Emax = np.nanmax([np.nanmax(E_true), np.nanmax(E_reco)])
        edges = np.logspace(np.log10(Emin), np.log10(Emax), 21)
        if bin_edges_true is None:
            bin_edges_true = edges
        if bin_edges_reco is None:
            bin_edges_reco = edges

    H, reco_edges, true_edges = np.histogram2d(E_reco, E_true, bins=(bin_edges_reco, bin_edges_true))
    # Normalize columns (for each true bin j): sum_i R[i,j] = 1
    R = H.copy().astype(float)
    col_sums = R.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    R /= col_sums
    return R, reco_edges, true_edges


def _second_diff_matrix(n: int) -> np.ndarray:
    """2nd difference operator L (n-2 by n) for smoothness regularization."""
    L = np.zeros((n-2, n))
    for i in range(n-2):
        L[i, i] = 1.0
        L[i, i+1] = -2.0
        L[i, i+2] = 1.0
    return L


def tikhonov_unfold(
    M: np.ndarray,
    R: np.ndarray,
    tau: float = 1.0,
    weights: Optional[np.ndarray] = None,
    non_negative: bool = True,
) -> np.ndarray:
    """
    Solve (R^T W R + tau L^T L) N = R^T W M for N.
    - M: measured counts vector (len m)
    - R: response matrix (m x n)
    - tau: regularization strength
    - weights: optional per-bin weights for M (e.g. 1/sigma^2). If None, use 1/max(M,1).
    - non_negative: clip negatives to zero at the end.
    """
    m, n = R.shape
    M = M.astype(float).reshape(m)
    if weights is None:
        sigma2 = np.maximum(M, 1.0)  # Poisson variance ~ N
        W = 1.0 / sigma2
    else:
        W = weights.reshape(m)
    # Weighted design
    Wsqrt = np.sqrt(W)
    RW = (R.T * W) @ R   # R^T W R
    RtWM = R.T @ (W * M) # R^T W M

    L = _second_diff_matrix(n)
    A = RW + tau * (L.T @ L)
    b = RtWM

    # Solve
    N = np.linalg.solve(A + 1e-12*np.eye(n), b)
    if non_negative:
        N = np.clip(N, 0, None)
    return N


def unfold_to_flux(
    unfolded_counts: np.ndarray,
    true_edges: np.ndarray,
    eff_area_m2: np.ndarray,
    exposure_sec: float,
) -> pd.DataFrame:
    """
    Convert unfolded counts per true-energy bin into differential flux.
    Returns DataFrame with e_low,e_high,e_center,N_true,dE,A_eff,T_sec,flux,flux_err (Poisson approx).
    """
    e_low = true_edges[:-1]
    e_high = true_edges[1:]
    e_center = np.sqrt(e_low * e_high)
    dE = e_high - e_low

    N = unfolded_counts.astype(float)
    A = eff_area_m2.astype(float)
    T = float(exposure_sec)

    with np.errstate(divide="ignore", invalid="ignore"):
        flux = N / (A * T * dE)
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        # rough error bar: sqrt(N)/(...)
        dN = np.sqrt(np.maximum(N, 1.0))
        flux_err = dN / (A * T * dE)
        flux_err = np.nan_to_num(flux_err, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame({
        "e_low": e_low, "e_high": e_high, "e_center": e_center, "dE": dE,
        "N_true": N, "A_eff_m2": A, "T_sec": T, "flux": flux, "flux_err": flux_err
    })



# ===== Auto tau (L-curve) & plots =====

def _solve_tikhonov_components(M: np.ndarray, R: np.ndarray, tau: float, weights=None) -> tuple[np.ndarray, float, float]:
    """Solve and return (N, misfit, smoothness)."""
    m, n = R.shape
    if weights is None:
        W = 1.0 / np.maximum(M, 1.0)
    else:
        W = weights.reshape(m)
    L = _second_diff_matrix(n)
    RW = (R.T * W) @ R
    RtWM = R.T @ (W * M)
    A = RW + tau * (L.T @ L)
    N = np.linalg.solve(A + 1e-12*np.eye(n), RtWM)
    resid = R @ N - M
    misfit = float(np.sqrt(np.sum((np.sqrt(W) * resid) ** 2)))
    smooth = float(np.linalg.norm(L @ N))
    return N, misfit, smooth

def select_tau_lcurve(M: np.ndarray, R: np.ndarray, tau_grid: np.ndarray | list[float], weights=None) -> tuple[float, pd.DataFrame]:
    """Pick tau at max curvature of L-curve (log-log)."""
    rows = []
    for tau in tau_grid:
        _, mis, sm = _solve_tikhonov_components(M, R, tau, weights=weights)
        rows.append((tau, mis, sm))
    df = pd.DataFrame(rows, columns=["tau","misfit","smoothness"])
    x = np.log(df["misfit"].to_numpy() + 1e-30)
    y = np.log(df["smoothness"].to_numpy() + 1e-30)
    t = np.log(df["tau"].to_numpy())
    x1 = np.gradient(x, t); y1 = np.gradient(y, t)
    x2 = np.gradient(x1, t); y2 = np.gradient(y1, t)
    kappa = np.abs(x1*y2 - y1*x2) / np.power((x1**2 + y1**2), 1.5)
    df["kappa"] = kappa
    best = int(np.nanargmax(kappa))
    return float(df.loc[best, "tau"]), df

def save_lcurve_plot(df_lc: pd.DataFrame, out_png: str = "plots/lcurve.png") -> None:
    #import matplotlib.pyplot as plt, os
    fig = plt.figure(figsize=(7,5))
    plt.plot(df_lc["misfit"], df_lc["smoothness"], marker="o")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Data misfit ||W^{1/2}(R N - M)||"); plt.ylabel("Smoothness ||L N||")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def save_response_matrix_plot(R: np.ndarray, reco_edges: np.ndarray, true_edges: np.ndarray, out_png: str = "plots/response_matrix.png") -> None:
    #import matplotlib.pyplot as plt, os
    fig = plt.figure(figsize=(6,5))
    plt.imshow(R, origin="lower", aspect="auto")
    plt.xlabel("True-energy bin index"); plt.ylabel("Reco-energy bin index")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def run_unfolding_pipeline(
    exp_df: pd.DataFrame,
    model_df: pd.DataFrame,
    eff_df: pd.DataFrame,
    tau: float = 1.0,
    reco_energy_col: str = "reconstructed_energy",
    true_energy_col: str = "energy",
    theta2_col: str = "theta2_1",
    theta2_max: float = 0.05,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    High-level helper:
      - builds response matrix from model_df (using eff_df true-bin edges),
      - histograms measured counts M from exp_df (reco energies, theta2 cut),
      - unfolds with Tikhonov,
      - converts to flux with A_eff and exposure (from 'por' 2-min portions).
    Returns: (flux_df, R, M)
    """
    # Use eff_df true-bin edges for both true & reco (simplify)
    true_edges = eff_df["e_low"].to_numpy()
    true_edges = np.append(true_edges, eff_df["e_high"].to_numpy()[-1])
    reco_edges = true_edges

    # Build response
    R, _, _ = build_response_matrix(model_df, true_energy_col=true_energy_col,
                                    reco_energy_col=reco_energy_col,
                                    bin_edges_true=true_edges, bin_edges_reco=reco_edges)

    # Measured counts in reco bins (theta2 selection)
    df = exp_df.copy()
    if theta2_col in df.columns:
        df = df[df[theta2_col] < theta2_max]
    E_reco = df[reco_energy_col].dropna().to_numpy()
    M, _ = np.histogram(E_reco, bins=reco_edges)

    # Exposure time from 'por' (2-min portions)
    from .analysis import compute_observation_time_minutes
    T_min = compute_observation_time_minutes(exp_df, portion_col="por", portion_duration_min=2.0)
    T_sec = T_min * 60.0

    # Auto-tau (L-curve) если запрошено
    if tau is None:
        # сетка подбора можно варьировать при желании
        tau_grid = np.logspace(-4, 2, 25)
        tau_chosen, df_lc = select_tau_lcurve(M, R, tau_grid)
        # сохранить график L-кривой (полезно для дебага)
        try:
            save_lcurve_plot(df_lc, out_png="plots/lcurve.png")
        except Exception:
            pass
        print(f"[Unfolding] Auto tau selected by L-curve: {tau_chosen:.3g}")
        tau = float(tau_chosen)

    # Unfold
    N_true = tikhonov_unfold(M, R, tau=tau, weights=None, non_negative=True)

    # Effective area (align bins with eff_df)
    A = eff_df["eff_area_m2"].to_numpy()

    # Convert to flux
    flux_df = unfold_to_flux(N_true, true_edges, A, T_sec)
    return flux_df, R, M
