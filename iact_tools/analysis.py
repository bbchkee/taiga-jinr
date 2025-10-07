
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SpectrumAnalyzer:
    def __init__(self, exp_data: pd.DataFrame):
        self.exp_data = exp_data

    def plot_spectrum(self, threshold=0.05, out_png: str = 'plots/spectrum_theta_cut.png'):
        df = self.exp_data[self.exp_data['theta2_1'] < threshold].copy()
        fig = plt.figure(figsize=(8,5))
        plt.hist(df['reconstructed_energy'], bins=np.linspace(0, 200, 50), alpha=0.7, label=f'theta2 < {threshold}')
        plt.xlabel('Energy (TeV)'); plt.ylabel('Counts'); plt.yscale('log'); plt.xlim(0, 200)
        plt.legend(); fig.tight_layout()
        import os; os.makedirs('plots', exist_ok=True)
        fig.savefig(out_png, dpi=150); plt.close(fig)

    def plot_all_spectrum(self, out_png: str = 'plots/spectrum_all.png'):
        fig = plt.figure(figsize=(8,5))
        plt.hist(self.exp_data['reconstructed_energy'], bins=np.linspace(0, 200, 50), alpha=0.7, label='all')
        plt.xlabel('Energy (TeV)'); plt.ylabel('Counts'); plt.yscale('log'); plt.xlim(0, 200)
        plt.legend(); fig.tight_layout()
        import os; os.makedirs('plots', exist_ok=True)
        fig.savefig(out_png, dpi=150); plt.close(fig)

    def plot_theta2_distribution(
        self,
        out_png_onoff: str = "plots/theta_on_off.png",
        out_png_diff: str = "plots/theta_diff.png",
        nbins: int = 50,
        theta2_max: float = 1.25,
        signal_max: float = 0.05,
        use_gamma_like: bool = True,
        model = False,
    ): # (:
        import os
        #import numpy as np
        #import pandas as pd
        #import matplotlib.pyplot as plt
        
        if not model:
            df = self.exp_data
        else:
            df = self.model_data
        # TODO добавить для модельных данных theta2 и спектр
        if df is None or df.empty:
            raise ValueError("exp_data is empty")

        os.makedirs(os.path.dirname(out_png_onoff) or ".", exist_ok=True)

        # --- ON выборка ---
        on_col = "theta2_1" if "theta2_1" in df.columns else ("theta2" if "theta2" in df.columns else None)
        if on_col is None:
            raise KeyError("Не найдены колонки 'theta2_1' или 'theta2' для ON")

        idx = df.index
        if use_gamma_like:
            suf = on_col.split("_")[-1] if "_" in on_col else None
            gl_col = f"gamma_like_{suf}" if suf and f"gamma_like_{suf}" in df.columns else ("gamma_like" if "gamma_like" in df.columns else None)
            on_mask = df.get(gl_col, pd.Series(True, index=idx)).astype(bool)
            on = df.loc[on_mask, on_col].dropna().to_numpy()
        else:
            on = df[on_col].dropna().to_numpy()

        # --- OFF выборки (theta2_2..theta2_6) ---
        off_cols = [c for c in (f"theta2_{i}" for i in range(2, 7)) if c in df.columns]
        bins = np.linspace(0.0, theta2_max, nbins)
        bins_sig = np.linspace(0.0, signal_max, max(5, int(signal_max / theta2_max * nbins)))

        N_off_all = []
        N_off_all_sig = []

        for c in off_cols:
            if use_gamma_like:
                suf = c.split("_")[-1]
                gl_col = f"gamma_like_{suf}"
                mask = df.get(gl_col, pd.Series(True, index=idx)).astype(bool)
                arr = df.loc[mask, c].dropna().to_numpy()
            else:
                arr = df[c].dropna().to_numpy()
            if arr.size:
                N_off_all.append(np.histogram(arr, bins=bins)[0])
                N_off_all_sig.append(np.histogram(arr, bins=bins_sig)[0])

        # Фоллбэк: одиночная колонка theta2_off
        if not N_off_all:
            if "theta2_off" in df.columns:
                arr = df["theta2_off"].dropna().to_numpy()
                if arr.size:
                    N_off_all.append(np.histogram(arr, bins=bins)[0])
                    N_off_all_sig.append(np.histogram(arr, bins=bins_sig)[0])
            else:
                raise ValueError("Нет фоновых колонок: ни theta2_2..theta2_6, ни theta2_off")

        m_off = len(N_off_all)
        alpha = 1.0 / m_off  # Li & Ma α = 1/число OFF областей

        # --- Гистограммы и усреднённый OFF ---
        N_on, edges = np.histogram(on, bins=bins)
        N_off_avg = np.mean(N_off_all, axis=0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.9

        # Figure 1: ON vs OFF(avg)
        fig = plt.figure(figsize=(8, 5))
        plt.bar(centers, N_on, width=width, alpha=0.7, label="Theta² ON")
        plt.bar(centers, N_off_avg, width=width, alpha=0.7, label=f"Background avg ({m_off} OFF)")
        plt.xlabel(r"$\theta^2$ (deg$^2$)")
        plt.ylabel("Counts")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_png_onoff, dpi=150)
        plt.close(fig)

        # Figure 2: Difference ON - OFF(avg)
        diff = N_on - N_off_avg
        fig = plt.figure(figsize=(8, 5))
        plt.step(centers, diff, where="mid", label="ON - OFF(avg)", linewidth=2)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel(r"$\theta^2$ (deg$^2$)")
        plt.ylabel("Counts")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Li & Ma в узком окне
        N_on_sig = np.histogram(on, bins=bins_sig)[0].sum()
        N_off_sig_total = np.sum(N_off_all_sig)  # сумма по всем OFF и по всем бинам в окне
        if N_on_sig > 0 and N_off_sig_total > 0:
            S = np.sqrt(
                2.0 * (
                    N_on_sig * np.log(((1 + alpha) / alpha) * (N_on_sig / (N_on_sig + N_off_sig_total))) +
                    N_off_sig_total * np.log((1 + alpha) * (N_off_sig_total / (N_on_sig + N_off_sig_total)))
                )
            )
        else:
            S = np.nan

        plt.figtext(0.14, 0.01, f"N_on = {N_on_sig}, N_off = {N_off_sig_total:.0f} (α={alpha:.2f}, signal area = {theta2_max:.2f}), S = {np.nan if np.isnan(S) else round(S,2)}")
        fig.tight_layout()
        fig.savefig(out_png_diff, dpi=150)
        plt.close(fig)


# ======= Observation time & physical spectrum =======

def compute_observation_time_minutes(df: pd.DataFrame, portion_col: str = "por", portion_duration_min: float = 3.0) -> float:
    """
    Estimate observation time (in minutes) from portion IDs.
    Assumptions:
      - Within a single observing session, portion IDs (e.g., 'por') are unique and generally increase.
      - When a new session begins, the portion IDs reset (e.g., go back to 0).
    We scan the sequence of portion IDs, detect resets where diff < 0, and count unique IDs per segment.
    Returns total_minutes = (#unique portions across all sessions) * portion_duration_min
    """
    if portion_col not in df.columns:
        raise KeyError(f"Column '{portion_col}' not found in DataFrame.")
    por = df[portion_col].to_numpy()
    if por.size == 0:
        return 0.0

    # Detect session resets: whenever difference < 0
    #import numpy as np
    d = np.diff(por.astype(float), prepend=por[0])
    resets = (d < 0).astype(int)
    session_ids = np.cumsum(resets)

    # Count unique portions per session and sum
    s = pd.Series(por)
    unique_counts = s.groupby(session_ids).nunique()
    total_portions = int(unique_counts.sum())
    return total_portions * float(portion_duration_min)


def compute_physical_spectrum(
    exp_df: pd.DataFrame,
    eff_df: pd.DataFrame,
    energy_col_reco: str = "reconstructed_energy",
    theta2_col: str = "theta2_1",
    theta2_max: float = 0.05,
) -> pd.DataFrame:
    """
    Compute a simple differential spectrum:
      dN / (dE * dt * A_eff)
    - Select ON events with theta2 < theta2_max (no background subtraction here).
    - Bin them using eff_df bin edges (e_low/e_high).
    - Use observation time derived from portion IDs ('por'), 2 minutes per portion.
    - Use effective area per bin from eff_df['eff_area_m2'].
    Returns a DataFrame with columns:
      e_low, e_high, e_center, N, dE, T_sec, A_eff_m2, flux, flux_err
    """
    # Observation time
    T_min = compute_observation_time_minutes(exp_df, portion_col="por", portion_duration_min=3.0)
    T_sec = T_min * 60.0

    # Event selection by theta2
    df = exp_df.copy()
    if theta2_col not in df.columns:
        # try fallback name
        theta2_col = "theta2" if "theta2" in df.columns else None
    if theta2_col is not None and theta2_col in df.columns:
        df = df[df[theta2_col] < theta2_max]

    # Binning per eff_df edges
    edges = eff_df["e_low"].to_numpy()
    edges = np.append(edges, eff_df["e_high"].to_numpy()[-1])
    centers = eff_df["e_center"].to_numpy()
    area = eff_df["eff_area_m2"].to_numpy()

    reco = df[energy_col_reco].dropna().to_numpy()
    N, _ = np.histogram(reco, bins=edges)
    dE = np.diff(edges)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = N.astype(float) / (area * T_sec * dE)
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        # Poisson error on counts -> propagate
        N_err = np.sqrt(np.maximum(N, 1.0))
        flux_err = N_err / (area * T_sec * dE)
        flux_err = np.nan_to_num(flux_err, nan=0.0, posinf=0.0, neginf=0.0)

    out = pd.DataFrame({
        "e_low": edges[:-1],
        "e_high": edges[1:],
        "e_center": centers,
        "N": N,
        "dE": dE,
        "T_sec": T_sec,
        "A_eff_m2": area,
        "flux": flux,
        "flux_err": flux_err,
    })
    return out


def plot_physical_spectrum(
    df_spec: pd.DataFrame,
    out_png: str = "plots/physical_spectrum.png",
    logy: bool = True,
    scale_power: float = 2.4,   # F(E)*E^2.4 by default
):
    """
    Plot the differential spectrum vs energy with error bars.
    """
    # TODO fix hardcoded m2 to mc2 convertion
    E  = df_spec["e_center"].to_numpy()
    F  = df_spec["flux"].to_numpy() / 10**5 # from m2 to cm2 
    dF = df_spec["flux_err"].to_numpy() / 10**5 # from m2 to cm2

    # Scaling F(E) -> F(E)*E^p и ошибок
    if scale_power is not None and float(scale_power) != 0.0:
        F_plot  = F  * (E ** scale_power)
        dF_plot = dF * (E ** scale_power)
        # from [TeV^-1 cm^-2 s^-1] * TeV^p -> TeV^(p-1) cm^-2 s^-1
        y_unit_exp = scale_power - 1.0
        y_label = rf"$F(E)\,E^{{{scale_power:g}}}$, TeV$^{{{y_unit_exp:g}}}$ cm$^{{-2}}$ s$^{{-1}}$"
    else:
        F_plot, dF_plot = F, dF
        y_label = r"$F(E)$, TeV$^{-1}$ cm$^{-2}$ s$^{-1}$"

    fig = plt.figure(figsize=(8,5))
    # if Y is log show only positive
    mask = ~np.isnan(F_plot) & (F_plot > 0 if logy else np.isfinite(F_plot))
    plt.errorbar(E[mask], F_plot[mask], yerr=dF_plot[mask], fmt="o", capsize=2)
 
    plt.xlabel("Энергия, ТэВ")
    plt.ylabel(y_label)
    if (E > 0).any():
        plt.xscale("log")
    if logy and (F_plot[mask] > 0).any():
        plt.yscale("log")

    plt.grid()
    import os; os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
