
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
    ): # (:
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        df = self.exp_data
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

        plt.figtext(0.14, 0.01, f"N_on = {N_on_sig}, N_off = {N_off_sig_total:.0f} (α={alpha:.2f}), S = {np.nan if np.isnan(S) else round(S,2)}")
        fig.tight_layout()
        fig.savefig(out_png_diff, dpi=150)
        plt.close(fig)
