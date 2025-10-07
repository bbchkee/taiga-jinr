
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import linear_model, calculate_theta2_classic

class GammaSpectrumReconstructor:
    """Energy reconstruction and selection utilities."""
    def __init__(self, model_data: pd.DataFrame, bins=None, regressor=None):
        self.model_data = model_data
        self.dist_bins = bins if bins is not None else np.arange(0, 5, 0.3)
        self.regressor = regressor
        self.energy_fit_params = {}
        self.cuts = None

# FILTERS & QUALITY CUTS ============================================================

    def apply_good_cut(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Кат на хорошесть данных (шумы, темпы счета, погода) 
        '''
        df = df.copy()
        mask = (
            (df['tracking'] == 1) &
            (df['good'] == 1) &
            (df['edge'] == 0) &
            (df['tel_el'] < 60) &
            (df['tel_el'] > 50) &
            (df['weather_mark'] > 5) &
            (df['star'] == 0) 
        )
        return df[mask]

    @staticmethod
    def filter_model_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = (
            (df['size'] > 120) &
            (df['dist[0]'].between(0.36, 1.44)) &
            (df['width[0]'] > 0.024) &
            (df['width[0]'] < 0.068 * np.log10(df['size']) - 0.047) &
            (df['length[0]'] < 0.145 * np.log10(df['size']))
        )
        return df[mask]

    def add_background_alphas_dists_thetas(self, df: pd.DataFrame) -> pd.DataFrame:
    # TODO добавить расчет для модельных данных;
    # откуда брать x,y ?

        df = df.copy()
        r = np.sqrt(df['source_x']**2 + df['source_y']**2)
        angle0 = np.arctan2(df['source_y'], df['source_x'])
        u_angle = np.arctan(df['a_axis'])  # beware: assumes 'a_axis' ~ tan(angle)

        angles_deg = [0, 180, 60, 120, 240, 300]
        angles_rad = [angle0 + np.deg2rad(a) for a in angles_deg]

        for i, angle in enumerate(angles_rad, start=1):
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            dx = x - df['Xc']
            dy = y - df['Yc']
            dist = np.sqrt(dx**2 + dy**2)
            v_angle = np.arctan2(dy, dx)
            delta = np.abs(v_angle - u_angle)
            delta = np.mod(delta, 2 * np.pi)
            alpha = np.minimum(delta, 2 * np.pi - delta)
            alpha_deg = np.rad2deg(alpha)
            alpha_deg = np.where(alpha_deg > 90, 180 - alpha_deg, alpha_deg)
            df[f'dist{i}'] = dist
            df[f'alpha{i}'] = alpha_deg

        for i in range(1, 7):
            df[f'theta2_{i}'] = df.apply(
                lambda row: calculate_theta2_classic(
                    row.get(f'dist{i}', np.nan),
                    row['size'],
                    row['width'],
                    row['length'],
                    row.get(f'alpha{i}', np.nan)
                ),
                axis=1
            )
        return df

    def filter_experiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO загрузку катов с json
        # и вообще эту канитель надо переделать и слить с classity_by_cuts
        df = df.copy()
        df['gamma_like_1'] = (
            (df['size'] > 120) &
            (df['dist1'].between(0.36, 1.44)) &
            (df['width'] > 0.024) &
            (df['width'] < 0.068 * np.log10(df['size']) - 0.047) &
            (df['length'] < 0.145 * np.log10(df['size']))
        )
        df['gamma_like_2'] = (
            (df['size'] > 120) &
            (df['dist2'].between(0.36, 1.44)) &
            (df['width'] > 0.024) &
            (df['width'] < 0.068 * np.log10(df['size']) - 0.047) &
            (df['length'] < 0.145 * np.log10(df['size']))
        )
        df['theta2'] = df.apply(lambda r: calculate_theta2_classic(r['dist1'], r['size'], r['width'], r['length'], r['alpha1']), axis=1)
        df['theta2_off'] = df.apply(lambda r: calculate_theta2_classic(r['dist2'], r['size'], r['width'], r['length'], r['alpha2']), axis=1)
        return df

    @staticmethod
    def compute_dynamic_cuts(model_data: pd.DataFrame, exp_data: pd.DataFrame) -> dict:
        # TODO квантили аргументами
        # отрисовку гистограмм параметров для модельных и экспериментальных, для дебага
        gamma_width = model_data['width[0]']
        gamma_length = model_data['length[0]']
        bg_width = exp_data['width']
        bg_length = exp_data['length']

        width_min = max(gamma_width.quantile(0.05), bg_width.quantile(0.01))
        width_max = min(gamma_width.quantile(0.95), bg_width.quantile(0.99))
        length_max = min(gamma_length.quantile(0.95), bg_length.quantile(0.99))

        return {'width_min': float(width_min), 'width_max': float(width_max), 'length_max': float(length_max)}

    def tune_cuts(self, exp_data: pd.DataFrame):
        self.cuts = self.compute_dynamic_cuts(self.model_data, exp_data)

# RECO  ============================================================

    def reconstruct_energy(self, exp_data: pd.DataFrame) -> pd.DataFrame:
        if not self.energy_fit_params:
            raise ValueError("Call fit_energy_size() before reconstruct_energy().")
        df = exp_data.copy()
        energies = []
        for _, row in df.iterrows():
            E = np.nan
            for (low, high), p in self.energy_fit_params.items():
                if low <= row.get('dist1', np.nan) < high:
                    E = 10 ** (linear_model(np.log10(row['size']), *p))
                    break
            energies.append(E)
        df['reconstructed_energy'] = energies
        return df

    def reconstruct_energy_for_model_data(self, df_model: pd.DataFrame) -> pd.DataFrame:
        if not self.energy_fit_params:
            raise ValueError("Call fit_energy_size() before reconstruct_energy_for_model_data().")
        df = df_model.copy()
        energies = []
        for _, row in df.iterrows():
            E = np.nan
            for (low, high), p in self.energy_fit_params.items():
                if low <= row.get('dist[0]', np.nan) < high:
                    E = 10 ** (linear_model(np.log10(row['size']), *p))
                    break
            energies.append(E)
        df['reconstructed_energy'] = energies
        return df


# CLASSIFIERS ============================================================

    @staticmethod
    def train_rf_classifier(model_df: pd.DataFrame, exp_df: pd.DataFrame):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils import shuffle
        model_df = model_df.rename(columns={'dist[0]': 'dist', 'width[0]': 'width', 'length[0]': 'length'})
        exp_df = exp_df.rename(columns={'dist1': 'dist'})
        features = ['size', 'width', 'length', 'dist', 'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']
        model_df = model_df[features].dropna()
        exp_df = exp_df[features].dropna()
        model_df['label'] = 1
        exp_df['label'] = 0
        df_all = pd.concat([model_df, exp_df], ignore_index=True)
        df_all = shuffle(df_all, random_state=42)
        X = df_all[features]
        y = df_all['label']
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        return clf
    
    def train_nn_classifier(model_df: pd.DataFrame, exp_df: pd.DataFrame):
        # TODO
        return df
    
    def dump_rf_classifier(self):
        '''dump trained rf classifier '''
        # TODO
        return 0

    def dump_nn_classifier(self):
        '''dump trained nn classifier '''
        # TODO
        return 0

    def classify_by_cuts(self, df: pd.DataFrame, cuts=None) -> pd.DataFrame:
        # TODO load cuts from json
        df = df.copy()
        cuts = cuts or {'width_min': 0.024, 'width_max': 0.068, 'length_max': 0.145}
        for i in range(1, 7):
            dist_col = f'dist{i}'
            gamma_col = f'gamma_like_{i}'
            if dist_col not in df.columns:
                continue
            df[gamma_col] = (
                (df.get('CR_portion', 6) > 5) &
                (df.get('weather_mark', 5) >= 4) &
                (df['size'] > 120) &
                (df[dist_col].between(0.36, 1.44)) &
                (df['width'] > cuts['width_min']) &
                (df['width'] < cuts['width_max'] * np.log10(df['size']) - 0.047) &
                (df['length'] < cuts['length_max'] * np.log10(df['size']))
            )
        return df
    
    @staticmethod
    def classify_by_rf(df: pd.DataFrame, clf, threshold: float = 0.8) -> pd.DataFrame:
        df = df.copy()
        features = ['size', 'width', 'length', 'dist', 'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']
        for i in range(1, 7):
            dist_col = f'dist{i}'
            flag_col = f'gamma_like_{i}'
            cols = ['size', 'width', 'length', dist_col, 'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']
            if not all(c in df.columns for c in cols):
                continue
            temp = df[cols].copy()
            temp.columns = features
            mask = temp.notna().all(axis=1)
            proba = np.zeros(len(df))
            proba[mask] = clf.predict_proba(temp[mask])[:, 1]
            df[flag_col] = proba > threshold
        return df
    
    @staticmethod
    def classify_by_nn(df: pd.DataFrame, clf, threshold: float = 0.8) -> pd.DataFrame:
        # TODO
        return df

    def fit_energy_size(self, out_dir: str = "plots") -> dict:
        filtered = self.model_data.dropna(subset=['size', 'dist[0]', 'energy'])
        params = {}
        for i in range(len(self.dist_bins) - 1):
            lo, hi = self.dist_bins[i], self.dist_bins[i + 1]
            bin_data = filtered[(filtered['dist[0]'] >= lo) & (filtered['dist[0]'] < hi)]
            if len(bin_data) < 3:
                continue
            try:
                x = np.log10(bin_data['size'])
                y = np.log10(bin_data['energy'])
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(linear_model, x, y)
                params[(lo, hi)] = popt
                # Optional preview plot
                fig = plt.figure(figsize=(5,4))
                xs = np.sort(x.values)
                ys = linear_model(xs, *popt)
                plt.plot(x, y, 'o', alpha=0.5, label='data')
                plt.plot(xs, ys, '--', label='fit')
                plt.xlabel('log10(size)'); plt.ylabel('log10(E, TeV)'); plt.grid(True); plt.legend()
                fig.tight_layout()
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(f"{out_dir}/size_fit_{i}.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                # Skip bad bins silently (log outside if needed)
                pass
        self.energy_fit_params = params
        return params

# PLOTTING ============================================================

    def plot_theta2_distribution_with_background_average(self, exp_data: pd.DataFrame, prefix: str = "") -> None:
        bins = np.linspace(0, 1.25, 50)
        bins005 = np.linspace(0, 0.05, 5)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        on = exp_data.get('gamma_like_1', pd.Series(False, index=exp_data.index))
        on_data = exp_data.loc[on, 'theta2_1'].dropna()

        N_on, _ = np.histogram(on_data, bins=bins)
        N_on005, _ = np.histogram(on_data, bins=bins005)

        N_off_all, N_off_all005 = [], []
        for i in range(2, 7):
            theta_col = f'theta2_{i}'
            flag_col = f'gamma_like_{i}'
            if theta_col not in exp_data.columns or flag_col not in exp_data.columns:
                continue
            data = exp_data.loc[exp_data[flag_col], theta_col].dropna()
            h, _ = np.histogram(data, bins=bins)
            h005, _ = np.histogram(data, bins=bins005)
            N_off_all.append(h); N_off_all005.append(h005)

        if not N_off_all:
            return

        N_off_avg = np.mean(N_off_all, axis=0)
        # Plot ON vs averaged OFF
        #import matplotlib.pyplot as plt
        bw = (bins[1] - bins[0]) * 0.5
        fig = plt.figure(figsize=(8,5))
        plt.bar(bin_centers - bw/2, N_on, width=bw, label='source')
        plt.bar(bin_centers + bw/2, N_off_avg, width=bw, label='background avg (5 pts)')
        plt.xlabel(r"$\\theta^2$ (deg$^2$)"); plt.ylabel('Counts'); plt.legend(); plt.grid(True, alpha=0.3)
        fig.tight_layout()
        os.makedirs('plots', exist_ok=True)
        fig.savefig(f'plots/theta2_comp_{prefix}.png', dpi=150)
        plt.close(fig)

        # Excess + Li&Ma significance (alpha = 1/5)
        alpha = 1/5.0
        N_on_total = N_on005.sum()
        N_off_total = np.sum(N_off_all005)
        # Li & Ma Eq. 17 (1983)
        S = np.nan
        if N_on_total > 0 and N_off_total > 0:
            term1 = N_on_total * np.log((1 + alpha) / alpha * (N_on_total / (N_on_total + N_off_total)))
            term2 = N_off_total * np.log((1 + alpha) * (N_off_total / (N_on_total + N_off_total)))
            S = float(np.sqrt(2 * (term1 + term2)))

        fig = plt.figure(figsize=(8,5))
        diff = N_on - N_off_avg
        plt.plot(bin_centers, diff, marker='o', linestyle='-', label='N_on - N_off')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel(r"$\\theta^2$"); plt.ylabel('N_on - N_off'); plt.title('Excess (ON - OFF)')
        plt.grid(True, linestyle=":", alpha=0.6); plt.legend()
        plt.figtext(0.15, 0.01, f"N_on={N_on_total}, N_off={N_off_total*alpha:.1f}, S={S:.2f}")
        fig.tight_layout()
        fig.savefig(f'plots/theta_diff_{prefix}.png', dpi=150)
        plt.close(fig)
