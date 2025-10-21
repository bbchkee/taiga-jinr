import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Any, Optional
from .models import linear_model, calculate_theta2_classic
from scipy.optimize import curve_fit

def linear_model(x, a, b):
    """Линейная модель для фитирования зависимости size vs energy."""
    return x * a + b

# ------------------------------------------------------------------
# Shared, synchronized energy-bin generator (using everywhere)
# ------------------------------------------------------------------
def fixed_energy_bins(n_bins: int = 20, e_min_tev: float = 1.0, e_max_tev: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (edges, centers) — логарифмически равномерные бины от e_min до e_max.
    Эти бины должны использоваться последовательно в:
      - эффективной площади,
      - матрице отклика (true и reco оси),
      - анфолдинге,
      - отрисовке спектров.
    """
    edges = np.logspace(np.log10(e_min_tev), np.log10(e_max_tev), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers

def _pick_first_col(df, candidates):
    """Вернуть имя первой существующей колонки из candidates, иначе None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _build_dist_bins_from_params(params_dict):
    """
    Из self.energy_fit_params вида {(lo,hi): {...}} собрать отсортированный список бинов.
    Возвращает список [(lo,hi,params), ...] отсортированный по lo.
    """
    bins = []
    for (lo, hi), p in params_dict.items():
        bins.append((float(lo), float(hi), p))
    bins.sort(key=lambda t: t[0])
    return bins

def _flat_logE_weights(y_log10: np.ndarray, bins: int = 20, clip_q: float = 0.99) -> np.ndarray:
    """Сделать веса так, чтобы вклад был примерно равномерным по log10(E)."""
    y = y_log10[np.isfinite(y_log10)]
    if len(y) < 5:
        return np.ones_like(y_log10)
    hist, edges = np.histogram(y, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # оценка частоты в точке y_i
    freq = np.interp(y_log10, centers, np.maximum(hist, 1))
    w = 1.0 / freq
    # нормировка и клиппинг
    med = np.median(w[np.isfinite(w)]) if np.any(np.isfinite(w)) else 1.0
    w = w / (med if med > 0 else 1.0)
    if 0 < clip_q < 1:
        thr = np.quantile(w[np.isfinite(w)], clip_q)
        w = np.minimum(w, thr)
    return np.where(np.isfinite(w), w, 1.0)

def _apply_energy_reco_to_df(self, df: pd.DataFrame, dist_col_candidates: List[str], clamp_outside=True) -> pd.DataFrame:
    if not self.energy_fit_params:
        raise ValueError("Call fit_energy_size() before reconstruct_energy*().")
    out = df.copy()
    dist_col = _pick_first_col(out, dist_col_candidates)
    if dist_col is None or 'size' not in out.columns:
        out["reconstructed_energy"] = np.nan
        return out

    size = out["size"].to_numpy(copy=False)
    dist = out[dist_col].to_numpy(copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(np.where(size>0, size, np.nan))

    E_rec = np.full(len(out), np.nan, dtype=float)
    bins = _build_dist_bins_from_params(self.energy_fit_params)
    if not bins:
        out["reconstructed_energy"] = E_rec
        return out

    lo_edges = np.array([b[0] for b in bins], dtype=float)
    hi_edges = np.array([b[1] for b in bins], dtype=float)

    for (lo, hi, p) in bins:
        if isinstance(p, (tuple, list)) and len(p)==2:
            a, b = float(p[0]), float(p[1]); c, mu, sd = 0.0, 0.0, 1.0
        else:
            a = float(p.get("a",0.0)); b = float(p.get("b",0.0)); c = float(p.get("c",0.0))
            mu = float(p.get("mu_dist",0.0)); sd = float(p.get("sigma_dist",1.0)) or 1.0

        mask = (dist>=lo) & (dist<hi) & np.isfinite(x)
        if not np.any(mask): continue
        z = (dist[mask]-mu)/sd
        logE = a*x[mask] + c*z + b
        E_rec[mask] = np.power(10.0, logE)

    if clamp_outside:
        outside = (np.isfinite(x)) & ~np.isfinite(E_rec)
        if np.any(outside):
            centers = 0.5*(lo_edges+hi_edges)
            idx = np.argmin(np.abs(dist[outside,None]-centers[None,:]), axis=1)
            for k, j in enumerate(idx):
                p = bins[j][2]
                if isinstance(p,(tuple,list)) and len(p)==2:
                    a, b = float(p[0]), float(p[1]); c, mu, sd = 0.0,0.0,1.0
                else:
                    a = float(p.get("a",0.0)); b=float(p.get("b",0.0)); c=float(p.get("c",0.0))
                    mu=float(p.get("mu_dist",0.0)); sd=float(p.get("sigma_dist",1.0)) or 1.0
                xx = x[outside][k]; dd = dist[outside][k]
                if np.isfinite(xx):
                    z = (dd-mu)/sd
                    E_rec[outside][k] = 10.0**(a*xx + c*z + b)

    # пост-калибровка gain(log10 E_rec), если задана
    if self._gain_nodes is not None and self._gain_vals is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.log10(np.clip(E_rec, 1e-30, None))
        g = np.interp(t, self._gain_nodes, self._gain_vals, left=1.0, right=1.0)
        E_rec = E_rec * g

    out["reconstructed_energy"] = E_rec
    return out


class GammaSpectrumReconstructor:
    """Energy reconstruction and selection utilities."""
    def __init__(self, model_data: pd.DataFrame, bins=None, regressor=None,
                 energy_bins: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.model_data = model_data
        self.dist_bins = bins if bins is not None else np.arange(0, 2.0, 0.3)
        self.regressor = regressor
        self.energy_fit_params: Dict[Tuple[float,float], Any] = {}
        self.cuts = None
        # синхронизированные бины энергии (используй их в R/eff/unfold)
        self.energy_edges, self.energy_centers = energy_bins if energy_bins is not None else fixed_energy_bins()
        # опциональная пост-калибровка (gain(log10 E_rec))
        self._gain_nodes: Optional[np.ndarray] = None
        self._gain_vals: Optional[np.ndarray] = None
        self.energy_fit_params = None
        self.regressor = None

# FILTERS & QUALITY CUTS ============================================================

    def apply_good_cut(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Кат на хорошесть данных (шумы, темпы счета, погода) 
        '''
        df = df.copy()
        mask = (
            (df['tracking'] == 1) &
            (df['good'] == 1) &
            (df['edge'] == 0) &
            #(df['dist1'] < 1.3) &
            (df['CR_portion'] > 5) &
            (df['tel_el'] < 60) &
            (df['tel_el'] > 50) &
            (df['weather_mark'] > 9) &
            (df['star'] == 0) 
        )
        return df[mask]

    @staticmethod
    def filter_model_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = (
            (df['size'] > 120) &
            #(df['dist[0]'] < 1.3) &
            #(df.get('weather_mark', 5) >= 4) &
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
    
    def reconstruct_energy(self, exp_data):
        """Восстановление энергии для экспериментальных данных с учетом бинов dist[0]."""
        reconstructed_energy = []
        for _, row in exp_data.iterrows():
            for (low, high), params in self.energy_fit_params.items():
                if low <= row['dist1'] < high:
                    reconstructed_energy.append(10**(linear_model(np.log10(row['size']), *params)))
                    break
            else:
                reconstructed_energy.append(np.nan)
        exp_data['reconstructed_energy'] = reconstructed_energy
        return exp_data
        
    def reconstruct_energy_for_model_data(self, exp_data):
        """Восстановление энергии для экспериментальных данных с учетом бинов dist[0]."""
        reconstructed_energy = []
        for _, row in exp_data.iterrows():
            for (low, high), params in self.energy_fit_params.items():
                if low <= row['dist[0]'] < high:
                    reconstructed_energy.append(10**(linear_model(np.log10(row['size']), *params)))
                    break
            else:
                reconstructed_energy.append(np.nan)
        exp_data['reconstructed_energy'] = reconstructed_energy
        return exp_data
        
    def reconstruct_energy_complex(self, exp_data: pd.DataFrame) -> pd.DataFrame:
        """
        Восстановление энергии для ЭКСПЕРИМЕНТАЛЬНЫХ данных с учётом новых фит-параметров.
        Используются параметры по бинам dist: (a,b,c,mu_dist,sigma_dist).
        Для совместимости со старыми параметрами (a,b) член c*z просто будет равен 0.

        Алгоритм:
          - берём dist-колонку из набора кандидатов ['dist1','dist','dist[0]'] что найдётся в df,
          - считаем x=log10(size), z=(dist - mu_dist)/sigma_dist для соответствующего бина,
          - logE = a*x + c*z + b,  E = 10**logE.
          - события вне диапазона биннинга по dist — клэмпим к ближайшему бину (настраивается).
        """
        return _apply_energy_reco_to_df(self, exp_data, dist_col_candidates=['dist1','dist','dist[0]'], clamp_outside=True)

    def reconstruct_energy_for_model_data_complex(self, df_model: pd.DataFrame) -> pd.DataFrame:
        """
        Восстановление энергии для МОДЕЛЬНЫХ данных (обычно есть 'dist[0]').
        Совместимо с новыми (a,b,c,mu_dist,sigma_dist) и старыми (a,b) параметрами.
        """
        return _apply_energy_reco_to_df(self, df_model, dist_col_candidates=['dist[0]','dist','dist0','dist1'], clamp_outside=True)


# CLASSIFIERS ============================================================

    @staticmethod
    def train_rf_classifier(model_df: pd.DataFrame, exp_df: pd.DataFrame):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils import shuffle

        # Приведение имён колонок к одному виду
        model_df = model_df.rename(columns={
            'dist[0]': 'dist',
            'width[0]': 'width',
            'length[0]': 'length'
        })
        exp_df = exp_df.rename(columns={'dist1': 'dist'})

        features = [
            'size', 'width', 'length', 'dist',
            'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w'
        ]

        model_df = model_df[features].dropna()
        exp_df = exp_df[features].dropna()

        # ---------- 🔧 Автобалансировка ----------
        n_model = len(model_df)
        n_exp = len(exp_df)

        if n_exp > n_model:
            exp_df = exp_df.sample(n=n_model, random_state=42)
        elif n_model > n_exp:
            # опционально: можно undersample model_df, если хочешь строгий баланс
            model_df = model_df.sample(n=n_exp, random_state=42)
        # ---------------------------------------

        # Добавляем метки классов
        model_df['label'] = 1
        exp_df['label'] = 0

        # Объединяем и перемешиваем
        df_all = pd.concat([model_df, exp_df], ignore_index=True)
        df_all = shuffle(df_all, random_state=42)

        X = df_all[features]
        y = df_all['label']

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
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
                #(df.get('CR_portion', 6) > 5) &
                #(df.get('weather_mark', 5) >= 4) &
                (df['size'] > 120) &
                (df[dist_col].between(0.36, 1.44)) &
                (df['width'] > cuts['width_min']) &
                (df['width'] < cuts['width_max'] * np.log10(df['size']) - 0.047) &
                (df['length'] < cuts['length_max'] * np.log10(df['size']))
            )
        return df
    
    #@staticmethod
    def classify_by_rf(self, df: pd.DataFrame, clf, threshold: float = 0.8) -> pd.DataFrame:
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

    def fit_energy_size(self, out_dir = 'plots'):
        filtered_data = self.model_data
        fit_params = {}
        for i in range(len(self.dist_bins) - 1):
            bin_mask = (filtered_data['dist[0]'] >= self.dist_bins[i]) & (filtered_data['dist[0]'] < self.dist_bins[i + 1])
            bin_data = filtered_data[bin_mask]
                
            if len(bin_data) >= 3:  # ✅ хотя бы 3 точки (на всякий случай)
                try:
                    x = np.log10(bin_data['size'])
                    y = np.log10(bin_data['energy'])
                    popt, _ = curve_fit(linear_model, x, y)
                    fit_params[(self.dist_bins[i], self.dist_bins[i + 1])] = popt

                    # График
                    plt.plot(x, y, 'o', alpha=0.5, label='s-e data')
                    plt.plot(np.sort(x), linear_model(np.sort(x), *popt), linestyle='--')
                    plt.xlabel('log10(size)')
                    plt.ylabel('log10(E, TeV)')
                    plt.title(f'dist {self.dist_bins[i]:.2f} - {self.dist_bins[i+1]:.2f}')
                    plt.xlim(2, 4.5)
                    plt.ylim(0, 2.5)
                    plt.grid()
                    plt.savefig(out_dir+f'/size_fit_{i}.png')
                    plt.clf()
                except Exception as e:
                    print(f"[!] Ошибка фита в бинe {self.dist_bins[i]}–{self.dist_bins[i+1]}: {e}")
            else:
                print(f"[!] Пропускаем бин {self.dist_bins[i]}–{self.dist_bins[i+1]}: мало точек ({len(bin_data)})")
            self.energy_fit_params = fit_params
        return fit_params


    def fit_energy_size_complex(
        self,
        out_dir: str = "plots",
        verbose: bool = True,
        weighting: str = "flat_logE",  # 'flat_logE' | 'powerlaw' | 'none'
        spectral_index: float = 2.6,   # для weighting='powerlaw' (E^{-gamma})
        use_dist_term: bool = True,
        use_binned_regression: bool = True,
        size_bins_per_dist: int = 12,
        min_points_bin: int = 20,
        weight_clip_q: float = 0.99,
    ) -> dict:
        """
        Калибровка log10(E) по log10(size) и (опц.) по dist:
            log10(E_TeV) = a*log10(size) + c*z + b, z=(dist-mu)/sigma.

        Устойчивость:
          • веса по умолчанию выравнивают вклад по log10(E) (weighting='flat_logE') —
            это устраняет расплытие из-за смеси спектров;
          • опц. медианные «гист-точки» по log10(size);
          • нормировка dist в бине;
          • логирование и PNG-диагностика в out_dir/size_fit_<i>.png.

        Возвращает self.energy_fit_params: {(lo,hi): {'a','b','c','mu_dist','sigma_dist','n'}}.
        """
        os.makedirs(out_dir, exist_ok=True)

        need = ['size', 'dist[0]', 'energy']
        if any(c not in self.model_data.columns for c in need):
            raise KeyError(f"fit_energy_size: отсутствуют {need}. Есть: {list(self.model_data.columns)}")

        df = self.model_data.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
        df = df[(df['size']>0) & (df['energy']>0)]
        if verbose:
            print(f"[fit_energy_size] rows after filter: {len(df)}")

        params: Dict[Tuple[float,float], Any] = {}
        saved = 0

        for i in range(len(self.dist_bins)-1):
            lo, hi = self.dist_bins[i], self.dist_bins[i+1]
            dd = df[(df['dist[0]']>=lo) & (df['dist[0]']<hi)].copy()
            if len(dd) < max(10, min_points_bin):
                if verbose: print(f"[fit_energy_size] bin {i} [{lo},{hi}): too few rows ({len(dd)}), skip")
                continue

            x_all = np.log10(dd['size'].to_numpy())
            y_all = np.log10(dd['energy'].to_numpy())
            d_all = dd['dist[0]'].to_numpy()
            m = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(d_all)
            x_all, y_all, d_all = x_all[m], y_all[m], d_all[m]
            if len(x_all) < max(10, min_points_bin):
                if verbose: print(f"[fit_energy_size] bin {i}: finite rows < {min_points_bin}, skip")
                continue

            mu_d = float(np.mean(d_all))
            sd_d = float(np.std(d_all)) if np.std(d_all)>0 else 1.0
            z_all = (d_all - mu_d)/sd_d

            # веса
            if weighting == "powerlaw":
                w_all = (10.0**y_all) ** (-float(spectral_index))
            elif weighting == "flat_logE":
                w_all = _flat_logE_weights(y_all, bins=20, clip_q=weight_clip_q)
            else:
                w_all = np.ones_like(y_all)
            w_all = np.where(np.isfinite(w_all), w_all, 1.0)

            # медианные точки по log10(size)
            if use_binned_regression:
                edges = np.linspace(np.min(x_all), np.max(x_all), size_bins_per_dist+1)
                xm, ym, zm, wm = [], [], [], []
                for j in range(len(edges)-1):
                    sel = (x_all>=edges[j]) & (x_all<edges[j+1])
                    if np.sum(sel) < 5: continue
                    xm.append(np.median(x_all[sel]))
                    ym.append(np.median(y_all[sel]))
                    zm.append(np.median(z_all[sel]))
                    wm.append(np.median(w_all[sel]))
                x_fit = np.array(xm); y_fit = np.array(ym); z_fit = np.array(zm); w_fit = np.array(wm)
                if len(x_fit) < 5:
                    x_fit, y_fit, z_fit, w_fit = x_all, y_all, z_all, w_all
            else:
                x_fit, y_fit, z_fit, w_fit = x_all, y_all, z_all, w_all

            # взвешенная лин. регрессия
            if use_dist_term:
                X = np.column_stack([x_fit, z_fit, np.ones_like(x_fit)])
            else:
                X = np.column_stack([x_fit, np.ones_like(x_fit)])
            sw = np.sqrt(np.clip(w_fit, 1e-12, None))
            Xw = X * sw[:,None]
            yw = y_fit * sw
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            if use_dist_term:
                a, c, b = map(float, beta)
            else:
                a, b = map(float, beta); c = 0.0

            params[(lo,hi)] = {"a":a, "b":b, "c":c, "mu_dist":mu_d, "sigma_dist":sd_d, "n":int(len(x_all))}

            # диаграмма
            try:
                fig = plt.figure(figsize=(6,5))
                xs = np.linspace(np.min(x_all), np.max(x_all), 100)
                ys = a*xs + b
                plt.plot(x_all, y_all, 'o', alpha=0.35, label=f"data (N={len(x_all)})")
                if use_binned_regression and len(x_fit)!=len(x_all):
                    plt.plot(x_fit, y_fit, 's', alpha=0.9, label='medians by size-bin')
                label = f"fit: y={a:.3f}x + {b:.3f}" + (f" + {c:.3f}·z" if use_dist_term else "")
                plt.plot(xs, ys, '--', label=label)
                plt.xlabel("log10(size)"); plt.ylabel("log10(E, TeV)")
                plt.title(f"dist in [{lo}, {hi}) (μ={mu_d:.2f}, σ={sd_d:.2f}), weighting={weighting}")
                plt.grid(True, alpha=0.5); plt.legend()
                fig.tight_layout()
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(os.path.join(out_dir, f"size_fit_{i}.png"), dpi=150)
                plt.close(fig); saved += 1
            except Exception as e:
                if verbose: print(f"[fit_energy_size] plot failed in bin {i}: {e}")

        if verbose:
            print(f"[fit_energy_size] fitted bins: {len(params)}; saved plots: {saved}")

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
