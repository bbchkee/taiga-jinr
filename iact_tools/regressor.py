# iact_tools/nn_regressor.py
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split

try:
    # optional pretty pictures
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---- column aliases: model vs experiment ----
MODEL_ALIASES = {
    "size":   ["size", "hillas_size", "SIZE"],
    "dist":   ["dist[0]", "dist", "dist0"],
    "width":  ["width[0]", "width"],
    "length": ["length[0]", "length"],
    "numpix": ["numb_pix", "num_pix", "numpix"],
    "energy": ["energy", "true_energy", "mc_energy", "E"],
    "erec":   ["reconstructed_energy"],
}
EXP_ALIASES = {
    "size":   ["size", "hillas_size", "SIZE"],
    "dist":   ["dist1", "dist", "dist[0]"],
    "width":  ["width"],
    "length": ["length"],
    "numpix": ["numb_pix", "num_pix", "numpix"],
    "energy": ["energy"],  # only in model
    "erec":   ["reconstructed_energy"],
}

def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


class _MLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


class GammaNNRegressor:
    """
    Neural regressor for energy reconstruction:
      inputs (subset): size, dist, width, length, numb_pix
      target: true energy (TeV), learning on model data after cuts.
    - scalins X and Y (StandardScaler)
    - device: cuda/CPU auto
    - give methods fit_on_model(...) и predict_for_df(...)

    Parameters:
      use_dist/use_wl/use_npix — use features if exists in data
      test_size/seed — train/val separation
    """
    def __init__(self,
                 use_dist: bool = True,
                 use_wl: bool = True,
                 use_npix: bool = True,
                 model_path: str = "gamma_regressor.pth",
                 seed: int = 42):
        self.use_dist = use_dist
        self.use_wl = use_wl
        self.use_npix = use_npix
        self.model_path = model_path
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model: Optional[_MLP] = None

        # diagnostics
        self._train_loader = None
        self._test_loader  = None
        self._df_cached: Optional[pd.DataFrame] = None
        self._test_index = None
        self._feat_order: List[str] = []

    # ---------- columns and features ----------
    @staticmethod
    def _normalize_columns(df: pd.DataFrame, model_like: bool) -> pd.DataFrame:
        alias = MODEL_ALIASES if model_like else EXP_ALIASES
        out = df.copy()
        # mapping to standard names
        mapping = {}
        for std, pool in alias.items():
            src = _pick(out, pool)
            if src:
                mapping[src] = std
        out = out.rename(columns=mapping)
        return out

    def _features_from_df(self, df: pd.DataFrame, model_like: bool, dropna=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        dfn = self._normalize_columns(df, model_like=model_like)

        needed = ["size"]
        if self.use_dist:  needed += ["dist"]
        if self.use_wl:    needed += ["width", "length"]
        if self.use_npix:  needed += ["numpix"]

        ycol = "energy" if model_like else None

        cols_present = [c for c in needed if c in dfn.columns]
        if dropna:
            dfn = dfn.replace([np.inf, -np.inf], np.nan)
            keep = ["size"] + ([ycol] if ycol else []) + cols_present
            dfn = dfn.dropna(subset=[c for c in keep if c in dfn.columns])

        # size>0
        if "size" in dfn.columns:
            dfn = dfn[dfn["size"] > 0]

        Xcols = []
        self._feat_order = []
        # log(size) to make it more "linear" feature
        Xcols.append(np.log10(dfn["size"].to_numpy().astype(np.float32)))
        self._feat_order.append("log10(size)")

        if self.use_dist and "dist" in dfn.columns:
            Xcols.append(dfn["dist"].to_numpy().astype(np.float32))
            self._feat_order.append("dist")

        if self.use_wl:
            if "width" in dfn.columns:
                Xcols.append(dfn["width"].to_numpy().astype(np.float32))
                self._feat_order.append("width")
            if "length" in dfn.columns:
                Xcols.append(dfn["length"].to_numpy().astype(np.float32))
                self._feat_order.append("length")

        if self.use_npix and "numpix" in dfn.columns:
            Xcols.append(dfn["numpix"].to_numpy().astype(np.float32))
            self._feat_order.append("numpix")

        if not Xcols:
            raise RuntimeError("Нет доступных признаков (size/dist/width/length/numb_pix)")

        X = np.column_stack(Xcols).astype(np.float32)
        y = None
        if ycol and ycol in dfn.columns:
            y = dfn[ycol].to_numpy().astype(np.float32).reshape(-1, 1)

        return X, y

    # ---------- LEARNING ----------
    def fit_on_model(self,
                     model_df: pd.DataFrame,
                     epochs: int = 200,
                     batch_size: int = 256,
                     lr: float = 3e-4,
                     patience: int = 10,
                     val_size: float = 0.2,
                     out_dir: Optional[str] = "fit_plots",
                     verbose: bool = True,
                     balance_bins: int = 20,
                     balance_clip: float = 0.99):
        """
        Обучение на МОДЕЛЬНЫХ данных с авто-балансировкой по log10(E_true):
          - строим бины по log10(E_true)
          - задаём веса w_i ~ 1/freq(bin_i) (клиппинг по quantile=balance_clip)
          - train/val делим стратифицированно по бину
          - для train используем WeightedRandomSampler (replacement=True)

        Параметры balance_* позволяют подкрутить сглаживание.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_raw, y_raw = self._features_from_df(model_df, model_like=True, dropna=True)
        if y_raw is None or len(y_raw) < 10:
            raise RuntimeError("В модельных данных нет 'energy' или слишком мало событий для обучения.")

        # --- Балансировка по log10(E_true) ДО скейлинга
        y_true = y_raw.reshape(-1)  # в TeV
        y_log10 = np.log10(np.clip(y_true, 1e-30, None))

        # Бины (равномерные по logE)
        lo, hi = np.min(y_log10), np.max(y_log10)
        edges = np.linspace(lo, hi, balance_bins + 1)
        bins = np.digitize(y_log10, edges) - 1  # 0..bins-1
        bins = np.clip(bins, 0, balance_bins - 1)

        # Частоты и веса
        counts = np.bincount(bins, minlength=balance_bins).astype(float)
        counts[counts == 0] = 1.0
        weights = 1.0 / counts[bins]
        # нормировка и клиппинг хвостов
        med = np.median(weights)
        if med > 0:
            weights = weights / med
        if 0 < balance_clip < 1:
            thr = np.quantile(weights, balance_clip)
            weights = np.clip(weights, None, thr)

        # --- scaling (X и y в стандартах модели)
        X = self.scaler_X.fit_transform(X_raw)
        y = self.scaler_y.fit_transform(y_raw)

        # --- train/val split: стратифицированно по бину logE
        idx = np.arange(len(X))
        idx_tr, idx_te = train_test_split(
            idx, test_size=val_size, random_state=self.seed, stratify=bins
        )
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]
        w_tr = weights[idx_tr]  # только train-веса

        # --- DataLoaders
        # train: взвешенная выборка по WeightedRandomSampler
        tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(w_tr, dtype=torch.double),
            num_samples=len(X_tr),
            replacement=True
        )
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)

        # val: обычный DataLoader без весов (чтобы метрика была честной)
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
            batch_size=batch_size, shuffle=False
        )

        self._train_loader, self._test_loader = tr_loader, te_loader
        self._df_cached = model_df.copy()
        self._test_index = None

        # --- модель/оптимизатор/шедулер
        self.model = _MLP(in_features=X.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best = np.inf; bad = 0
        self.model.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total += loss.item()

            # валидация
            val = 0.0
            self.model.eval()
            with torch.no_grad():
                for xb, yb in te_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val += criterion(pred, yb).item()
            self.model.train()

            avg_tr = total / max(1, len(tr_loader))
            avg_te = val / max(1, len(te_loader))
            scheduler.step(avg_te)

            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"[NN] epoch {ep:03d}  train={avg_tr:.5f}  val={avg_te:.5f}")

            if avg_te < best - 1e-6:
                best = avg_te; bad = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"[NN] early stop at epoch {ep}, best val={best:.5f}")
                    break

        # загрузим лучшую
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)

        if out_dir is not None and plt is not None:
            os.makedirs(out_dir, exist_ok=True)
            self._plot_val_plane(out_dir)

    # ---------- applying to some DF ----------
    def predict_for_df(self, df: pd.DataFrame, model_like: bool) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Модель не обучена. Вызовите fit_on_model(...) или load_model().")
        X_raw, _ = self._features_from_df(df, model_like=model_like, dropna=False)
        # if size are absent/NaN - return NaN
        mask_valid = np.isfinite(X_raw).all(axis=1)
        X = np.zeros_like(X_raw, dtype=np.float32)
        y_pred = np.full(len(df), np.nan, dtype=np.float32)
        if np.any(mask_valid):
            X[mask_valid] = self.scaler_X.transform(X_raw[mask_valid])
            with torch.no_grad():
                tX = torch.tensor(X[mask_valid], dtype=torch.float32, device=self.device)
                y = self.model(tX).cpu().numpy()
                y = self.scaler_y.inverse_transform(y).reshape(-1)
            y_pred[mask_valid] = y
        return pd.Series(y_pred, index=df.index, name="reconstructed_energy")

    # ---------- utilities ----------
    def load_model(self):
        self.model = _MLP(in_features=len(self._feat_order) if self._feat_order else 5)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)

    def _plot_val_plane(self, out_dir: str):
        if self._test_loader is None or plt is None or sns is None:
            return
        true_E, pred_E = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in self._test_loader:
                xb = xb.to(self.device)
                pred = self.model(xb).cpu().numpy()
                pred = self.scaler_y.inverse_transform(pred)
                yb = self.scaler_y.inverse_transform(yb.numpy())
                true_E.extend(yb.flatten()); pred_E.extend(pred.flatten())
        bins = np.linspace(0, 200, 41)
        fig = plt.figure(figsize=(7,6))
        sns.histplot(x=np.array(true_E), y=np.array(pred_E), bins=[bins, bins], cbar=True)
        plt.xlabel("True E (TeV)"); plt.ylabel("Pred E (TeV)")
        plt.xlim(0,200); plt.ylim(0,200); plt.grid(True, ls="--", alpha=0.4)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nn_energy_true_vs_pred.png"), dpi=150); plt.close(fig)
