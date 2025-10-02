
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import COMMON_RANGES

class GammaDataLoader:
    """Encapsulates model/experiment CSV loading and quick-look plots."""
    def __init__(self, model_path: str, exp_path: str):
        self.model_path = model_path
        self.exp_path = exp_path
        self.model_data = None
        self.exp_data = None

    def load_model_data(self, pattern: str = "*.csv") -> pd.DataFrame:
        files = glob.glob(os.path.join(self.model_path, pattern))
        frames = [pd.read_csv(f) for f in tqdm(files, desc="Loading model data")]
        self.model_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.model_data

    def load_experiment_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.exp_path, file_pattern), recursive=True))
        frames = [pd.read_csv(f) for f in tqdm(files, desc="Loading experiment data")]
        self.exp_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.exp_data

    def _hist4(self, df: pd.DataFrame, cols: list[str], title: str, out_png: str):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14)
        for ax, col in zip(axes.flatten(), cols):
            if col in df.columns:
                lo, hi = COMMON_RANGES[col]
                bins = np.linspace(lo, hi, 30)
                ax.hist(df[col], bins=bins, alpha=0.7, edgecolor="black")
                ax.set_title(col)
                ax.set_xlabel(col)
                ax.set_ylabel("N")
                ax.set_xlim(lo, hi)
            else:
                ax.set_title(f"{col} not found")
                ax.axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    def plot_model_pars(self, out_png: str = "plots/pars_hist_model.png"):
        if self.model_data is None:
            raise ValueError("Call load_model_data() first")
        self._hist4(self.model_data, ["size", "dist[0]", "width[0]", "length[0]"], "Model:", out_png)

    def plot_exp_pars(self, out_png: str = "plots/pars_hist_exp.png"):
        if self.exp_data is None:
            raise ValueError("Call load_experiment_data() first")
        self._hist4(self.exp_data, ["size", "dist1", "width", "length"], "Experiment:", out_png)
