
"""
Effective area computation for IACT pipeline.

This module adds functions to compute the effective area A_eff(E) from:
- a "sums" file with *all thrown* MC energies (unbiased, pre-trigger), and
- a model CSV with *passed* (e.g., triggered & selected) events.

We assume MC events were thrown uniformly over a disk of radius R (meters).
Then:  A_eff(E_i) = (N_passed(E_i) / N_thrown(E_i)) * (pi * R^2).

Author: ChatGPT (assistant)
"""

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable
import warnings
from pathlib import Path
import glob

Number = Union[int, float]
PathLike = Union[str, os.PathLike]

DEFAULT_RADIUS_M = 500.0  # meters (feel free to adjust or make configurable)


def _to_energy_array(obj: Union[pd.DataFrame, pd.Series, Iterable[Number], np.ndarray],
                     energy_col: str = "energy") -> np.ndarray:
    """
    Convert common inputs into a 1D numpy array of energies.
    - If DataFrame: use `energy_col`.
    - If Series / ndarray / list: convert to 1D array directly.
    """
    if isinstance(obj, pd.DataFrame):
        if energy_col not in obj.columns:
            raise KeyError(f"Energy column '{energy_col}' not found in DataFrame columns: {list(obj.columns)}")
        arr = obj[energy_col].to_numpy(dtype=float)
    elif isinstance(obj, pd.Series):
        arr = obj.to_numpy(dtype=float)
    else:
        arr = np.asarray(list(obj), dtype=float)
    return arr.reshape(-1)


def _load_sums_energies(sums_path: PathLike, energy_col: str | None = "energy") -> np.ndarray:
    """
    Загружает энергии из файла 'sums'. Поддерживаются форматы:
    1) CSV/TSV с колонкой (по умолчанию 'energy').
    2) Текстовые файлы с одной или несколькими колонками (берётся первая).
    
    Возвращает: 1D numpy array энергий.
    """
    sums_path = str(sums_path)
    _, ext = os.path.splitext(sums_path.lower())

    if ext in (".csv", ".tsv", ".txt", ".dat"):
        # Пробуем pandas
        try:
            sep = "," if ext == ".csv" else None  # автоопределение
            df = pd.read_csv(sums_path, sep=sep, engine="python")

            # Если есть заданная колонка
            if energy_col and energy_col in df.columns:
                return df[energy_col].to_numpy(dtype=float).reshape(-1) / 10**12

            # Если нет — используем первую числовую
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].to_numpy(dtype=float).reshape(-1) / 10**12

            # Если нет числовых колонок — берём первую колонку вообще
            if len(df.columns) > 0:
                return df.iloc[:, 0].to_numpy(dtype=float).reshape(-1) / 10**12

        except Exception:
            pass  # fallback на numpy

        # Fallback: plain текст
        try:
            arr = np.loadtxt(sums_path, dtype=float)
            # Если файл 2D — берём первую колонку
            if arr.ndim > 1:
                arr = arr[:, 0]
            return np.asarray(arr, dtype=float).reshape(-1) / 10**12
        except Exception as e:
            raise ValueError(f"Не удалось прочитать файл sums '{sums_path}': {e}")

    else:
        # Неизвестное расширение — пробуем loadtxt
        try:
            arr = np.loadtxt(sums_path, dtype=float)
            if arr.ndim > 1:
                arr = arr[:, 0]
            return np.asarray(arr, dtype=float).reshape(-1) / 10**12
        except Exception as e:
            raise ValueError(f"Не удалось прочитать файл sums '{sums_path}': {e}")



def _load_model_energies(model_path_or_df: Union[PathLike, pd.DataFrame],
                         energy_col: str = "energy") -> np.ndarray:
    """
    Load energies of PASSED (triggered/selected) events from model CSV or DataFrame.
    """
    if isinstance(model_path_or_df, pd.DataFrame):
        return _to_energy_array(model_path_or_df, energy_col=energy_col)
    else:
        model_path = str(model_path_or_df)
        df = pd.read_csv(model_path)
        if energy_col not in df.columns:
            # try a couple of common aliases
            for alt in ("mc_energy", "true_energy", "E", "Energy", "ENERGY"):
                if alt in df.columns:
                    energy_col = alt
                    break
        if energy_col not in df.columns:
            raise KeyError(f"Не найден столбец энергии '{energy_col}' в '{model_path}'. Доступные колонки: {list(df.columns)}")
        return df[energy_col].to_numpy(dtype=float).reshape(-1)


def compute_effective_area(
    sums_energies: Union[np.ndarray, Iterable[Number], pd.Series, pd.DataFrame],
    passed_energies: Union[np.ndarray, Iterable[Number], pd.Series, pd.DataFrame],
    n_bins: int = 20,
    radius_m: float = DEFAULT_RADIUS_M,
    energy_col: str = "energy",
    energy_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute effective area from thrown (sums) and passed energies.

    Parameters
    ----------
    sums_energies : array-like or DataFrame
        Energies of ALL thrown MC events (pre-trigger). If DataFrame, uses `energy_col`.
    passed_energies : array-like or DataFrame
        Energies of PASSED events (post-trigger/selection). If DataFrame, uses `energy_col`.
    n_bins : int
        Number of logarithmic bins.
    radius_m : float
        Radius of the uniform throw disk (in meters).
    energy_col : str
        Name of energy column if inputs are DataFrames.
    energy_range : (emin, emax), optional
        If provided, binning range (in same units as energies). If None, auto from data.

    Returns
    -------
    bin_edges : np.ndarray shape (n_bins+1,)
    bin_centers : np.ndarray shape (n_bins,)
    total_counts : np.ndarray shape (n_bins,)
    passed_counts : np.ndarray shape (n_bins,)
    eff_area_m2 : np.ndarray shape (n_bins,)
    """
    E_all = _to_energy_array(sums_energies, energy_col=energy_col)
    E_pass = _to_energy_array(passed_energies, energy_col=energy_col)

    if energy_range is None:
        if E_all.size == 0 or E_pass.size == 0:
            raise ValueError("Пустые массивы энергий для расчёта.")
        log_min = np.log10(min(E_all.min(), E_pass.min()))
        log_max = np.log10(max(E_all.max(), E_pass.max()))
    else:
        emin, emax = energy_range
        if emin <= 0 or emax <= 0 or emax <= emin:
            raise ValueError("Некорректный energy_range. Ожидается 0 < emin < emax.")
        log_min = np.log10(emin)
        log_max = np.log10(emax)

    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    total_counts, _ = np.histogram(E_all, bins=bin_edges)
    passed_counts, _ = np.histogram(E_pass, bins=bin_edges)

    area_m2 = math.pi * float(radius_m) ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        eff_area = (passed_counts.astype(float) / total_counts.astype(float)) * area_m2
        eff_area = np.nan_to_num(eff_area, nan=0.0, posinf=0.0, neginf=0.0)

    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return bin_edges, bin_centers, total_counts, passed_counts, eff_area


def compute_effective_area_from_files(
    sums_path: PathLike,
    model_path_or_df: Union[PathLike, pd.DataFrame],
    n_bins: int = 20,
    radius_m: float = DEFAULT_RADIUS_M,
    energy_col: str = "energy",
    energy_range: Optional[Tuple[float, float]] = None,
    model_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    model_glob: str = "**/*.csv",
) -> pd.DataFrame:
    """
    High-level convenience that loads inputs from files and returns a tidy DataFrame.

    Returns
    -------
    DataFrame with columns:
      - e_low, e_high, e_center
      - total_counts, passed_counts
      - eff_area_m2
    """
#    E_all = _load_sums_energies(sums_path, energy_col=energy_col)
#    E_pass = _load_model_energies(model_path_or_df, energy_col=energy_col)
    E_all = _load_sums_energies(sums_path, energy_col=energy_col)

    # --- загрузка model_df целиком (а не только энергий) ---
    if isinstance(model_path_or_df, pd.DataFrame):
        model_df = model_path_or_df.copy()
    else:
        mpath = Path(model_path_or_df)
        if mpath.is_dir():
            files = sorted(mpath.rglob(model_glob))
            if not files:
                warnings.warn(f"No model CSV files found under {mpath} (pattern {model_glob}).")
                model_df = pd.DataFrame()
            else:
                model_df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        else:
            model_df = pd.read_csv(mpath)

    n_before = len(model_df)
    # --- применяем каты, если попросили ---
    if model_filter is not None and not model_df.empty:
        try:
            model_df = model_filter(model_df)
        except Exception as e:
            warnings.warn(f"model_filter failed ({e}). Using unfiltered model_df.")
    n_after = len(model_df)
    if model_filter is not None:
        print(f"[A_eff] Model events: {n_before} -> after cuts: {n_after}")

    if energy_col not in model_df.columns:
        raise KeyError(f"Model data has no '{energy_col}' column after filtering.")
    E_pass = model_df[energy_col].to_numpy()

    edges, centers, total, passed, area = compute_effective_area(
        E_all, E_pass, n_bins=n_bins, radius_m=radius_m, energy_col=energy_col, energy_range=energy_range
    )

    df = pd.DataFrame({
        "e_low": edges[:-1],
        "e_high": edges[1:],
        "e_center": centers,
        "total_counts": total,
        "passed_counts": passed,
        "eff_area_m2": area,
    })
    return df


def save_effective_area_plot(
    df: pd.DataFrame,
    out_png: PathLike,
    title: str = "Эффективная площадь телескопа",
    xlabel: str = "Энергия (ТэВ)",
    ylabel: str = "Эффективная площадь (м²)",
) -> None:
    """
    Save a step-plot of effective area vs. energy.
    Note: follows the plotting rules (matplotlib, single chart, no custom colors).
    """
    #import matplotlib.pyplot as plt

    x = df["e_center"].to_numpy()
    y = df["eff_area_m2"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.step(x, y, where="mid")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.tight_layout()
    out_png = str(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def write_effective_area_products(
    df: pd.DataFrame,
    out_dir: PathLike = "out",
    basename: str = "effective_area",
    save_plot: bool = True,
) -> str:
    """
    Save the computed effective area table and (optionally) the plot to disk.

    Returns
    -------
    The path to the CSV file.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{basename}.csv")
    png_path = os.path.join(out_dir, f"{basename}.png")

    df.to_csv(csv_path, index=False)
    if save_plot:
        save_effective_area_plot(df, png_path)

    return csv_path
