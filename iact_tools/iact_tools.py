import os
import glob
import math
import time

import joblib

from datetime import timedelta

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg') # Headless, чтоб не падал при plt.show() при запуске на удаленном сервере

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm


from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.stats import norm

import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def linear_model(x, a, b):
    """Линейная модель для фитирования зависимости size vs energy."""
    return x * a + b

def square_model(x, b, a, c):
    return (x)**a + b*x + c

def calculate_theta2_classic(dist, size, width, length, alpha):
    '''Это от Свешниковой, здесь рассчитывается угол 
    между направлением события и направлением на источник TODO
    все в градусах, width и length у нее исходно в сантиметрах,
    здесь уже все переделано на градусы.
    '''
    if dist < 0.9045:
        ksi = 1.3
    elif 0.9045 < dist < 1.206:
        ksi = 1.6
    elif dist > 1.206:
        if size < 700.0:
            ksi = 2.0
        elif 700.0 < size < 1000.0:
            ksi = 1.9
        elif 1000.0 < size < 1250.0:
            ksi = 1.75
        elif 1250.0 < size < 1500.0:
            ksi = 1.5
        else:
            ksi = 1.3  # Значение по умолчанию, если size выходит за указанные границы
    elif dist > 1.3:
        return None
    else:
        return None
        #return -1

    disp = ksi * (1.0 - width / length)
    theta2 = disp ** 2 + dist ** 2 - 2.0 * disp * dist * math.cos(alpha / 57.3)
    
    return theta2

def calc_seconds(directory, file_extension=".csv"):
    '''
    Считает длительность рана (по числу 2-минутных порций)
    TODO!
    '''

    total_unique_sum = 0
    # Перебираем все файлы в указанной директории
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):  # Проверяем расширение файла
            file_path = os.path.join(directory, filename)
                # Читаем файл в DataFrame
            df = pd.read_csv(file_path)
            if not df.empty:
                # Считаем количество уникальных значений в первой колонке
                unique_values_count = df.iloc[:, 0].nunique()
                total_unique_sum += unique_values_count
    
    return total_unique_sum*2*60

def estimate_observation_time(exp_data):
    """
    Оценка времени наблюдений:
    - Сортируем по минимальному unix_time_ns каждой порции.
    - Считаем уникальные интервалы по 2 минуты, не перекрываясь.
    """
    if 'por' not in exp_data.columns or 'unix_time_ns' not in exp_data.columns:
        print("[!] Не хватает колонок 'por' и 'unix_time_ns'")
        return 0

    # Получаем минимальные времена на каждую порцию
    min_times = exp_data.groupby('por')['unix_time_ns'].min().sort_values().values

    portion_duration_ns = 2 * 60 * 1_000_000_000  # 2 минуты в наносекундах
    count = 0
    current_end = 0

    for t in min_times:
        if t > current_end:
            count += 1
            current_end = t + portion_duration_ns

    total_seconds = count * 2 * 60
    return total_seconds


class GammaDataLoader:
    """Класс для загрузки данных и отрисовки их параметров."""

    def __init__(self, model_path = None, exp_path = None, sums_data = None):
        self.model_path = model_path
        self.exp_path = exp_path
        self.model_data = pd.DataFrame()
        self.meta_data = sums_data
        self.exp_data = None
        self.loaded_model_files = set()  # для исключения повторной загрузки

    def load_model_data(self, model_path, pattern='*.csv'):
        """Загрузка модельных данных"""
        
        model_files = glob.glob(os.path.join(model_path, pattern))
        new_data = []
        
        for file in tqdm(model_files, desc="Loading model data"):
            if file not in self.loaded_model_files:
                try:
                    df = pd.read_csv(file)
                    new_data.append(df)
                    self.loaded_model_files.add(file)
                except Exception as e:
                    print(f"Ошибка при загрузке {file}: {e}")
        
        if new_data:
            new_df = pd.concat(new_data, ignore_index=True)
            self.model_data = pd.concat([self.model_data, new_df], ignore_index=True)
        
        return self.model_data
    
    def load_meta_data(self, sums_path, telescope_id=1, n_telescopes=5): # TODO
        """
        Загрузка энергии из файла sums для заданного телескопа.

        :param sums_path: Путь к файлу `sums`.
        :param telescope_id: Номер телескопа (от 1 до n_telescopes).
        :param n_telescopes: Общее количество телескопов (по умолчанию 5).
        :return: DataFrame с одной колонкой 'energy'.
        TODO
        """
        assert 1 <= telescope_id <= n_telescopes, "Неверный номер телескопа."

        try:
            # Пропустить первые 2 строки
            df = pd.read_csv(sums_path, delim_whitespace=True, skiprows=2, header=None)
        except Exception as e:
            print(f"Ошибка при чтении файла sums: {e}")
            return pd.DataFrame(columns=['energy'])

        # Выбираем строки, соответствующие заданному телескопу
        selected_rows = df.iloc[telescope_id - 1::n_telescopes, -1]

        # Сохраняем как одноколоночный DataFrame
        energy_df = pd.DataFrame({'energy': selected_rows.values})

        # Сохраняем результат
        self.meta_data = energy_df

        return energy_df

    def compute_effective_area(meta_data, model_data, n_bins=20, out_path = 'out/'):
        """
        Строит эффективную площадь на основе разыгранных и прошедших событий.

        :param meta_data: DataFrame с колонкой 'energy' (истинная энергия всех событий).
        :param model_data: DataFrame с колонкой 'energy' (энергии событий после триггера).
        :param n_bins: Кол-во логарифмических бинов.
        """
        energies_all = meta_data['energy'].values
        energies_triggered = model_data['energy'].values

            # Создаем логарифмические бины по энергии
        log_min = np.log10(min(energies_all.min(), energies_triggered.min()))
        log_max = np.log10(max(energies_all.max(), energies_triggered.max()))
        bins = np.logspace(log_min, log_max, n_bins + 1)

        # Гистограмма: сколько событий всего (MC) и сколько прошло (триггер)
        total_counts, _ = np.histogram(energies_all, bins=bins)
        passed_counts, _ = np.histogram(energies_triggered, bins=bins)

            # Эффективная площадь: отношение * площадь круга
        with np.errstate(divide='ignore', invalid='ignore'):
            eff_area = (passed_counts / total_counts) * AREA
            eff_area = np.nan_to_num(eff_area)  # заменим NaN и inf на 0

            # Центры бинов
        bin_centers = np.sqrt(bins[:-1] * bins[1:])

            # График
        plt.figure(figsize=(8, 5))
        plt.step(bin_centers, eff_area, where='mid')
        plt.xscale('log')
        plt.yscale('linear')
        plt.xlabel('Энергия (GeV)')
        plt.ylabel('Эффективная площадь (м²)')
        plt.title('Эффективная площадь телескопа')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'eff_area.png'))
        #plt.show()

        return bin_centers, eff_area
   
    def reset_model_data(self):
        """Сброс загруженных модельных данных."""
        self.model_data = pd.DataFrame()
        self.loaded_model_files = set()

    # ranges for plotting
    COMMON_RANGES = {
        "size": (0, 3000),
        "dist[0]": (0, 6.0),
        "dist1": (0, 6.0),  # dist1 и dist[0] это одно и то же
        "width[0]": (0, 2.5),
        "width": (0, 2.5),  # width[0] и width это одно и то же
        "length[0]": (0, 2.5),
        "length": (0, 2.5),  # length[0] и length это одно и то же
    } 
        
    def plot_model_pars(self):
        """Гистограммы параметров size, dist[0], width[0], length[0] из self.model_data"""
        common_ranges = {
        "size": (0, 3000),
        "dist[0]": (0, 6.0),
        "dist1": (0, 6.0),  # dist1 и dist[0] это одно и то же
        "width[0]": (0, 2.5),
        "width": (0, 2.5),  # width[0] и width это одно и то же
        "length[0]": (0, 2.5),
        "length": (0, 2.5),  # length[0] и length это одно и то же
        }
        params = ["size", "dist[0]", "width[0]", "length[0]"]
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Model:", fontsize=14)
    
        for ax, param in zip(axes.flatten(), params):
            if param in self.model_data.columns:
                bins = np.linspace(common_ranges[param][0], common_ranges[param][1], 30)
                ax.hist(self.model_data[param], bins=bins, alpha=0.7, color="blue", edgecolor="black")
                ax.set_title(f"{param}")
                ax.set_xlabel(param)
                ax.set_ylabel("N")
                ax.set_xlim(common_ranges[param])
            else:
                ax.set_title(f"{param} not found in model_data")
                ax.axis("off")
    
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('plots/pars_hist_model.png')
        plt.show()
            
    def plot_exp_pars(self):
        """Гистограммы параметров size, dist1, width, length из self.exp_data"""
        common_ranges = {
        "size": (0, 3000),
        "dist[0]": (0, 6.0),
        "dist1": (0, 6.0),  # dist1 и dist[0] это одно и то же
        "width[0]": (0, 2.5),
        "width": (0, 2.5),  # width[0] и width это одно и то же
        "length[0]": (0, 2.5),
        "length": (0, 2.5),  # length[0] и length это одно и то же
        }
        params = ["size", "dist1", "width", "length"]
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Experiment:", fontsize=14)
    
        for ax, param in zip(axes.flatten(), params):
            if param in self.exp_data.columns:
                bins = np.linspace(common_ranges[param][0], common_ranges[param][1], 30)
                ax.hist(self.exp_data[param], bins=bins, alpha=0.7, color="green", edgecolor="black")
                ax.set_title(f"{param}")
                ax.set_xlabel(param)
                ax.set_ylabel("N")
                ax.set_xlim(common_ranges[param])
            else:
                ax.set_title(f"{param} not found in exp_data")
                ax.axis("off")
    
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('plots/pars_hist_exp.png')
        plt.show()
        
    def plot_param_time_distributions(self, exp_data, out_path='plots', prefix=''):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        import os
        import pandas as pd

        if 'unix_time_ns' not in exp_data.columns:
            print("❌ exp_data должен содержать колонку 'unix_time_ns'")
            return

        os.makedirs(out_path, exist_ok=True)
        params = ['size', 'width', 'length', 'dist1']
        exp_data = exp_data.copy()

        # Преобразуем метку времени
        exp_data['timestamp'] = pd.to_datetime(exp_data['unix_time_ns'], unit='ns', errors='coerce')
        exp_data['day'] = exp_data['timestamp'].dt.floor('D')

        for param in params:
            if param not in exp_data.columns:
                print(f"[!] Пропускаю {param}: колонки нет в exp_data")
                continue

            # Удаляем строки с nan в нужных колонках
            df_valid = exp_data[['day', param]].dropna()

            if df_valid.empty:
                print(f"[!] Пропускаю {param}: нет данных после фильтрации NaN")
                continue

            x = df_valid['day']
            y = df_valid[param]

            x_num = mdates.date2num(x)

            plt.figure(figsize=(12, 5))
            plt.hist2d(x_num, y, bins=[np.arange(x_num.min(), x_num.max() + 1.5), 100],
                       cmap='viridis', cmin=1)
            plt.colorbar(label='Counts')
            plt.xlabel('Date')
            plt.ylabel(param)
            plt.title(f'{param} vs Time (daily bins)')
            plt.grid(True, linestyle='--', alpha=0.3)

            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(out_path, f'{prefix}_{param}_vs_time_2d.png'))
            plt.clf()

    def analyze_exp_data_sessions(self, exp_data: pd.DataFrame, out_path='plots/session_analysis'):
        def safe_hist2d(x, y, bins, cmap, cmin, xlabel, ylabel, title, save_path, is_date=False, log_x=False):
            try:
                mask = np.isfinite(x) & np.isfinite(y)
                if not np.any(mask):
                    print(f"[!] Пропуск {title} — все значения нечисловые или бесконечные.")
                    return
                x = x[mask]
                y = y[mask]
                plt.figure(figsize=(14, 5))
                plt.hist2d(x, y, bins=bins, cmap=cmap, cmin=cmin)
                plt.colorbar(label='Counts')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                if is_date:
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    plt.gcf().autofmt_xdate()
                if log_x:
                    plt.xscale('log')
                    plt.xlim(1, 1000)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
            except Exception as e:
                print(f"[!] Ошибка при построении {title}: {e}")

        def safe_hist1d(data, bins, xlabel, ylabel, title, save_path):
            try:
                data = data[np.isfinite(data)]
                if data.empty:
                    print(f"[!] Пропуск {title} — невалидные данные.")
                    return
                if not np.isfinite(data.min()) or not np.isfinite(data.max()):
                    print(f"[!] Пропуск {title} — границы невалидны.")
                    return
                plt.figure()
                plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(False)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
            except Exception as e:
                print(f"[!] Ошибка при построении {title}: {e}")

        os.makedirs(out_path, exist_ok=True)
        exp_data = exp_data.copy()
        exp_data['timestamp'] = pd.to_datetime(exp_data['unix_time_ns'], unit='ns', errors='coerce')
        if not exp_data['timestamp'].is_monotonic_increasing:
            exp_data = exp_data.sort_values('timestamp')

        ignore_columns = {'unix_time_ns', 'timestamp', 'gamma_like_1', 'gamma_like', 'event_numb', 'tel_az', 'tel_el', 'por', 'delta_time', 'source_az', 'source_el', 'tel_ra', 'tel_dec', 'source_ra', 'source_dec', 'source_x', 'source_y'}
        numeric_columns = [col for col in exp_data.select_dtypes(include=[np.number]).columns if col not in ignore_columns]

        print(f"[*] Построение глобальных 2D-гистограмм по времени для {len(numeric_columns)} параметров")
        for i, param in enumerate(numeric_columns):
            print(f"  [{i+1}/{len(numeric_columns)}] {param}")
            x = exp_data['timestamp']
            y = exp_data[param]
            x_num = mdates.date2num(x)
            mask = np.isfinite(x_num) & np.isfinite(y)
            if not np.any(mask):
                print(f"[!] Пропуск глобальной 2D-гистограммы для {param}")
                continue
            bins = [np.arange(x_num[mask].min(), x_num[mask].max() + 1.5), 100]
            save_path = os.path.join(out_path, f'{param}_vs_time_2d.png')
            safe_hist2d(x_num, y.values, bins=bins, cmap='viridis', cmin=1,
                        xlabel='Date', ylabel=param, title=f'{param} vs Time',
                        save_path=save_path, is_date=True)

        time_diffs = exp_data['timestamp'].diff().dt.total_seconds().fillna(0)
        session_starts = [0] + list(exp_data.index[time_diffs >= 604800])
        session_starts.append(len(exp_data))

        print(f"[*] Разбивка на {len(session_starts)-1} сессий и построение графиков по ним")
        for i in range(len(session_starts) - 1):
            print(f"  [Session {i}]")
            session_dir = os.path.join(out_path, f'session_{i}')
            os.makedirs(session_dir, exist_ok=True)
            session_data = exp_data.iloc[session_starts[i]:session_starts[i + 1]]

            for j, param in enumerate(numeric_columns):
                print(f"    [{j+1}/{len(numeric_columns)}] {param}")
                session_param = session_data[[param, 'size']].copy()
                session_param = session_param[np.isfinite(session_param[param])]

                if not session_param[param].empty:
                    save_path_1d = os.path.join(session_dir, f'{param}_hist.png')
                    safe_hist1d(session_param[param], bins=30, xlabel=param, ylabel='Counts',
                                title=f'{param} Histogram (Session {i})', save_path=save_path_1d)

                if param != 'size':
                    valid = np.isfinite(session_param['size']) & np.isfinite(session_param[param])
                    if valid.any():
                        bins = [np.logspace(0, np.log10(1000), 50), 50]
                        save_path = os.path.join(session_dir, f'{param}_vs_size_2d.png')
                        safe_hist2d(session_param['size'].values, session_param[param].values,
                                    bins=bins, cmap='plasma', cmin=1,
                                    xlabel='size', ylabel=param,
                                    title=f'{param} vs size (Session {i})',
                                    save_path=save_path, log_x=True)

        print("[✓] Анализ завершён.")

    def load_experiment_data(self, file_pattern = "*hillas_14_7.0fix.csv"):
        """Загрузка экспериментальных данных с прогресс-баром.
        Принимает сразу папку с иактом за весь сезон, 
        если надо вычитать другой период, надо изменить file_pattern
        """
        #file_pattern = os.path.join(self.exp_path, "**/*hillas_14_7.0fix.csv") # это для 20-21
        file_pattern = os.path.join(self.exp_path, file_pattern) # вариативный
        #file_pattern = os.path.join(self.exp_path, "/*hillas_14_7.0fix.csv") # это для 19-20
        #taiga607_hillas_iact05_14_7fix_cb0.csv name example
        #/home/bzzkv/soft/taiga/msu/k38/taiga_pool/iact_trigger/from_ISU/gamma/gamma_old_cone/bpe607_31m_da1.2_md5/IACT01_20_21/b0/14-7fix/taiga607_hillas_iact01_14_7fix_cb0.csv
        exp_files = sorted(glob.glob(file_pattern, recursive=True))
        print('Loading data from '+self.exp_path)
        #print(file_pattern)
        #file_pattern = '231119.01_out_hillas_14_7.0fix.csv'
        #print(exp_files)
        exp_data = []
        # костыль для обработки только первого иакта
        for file in tqdm(exp_files, desc="Loading experiment data"):
            exp_data.append(pd.read_csv(file))
        return pd.concat(exp_data, ignore_index=True)

class GammaSpectrumReconstructor:
    """Класс для восстановления энергии и построения спектра."""

    def __init__(self, model_data, bins = None, regressor=None):
        #self.model_data = self.filter_model_data(model_data)
        self.model_data = model_data # csv file
        #self.sums_data = sums_data # sums file
        self.dist_bins = bins if bins else np.arange(0, 5, 0.3)
        #self.energy_fit_params = self.fit_energy_size()
        self.regressor = regressor
        self.cuts = None

    def filter_model_data(self, df):
        """Отсеивание модельных событий для последующего фита"""
        df = df[(df['size'] > 120) & 
                (df['dist[0]'].between(0.36, 1.44)) &
                (df['width[0]'].between(0.024, 0.068 * np.log10(df['size']) - 0.047)) &
                (df['length[0]'] < 0.145 * np.log10(df['size']))]# &
                #(2 * df['dist[0]'] ** 2 * (1 - np.cos(np.radians(df['alpha[0]']))) < 10) &
                #(df['alpha[0]'] < ALPHA_CUT)]
        return df

    def add_background_alphas_dists_thetas_abs(self, df):
        """
        Добавляет в DataFrame alpha1–alpha6 и dist1–dist6:
        источник, антиисточник и 4 фоно-точки на окружности вокруг центра камеры.
        Углы alpha считаются как абсолютные отклонения от главной оси эллипса.
        """

        df = df.copy()

        # Вектор от CoG до источника
        dx = df['source_x'] - df['Xc']
        dy = df['source_y'] - df['Yc']
        v_angle = np.arctan2(dy, dx)  # угол в радианах

        # Направление главной оси эллипса: от вектора на источник - alpha1
#        u_angle = v_angle - np.deg2rad(df['alpha1'])  # главная ось в радианах
        u_angle = np.arctan(df['a_axis'])

        u_x = np.cos(u_angle)
        u_y = np.sin(u_angle)

        # Расстояние от центра камеры до источника — радиус окружности
        r = np.sqrt(df['source_x']**2 + df['source_y']**2)

        # Угол на источник из центра камеры
        angle0 = np.arctan2(df['source_y'], df['source_x'])

        # Углы точек: 0° (источник), 60°, 120°, 180° (антиисточник), 240°, 300°
        angles_deg = [0, 180, 60, 120, 240, 300]
        angles_rad = [angle0 + np.deg2rad(a) for a in angles_deg]

        for i, angle in enumerate(angles_rad, start=1):
            # Координаты точки на окружности
            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Вектор от CoG до точки
            dx = x - df['Xc']
            dy = y - df['Yc']
            dist = np.sqrt(dx**2 + dy**2)

            # Угол между направлением на точку и главной осью (в [0°, 180°])
            v_angle = np.arctan2(dy, dx)
            delta = np.abs(v_angle - u_angle)
            delta = np.mod(delta, 2 * np.pi)
            alpha = np.minimum(delta, 2 * np.pi - delta)
            alpha_deg = np.rad2deg(alpha)
            alpha_deg = np.where(alpha_deg > 90, 180 - alpha_deg, alpha_deg)

            df[f'dist{i}'] = dist
            df[f'alpha{i}'] = alpha_deg

        # Считам theta2 для 5 точек шума и 1 точки сигнала
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

    def add_background_alphas_dists_thetas_fast(self, df):
        ''' Более быстрая версия add_background_alphas_dists_thetas_abs, предпочтительна для использования/
        Включает векторизированную версию calc_theta2_classic '''
        df = df.copy()

        dx_src = df['source_x'] - df['Xc']
        dy_src = df['source_y'] - df['Yc']
        v_angle = np.arctan2(dy_src, dx_src)
        u_angle = np.arctan(df['a_axis'])

        r = np.sqrt(df['source_x']**2 + df['source_y']**2)
        angle0 = np.arctan2(df['source_y'], df['source_x'])

        angles_deg = np.array([0, 180, 60, 120, 240, 300])
        angles_rad = np.deg2rad(angles_deg).reshape(1, -1)
        angle0 = angle0.values.reshape(-1, 1)
        r = r.values.reshape(-1, 1)

        all_angles = angle0 + angles_rad
        x_all = r * np.cos(all_angles)
        y_all = r * np.sin(all_angles)

        dx_all = x_all - df['Xc'].values.reshape(-1, 1)
        dy_all = y_all - df['Yc'].values.reshape(-1, 1)
        dist_all = np.sqrt(dx_all**2 + dy_all**2)

        v_angle_all = np.arctan2(dy_all, dx_all)
        u_angle = u_angle.values.reshape(-1, 1)
        delta = np.abs(v_angle_all - u_angle)
        delta = np.mod(delta, 2 * np.pi)
        alpha = np.minimum(delta, 2 * np.pi - delta)
        alpha_deg = np.rad2deg(alpha)
        alpha_deg = np.where(alpha_deg > 90, 180 - alpha_deg, alpha_deg)

        for i in range(6):
            df[f'dist{i+1}'] = dist_all[:, i]
            df[f'alpha{i+1}'] = alpha_deg[:, i]

        # theta2 расчёт (векторизованный аналог calculate_theta2_classic)
        size = df['size'].values
        width = df['width'].values
        length = df['length'].values

        for i in range(6):
            dist = dist_all[:, i]
            alpha = alpha_deg[:, i]

            ksi = np.full_like(dist, 1.3)

            mask1 = (dist >= 0.9045) & (dist < 1.206)
            ksi[mask1] = 1.6

            mask2 = dist >= 1.206
            mask2a = mask2 & (size < 700)
            mask2b = mask2 & (size >= 700) & (size < 1000)
            mask2c = mask2 & (size >= 1000) & (size < 1250)
            mask2d = mask2 & (size >= 1250) & (size < 1500)
            mask2e = mask2 & (size >= 1500)

            ksi[mask2a] = 2.0
            ksi[mask2b] = 1.9
            ksi[mask2c] = 1.75
            ksi[mask2d] = 1.5
            ksi[mask2e] = 1.3

            disp = ksi * (1.0 - width / length)
            theta2 = disp**2 + dist**2 - 2.0 * disp * dist * np.cos(np.deg2rad(alpha))

            df[f'theta2_{i+1}'] = theta2

        return df

    def filter_experiment_data(self, df):
        """Первичное отсеивание фоновых событий по классическим катам. 
        Добавление флага gamma_like_1 (флаг первичного отсеивания) и theta2.
        После отрабатывания этой функции дальнейшее отсеивание можно делать по theta2."""
        df['gamma_like'] = (
                (df['size'] > 120) & 
                (df['dist1'].between(0.36, 1.44)) &
                (df['width'] > 0.024) & 
                (df['width'] < 0.068 * np.log10(df['size']) - 0.047) &
                (df['length'] < 0.145 * np.log10(df['size']))# &
        )

        df['gamma_like_2'] = (
                (df['size'] > 120) & 
                (df['dist2'].between(0.36, 1.44)) &
                (df['width'] > 0.024) & 
                (df['width'] < 0.068 * np.log10(df['size']) - 0.047) &
                (df['length'] < 0.145 * np.log10(df['size']))# &
        )

        
        df['theta2'] = df.apply(lambda row: calculate_theta2_classic(row['dist1'], row['size'], row['width'], row['length'], row['alpha1']), axis=1)
        df['theta2_off'] = df.apply(lambda row: calculate_theta2_classic(row['dist2'], row['size'], row['width'], row['length'], row['alpha2']), axis=1)
        return df

    def train_gamma_classifier_nn(self, model_df, exp_df, path = ''):
        """Обучает MLP-классификатор с нормализацией признаков и отладкой."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import roc_auc_score
        import matplotlib.pyplot as plt

        model_df = model_df.rename(columns={'dist[0]': 'dist', 'width[0]': 'width', 'length[0]': 'length'})
        model_df = model_df[model_df['dist'].between(0.3,1.5)]
        
        #exp_df = exp_df.rename(columns={'dist1': 'dist'})  # вариант с расчетом по источнику
        exp_df = exp_df.rename(columns={'dist2': 'dist'})  # для верности берем антиисточник

        features = ['size', 'width', 'length', 'dist',
                    'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']

        model_df = model_df[features].dropna()
        exp_df = exp_df[exp_df['dist'].between(0.3,1.5)]
        exp_df = exp_df[features].dropna()

        model_df['label'] = 1
        exp_df['label'] = 0

        df_all = pd.concat([model_df, exp_df], ignore_index=True)
        df_all = shuffle(df_all, random_state=42)

        X = df_all[features]
        y = df_all['label']
        print("Labels distribution:", np.bincount(y))

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(8,16,8,4),
                activation='relu',
                solver='adam',
                alpha=1e-3,
                max_iter=100,
                early_stopping=True,         # ← теперь явно включено
                n_iter_no_change=3,          # ← patience = 3
                tol=1e-3,                    # ← чувствительность к улучшению
                validation_fraction=0.1,
                verbose=True,
                random_state=42
            ))

        ])


        clf.fit(X, y)

        # AUC
        y_proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        print(f"\n[INFO] AUC: {auc:.4f}")

        # Кривая лосса
        mlp = clf.named_steps['mlp']
        if hasattr(mlp, 'loss_curve_'):
            plt.plot(mlp.loss_curve_)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("MLP Training Loss")
            plt.grid()
            plt.savefig(path+'loss.png')

        return clf

    def train_forest_classifier(self, model_df, exp_df):
        """Обучает RandomForest классификатор по аналогии с NN-классификатором.
        TODO! сейчас часть функционала в обертке (динамическое вычисление порогов) надо вынести сюда.
        Может общий класс для классификаторов сделать."""

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils import shuffle

        # Используем те же фичи, что и для нейросети
        features = ['size', 'width', 'length', 'dist',
                    'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']

        # Переименование колонок
        model_df = model_df.rename(columns={'dist[0]': 'dist', 'width[0]': 'width', 'length[0]': 'length'})
        exp_df = exp_df.rename(columns={'dist2': 'dist'})  # dist2, чтобы точно поменьше гамма зацепить

        # Фильтр по dist: 0.3 < dist < 1.5
        model_df = model_df[model_df['dist'].between(0.3, 1.5)]
        exp_df = exp_df[exp_df['dist'].between(0.3, 1.5)]

        # Отбор признаков и фильтрация
        model_df = model_df[features].dropna()
        exp_df = exp_df[features].dropna()

        model_df['label'] = 1
        exp_df['label'] = 0
        df_all = pd.concat([model_df, exp_df], ignore_index=True)
        df_all = shuffle(df_all, random_state=42)

        X = df_all[features].values
        y = df_all['label'].values

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X, y)

        print('[INFO] RandomForestClassifier обучен')
        return clf

    def save_classifier(self, clf, path='gamma_rf_classifier.pkl'):
        """Сохраняет обученный классификатор и список фичей в файл."""
        #if not hasattr(self, 'rf_clf') or self.rf_clf is None:
        #    raise ValueError("Классификатор не обучен!")

        data = {
            'classifier': clf,
            'features': ['size', 'width', 'length', 'dist',
                         'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']
        }
        joblib.dump(data, path)
        print(f"[+] Классификатор сохранён в {path}")

    def apply_forest_cut(self, df, clf, threshold=0.8):
        """Применяет обученный классификатор к df и ставит флаги gamma_like_{i} с учётом катов по dist."""
        features = ['size', 'width', 'length', 'dist',
                    'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']

        for i in range(1, 7):
            dist_col = f'dist{i}'
            flag_col = f'gamma_like_{i}'

            temp = df[['size', 'width', 'length', dist_col,
                       'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']].copy()
            temp.columns = features

            mask_valid = temp.notna().all(axis=1)
            mask_dist = df[dist_col].between(0.3, 1.5)
            mask = mask_valid & mask_dist

            proba = np.zeros(len(df))
            proba[mask] = clf.predict_proba(temp[mask])[:, 1]

            df[flag_col] = np.nan
            df.loc[mask, flag_col] = proba[mask] > threshold

        return df

    def apply_nn_cut(self, df, clf, threshold=0.8):
        """Применяет обученный MLPClassifier к df и ставит флаги gamma_like_N с учётом катов по dist."""
        features = ['size', 'width', 'length', 'dist',
                    'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']

        for i in range(1, 7):
            dist_col = f'dist{i}'
            flag_col = f'gamma_like_{i}'

            temp = df[['size', 'width', 'length', dist_col,
                       'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']].copy()
            temp.columns = features

            # маска на NaN и подходящий dist
            mask_valid = temp.notna().all(axis=1)
            mask_dist = df[dist_col].between(0.3, 1.5)
            mask = mask_valid & mask_dist

            proba = np.zeros(len(df))
            proba[mask] = clf.predict_proba(temp[mask])[:, 1]

            df[flag_col] = np.nan
            df.loc[mask, flag_col] = proba[mask] > threshold

        return df

    def plot_classifier_output(self, model_df, exp_df, clf, path = '', threshold = 0.8):
        """Строит гистограмму выходов классификатора для гамма и фона (модель и эксперимент)."""
        model_df = model_df.rename(columns={'dist[0]': 'dist', 'width[0]': 'width', 'length[0]': 'length'})
        exp_df = exp_df.rename(columns={'dist1': 'dist'})

        features = ['size', 'width', 'length', 'dist',
                    'skewness_l', 'kurtosis_l', 'skewness_w', 'kurtosis_w']

        model_df = model_df[features].dropna()
        exp_df = exp_df[features].dropna()

        # Получаем вероятности
        gamma_probs = clf.predict_proba(model_df)[:, 1]
        bg_probs = clf.predict_proba(exp_df)[:, 1]

        # Рисуем
        plt.figure(figsize=(8, 5))
        sns.histplot(gamma_probs, bins=50, label='Gamma (model)', color='blue', stat='density', kde=True, alpha=0.5)
        sns.histplot(bg_probs, bins=50, label='Background (exp)', color='red', stat='density', kde=True, alpha=0.5)

        plt.xlabel('Classifier Output (P(gamma))')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.2f}')
        plt.tight_layout()
        plt.savefig(path+'classifier_output_gamma_vs_bg.png')
        plt.show()

    def plot_param_distributions(self, model_data, exp_data, out_path='plots'):
        plt.figure(figsize=(15, 10))

        # Size
        plt.subplot(2, 2, 1)
        plt.hist(model_data['size'], bins=50, alpha=0.6, label='Gamma', color='blue')
        plt.hist(exp_data['size'], bins=50, alpha=0.6, label='Background', color='red')
        plt.xlabel('size')
        plt.legend()

        # Dist
        plt.subplot(2, 2, 2)
        plt.hist(model_data['dist[0]'], bins=50, alpha=0.6, label='Gamma', color='blue')
        plt.hist(exp_data['dist1'], bins=50, alpha=0.6, label='Background', color='red')
        plt.xlabel('dist')
        plt.legend()

        # Width vs Length
        plt.subplot(2, 2, 3)
        plt.scatter(model_data['width[0]'], model_data['length[0]'], s=5, alpha=0.3, label='Gamma', color='blue')
        plt.scatter(exp_data['width'], exp_data['length'], s=5, alpha=0.3, label='Background', color='red')
        plt.xlabel('width')
        plt.ylabel('length')
        plt.legend()

        plt.tight_layout()
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(os.path.join(out_path, "params_comparison.png"))
        plt.close()

    def plot_detailed_param_distributions(self, model_data, exp_data, out_path='plots', prefix=''):

        # Настройки
        param_map_model = {
            'size': 'size',
            'dist1': 'dist[0]',
            'width': 'width[0]',
            'length': 'length[0]'
        }
        param_map_exp = {
            'size': 'size',
            'dist1': 'dist1',
            'width': 'width',
            'length': 'length'
        }
        
        params = ['size', 'dist1', 'width', 'length']
        groups = {
            "Model": model_data,
            "Exp (all)": exp_data,
            "Exp (gamma_like)": exp_data[exp_data['gamma_like_1'] == True],
            "Exp (hadron_like)": exp_data[exp_data['gamma_like_1'] == False],
        }

        fig, axes = plt.subplots(len(params), len(groups), figsize=(16, 12))
        for col_idx, (group_name, data) in enumerate(groups.items()):
            for row_idx, param in enumerate(params):
                ax = axes[row_idx][col_idx]
                col_name = param_map_model[param] if group_name == "Model" else param_map_exp[param]

                if col_name in data.columns:
                    if param == 'size':
                        ax.hist(data[col_name], bins=30, alpha=0.7, edgecolor='black', range=(min(data[col_name]), max(data[col_name])))
                        ax.set_yscale('log')
                        ax.set_xscale('log')
                        #ax.set_xlim(0, 500)
                    else:
                        ax.hist(data[col_name], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{group_name}\n{param}' if row_idx == 0 else f'{param}')
                ax.set_xlabel(param)
                ax.set_ylabel("N")
                ax.grid(True)

        plt.tight_layout()
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(os.path.join(out_path, f"{prefix}_params_detailed_comparison.png"))
        plt.show()

        # Scatter width vs length
        fig, axes = plt.subplots(1, len(groups), figsize=(16, 4))
        for ax, (group_name, data) in zip(axes, groups.items()):
            width_col = param_map_model['width'] if group_name == "Model" else param_map_exp['width']
            length_col = param_map_model['length'] if group_name == "Model" else param_map_exp['length']

            if width_col in data.columns and length_col in data.columns:
                ax.scatter(data[width_col], data[length_col], s=5, alpha=0.4)
                ax.set_xlabel("width")
                ax.set_ylabel("length")
                ax.set_title(f"{group_name}")
                ax.grid(True)
                ax.set_xlim(0, 2.5)
                ax.set_ylim(0, 2.5)

        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"{prefix}_width_vs_length_scatter.png"))
        plt.show()

    def apply_first_cut(self, df, cuts=None):
        if cuts is None:
            cuts = {
                'width_min': 0.024,
                'width_max': 0.068,
                'length_max': 0.145
            }

        for i in range(1, 7):
            dist_col = f'dist{i}'
            gamma_col = f'gamma_like_{i}'
            df[gamma_col] = (
                (df['CR_portion'] > 5) &
                (df['weather_mark'] >= 4) &
                (df['size'] > 120) &
                (df[dist_col].between(0.36, 1.44)) &
                (df['width'] > cuts['width_min']) &
                (df['width'] < cuts['width_max']*np.log10(df['size']) - 0.047) &
                (df['length'] < cuts['length_max']*np.log10(df['size']))
            )
        return df

    def apply_good_cut(self, exp_data):
        ''' Кат на хорошесть данных (шумы, темпы счета, погода) 
        '''
        exp_data = exp_data[
            (exp_data['tracking'] == 1) &
            (exp_data['good'] == 1) &
            (exp_data['edge'] == 0) &
            (exp_data['tel_el'] < 60) &
            (exp_data['tel_el'] > 50) &
            (exp_data['weather_mark'] > 8) &
            (exp_data['star'] == 0) &
            (exp_data['CR_portion'] > 6)
        ]
        return exp_data

    def fit_energy_size(self):
        ''' Фитируем size от энергии (линейной функцией по логарифмам параметров) '''
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
                    plt.ylim(1, 2.5)
                    plt.grid()
                    plt.savefig(f'plots/size_fit_{i}.png')
                    plt.clf()
                except Exception as e:
                    print(f"[!] Ошибка фита в бинe {self.dist_bins[i]}–{self.dist_bins[i+1]}: {e}")
            else:
                print(f"[!] Пропускаем бин {self.dist_bins[i]}–{self.dist_bins[i+1]}: мало точек ({len(bin_data)})")

        return fit_params

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

    def reconstruct_energy_regressor(self, exp_data):
        """Восстановление энергии с помощью обученного регрессора по эксп данным."""
        if self.regressor is None:
            raise ValueError("Regressor model is not provided!")
        
        # Переименовываем колонки в формат, понятный регрессору
        exp_data = exp_data.rename(columns={
            'dist1': 'dist[0]',
            'width': 'width[0]',
            'length': 'length[0]'
        })
        
        feature_columns = ['size', 'dist[0]', 'width[0]', 'length[0]', 'numb_pix']
        valid_rows = exp_data[feature_columns].notna().all(axis=1)
        features = exp_data.loc[valid_rows, feature_columns].values.astype(np.float32)
        features = self.regressor.scaler_X.transform(features)  # Нормализация входных данных
        
        self.regressor.model.eval()
        with torch.no_grad():
            predicted_energy = self.regressor.model(torch.tensor(features)).cpu().numpy()
            predicted_energy = self.regressor.scaler_y.inverse_transform(predicted_energy)
        
        exp_data.loc[valid_rows, 'predicted_energy'] = predicted_energy.flatten()
        return exp_data

    def reconstruct_energy_for_model_data(self, exp_data):
        """Восстановление энергии по модельным данным"""
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

    def plot_theta2_distribution_with_background_average(self, exp_data, path='out', prefix = '', exp_path = ''):
        """Построение theta² с усреднением по фоновым точкам."""
        #bins = np.linspace(0, 5, 250)
        bins = np.linspace(0, 1.25, 50)
        bins005 = np.linspace(0, 0.05, 5)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_centers005 = 0.5 * (bins005[:-1] + bins005[1:])

        # Сигнальные данные
        #on_data = exp_data[exp_data['gamma_like']]['theta2_1']
        #on_data = exp_data[exp_data['gamma_like_1']]['theta2_1']# рабочая версия
        on_data = exp_data.loc[exp_data['gamma_like_1'] == True, 'theta2_1'] # чтоб не ломалоcь по NaN

        N_on, _ = np.histogram(on_data, bins=bins)
        N_on005, _ = np.histogram(on_data, bins=bins005)
        
#        print(N_on005)
        # Список фоновых theta²
        bg_cols = [f'theta2_{i}' for i in range(2, 7)]
        N_off_all = []
        N_off_all005 = []
        
        # Фоновая выборка: theta2_2–6 с соответствующими gamma_like_N
        for i in range(2, 7):
            theta_col = f'theta2_{i}'
            gamma_flag = f'gamma_like_{i}'
            #gamma_flag = f'gamma_like_1'

            if theta_col in exp_data.columns:
                #data = exp_data[exp_data[gamma_flag]][theta_col]
                data = exp_data.loc[exp_data[gamma_flag] == True, theta_col]
                #data = exp_data[theta_col]
                
                hist, _ = np.histogram(data, bins=bins)
                N_off_all.append(hist)
                
                hist005, _ = np.histogram(data, bins=bins005)
                N_off_all005.append(hist005)
                
                

        if not N_off_all:
            print("Нет данных по фону для усреднения.")
            return

        # Усреднённый фон
        N_off_avg = np.mean(N_off_all, axis=0)
        N_off_avg005 = np.mean(N_off_all005, axis=0)

        # Рисуем
        bar_width = 0.0125
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers - bar_width/2, N_on, width=bar_width, label='source', alpha = 0.7, color='dodgerblue')
        plt.bar(bin_centers + bar_width/2, N_off_avg, width=bar_width, label='background avg 5 points', alpha = 0.7, color='orange')
#        plt.bar(bin_centers - bar_width/2, N_on, width=bar_width, label='source', alpha = 0.5, color='dodgerblue')
#        plt.bar(bin_centers + bar_width/2, N_off_avg, width=bar_width, label='background avg 5 points', alpha = 0.5, color='orange')

        plt.xlabel(r'$\theta^2$ (deg$^2$)')
        plt.ylabel('Counts')
        plt.legend()
        #plt.title('Theta² distribution with background average')
        plt.title(f'{prefix}')
        
        plt.tight_layout()
        plt.grid()
        plt.savefig(path+'theta_comp_' + prefix + '.png')
        plt.show()

        plt.figure(figsize=(8, 5))

        # Разность
        diff = N_on - N_off_avg
        plt.plot(bin_centers, diff, marker='o', linestyle='-', label='N_on - N_off', color='black')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Theta²")
        plt.ylabel("N_on - N_off")
        #plt.title("Excess (ON - OFF)")
        plt.title(f'Excess ON-OFF {prefix}')
        
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        
        # Сигма (Li & Ma)
        N_on_total = N_on005.sum()



        N_off_total = np.sum(N_off_all005)
        print(N_on_total, 'N on')
        print(N_off_total, 'N off')
        print(N_off_total/5, 'N off / 5')
        try:
            S_lima = round(
                np.sqrt(
                    2 * N_on_total * np.log(6 * (N_on_total / (N_on_total + N_off_total))) +
                    2 * N_off_total * np.log(1.2 * (N_off_total / (N_on_total + N_off_total)))
                ), 2
            )
        except:
            S_lima = np.nan
        
        plt.figtext(0.15, 0.01, f"N_on = {N_on_total}, N_off = {N_off_total/5.}, S = {S_lima}")


        duration_hours = estimate_observation_time(exp_data)/3600
        duration_hours_total = calc_seconds(exp_path)/3600
        
        #plt.figtext(0.15, 0.06, f"Observation time ≈ {round(duration_hours, 1)} h")
        plt.figtext(0.15, 0.04, f"Observation time total ≈ {duration_hours_total:.1f} h")
        #plt.figtext(0.15, 0.08, f"Observation time after cuts ≈ {duration_hours:.1f} h")

        plt.tight_layout()
        plt.savefig(path+'theta_diff_' + prefix + '.png')
        plt.show()
        
    def plot_alpha_distribution_with_background_average(self, exp_data, path = 'out', prefix=''):
        """Построение alpha с усреднением по фоновым точкам."""
        
        bins = np.linspace(0, 50, 50)           # Градусы от 0 до 50
        bins10 = np.linspace(0, 1, 5)           # Узкая область сигнала (0–10°)
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_centers10 = 0.5 * (bins10[:-1] + bins10[1:])

        # ON данные (alpha1)
        #on_data = exp_data[exp_data['gamma_like_1']]['alpha1']
        on_data = exp_data.loc[exp_data['gamma_like_1'] == True, 'alpha1']

        N_on, _ = np.histogram(on_data, bins=bins)
        N_on10, _ = np.histogram(on_data, bins=bins10)

        # Фоновые alpha: alpha2 ... alpha6
        N_off_all = []
        N_off_all10 = []

        for i in range(2, 7):
            alpha_col = f'alpha{i}'
            gamma_flag = f'gamma_like_{i}'
            if alpha_col in exp_data.columns:
                #data = exp_data[exp_data[gamma_flag]][alpha_col]
                data = exp_data.loc[exp_data[gamma_flag] == True, alpha_col]

                hist, _ = np.histogram(data, bins=bins)
                hist10, _ = np.histogram(data, bins=bins10)

                N_off_all.append(hist)
                N_off_all10.append(hist10)

        if not N_off_all:
            print("Нет данных по фону для усреднения.")
            return

        N_off_avg = np.mean(N_off_all, axis=0)
        N_off_avg10 = np.mean(N_off_all10, axis=0)

        # Рисуем основную гистограмму
        bar_width = 1.5
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers - bar_width/2, N_on, width=bar_width, label='source (alpha1)', alpha=0.7, color='dodgerblue')
        plt.bar(bin_centers + bar_width/2, N_off_avg, width=bar_width, label='background avg (alpha2–6)', alpha=0.7, color='orange')
        plt.xlabel(r'$\alpha$ (deg)')
        plt.ylabel('Counts')
        plt.legend()
        plt.title('Alpha distribution with background average')
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{path}alpha_comp_{prefix}.png')
        plt.show()

        # Разность
        plt.figure(figsize=(8, 5))
        diff = N_on - N_off_avg
        plt.plot(bin_centers, diff, marker='o', linestyle='-', label='N_on - N_off', color='black')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel(r'$\alpha$ (deg)')
        plt.ylabel("N_on - N_off")
        plt.title("Excess (ON - OFF) in Alpha (1 deg)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()

        # Сигнификанс (Li & Ma) в узкой области
        N_on_total = N_on10.sum()
        N_off_total = np.sum(N_off_all10)
        print(f"N_on = {N_on_total}")
        print(f"N_off = {N_off_total}")
        print(f"N_off / 5 = {N_off_total / 5}")

        try:
            S_lima = round(
                np.sqrt(
                    2 * N_on_total * np.log(6 * (N_on_total / (N_on_total + N_off_total))) +
                    2 * N_off_total * np.log(1.2 * (N_off_total / (N_on_total + N_off_total)))
                ), 2
            )
        except Exception as e:
            print("Ошибка при вычислении сигнификанса:", e)
            S_lima = np.nan

        plt.figtext(0.15, 0.01, f"N_on = {N_on_total}, N_off = {N_off_total/5:.1f}, S = {S_lima}")
        plt.tight_layout()
        plt.savefig(f'{path}alpha_diff_{prefix}.png')
        plt.show()

class SpectrumAnalyzer:
    """Класс для анализа и построения спектра."""

    def __init__(self, exp_data):
        self.exp_data = exp_data

    def plot_spectrum(self, threshold=0.05):
        """Построение спектра только для событий с theta2 < threshold."""
        #scaling_factor = # надо его динамически
        #scaling_factor = 10000 * 6 * 10 ** 4  * 479160 # cm * s
        filtered_data_theta = self.exp_data[self.exp_data['theta2'] < threshold]
        filtered_data = filtered_data_theta
        #filtered_data = filtered_data_theta[filtered_data_theta['width'] < 0.6]
        #weights=np.ones(len(filtered_data)) / scaling_factor
        plt.hist(filtered_data['reconstructed_energy'], bins=np.linspace(0, 200, 50), alpha=0.7, label=f'theta2 < {threshold}')
        plt.xlabel('Energy (TeV)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.xlim(0,200)
        plt.legend()
        plt.show()

    def plot_all_spectrum(self, threshold=0.05):
        """Построение спектра только для событий с theta2 < threshold."""
        #scaling_factor = # надо его динамически
        #scaling_factor = 1000 * 4791600 # cm * s
        #filtered_data = self.exp_data
        #weights=np.ones(len(filtered_data)) / scaling_factor
        plt.hist(self.exp_data['reconstructed_energy'], bins=np.linspace(0, 200, 50), alpha=0.7, label=f'all')
        plt.xlabel('Energy (TeV)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.xlim(0,200)
        plt.legend()
        plt.show()

    def plot_comparison(self, threshold=0.05):
        """Построение спектра только для событий с theta2 < threshold."""
        #scaling_factor = # надо его динамически
        #scaling_factor = 1000 * 4791600 # cm * s
        #filtered_data = self.exp_data
        #weights=np.ones(len(filtered_data)) / scaling_factor
        filtered_data_theta = self.exp_data[self.exp_data['theta2'] < threshold]
        filtered_data = filtered_data_theta
        plt.hist(filtered_data['reconstructed_energy'], bins=np.linspace(0, 200, 50), alpha=0.7, color = 'blue', label=f'classic')
        plt.hist(filtered_data[filtered_data['reconstructed_energy'] > 25]['predicted_energy'], bins=np.linspace(0, 200, 50), alpha=0.7, color = 'red', label=f'regressor')
        plt.xlabel('Energy (TeV)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.xlim(0,200)
        plt.legend()
        plt.savefig('plots/spectrum_comparison.png')
        plt.show()
    

    def plot_theta2_distribution(self):
        """Построение распределения theta2."""
        plt.hist(self.exp_data['theta2'], bins=100, alpha=0.7, label='Theta2 ON')
        plt.hist(self.exp_data['theta2_off'], bins=100, alpha=0.7, label='Theta2 OFF')
        plt.xlabel('Theta2')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()
        
        on_data = self.exp_data['theta2']
        off_data = self.exp_data['theta2_off']
        
        bins = np.linspace(min(on_data.min(), off_data.min()), max(on_data.max(), off_data.max()), 25)
        # Вычисляем гистограммы
        hist_on, bin_edges = np.histogram(on_data, bins=bins)
        hist_off, _ = np.histogram(off_data, bins=bins)
        
        # Вычисляем разницу
        hist_diff = hist_on - hist_off
        
        # Строим гистограмму разницы
        plt.figure(figsize=(8, 6))
        plt.step(bin_edges[:-1], hist_diff, where='mid', label='Theta2 ON - Theta2 OFF', linestyle='-', linewidth=2)
        
        plt.xlabel("Theta²")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig('plots/theta_diff.png')
        plt.show()
    
    def plot_theta2_distribution_with_cuts(self, prefix = ''):
        """Построение распределения theta2."""
        bins = bins = np.linspace(0, 5, 20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        on_data = self.exp_data[self.exp_data['gamma_like']]['theta2']
        off_data = self.exp_data[self.exp_data['gamma_like_anti']]['theta2_off']
        
        N_on, _ = np.histogram(on_data, bins=bins)
        N_off, _ = np.histogram(off_data, bins=bins)
        
        bar_width = 0.1  # ширина баров
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers - bar_width/2, N_on, width=bar_width, label='source', color='dodgerblue')
        plt.bar(bin_centers + bar_width/2, N_off, width=bar_width, label='anti-source', color='orange')

        plt.xlabel('Theta2')
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig('plots/theta_comp_'+prefix+'.png')
        plt.show()
        
        on_data = self.exp_data[self.exp_data['gamma_like']]['theta2']
        off_data = self.exp_data[self.exp_data['gamma_like_anti']]['theta2_off']
        
        bins = np.linspace(min(on_data.min(), off_data.min()), max(on_data.max(), off_data.max()), 15)
        # Вычисляем гистограммы
        hist_on, bin_edges = np.histogram(on_data, bins=bins)
        hist_off, _ = np.histogram(off_data, bins=bins)
        
        # Вычисляем разницу
        hist_diff = hist_on - hist_off
        
        # Строим гистограмму разницы
        plt.figure(figsize=(8, 6))
        plt.step(bin_edges[:-1], hist_diff, where='mid', label='Theta2 ON - Theta2 OFF', linestyle='-', linewidth=2)
        
        plt.xlabel("Theta²")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig('plots/theta_diff'+prefix+'.png')
        plt.show()


# REGRESSOR BLOCK 

class GammaShowerRegressor:
    def __init__(self, model_path='gamma_regressor.pth'):
        self.model_path = model_path
        self.model = self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.df = None  # Сохраняем DataFrame для сравнения

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(5, 64),  # Теперь 5 входных параметров
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout для борьбы с оверфитом
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def load_data(self, df):
        df = df[['size', 'dist[0]', 'width[0]', 'length[0]', 'numb_pix', 'energy', 'reconstructed_energy']].dropna()  # Удаляем строки с NaN
        self.df = df  # Сохраняем DataFrame для последующего анализа
        
        features = df[['size', 'dist[0]', 'width[0]', 'length[0]', 'numb_pix']].values.astype(np.float32)
        target = df['energy'].values.astype(np.float32).reshape(-1, 1)
        
        features = self.scaler_X.fit_transform(features)  # Нормализация признаков
        target = self.scaler_y.fit_transform(target)  # Нормализация целевой переменной
        
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            features, target, df.index, test_size=0.2, random_state=42
        )
        
        self.test_indices = test_indices  # Сохраняем индексы тестовой выборки
        
        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32, shuffle=False)
    
    def train(self, epochs=500):
        best_loss = float('inf')
        patience = 5  # Остановим обучение, если 20 эпох нет улучшений
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
            
            # Reduce LR если нужно
            self.scheduler.step(avg_loss)
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        print("Model loaded successfully")
    
    def plot_energy_histograms(self):
        self.model.eval()
        true_energies = []
        predicted_energies = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch).cpu().numpy()
                
                # Обратная нормализация
                predictions = self.scaler_y.inverse_transform(predictions)
                y_batch = self.scaler_y.inverse_transform(y_batch.cpu().numpy())
                
                true_energies.extend(y_batch.flatten())
                predicted_energies.extend(predictions.flatten())
        
        # Одинаковый биннинг
        bins = np.arange(0, 205, 5)
        
        # True Energy vs Predicted Energy
        plt.figure(figsize=(8, 6))
        sns.histplot(x=true_energies, y=predicted_energies, bins=[bins, bins], cmap='plasma', cbar=True, edgecolor=None)
        plt.xlabel("True Energy")
        plt.ylabel("Predicted Energy")
        plt.xlim(0, 200)
        plt.ylim(0, 200)
        plt.grid()
        plt.title("Regressor")
        plt.savefig('plots/energy_diff_regressor.png')
        plt.show()
        
        # Energy vs Reconstructed Energy (на тестовых данных)
        if self.df is not None and self.test_indices is not None:
            test_df = self.df.loc[self.test_indices]
            plt.figure(figsize=(8, 6))
            sns.histplot(x=test_df['energy'], y=test_df['reconstructed_energy'], bins=[bins, bins], cmap='viridis', cbar=True, edgecolor=None)
            plt.xlabel("True Energy")
            plt.ylabel("Reconstructed Energy")
            plt.xlim(0, 200)
            plt.ylim(0, 200)
            plt.grid()
            plt.title("Classic")
            plt.savefig('plots/energy_diff_classic.png')
            plt.show()
            
            # Гистограмма отклонений
            #error_regressor = np.array(predicted_energies) - np.array(true_energies)
            #error_classic = test_df['reconstructed_energy'] - test_df['energy']

            # относительные
            error_regressor = (np.array(predicted_energies) - np.array(test_df['energy']))/np.array(test_df['energy'])
            error_classic = np.array(test_df['reconstructed_energy'] - np.array(test_df['energy']))/np.array(test_df['energy'])
            
            
            plt.figure(figsize=(8, 6))
            sns.histplot(error_regressor, bins=50, label='Regressor', color='blue', alpha=0.6)
            sns.histplot(error_classic, bins=50, label='Classic', color='red', alpha=0.6)
            
            plt.xlabel("Energy Residual (abs)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid()
            plt.savefig('plots/residuals_comparison.png')
            plt.show()


def run_pipeline(model_path, exp_path):
    """Запуск полного анализа."""
    loader = GammaDataLoader(model_path, exp_path)
    model_data = loader.load_model_data()
    exp_data = loader.load_experiment_data()

    reconstructor = GammaSpectrumReconstructor(model_data)
    exp_data = reconstructor.reconstruct_energy(exp_data)
    #exp_data = reconstructor.reconstruct_energy_regressor(exp_data)
    #exp_data = reconstructor.calculate_theta2(exp_data)
    exp_data = reconstructor.filter_events(exp_data)

    analyzer = SpectrumAnalyzer(exp_data)
    analyzer.plot_spectrum()
    analyzer.plot_theta2_distribution()

    
