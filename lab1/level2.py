import os
import pathlib
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH: pathlib.Path = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR: pathlib.Path = SCRIPT_PATH.parent
BUILD_DIR: pathlib.Path = SCRIPT_DIR / 'build'

BUILD_DIR.mkdir(parents=True, exist_ok=True)

def generate_trend_quadratic(n: int, a: float = 0.00005, b: float = 0.01, c: float = 10.0) -> np.ndarray:
    t: np.ndarray = np.arange(n)
    return a * t**2 + b * t + c

def generate_noise_normal(n: int, mean: float = 0.0, std: float = 2.0) -> np.ndarray:
    return np.random.normal(mean, std, n)

def generate_noise_exponential(n: int, scale: float = 2.0) -> np.ndarray:
    return np.random.exponential(scale, n)

def calculate_statistics(data: np.ndarray, label: str) -> Tuple[float, float, float]:
    mean_val: float = float(np.mean(data))
    var_val: float = float(np.var(data))
    std_val: float = float(np.std(data))
    
    print(f"--- {label} ---")
    print(f"Expected Value: {mean_val:.4f}")
    print(f"Variance: {var_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print("-" * 30)
    
    return mean_val, var_val, std_val

def plot_histogram(data: np.ndarray, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(BUILD_DIR / filename)
    plt.close()

def plot_dynamics(trend: np.ndarray, noisy_data: np.ndarray, title: str, filename: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(noisy_data, label='Noisy Data', alpha=0.6, color='gray')
    plt.plot(trend, label='Ideal Trend', color='red', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(BUILD_DIR / filename)
    plt.close()

def process_real_data(filepath: pathlib.Path) -> None:
    df: pd.DataFrame = pd.read_csv(filepath, sep='\t')
    df_clean: pd.DataFrame = df[(df['Купівля'] > 0) & (df['Продаж'] > 0)].copy()
    
    indicators: List[str] = ['Купівля', 'Продаж', 'КурсНбу']
    
    for ind in indicators:
        if ind in df_clean.columns:
            data: np.ndarray = df_clean[ind].values
            calculate_statistics(data, f"Real Data: {ind}")
            
            plt.figure(figsize=(10, 4))
            plt.plot(data, marker='o', linestyle='-')
            plt.title(f"Dynamics: {ind}")
            plt.grid(True)
            plt.savefig(BUILD_DIR / f'real_dynamics_{ind}.png')
            plt.close()
            
            plot_histogram(data, f"Histogram: {ind}", f'real_hist_{ind}.png')

if __name__ == "__main__":
    N: int = 1000
    
    trend: np.ndarray = generate_trend_quadratic(N)
    noise_norm: np.ndarray = generate_noise_normal(N, mean=0.0, std=5.0)
    data_synthetic: np.ndarray = trend + noise_norm
    
    calculate_statistics(data_synthetic, "Synthetic Data")
    plot_dynamics(trend, data_synthetic, "Additive Model", "synthetic_dynamics.png")
    plot_histogram(data_synthetic, "Histogram Synthetic", "synthetic_histogram.png")
    
    real_data_path: pathlib.Path = SCRIPT_DIR / 'Lab_work_1' / 'oschad.xlsx.csv'
    
    if real_data_path.exists():
        process_real_data(real_data_path)
    else:
        print(f"File not found: {real_data_path}")