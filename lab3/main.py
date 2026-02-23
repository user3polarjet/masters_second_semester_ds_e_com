import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Налаштування шляхів для збереження
SCRIPT_PATH: pathlib.Path = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR: pathlib.Path = SCRIPT_PATH.parent
BUILD_DIR: pathlib.Path = SCRIPT_DIR / 'build'

BUILD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. ГЕНЕРАЦІЯ ДАНИХ (Нелінійний процес)
# -----------------------------------------------------------------------------

def generate_nonlinear_data(n: int, q_av: float = 5.0, nav_percent: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Генерація експоненційного тренду, нормального шуму та аномалій."""
    t: np.ndarray = np.arange(n)
    
    # Ідеальний нелінійний (експоненційний) тренд: y = 10 * exp(0.003 * t)
    trend: np.ndarray = 10.0 * np.exp(0.003 * t)
    
    # Нормальний шум
    noise: np.ndarray = np.random.normal(0.0, 5.0, n)
    data_normal: np.ndarray = trend + noise
    
    # Додавання аномалій (викидів)
    data_anomalous: np.ndarray = data_normal.copy()
    num_anomalies: int = int(n * nav_percent)
    anomaly_indices: np.ndarray = np.random.choice(n, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_magnitude: float = float(np.random.normal(0.0, q_av * 5.0))
        data_anomalous[idx] += anomaly_magnitude + (100.0 if np.random.rand() > 0.5 else -100.0)
        
    return trend, data_normal, data_anomalous

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Обчислення коефіцієнта детермінації R^2."""
    ss_res: float = float(np.sum((y_true - y_pred) ** 2))
    ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / ss_tot)

# -----------------------------------------------------------------------------
# 2. R&D: АДАПТИВНЕ ОЧИЩЕННЯ ВІД АНОМАЛІЙ (Група 3, п. 3.1)
# -----------------------------------------------------------------------------

def adaptive_ema_anomaly_detector(data: np.ndarray, alpha: float = 0.1, z_thresh: float = 3.0) -> np.ndarray:
    """
    R&D: Алгоритм очищення на базі експоненційного ковзного середнього (EMA).
    Параметри алгоритму "навчаються" локальній дисперсії даних.
    """
    n: int = len(data)
    cleaned: np.ndarray = data.copy()
    
    ema: float = float(data[0])
    emvar: float = 0.0 # Експоненційна дисперсія
    
    for i in range(1, n):
        diff: float = float(data[i]) - ema
        
        # Оновлення локальної дисперсії
        emvar = (1 - alpha) * (emvar + alpha * diff**2)
        std_dev: float = np.sqrt(emvar)
        
        # Динамічний поріг виявлення
        if i > 10 and abs(diff) > z_thresh * std_dev:
            # Детекція: замінюємо аномалію на поточне математичне сподівання
            cleaned[i] = ema
        else:
            # Навчання: оновлюємо модель тільки на чистих даних
            ema = ema + alpha * diff
            
    return cleaned

# -----------------------------------------------------------------------------
# 3. ПОЛІНОМІАЛЬНИЙ МНК ТА АЛЬФА-БЕТА ФІЛЬТР (Група 1 і 2)
# -----------------------------------------------------------------------------

def polynomial_lsm(data: np.ndarray, forecast_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Класичний МНК для полінома 2-го ступеня (Група 1, п.5-6)."""
    n: int = len(data)
    total_steps: int = int(n * (1 + forecast_ratio))
    
    Yin: np.ndarray = data.reshape(-1, 1)
    F: np.ndarray = np.ones((n, 3))
    for i in range(n):
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        
    FT: np.ndarray = F.T
    C: np.ndarray = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    
    smoothed: np.ndarray = F.dot(C).flatten()
    
    F_extrapol: np.ndarray = np.ones((total_steps, 3))
    for i in range(total_steps):
        F_extrapol[i, 1] = float(i)
        F_extrapol[i, 2] = float(i * i)
        
    extrapolated: np.ndarray = F_extrapol.dot(C).flatten()
    return smoothed, extrapolated, np.arange(total_steps)

def alpha_beta_filter(data: np.ndarray, alpha_min: float = 0.05, beta_min: float = 0.001) -> np.ndarray:
    """Рекурентний a-b фільтр із захистом від розбіжності (Група 2, п.5)."""
    n: int = len(data)
    y_out: np.ndarray = np.zeros(n)
    
    y_est: float = float(data[0])
    v_est: float = float(data[1] - data[0]) if n > 1 else 0.0
    y_out[0] = y_est
    
    for i in range(1, n):
        y_pred: float = y_est + v_est
        
        alpha: float = max((2.0 * (2.0 * i - 1.0)) / (i * (i + 1.0)), alpha_min)
        beta: float = max(6.0 / (i * (i + 1.0)), beta_min)
        
        residual: float = float(data[i]) - y_pred
        y_est = y_pred + alpha * residual
        v_est = v_est + beta * residual
        y_out[i] = y_est
        
    return y_out

# -----------------------------------------------------------------------------
# 4. R&D: НЕЛІНІЙНЕ НАВЧАННЯ МНК (Група 3, п. 3.2)
# -----------------------------------------------------------------------------

def custom_exponential_lsm(data: np.ndarray, forecast_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    R&D: Власний алгоритм МНК для нелінійної моделі y = A * exp(B * t).
    Використовує логарифмічну лінеаризацію: ln(y) = ln(A) + B * t.
    """
    n: int = len(data)
    
    # Зсув для уникнення логарифма від від'ємних чисел (через шум)
    min_val: float = float(np.min(data))
    shift: float = abs(min_val) + 1.0 if min_val <= 0 else 0.0
    
    y_shifted: np.ndarray = data + shift
    ln_y: np.ndarray = np.log(y_shifted).reshape(-1, 1)
    
    # Формування матриць для лінійної моделі ln(y)
    F: np.ndarray = np.ones((n, 2))
    F[:, 1] = np.arange(n)
    
    # Матричні операції МНК: C = (F^T * F)^(-1) * F^T * ln_y
    FT: np.ndarray = F.T
    C: np.ndarray = np.linalg.inv(FT.dot(F)).dot(FT).dot(ln_y)
    
    ln_A: float = float(C[0, 0])
    B: float = float(C[1, 0])
    A: float = float(np.exp(ln_A))
    
    print(f"\n--- R&D Нелінійна модель ---")
    print(f"y(t) = {A:.4f} * exp({B:.6f} * t) - {shift:.4f}")
    
    # Прогнозування (Екстраполяція нелінійного процесу)
    total_steps: int = int(n * (1 + forecast_ratio))
    t_all: np.ndarray = np.arange(total_steps)
    
    smoothed: np.ndarray = A * np.exp(B * np.arange(n)) - shift
    extrapolated: np.ndarray = A * np.exp(B * t_all) - shift
    
    return smoothed, extrapolated, t_all

# -----------------------------------------------------------------------------
# ГОЛОВНИЙ БЛОК
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N_SAMPLES: int = 800
    
    print("1. Генерація експоненційної вибірки з аномаліями...")
    trend: np.ndarray
    data_norm: np.ndarray
    data_anom: np.ndarray
    trend, data_norm, data_anom = generate_nonlinear_data(N_SAMPLES)
    
    print("2. R&D Очищення вибірки (Адаптивний EMA детектор)...")
    data_clean: np.ndarray = adaptive_ema_anomaly_detector(data_anom, alpha=0.1, z_thresh=2.5)
    
    print("3. Навчання класичної поліноміальної моделі...")
    poly_smoothed: np.ndarray
    poly_extrapol: np.ndarray
    t_extrapol: np.ndarray
    poly_smoothed, poly_extrapol, t_extrapol = polynomial_lsm(data_clean, forecast_ratio=0.5)
    r2_poly: float = calculate_r2(data_clean, poly_smoothed)
    
    print("4. Рекурентна фільтрація Alpha-Beta...")
    ab_smoothed: np.ndarray = alpha_beta_filter(data_clean)
    r2_ab: float = calculate_r2(data_clean, ab_smoothed)
    
    print("5. R&D Навчання нелінійної (експоненційної) моделі...")
    exp_smoothed: np.ndarray
    exp_extrapol: np.ndarray
    exp_smoothed, exp_extrapol, _ = custom_exponential_lsm(data_clean, forecast_ratio=0.5)
    r2_exp: float = calculate_r2(data_clean, exp_smoothed)
    
    # Аналіз та оптимізація моделі (Група 1, п.4)
    print("\n--- ПОРІВНЯЛЬНИЙ АНАЛІЗ ЯКОСТІ МОДЕЛЕЙ (R^2) ---")
    print(f"Класичний поліном 2-го ступеня: {r2_poly:.4f}")
    print(f"Альфа-Бета фільтр (згладжування): {r2_ab:.4f}")
    print(f"R&D Нелінійна експоненційна модель: {r2_exp:.4f}")
    print("ВИСНОВОК: Експоненційна модель краще апроксимує фізичну суть згенерованого процесу.")
    
    # --- ВІЗУАЛІЗАЦІЯ ---
    plt.figure(figsize=(16, 12))

    # Графік 1: Робота R&D детектора аномалій
    plt.subplot(3, 1, 1)
    plt.plot(data_anom, label='Вхідні дані з аномаліями', color='lightgray', marker='.', linestyle='none')
    plt.plot(data_clean, label='R&D Адаптивне очищення', color='blue', alpha=0.7)
    plt.title("Група 3.1: Адаптивний EMA детектор аномалій (Динамічний поріг)")
    plt.legend()
    plt.grid(True)

    # Графік 2: Порівняння класики (Поліном) та R&D (Експонента)
    plt.subplot(3, 1, 2)
    plt.plot(data_clean, color='blue', alpha=0.2, label='Очищені дані')
    plt.plot(t_extrapol, poly_extrapol, label=f'Поліноміальна екстраполяція (R²={r2_poly:.2f})', color='red', linestyle='--')
    plt.plot(t_extrapol, exp_extrapol, label=f'R&D Експоненційна екстраполяція (R²={r2_exp:.2f})', color='green', linewidth=2)
    plt.axvline(x=N_SAMPLES, color='purple', linestyle=':', label='Початок прогнозу')
    plt.title("Група 1 та 3.2: Оптимізація вибору моделі (Поліном vs Експонента)")
    plt.legend()
    plt.grid(True)
    
    # Графік 3: Альфа-бета фільтр
    plt.subplot(3, 1, 3)
    plt.plot(data_clean, color='blue', alpha=0.2, label='Очищені дані')
    plt.plot(ab_smoothed, label=f'Альфа-Бета фільтр (R²={r2_ab:.2f})', color='orange', linewidth=2)
    plt.title("Група 2: Рекурентна Альфа-Бета фільтрація із захистом від розбіжності")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_filepath: pathlib.Path = BUILD_DIR / 'lab2_level4_results.svg'
    plt.savefig(output_filepath, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"\nГрафіки успішно збережено у: {output_filepath}")