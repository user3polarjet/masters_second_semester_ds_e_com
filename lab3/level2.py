import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_PATH: pathlib.Path = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR: pathlib.Path = SCRIPT_PATH.parent
BUILD_DIR: pathlib.Path = SCRIPT_DIR / 'build'

BUILD_DIR.mkdir(parents=True, exist_ok=True)

def normalize_criteria(df: pd.DataFrame, max_cols: list[str], min_cols: list[str]) -> pd.DataFrame:
    """Мінімаксна нормалізація. Всі показники зводяться до діапазону [0; 1], де 1 - найкраще."""
    norm_df: pd.DataFrame = df.copy()
    
    # Максимізовані: чим більше, тим краще (1.0 = максимум)
    for col in max_cols:
        col_min: float = float(norm_df[col].min())
        col_max: float = float(norm_df[col].max())
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 1.0
            
    # Мінімізовані: чим менше, тим краще (1.0 = мінімум)
    for col in min_cols:
        col_min: float = float(norm_df[col].min())
        col_max: float = float(norm_df[col].max())
        if col_max > col_min:
            norm_df[col] = (col_max - norm_df[col]) / (col_max - col_min)
        else:
            norm_df[col] = 1.0
            
    return norm_df

def calculate_integral_score(norm_df: pd.DataFrame, weights: list[float], cols: list[str]) -> np.ndarray:
    """Розрахунок зваженої суми для кожної альтернативи."""
    data_matrix: np.ndarray = norm_df[cols].values
    weights_array: np.ndarray = np.array(weights)
    return np.dot(data_matrix, weights_array)

def plot_ranking(labels: list[str], scores: np.ndarray, transport_types: list[str], filepath: pathlib.Path) -> None:
    """Побудова та збереження горизонтального графіка SVG."""
    sorted_indices: np.ndarray = np.argsort(scores)[::-1]
    sorted_labels: list[str] = [labels[i] for i in sorted_indices]
    sorted_scores: np.ndarray = scores[sorted_indices]
    sorted_types: list[str] = [transport_types[i] for i in sorted_indices]
    
    # Кольори залежно від типу транспорту
    color_map = {'Car': '#ff9999', 'Transit': '#99ccff', 'Walk': '#99ff99'}
    colors = [color_map.get(t, 'gray') for t in sorted_types]
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(sorted_labels, sorted_scores, color=colors, edgecolor='black')
    
    # Додавання числових значень біля колонок
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.3f}', va='center', fontsize=11)
        
    plt.title('Route to University Ranking (MCDA)')
    plt.xlabel('Integral Effectiveness Score')
    plt.xlim(0, max(sorted_scores) * 1.15)
    plt.gca().invert_yaxis() # Найкращий маршрут зверху
    
    # Створення легенди
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
    plt.legend(handles=legend_handles, loc='lower right')
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Скрипт очікує, що файл routes_data.csv лежить поруч із ним
    data_file: pathlib.Path = SCRIPT_DIR / 'routes_data.csv'
    
    if not data_file.exists():
        print(f"Помилка: Файл {data_file.name} не знайдено у папці зі скриптом!")
        exit(1)
        
    df: pd.DataFrame = pd.read_csv(data_file)
    
    # 1. Визначення критеріїв
    minimized_criteria: list[str] = ['Total_Time_min', 'Walking_Time_min', 'Transfers_count', 'Cost_UAH', 'Traffic_Jam_Risk']
    maximized_criteria: list[str] = ['Comfort_Level', 'Reliability']
    all_criteria: list[str] = minimized_criteria + maximized_criteria
    
    # 2. Вагові коефіцієнти (сума = 1.0)
    # Порядок: Total_Time, Walking_Time, Transfers, Cost, Jam_Risk | Comfort, Reliability
    weights: list[float] = [0.30, 0.10, 0.05, 0.20, 0.10, 0.15, 0.10]
    
    # 3. Нормалізація
    normalized_df: pd.DataFrame = normalize_criteria(df, maximized_criteria, minimized_criteria)
    
    # 4. Обчислення інтегральної оцінки
    scores: np.ndarray = calculate_integral_score(normalized_df, weights, all_criteria)
    df['Integral_Score'] = scores
    
    # 5. Вивід результатів
    print("--- MCDA Route Effectiveness Ranking ---")
    sorted_df: pd.DataFrame = df[['Route_Name', 'Transport_Type', 'Integral_Score']].sort_values(by='Integral_Score', ascending=False)
    
    for index, row in sorted_df.iterrows():
        print(f"[{row['Transport_Type']:^7}] {row['Route_Name']:<25} : {row['Integral_Score']:.4f}")
        
    # 6. Збереження SVG графіка
    output_image_path: pathlib.Path = BUILD_DIR / 'routes_ranking.svg'
    plot_ranking(df['Route_Name'].tolist(), scores, df['Transport_Type'].tolist(), output_image_path)
    print(f"\nRanking chart saved to: {output_image_path}")