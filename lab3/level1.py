import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_PATH: pathlib.Path = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR: pathlib.Path = SCRIPT_PATH.parent
BUILD_DIR: pathlib.Path = SCRIPT_DIR / 'build'

BUILD_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_data(filepath: pathlib.Path) -> None:
    data: dict[str, list[float | str]] = {
        'Housing': ['House_A', 'House_B', 'House_C', 'House_D', 'House_E', 'House_F', 'House_G', 'House_H'],
        'Area_m2': [45.0, 60.0, 85.0, 50.0, 120.0, 75.0, 90.0, 55.0],
        'Rooms': [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 3.0, 2.0],
        'Floor': [5.0, 2.0, 12.0, 8.0, 20.0, 3.0, 15.0, 1.0],
        'Balconies': [1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 0.0],
        'Price_USD': [40000.0, 55000.0, 80000.0, 48000.0, 150000.0, 70000.0, 95000.0, 38000.0],
        'Dist_Metro_min': [15.0, 5.0, 25.0, 10.0, 40.0, 8.0, 20.0, 12.0],
        'Crime_Rate': [4.0, 2.0, 3.0, 5.0, 1.0, 3.0, 2.0, 6.0],
        'Age_Years': [10.0, 5.0, 20.0, 15.0, 2.0, 8.0, 25.0, 30.0],
        'Dist_Center_km': [10.0, 5.0, 15.0, 8.0, 25.0, 6.0, 12.0, 9.0],
        'Utility_Cost': [80.0, 100.0, 150.0, 90.0, 250.0, 120.0, 160.0, 70.0],
        'Noise_Level_dB': [60.0, 50.0, 40.0, 65.0, 30.0, 55.0, 45.0, 70.0],
        'Dist_Supermarket_min': [5.0, 2.0, 10.0, 3.0, 15.0, 4.0, 8.0, 1.0]
    }
    df: pd.DataFrame = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def normalize_criteria(df: pd.DataFrame, max_cols: list[str], min_cols: list[str]) -> pd.DataFrame:
    norm_df: pd.DataFrame = df.copy()
    
    for col in max_cols:
        col_min: float = float(norm_df[col].min())
        col_max: float = float(norm_df[col].max())
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 1.0
            
    for col in min_cols:
        col_min: float = float(norm_df[col].min())
        col_max: float = float(norm_df[col].max())
        if col_max > col_min:
            norm_df[col] = (col_max - norm_df[col]) / (col_max - col_min)
        else:
            norm_df[col] = 1.0
            
    return norm_df

def calculate_integral_score(norm_df: pd.DataFrame, weights: list[float], cols: list[str]) -> np.ndarray:
    data_matrix: np.ndarray = norm_df[cols].values
    weights_array: np.ndarray = np.array(weights)
    scores: np.ndarray = np.dot(data_matrix, weights_array)
    return scores

def plot_ranking(labels: list[str], scores: np.ndarray, filepath: pathlib.Path) -> None:
    sorted_indices: np.ndarray = np.argsort(scores)[::-1]
    sorted_labels: list[str] = [labels[i] for i in sorted_indices]
    sorted_scores: np.ndarray = scores[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_labels, sorted_scores, color='skyblue', edgecolor='black')
    
    for bar in bars:
        yval: float = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
        
    plt.title('Housing Options Ranking (MCDA)')
    plt.xlabel('Housing Options')
    plt.ylabel('Integral Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    data_file: pathlib.Path = BUILD_DIR / 'housing_data.csv'
    
    if not data_file.exists():
        create_sample_data(data_file)
        
    df: pd.DataFrame = pd.read_csv(data_file)
    
    maximized_criteria: list[str] = ['Area_m2', 'Rooms', 'Floor', 'Balconies']
    minimized_criteria: list[str] = [
        'Price_USD', 'Dist_Metro_min', 'Crime_Rate', 'Age_Years', 
        'Dist_Center_km', 'Utility_Cost', 'Noise_Level_dB', 'Dist_Supermarket_min'
    ]
    
    all_criteria: list[str] = maximized_criteria + minimized_criteria
    
    criteria_weights: list[float] = [
        0.15, 0.05, 0.05, 0.05,
        0.20, 0.15, 0.10, 0.05, 
        0.05, 0.05, 0.05, 0.05  
    ]
    
    normalized_df: pd.DataFrame = normalize_criteria(df, maximized_criteria, minimized_criteria)
    
    scores: np.ndarray = calculate_integral_score(normalized_df, criteria_weights, all_criteria)
    df['Integral_Score'] = scores
    
    print("--- MCDA Results (Sorted) ---")
    sorted_df: pd.DataFrame = df[['Housing', 'Integral_Score']].sort_values(by='Integral_Score', ascending=False)
    for index, row in sorted_df.iterrows():
        print(f"{row['Housing']}: {row['Integral_Score']:.4f}")
        
    output_image_path: pathlib.Path = BUILD_DIR / 'housing_ranking.svg'
    plot_ranking(df['Housing'].tolist(), scores, output_image_path)
    print(f"\nRanking chart saved to: {output_image_path}")