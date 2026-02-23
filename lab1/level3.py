import os
import pathlib
import requests
import bs4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataclasses

SCRIPT_PATH: pathlib.Path = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR: pathlib.Path = SCRIPT_PATH.parent
BUILD_DIR: pathlib.Path = SCRIPT_DIR / 'build'

BUILD_DIR.mkdir(parents=True, exist_ok=True)

@dataclasses.dataclass
class PopulationRecord:
    year: int
    population: int

def parse_world_population(url: str) -> list[PopulationRecord]:
    headers: dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response: requests.Response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup: bs4.BeautifulSoup = bs4.BeautifulSoup(response.text, 'html.parser')
    table: bs4.element.Tag | None = soup.find('table')
    
    records: list[PopulationRecord] = []
    
    if table:
        tbody: bs4.element.Tag | None = table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                cols: bs4.ResultSet = row.find_all('td')
                if len(cols) >= 2:
                    year_str: str = cols[0].text.strip()
                    pop_str: str = cols[1].text.strip().replace(',', '')
                    try:
                        records.append(PopulationRecord(
                            year=int(year_str), 
                            population=int(pop_str)
                        ))
                    except ValueError:
                        continue
                        
    records.sort(key=lambda x: x.year)
    return records

def save_to_csv(records: list[PopulationRecord], filepath: pathlib.Path) -> None:
    df: pd.DataFrame = pd.DataFrame([r.__dict__ for r in records])
    df.to_csv(filepath, index=False)

def calculate_statistics(data: np.ndarray, label: str) -> tuple[float, float, float]:
    mean_val: float = float(np.mean(data))
    var_val: float = float(np.var(data))
    std_val: float = float(np.std(data))
    
    print(f"--- {label} ---")
    print(f"Expected Value: {mean_val:.4f}")
    print(f"Variance: {var_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print("-" * 30)
    
    return mean_val, var_val, std_val

def fit_polynomial_trend(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    coeffs: np.ndarray = np.polyfit(x, y, degree)
    trend: np.ndarray = np.polyval(coeffs, x)
    return trend

def synthesize_data(trend: np.ndarray, std_dev: float) -> np.ndarray:
    noise: np.ndarray = np.random.normal(0.0, std_dev, len(trend))
    return trend + noise

def plot_trend_analysis(x: np.ndarray, y_real: np.ndarray, y_trend: np.ndarray, y_synthetic: np.ndarray, filepath: pathlib.Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_real, label='Real Data (Parsed)', marker='o', color='blue', alpha=0.6)
    plt.plot(x, y_trend, label='Calculated Trend (Degree 2)', color='red', linewidth=2)
    plt.plot(x, y_synthetic, label='Synthesized Data', linestyle='--', color='green', alpha=0.7)
    plt.title("Trend Analysis and Data Synthesis")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_histograms(real_data: np.ndarray, synthetic_data: np.ndarray, filepath: pathlib.Path) -> None:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(real_data, bins=15, color='blue', edgecolor='black', alpha=0.6)
    plt.title("Real Data Histogram")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(synthetic_data, bins=15, color='green', edgecolor='black', alpha=0.6)
    plt.title("Synthetic Data Histogram")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":
    url: str = "https://www.worldometers.info/world-population/world-population-by-year/"
    
    parsed_records: list[PopulationRecord] = parse_world_population(url)
    
    if not parsed_records:
        print("Parsing failed or no data found.")
    else:
        csv_path: pathlib.Path = BUILD_DIR / 'parsed_population.csv'
        save_to_csv(parsed_records, csv_path)
        
        years: np.ndarray = np.array([r.year for r in parsed_records])
        population: np.ndarray = np.array([r.population for r in parsed_records], dtype=float)
        
        real_mean: float
        real_var: float
        real_std: float
        real_mean, real_var, real_std = calculate_statistics(population, "Parsed Real Data")
        
        trend: np.ndarray = fit_polynomial_trend(years, population, degree=2)
        
        residuals: np.ndarray = population - trend
        residual_mean: float
        residual_var: float
        residual_std: float
        residual_mean, residual_var, residual_std = calculate_statistics(residuals, "Trend Residuals (Noise)")
        
        synthetic_population: np.ndarray = synthesize_data(trend, residual_std)
        
        calculate_statistics(synthetic_population, "Synthesized Data")
        
        plot_trend_analysis(
            years, 
            population, 
            trend, 
            synthetic_population, 
            BUILD_DIR / 'trend_analysis.png'
        )
        
        plot_histograms(
            population, 
            synthetic_population, 
            BUILD_DIR / 'comparison_histograms.png'
        )