import os
import pathlib
import requests
import dataclasses
import matplotlib.pyplot
import numpy
import datetime
import torch
import sklearn.preprocessing

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

@dataclasses.dataclass
class TemperatureData:
    dates: list[str]
    temperatures: list[float]

@dataclasses.dataclass
class MLHyperparameters:
    window_size: int
    learning_rate: float
    epochs: int

@dataclasses.dataclass
class TrainingMetrics:
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float

@dataclasses.dataclass
class GridSearchResult:
    best_params: MLHyperparameters
    best_model_state: dict[str, torch.Tensor]
    metrics: TrainingMetrics
    scaler: sklearn.preprocessing.MinMaxScaler

def get_extended_historical_data(total_days: int) -> TemperatureData:
    today = datetime.date.today()
    archive_end_date = today - datetime.timedelta(days=7)
    archive_start_date = today - datetime.timedelta(days=total_days)

    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    archive_params = {
        "latitude": 50.4501,
        "longitude": 30.5234,
        "start_date": archive_start_date.isoformat(),
        "end_date": archive_end_date.isoformat(),
        "daily": ["temperature_2m_max"],
        "timezone": "Europe/Kyiv"
    }
    archive_response = requests.get(archive_url, params=archive_params)
    archive_data = archive_response.json()

    recent_url = "https://api.open-meteo.com/v1/forecast"
    recent_params = {
        "latitude": 50.4501,
        "longitude": 30.5234,
        "past_days": 6,
        "forecast_days": 1,
        "daily": ["temperature_2m_max"],
        "timezone": "Europe/Kyiv"
    }
    recent_response = requests.get(recent_url, params=recent_params)
    recent_data = recent_response.json()

    combined_dates = archive_data["daily"]["time"] + recent_data["daily"]["time"][:-1]
    combined_temps = archive_data["daily"]["temperature_2m_max"] + recent_data["daily"]["temperature_2m_max"][:-1]

    return TemperatureData(dates=combined_dates, temperatures=combined_temps)

def create_time_series_sequences(data: numpy.ndarray, window_size: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    x_sequences: list[numpy.ndarray] = []
    y_targets: list[float] = []
    
    for i in range(len(data) - window_size):
        x_sequences.append(data[i : i + window_size, 0])
        y_targets.append(float(data[i + window_size, 0]))
        
    return numpy.array(x_sequences), numpy.array(y_targets)

def train_and_evaluate_model(
    x_train: torch.Tensor, y_train: torch.Tensor, 
    x_val: torch.Tensor, y_val: torch.Tensor, 
    params: MLHyperparameters
) -> tuple[dict[str, torch.Tensor], TrainingMetrics]:
    
    model = torch.nn.Sequential(
        torch.nn.Linear(params.window_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1)
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float('inf')
    best_state_dict: dict[str, torch.Tensor] = {}

    for epoch in range(params.epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(x_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            
        current_train_loss = float(train_loss.item())
        current_val_loss = float(val_loss.item())
        
        train_losses.append(current_train_loss)
        val_losses.append(current_val_loss)
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    metrics = TrainingMetrics(
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss
    )
    
    return best_state_dict, metrics

def perform_grid_search(
    history: TemperatureData, 
    window_sizes: list[int], 
    learning_rates: list[float], 
    epochs: int = 150
) -> GridSearchResult:
    
    raw_temps = numpy.array(history.temperatures).reshape(-1, 1)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_temps = scaler.fit_transform(raw_temps)
    
    best_overall_val_loss = float('inf')
    best_overall_params: MLHyperparameters = MLHyperparameters(0, 0.0, 0)
    best_overall_state: dict[str, torch.Tensor] = {}
    best_overall_metrics: TrainingMetrics = TrainingMetrics([], [], 0.0)

    print("=== ПОЧАТОК GRID SEARCH ===")
    
    for w_size in window_sizes:
        for lr in learning_rates:
            x_seq, y_target = create_time_series_sequences(scaled_temps, w_size)
            
            split_idx = int(len(x_seq) * 0.8)
            x_train_np, x_val_np = x_seq[:split_idx], x_seq[split_idx:]
            y_train_np, y_val_np = y_target[:split_idx], y_target[split_idx:]
            
            x_train = torch.tensor(x_train_np, dtype=torch.float32)
            y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
            x_val = torch.tensor(x_val_np, dtype=torch.float32)
            y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
            
            current_params = MLHyperparameters(window_size=w_size, learning_rate=lr, epochs=epochs)
            
            best_state, metrics = train_and_evaluate_model(x_train, y_train, x_val, y_val, current_params)
            
            print(f"Вікно: {w_size:2d} | LR: {lr:.4f} --> Train Loss: {metrics.train_losses[-1]:.6f} | Val Loss: {metrics.best_val_loss:.6f}")
            
            if metrics.best_val_loss < best_overall_val_loss:
                best_overall_val_loss = metrics.best_val_loss
                best_overall_params = current_params
                best_overall_state = best_state
                best_overall_metrics = metrics

    print("=== ЗАВЕРШЕННЯ GRID SEARCH ===")
    print(f"Найкращі параметри: Вікно = {best_overall_params.window_size}, LR = {best_overall_params.learning_rate}")
    print(f"Найкращий результат валідації (MSE): {best_overall_val_loss:.6f}\n")

    return GridSearchResult(
        best_params=best_overall_params,
        best_model_state=best_overall_state,
        metrics=best_overall_metrics,
        scaler=scaler
    )

def plot_learning_curves(metrics: TrainingMetrics) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    fig = matplotlib.pyplot.figure(figsize=(10, 5))
    
    epochs_range = numpy.arange(1, len(metrics.train_losses) + 1)
    matplotlib.pyplot.plot(epochs_range, metrics.train_losses, label='Train Loss (MSE)', color='#4b5de4', linewidth=2)
    matplotlib.pyplot.plot(epochs_range, metrics.val_losses, label='Validation Loss (MSE)', color='#EAA228', linewidth=2)
    
    matplotlib.pyplot.title('Криві навчання: Оцінка перенавчання (Overfitting / Underfitting)')
    matplotlib.pyplot.xlabel('Епоха навчання')
    matplotlib.pyplot.ylabel('Середньоквадратична похибка (MSE)')
    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.6)
    
    output_filepath = BUILD_DIR / 'learning_curves_ann.svg'
    matplotlib.pyplot.savefig(output_filepath, bbox_inches='tight', dpi=300)
    matplotlib.pyplot.close(fig)

def forecast_future_ann(
    history: TemperatureData, 
    grid_result: GridSearchResult, 
    days_ahead: int
) -> TemperatureData:
    
    best_params = grid_result.best_params
    
    model = torch.nn.Sequential(
        torch.nn.Linear(best_params.window_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1)
    )
    model.load_state_dict(grid_result.best_model_state)
    model.eval()
    
    raw_temps = numpy.array(history.temperatures).reshape(-1, 1)
    scaled_temps = grid_result.scaler.transform(raw_temps)
    
    last_date_str = history.dates[-1]
    last_date = datetime.date.fromisoformat(last_date_str)
    
    future_dates: list[str] = []
    future_temps: list[float] = []
    
    current_window = scaled_temps[-best_params.window_size:, 0].tolist()
    
    with torch.no_grad():
        for i in range(days_ahead):
            current_input = torch.tensor([current_window], dtype=torch.float32)
            predicted_scaled = model(current_input).item()
            
            current_window.append(predicted_scaled)
            current_window.pop(0)
            
            predicted_temp = float(grid_result.scaler.inverse_transform([[predicted_scaled]])[0, 0])
            
            predicted_date = last_date + datetime.timedelta(days=i + 1)
            future_dates.append(predicted_date.isoformat())
            future_temps.append(predicted_temp)
            
    return TemperatureData(dates=future_dates, temperatures=future_temps)

def plot_temperature_forecast(history: TemperatureData, forecast: TemperatureData) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    visual_history_dates = history.dates[-60:]
    visual_history_temps = history.temperatures[-60:]
    
    all_dates = visual_history_dates + forecast.dates
    total_days = len(all_dates)
    x_positions = numpy.arange(total_days)
    
    fig = matplotlib.pyplot.figure(figsize=(14, 7))
    
    hist_x = numpy.arange(len(visual_history_dates))
    matplotlib.pyplot.plot(
        hist_x, 
        visual_history_temps, 
        color='#4b5de4', 
        marker='o', 
        markersize=4,
        linestyle='-', 
        linewidth=2,
        label='Історичні дані (Останні 60 днів)'
    )
    
    forecast_x = numpy.arange(len(visual_history_dates) - 1, total_days)
    forecast_temps = [visual_history_temps[-1]] + forecast.temperatures
    
    matplotlib.pyplot.plot(
        forecast_x, 
        forecast_temps, 
        color='#EAA228', 
        marker='X', 
        markersize=8,
        linestyle='--', 
        linewidth=2.5,
        label='ANN Прогноз (PyTorch)'
    )
    
    matplotlib.pyplot.title('Прогноз температури: PyTorch ANN з оптимізацією гіперпараметрів')
    matplotlib.pyplot.xlabel('Дати')
    matplotlib.pyplot.ylabel('Температура (°C)')
    
    matplotlib.pyplot.xticks(x_positions[::3], all_dates[::3], rotation=45, ha='right', fontsize=8)
    matplotlib.pyplot.legend(loc='upper left')
    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.6)
    
    output_filepath = BUILD_DIR / 'temperature_prediction_ann_optimized.svg'
    matplotlib.pyplot.savefig(output_filepath, bbox_inches='tight', dpi=300)
    matplotlib.pyplot.close(fig)

def main() -> None:
    print("Завантаження даних за 365 днів...")
    history = get_extended_historical_data(total_days=365)
    
    # Генерація параметрів за допомогою numpy.linspace
    # Вікна: 4 значення від 3 до 15 (приводяться до int: 3, 7, 11, 15)
    window_sizes_to_test: list[int] = [int(w) for w in numpy.linspace(3, 50, num=20)]
    
    # Швидкість навчання: 3 значення від 0.001 до 0.01 (0.001, 0.0055, 0.01)
    learning_rates_to_test: list[float] = [float(lr) for lr in numpy.linspace(0.0001, 0.01, num=30)]
    
    grid_result = perform_grid_search(
        history=history, 
        window_sizes=window_sizes_to_test, 
        learning_rates=learning_rates_to_test,
        epochs=200
    )
    
    plot_learning_curves(grid_result.metrics)
    
    forecast = forecast_future_ann(history, grid_result, days_ahead=7)
    
    plot_temperature_forecast(history, forecast)
    print("Усі графіки успішно збережено у директорію 'build/'")

if __name__ == '__main__':
    main()