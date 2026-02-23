import os
import pathlib
import requests
import dataclasses
import matplotlib.pyplot
import numpy
import datetime

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

@dataclasses.dataclass
class OpenMeteoParams:
    latitude: float
    longitude: float
    daily: list[str]
    past_days: int
    forecast_days: int
    timezone: str

@dataclasses.dataclass
class TemperatureData:
    dates: list[str]
    temperatures: list[float]

def get_historical_data(past_days: int) -> TemperatureData:
    url = "https://api.open-meteo.com/v1/forecast"
    
    request_params = OpenMeteoParams(
        latitude=50.4501,
        longitude=30.5234,
        daily=["temperature_2m_max"],
        past_days=past_days,
        forecast_days=1,
        timezone="Europe/Kyiv"
    )
    
    response = requests.get(url, params=dataclasses.asdict(request_params))
    data = response.json()
    
    dates = data["daily"]["time"][:-1]
    temps = data["daily"]["temperature_2m_max"][:-1]
    
    return TemperatureData(dates=dates, temperatures=temps)

def forecast_temperatures(history: TemperatureData, days_ahead: int) -> TemperatureData:
    x_hist = numpy.arange(len(history.dates))
    y_hist = numpy.array(history.temperatures)
    
    model_coefficients = numpy.polyfit(x_hist, y_hist, 2)
    predict_function = numpy.poly1d(model_coefficients)
    
    last_date_str = history.dates[-1]
    last_date = datetime.date.fromisoformat(last_date_str)
    
    future_dates: list[str] = []
    future_temps: list[float] = []
    
    start_idx = len(x_hist)
    for i in range(days_ahead):
        current_idx = start_idx + i
        predicted_temp = predict_function(current_idx)
        predicted_date = last_date + datetime.timedelta(days=i + 1)
        future_dates.append(predicted_date.isoformat())
        future_temps.append(float(predicted_temp))
        
    return TemperatureData(dates=future_dates, temperatures=future_temps)

def build_olap_cube(history: TemperatureData, forecast: TemperatureData) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    all_dates = history.dates + forecast.dates
    total_days = len(all_dates)
    x_positions = numpy.arange(total_days)
    
    fig = matplotlib.pyplot.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    hist_x = numpy.arange(len(history.dates))
    ax.bar(
        hist_x, 
        history.temperatures, 
        zs=1, 
        zdir='y', 
        color='#4b5de4', 
        alpha=0.8, 
        label='Історичні дані (Факт)'
    )
    
    forecast_x = numpy.arange(len(history.dates), total_days)
    ax.bar(
        forecast_x, 
        forecast.temperatures, 
        zs=2, 
        zdir='y', 
        color='#EAA228', 
        alpha=0.8, 
        label='Data Mining (Прогноз)'
    )
    
    ax.set_xlabel('Дати')
    ax.set_ylabel('Вимір OLAP (1=Факт, 2=Прогноз)')
    ax.set_zlabel('Температура (°C)')
    
    ax.set_yticks([1, 2])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_dates, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left')
    
    output_filepath = BUILD_DIR / 'temperature_prediction_olap.svg'
    matplotlib.pyplot.savefig(output_filepath, bbox_inches='tight', dpi=300)
    matplotlib.pyplot.close(fig)

def main() -> None:
    history = get_historical_data(past_days=30)
    forecast = forecast_temperatures(history, days_ahead=7)
    build_olap_cube(history, forecast)

if __name__ == '__main__':
    main()