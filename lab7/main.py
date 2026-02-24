import os
import pathlib
import datetime
import dataclasses
import contextlib
import typing

import pandas
import numpy
import matplotlib.pyplot

SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

@dataclasses.dataclass
class ForecastResult:
    region: str
    historical_dates: list[datetime.datetime]
    historical_sales: list[float]
    forecast_dates: list[datetime.datetime]
    forecast_sales: list[float]
    slope: float
    intercept: float

def perform_exploratory_data_analysis(df: pandas.DataFrame) -> None:
    print(df.head())
    print(df.info())
    print(df.describe(include='all'))
    print(df.isna().sum())

def preprocess_data(df: pandas.DataFrame) -> pandas.DataFrame:
    df_cleaned = df.copy()
    df_cleaned['OrderDate'] = pandas.to_datetime(df_cleaned['OrderDate'], errors='coerce')
    df_cleaned.dropna(subset=['OrderDate', 'Region', 'Total'], inplace=True)
    return df_cleaned

def aggregate_monthly_sales(df: pandas.DataFrame, region: str) -> pandas.DataFrame:
    df_region = df[df['Region'] == region].copy()
    df_region['Month'] = df_region['OrderDate'].dt.to_period('M')
    df_monthly = df_region.groupby('Month')['Total'].sum().reset_index()
    df_monthly['Month_DT'] = df_monthly['Month'].dt.to_timestamp()
    df_monthly.sort_values('Month_DT', inplace=True)
    df_monthly.reset_index(drop=True, inplace=True)
    return df_monthly

def fit_ordinary_least_squares(x_values: list[int], y_values: list[float]) -> tuple[float, float]:
    assert len(x_values) >= 2
    x_arr = numpy.array(x_values)
    y_arr = numpy.array(y_values)
    design_matrix = numpy.vstack([x_arr, numpy.ones(len(x_arr))]).T
    slope, intercept = numpy.linalg.lstsq(design_matrix, y_arr, rcond=None)[0]
    return float(slope), float(intercept)

def forecast_sales_for_region(df_monthly: pandas.DataFrame, region: str, months_to_predict: int) -> ForecastResult:
    dates = df_monthly['Month_DT'].tolist()
    sales = df_monthly['Total'].tolist()
    
    assert dates
    assert sales

    start_date = dates[0]
    x_hist = [(d - start_date).days for d in dates]
    
    slope, intercept = fit_ordinary_least_squares(x_hist, sales)
    
    forecast_dates = []
    forecast_sales = []
    last_date = dates[-1]
    
    for i in range(1, months_to_predict + 1):
        next_date = last_date + datetime.timedelta(days=30 * i)
        forecast_dates.append(next_date)
        x_pred = (next_date - start_date).days
        y_pred = slope * x_pred + intercept
        y_pred = max(0.0, y_pred)
        forecast_sales.append(y_pred)
        
    return ForecastResult(
        region=region,
        historical_dates=dates,
        historical_sales=sales,
        forecast_dates=forecast_dates,
        forecast_sales=forecast_sales,
        slope=slope,
        intercept=intercept
    )

def save_forecast_plot(result: ForecastResult) -> None:
    assert result.historical_dates
        
    fig, ax = matplotlib.pyplot.subplots(figsize=(12, 7))
    ax.plot(result.historical_dates, result.historical_sales, label='Historical Total', marker='o', color='#1f77b4', linewidth=2)
    ax.plot(result.forecast_dates, result.forecast_sales, label='Forecast Total (OLS)', linestyle='--', marker='s', color='#d62728', linewidth=2)
    
    start_date = result.historical_dates[0]
    all_dates = result.historical_dates + result.forecast_dates
    trend_x = [(d - start_date).days for d in all_dates]
    trend_y = [max(0.0, result.slope * x + result.intercept) for x in trend_x]
    
    ax.plot(all_dates, trend_y, label='Trend Line', linestyle=':', color='#2ca02c')
    ax.set_title(f'Performance Forecast for Region: {result.region}', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Revenue', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.autofmt_xdate()
    file_path = BUILD_DIR / f'forecast_plot_{result.region}.svg'
    fig.savefig(file_path, format='svg', bbox_inches='tight')
    matplotlib.pyplot.close(fig)

def save_forecast_table(result: ForecastResult) -> None:
    assert result.forecast_dates
        
    data_dict: dict[str, typing.Any] = {
        'Region': [result.region] * len(result.forecast_dates),
        'Forecast Date': [d.strftime('%Y-%m-%d') for d in result.forecast_dates],
        'Forecast Total': [round(s, 2) for s in result.forecast_sales]
    }
    
    df_forecast = pandas.DataFrame(data_dict)
    file_path = BUILD_DIR / f'forecast_table_{result.region}.csv'
    df_forecast.to_csv(file_path, index=False, encoding='utf-8')

def analyze_and_predict_by_region(df_cleaned: pandas.DataFrame) -> None:
    regions = df_cleaned['Region'].unique().tolist()
    for r in regions:
        df_monthly = aggregate_monthly_sales(df_cleaned, r)
        result = forecast_sales_for_region(df_monthly, r, 6)
        if result.historical_dates:
            save_forecast_plot(result)
            save_forecast_table(result)

def process_item_performance(df_cleaned: pandas.DataFrame) -> None:
    df_item = df_cleaned.groupby('Item')['Total'].sum().reset_index()
    df_item.sort_values('Total', ascending=False, inplace=True)
    
    file_path = BUILD_DIR / 'item_performance.csv'
    df_item.to_csv(file_path, index=False, encoding='utf-8')
    
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6))
    ax.bar(df_item['Item'], df_item['Total'], color='#9467bd')
    ax.set_title('Item Performance (Total Sales)', fontsize=14)
    ax.set_xlabel('Item', fontsize=12)
    ax.set_ylabel('Total Sales', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.autofmt_xdate()
    
    plot_path = BUILD_DIR / 'item_performance.svg'
    fig.savefig(plot_path, format='svg', bbox_inches='tight')
    matplotlib.pyplot.close(fig)

def main() -> None:
    with (
        open(BUILD_DIR / 'logs.txt', 'w') as logsio,
        contextlib.redirect_stdout(logsio)
    ):
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        df = pandas.read_csv(SCRIPT_DIR / 'Data_Set_tabl_2' / 'Data_Set_3.csv')
        perform_exploratory_data_analysis(df)
        df_cleaned = preprocess_data(df)
        analyze_and_predict_by_region(df_cleaned)
        process_item_performance(df_cleaned)

if __name__ == '__main__':
    main()