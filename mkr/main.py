from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Observation:
    day_index: int
    day: date
    cases: float


@dataclass(frozen=True)
class SmoothedObservation:
    day_index: int
    day: date
    cases: float
    moving_average: float


@dataclass(frozen=True)
class LinearTrend:
    intercept: float
    slope: float
    r_squared: float
    pearson: float
    first_value: float
    last_value: float
    percent_change: float
    direction: str


@dataclass(frozen=True)
class SegmentSummary:
    segment_index: int
    start_day: date
    end_day: date
    mean_cases: float
    min_cases: float
    max_cases: float
    slope: float
    direction: str


@dataclass(frozen=True)
class AnalysisResult:
    observations: tuple[Observation, ...]
    smoothed: tuple[SmoothedObservation, ...]
    trend: LinearTrend
    segments: tuple[SegmentSummary, ...]


@dataclass(frozen=True)
class SeriesConfig:
    start_date: date
    days: int
    seed: int
    base_level: float
    daily_linear_growth: float
    wave_amplitude: float
    wave_center: float
    wave_width: float
    weekly_amplitude: float
    noise_sigma: float


def gaussian_wave(x: float, center: float, width: float, amplitude: float) -> float:
    return amplitude * math.exp(-0.5 * ((x - center) / width) ** 2)


def generate_cases(config: SeriesConfig) -> tuple[Observation, ...]:
    rng = random.Random(config.seed)
    observations: list[Observation] = []
    for day_index in range(config.days):
        linear_component = config.base_level + config.daily_linear_growth * day_index
        wave_component = gaussian_wave(float(day_index), config.wave_center, config.wave_width, config.wave_amplitude)
        weekly_component = config.weekly_amplitude * math.sin(2.0 * math.pi * float(day_index) / 7.0)
        noise_component = rng.gauss(0.0, config.noise_sigma)
        cases = max(0.0, linear_component + wave_component + weekly_component + noise_component)
        observations.append(Observation(day_index, config.start_date + timedelta(days=day_index), round(cases, 2)))
    return tuple(observations)


def moving_average(observations: Sequence[Observation], window: int) -> tuple[SmoothedObservation, ...]:
    half = window // 2
    result: list[SmoothedObservation] = []
    for index, item in enumerate(observations):
        left = max(0, index - half)
        right = min(len(observations), index + half + 1)
        values = [point.cases for point in observations[left:right]]
        result.append(SmoothedObservation(item.day_index, item.day, item.cases, round(statistics.mean(values), 2)))
    return tuple(result)


def linear_regression(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[float, float]:
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0.0:
        return y_mean, 0.0
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return intercept, slope


def coefficient_of_determination(x_values: Sequence[float], y_values: Sequence[float], intercept: float, slope: float) -> float:
    y_mean = statistics.mean(y_values)
    total = sum((y - y_mean) ** 2 for y in y_values)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(x_values, y_values))
    if total == 0.0:
        return 1.0
    return 1.0 - residual / total


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denominator == 0.0 or y_denominator == 0.0:
        return 0.0
    return numerator / (x_denominator * y_denominator)


def define_direction(slope: float, threshold: float) -> str:
    if slope > threshold:
        return "зростаючий"
    if slope < -threshold:
        return "спадний"
    return "стабільний"


def estimate_linear_trend(smoothed: Sequence[SmoothedObservation]) -> LinearTrend:
    x_values = [float(item.day_index) for item in smoothed]
    y_values = [item.moving_average for item in smoothed]
    intercept, slope = linear_regression(x_values, y_values)
    first_value = intercept + slope * x_values[0]
    last_value = intercept + slope * x_values[-1]
    percent_change = ((last_value - first_value) / first_value) * 100.0 if first_value != 0.0 else 0.0
    r_squared = coefficient_of_determination(x_values, y_values, intercept, slope)
    pearson = pearson_correlation(x_values, y_values)
    direction = define_direction(slope, 0.05)
    return LinearTrend(
        intercept=round(intercept, 4),
        slope=round(slope, 4),
        r_squared=round(r_squared, 4),
        pearson=round(pearson, 4),
        first_value=round(first_value, 2),
        last_value=round(last_value, 2),
        percent_change=round(percent_change, 2),
        direction=direction,
    )


def split_segments(smoothed: Sequence[SmoothedObservation], segment_size: int) -> tuple[SegmentSummary, ...]:
    result: list[SegmentSummary] = []
    for segment_index, start in enumerate(range(0, len(smoothed), segment_size), start=1):
        segment = smoothed[start:start + segment_size]
        x_values = [float(item.day_index) for item in segment]
        y_values = [item.moving_average for item in segment]
        intercept, slope = linear_regression(x_values, y_values)
        values = [item.cases for item in segment]
        result.append(
            SegmentSummary(
                segment_index=segment_index,
                start_day=segment[0].day,
                end_day=segment[-1].day,
                mean_cases=round(statistics.mean(values), 2),
                min_cases=round(min(values), 2),
                max_cases=round(max(values), 2),
                slope=round(slope, 4),
                direction=define_direction(slope, 0.05),
            )
        )
    return tuple(result)


def analyze(config: SeriesConfig) -> AnalysisResult:
    observations = generate_cases(config)
    smoothed = moving_average(observations, 7)
    trend = estimate_linear_trend(smoothed)
    segments = split_segments(smoothed, 30)
    return AnalysisResult(observations, smoothed, trend, segments)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def save_observations(path: Path, observations: Sequence[Observation]) -> None:
    rows = ([str(item.day_index), item.day.isoformat(), f"{item.cases:.2f}"] for item in observations)
    write_csv(path, ["day_index", "date", "cases"], rows)


def save_smoothed(path: Path, smoothed: Sequence[SmoothedObservation]) -> None:
    rows = ([str(item.day_index), item.day.isoformat(), f"{item.cases:.2f}", f"{item.moving_average:.2f}"] for item in smoothed)
    write_csv(path, ["day_index", "date", "cases", "moving_average_7"], rows)


def save_segments(path: Path, segments: Sequence[SegmentSummary]) -> None:
    rows = (
        [
            str(item.segment_index),
            item.start_day.isoformat(),
            item.end_day.isoformat(),
            f"{item.mean_cases:.2f}",
            f"{item.min_cases:.2f}",
            f"{item.max_cases:.2f}",
            f"{item.slope:.4f}",
            item.direction,
        ]
        for item in segments
    )
    write_csv(path, ["segment", "start_date", "end_date", "mean", "min", "max", "slope", "direction"], rows)


def save_summary(path: Path, result: AnalysisResult) -> None:
    lines = [
        "Результати визначення тренду часового ряду захворюваності",
        f"Кількість дискретних вимірів: {len(result.observations)}",
        f"Напрям загального тренду: {result.trend.direction}",
        f"Коефіцієнт нахилу лінії тренду: {result.trend.slope:.4f} випадків/день",
        f"Оцінка на початку періоду: {result.trend.first_value:.2f}",
        f"Оцінка наприкінці періоду: {result.trend.last_value:.2f}",
        f"Відносна зміна за 180 днів: {result.trend.percent_change:.2f}%",
        f"Коефіцієнт детермінації R^2: {result.trend.r_squared:.4f}",
        f"Коефіцієнт кореляції Пірсона: {result.trend.pearson:.4f}",
        "",
        "Помесячна сегментація:",
    ]
    for item in result.segments:
        lines.append(
            f"{item.segment_index}: {item.start_day.isoformat()} - {item.end_day.isoformat()}, "
            f"середнє={item.mean_cases:.2f}, slope={item.slope:.4f}, напрям={item.direction}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trend(path: Path, result: AnalysisResult) -> None:
    x_values = [item.day_index for item in result.smoothed]
    cases = [item.cases for item in result.smoothed]
    averages = [item.moving_average for item in result.smoothed]
    trend_values = [result.trend.intercept + result.trend.slope * item.day_index for item in result.smoothed]
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, cases, linewidth=1.0, label="Денні виміри")
    plt.plot(x_values, averages, linewidth=2.0, label="Ковзне середнє 7 днів")
    plt.plot(x_values, trend_values, linewidth=2.0, linestyle="--", label="Лінія тренду")
    plt.xlabel("День спостереження")
    plt.ylabel("Кількість випадків")
    plt.title("Визначення тренду захворюваності за 180 днів")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_segments(path: Path, segments: Sequence[SegmentSummary]) -> None:
    labels = [f"{item.segment_index}" for item in segments]
    means = [item.mean_cases for item in segments]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, means)
    plt.xlabel("Номер 30-денного сегмента")
    plt.ylabel("Середня кількість випадків")
    plt.title("Середня захворюваність за 30-денними сегментами")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_pipeline(project_dir: Path) -> None:
    build_dir = project_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    config = SeriesConfig(
        start_date=date(2025, 1, 1),
        days=180,
        seed=31,
        base_level=95.0,
        daily_linear_growth=0.36,
        wave_amplitude=82.0,
        wave_center=92.0,
        wave_width=28.0,
        weekly_amplitude=9.0,
        noise_sigma=8.0,
    )
    result = analyze(config)
    save_observations(build_dir / "observations.csv", result.observations)
    save_smoothed(build_dir / "moving_average.csv", result.smoothed)
    save_segments(build_dir / "segment_summary.csv", result.segments)
    save_summary(build_dir / "summary.txt", result)
    plot_trend(build_dir / "covid_trend.png", result)
    plot_segments(build_dir / "segment_means.png", result.segments)
    print((build_dir / "summary.txt").read_text(encoding="utf-8"))


def main() -> None:
    run_pipeline(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
