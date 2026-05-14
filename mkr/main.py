from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

Direction = Literal["max", "min"]


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    weight: float
    direction: Direction
    unit: str


@dataclass(frozen=True, slots=True)
class Alternative:
    name: str
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    name: str
    normalized_values: dict[str, float]
    additive_score: float
    nonlinear_score: float
    topsis_score: float
    pareto_efficient: bool


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    names: tuple[str, ...]
    total_cost: float
    average_score: float
    portfolio_score: float


def create_criteria() -> tuple[Criterion, ...]:
    return (
        Criterion("throughput_mbps", 0.16, "max", "Mbps"),
        Criterion("coverage_km", 0.16, "max", "km"),
        Criterion("autonomy_h", 0.13, "max", "h"),
        Criterion("availability", 0.16, "max", "probability"),
        Criterion("security_score", 0.14, "max", "points"),
        Criterion("mobility_score", 0.10, "max", "points"),
        Criterion("deployment_time_min", 0.08, "min", "min"),
        Criterion("cost_usd", 0.07, "min", "USD"),
    )


def create_alternatives() -> tuple[Alternative, ...]:
    return (
        Alternative(
            "LTE field router",
            {
                "throughput_mbps": 80.0,
                "coverage_km": 10.0,
                "autonomy_h": 8.0,
                "availability": 0.88,
                "security_score": 7.5,
                "mobility_score": 8.0,
                "deployment_time_min": 6.0,
                "cost_usd": 1200.0,
            },
        ),
        Alternative(
            "VHF UHF handheld radio",
            {
                "throughput_mbps": 0.064,
                "coverage_km": 35.0,
                "autonomy_h": 24.0,
                "availability": 0.92,
                "security_score": 6.0,
                "mobility_score": 9.0,
                "deployment_time_min": 3.0,
                "cost_usd": 650.0,
            },
        ),
        Alternative(
            "Satellite terminal",
            {
                "throughput_mbps": 15.0,
                "coverage_km": 1000.0,
                "autonomy_h": 6.0,
                "availability": 0.96,
                "security_score": 8.0,
                "mobility_score": 5.0,
                "deployment_time_min": 10.0,
                "cost_usd": 4500.0,
            },
        ),
        Alternative(
            "Mesh network node",
            {
                "throughput_mbps": 20.0,
                "coverage_km": 5.0,
                "autonomy_h": 14.0,
                "availability": 0.86,
                "security_score": 8.5,
                "mobility_score": 8.0,
                "deployment_time_min": 8.0,
                "cost_usd": 900.0,
            },
        ),
        Alternative(
            "Smartphone gateway",
            {
                "throughput_mbps": 50.0,
                "coverage_km": 8.0,
                "autonomy_h": 10.0,
                "availability": 0.75,
                "security_score": 5.5,
                "mobility_score": 10.0,
                "deployment_time_min": 2.0,
                "cost_usd": 700.0,
            },
        ),
        Alternative(
            "Portable SDR relay",
            {
                "throughput_mbps": 5.0,
                "coverage_km": 20.0,
                "autonomy_h": 12.0,
                "availability": 0.82,
                "security_score": 7.0,
                "mobility_score": 7.0,
                "deployment_time_min": 12.0,
                "cost_usd": 1800.0,
            },
        ),
    )


def ensure_valid_weights(criteria: tuple[Criterion, ...]) -> None:
    weight_sum = sum(criterion.weight for criterion in criteria)
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Weight sum must be 1.0, got {weight_sum:.6f}")


def ensure_complete_values(criteria: tuple[Criterion, ...], alternatives: tuple[Alternative, ...]) -> None:
    expected = {criterion.name for criterion in criteria}
    for alternative in alternatives:
        actual = set(alternative.values.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Invalid values for {alternative.name}: missing={missing}, extra={extra}")


def criterion_bounds(criteria: tuple[Criterion, ...], alternatives: tuple[Alternative, ...]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for criterion in criteria:
        values = [alternative.values[criterion.name] for alternative in alternatives]
        bounds[criterion.name] = (min(values), max(values))
    return bounds


def normalize_value(value: float, lower: float, upper: float, direction: Direction) -> float:
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-12):
        return 1.0
    if direction == "max":
        return (value - lower) / (upper - lower)
    return (upper - value) / (upper - lower)


def normalize_alternative(
    criteria: tuple[Criterion, ...],
    alternative: Alternative,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for criterion in criteria:
        lower, upper = bounds[criterion.name]
        normalized[criterion.name] = normalize_value(
            alternative.values[criterion.name],
            lower,
            upper,
            criterion.direction,
        )
    return normalized


def additive_score(criteria: tuple[Criterion, ...], normalized: dict[str, float]) -> float:
    return sum(criterion.weight * normalized[criterion.name] for criterion in criteria)


def nonlinear_compromise_score(criteria: tuple[Criterion, ...], normalized: dict[str, float]) -> float:
    epsilon = 1e-9
    result = 1.0
    for criterion in criteria:
        result *= (epsilon + normalized[criterion.name]) ** criterion.weight
    return result


def weighted_distance_to_point(
    criteria: tuple[Criterion, ...],
    normalized: dict[str, float],
    point_value: float,
) -> float:
    return math.sqrt(
        sum(
            criterion.weight * (normalized[criterion.name] - point_value) ** 2
            for criterion in criteria
        )
    )


def topsis_score(criteria: tuple[Criterion, ...], normalized: dict[str, float]) -> float:
    distance_to_ideal = weighted_distance_to_point(criteria, normalized, 1.0)
    distance_to_anti_ideal = weighted_distance_to_point(criteria, normalized, 0.0)
    denominator = distance_to_ideal + distance_to_anti_ideal
    if math.isclose(denominator, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return 0.0
    return distance_to_anti_ideal / denominator


def is_better_or_equal(left: float, right: float, direction: Direction) -> bool:
    if direction == "max":
        return left >= right
    return left <= right


def is_strictly_better(left: float, right: float, direction: Direction) -> bool:
    if direction == "max":
        return left > right
    return left < right


def dominates(left: Alternative, right: Alternative, criteria: tuple[Criterion, ...]) -> bool:
    better_or_equal = all(
        is_better_or_equal(left.values[criterion.name], right.values[criterion.name], criterion.direction)
        for criterion in criteria
    )
    strictly_better = any(
        is_strictly_better(left.values[criterion.name], right.values[criterion.name], criterion.direction)
        for criterion in criteria
    )
    return better_or_equal and strictly_better


def pareto_flags(criteria: tuple[Criterion, ...], alternatives: tuple[Alternative, ...]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for alternative in alternatives:
        dominated = any(
            dominates(other, alternative, criteria)
            for other in alternatives
            if other.name != alternative.name
        )
        flags[alternative.name] = not dominated
    return flags


def evaluate(criteria: tuple[Criterion, ...], alternatives: tuple[Alternative, ...]) -> list[EvaluationResult]:
    ensure_valid_weights(criteria)
    ensure_complete_values(criteria, alternatives)
    bounds = criterion_bounds(criteria, alternatives)
    flags = pareto_flags(criteria, alternatives)
    results: list[EvaluationResult] = []
    for alternative in alternatives:
        normalized = normalize_alternative(criteria, alternative, bounds)
        results.append(
            EvaluationResult(
                alternative.name,
                normalized,
                additive_score(criteria, normalized),
                nonlinear_compromise_score(criteria, normalized),
                topsis_score(criteria, normalized),
                flags[alternative.name],
            )
        )
    return sorted(results, key=lambda item: item.additive_score, reverse=True)


def result_by_name(results: list[EvaluationResult]) -> dict[str, EvaluationResult]:
    return {result.name: result for result in results}


def alternative_by_name(alternatives: tuple[Alternative, ...]) -> dict[str, Alternative]:
    return {alternative.name: alternative for alternative in alternatives}


def select_portfolio(
    alternatives: tuple[Alternative, ...],
    results: list[EvaluationResult],
    budget: float,
    max_items: int,
) -> PortfolioResult:
    scores = result_by_name(results)
    best = PortfolioResult((), 0.0, 0.0, -1.0)
    for size in range(1, max_items + 1):
        for combination in itertools.combinations(alternatives, size):
            total_cost = sum(item.values["cost_usd"] for item in combination)
            if total_cost > budget:
                continue
            average_score = sum(scores[item.name].additive_score for item in combination) / size
            portfolio_score = average_score * math.log2(size + 1.0)
            if portfolio_score > best.portfolio_score:
                best = PortfolioResult(
                    tuple(item.name for item in combination),
                    total_cost,
                    average_score,
                    portfolio_score,
                )
    return best


def write_criteria(path: Path, criteria: tuple[Criterion, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("criterion", "weight", "direction", "unit"))
        for criterion in criteria:
            writer.writerow((criterion.name, f"{criterion.weight:.4f}", criterion.direction, criterion.unit))


def write_source_values(path: Path, criteria: tuple[Criterion, ...], alternatives: tuple[Alternative, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("alternative", *(criterion.name for criterion in criteria)))
        for alternative in alternatives:
            writer.writerow((alternative.name, *(alternative.values[criterion.name] for criterion in criteria)))


def write_normalized_values(path: Path, criteria: tuple[Criterion, ...], results: list[EvaluationResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("alternative", *(criterion.name for criterion in criteria)))
        for result in results:
            writer.writerow((result.name, *(f"{result.normalized_values[criterion.name]:.4f}" for criterion in criteria)))


def write_ranking(path: Path, results: list[EvaluationResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("rank", "alternative", "additive_score", "nonlinear_score", "topsis_score", "pareto_efficient"))
        for index, result in enumerate(results, start=1):
            writer.writerow(
                (
                    index,
                    result.name,
                    f"{result.additive_score:.4f}",
                    f"{result.nonlinear_score:.4f}",
                    f"{result.topsis_score:.4f}",
                    str(result.pareto_efficient),
                )
            )


def write_summary(path: Path, results: list[EvaluationResult], portfolio: PortfolioResult) -> None:
    best = results[0]
    with path.open("w", encoding="utf-8") as file:
        file.write(f"Best alternative: {best.name}\n")
        file.write(f"Additive score: {best.additive_score:.4f}\n")
        file.write(f"Nonlinear compromise score: {best.nonlinear_score:.4f}\n")
        file.write(f"TOPSIS score: {best.topsis_score:.4f}\n")
        file.write(f"Pareto efficient: {best.pareto_efficient}\n")
        file.write("\n")
        file.write(f"Best portfolio: {', '.join(portfolio.names)}\n")
        file.write(f"Portfolio cost: {portfolio.total_cost:.2f}\n")
        file.write(f"Average score: {portfolio.average_score:.4f}\n")
        file.write(f"Portfolio score: {portfolio.portfolio_score:.4f}\n")


def plot_ranking(path: Path, results: list[EvaluationResult]) -> None:
    names = [result.name for result in results]
    scores = [result.additive_score for result in results]
    positions = list(range(len(results)))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(positions, scores)
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.invert_yaxis()
    axis.set_xlabel("Integrated effectiveness score")
    axis.set_title("Ranking of mobile communication alternatives")
    axis.set_xlim(0.0, 1.0)
    for position, score in zip(positions, scores, strict=True):
        axis.text(score + 0.01, position, f"{score:.3f}", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_cost_effectiveness(path: Path, alternatives: tuple[Alternative, ...], results: list[EvaluationResult]) -> None:
    alternatives_map = alternative_by_name(alternatives)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for result in results:
        alternative = alternatives_map[result.name]
        cost = alternative.values["cost_usd"]
        marker = "o" if result.pareto_efficient else "x"
        axis.scatter(cost, result.additive_score, marker=marker, s=80)
        axis.annotate(result.name, (cost, result.additive_score), textcoords="offset points", xytext=(6, 5))
    axis.set_xlabel("Cost, USD")
    axis.set_ylabel("Integrated effectiveness score")
    axis.set_title("Cost and effectiveness with Pareto-efficient alternatives")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run() -> None:
    build_dir = Path("build")
    build_dir.mkdir(parents=True, exist_ok=True)
    criteria = create_criteria()
    alternatives = create_alternatives()
    results = evaluate(criteria, alternatives)
    portfolio = select_portfolio(alternatives, results, budget=5000.0, max_items=2)
    write_criteria(build_dir / "criteria.csv", criteria)
    write_source_values(build_dir / "source_values.csv", criteria, alternatives)
    write_normalized_values(build_dir / "normalized_values.csv", criteria, results)
    write_ranking(build_dir / "ranking.csv", results)
    write_summary(build_dir / "summary.txt", results, portfolio)
    plot_ranking(build_dir / "effectiveness_ranking.png", results)
    plot_cost_effectiveness(build_dir / "cost_effectiveness_pareto.png", alternatives, results)
    print((build_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    run()
