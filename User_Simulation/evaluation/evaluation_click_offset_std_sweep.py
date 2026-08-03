"""Evaluate Gaussian click-offset variability against held-out real clicks.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_click_offset_std_sweep
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from User_Simulation.evaluation.evaluation_baseline import (
    TEXT_DATA_ROOT,
    load_text_click_data,
    write_json,
)
from User_Simulation.evaluation.synthetic_profiles import split_sessions


CLOCK_PERIOD_COL = "Clock Period (s)"
CLICK_OFFSET_COL = "Click Time Relative (s)"
DEFAULT_USERS = ("A", "B", "C", "D", "F", "G")
DEFAULT_STD_MULTIPLIERS = (0.5, 1.0, 1.5)
METRIC_COLUMNS = (
    "quantile_mae_ms",
    "quantile_rmse_ms",
    "ks_statistic",
    "centered_ks_statistic",
    "wasserstein_ms",
    "mean_bias_ms",
    "std_difference_ms",
    "real_mean_ms",
    "synthetic_mean_ms",
    "real_std_ms",
    "synthetic_std_ms",
    "real_lag1_autocorrelation",
    "synthetic_lag1_autocorrelation",
    "lag1_autocorrelation_abs_difference",
    "real_outside_half_clock_rate",
    "synthetic_outside_half_clock_rate",
    "normalization_scale_ms",
    "normalized_wasserstein",
    "normalized_quantile_rmse",
    "normalized_mean_bias",
    "normalized_std_difference",
    "ranking_score",
)


def discover_user_ids(data_root: Path = TEXT_DATA_ROOT) -> list[str]:
    available = {
        path.name.removeprefix("user_").removesuffix("_text_click_data_clean.csv")
        for path in data_root.glob("user_*_text_click_data_clean.csv")
    }
    return [user_id for user_id in DEFAULT_USERS if user_id in available]


def _clean_period_offset_rows(click_df: pd.DataFrame) -> pd.DataFrame:
    result = click_df.copy()
    result[CLOCK_PERIOD_COL] = pd.to_numeric(result[CLOCK_PERIOD_COL], errors="coerce")
    result[CLICK_OFFSET_COL] = pd.to_numeric(result[CLICK_OFFSET_COL], errors="coerce")
    invalid = result[[CLOCK_PERIOD_COL, CLICK_OFFSET_COL]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} click rows without a numeric clock period and offset"
        )
    return result


def build_clock_regimes(
    train_click_df: pd.DataFrame,
    absolute_tolerance_s: float = 0.02,
    relative_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Cluster observed training periods without consulting validation data."""

    periods = pd.to_numeric(train_click_df[CLOCK_PERIOD_COL], errors="coerce").dropna()
    if periods.empty:
        raise ValueError("Cannot build clock regimes without training clock periods")

    period_counts = periods.value_counts().sort_index()
    clusters: list[dict[str, Any]] = []
    for period, count in period_counts.items():
        period = float(period)
        count = int(count)
        if not clusters:
            clusters.append({"values": [(period, count)]})
            continue

        values = clusters[-1]["values"]
        total = sum(item_count for _, item_count in values)
        center = sum(value * item_count for value, item_count in values) / total
        tolerance = max(absolute_tolerance_s, abs(center) * relative_tolerance)
        if abs(period - center) <= tolerance:
            values.append((period, count))
        else:
            clusters.append({"values": [(period, count)]})

    rows = []
    for index, cluster in enumerate(clusters):
        values = cluster["values"]
        total = sum(count for _, count in values)
        center = sum(value * count for value, count in values) / total
        rows.append(
            {
                "regime_id": f"regime_{index + 1}_{center:.3f}s",
                "period_center_s": float(center),
                "period_min_s": float(min(value for value, _ in values)),
                "period_max_s": float(max(value for value, _ in values)),
                "train_period_rows": int(total),
            }
        )
    return pd.DataFrame(rows)


def assign_clock_regimes(
    click_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    absolute_tolerance_s: float = 0.02,
    relative_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Assign rows to their nearest compatible training-derived regime."""

    if regimes_df.empty:
        raise ValueError("At least one clock regime is required")
    result = click_df.copy()
    centers = regimes_df["period_center_s"].to_numpy(dtype=float)
    regime_ids = regimes_df["regime_id"].astype(str).to_numpy()
    assignments = []
    for period in pd.to_numeric(result[CLOCK_PERIOD_COL], errors="coerce"):
        if pd.isna(period):
            raise ValueError("Clock-period rows must be numeric before regime assignment")
        differences = np.abs(centers - float(period))
        nearest = int(np.argmin(differences))
        tolerance = max(
            absolute_tolerance_s,
            abs(float(centers[nearest])) * relative_tolerance,
        )
        if float(differences[nearest]) > tolerance:
            raise ValueError(
                f"Clock period {float(period):.6f}s has no compatible training regime; "
                f"nearest is {float(centers[nearest]):.6f}s"
            )
        assignments.append(regime_ids[nearest])
    result["Clock Regime"] = assignments
    return result


def build_regime_statistics(
    assigned_train_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    minimum_training_clicks: int = 20,
    low_sample_threshold: int = 50,
) -> pd.DataFrame:
    rows = []
    for regime in regimes_df.to_dict("records"):
        regime_rows = assigned_train_df[assigned_train_df["Clock Regime"] == regime["regime_id"]]
        offsets = regime_rows[CLICK_OFFSET_COL].to_numpy(dtype=float)
        count = int(offsets.size)
        rows.append(
            {
                **regime,
                "train_click_rows": count,
                "offset_mean_s": float(np.mean(offsets)) if count else np.nan,
                "offset_std_s": float(np.std(offsets, ddof=0)) if count else np.nan,
                "eligible": bool(count >= minimum_training_clicks),
                "low_sample": bool(count < low_sample_threshold),
            }
        )
    return pd.DataFrame(rows)


def build_validation_schedule(validation_click_df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable chronological schedule for later simulator comparisons."""

    schedule = validation_click_df.copy()
    schedule["Source Row"] = schedule.index
    sort_columns = [
        column
        for column in ("Session Num", "Phrase Num", "Selection Num", "Click Num", "Source Row")
        if column in schedule
    ]
    schedule = schedule.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    schedule.insert(0, "Click Index", np.arange(1, len(schedule) + 1))
    return schedule


def _trial_rng(base_seed: int, user_id: str, multiplier_index: int, trial: int) -> np.random.Generator:
    user_entropy = list(user_id.encode("utf-8"))
    seed = np.random.SeedSequence([base_seed, multiplier_index, trial, *user_entropy])
    return np.random.default_rng(seed)


def generate_offsets(
    schedule_df: pd.DataFrame,
    regime_stats_df: pd.DataFrame,
    std_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    synthetic = np.empty(len(schedule_df), dtype=float)
    stats_by_regime = regime_stats_df.set_index("regime_id")
    for regime_id, indices in schedule_df.groupby("Clock Regime", sort=False).groups.items():
        if regime_id not in stats_by_regime.index:
            raise ValueError(f"Missing training statistics for {regime_id}")
        stats = stats_by_regime.loc[regime_id]
        if not bool(stats["eligible"]):
            raise ValueError(
                f"{regime_id} has only {int(stats['train_click_rows'])} training clicks"
            )
        positions = schedule_df.index.get_indexer(indices)
        synthetic[positions] = rng.normal(
            loc=float(stats["offset_mean_s"]),
            scale=float(stats["offset_std_s"]) * std_multiplier,
            size=len(positions),
        )
    return synthetic


def lag1_autocorrelation(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    if values_array.size < 3 or np.std(values_array[:-1]) == 0 or np.std(values_array[1:]) == 0:
        return np.nan
    return float(np.corrcoef(values_array[:-1], values_array[1:])[0, 1])


def calculate_offset_metrics(
    real_offsets_s: Sequence[float],
    synthetic_offsets_s: Sequence[float],
    clock_periods_s: Sequence[float],
) -> dict[str, float]:
    real_ms = np.asarray(real_offsets_s, dtype=float) * 1000.0
    synthetic_ms = np.asarray(synthetic_offsets_s, dtype=float) * 1000.0
    periods_ms = np.asarray(clock_periods_s, dtype=float) * 1000.0
    if real_ms.size == 0 or real_ms.shape != synthetic_ms.shape or real_ms.shape != periods_ms.shape:
        raise ValueError("Real offsets, synthetic offsets, and clock periods need equal non-zero shapes")

    quantile_errors = np.sort(synthetic_ms) - np.sort(real_ms)
    real_mean = float(np.mean(real_ms))
    synthetic_mean = float(np.mean(synthetic_ms))
    real_std = float(np.std(real_ms, ddof=0))
    synthetic_std = float(np.std(synthetic_ms, ddof=0))
    scale = real_std if real_std > 1e-9 else max(abs(real_mean), 1.0)
    wasserstein_ms = float(wasserstein_distance(real_ms, synthetic_ms))
    quantile_rmse_ms = float(np.sqrt(np.mean(np.square(quantile_errors))))
    mean_bias_ms = synthetic_mean - real_mean
    std_difference_ms = synthetic_std - real_std
    ks_statistic = float(ks_2samp(real_ms, synthetic_ms).statistic)
    centered_ks_statistic = float(
        ks_2samp(real_ms - real_mean, synthetic_ms - synthetic_mean).statistic
    )
    real_lag1 = lag1_autocorrelation(real_ms)
    synthetic_lag1 = lag1_autocorrelation(synthetic_ms)
    lag1_difference = (
        abs(synthetic_lag1 - real_lag1)
        if np.isfinite(real_lag1) and np.isfinite(synthetic_lag1)
        else np.nan
    )
    normalized_std_difference = abs(std_difference_ms) / scale
    ranking_score = 0.70 * normalized_std_difference + 0.30 * centered_ks_statistic
    return {
        "quantile_mae_ms": float(np.mean(np.abs(quantile_errors))),
        "quantile_rmse_ms": quantile_rmse_ms,
        "ks_statistic": ks_statistic,
        "centered_ks_statistic": centered_ks_statistic,
        "wasserstein_ms": wasserstein_ms,
        "mean_bias_ms": mean_bias_ms,
        "std_difference_ms": std_difference_ms,
        "real_mean_ms": real_mean,
        "synthetic_mean_ms": synthetic_mean,
        "real_std_ms": real_std,
        "synthetic_std_ms": synthetic_std,
        "real_lag1_autocorrelation": real_lag1,
        "synthetic_lag1_autocorrelation": synthetic_lag1,
        "lag1_autocorrelation_abs_difference": lag1_difference,
        "real_outside_half_clock_rate": float(np.mean(np.abs(real_ms) > periods_ms / 2.0)),
        "synthetic_outside_half_clock_rate": float(
            np.mean(np.abs(synthetic_ms) > periods_ms / 2.0)
        ),
        "normalization_scale_ms": scale,
        "normalized_wasserstein": wasserstein_ms / scale,
        "normalized_quantile_rmse": quantile_rmse_ms / scale,
        "normalized_mean_bias": abs(mean_bias_ms) / scale,
        "normalized_std_difference": normalized_std_difference,
        "ranking_score": ranking_score,
    }


def evaluate_trial_scopes(
    user_id: str,
    trial: int,
    std_multiplier: float,
    schedule_df: pd.DataFrame,
    synthetic_offsets_s: np.ndarray,
    regime_stats_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(schedule_df), dtype=bool))
    ]
    for regime_id in schedule_df["Clock Regime"].drop_duplicates():
        scopes.append(
            (
                "regime",
                str(regime_id),
                schedule_df["Clock Regime"].eq(regime_id).to_numpy(),
            )
        )

    stats_by_regime = regime_stats_df.set_index("regime_id")
    rows = []
    for scope, regime_id, mask in scopes:
        if scope == "overall":
            train_rows = int(
                regime_stats_df[
                    regime_stats_df["regime_id"].isin(schedule_df["Clock Regime"].unique())
                ]["train_click_rows"].sum()
            )
            low_sample = bool(
                regime_stats_df[
                    regime_stats_df["regime_id"].isin(schedule_df["Clock Regime"].unique())
                ]["low_sample"].any()
            )
        else:
            stats = stats_by_regime.loc[regime_id]
            train_rows = int(stats["train_click_rows"])
            low_sample = bool(stats["low_sample"])
        metrics = calculate_offset_metrics(
            schedule_df.loc[mask, CLICK_OFFSET_COL],
            synthetic_offsets_s[mask],
            schedule_df.loc[mask, CLOCK_PERIOD_COL],
        )
        rows.append(
            {
                "user_id": user_id,
                "trial": trial,
                "std_multiplier": std_multiplier,
                "scope": scope,
                "regime_id": regime_id,
                "train_click_rows": train_rows,
                "validation_click_rows": int(mask.sum()),
                "validation_weight": float(mask.mean()),
                "low_sample": low_sample,
                **metrics,
            }
        )
    return rows


def aggregate_trial_metrics(trial_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["user_id", "std_multiplier", "scope", "regime_id"]
    for keys, group in trial_metrics_df.groupby(grouping, sort=True, dropna=False):
        row = dict(zip(grouping, keys))
        row.update(
            {
                "trials": int(group["trial"].nunique()),
                "train_click_rows": int(group["train_click_rows"].iloc[0]),
                "validation_click_rows": int(group["validation_click_rows"].iloc[0]),
                "validation_weight": float(group["validation_weight"].iloc[0]),
                "low_sample": bool(group["low_sample"].iloc[0]),
            }
        )
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if not values.empty else np.nan
            row[f"{metric}_p025"] = float(values.quantile(0.025)) if not values.empty else np.nan
            row[f"{metric}_p975"] = float(values.quantile(0.975)) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_multipliers(summary_df: pd.DataFrame) -> pd.DataFrame:
    overall = summary_df[summary_df["scope"] == "overall"].copy()
    selected = []
    for user_id, user_rows in overall.groupby("user_id", sort=True):
        user_rows = user_rows.assign(
            distance_from_one=(user_rows["std_multiplier"] - 1.0).abs()
        ).sort_values(
            ["ranking_score_mean", "distance_from_one", "std_multiplier"],
            kind="stable",
        )
        winner = user_rows.iloc[0]
        selected.append(
            {
                "user_id": user_id,
                "selected_std_multiplier": float(winner["std_multiplier"]),
                "ranking_score": float(winner["ranking_score_mean"]),
                "train_click_rows": int(winner["train_click_rows"]),
                "validation_click_rows": int(winner["validation_click_rows"]),
                "low_sample": bool(winner["low_sample"]),
                "confidence": "low_sample" if bool(winner["low_sample"]) else "standard",
            }
        )
    return pd.DataFrame(selected)


def build_score_summary(summary_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact table needed to choose and discuss SD multipliers."""

    selected_lookup = selected_df.set_index("user_id")["selected_std_multiplier"].to_dict()
    overall = summary_df[summary_df["scope"] == "overall"].copy()
    rows = []
    for row in overall.sort_values(["user_id", "std_multiplier"]).itertuples():
        selected_multiplier = selected_lookup[row.user_id]
        rows.append(
            {
                "user_id": row.user_id,
                "std_multiplier": float(row.std_multiplier),
                "selected": bool(np.isclose(float(row.std_multiplier), selected_multiplier)),
                "normalized_std_error": float(row.normalized_std_difference_mean),
                "centered_ks": float(row.centered_ks_statistic_mean),
                "score": float(row.ranking_score_mean),
                "train_click_rows": int(row.train_click_rows),
                "validation_click_rows": int(row.validation_click_rows),
                "low_sample": bool(row.low_sample),
            }
        )
    return pd.DataFrame(rows)


def _representative_trial(
    trial_metrics_df: pd.DataFrame,
    user_id: str,
    std_multiplier: float,
) -> int:
    rows = trial_metrics_df[
        (trial_metrics_df["user_id"] == user_id)
        & (trial_metrics_df["scope"] == "overall")
        & np.isclose(trial_metrics_df["std_multiplier"], std_multiplier)
    ].copy()
    median = float(rows["ranking_score"].median())
    rows["median_distance"] = (rows["ranking_score"] - median).abs()
    return int(rows.sort_values(["median_distance", "trial"]).iloc[0]["trial"])


def plot_user_results(
    user_id: str,
    schedule_df: pd.DataFrame,
    regime_stats_df: pd.DataFrame,
    trial_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_multiplier: float,
    std_multipliers: Sequence[float],
    base_seed: int,
    output_path: Path,
) -> None:
    colors = {
        multiplier: color
        for multiplier, color in zip(std_multipliers, ("#2563eb", "#16a34a", "#ea580c"))
    }
    synthetic_by_multiplier: dict[float, np.ndarray] = {}
    for multiplier_index, multiplier in enumerate(std_multipliers):
        trial = _representative_trial(trial_metrics_df, user_id, multiplier)
        synthetic_by_multiplier[multiplier] = generate_offsets(
            schedule_df,
            regime_stats_df,
            multiplier,
            _trial_rng(base_seed, user_id, multiplier_index, trial),
        )

    real_ms = schedule_df[CLICK_OFFSET_COL].to_numpy(dtype=float) * 1000.0
    figure = plt.figure(figsize=(11, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(3.5, 1.15))
    ecdf_axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])

    for values, label, color in [(real_ms, "Held-out real", "#171717")]:
        sorted_values = np.sort(values)
        ecdf_axis.plot(
            sorted_values,
            np.arange(1, len(values) + 1) / len(values),
            label=label,
            color=color,
            linewidth=2.0,
        )
    for multiplier in std_multipliers:
        values = np.sort(synthetic_by_multiplier[multiplier] * 1000.0)
        ecdf_axis.plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            label=f"Synthetic {multiplier:g}x SD",
            color=colors[multiplier],
            linewidth=1.6,
        )
    ecdf_axis.set_xlabel("Click offset (ms)")
    ecdf_axis.set_ylabel("ECDF")
    ecdf_axis.grid(alpha=0.18)
    ecdf_axis.legend(fontsize=9, loc="lower right")

    user_summary = summary_df[
        (summary_df["user_id"] == user_id) & (summary_df["scope"] == "overall")
    ].set_index("std_multiplier")
    trial_count = int(
        trial_metrics_df[trial_metrics_df["user_id"] == user_id]["trial"].nunique()
    )
    table_rows = []
    for multiplier in std_multipliers:
        row = user_summary.loc[multiplier]
        label = f"{multiplier:g}x" + (" *" if np.isclose(multiplier, selected_multiplier) else "")
        table_rows.append(
            [
                label,
                f"{row['normalized_std_difference_mean']:.3f}",
                f"{row['centered_ks_statistic_mean']:.3f}",
                f"{row['ranking_score_mean']:.3f}",
            ]
        )
    table_axis.axis("off")
    table_axis.set_title(f"{trial_count}-trial selection scores", pad=8, fontsize=11)
    table = table_axis.table(
        cellText=table_rows,
        colLabels=["SD multiplier", "Norm std error", "Centered KS", "Score"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    figure.suptitle(
        f"User {user_id} click-offset SD sweep | selected {selected_multiplier:g}x",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"click_offset_std_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _parse_csv_values(value: str, converter: Any) -> list[Any]:
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS), help="Comma-separated user ids.")
    parser.add_argument("--std-multipliers", default="0.5,1.0,1.5")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--absolute-regime-tolerance-s", type=float, default=0.02)
    parser.add_argument("--relative-regime-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-regime-training-clicks", type=int, default=20)
    parser.add_argument("--low-sample-threshold", type=int, default=50)
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Also write full trial metrics, detailed summary, and validation schedules.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> Path:
    user_ids = _parse_csv_values(args.users, str)
    std_multipliers = _parse_csv_values(args.std_multipliers, float)
    if not user_ids or not std_multipliers:
        raise ValueError("At least one user and standard-deviation multiplier are required")
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if any(multiplier < 0 for multiplier in std_multipliers):
        raise ValueError("Standard-deviation multipliers cannot be negative")

    output_dir = build_output_dir(args.output_dir)
    all_trial_rows: list[dict[str, Any]] = []
    all_schedules = []
    user_context: dict[str, dict[str, Any]] = {}
    split_config: dict[str, Any] = {}

    for user_id in user_ids:
        real_click_df = _clean_period_offset_rows(load_text_click_data(user_id))
        train_sessions, validation_sessions = split_sessions(
            real_click_df, args.validation_fraction
        )
        train_df = real_click_df[real_click_df["Session Num"].isin(train_sessions)].copy()
        validation_df = real_click_df[
            real_click_df["Session Num"].isin(validation_sessions)
        ].copy()
        regimes_df = build_clock_regimes(
            train_df,
            args.absolute_regime_tolerance_s,
            args.relative_regime_tolerance,
        )
        assigned_train_df = assign_clock_regimes(
            train_df,
            regimes_df,
            args.absolute_regime_tolerance_s,
            args.relative_regime_tolerance,
        )
        regime_stats_df = build_regime_statistics(
            assigned_train_df,
            regimes_df,
            args.minimum_regime_training_clicks,
            args.low_sample_threshold,
        )
        schedule_df = assign_clock_regimes(
            build_validation_schedule(validation_df),
            regimes_df,
            args.absolute_regime_tolerance_s,
            args.relative_regime_tolerance,
        )
        evaluated_regimes = set(schedule_df["Clock Regime"])
        ineligible = regime_stats_df[
            regime_stats_df["regime_id"].isin(evaluated_regimes)
            & ~regime_stats_df["eligible"]
        ]
        if not ineligible.empty:
            details = ", ".join(
                f"{row.regime_id} ({int(row.train_click_rows)} rows)"
                for row in ineligible.itertuples()
            )
            raise ValueError(f"User {user_id} has insufficient training data: {details}")

        schedule_output = schedule_df[
            [
                "Click Index",
                "Source Row",
                "Session Num",
                *[
                    column
                    for column in ("Phrase Num", "Selection Num", "Click Num")
                    if column in schedule_df
                ],
                CLOCK_PERIOD_COL,
                "Clock Regime",
                CLICK_OFFSET_COL,
            ]
        ].copy()
        schedule_output.insert(0, "User ID", user_id)
        all_schedules.append(schedule_output)

        for multiplier_index, multiplier in enumerate(std_multipliers):
            for trial in range(args.trials):
                synthetic = generate_offsets(
                    schedule_df,
                    regime_stats_df,
                    multiplier,
                    _trial_rng(args.seed, user_id, multiplier_index, trial),
                )
                all_trial_rows.extend(
                    evaluate_trial_scopes(
                        user_id,
                        trial,
                        multiplier,
                        schedule_df,
                        synthetic,
                        regime_stats_df,
                    )
                )

        user_context[user_id] = {
            "schedule": schedule_df,
            "regime_stats": regime_stats_df,
        }
        split_config[user_id] = {
            "train_sessions": train_sessions,
            "validation_sessions": validation_sessions,
            "regimes": regime_stats_df.to_dict("records"),
        }

    trial_metrics_df = pd.DataFrame(all_trial_rows)
    summary_df = aggregate_trial_metrics(trial_metrics_df)
    selected_df = select_best_multipliers(summary_df)
    score_summary_df = build_score_summary(summary_df, selected_df)
    schedules_df = pd.concat(all_schedules, ignore_index=True)

    score_summary_df.to_csv(output_dir / "std_sweep_scores.csv", index=False)
    selected_df.to_csv(output_dir / "selected_std_multipliers.csv", index=False)
    if args.write_diagnostics:
        trial_metrics_df.to_csv(output_dir / "trial_metrics.csv", index=False)
        summary_df.to_csv(output_dir / "sweep_summary_diagnostics.csv", index=False)
        schedules_df.to_csv(output_dir / "validation_clock_schedules.csv", index=False)

    selected_by_user = selected_df.set_index("user_id")
    for user_id in user_ids:
        plot_user_results(
            user_id=user_id,
            schedule_df=user_context[user_id]["schedule"],
            regime_stats_df=user_context[user_id]["regime_stats"],
            trial_metrics_df=trial_metrics_df,
            summary_df=summary_df,
            selected_multiplier=float(selected_by_user.loc[user_id, "selected_std_multiplier"]),
            std_multipliers=std_multipliers,
            base_seed=args.seed,
            output_path=output_dir / "plots" / f"user_{user_id}_click_offset_std_sweep.png",
        )

    write_json(
        output_dir / "run_config.json",
        {
            "users": user_ids,
            "std_multipliers": std_multipliers,
            "trials": args.trials,
            "write_diagnostics": bool(args.write_diagnostics),
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "regime_policy": {
                "source": "training sessions only",
                "absolute_tolerance_s": args.absolute_regime_tolerance_s,
                "relative_tolerance": args.relative_regime_tolerance,
                "minimum_training_clicks": args.minimum_regime_training_clicks,
                "low_sample_threshold": args.low_sample_threshold,
            },
            "sampling": "unbounded normal by training-derived clock regime",
            "ranking_components": [
                {"metric": "normalized_std_difference", "weight": 0.70},
                {"metric": "centered_ks_statistic", "weight": 0.30},
            ],
            "selection_scope": "one multiplier per user from overall held-out score",
            "default_outputs": [
                "std_sweep_scores.csv",
                "selected_std_multipliers.csv",
                "run_config.json",
                "plots/user_<id>_click_offset_std_sweep.png",
            ],
            "diagnostic_outputs": [
                "trial_metrics.csv",
                "sweep_summary_diagnostics.csv",
                "validation_clock_schedules.csv",
            ],
            "clock_schedule_contract": (
                "validation_clock_schedules.csv preserves chronological held-out periods "
                "when --write-diagnostics is enabled"
            ),
            "splits": split_config,
            "selected_multipliers": selected_df.to_dict("records"),
        },
    )

    print(f"Saved click-offset SD sweep to: {output_dir}")
    print(selected_df.to_string(index=False))
    return output_dir


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
