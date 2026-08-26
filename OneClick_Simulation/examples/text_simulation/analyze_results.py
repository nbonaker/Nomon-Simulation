"""Standardized analysis for phrase-level OneClick simulation results.

This module intentionally sits downstream of the simulator:

    run_full_study.py -> user_*.csv -> analyze_results.py -> metric tables

Metrics are reconstructed from raw phrase counters. Prediction usage is based
on the actual successful winning clock index, classified as prefix, BEST, or
argmax-literal by the simulator.

Example:

    python3 -m OneClick_Simulation.examples.text_simulation.analyze_results \
        OneClick_Simulation/examples/text_simulation/results/sim-MM_DD_YYYY-HH_MM

The command writes ``metrics_per_user.csv``, ``metrics_aggregate.csv``, and
``failure_reasons.csv`` to an ``analysis`` subdirectory by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = {
    "Num Clicks",
    "Num Selections",
    "Successful Word Click Count",
    "Successful Word Selection Count",
    "Successful Word Character Count",
    "Target Word Count",
    "Completed Word Count",
    "Failed Word Count",
    "Corrective Undo Action Count",
    "Enter Press Count",
    "Enter Misselection Count",
    "Prediction Selection Count",
    "Prefix Prediction Selection Count",
    "Best Prediction Selection Count",
    "Argmax Prediction Selection Count",
    "Prediction Selection Events",
    "Active Typing Time (s)",
    "Phrase Completed",
    "Phrase Failure Reason",
    "Failure Reason Counts",
    "Failure Events",
}

RATE_COLUMNS = [
    "Clicks per Character",
    "Active Typing Time (s/phrase)",
    "Correction Rate (undoes/successful word)",
    "Enter Misselection Rate",
    "Prefix Prediction Usage (%)",
    "Best Prediction Usage (%)",
    "Argmax Prediction Usage (%)",
    "Completion Rate",
]

RAW_COUNT_COLUMNS = [
    "Phrases Recorded",
    "Completed Phrase Count",
    "Attempted Word Count",
    "Completed Word Count",
    "Failed Word Count",
    "Failure Phrase Count",
    "Num Clicks",
    "Num Selections",
    "Successful Word Click Count",
    "Successful Word Selection Count",
    "Successful Word Character Count",
    "Corrective Undo Action Count",
    "Enter Press Count",
    "Enter Misselection Count",
    "Prediction Selection Count",
    "Prefix Prediction Selection Count",
    "Best Prediction Selection Count",
    "Argmax Prediction Selection Count",
]


class MissingRawFieldsError(ValueError):
    """Raised when input CSVs predate the raw phrase-level output schema."""


def _numeric_sum(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"Raw count column {column!r} contains missing values")
    return float(values.sum())


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse booleans without treating the string ``'False'`` as truthy."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = set(normalized.unique()) - allowed
    if unexpected:
        raise ValueError(f"Unrecognized boolean values: {sorted(unexpected)}")
    return normalized.isin({"true", "1"})


def _safe_percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator * 100.0


def _infer_user_id(path: Path) -> str:
    stem = path.stem
    return stem[len("user_") :] if stem.startswith("user_") else stem


def load_results(
    results_dir: Path | str,
    configuration: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Load and concatenate the phrase-level ``user_*.csv`` files.

    ``summary.csv`` is deliberately ignored because it contains means of
    already-derived phrase metrics rather than the raw counters needed here.
    Arbitrary configuration metadata can be attached now and used as grouping
    columns in a future multi-configuration analysis.
    """
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("user_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No user_*.csv files found in {results_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "user_id" not in frame.columns:
            frame["user_id"] = _infer_user_id(path)
        frame["Source File"] = str(path)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True, sort=False)
    missing = sorted(REQUIRED_RAW_COLUMNS - set(results.columns))
    if missing:
        raise MissingRawFieldsError(
            "Phrase CSVs do not contain the current raw schema; missing: "
            + ", ".join(missing)
        )

    results["Configuration"] = configuration or results_dir.name
    for key, value in (metadata or {}).items():
        if key in results.columns and not (results[key] == value).all():
            raise ValueError(f"Metadata {key!r} conflicts with an input CSV column")
        results[key] = value
    return results


def compute_clicks_per_character(phrases: pd.DataFrame) -> float:
    """Return successful-word clicks / characters in successful words."""
    clicks = _numeric_sum(phrases, "Successful Word Click Count")
    characters = _numeric_sum(phrases, "Successful Word Character Count")
    return clicks / characters if characters > 0 else np.nan


def compute_active_typing_time(phrases: pd.DataFrame) -> float:
    """Return mean active simulated seconds per recorded phrase."""
    values = pd.to_numeric(phrases["Active Typing Time (s)"], errors="raise")
    if values.isna().any():
        raise ValueError("Active Typing Time (s) contains missing values")
    return float(values.mean()) if len(values) else np.nan


def compute_correction_rate(phrases: pd.DataFrame) -> float:
    """Return corrective Undo actions / successfully completed words."""
    corrections = _numeric_sum(phrases, "Corrective Undo Action Count")
    completed = _numeric_sum(phrases, "Completed Word Count")
    return corrections / completed if completed > 0 else np.nan


def compute_enter_misselection_rate(phrases: pd.DataFrame) -> float:
    """Return Enter presses selecting a non-target clock / all Enter presses."""
    misselections = _numeric_sum(phrases, "Enter Misselection Count")
    presses = _numeric_sum(phrases, "Enter Press Count")
    return misselections / presses if presses > 0 else np.nan


def compute_prediction_usage(phrases: pd.DataFrame) -> Dict[str, float]:
    """Return each source's share of successful prediction selections."""
    counts = {
        "prefix": _numeric_sum(phrases, "Prefix Prediction Selection Count"),
        "best": _numeric_sum(phrases, "Best Prediction Selection Count"),
        "argmax": _numeric_sum(phrases, "Argmax Prediction Selection Count"),
    }
    total = sum(counts.values())
    if total == 0:
        return {source: np.nan for source in counts}
    return {source: count / total * 100.0 for source, count in counts.items()}


def compute_completion_rate(phrases: pd.DataFrame) -> float:
    """Return completed words / attempted words, including failures."""
    completed = _numeric_sum(phrases, "Completed Word Count")
    attempted = _numeric_sum(phrases, "Target Word Count")
    return completed / attempted if attempted > 0 else np.nan


def summarize_user(phrases: pd.DataFrame, user_id: object) -> Dict[str, object]:
    """Create one standardized metric row for one user/configuration."""
    predictions = compute_prediction_usage(phrases)
    failures = phrases["Phrase Failure Reason"].fillna("").astype(str).str.strip()
    phrase_completed = _as_bool(phrases["Phrase Completed"])
    attempted_words = _numeric_sum(phrases, "Target Word Count")
    completed_words = _numeric_sum(phrases, "Completed Word Count")
    failed_words = _numeric_sum(phrases, "Failed Word Count")

    row: Dict[str, object] = {
        "Configuration": phrases["Configuration"].iloc[0],
        "User": str(user_id),
        "Phrases Recorded": int(len(phrases)),
        "Completed Phrase Count": int(phrase_completed.sum()),
        "Attempted Word Count": int(attempted_words),
        "Completed Word Count": int(completed_words),
        "Failed Word Count": int(failed_words),
        "Failure Phrase Count": int((failures != "").sum()),
        "Num Clicks": int(_numeric_sum(phrases, "Num Clicks")),
        "Num Selections": int(_numeric_sum(phrases, "Num Selections")),
        "Successful Word Click Count": int(
            _numeric_sum(phrases, "Successful Word Click Count")
        ),
        "Successful Word Selection Count": int(
            _numeric_sum(phrases, "Successful Word Selection Count")
        ),
        "Successful Word Character Count": int(
            _numeric_sum(phrases, "Successful Word Character Count")
        ),
        "Corrective Undo Action Count": int(
            _numeric_sum(phrases, "Corrective Undo Action Count")
        ),
        "Enter Press Count": int(_numeric_sum(phrases, "Enter Press Count")),
        "Enter Misselection Count": int(
            _numeric_sum(phrases, "Enter Misselection Count")
        ),
        "Prediction Selection Count": int(
            _numeric_sum(phrases, "Prediction Selection Count")
        ),
        "Prefix Prediction Selection Count": int(
            _numeric_sum(phrases, "Prefix Prediction Selection Count")
        ),
        "Best Prediction Selection Count": int(
            _numeric_sum(phrases, "Best Prediction Selection Count")
        ),
        "Argmax Prediction Selection Count": int(
            _numeric_sum(phrases, "Argmax Prediction Selection Count")
        ),
        "Clicks per Character": compute_clicks_per_character(phrases),
        "Active Typing Time (s/phrase)": compute_active_typing_time(phrases),
        "Correction Rate (undoes/successful word)": compute_correction_rate(phrases),
        "Enter Misselection Rate": compute_enter_misselection_rate(phrases),
        "Prefix Prediction Usage (%)": predictions["prefix"],
        "Best Prediction Usage (%)": predictions["best"],
        "Argmax Prediction Usage (%)": predictions["argmax"],
        "Completion Rate": compute_completion_rate(phrases),
    }

    # Preserve caller-supplied configuration metadata when it is constant for
    # this result set.  Source/user/raw simulation columns are intentionally not
    # copied into the standardized schema.
    excluded = REQUIRED_RAW_COLUMNS | {"user_id", "Source File", "Configuration"}
    for column in phrases.columns:
        if column in excluded or column in row:
            continue
        values = phrases[column].dropna().unique()
        if len(values) == 1 and column.startswith("config_"):
            row[column] = values[0]
    return row


def summarize_users(results: pd.DataFrame) -> pd.DataFrame:
    """Return one row per configuration and user."""
    rows = []
    for (configuration, user_id), phrases in results.groupby(
        ["Configuration", "user_id"], sort=True, dropna=False
    ):
        del configuration
        rows.append(summarize_user(phrases, user_id))
    return pd.DataFrame(rows)


def summarize_all_users(
    per_user: pd.DataFrame,
    exclude_users: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Macro-average user metrics and sum raw counts by configuration.

    Rates are averaged over user rows, never recomputed from pooled phrase
    counts.  This gives each user equal weight.  Raw counts remain available as
    sums for auditing.
    """
    excluded = {str(user) for user in (exclude_users or [])}
    included = per_user[~per_user["User"].astype(str).isin(excluded)].copy()
    rows = []
    for configuration, users in included.groupby("Configuration", sort=True):
        row: Dict[str, object] = {
            "Configuration": configuration,
            "User": "AGGREGATE",
            "Users Included": int(len(users)),
        }
        for column in RAW_COUNT_COLUMNS:
            row[column] = int(pd.to_numeric(users[column], errors="raise").sum())
        for column in RATE_COLUMNS:
            # pandas mean skips unresolved NaN TODO values. If every user is NaN,
            # the aggregate remains NaN rather than fabricating a zero.
            row[column] = pd.to_numeric(users[column], errors="raise").mean()
        for column in users.columns:
            if column.startswith("config_") and users[column].nunique(dropna=True) == 1:
                row[column] = users[column].dropna().iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_failure_reasons(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every structured terminal failure reason."""
    rows = []
    for _, phrase in results.iterrows():
        raw_events = phrase["Failure Events"]
        try:
            events = json.loads(raw_events) if pd.notna(raw_events) else []
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid Failure Events JSON: {raw_events!r}") from exc
        if not isinstance(events, list):
            raise ValueError("Failure Events must be a JSON array")
        for event in events:
            if not isinstance(event, dict) or "reason" not in event:
                raise ValueError("Each Failure Events item must contain a reason")
            rows.append(
                {
                    "Configuration": phrase["Configuration"],
                    "User": phrase["user_id"],
                    "Failure Reason": event["reason"],
                    "Failure Stage": event.get("stage", "unknown"),
                    "Failure Limit": event.get("limit", ""),
                    "Failure Guard": event.get("guard", ""),
                    "Failure Count": 1,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Configuration",
                "User",
                "Failure Reason",
                "Failure Stage",
                "Failure Limit",
                "Failure Guard",
                "Failure Count",
            ]
        )
    return (
        pd.DataFrame(rows)
        .groupby(
            [
                "Configuration",
                "User",
                "Failure Reason",
                "Failure Stage",
                "Failure Limit",
                "Failure Guard",
            ],
            as_index=False,
        )["Failure Count"]
        .sum()
    )


def _parse_metadata(values: Sequence[str]) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Metadata must have KEY=VALUE form: {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Metadata key cannot be empty")
        if not key.startswith("config_"):
            key = "config_" + key
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def analyze_results(
    results_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    configuration: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
    exclude_users: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the analysis and write the three output tables."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "analysis"
    raw = load_results(results_dir, configuration=configuration, metadata=metadata)
    per_user = summarize_users(raw)
    aggregate = summarize_all_users(per_user, exclude_users=exclude_users)
    failures = summarize_failure_reasons(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(output_dir / "metrics_per_user.csv", index=False)
    aggregate.to_csv(output_dir / "metrics_aggregate.csv", index=False)
    failures.to_csv(output_dir / "failure_reasons.csv", index=False)
    return per_user, aggregate, failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze phrase-level CSVs from the OneClick full study."
    )
    parser.add_argument("results_dir", type=Path, help="Directory containing user_*.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--configuration",
        help="Configuration label (defaults to the result directory name)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach configuration metadata; repeat for multiple fields",
    )
    parser.add_argument(
        "--exclude-user",
        action="append",
        default=[],
        help="Exclude a user from the aggregate only (for example, synthetic user P)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    metadata = _parse_metadata(args.metadata)
    per_user, aggregate, failures = analyze_results(
        args.results_dir,
        output_dir=args.output_dir,
        configuration=args.configuration,
        metadata=metadata,
        exclude_users=args.exclude_user,
    )
    output_dir = args.output_dir or args.results_dir / "analysis"
    print(per_user.to_string(index=False))
    print("\nAggregate (equal weight per user):")
    print(aggregate.to_string(index=False))
    print(f"\nFailure rows: {len(failures)}")
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
