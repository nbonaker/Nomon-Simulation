"""Create an OG Nomon versus OneClick presentation with exact failure causes."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OG_COLOR = "#4C78A8"
ONECLICK_COLOR = "#54A24B"
FAILURE_REASONS = OrderedDict(
    [
        ("completed", ("Completed", ONECLICK_COLOR)),
        ("undo_click_budget_exhausted", ("Undo selection budget exhausted", "#E45756")),
        ("target_not_displayed", ("Target not displayed", "#F2A541")),
        ("target_enter_retries_exhausted", ("Target Enter retries exhausted", "#EDC948")),
        ("word_click_budget_letters", ("Letter click budget exhausted", "#B279A2")),
        ("word_click_budget_target_enter", ("Target Enter click budget exhausted", "#9C755F")),
        ("word_click_budget_between_attempts", ("Between-attempt click budget exhausted", "#FF9DA7")),
        ("click_stream_exhausted_between_words", ("Input exhausted between words", "#79706E")),
        ("click_stream_exhausted_letters", ("Input exhausted during letters", "#BAB0AC")),
        ("click_stream_exhausted_target_enter", ("Input exhausted during target Enter", "#86BCB6")),
        ("click_stream_exhausted_undo", ("Input exhausted during Undo", "#5F9ED1")),
        ("word_attempts_exhausted", ("Word attempts exhausted", "#8C8C8C")),
        ("final_text_mismatch", ("Final text mismatch", "#333333")),
        ("unclassified_failure", ("Unclassified failure", "#D0D0D0")),
    ]
)


def find_latest_telemetry_run(outputs_dir: Path = OUTPUTS_DIR) -> Path:
    for run_dir in sorted(outputs_dir.glob("nomon_oneclick_*comparison_*"), reverse=True):
        summary_path = run_dir / "comparison_summary.csv"
        system_path = run_dir / "system_phrase_results.csv"
        if not (summary_path.is_file() and system_path.is_file()):
            continue
        columns = pd.read_csv(system_path, nrows=0).columns
        if "phrase_failure_reason" in columns:
            return run_dir
    raise FileNotFoundError("No OneClick comparison with exact failure telemetry found")


def load_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(run_dir / "comparison_summary.csv").sort_values("user_id")
    systems = pd.read_csv(run_dir / "system_phrase_results.csv")
    required_summary = {
        "user_id",
        "og_phrase_completion_rate",
        "oneclick_phrase_completion_rate",
        "mutually_completed_phrase_trials",
        "mutually_completed_og_mean_clicks",
        "mutually_completed_oneclick_mean_clicks",
    }
    required_system = {
        "user_id",
        "system",
        "phrase_completed",
        "phrase_failure_reason",
        "phrase_failure_stage",
        "phrase_failure_limit",
        "phrase_failure_guard",
        "failed_target_word",
        "failed_word_position",
        "failed_word_attempt",
        "failure_word_click_count",
        "failure_letter_press_count",
        "failure_target_enter_attempt_count",
        "failure_undo_attempt_count",
    }
    missing_summary = sorted(required_summary - set(summary.columns))
    missing_system = sorted(required_system - set(systems.columns))
    if missing_summary:
        raise ValueError(f"Comparison summary lacks columns: {missing_summary}")
    if missing_system:
        raise ValueError(f"System results lack failure telemetry: {missing_system}")
    oneclick = systems[systems["system"] == "oneclick"].copy()
    if oneclick.empty:
        raise ValueError("System results contain no protected OneClick rows")
    return summary, oneclick


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def classified_failures(oneclick: pd.DataFrame) -> pd.DataFrame:
    frame = oneclick.copy()
    completed = frame["phrase_completed"].fillna(False).astype(bool)
    reasons = frame["phrase_failure_reason"].fillna("").astype(str).str.strip()
    frame["outcome_reason"] = np.where(
        completed,
        "completed",
        np.where(reasons.isin(FAILURE_REASONS), reasons, "unclassified_failure"),
    )
    return frame


def outcome_counts(
    frame: pd.DataFrame,
    users: list[str],
    include_completed: bool,
) -> pd.DataFrame:
    data = frame if include_completed else frame[frame["outcome_reason"] != "completed"]
    categories = list(FAILURE_REASONS)
    if not include_completed:
        categories.remove("completed")
    return pd.crosstab(data["user_id"], data["outcome_reason"]).reindex(
        index=users,
        columns=categories,
        fill_value=0,
    )


def plot_completion(axis, summary: pd.DataFrame) -> None:
    users = summary["user_id"].astype(str).tolist()
    x = np.arange(len(users))
    width = 0.36
    for offset, column, label, color in [
        (-width / 2, "og_phrase_completion_rate", "OG Nomon", OG_COLOR),
        (width / 2, "oneclick_phrase_completion_rate", "OneClick", ONECLICK_COLOR),
    ]:
        values = summary[column].to_numpy(float)
        bars = axis.bar(x + offset, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x)
    axis.set_xticklabels(users)
    axis.set_ylim(0, 1.18)
    axis.set_ylabel("Completed phrase trials")
    axis.set_title("Phrase completion rate")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, ncol=2, loc="upper center")


def plot_paired_clicks(axis, summary: pd.DataFrame) -> None:
    users = summary["user_id"].astype(str).tolist()
    x = np.arange(len(users))
    width = 0.36
    counts = summary["mutually_completed_phrase_trials"].astype(int).to_numpy()
    all_values = []
    for offset, column, label, color in [
        (-width / 2, "mutually_completed_og_mean_clicks", "OG Nomon", OG_COLOR),
        (width / 2, "mutually_completed_oneclick_mean_clicks", "OneClick", ONECLICK_COLOR),
    ]:
        values = summary[column].to_numpy(float)
        all_values.append(values)
        bars = axis.bar(x + offset, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    combined = np.concatenate(all_values) if all_values else np.array([])
    finite_values = combined[np.isfinite(combined)]
    ymax = float(finite_values.max()) if finite_values.size else 1.0
    for user_x, count in zip(x, counts):
        axis.text(user_x, ymax * 1.13, f"n={count}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(users)
    axis.set_ylim(0, ymax * 1.25)
    axis.set_ylabel("Mean clicks per completed phrase")
    axis.set_title("Clicks on mutually completed phrases")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, ncol=2, loc="upper center")


def plot_stacked_outcomes(
    axis,
    counts: pd.DataFrame,
    title: str,
    label_threshold: float,
) -> tuple[list, list[str]]:
    totals = counts.sum(axis=1)
    proportions = counts.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    users = counts.index.astype(str).tolist()
    left = np.zeros(len(users))
    for reason in counts.columns:
        label, color = FAILURE_REASONS[reason]
        values = proportions[reason].to_numpy(float)
        bars = axis.barh(users, values, left=left, color=color, label=label)
        for row_index, (start, value, count) in enumerate(
            zip(left, values, counts[reason].to_numpy(int))
        ):
            if value >= label_threshold:
                axis.text(
                    start + value / 2,
                    row_index,
                    f"{count}\n{value * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if reason in {"undo_click_budget_exhausted", "final_text_mismatch"} else "#222222",
                    fontweight="bold",
                )
        left += values
    axis.set_yticks(np.arange(len(users)))
    axis.set_yticklabels([f"{user} (n={int(total)})" for user, total in zip(users, totals)])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of phrase trials")
    axis.set_title(title)
    axis.invert_yaxis()
    return axis.get_legend_handles_labels()


def failure_detail_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "user_id",
        "trial",
        "Comparison Phrase ID",
        "Target Phrase",
        "Typed Text",
        "phrase_failure_reason",
        "phrase_failure_stage",
        "phrase_failure_limit",
        "phrase_failure_guard",
        "failed_target_word",
        "failed_word_position",
        "failed_word_attempt",
        "failure_word_click_count",
        "failure_letter_press_count",
        "failure_target_enter_attempt_count",
        "failure_undo_attempt_count",
    ]
    return frame[~frame["phrase_completed"].fillna(False).astype(bool)][columns].copy()


def create_oneclick_failure_comparison(run_dir: Path) -> list[Path]:
    configure_style()
    summary, oneclick = load_data(run_dir)
    classified = classified_failures(oneclick)
    users = summary["user_id"].astype(str).tolist()
    overall_counts = outcome_counts(classified, users, include_completed=True)
    failure_counts = outcome_counts(classified, users, include_completed=False)
    observed_reasons = [
        reason
        for reason in FAILURE_REASONS
        if overall_counts.get(reason, pd.Series(dtype=int)).sum() > 0
    ]
    overall_counts = overall_counts[observed_reasons]
    failure_counts = failure_counts[[reason for reason in observed_reasons if reason != "completed"]]

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(15, 11), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        top=0.965,
        bottom=0.17,
        hspace=0.42,
        wspace=0.28,
    )
    completion_axis = figure.add_subplot(grid[0, 0])
    clicks_axis = figure.add_subplot(grid[0, 1])
    overall_axis = figure.add_subplot(grid[1, 0])
    failures_axis = figure.add_subplot(grid[1, 1])
    plot_completion(completion_axis, summary)
    plot_paired_clicks(clicks_axis, summary)
    handles, labels = plot_stacked_outcomes(
        overall_axis,
        overall_counts,
        "OneClick outcomes — all phrase trials",
        label_threshold=0.10,
    )
    plot_stacked_outcomes(
        failures_axis,
        failure_counts,
        "OneClick failure causes — failed trials only",
        label_threshold=0.10,
    )
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=min(3, max(len(labels), 1)),
        frameon=False,
    )

    png_path = plot_dir / "team_oneclick_failure_comparison.png"
    pdf_path = plot_dir / "team_oneclick_failure_comparison.pdf"
    detail_path = plot_dir / "team_oneclick_failure_details.csv"
    figure.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    failure_detail_table(classified).to_csv(detail_path, index=False)
    return [png_path, pdf_path, detail_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_telemetry_run()
    outputs = create_oneclick_failure_comparison(run_dir)
    print(f"Created exact-failure OneClick comparison from: {run_dir}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
