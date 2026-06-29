import argparse
from pathlib import Path

import pandas as pd


"""Supplemental timing analysis for the cleaned PrairieLearn data.

This script is NOT the main data-cleaning pipeline. The model-/pipeline-ready
dataset is produced by experiments/00_clean_raw_data.py and saved at:

    data/processed/cleaned_interactions.csv

This script only creates auxiliary analysis files:

1. data/processed/question_attempt_time_summary.csv
   One row per item_id/question_id, aggregated across all students.

2. data/processed/student_question_attempt_time_summary.csv
   One row per user_id + item_id pair, showing that student's observed timing
   pattern on that question.

3. data/processed/cleaning_report.md
   Human-readable report with cleaning notes and timing summaries.

The raw PrairieLearn export does not include true page-open or active-time
fields. All timing values here are inferred from submission timestamps, so they
should be treated as approximate behavioral summaries, not exact time-on-task.
"""


DEFAULT_RAW_PATH = Path("data/anonymized_submissions_ECS32A_sq26.csv")

# Main cleaned dataset used by downstream preprocessor/oracle/planner code.
# The two timing-summary CSV files below are auxiliary and should not replace it.
DEFAULT_CLEANED_PATH = Path("data/processed/cleaned_interactions.csv")
DEFAULT_REPORT_PATH = Path("data/processed/cleaning_report.md")

# Auxiliary table: one row per question/item, aggregated across students.
DEFAULT_QUESTION_TIME_PATH = Path("data/processed/question_attempt_time_summary.csv")

# Auxiliary table: one row per student-question pair.
DEFAULT_STUDENT_QUESTION_TIME_PATH = Path(
    "data/processed/student_question_attempt_time_summary.csv"
)
MISSING_STRINGS = {"", "nan", "none", "null", "nah"}
REPORT_START = "<!-- BEGIN ATTEMPT_TIME_ANALYSIS -->"
REPORT_END = "<!-- END ATTEMPT_TIME_ANALYSIS -->"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze unfinished submissions and timestamp-based attempt timing."
    )
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--cleaned-path", default=str(DEFAULT_CLEANED_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--question-time-path", default=str(DEFAULT_QUESTION_TIME_PATH))
    parser.add_argument(
        "--student-question-time-path",
        default=str(DEFAULT_STUDENT_QUESTION_TIME_PATH),
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=60.0,
        help=(
            "Maximum same-student/same-question gap treated as plausible work time. "
            "Longer gaps are reported but excluded from timing averages."
        ),
    )
    return parser.parse_args()


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def missing_value_mask(series):
    as_text = series.astype("string").str.strip()
    return series.isna() | as_text.str.lower().isin(MISSING_STRINGS)


def normalize_student_id(value):
    value = str(value).strip()
    if value.isdigit():
        value = value.lstrip("0")
        return value if value else "0"
    return value


def load_raw(path):
    raw = standardize_columns(pd.read_csv(path, dtype=str))
    if "student_id" not in raw.columns and "anon_student_id" in raw.columns:
        raw = raw.rename(columns={"anon_student_id": "student_id"})

    required = {"student_id", "question_id", "timestamp", "is_correct"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw CSV missing columns required for timing analysis: {missing}")

    raw = raw.copy()
    raw["_original_row_order"] = range(len(raw))
    raw["student_id"] = raw["student_id"].apply(normalize_student_id)
    raw["question_id"] = raw["question_id"].astype(str).str.strip()
    raw["timestamp"] = raw["timestamp"].astype(str).str.strip()
    raw["timestamp_dt"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    return raw


def load_cleaned_summary(path):
    """Read the main cleaned CSV and return a compact validation summary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned interaction CSV does not exist: {path}")

    cleaned = standardize_columns(pd.read_csv(path, dtype=str))
    expected_columns = ["user_id", "item_id", "is_correct", "timestamp"]
    has_expected = set(expected_columns).issubset(cleaned.columns)
    duplicate_count = (
        cleaned.duplicated(["user_id", "item_id", "timestamp"]).sum()
        if has_expected
        else pd.NA
    )
    null_count = (
        cleaned[expected_columns].isna().sum().sum()
        if has_expected
        else pd.NA
    )

    return {
        "path": str(path),
        "rows": len(cleaned),
        "columns": list(cleaned.columns),
        "exact_schema": list(cleaned.columns) == expected_columns,
        "null_values": int(null_count) if not pd.isna(null_count) else pd.NA,
        "duplicates": int(duplicate_count) if not pd.isna(duplicate_count) else pd.NA,
    }


def build_unfinished_mask(raw):
    is_correct_missing = missing_value_mask(raw["is_correct"])
    if "score" in raw.columns:
        score_missing = missing_value_mask(raw["score"])
        return is_correct_missing & score_missing
    return is_correct_missing


def assessment_column(df):
    if "assessment_title" in df.columns:
        return "assessment_title"
    if "assessment_name" in df.columns:
        return "assessment_name"
    return None


def unfinished_by_assessment(raw, unfinished_mask):
    column = assessment_column(raw)
    if column is None:
        return pd.DataFrame()

    summary = (
        raw.assign(_unfinished=unfinished_mask)
        .groupby(column, dropna=False)
        .agg(total_rows=("_unfinished", "size"), unfinished_rows=("_unfinished", "sum"))
        .reset_index()
        .rename(columns={column: "assessment"})
    )
    summary["unfinished_rate"] = summary["unfinished_rows"] / summary["total_rows"]
    return summary.sort_values(
        ["unfinished_rows", "unfinished_rate"],
        ascending=[False, False],
    )


def prepare_completed_attempts(raw, unfinished_mask):
    completed = raw.loc[~unfinished_mask].copy()
    completed["is_correct_int"] = pd.to_numeric(
        completed["is_correct"],
        errors="coerce",
    )
    invalid = completed["is_correct_int"].isna() | ~completed["is_correct_int"].isin([0, 1])
    if invalid.any():
        print("Invalid completed is_correct values:")
        print(completed.loc[invalid, "is_correct"].value_counts(dropna=False).to_string())
        raise ValueError("Completed rows contain invalid is_correct values.")

    completed = completed.loc[~completed["timestamp_dt"].isna()].copy()
    completed = completed.sort_values(
        ["student_id", "question_id", "timestamp_dt", "_original_row_order"],
        ascending=[True, True, True, True],
    )
    completed["recomputed_attempt_number"] = (
        completed.groupby(["student_id", "question_id"]).cumcount() + 1
    )
    return completed


def add_attempt_intervals(completed, max_gap_minutes):
    completed = completed.copy()
    group_cols = ["student_id", "question_id"]
    completed["prev_timestamp_dt"] = completed.groupby(group_cols)["timestamp_dt"].shift(1)
    completed["prev_is_correct"] = completed.groupby(group_cols)["is_correct_int"].shift(1)
    completed["gap_seconds"] = (
        completed["timestamp_dt"] - completed["prev_timestamp_dt"]
    ).dt.total_seconds()
    completed["gap_minutes"] = completed["gap_seconds"] / 60.0

    max_gap_seconds = max_gap_minutes * 60.0
    completed["is_repeat_attempt"] = completed["recomputed_attempt_number"] > 1
    completed["plausible_gap"] = (
        completed["is_repeat_attempt"]
        & completed["gap_seconds"].notna()
        & (completed["gap_seconds"] > 0)
        & (completed["gap_seconds"] <= max_gap_seconds)
    )
    return completed


def timing_reduction_summary(attempts, min_previous_gap_minutes=1.0):
    intervals = attempts.loc[attempts["plausible_gap"]].copy()
    if intervals.empty:
        return {
            "all_pair_count": 0,
            "robust_pair_count": 0,
            "min_previous_gap_minutes": min_previous_gap_minutes,
            "median_reduction_rate": pd.NA,
            "percent_decreased": pd.NA,
        }

    intervals = intervals.sort_values(
        ["student_id", "question_id", "recomputed_attempt_number"]
    )
    group_cols = ["student_id", "question_id"]
    intervals["prev_gap_minutes"] = intervals.groupby(group_cols)["gap_minutes"].shift(1)
    comparable = intervals.loc[
        intervals["prev_gap_minutes"].notna() & (intervals["prev_gap_minutes"] > 0)
    ].copy()
    if comparable.empty:
        return {
            "all_pair_count": 0,
            "robust_pair_count": 0,
            "min_previous_gap_minutes": min_previous_gap_minutes,
            "median_reduction_rate": pd.NA,
            "percent_decreased": pd.NA,
        }

    robust = comparable.loc[
        comparable["prev_gap_minutes"] >= min_previous_gap_minutes
    ].copy()
    if robust.empty:
        return {
            "all_pair_count": len(comparable),
            "robust_pair_count": 0,
            "min_previous_gap_minutes": min_previous_gap_minutes,
            "median_reduction_rate": pd.NA,
            "percent_decreased": pd.NA,
        }

    robust["reduction_rate"] = (
        robust["prev_gap_minutes"] - robust["gap_minutes"]
    ) / robust["prev_gap_minutes"]
    return {
        "all_pair_count": len(comparable),
        "robust_pair_count": len(robust),
        "min_previous_gap_minutes": min_previous_gap_minutes,
        "median_reduction_rate": robust["reduction_rate"].median(),
        "percent_decreased": (robust["reduction_rate"] > 0).mean(),
    }


def question_time_summary(attempts):
    intervals = attempts.loc[attempts["plausible_gap"]].copy()
    completed_counts = (
        attempts.groupby("question_id")
        .agg(
            completed_submissions=("question_id", "size"),
            students=("student_id", "nunique"),
        )
        .reset_index()
    )
    if intervals.empty:
        completed_counts["valid_repeat_intervals"] = 0
        completed_counts["mean_minutes"] = pd.NA
        completed_counts["median_minutes"] = pd.NA
        completed_counts["p75_minutes"] = pd.NA
        return completed_counts.rename(columns={"question_id": "item_id"})

    timing = (
        intervals.groupby("question_id")
        .agg(
            valid_repeat_intervals=("gap_minutes", "size"),
            mean_minutes=("gap_minutes", "mean"),
            median_minutes=("gap_minutes", "median"),
            p75_minutes=("gap_minutes", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    summary = completed_counts.merge(timing, on="question_id", how="left")
    summary["valid_repeat_intervals"] = summary["valid_repeat_intervals"].fillna(0).astype(int)
    summary = summary.rename(columns={"question_id": "item_id"})
    return summary.sort_values(
        ["valid_repeat_intervals", "mean_minutes"],
        ascending=[False, False],
    )


def student_question_time_summary(attempts):
    """Estimate time spent per student/question from observed submission gaps.

    The first-attempt duration is unknown because the export has no page-open
    timestamp. For repeated completed submissions, valid gaps between adjacent
    submissions approximate continued work on the same question.
    """
    intervals = attempts.loc[attempts["plausible_gap"]].copy()

    agg_spec = {
        "completed_submissions": ("question_id", "size"),
        "first_completed_timestamp": ("timestamp", "first"),
        "last_completed_timestamp": ("timestamp", "last"),
        "first_completed_dt": ("timestamp_dt", "first"),
        "last_completed_dt": ("timestamp_dt", "last"),
        "correct_submissions": ("is_correct_int", "sum"),
        "final_is_correct": ("is_correct_int", "last"),
    }
    if "variant_id" in attempts.columns:
        agg_spec["unique_variants"] = ("variant_id", "nunique")

    base = attempts.groupby(["student_id", "question_id"]).agg(**agg_spec).reset_index()
    base["observed_span_minutes"] = (
        base["last_completed_dt"] - base["first_completed_dt"]
    ).dt.total_seconds() / 60.0
    base = base.drop(columns=["first_completed_dt", "last_completed_dt"])
    if "unique_variants" in base.columns:
        base["same_variant_extra_submissions"] = (
            base["completed_submissions"] - base["unique_variants"]
        )

    if intervals.empty:
        base["valid_repeat_intervals"] = 0
        base["total_valid_repeat_minutes"] = 0.0
        base["mean_repeat_minutes"] = pd.NA
        base["median_repeat_minutes"] = pd.NA
    else:
        timing = (
            intervals.groupby(["student_id", "question_id"])
            .agg(
                valid_repeat_intervals=("gap_minutes", "size"),
                total_valid_repeat_minutes=("gap_minutes", "sum"),
                mean_repeat_minutes=("gap_minutes", "mean"),
                median_repeat_minutes=("gap_minutes", "median"),
            )
            .reset_index()
        )
        base = base.merge(timing, on=["student_id", "question_id"], how="left")
        base["valid_repeat_intervals"] = (
            base["valid_repeat_intervals"].fillna(0).astype(int)
        )
        base["total_valid_repeat_minutes"] = (
            base["total_valid_repeat_minutes"].fillna(0.0)
        )

    base = base.rename(
        columns={
            "student_id": "user_id",
            "question_id": "item_id",
        }
    )
    return base.sort_values(
        ["valid_repeat_intervals", "observed_span_minutes"],
        ascending=[False, False],
    )


def attempt_number_summary(attempts):
    intervals = attempts.loc[attempts["plausible_gap"]].copy()
    if intervals.empty:
        return pd.DataFrame()
    return (
        intervals.groupby("recomputed_attempt_number")
        .agg(
            valid_intervals=("gap_minutes", "size"),
            mean_minutes=("gap_minutes", "mean"),
            median_minutes=("gap_minutes", "median"),
            after_incorrect_intervals=("prev_is_correct", lambda s: int((s == 0).sum())),
        )
        .reset_index()
        .sort_values("recomputed_attempt_number")
    )


def format_percent(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def format_minutes(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2f}"


def markdown_table(df, columns, max_rows=10):
    if df.empty:
        return "No rows available."

    rows = df.loc[:, columns].head(max_rows).copy()
    formatted_rows = []
    for _, row in rows.iterrows():
        formatted = []
        for col in columns:
            value = row[col]
            if "rate" in col or "percent" in col:
                formatted.append(format_percent(value))
            elif "minutes" in col:
                formatted.append(format_minutes(value))
            elif (
                col.endswith("_rows")
                or col.endswith("_submissions")
                or col.endswith("_intervals")
                or col.endswith("_variants")
                or col.endswith("_extra_submissions")
                or col in {"students", "recomputed_attempt_number"}
            ):
                formatted.append(str(int(value)) if not pd.isna(value) else "NA")
            else:
                formatted.append(str(value))
        formatted_rows.append(formatted)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted_rows]
    return "\n".join([header, separator] + body)


def build_report_section(
    raw,
    cleaned_summary,
    unfinished_mask,
    attempts,
    assessment_summary,
    question_summary,
    student_question_summary,
    max_gap_minutes,
):
    total_raw = len(raw)
    unfinished_count = int(unfinished_mask.sum())
    unfinished_rate = unfinished_count / total_raw if total_raw else 0
    repeat_count = int(attempts["is_repeat_attempt"].sum())
    plausible_count = int(attempts["plausible_gap"].sum())
    long_or_bad_count = repeat_count - plausible_count
    plausible = attempts.loc[attempts["plausible_gap"]]
    after_incorrect = plausible.loc[plausible["prev_is_correct"] == 0]
    reduction = timing_reduction_summary(attempts)
    by_attempt = attempt_number_summary(attempts)

    lines = [
        REPORT_START,
        "",
        "## Supplemental Attempt-Time Analysis",
        "",
        "This section was generated by `experiments/00b_analyze_attempt_timing.py`.",
        "",
        "Important limitation: the PrairieLearn export does not include a true "
        "page-open timestamp or active time-on-task field. The timing metrics below "
        "use the elapsed time between consecutive completed submissions for the same "
        "student and question. First-attempt duration cannot be measured from this "
        "CSV alone.",
        "",
        f"Gaps longer than `{max_gap_minutes:.0f}` minutes are treated as likely idle/cross-session gaps "
        "and are excluded from average-time calculations.",
        "",
        "### File Roles",
        "",
        "Files changed or produced by this data-cleaning work:",
        "",
        "| file | role | main pipeline input? |",
        "| --- | --- | --- |",
        "| `experiments/00_clean_raw_data.py` | Main cleaning script. Produces the model-ready four-column interaction CSV. | Yes |",
        "| `data/processed/cleaned_interactions.csv` | Cleaned usable data for `src/data_engine/preprocessor.py`, Oracle, and Planner. | Yes |",
        "| `experiments/00b_analyze_attempt_timing.py` | Supplemental timing/QA analysis script. Does not replace the cleaned data. | No |",
        "| `data/processed/question_attempt_time_summary.csv` | Auxiliary per-question timing summary, one row per `item_id`. | No |",
        "| `data/processed/student_question_attempt_time_summary.csv` | Auxiliary per-student-per-question timing summary, one row per `user_id,item_id`. | No |",
        "| `data/processed/cleaning_report.md` | Human-readable cleaning and QA log. | No |",
        "",
        "Main cleaned CSV sanity check:",
        "",
        f"- Path: `{cleaned_summary['path']}`",
        f"- Rows: `{cleaned_summary['rows']}`",
        f"- Columns: `{','.join(cleaned_summary['columns'])}`",
        f"- Exact schema `user_id,item_id,is_correct,timestamp`: `{cleaned_summary['exact_schema']}`",
        f"- Null values in required columns: `{cleaned_summary['null_values']}`",
        f"- Duplicate `user_id,item_id,timestamp` rows: `{cleaned_summary['duplicates']}`",
        "",
        "`question_attempt_time_summary.csv` and "
        "`student_question_attempt_time_summary.csv` are not the same file. The "
        "first aggregates timing by question across all students; the second keeps "
        "separate rows for each student-question pair.",
        "",
        "### Unfinished / No-Score Rows",
        "",
        f"- Raw rows: `{total_raw}`",
        f"- Rows with both `is_correct` and `score` missing: `{unfinished_count}`",
        f"- Overall unfinished/no-score rate: `{unfinished_rate:.2%}`",
        "",
        "Top assessments by unfinished/no-score rate among the largest contributors:",
        "",
        markdown_table(
            assessment_summary,
            ["assessment", "total_rows", "unfinished_rows", "unfinished_rate"],
            max_rows=10,
        ),
        "",
        "### Timestamp-Based Attempt Timing",
        "",
        f"- Completed submissions analyzed: `{len(attempts)}`",
        f"- Repeat attempts with a previous completed submission: `{repeat_count}`",
        f"- Plausible repeat-attempt intervals used for timing: `{plausible_count}`",
        f"- Repeat intervals excluded as nonpositive or over cap: `{long_or_bad_count}`",
        f"- Mean repeat-attempt interval: `{format_minutes(plausible['gap_minutes'].mean())}` minutes",
        f"- Median repeat-attempt interval: `{format_minutes(plausible['gap_minutes'].median())}` minutes",
        f"- Mean interval after an incorrect previous attempt: `{format_minutes(after_incorrect['gap_minutes'].mean())}` minutes",
        f"- Median interval after an incorrect previous attempt: `{format_minutes(after_incorrect['gap_minutes'].median())}` minutes",
        "",
        "Attempt-number timing summary:",
        "",
        markdown_table(
            by_attempt,
            [
                "recomputed_attempt_number",
                "valid_intervals",
                "mean_minutes",
                "median_minutes",
                "after_incorrect_intervals",
            ],
            max_rows=8,
        ),
        "",
        "Consecutive interval reduction summary:",
        "",
        f"- All comparable consecutive interval pairs: `{reduction['all_pair_count']}`",
        f"- Robust pairs with previous interval >= `{reduction['min_previous_gap_minutes']:.1f}` minute: `{reduction['robust_pair_count']}`",
        f"- Median reduction rate on robust pairs: `{format_percent(reduction['median_reduction_rate'])}`",
        f"- Share of robust pairs where the later interval was shorter: `{format_percent(reduction['percent_decreased'])}`",
        "",
        "The mean reduction rate is intentionally not reported because ratios become "
        "unstable when the previous interval is only a few seconds.",
        "",
        "### Average Repeat-Attempt Time By Question",
        "",
        "`data/processed/question_attempt_time_summary.csv` contains the full per-question table. "
        "Top rows by number of valid repeat intervals:",
        "",
        markdown_table(
            question_summary,
            [
                "item_id",
                "completed_submissions",
                "students",
                "valid_repeat_intervals",
                "mean_minutes",
                "median_minutes",
                "p75_minutes",
            ],
            max_rows=15,
        ),
        "",
        "### Observed Time By Student And Question",
        "",
        "`data/processed/student_question_attempt_time_summary.csv` contains the "
        "student-question level table. `observed_span_minutes` is the time from "
        "the first completed submission to the last completed submission for that "
        "student/question. It is 0 for single-submission cases and does not include "
        "unknown time before the first submission.",
        "",
        "`completed_submissions` is counted under the same `question_id` and does "
        "not separate PrairieLearn variants. When `unique_variants` is close to "
        "`completed_submissions`, the high count mostly means the student saw many "
        "different variants of the same question template, not the exact same static "
        "problem repeated unchanged.",
        "",
        markdown_table(
            student_question_summary,
            [
                "user_id",
                "item_id",
                "completed_submissions",
                "unique_variants",
                "same_variant_extra_submissions",
                "valid_repeat_intervals",
                "observed_span_minutes",
                "total_valid_repeat_minutes",
                "mean_repeat_minutes",
                "final_is_correct",
            ],
            max_rows=15,
        ),
        "",
        REPORT_END,
        "",
    ]
    return "\n".join(lines)


def update_report(report_path, section):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
    else:
        existing = ""

    if REPORT_START in existing and REPORT_END in existing:
        before = existing.split(REPORT_START)[0].rstrip()
        after = existing.split(REPORT_END, 1)[1].lstrip()
        new_text = before + "\n\n" + section.rstrip() + "\n"
        if after:
            new_text += "\n" + after
    else:
        new_text = existing.rstrip() + "\n\n" + section.rstrip() + "\n"

    report_path.write_text(new_text, encoding="utf-8")


def main():
    args = parse_args()
    raw = load_raw(args.raw_path)
    cleaned_summary = load_cleaned_summary(args.cleaned_path)
    unfinished_mask = build_unfinished_mask(raw)
    completed = prepare_completed_attempts(raw, unfinished_mask)
    attempts = add_attempt_intervals(completed, args.max_gap_minutes)
    assessment_summary = unfinished_by_assessment(raw, unfinished_mask)
    question_summary = question_time_summary(attempts)
    student_question_summary = student_question_time_summary(attempts)

    question_path = Path(args.question_time_path)
    question_path.parent.mkdir(parents=True, exist_ok=True)
    question_summary.to_csv(question_path, index=False)

    student_question_path = Path(args.student_question_time_path)
    student_question_path.parent.mkdir(parents=True, exist_ok=True)
    student_question_summary.to_csv(student_question_path, index=False)

    section = build_report_section(
        raw=raw,
        cleaned_summary=cleaned_summary,
        unfinished_mask=unfinished_mask,
        attempts=attempts,
        assessment_summary=assessment_summary,
        question_summary=question_summary,
        student_question_summary=student_question_summary,
        max_gap_minutes=args.max_gap_minutes,
    )
    update_report(args.report_path, section)

    print(f"Wrote per-question timing summary to {question_path}")
    print(f"Wrote per-student-question timing summary to {student_question_path}")
    print(f"Updated cleaning report at {args.report_path}")


if __name__ == "__main__":
    main()
