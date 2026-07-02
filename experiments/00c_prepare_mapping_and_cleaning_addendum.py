import argparse
from pathlib import Path

import pandas as pd


"""Build question mapping template and cleaning-report addenda.

This supplemental script covers three handoff/QA deliverables:

1. data/question_concept_mapping_template.csv
   One row per unique cleaned item_id, with assessment_title filled from the raw
   PrairieLearn export and concept_id left blank for DAG mapping.

2. data/processed/cleaning_report.md
   Adds per-assessment cleaning statistics.

3. data/processed/cleaning_report.md
   Adds high-retry outlier statistics and a keep-vs-filter recommendation.
"""


DEFAULT_RAW_PATH = Path("data/anonymized_submissions_ECS32A_sq26.csv")
DEFAULT_CLEANED_PATH = Path("data/processed/cleaned_interactions.csv")
DEFAULT_TEMPLATE_PATH = Path("data/question_concept_mapping_template.csv")
DEFAULT_REPORT_PATH = Path("data/processed/cleaning_report.md")

MISSING_STRINGS = {"", "nan", "none", "null", "nah"}

TEMPLATE_START = "<!-- BEGIN QUESTION_CONCEPT_MAPPING_TEMPLATE -->"
TEMPLATE_END = "<!-- END QUESTION_CONCEPT_MAPPING_TEMPLATE -->"
ASSESSMENT_STATS_START = "<!-- BEGIN ASSESSMENT_CLEANING_STATS -->"
ASSESSMENT_STATS_END = "<!-- END ASSESSMENT_CLEANING_STATS -->"
HIGH_RETRY_START = "<!-- BEGIN HIGH_RETRY_OUTLIERS -->"
HIGH_RETRY_END = "<!-- END HIGH_RETRY_OUTLIERS -->"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mapping template and cleaning-report addenda."
    )
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--cleaned-path", default=str(DEFAULT_CLEANED_PATH))
    parser.add_argument("--template-path", default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=60.0,
        help="Maximum repeat-attempt gap treated as plausible active work time.",
    )
    return parser.parse_args()


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_student_id(value):
    value = str(value).strip()
    if value.isdigit():
        value = value.lstrip("0")
        return value if value else "0"
    return value


def missing_value_mask(series):
    as_text = series.astype("string").str.strip()
    return series.isna() | as_text.str.lower().isin(MISSING_STRINGS)


def load_raw(path):
    raw = standardize_columns(pd.read_csv(path, dtype=str))
    if "student_id" not in raw.columns and "anon_student_id" in raw.columns:
        raw = raw.rename(columns={"anon_student_id": "student_id"})

    required = {"student_id", "question_id", "timestamp", "is_correct"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw CSV missing required columns: {sorted(missing)}")

    raw = raw.copy()
    raw["_original_row_order"] = range(len(raw))
    raw["student_id"] = raw["student_id"].apply(normalize_student_id)
    raw["question_id"] = raw["question_id"].astype(str).str.strip()
    raw["timestamp"] = raw["timestamp"].astype(str).str.strip()
    raw["timestamp_dt"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    return raw


def load_cleaned_item_ids(path):
    cleaned = standardize_columns(pd.read_csv(path, dtype=str))
    if "item_id" not in cleaned.columns:
        raise ValueError(f"Cleaned CSV must contain item_id. Found: {list(cleaned.columns)}")

    item_ids = (
        cleaned["item_id"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
    )
    return item_ids.tolist()


def assessment_column(df):
    if "assessment_title" in df.columns:
        return "assessment_title"
    if "assessment_name" in df.columns:
        return "assessment_name"
    raise ValueError("Raw CSV must contain assessment_title or assessment_name.")


def build_unfinished_mask(raw):
    is_correct_missing = missing_value_mask(raw["is_correct"])
    if "score" in raw.columns:
        score_missing = missing_value_mask(raw["score"])
        return is_correct_missing & score_missing
    return is_correct_missing


def prepare_completed_attempts(raw, unfinished_mask):
    completed = raw.loc[~unfinished_mask].copy()
    completed["is_correct_int"] = pd.to_numeric(
        completed["is_correct"], errors="coerce"
    )
    invalid = completed["is_correct_int"].isna() | ~completed["is_correct_int"].isin(
        [0, 1]
    )
    if invalid.any():
        invalid_counts = completed.loc[invalid, "is_correct"].value_counts(dropna=False)
        print(invalid_counts.to_string())
        raise ValueError("Completed rows contain invalid is_correct values.")

    completed = completed.sort_values(
        ["student_id", "question_id", "timestamp", "_original_row_order"],
        ascending=[True, True, True, True],
    ).copy()
    completed["recomputed_attempt_number"] = (
        completed.groupby(["student_id", "question_id"]).cumcount() + 1
    )
    return completed


def deduplicate_like_cleaning_script(completed):
    deduped = completed.sort_values(
        [
            "student_id",
            "question_id",
            "timestamp",
            "recomputed_attempt_number",
            "_original_row_order",
        ],
        ascending=[True, True, True, False, False],
    ).copy()
    return deduped.drop_duplicates(
        subset=["student_id", "question_id", "timestamp"],
        keep="first",
    ).copy()


def add_attempt_intervals(completed, max_gap_minutes):
    completed = completed.copy()
    group_cols = ["student_id", "question_id"]
    completed["prev_timestamp_dt"] = completed.groupby(group_cols)["timestamp_dt"].shift(
        1
    )
    completed["prev_is_correct"] = completed.groupby(group_cols)["is_correct_int"].shift(
        1
    )
    completed["gap_seconds"] = (
        completed["timestamp_dt"] - completed["prev_timestamp_dt"]
    ).dt.total_seconds()
    completed["gap_minutes"] = completed["gap_seconds"] / 60.0
    completed["is_repeat_attempt"] = completed["recomputed_attempt_number"] > 1
    completed["plausible_gap"] = (
        completed["is_repeat_attempt"]
        & completed["gap_seconds"].notna()
        & (completed["gap_seconds"] > 0)
        & (completed["gap_seconds"] <= max_gap_minutes * 60.0)
    )
    return completed


def first_nonempty(values):
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def multi_assessment_items(raw, item_ids):
    assess_col = assessment_column(raw)
    item_id_set = set(item_ids)
    relevant = raw.loc[raw["question_id"].isin(item_id_set)].copy()
    rows = []
    for question_id, group in relevant.groupby("question_id", dropna=False, sort=False):
        titles = [
            str(value).strip()
            for value in group[assess_col]
            if pd.notna(value) and str(value).strip()
        ]
        unique_titles = sorted(set(titles))
        if len(unique_titles) > 1:
            rows.append(
                {
                    "item_id": str(question_id),
                    "selected_assessment_title": first_nonempty(group[assess_col]),
                    "all_observed_assessment_titles": "; ".join(unique_titles),
                }
            )
    return pd.DataFrame(rows)


def build_question_template(raw, cleaned_item_ids):
    assess_col = assessment_column(raw)
    item_id_set = set(cleaned_item_ids)
    assessment_lookup = (
        raw.loc[raw["question_id"].isin(item_id_set)]
        .groupby("question_id", dropna=False, sort=False)
        .agg(assessment_title=(assess_col, first_nonempty))
        .reset_index()
        .rename(columns={"question_id": "item_id"})
    )

    template = pd.DataFrame({"item_id": cleaned_item_ids})
    template = template.merge(assessment_lookup, on="item_id", how="left")
    missing = template["assessment_title"].isna() | template["assessment_title"].eq("")
    if missing.any():
        missing_ids = ", ".join(template.loc[missing, "item_id"].head(20))
        raise ValueError(
            "Some cleaned item_id values have no assessment_title in the raw export: "
            f"{missing_ids}"
        )

    template["question_label"] = ""
    template["concept_id"] = ""
    template["_item_sort"] = pd.to_numeric(template["item_id"], errors="coerce")
    template = template.sort_values(
        ["assessment_title", "_item_sort", "item_id"],
        ascending=[True, True, True],
    )
    return template[
        ["item_id", "assessment_title", "question_label", "concept_id"]
    ].copy()


def assessment_cleaning_stats(raw, cleaned_rows):
    assess_col = assessment_column(raw)
    raw_counts = (
        raw.groupby(assess_col, dropna=False)
        .size()
        .reset_index(name="raw_rows")
        .rename(columns={assess_col: "assessment_title"})
    )
    cleaned_counts = (
        cleaned_rows.groupby(assess_col, dropna=False)
        .size()
        .reset_index(name="cleaned_rows")
        .rename(columns={assess_col: "assessment_title"})
    )
    stats = raw_counts.merge(cleaned_counts, on="assessment_title", how="left")
    stats["cleaned_rows"] = stats["cleaned_rows"].fillna(0).astype(int)
    stats["dropped_rows"] = stats["raw_rows"] - stats["cleaned_rows"]
    stats["drop_rate"] = stats["dropped_rows"] / stats["raw_rows"]
    return stats.sort_values("assessment_title").reset_index(drop=True)


def item_assessment_lookup(template):
    return template[["item_id", "assessment_title"]].copy()


def student_question_summary(attempts, template):
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

    intervals = attempts.loc[attempts["plausible_gap"]].copy()
    if intervals.empty:
        base["valid_repeat_intervals"] = 0
        base["median_repeat_minutes"] = pd.NA
    else:
        timing = (
            intervals.groupby(["student_id", "question_id"])
            .agg(
                valid_repeat_intervals=("gap_minutes", "size"),
                median_repeat_minutes=("gap_minutes", "median"),
            )
            .reset_index()
        )
        base = base.merge(timing, on=["student_id", "question_id"], how="left")
        base["valid_repeat_intervals"] = (
            base["valid_repeat_intervals"].fillna(0).astype(int)
        )

    base = base.rename(columns={"student_id": "user_id", "question_id": "item_id"})
    base["item_id"] = base["item_id"].astype(str)
    base = base.merge(item_assessment_lookup(template), on="item_id", how="left")
    return base.sort_values(
        ["completed_submissions", "observed_span_minutes"],
        ascending=[False, False],
    )


def retry_threshold_stats(student_question, attempts, thresholds):
    intervals = attempts.loc[attempts["plausible_gap"]].copy()
    rows = []
    for threshold in thresholds:
        high_pairs = student_question.loc[
            student_question["completed_submissions"] >= threshold,
            ["user_id", "item_id"],
        ].copy()
        if high_pairs.empty:
            gaps = pd.Series(dtype=float)
        else:
            high_pairs = high_pairs.rename(
                columns={"user_id": "student_id", "item_id": "question_id"}
            )
            intervals_for_pairs = intervals.merge(
                high_pairs, on=["student_id", "question_id"], how="inner"
            )
            gaps = intervals_for_pairs["gap_minutes"]

        rows.append(
            {
                "attempt_threshold": f">={threshold}",
                "student_question_pairs": len(high_pairs),
                "interval_count": int(gaps.size),
                "p25_minutes": gaps.quantile(0.25) if not gaps.empty else pd.NA,
                "median_minutes": gaps.median() if not gaps.empty else pd.NA,
                "p75_minutes": gaps.quantile(0.75) if not gaps.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def format_value(value, column):
    if pd.isna(value):
        return "NA"
    if column.endswith("_rate"):
        return f"{value:.2%}"
    if "minutes" in column:
        return f"{value:.2f}"
    if column in {
        "raw_rows",
        "dropped_rows",
        "cleaned_rows",
        "completed_submissions",
        "valid_repeat_intervals",
        "unique_variants",
        "same_variant_extra_submissions",
        "student_question_pairs",
        "interval_count",
    }:
        return str(int(value))
    if column == "final_is_correct":
        return str(int(value))
    return str(value)


def markdown_table(df, columns, max_rows=None):
    if df.empty:
        return "No rows available."

    rows = df.loc[:, columns]
    if max_rows is not None:
        rows = rows.head(max_rows)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append(
            "| "
            + " | ".join(format_value(row[column], column) for column in columns)
            + " |"
        )
    return "\n".join([header, separator] + body)


def build_template_report_section(template, raw, cleaned_item_ids):
    conflicts = multi_assessment_items(raw, cleaned_item_ids)
    lines = [
        TEMPLATE_START,
        "",
        "## Question-Concept Mapping Template",
        "",
        "`data/question_concept_mapping_template.csv` was generated by extracting "
        "the unique `item_id` values from `data/processed/cleaned_interactions.csv` "
        "and filling `assessment_title` from the raw PrairieLearn export. "
        "`question_label` is blank because this export has no human-readable "
        "question label/title column, and `concept_id` is blank for DAG mapping.",
        "",
        f"- Unique item_id values written: `{len(template)}`",
        f"- Unique item_id values read from cleaned interactions: `{len(cleaned_item_ids)}`",
        f"- Raw rows scanned: `{len(raw)}`",
        f"- Items appearing in multiple assessments: `{len(conflicts)}`",
    ]
    if not conflicts.empty:
        lines.extend(
            [
                "- When one item appears in multiple assessments, the template keeps "
                "one row and uses the first non-empty `assessment_title` observed in "
                "the raw export.",
                "",
                markdown_table(
                    conflicts,
                    [
                        "item_id",
                        "selected_assessment_title",
                        "all_observed_assessment_titles",
                    ],
                ),
            ]
        )
    lines.extend(["", TEMPLATE_END, ""])
    return "\n".join(lines)


def build_assessment_stats_section(stats):
    lines = [
        ASSESSMENT_STATS_START,
        "",
        "## Per-Assessment Cleaning Statistics",
        "",
        "Rows are counted by `assessment_title` from the PrairieLearn raw export. "
        "`dropped_rows` equals raw rows minus rows retained after the same "
        "unfinished-row filter and deduplication logic used by "
        "`experiments/00_clean_raw_data.py`. In the current repository, true "
        "orphan filtering is still skipped because the concept file has no "
        "`question_id` or `item_id` column.",
        "",
        markdown_table(
            stats,
            [
                "assessment_title",
                "raw_rows",
                "dropped_rows",
                "drop_rate",
                "cleaned_rows",
            ],
        ),
        "",
        ASSESSMENT_STATS_END,
        "",
    ]
    return "\n".join(lines)


def build_high_retry_section(student_question, threshold_stats, max_gap_minutes):
    top_outliers = student_question.head(10).copy()
    exactly_119 = student_question.loc[
        student_question["completed_submissions"] == 119
    ].copy()

    lines = [
        HIGH_RETRY_START,
        "",
        "## High-Retry Outlier Review",
        "",
        "This section summarizes repeated completed submissions by "
        "`user_id,item_id`. `completed_submissions` is counted after unfinished "
        "no-score rows are removed, and repeat-attempt intervals use consecutive "
        "submission timestamps for the same student and question.",
        "",
        f"Intervals longer than `{max_gap_minutes:.0f}` minutes, zero-length gaps, "
        "and negative gaps are excluded from interval quantiles because they are "
        "likely idle/cross-session artifacts rather than active work time.",
        "",
        "Retry threshold counts and interval distribution:",
        "",
        markdown_table(
            threshold_stats,
            [
                "attempt_threshold",
                "student_question_pairs",
                "interval_count",
                "p25_minutes",
                "median_minutes",
                "p75_minutes",
            ],
        ),
        "",
        "Highest-attempt student-question pairs:",
        "",
        markdown_table(
            top_outliers,
            [
                "user_id",
                "item_id",
                "assessment_title",
                "completed_submissions",
                "unique_variants",
                "valid_repeat_intervals",
                "observed_span_minutes",
                "median_repeat_minutes",
                "final_is_correct",
            ],
            max_rows=10,
        ),
        "",
    ]

    if not exactly_119.empty:
        lines.extend(
            [
                "The specifically flagged 119-attempt case is:",
                "",
                markdown_table(
                    exactly_119,
                    [
                        "user_id",
                        "item_id",
                        "assessment_title",
                        "completed_submissions",
                        "unique_variants",
                        "valid_repeat_intervals",
                        "observed_span_minutes",
                        "median_repeat_minutes",
                        "final_is_correct",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Recommendation: do not filter these rows out of "
            "`cleaned_interactions.csv`. Keep them in the cleaned data and handle "
            "them in preprocessing/modeling with explicit sensitivity checks, "
            "attempt caps, weighting, or diagnostic flags. The high counts are "
            "partly explained by PrairieLearn variants and short repeated-submit "
            "intervals, so dropping them during cleaning would remove real behavior "
            "before the modeling assumptions are finalized.",
            "",
            HIGH_RETRY_END,
            "",
        ]
    )
    return "\n".join(lines)


def replace_or_append_section(text, start_marker, end_marker, section):
    if start_marker in text and end_marker in text:
        before = text.split(start_marker, 1)[0].rstrip()
        after = text.split(end_marker, 1)[1].lstrip()
        updated = before + "\n\n" + section.rstrip() + "\n"
        if after:
            updated += "\n" + after
        return updated
    return text.rstrip() + "\n\n" + section.rstrip() + "\n"


def update_report(report_path, sections):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    for start_marker, end_marker, section in sections:
        text = replace_or_append_section(text, start_marker, end_marker, section)
    report_path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    raw = load_raw(args.raw_path)
    cleaned_item_ids = load_cleaned_item_ids(args.cleaned_path)
    unfinished_mask = build_unfinished_mask(raw)
    completed = prepare_completed_attempts(raw, unfinished_mask)
    cleaned_rows = deduplicate_like_cleaning_script(completed)
    attempts = add_attempt_intervals(completed, args.max_gap_minutes)

    template = build_question_template(raw, cleaned_item_ids)
    template_path = Path(args.template_path)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(template_path, index=False)

    assessment_stats = assessment_cleaning_stats(raw, cleaned_rows)
    student_question = student_question_summary(attempts, template)
    threshold_stats = retry_threshold_stats(
        student_question=student_question,
        attempts=attempts,
        thresholds=[10, 20, 50, 100],
    )

    sections = [
        (
            TEMPLATE_START,
            TEMPLATE_END,
            build_template_report_section(template, raw, cleaned_item_ids),
        ),
        (
            ASSESSMENT_STATS_START,
            ASSESSMENT_STATS_END,
            build_assessment_stats_section(assessment_stats),
        ),
        (
            HIGH_RETRY_START,
            HIGH_RETRY_END,
            build_high_retry_section(student_question, threshold_stats, args.max_gap_minutes),
        ),
    ]
    update_report(args.report_path, sections)

    print(f"Wrote question-concept mapping template to {template_path}")
    print(f"Updated cleaning report at {args.report_path}")


if __name__ == "__main__":
    main()
