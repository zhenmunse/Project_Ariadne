import argparse
from pathlib import Path

import pandas as pd


# This script turns raw PrairieLearn submission logs into the exact CSV schema
# expected by src/data_engine/preprocessor.py:
# user_id,item_id,is_correct,timestamp
DEFAULT_RAW_INPUT = Path("data/anonymized_submissions_ECS32A_sq26.csv")
DEFAULT_MAPPING_PATH = Path("data/ecs32a_concepts_required_full_v1.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/cleaned_interactions.csv")
OUTPUT_COLUMNS = ["user_id", "item_id", "is_correct", "timestamp"]

# Treat these text values as missing.
# "nah" is a defensive choice for possible exports; it was not observed as a
# literal value in the current ECS32A CSV, where unfinished rows are blank/NaN.
MISSING_STRINGS = {"", "nan", "none", "null", "nah"}


def parse_args():
    """Read command-line options for input, mapping, output, and dry-run mode."""
    parser = argparse.ArgumentParser(
        description="Clean PrairieLearn submission logs for the Ariadne pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_INPUT),
        help="Raw CSV file or directory containing raw CSV files.",
    )
    parser.add_argument(
        "--mapping-path",
        default=str(DEFAULT_MAPPING_PATH),
        help="CSV containing valid question IDs in a question_id or item_id column.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for cleaned_interactions.csv.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_student_id(value):
    """Normalize student IDs while preserving non-numeric anonymized IDs.

    Pure numeric IDs lose leading zeros, for example "000123" -> "123".
    IDs such as "student_030409cf789" are left unchanged except for stripping
    spaces, because their zeros are part of the anonymized identifier.
    """
    value = str(value).strip()
    if value.isdigit():
        value = value.lstrip("0")
        return value if value else "0"
    return value


def standardize_columns(df):
    """Strip spaces around column names without changing their case."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_raw_csv_files(raw_input):
    """Return one CSV file or all CSV files inside a directory."""
    raw_input = Path(raw_input)

    if raw_input.is_file():
        if raw_input.suffix.lower() != ".csv":
            raise ValueError(f"Raw input file must be a CSV: {raw_input}")
        return [raw_input]

    if not raw_input.exists():
        raise FileNotFoundError(f"Raw input path does not exist: {raw_input}")

    if not raw_input.is_dir():
        raise ValueError(f"Raw input must be a CSV file or directory: {raw_input}")

    csv_files = sorted(raw_input.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_input}")

    return csv_files


def load_raw_data(raw_input):
    """Load raw PrairieLearn CSV data and concatenate multiple files if needed."""
    print("Loading raw data...")
    csv_files = find_raw_csv_files(raw_input)
    frames = []

    print(f"Reading raw CSV files from {raw_input}")
    for path in csv_files:
        # Read as strings so pandas does not remove leading zeros from IDs.
        df = pd.read_csv(path, dtype=str)
        df = standardize_columns(df)
        frames.append(df)
        print(f"- {path}: {len(df)} rows")

    raw_df = pd.concat(frames, ignore_index=True)
    print(f"Total raw rows loaded: {len(raw_df)}")
    return raw_df


def load_valid_question_ids(mapping_path, raw_df=None):
    """Load valid question IDs from the concept/item mapping CSV.

    The ideal file contains a question_id or item_id column. The current
    concepts-only CSV in this repo does not have that column, so when raw_df is
    provided this function falls back to all question_id values in the raw log.
    That lets Week 1 cleaning run, while clearly warning that true orphan
    filtering still needs a real question-to-concept mapping file.
    """
    print("Loading concept mapping...")
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Concept mapping file does not exist: {mapping_path}")

    # Read as strings for the same reason: question/item IDs are identifiers,
    # not numbers we should normalize arithmetically.
    mapping = standardize_columns(pd.read_csv(mapping_path, dtype=str))

    if "question_id" in mapping.columns:
        id_column = "question_id"
    elif "item_id" in mapping.columns:
        id_column = "item_id"
    else:
        if raw_df is None or "question_id" not in raw_df.columns:
            raise ValueError(
                "Concept mapping file must contain a question_id or item_id column. "
                f"Found columns: {list(mapping.columns)}"
            )

        print(
            "WARNING: Concept mapping file has no question_id or item_id column. "
            "Using all raw question_id values as a temporary valid set; "
            "true orphan filtering is skipped until a real question mapping is added."
        )
        valid_ids = (
            raw_df["question_id"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
        )
        return set(valid_ids)

    valid_ids = (
        mapping[id_column]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )
    return set(valid_ids)


def validate_required_columns(df):
    """Check that the raw CSV contains every column needed for cleaning."""
    has_student_id = "student_id" in df.columns
    has_anon_student_id = "anon_student_id" in df.columns

    missing = []
    if not has_student_id and not has_anon_student_id:
        missing.append("student_id or anon_student_id")

    for column in ["question_id", "timestamp", "is_correct", "attempt_number"]:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def canonicalize_student_column(df):
    """Use anon_student_id as student_id when the raw export uses that name."""
    df = df.copy()
    if "student_id" not in df.columns and "anon_student_id" in df.columns:
        df = df.rename(columns={"anon_student_id": "student_id"})
    return df


def missing_value_mask(series):
    """Return True for real nulls and configured missing-value strings."""
    as_text = series.astype("string").str.strip()
    return series.isna() | as_text.str.lower().isin(MISSING_STRINGS)


def normalize_ids(df):
    """Normalize student IDs, question IDs, and timestamps into clean strings."""
    print("Normalizing IDs...")
    df = df.copy()
    df["student_id"] = df["student_id"].apply(normalize_student_id)
    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    return df


def drop_missing_is_correct(df):
    """Drop unfinished submissions that do not have a correctness label.

    In the real PrairieLearn export, rows with both is_correct and score empty
    are likely Save/autosave/page-close records. They are not completed
    submissions, so they should not count as learning attempts.

    If these unfinished rows are a large share of the export, print the
    assessment distribution so we can spot a possible PrairieLearn config issue.
    """
    print("Dropping missing is_correct...")
    is_correct_missing = missing_value_mask(df["is_correct"])

    if "score" in df.columns:
        score_missing = missing_value_mask(df["score"])
        # Only drop rows that look unfinished in both correctness and score.
        unfinished_mask = is_correct_missing & score_missing
    else:
        print(
            "No score column found; falling back to dropping rows where "
            "is_correct alone is missing."
        )
        unfinished_mask = is_correct_missing

    dropped = df[unfinished_mask].copy()
    drop_rate = len(dropped) / len(df) if len(df) else 0
    print(f"Rows dropped due to missing is_correct: {len(dropped)}")
    print(f"Missing is_correct drop rate: {drop_rate:.2%}")

    if not dropped.empty:
        print("Dropped missing is_correct by student_id:")
        print(dropped["student_id"].value_counts().head(20).to_string())
        print("Dropped missing is_correct by question_id:")
        print(dropped["question_id"].value_counts().head(20).to_string())

        if drop_rate > 0.10:
            assessment_column = None
            if "assessment_title" in dropped.columns:
                assessment_column = "assessment_title"
            elif "assessment_name" in dropped.columns:
                assessment_column = "assessment_name"

            if assessment_column is not None:
                print(
                    "WARNING: Missing is_correct drop rate exceeds 10%; "
                    f"top assessments by dropped rows ({assessment_column}):"
                )
                print(dropped[assessment_column].value_counts().head(20).to_string())
            else:
                print(
                    "WARNING: Missing is_correct drop rate exceeds 10%, "
                    "but no assessment_title or assessment_name column is available."
                )

    kept = df[~unfinished_mask].copy()
    return kept, len(dropped)


def convert_is_correct(df):
    """Convert is_correct to integer 0/1 and fail on anything else."""
    print("Converting is_correct...")
    df = df.copy()
    numeric = pd.to_numeric(df["is_correct"], errors="coerce")
    invalid_mask = numeric.isna() | ~numeric.isin([0, 1])

    if invalid_mask.any():
        invalid_counts = df.loc[invalid_mask, "is_correct"].value_counts(dropna=False)
        print("Invalid is_correct values:")
        print(invalid_counts.to_string())
        raise ValueError("is_correct must be convertible to int values 0 or 1.")

    df["is_correct"] = numeric.astype(int)
    return df


def recompute_attempt_number(df):
    """Recompute attempts by student and question instead of trusting PL export.

    PrairieLearn can reset attempt_number when the variant_id changes. For this
    learning-path project, different variants of the same question still
    represent repeated attempts on the same item/concept, so we count attempts
    within each (student_id, question_id) group after unfinished rows are removed.
    """
    print("Recomputing attempt_number by student_id/question_id...")
    df = df.copy()
    exported_attempt = pd.to_numeric(df["attempt_number"], errors="coerce")
    invalid_exported = exported_attempt.isna().sum()
    print(f"Rows with missing/invalid exported attempt_number ignored: {invalid_exported}")

    df = df.sort_values(
        by=["student_id", "question_id", "timestamp", "_original_row_order"],
        ascending=[True, True, True, True],
    ).copy()
    df["attempt_number"] = (
        df.groupby(["student_id", "question_id"]).cumcount() + 1
    )
    return df


def deduplicate(df):
    """Remove duplicate student/question/timestamp rows.

    If multiple rows share the same student_id, question_id, and timestamp, keep
    the row with the largest recomputed attempt_number. If that still ties, keep
    the later original row for deterministic behavior.
    """
    print("Deduplicating...")
    before = len(df)
    df = df.sort_values(
        by=[
            "student_id",
            "question_id",
            "timestamp",
            "attempt_number",
            "_original_row_order",
        ],
        ascending=[True, True, True, False, False],
    ).copy()
    df = df.drop_duplicates(
        subset=["student_id", "question_id", "timestamp"],
        keep="first",
    ).copy()
    removed = before - len(df)
    print(f"Duplicate rows removed: {removed}")
    return df, removed


def filter_orphans(df, valid_question_ids):
    """Drop questions that are not present in the supplied question mapping."""
    print("Filtering orphan questions...")
    orphan_mask = ~df["question_id"].isin(valid_question_ids)
    orphan_rows = df[orphan_mask].copy()
    orphan_ids = sorted(orphan_rows["question_id"].dropna().unique())

    print(f"Rows dropped due to orphan question_id: {len(orphan_rows)}")
    print(f"Unique orphan question_ids dropped: {len(orphan_ids)}")

    if orphan_ids:
        print("Orphan question_ids dropped:")
        for qid in orphan_ids[:50]:
            print(f"- {qid}")
        if len(orphan_ids) > 50:
            print(f"... and {len(orphan_ids) - 50} more")

    return df[~orphan_mask].copy(), len(orphan_rows)


def validate_cleaned_output(cleaned_df, valid_question_ids):
    """Run final safety checks before writing cleaned_interactions.csv."""
    print("Validating cleaned data...")
    if cleaned_df.empty:
        raise ValueError("Cleaned output has no rows.")

    duplicate_mask = cleaned_df.duplicated(
        subset=["user_id", "item_id", "timestamp"],
        keep=False,
    )
    if duplicate_mask.any():
        print("Duplicate examples:")
        print(cleaned_df[duplicate_mask].head(20).to_string(index=False))
        raise ValueError("Duplicate user_id/item_id/timestamp rows remain.")

    null_counts = cleaned_df[OUTPUT_COLUMNS].isna().sum()
    empty_text_counts = cleaned_df[["user_id", "item_id", "timestamp"]].apply(
        lambda col: col.astype("string").str.strip().eq("").sum()
    )
    if null_counts.sum() > 0 or empty_text_counts.sum() > 0:
        print("Null counts:")
        print(null_counts.to_string())
        print("Empty string counts:")
        print(empty_text_counts.to_string())
        raise ValueError("Cleaned output contains null or empty required values.")

    orphan_items = sorted(
        set(cleaned_df.loc[~cleaned_df["item_id"].isin(valid_question_ids), "item_id"])
    )
    if orphan_items:
        print("item_id values not in concept mapping:")
        for item_id in orphan_items[:50]:
            print(f"- {item_id}")
        raise ValueError("Cleaned output contains item_id values not in mapping.")

    if not set(cleaned_df["is_correct"].unique()) <= {0, 1}:
        raise ValueError("is_correct contains values outside {0, 1}.")

    parsed_ts = pd.to_datetime(cleaned_df["timestamp"], errors="coerce")
    bad_ts = parsed_ts.isna()
    if bad_ts.any():
        print(f"Unparseable timestamp count: {bad_ts.sum()}")
        print(cleaned_df.loc[bad_ts, "timestamp"].head(20).to_string(index=False))
        raise ValueError("timestamp contains unparseable values.")


def print_summary(stats, cleaned_df):
    """Print the row-count and timestamp summary required by the deliverable."""
    parsed_ts = pd.to_datetime(cleaned_df["timestamp"], errors="coerce")

    print("========== Cleaning Summary ==========")
    print(f"Rows before cleaning: {stats['rows_before']}")
    print(f"Rows dropped due to missing is_correct: {stats['missing_is_correct_dropped']}")
    print(f"Rows after missing is_correct filtering: {stats['after_missing_filtering']}")
    print(f"Duplicate rows removed: {stats['duplicate_rows_removed']}")
    print(f"Rows after deduplication: {stats['after_deduplication']}")
    print(f"Rows dropped due to orphan question_id: {stats['orphan_rows_dropped']}")
    print(f"Rows after orphan filtering: {stats['after_orphan_filtering']}")
    print(f"Rows final: {len(cleaned_df)}")
    print(f"Unique students final: {cleaned_df['user_id'].nunique()}")
    print(f"Unique questions final: {cleaned_df['item_id'].nunique()}")
    print(f"Timestamp min: {cleaned_df.loc[parsed_ts.idxmin(), 'timestamp']}")
    print(f"Timestamp max: {cleaned_df.loc[parsed_ts.idxmax(), 'timestamp']}")
    print(f"Output columns: {', '.join(cleaned_df.columns)}")
    print("======================================")


def clean_interactions(raw_df, valid_question_ids):
    """Run the full cleaning pipeline and return the four-column output frame."""
    stats = {"rows_before": len(raw_df)}

    df = standardize_columns(raw_df)
    validate_required_columns(df)
    df = canonicalize_student_column(df)
    df = df.copy()

    # Keep original order so tie-breaking is stable and reproducible.
    df["_original_row_order"] = range(len(df))

    df = normalize_ids(df)
    df, stats["missing_is_correct_dropped"] = drop_missing_is_correct(df)
    stats["after_missing_filtering"] = len(df)

    df = convert_is_correct(df)
    df = recompute_attempt_number(df)
    df, stats["duplicate_rows_removed"] = deduplicate(df)
    stats["after_deduplication"] = len(df)

    df, stats["orphan_rows_dropped"] = filter_orphans(df, valid_question_ids)
    stats["after_orphan_filtering"] = len(df)

    # Remove internal helper columns before final schema selection. OUTPUT_COLUMNS
    # already prevents leakage, but this keeps future edits safer.
    df = df.drop(columns=["_original_row_order"], errors="ignore")

    # Rename to the exact schema expected by the next preprocessing step.
    cleaned_df = df.rename(
        columns={"student_id": "user_id", "question_id": "item_id"}
    )[OUTPUT_COLUMNS].copy()
    cleaned_df["user_id"] = cleaned_df["user_id"].astype(str)
    cleaned_df["item_id"] = cleaned_df["item_id"].astype(str)
    cleaned_df["is_correct"] = cleaned_df["is_correct"].astype(int)
    cleaned_df["timestamp"] = cleaned_df["timestamp"].astype(str)

    validate_cleaned_output(cleaned_df, valid_question_ids)
    print_summary(stats, cleaned_df)
    return cleaned_df


def main():
    """CLI entry point."""
    args = parse_args()
    raw_df = load_raw_data(args.raw_dir)
    valid_question_ids = load_valid_question_ids(args.mapping_path, raw_df)
    cleaned_df = clean_interactions(raw_df, valid_question_ids)

    if args.dry_run:
        print("Dry run enabled. Output file was not written.")
        return

    print("Writing output...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Wrote cleaned data to {output_path}")


if __name__ == "__main__":
    main()
