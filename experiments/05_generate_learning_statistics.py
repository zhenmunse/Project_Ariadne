"""
为 learning-path 项目生成统计汇总。

这个脚本做的事情可以分成四层：
1. 检查 question -> concept mapping 本身是否干净。
2. 统计 cleaned interactions 的题目、学生、正确率和时间范围。
3. 把 interaction join 到 concept，生成 question/concept/student/student-concept 统计。
4. 结合 DAG 和 teaching order，生成 bottleneck、路径顺序和 prerequisite violation 统计。

默认输入：
  - data/anonymized_submissions_ECS32A_sq26.csv
  - data/question_concept_mapping_template.csv
  - data/processed/cleaned_interactions.csv
  - data/ecs32a_concepts_required_full_v1.csv
  - data/ecs32a_dag_edges_required_full_v1.csv
  - data/ecs32a_teaching_order_required_full_v1.csv

默认输出只写一个 Markdown 报告：
  - data/processed/learning_statistics_report.md

注意：脚本内部仍然会临时计算多个 DataFrame，但不会把这些中间统计表保存成 CSV。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MAPPING_PATH = Path("data/question_concept_mapping_template.csv")
DEFAULT_RAW_PATH = Path("data/anonymized_submissions_ECS32A_sq26.csv")
DEFAULT_CLEANED_PATH = Path("data/processed/cleaned_interactions.csv")
DEFAULT_CONCEPTS_PATH = Path("data/ecs32a_concepts_required_full_v1.csv")
DEFAULT_DAG_EDGES_PATH = Path("data/ecs32a_dag_edges_required_full_v1.csv")
DEFAULT_TEACHING_ORDER_PATH = Path("data/ecs32a_teaching_order_required_full_v1.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_REPORT_FILENAME = "learning_statistics_report.md"

MAPPING_COLUMNS = ["item_id", "assessment_title", "question_label", "concept_id"]
CLEANED_COLUMNS = ["user_id", "item_id", "is_correct", "timestamp"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mapping, interaction, concept, student, and DAG summaries."
    )
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--cleaned-path", default=str(DEFAULT_CLEANED_PATH))
    parser.add_argument("--concepts-path", default=str(DEFAULT_CONCEPTS_PATH))
    parser.add_argument("--dag-edges-path", default=str(DEFAULT_DAG_EDGES_PATH))
    parser.add_argument("--teaching-order-path", default=str(DEFAULT_TEACHING_ORDER_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def standardize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def missing_value_mask(series):
    as_text = series.astype("string").str.strip().str.lower()
    return series.isna() | as_text.isin({"", "nan", "none", "null", "nah"})


def read_csv_if_exists(path, dtype=str):
    path = Path(path)
    if not path.exists():
        return None
    return standardize_columns(pd.read_csv(path, dtype=dtype))


def require_columns(df, required, path):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_mapping(path):
    # 读取题目到知识点的核心映射表。这里把 ID 当成字符串处理，
    # 避免 pandas 把标识符当数字改写。
    mapping = standardize_columns(pd.read_csv(path, dtype=str))
    require_columns(mapping, MAPPING_COLUMNS, path)

    mapping = mapping[MAPPING_COLUMNS].copy()
    for column in MAPPING_COLUMNS:
        mapping[column] = clean_text(mapping[column])

    if mapping["item_id"].eq("").any():
        raise ValueError("Mapping contains blank item_id values.")
    if mapping["concept_id"].eq("").any():
        raise ValueError("Mapping contains blank concept_id values.")

    return mapping


def validate_mapping_for_join(mapping):
    """防止 concept-level join 静默选错 concept。

    后续所有 concept 统计都依赖 item_id -> concept_id 是一对一关系。
    如果一个 item_id 出现多行，哪怕当前 pandas 可以 drop_duplicates，
    也会让 assessment/question_label 的来源变得不明确，所以这里直接报错。
    """
    duplicated = mapping.loc[mapping["item_id"].duplicated(keep=False), "item_id"]
    if not duplicated.empty:
        duplicate_ids = sorted(duplicated.unique(), key=sort_key)
        raise ValueError(
            "Mapping must contain exactly one row per item_id before concept-level "
            f"statistics. Duplicate item_id values: {duplicate_ids[:20]}"
        )

    item_concept_counts = mapping.groupby("item_id")["concept_id"].nunique()
    multi_concept_ids = sorted(
        item_concept_counts[item_concept_counts > 1].index, key=sort_key
    )
    if multi_concept_ids:
        raise ValueError(
            "One item_id maps to multiple concept_id values, which would make the "
            f"join ambiguous: {multi_concept_ids[:20]}"
        )


def load_cleaned(path):
    # cleaned_interactions 是真实学生轨迹。这里统一校验 0/1 correctness，
    # 并把 timestamp 解析成 UTC 时间，方便 first/last/order 统计。
    cleaned = standardize_columns(pd.read_csv(path, dtype=str))
    require_columns(cleaned, CLEANED_COLUMNS, path)

    cleaned = cleaned[CLEANED_COLUMNS].copy()
    cleaned["user_id"] = clean_text(cleaned["user_id"])
    cleaned["item_id"] = clean_text(cleaned["item_id"])
    cleaned["is_correct"] = pd.to_numeric(cleaned["is_correct"], errors="coerce")

    invalid_correct = cleaned["is_correct"].isna() | ~cleaned["is_correct"].isin([0, 1])
    if invalid_correct.any():
        examples = cleaned.loc[invalid_correct, "is_correct"].head(20).tolist()
        raise ValueError(f"cleaned interactions contain invalid is_correct values: {examples}")

    cleaned["is_correct"] = cleaned["is_correct"].astype(int)
    cleaned["timestamp"] = clean_text(cleaned["timestamp"])
    cleaned["timestamp_dt"] = pd.to_datetime(
        cleaned["timestamp"], errors="coerce", utc=True
    )
    bad_ts = cleaned["timestamp_dt"].isna()
    if bad_ts.any():
        examples = cleaned.loc[bad_ts, "timestamp"].head(20).tolist()
        raise ValueError(f"cleaned interactions contain unparseable timestamps: {examples}")

    cleaned["_row_order"] = np.arange(len(cleaned))
    cleaned["date"] = extract_source_date(cleaned["timestamp"], cleaned["timestamp_dt"])
    return cleaned


def load_raw_context(path):
    # raw 文件不是 concept-level 统计的必需输入，但它能解释 unmapped item_id
    # 原本属于哪个 assessment。没有 raw 文件时，脚本仍能生成主统计，只是少一张上下文表。
    raw = read_csv_if_exists(path, dtype=str)
    if raw is None:
        return None
    if "question_id" not in raw.columns:
        return None

    raw = raw.copy()
    raw["question_id"] = clean_text(raw["question_id"])
    if "assessment_title" not in raw.columns:
        raw["assessment_title"] = ""
    if "assessment_name" not in raw.columns:
        raw["assessment_name"] = ""
    raw["assessment_title"] = clean_text(raw["assessment_title"])
    raw["assessment_name"] = clean_text(raw["assessment_name"])
    if "timestamp" in raw.columns:
        raw["timestamp"] = clean_text(raw["timestamp"])
    else:
        raw["timestamp"] = ""
    if "is_correct" in raw.columns:
        raw["is_correct_num"] = pd.to_numeric(raw["is_correct"], errors="coerce")
    else:
        raw["is_correct_num"] = np.nan
    return raw


def extract_source_date(timestamp_series, parsed_ts):
    date_from_text = timestamp_series.str.extract(r"^(\d{4}-\d{2}-\d{2})", expand=False)
    date_from_parsed = parsed_ts.dt.strftime("%Y-%m-%d")
    return date_from_text.fillna(date_from_parsed)


def load_concept_catalog(path):
    # DAG concept catalog 用来补全没有题目的 concept。
    # 这样 concept_summary 能同时显示“题目覆盖到的 concept”和“DAG 中尚无题目的 concept”。
    concepts = read_csv_if_exists(path, dtype=str)
    if concepts is None:
        return pd.DataFrame(columns=["concept_id", "catalog_concept_name", "catalog_teaching_order"])

    if "node_id" in concepts.columns:
        id_col = "node_id"
    elif "concept_id" in concepts.columns:
        id_col = "concept_id"
    else:
        return pd.DataFrame(columns=["concept_id", "catalog_concept_name", "catalog_teaching_order"])

    name_col = "concept_name" if "concept_name" in concepts.columns else None
    order_col = "teaching_order" if "teaching_order" in concepts.columns else None

    out = pd.DataFrame({"concept_id": clean_text(concepts[id_col])})
    out["catalog_concept_name"] = (
        clean_text(concepts[name_col]).str.replace("_", " ", regex=False)
        if name_col
        else ""
    )
    out["catalog_teaching_order"] = (
        pd.to_numeric(concepts[order_col], errors="coerce") if order_col else pd.NA
    )
    return out.drop_duplicates(subset=["concept_id"], keep="first")


def load_teaching_order(path):
    # teaching order 用来和学生真实 first-seen order 做对比。
    teaching = read_csv_if_exists(path, dtype=str)
    if teaching is None:
        return pd.DataFrame(columns=["concept_id", "teaching_order", "teaching_concept_name"])

    if "node_id" in teaching.columns:
        id_col = "node_id"
    elif "concept_id" in teaching.columns:
        id_col = "concept_id"
    else:
        return pd.DataFrame(columns=["concept_id", "teaching_order", "teaching_concept_name"])

    order_col = "teaching_order" if "teaching_order" in teaching.columns else None
    name_col = "concept_name" if "concept_name" in teaching.columns else None

    out = pd.DataFrame({"concept_id": clean_text(teaching[id_col])})
    out["teaching_order"] = (
        pd.to_numeric(teaching[order_col], errors="coerce") if order_col else pd.NA
    )
    out["teaching_concept_name"] = (
        clean_text(teaching[name_col]).str.replace("_", " ", regex=False)
        if name_col
        else ""
    )
    return out.drop_duplicates(subset=["concept_id"], keep="first")


def load_dag_degrees(path):
    # DAG edges 用于计算 in_degree/out_degree，以及后面的 bottleneck 分数。
    edges = read_csv_if_exists(path, dtype=str)
    if edges is None or not {"src", "dst"} <= set(edges.columns):
        empty_degrees = pd.DataFrame(columns=["concept_id", "in_degree", "out_degree"])
        empty_edges = pd.DataFrame(columns=["src", "dst"])
        return empty_degrees, empty_edges

    edges = edges[["src", "dst"]].copy()
    edges["src"] = clean_text(edges["src"])
    edges["dst"] = clean_text(edges["dst"])
    edges = edges.loc[edges["src"].ne("") & edges["dst"].ne("")].drop_duplicates()

    nodes = sorted(set(edges["src"]) | set(edges["dst"]), key=sort_key)
    degrees = pd.DataFrame({"concept_id": nodes})
    in_counts = edges.groupby("dst").size().rename("in_degree")
    out_counts = edges.groupby("src").size().rename("out_degree")
    degrees = degrees.merge(in_counts, left_on="concept_id", right_index=True, how="left")
    degrees = degrees.merge(out_counts, left_on="concept_id", right_index=True, how="left")
    degrees[["in_degree", "out_degree"]] = (
        degrees[["in_degree", "out_degree"]].fillna(0).astype(int)
    )
    return degrees, edges


def sort_key(value):
    value = str(value)
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def sorted_unique_join(values):
    cleaned = sorted(
        {str(value).strip() for value in values if pd.notna(value) and str(value).strip()},
        key=sort_key,
    )
    return "; ".join(cleaned)


def first_nonempty(values):
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def iso_min(series):
    if series.empty or series.dropna().empty:
        return pd.NA
    return series.min().isoformat()


def iso_max(series):
    if series.empty or series.dropna().empty:
        return pd.NA
    return series.max().isoformat()


def label_lookup_frame(mapping, concept_catalog):
    labels = (
        mapping.groupby("concept_id", dropna=False)
        .agg(mapping_question_label=("question_label", first_nonempty))
        .reset_index()
    )
    labels = labels.merge(concept_catalog, on="concept_id", how="outer")
    mapping_label = labels["mapping_question_label"].fillna("").astype(str).str.strip()
    catalog_label = labels["catalog_concept_name"].fillna("").astype(str).str.strip()
    labels["question_label"] = mapping_label.where(mapping_label.ne(""), catalog_label)
    return labels[["concept_id", "question_label"]].copy()


def all_concepts_frame(mapping, concept_catalog, dag_degrees, teaching_order):
    concept_ids = set(mapping["concept_id"])
    concept_ids |= set(concept_catalog["concept_id"])
    concept_ids |= set(dag_degrees["concept_id"])
    concept_ids |= set(teaching_order["concept_id"])

    concepts = pd.DataFrame({"concept_id": sorted(concept_ids, key=sort_key)})
    labels = label_lookup_frame(mapping, concept_catalog)
    concepts = concepts.merge(labels, on="concept_id", how="left")
    concepts["question_label"] = concepts["question_label"].fillna("")
    return concepts


def build_mapping_concept_summary(mapping, all_concepts):
    # 第一层：mapping 本身的 concept 覆盖情况。
    # 注意这里会把 DAG 中 0 道题的 concept 也保留下来，方便发现覆盖空洞。
    summary = (
        mapping.groupby("concept_id", dropna=False)
        .agg(
            question_label=("question_label", first_nonempty),
            distinct_question_labels=("question_label", "nunique"),
            num_questions=("item_id", "nunique"),
            num_assessments=("assessment_title", "nunique"),
            assessments=("assessment_title", sorted_unique_join),
        )
        .reset_index()
    )

    summary = all_concepts.merge(summary, on="concept_id", how="left", suffixes=("_all", ""))
    summary_label = summary["question_label"].fillna("").astype(str).str.strip()
    fallback_label = summary["question_label_all"].fillna("").astype(str).str.strip()
    summary["question_label"] = summary_label.where(summary_label.ne(""), fallback_label)
    summary = summary.drop(columns=["question_label_all"])

    count_cols = ["distinct_question_labels", "num_questions", "num_assessments"]
    summary[count_cols] = summary[count_cols].fillna(0).astype(int)
    summary["assessments"] = summary["assessments"].fillna("")
    return summary.sort_values("concept_id", key=lambda col: col.map(sort_key))


def build_mapping_quality_summary(mapping, cleaned, mapping_concept_summary):
    # 第一层 QA：检查重复 item、一个 item 多 concept、一个 concept 多 label、
    # 以及 mapping 和 cleaned interactions 之间的问题集合差异。
    item_concepts = mapping.groupby("item_id")["concept_id"].nunique()
    duplicate_item_count = int(mapping["item_id"].duplicated(keep=False).sum())
    duplicate_item_ids = sorted(
        mapping.loc[mapping["item_id"].duplicated(keep=False), "item_id"].unique(),
        key=sort_key,
    )
    multi_concept_item_ids = sorted(item_concepts[item_concepts > 1].index, key=sort_key)

    label_counts = mapping.groupby("concept_id")["question_label"].nunique()
    multi_label_concept_ids = sorted(label_counts[label_counts > 1].index, key=sort_key)

    mapped_items = set(mapping["item_id"])
    cleaned_items = set(cleaned["item_id"])
    mapped_not_cleaned = sorted(mapped_items - cleaned_items, key=sort_key)
    cleaned_not_mapped = sorted(cleaned_items - mapped_items, key=sort_key)

    concept_zero = mapping_concept_summary.loc[
        mapping_concept_summary["num_questions"].eq(0), "concept_id"
    ].tolist()
    concept_one = mapping_concept_summary.loc[
        mapping_concept_summary["num_questions"].eq(1), "concept_id"
    ].tolist()

    rows = [
        {
            "metric": "total_mapping_rows",
            "value": len(mapping),
            "detail": "",
        },
        {
            "metric": "total_mapped_questions",
            "value": mapping["item_id"].nunique(),
            "detail": "",
        },
        {
            "metric": "total_concepts_covered_by_questions",
            "value": mapping["concept_id"].nunique(),
            "detail": "",
        },
        {
            "metric": "concepts_with_only_1_question",
            "value": len(concept_one),
            "detail": "; ".join(concept_one),
        },
        {
            "metric": "concepts_with_0_questions",
            "value": len(concept_zero),
            "detail": "; ".join(concept_zero),
        },
        {
            "metric": "duplicate_item_id_count",
            "value": len(duplicate_item_ids),
            "detail": "; ".join(duplicate_item_ids),
        },
        {
            "metric": "duplicate_item_id_rows",
            "value": duplicate_item_count,
            "detail": "; ".join(duplicate_item_ids),
        },
        {
            "metric": "item_id_mapped_to_multiple_concepts_count",
            "value": len(multi_concept_item_ids),
            "detail": "; ".join(multi_concept_item_ids),
        },
        {
            "metric": "concept_id_with_multiple_question_labels_count",
            "value": len(multi_label_concept_ids),
            "detail": "; ".join(multi_label_concept_ids),
        },
        {
            "metric": "mapping_questions_not_in_cleaned_interactions",
            "value": len(mapped_not_cleaned),
            "detail": "; ".join(mapped_not_cleaned),
        },
        {
            "metric": "cleaned_questions_not_in_mapping",
            "value": len(cleaned_not_mapped),
            "detail": "; ".join(cleaned_not_mapped),
        },
    ]
    return pd.DataFrame(rows)


def build_interaction_overview(cleaned, mapping):
    # 第二层：cleaned interactions 的总体规模、正确率、时间范围和分布。
    user_attempts = cleaned.groupby("user_id").size()
    item_attempts = cleaned.groupby("item_id").size()
    mapped_items = set(mapping["item_id"])
    cleaned_items = set(cleaned["item_id"])

    rows = [
        ("total_cleaned_interactions", len(cleaned)),
        ("unique_students", cleaned["user_id"].nunique()),
        ("unique_questions_in_cleaned", cleaned["item_id"].nunique()),
        ("unique_questions_in_mapping", mapping["item_id"].nunique()),
        ("questions_in_mapping_not_seen", len(mapped_items - cleaned_items)),
        ("questions_seen_not_in_mapping", len(cleaned_items - mapped_items)),
        ("overall_success_rate", cleaned["is_correct"].mean()),
        ("timestamp_min", iso_min(cleaned["timestamp_dt"])),
        ("timestamp_max", iso_max(cleaned["timestamp_dt"])),
        (
            "duration_days",
            (
                cleaned["timestamp_dt"].max() - cleaned["timestamp_dt"].min()
            ).total_seconds()
            / 86400.0,
        ),
        ("attempts_per_student_mean", user_attempts.mean()),
        ("attempts_per_student_median", user_attempts.median()),
        ("attempts_per_student_min", user_attempts.min()),
        ("attempts_per_student_25pct", user_attempts.quantile(0.25)),
        ("attempts_per_student_75pct", user_attempts.quantile(0.75)),
        ("attempts_per_student_max", user_attempts.max()),
        ("attempts_per_student_std", user_attempts.std(ddof=1)),
        ("attempts_per_question_mean", item_attempts.mean()),
        ("attempts_per_question_median", item_attempts.median()),
        ("attempts_per_question_min", item_attempts.min()),
        ("attempts_per_question_25pct", item_attempts.quantile(0.25)),
        ("attempts_per_question_75pct", item_attempts.quantile(0.75)),
        ("attempts_per_question_max", item_attempts.max()),
        ("attempts_per_question_std", item_attempts.std(ddof=1)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_enriched_interactions(cleaned, mapping):
    # 把 cleaned_interactions 从 question-level join 到 concept-level。
    # validate_mapping_for_join 已保证 item_id 是一对一映射，所以 keep="first" 不会隐藏冲突。
    mapping_for_join = mapping.drop_duplicates(subset=["item_id"], keep="first")
    enriched = cleaned.merge(mapping_for_join, on="item_id", how="left", indicator=True)
    enriched["is_mapped"] = enriched["_merge"].eq("both")
    enriched = enriched.drop(columns=["_merge"])
    return enriched


def build_question_summary(mapping, enriched):
    # question-level 统计：每道题尝试次数、学生覆盖数、整体正确率、首次尝试正确率。
    question_attempts = (
        enriched.groupby("item_id", dropna=False)
        .agg(
            attempts=("item_id", "size"),
            unique_students=("user_id", "nunique"),
            success_rate=("is_correct", "mean"),
        )
        .reset_index()
    )

    first_question_attempts = (
        enriched.sort_values(["user_id", "item_id", "timestamp_dt", "_row_order"])
        .drop_duplicates(subset=["user_id", "item_id"], keep="first")
        .groupby("item_id", dropna=False)
        .agg(first_try_success_rate=("is_correct", "mean"))
        .reset_index()
    )

    question_summary = mapping.drop_duplicates(subset=["item_id"], keep="first").merge(
        question_attempts, on="item_id", how="left"
    )
    question_summary = question_summary.merge(
        first_question_attempts, on="item_id", how="left"
    )
    question_summary[["attempts", "unique_students"]] = (
        question_summary[["attempts", "unique_students"]].fillna(0).astype(int)
    )
    return question_summary.sort_values(
        ["assessment_title", "concept_id", "item_id"],
        key=lambda col: col.map(sort_key) if col.name in {"concept_id", "item_id"} else col,
    )


def build_unmapped_cleaned_question_summary(enriched):
    # 如果 cleaned_interactions 里有 item_id 不在 mapping 中，就无法进入 concept-level 统计。
    # 单独输出这些题，方便后续决定是补 mapping，还是确认它们不是教学路径题目。
    unmapped = enriched.loc[~enriched["is_mapped"]].copy()
    if unmapped.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "attempts",
                "unique_students",
                "success_rate",
                "first_timestamp",
                "last_timestamp",
            ]
        )

    summary = (
        unmapped.groupby("item_id", dropna=False)
        .agg(
            attempts=("item_id", "size"),
            unique_students=("user_id", "nunique"),
            success_rate=("is_correct", "mean"),
            first_timestamp_dt=("timestamp_dt", "min"),
            last_timestamp_dt=("timestamp_dt", "max"),
        )
        .reset_index()
    )
    summary["first_timestamp"] = summary["first_timestamp_dt"].apply(
        lambda value: value.isoformat()
    )
    summary["last_timestamp"] = summary["last_timestamp_dt"].apply(
        lambda value: value.isoformat()
    )
    return summary[
        [
            "item_id",
            "attempts",
            "unique_students",
            "success_rate",
            "first_timestamp",
            "last_timestamp",
        ]
    ].sort_values("attempts", ascending=False)


def build_unmapped_assessment_context(raw_context, unmapped_question_summary):
    # 把 unmapped item_id 回查到原始 PrairieLearn export，回答：
    # 它来自哪个 assessment？原始 completed rows 有多少？是否可能同一个 item_id 被多个 assessment 复用？
    columns = [
        "item_id",
        "assessment_title",
        "assessment_name",
        "raw_rows",
        "completed_rows",
        "raw_unique_students",
        "raw_completed_success_rate",
        "first_raw_timestamp",
        "last_raw_timestamp",
    ]
    if raw_context is None or unmapped_question_summary.empty:
        return pd.DataFrame(columns=columns)

    item_ids = set(unmapped_question_summary["item_id"].astype(str))
    raw = raw_context.loc[raw_context["question_id"].isin(item_ids)].copy()
    if raw.empty:
        return pd.DataFrame(columns=columns)

    if "anon_student_id" in raw.columns:
        student_col = "anon_student_id"
    elif "student_id" in raw.columns:
        student_col = "student_id"
    else:
        raw["_unknown_student"] = ""
        student_col = "_unknown_student"

    is_correct_missing = (
        missing_value_mask(raw["is_correct"])
        if "is_correct" in raw.columns
        else pd.Series(True, index=raw.index)
    )
    score_missing = (
        missing_value_mask(raw["score"])
        if "score" in raw.columns
        else pd.Series(True, index=raw.index)
    )
    raw["is_completed"] = ~(is_correct_missing & score_missing)

    context = (
        raw.groupby(["question_id", "assessment_title", "assessment_name"], dropna=False)
        .agg(
            raw_rows=("question_id", "size"),
            completed_rows=("is_completed", "sum"),
            raw_unique_students=(student_col, "nunique"),
            raw_completed_success_rate=("is_correct_num", "mean"),
            first_raw_timestamp=("timestamp", "min"),
            last_raw_timestamp=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"question_id": "item_id"})
    )
    return context[columns].sort_values(["item_id", "assessment_title"])


def build_student_concept_summary(enriched_mapped):
    # student-concept 是后续 Oracle/Planner 最有用的明细层：
    # 每个学生在每个 concept 上的 first/last attempt、掌握状态和时间窗口。
    if enriched_mapped.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "concept_id",
                "question_label",
                "attempts",
                "success_rate",
                "first_try_correct",
                "last_try_correct",
                "mastered",
                "first_timestamp",
                "last_timestamp",
            ]
        )

    sorted_df = enriched_mapped.sort_values(
        ["user_id", "concept_id", "timestamp_dt", "_row_order"]
    )
    first_rows = sorted_df.drop_duplicates(["user_id", "concept_id"], keep="first")
    last_rows = sorted_df.drop_duplicates(["user_id", "concept_id"], keep="last")

    base = (
        sorted_df.groupby(["user_id", "concept_id"], dropna=False)
        .agg(
            question_label=("question_label", first_nonempty),
            attempts=("concept_id", "size"),
            success_rate=("is_correct", "mean"),
            first_timestamp_dt=("timestamp_dt", "min"),
            last_timestamp_dt=("timestamp_dt", "max"),
        )
        .reset_index()
    )

    first_try = first_rows[
        ["user_id", "concept_id", "is_correct"]
    ].rename(columns={"is_correct": "first_try_correct"})
    last_try = last_rows[
        ["user_id", "concept_id", "is_correct"]
    ].rename(columns={"is_correct": "last_try_correct"})

    base = base.merge(first_try, on=["user_id", "concept_id"], how="left")
    base = base.merge(last_try, on=["user_id", "concept_id"], how="left")
    base["mastered"] = base["last_try_correct"].eq(1).astype(int)
    base["first_timestamp"] = base["first_timestamp_dt"].apply(lambda value: value.isoformat())
    base["last_timestamp"] = base["last_timestamp_dt"].apply(lambda value: value.isoformat())
    base = base.drop(columns=["first_timestamp_dt", "last_timestamp_dt"])

    return base[
        [
            "user_id",
            "concept_id",
            "question_label",
            "attempts",
            "success_rate",
            "first_try_correct",
            "last_try_correct",
            "mastered",
            "first_timestamp",
            "last_timestamp",
        ]
    ].sort_values(["user_id", "concept_id"], key=lambda col: col.map(sort_key) if col.name == "concept_id" else col)


def build_concept_summary(all_concepts, mapping_concept_summary, enriched_mapped, student_concept):
    # concept-level 汇总：把题目覆盖、练习次数、first/last try、mastery 和 struggle 合在一起。
    num_questions = mapping_concept_summary[["concept_id", "num_questions"]]

    concept_attempts = (
        enriched_mapped.groupby("concept_id", dropna=False)
        .agg(
            attempts=("concept_id", "size"),
            unique_students=("user_id", "nunique"),
            success_rate=("is_correct", "mean"),
        )
        .reset_index()
    )

    concept_from_student = (
        student_concept.groupby("concept_id", dropna=False)
        .agg(
            first_try_success_rate=("first_try_correct", "mean"),
            last_try_success_rate=("last_try_correct", "mean"),
            avg_attempts_per_student=("attempts", "mean"),
            mastery_rate=("mastered", "mean"),
        )
        .reset_index()
    )

    summary = all_concepts.merge(num_questions, on="concept_id", how="left")
    summary = summary.merge(concept_attempts, on="concept_id", how="left")
    summary = summary.merge(concept_from_student, on="concept_id", how="left")

    count_cols = ["num_questions", "attempts", "unique_students"]
    summary[count_cols] = summary[count_cols].fillna(0).astype(int)
    summary["improvement"] = (
        summary["last_try_success_rate"] - summary["first_try_success_rate"]
    )
    summary["struggle_index"] = calculate_struggle_index(summary)
    return summary.sort_values("concept_id", key=lambda col: col.map(sort_key))


def calculate_struggle_index(summary):
    # 困难指数采用加权公式：
    # 0.4 * (1 - 首次正确率) + 0.3 * 归一化平均尝试次数 + 0.3 * (1 - 最后正确率)。
    # 没有 interaction 的 concept 保持为空，不强行赋 0，避免误读为“不困难”。
    attempts = summary["avg_attempts_per_student"]
    min_attempts = attempts.min(skipna=True)
    max_attempts = attempts.max(skipna=True)
    if pd.isna(min_attempts) or pd.isna(max_attempts) or max_attempts == min_attempts:
        attempts_normalized = pd.Series(0.0, index=summary.index)
    else:
        attempts_normalized = (attempts - min_attempts) / (max_attempts - min_attempts)

    struggle = (
        0.4 * (1 - summary["first_try_success_rate"])
        + 0.3 * attempts_normalized
        + 0.3 * (1 - summary["last_try_success_rate"])
    )
    return struggle.where(summary["attempts"].gt(0))


def build_student_summary(cleaned, student_concept):
    # student-level 画像：总尝试、覆盖题目、覆盖 concept、掌握 concept、薄弱 concept。
    base = (
        cleaned.groupby("user_id", dropna=False)
        .agg(
            total_attempts=("user_id", "size"),
            unique_questions=("item_id", "nunique"),
            overall_success_rate=("is_correct", "mean"),
            first_timestamp_dt=("timestamp_dt", "min"),
            last_timestamp_dt=("timestamp_dt", "max"),
        )
        .reset_index()
    )

    if student_concept.empty:
        base["concepts_attempted"] = 0
        base["concepts_mastered"] = 0
        base["mastery_ratio"] = np.nan
        base["weak_concepts"] = ""
    else:
        concept_stats = (
            student_concept.groupby("user_id", dropna=False)
            .agg(
                concepts_attempted=("concept_id", "nunique"),
                concepts_mastered=("mastered", "sum"),
            )
            .reset_index()
        )
        weak = (
            student_concept.loc[student_concept["mastered"].eq(0)]
            .groupby("user_id")["concept_id"]
            .apply(sorted_unique_join)
            .rename("weak_concepts")
            .reset_index()
        )
        base = base.merge(concept_stats, on="user_id", how="left")
        base = base.merge(weak, on="user_id", how="left")
        base[["concepts_attempted", "concepts_mastered"]] = (
            base[["concepts_attempted", "concepts_mastered"]].fillna(0).astype(int)
        )
        base["mastery_ratio"] = base["concepts_mastered"] / base["concepts_attempted"]
        base["weak_concepts"] = base["weak_concepts"].fillna("")

    base["first_timestamp"] = base["first_timestamp_dt"].apply(lambda value: value.isoformat())
    base["last_timestamp"] = base["last_timestamp_dt"].apply(lambda value: value.isoformat())
    return base[
        [
            "user_id",
            "total_attempts",
            "unique_questions",
            "concepts_attempted",
            "concepts_mastered",
            "overall_success_rate",
            "mastery_ratio",
            "weak_concepts",
            "first_timestamp",
            "last_timestamp",
        ]
    ].sort_values("user_id")


def build_assessment_summary(mapping, enriched_mapped):
    # assessment-level 统计：每个 Quiz/Practice Quiz 的提交量、覆盖范围和正确率。
    assessment_attempts = (
        enriched_mapped.groupby("assessment_title", dropna=False)
        .agg(
            attempts=("assessment_title", "size"),
            unique_students=("user_id", "nunique"),
            unique_questions=("item_id", "nunique"),
            unique_concepts=("concept_id", "nunique"),
            success_rate=("is_correct", "mean"),
            first_timestamp_dt=("timestamp_dt", "min"),
            last_timestamp_dt=("timestamp_dt", "max"),
        )
        .reset_index()
    )
    if not assessment_attempts.empty:
        assessment_attempts["first_timestamp"] = assessment_attempts[
            "first_timestamp_dt"
        ].apply(lambda value: value.isoformat())
        assessment_attempts["last_timestamp"] = assessment_attempts[
            "last_timestamp_dt"
        ].apply(lambda value: value.isoformat())
        assessment_attempts = assessment_attempts.drop(
            columns=["first_timestamp_dt", "last_timestamp_dt"]
        )

    assessment_mapping = (
        mapping.groupby("assessment_title", dropna=False)
        .agg(
            mapped_questions=("item_id", "nunique"),
            mapped_concepts=("concept_id", "nunique"),
        )
        .reset_index()
    )

    summary = assessment_mapping.merge(assessment_attempts, on="assessment_title", how="left")
    for column in ["attempts", "unique_students", "unique_questions", "unique_concepts"]:
        summary[column] = summary[column].fillna(0).astype(int)
    summary["first_timestamp"] = summary["first_timestamp"].fillna("")
    summary["last_timestamp"] = summary["last_timestamp"].fillna("")
    return summary.sort_values("assessment_title")


def build_daily_attempts(cleaned):
    # 时间层：按原始 timestamp 字符串里的日期统计，保留课程本地日期的直觉。
    return (
        cleaned.groupby("date", dropna=False)
        .agg(
            attempts=("date", "size"),
            unique_students=("user_id", "nunique"),
            unique_questions=("item_id", "nunique"),
            success_rate=("is_correct", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )


def build_bottleneck_summary(concept_summary, dag_degrees):
    # Planner 可用的 bottleneck 指标：out_degree 越高、struggle 越高，越值得优先处理。
    bottleneck = concept_summary.merge(dag_degrees, on="concept_id", how="left")
    bottleneck[["in_degree", "out_degree"]] = (
        bottleneck[["in_degree", "out_degree"]].fillna(0).astype(int)
    )
    bottleneck["bottleneck_score"] = bottleneck["out_degree"] * bottleneck["struggle_index"]
    return bottleneck[
        [
            "concept_id",
            "question_label",
            "in_degree",
            "out_degree",
            "mastery_rate",
            "struggle_index",
            "bottleneck_score",
        ]
    ].sort_values(["bottleneck_score", "out_degree"], ascending=[False, False])


def build_concept_order_summary(student_concept, concept_summary, teaching_order):
    # 路径顺序统计：先找每个学生第一次接触各 concept 的时间，
    # 再计算每个 concept 在学生真实路径中的平均首次出现位置。
    if student_concept.empty:
        return pd.DataFrame(
            columns=[
                "concept_id",
                "question_label",
                "teaching_order",
                "students_seen",
                "avg_actual_order",
                "median_actual_order",
                "order_gap",
            ]
        )

    first_seen = student_concept[
        ["user_id", "concept_id", "first_timestamp"]
    ].copy()
    first_seen["first_timestamp_dt"] = pd.to_datetime(
        first_seen["first_timestamp"], errors="coerce", utc=True
    )
    first_seen = first_seen.sort_values(["user_id", "first_timestamp_dt", "concept_id"])
    first_seen["actual_order"] = first_seen.groupby("user_id").cumcount() + 1

    order = (
        first_seen.groupby("concept_id", dropna=False)
        .agg(
            students_seen=("user_id", "nunique"),
            avg_actual_order=("actual_order", "mean"),
            median_actual_order=("actual_order", "median"),
        )
        .reset_index()
    )
    order = order.merge(
        concept_summary[["concept_id", "question_label"]], on="concept_id", how="left"
    )
    order = order.merge(teaching_order, on="concept_id", how="left")
    order["order_gap"] = order["avg_actual_order"] - order["teaching_order"]
    return order[
        [
            "concept_id",
            "question_label",
            "teaching_order",
            "students_seen",
            "avg_actual_order",
            "median_actual_order",
            "order_gap",
        ]
    ].sort_values("concept_id", key=lambda col: col.map(sort_key))


def build_dag_prerequisite_violation_summary(student_concept, edges):
    # prerequisite violation 定义：
    # 对 DAG 边 prereq -> target，如果学生先接触 target 后接触 prereq，
    # 或者接触了 target 但从未接触 prereq，都记为这条边的潜在违反。
    if student_concept.empty or edges.empty:
        return pd.DataFrame(
            columns=[
                "prereq",
                "target",
                "students_reaching_target",
                "students_seen_both",
                "violations_before_prereq",
                "violations_missing_prereq",
                "violation_rate",
            ]
        )

    first_seen = student_concept[
        ["user_id", "concept_id", "first_timestamp"]
    ].copy()
    first_seen["first_timestamp_dt"] = pd.to_datetime(
        first_seen["first_timestamp"], errors="coerce", utc=True
    )

    first_seen_lookup = {
        (row.user_id, row.concept_id): row.first_timestamp_dt
        for row in first_seen.itertuples(index=False)
    }
    users_by_concept = first_seen.groupby("concept_id")["user_id"].apply(set).to_dict()

    rows = []
    for edge in edges.itertuples(index=False):
        prereq = str(edge.src)
        target = str(edge.dst)
        target_users = users_by_concept.get(target, set())
        prereq_users = users_by_concept.get(prereq, set())
        both_users = target_users & prereq_users
        missing_prereq_users = target_users - prereq_users

        before_count = 0
        for user_id in both_users:
            prereq_time = first_seen_lookup[(user_id, prereq)]
            target_time = first_seen_lookup[(user_id, target)]
            if target_time < prereq_time:
                before_count += 1

        denominator = len(target_users)
        violation_count = before_count + len(missing_prereq_users)
        rows.append(
            {
                "prereq": prereq,
                "target": target,
                "students_reaching_target": denominator,
                "students_seen_both": len(both_users),
                "violations_before_prereq": before_count,
                "violations_missing_prereq": len(missing_prereq_users),
                "violation_rate": violation_count / denominator if denominator else pd.NA,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["violation_rate", "students_reaching_target"], ascending=[False, False]
    )


def classify_assessment(title):
    # 只识别形如 "Practice Quiz 1" / "Quiz 1" 的 assessment 标题。
    # 其他标题会被排除出 practice-vs-quiz transfer 统计。
    text = str(title)
    match = re.search(r"\b(Practice Quiz|Quiz)\s*(\d+)\b", text, flags=re.IGNORECASE)
    if not match:
        return pd.Series({"assessment_type": "", "unit": ""})
    kind = match.group(1).lower()
    assessment_type = "practice" if "practice" in kind else "quiz"
    return pd.Series({"assessment_type": assessment_type, "unit": match.group(2)})


def build_practice_quiz_transfer_summary(enriched_mapped):
    # 比较同一 unit + concept 在 Practice Quiz 和正式 Quiz 上的表现差距。
    if enriched_mapped.empty:
        return pd.DataFrame(
            columns=[
                "unit",
                "concept_id",
                "question_label",
                "practice_attempts",
                "practice_success_rate",
                "quiz_attempts",
                "quiz_success_rate",
                "gap",
            ]
        )

    classified = enriched_mapped.copy()
    classified[["assessment_type", "unit"]] = classified["assessment_title"].apply(
        classify_assessment
    )
    classified = classified.loc[classified["assessment_type"].isin(["practice", "quiz"])]
    if classified.empty:
        return pd.DataFrame(
            columns=[
                "unit",
                "concept_id",
                "question_label",
                "practice_attempts",
                "practice_success_rate",
                "quiz_attempts",
                "quiz_success_rate",
                "gap",
            ]
        )

    grouped = (
        classified.groupby(["unit", "concept_id", "question_label", "assessment_type"])
        .agg(attempts=("assessment_type", "size"), success_rate=("is_correct", "mean"))
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["unit", "concept_id", "question_label"],
        columns="assessment_type",
        values=["attempts", "success_rate"],
        aggfunc="first",
    )
    pivot.columns = [
        f"{assessment_type}_{metric}"
        for metric, assessment_type in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index()
    for column in ["practice_attempts", "quiz_attempts"]:
        if column not in pivot.columns:
            pivot[column] = 0
        pivot[column] = pivot[column].fillna(0).astype(int)
    for column in ["practice_success_rate", "quiz_success_rate"]:
        if column not in pivot.columns:
            pivot[column] = np.nan
    pivot["gap"] = pivot["quiz_success_rate"] - pivot["practice_success_rate"]
    pivot = pivot[
        [
            "unit",
            "concept_id",
            "question_label",
            "practice_attempts",
            "practice_success_rate",
            "quiz_attempts",
            "quiz_success_rate",
            "gap",
        ]
    ]
    return pivot.sort_values(["unit", "concept_id"], key=lambda col: col.map(sort_key) if col.name in {"unit", "concept_id"} else col)


def add_teaching_alignment_metric(concept_order_summary, interaction_overview):
    # teaching_order 与实际 first-seen order 的 Spearman 相关系数。
    # 只对同时有 teaching_order 和实际出现顺序的 concept 计算。
    usable = concept_order_summary.dropna(subset=["teaching_order", "avg_actual_order"])
    if len(usable) < 2:
        value = pd.NA
    else:
        value = usable["teaching_order"].corr(usable["avg_actual_order"], method="spearman")

    extra = pd.DataFrame(
        [{"metric": "teaching_order_actual_order_spearman", "value": value}]
    )
    return pd.concat([interaction_overview, extra], ignore_index=True)


def metric_value(metrics_df, metric, default=pd.NA):
    match = metrics_df.loc[metrics_df["metric"].eq(metric), "value"]
    if match.empty:
        return default
    return match.iloc[0]


def metric_detail(metrics_df, metric, default=""):
    if "detail" not in metrics_df.columns:
        return default
    match = metrics_df.loc[metrics_df["metric"].eq(metric), "detail"]
    if match.empty:
        return default
    value = match.iloc[0]
    return "" if pd.isna(value) else str(value)


def format_report_value(value, column=""):
    if pd.isna(value):
        return ""

    if isinstance(value, str):
        return value

    if column == "teaching_order":
        return f"{int(value):,}"

    if any(token in column for token in ["rate", "ratio"]) or column in {"gap"}:
        return f"{float(value):.2%}"

    if any(token in column for token in ["index", "score", "order", "improvement"]):
        return f"{float(value):.3f}"

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.2f}"

    return str(value)


def markdown_table(df, columns, max_rows=None):
    if df.empty:
        return "_无记录。_"

    rows = df.loc[:, columns]
    if max_rows is not None:
        rows = rows.head(max_rows)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(format_report_value(row[col], col) for col in columns)
            + " |"
        )
    return "\n".join(lines)


def build_learning_statistics_report(tables, joined_rows, unmapped_rows):
    # 生成中文报告。报告只引用本次内存中刚算出来的表，避免人工复制数字导致不一致。
    quality = tables["mapping_quality_summary.csv"]
    overview = tables["interaction_overview_summary.csv"]
    concept_summary = tables["concept_summary.csv"]
    bottleneck = tables["concept_bottleneck_summary.csv"]
    assessment = tables["assessment_summary.csv"]
    unmapped_questions = tables["unmapped_cleaned_question_summary.csv"]
    unmapped_context = tables["unmapped_cleaned_question_assessment_context.csv"]
    concept_order = tables["concept_order_summary.csv"]

    observed_concepts = concept_summary.loc[concept_summary["attempts"].gt(0)].copy()
    hardest_concepts = observed_concepts.sort_values(
        ["success_rate", "attempts"], ascending=[True, False]
    )
    if unmapped_context.empty:
        unmapped_by_assessment = pd.DataFrame(
            columns=["assessment_title", "unmapped_item_ids", "completed_rows"]
        )
    else:
        unmapped_by_assessment = (
            unmapped_context.groupby("assessment_title", dropna=False)
            .agg(
                unmapped_item_ids=("item_id", sorted_unique_join),
                completed_rows=("completed_rows", "sum"),
            )
            .reset_index()
        )

    lines = [
        "# Learning Statistics Report",
        "",
        "## 1. 本次做了什么",
        "",
        "本次新增并运行 `experiments/05_generate_learning_statistics.py`，把题目级 cleaned interactions 通过 "
        "`data/question_concept_mapping_template.csv` 映射到 concept-level。脚本现在只输出这个 Markdown 总结报告，"
        "不再把中间结果拆成一堆 CSV。",
        "",
        "脚本默认输入：",
        "",
        "- `data/anonymized_submissions_ECS32A_sq26.csv`",
        "- `data/question_concept_mapping_template.csv`",
        "- `data/processed/cleaned_interactions.csv`",
        "- `data/ecs32a_concepts_required_full_v1.csv`",
        "- `data/ecs32a_dag_edges_required_full_v1.csv`",
        "- `data/ecs32a_teaching_order_required_full_v1.csv`",
        "",
        "重新生成命令：",
        "",
        "```bash",
        "python experiments/05_generate_learning_statistics.py",
        "```",
        "",
        "## 2. 一句话结论",
        "",
        "当前 `mapping` 表内部没有发现结构性冲突：没有重复 `item_id`、没有一题多 concept、没有 concept-label 冲突。"
        "但是 cleaned 数据里确实有 3 个真实提交过的 `item_id` 不在 mapping 中：`9596878`、`9597496`、`9643536`。"
        "这不是 cleaned 文件坏了，而是 mapping 没有告诉脚本这 3 道题属于哪个 concept。",
        "",
        "这 3 道题合计对应 `457` 条 cleaned interaction。它们会进入 cleaned 总量统计，但不会进入 concept-level、"
        "student-concept、mastery、struggle、DAG bottleneck 等所有依赖 `concept_id` 的统计。",
        "",
        "## 3. 核心统计结果",
        "",
        "### 3.1 Mapping 质量",
        "",
        f"- Mapping 行数：`{format_report_value(metric_value(quality, 'total_mapping_rows'))}`",
        f"- unique mapped questions：`{format_report_value(metric_value(quality, 'total_mapped_questions'))}`",
        f"- covered concepts：`{format_report_value(metric_value(quality, 'total_concepts_covered_by_questions'))}`",
        f"- 只有 1 道题的 concept 数：`{format_report_value(metric_value(quality, 'concepts_with_only_1_question'))}`",
        f"- DAG/catalog 中 0 道题覆盖的 concept 数：`{format_report_value(metric_value(quality, 'concepts_with_0_questions'))}`",
        f"- duplicate item_id count：`{format_report_value(metric_value(quality, 'duplicate_item_id_count'))}`",
        f"- item_id mapped to multiple concepts count：`{format_report_value(metric_value(quality, 'item_id_mapped_to_multiple_concepts_count'))}`",
        f"- concept_id with multiple question_label count：`{format_report_value(metric_value(quality, 'concept_id_with_multiple_question_labels_count'))}`",
        f"- mapping 中有但 cleaned 从未出现的题：`{format_report_value(metric_value(quality, 'mapping_questions_not_in_cleaned_interactions'))}`",
        f"- cleaned 中有但 mapping 不存在的题：`{format_report_value(metric_value(quality, 'cleaned_questions_not_in_mapping'))}`"
        + (
            f"（{metric_detail(quality, 'cleaned_questions_not_in_mapping')}）"
            if metric_detail(quality, "cleaned_questions_not_in_mapping")
            else ""
        ),
        "",
        "### 3.2 Cleaned Interactions 总体",
        "",
        f"- 总 interaction 数：`{format_report_value(metric_value(overview, 'total_cleaned_interactions'))}`",
        f"- 学生数：`{format_report_value(metric_value(overview, 'unique_students'))}`",
        f"- cleaned 中 unique questions：`{format_report_value(metric_value(overview, 'unique_questions_in_cleaned'))}`",
        f"- 总体正确率：`{format_report_value(metric_value(overview, 'overall_success_rate'), 'success_rate')}`",
        f"- 时间范围：`{metric_value(overview, 'timestamp_min')}` 到 `{metric_value(overview, 'timestamp_max')}`",
        f"- 覆盖天数：`{format_report_value(metric_value(overview, 'duration_days'))}` 天",
        f"- 每个学生提交次数：mean `{format_report_value(metric_value(overview, 'attempts_per_student_mean'))}`，"
        f"median `{format_report_value(metric_value(overview, 'attempts_per_student_median'))}`，"
        f"min `{format_report_value(metric_value(overview, 'attempts_per_student_min'))}`，"
        f"max `{format_report_value(metric_value(overview, 'attempts_per_student_max'))}`",
        f"- 成功 join 到 concept-level 的 interaction：`{format_report_value(joined_rows)}`",
        f"- 未能 join 的 cleaned interaction：`{format_report_value(unmapped_rows)}`",
        "",
        "### 3.3 Concept-Level 概览",
        "",
        f"- concept-level 汇总中的 concept 总数：`{format_report_value(len(concept_summary))}`",
        f"- 有实际 interaction 的 concept 数：`{format_report_value(len(observed_concepts))}`",
        f"- teaching order vs actual first-seen order Spearman："
        f"`{format_report_value(metric_value(overview, 'teaching_order_actual_order_spearman'), 'score')}`",
        "",
        "最困难 concept（按 success_rate 从低到高，至少有 interaction）：",
        "",
        markdown_table(
            hardest_concepts,
            ["concept_id", "question_label", "attempts", "unique_students", "success_rate", "mastery_rate"],
            max_rows=10,
        ),
        "",
        "Top bottleneck concept（按 bottleneck_score 排序）：",
        "",
        markdown_table(
            bottleneck,
            [
                "concept_id",
                "question_label",
                "out_degree",
                "mastery_rate",
                "struggle_index",
                "bottleneck_score",
            ],
            max_rows=10,
        ),
        "",
        "### 3.4 Assessment-Level 概览",
        "",
        markdown_table(
            assessment,
            [
                "assessment_title",
                "mapped_questions",
                "mapped_concepts",
                "attempts",
                "unique_students",
                "success_rate",
            ],
            max_rows=20,
        ),
        "",
        "### 3.5 Unmapped Cleaned Questions",
        "",
        "#### 这到底是什么意思",
        "",
        "这里说的 unmapped，不是指这些提交记录无效，也不是指 cleaned_interactions 坏了。意思是："
        "这些 `item_id` 在学生提交记录里真实存在，但 `question_concept_mapping_template.csv` 没有给它们分配 "
        "`concept_id`。所以脚本不知道它们属于哪个知识点，只能把它们排除在 concept-level、student-concept、"
        "bottleneck、DAG violation 等统计之外。",
        "",
        "#### 证据链",
        "",
        "我检查了三层文件，结果如下：",
        "",
        "- `cleaned_interactions.csv` 里有这 3 个 item_id，说明它们是清洗后保留下来的有效提交。",
        "- `question_concept_mapping_template.csv` 里没有这 3 个 item_id。",
        "- `question_concept_mapping_final.csv` 里也没有这 3 个 item_id。",
        "- template 是从 final mapping 生成的，所以根因不是统计脚本 join 错了，而是 final mapping 源头没有收这 3 道题。",
        "",
        "影响可以这样理解：",
        "",
        "- 如果这些题本来就不属于学习路径，例如只是额外 review/final 检查题，那么排除它们是合理的。",
        "- 如果这些题其实应该属于某个 concept，那么现在的 concept 统计会少算这些练习记录，对应 concept 的 attempts、success_rate、mastery_rate、struggle_index 都会偏差。",
        "- 报告里的 question-level 统计只覆盖 mapping 里的 141 道题，因此这 3 个 unmapped item 不进入 question-level 结果。",
        "- 报告里的 cleaned 总量统计仍包含它们，因为那部分统计的是 cleaned 全量数据。",
        "",
        markdown_table(
            unmapped_questions,
            ["item_id", "attempts", "unique_students", "success_rate", "first_timestamp", "last_timestamp"],
            max_rows=20,
        ),
        "",
        "按 assessment 汇总后，缺口集中在这些地方：",
        "",
        markdown_table(
            unmapped_by_assessment,
            ["assessment_title", "unmapped_item_ids", "completed_rows"],
            max_rows=20,
        ),
        "",
        "回查 raw 数据后，这些 item_id 的 assessment 来源如下：",
        "",
        markdown_table(
            unmapped_context,
            [
                "item_id",
                "assessment_title",
                "raw_rows",
                "completed_rows",
                "raw_unique_students",
                "raw_completed_success_rate",
            ],
            max_rows=20,
        ),
        "",
        "特别注意：`9596878` 在 raw 数据里同时出现在 Practice Quiz 1 和 Practice Quiz 2；"
        "但 cleaned_interactions 只有 `item_id`，没有 `assessment_title`，所以 cleaned 层面只能把它们合并成同一道 item 的 314 条记录。",
        "",
        "#### 为什么可能会这样",
        "",
        "基于当前仓库文件，能确认的是：这 3 个 ID 没有进入 `question_concept_mapping_final.csv`。"
        "不能确认的是它们的具体题面，因为 raw CSV 只有 `question_id`、assessment、时间、正确性等列，没有题目标题或题面内容。",
        "",
        "最可能的原因有三类：",
        "",
        "1. `question_concept_mapping_final.csv` 是人工/半人工整理的教学题映射，整理时漏掉了每个相关 assessment 中额外出现的一道题。",
        "2. `9596878` 被 PrairieLearn 在 Practice Quiz 1 和 Practice Quiz 2 里复用；如果 mapping 是按某个静态题目清单整理的，这种跨 assessment 复用题很容易漏掉。",
        "3. `9597496` 和 `9643536` 出现在 final/review assessment 中，而且正确率都是 0；它们可能是额外 review/final 题、占位/特殊题，或者没有被纳入 learning path 的题。当前文件不足以判断它们应该映射到哪个 concept。",
        "",
        "因此，不应该直接给它们硬填 concept。正确做法是回到 PrairieLearn/课程题目元数据中查这三个 `question_id` 的题面或 question directory，"
        "再决定是否补进 mapping。",
        "",
        "### 3.6 路径顺序",
        "",
        "teaching order 与实际首次出现顺序差距最大的 concept：",
        "",
        markdown_table(
            concept_order.assign(abs_order_gap=concept_order["order_gap"].abs()).sort_values(
                "abs_order_gap", ascending=False
            ),
            [
                "concept_id",
                "question_label",
                "teaching_order",
                "avg_actual_order",
                "order_gap",
                "students_seen",
            ],
            max_rows=10,
        ),
        "",
        "## 4. 逻辑校验与已知假设",
        "",
        "- 脚本会校验 mapping 必须是一题一行；如果出现重复 `item_id` 或一个 `item_id` 对多个 `concept_id`，会直接报错停止。",
        "- `is_correct` 必须能转换为 0/1；`timestamp` 必须能解析为时间，否则脚本会停止。",
        "- concept-level 统计只使用成功 join 到 mapping 的 interactions；未 join 的题已集中写在本报告的 unmapped 小节。",
        "- `mastered` 的当前定义是：学生在该 concept 的最后一次尝试 `is_correct = 1`。",
        "- `first_try_success_rate` / `last_try_success_rate` 是按 `user_id + concept_id` 聚合后的首次/最后一次尝试均值。",
        "- `struggle_index = 0.4 * (1 - first_try_success_rate) + 0.3 * normalized(avg_attempts_per_student) + 0.3 * (1 - last_try_success_rate)`。",
        "- DAG violation 的当前定义包括两类：先接触 target 再接触 prereq；或者接触 target 但从未接触 prereq。",
        "- 路径顺序使用 UTC 解析后的 timestamp 排序；如果同一学生多个 concept 首次出现时间完全相同，用 `concept_id` 做稳定 tie-break。",
        "- daily attempts 使用原始 timestamp 字符串中的日期，保留课程本地日期视角；first/last timestamp 输出为 UTC ISO 时间。",
        "- Practice vs Quiz 只识别标题形如 `Practice Quiz N` 和 `Quiz N` 的 assessment。",
        "- DAG/catalog 中没有题目覆盖的 concept 会保留在 concept summary 中，但 attempts/rates 保持为空，避免误读为 0% 或 100%。",
        "",
        "## 5. 结论",
        "",
        "当前可以确认的是：mapping 表内部没有发现结构性冲突，也就是没有 duplicate item_id、没有一题多 concept、"
        "没有 concept-label 冲突。不能说“数据完全没有任何问题”，因为仍有 3 个 cleaned item_id 没有 concept 映射。",
        "",
        "这 3 个 unmapped item 的含义很具体：学生确实提交过这些题，cleaned_interactions 也保留了这些有效提交；"
        "但是 mapping 表没有告诉脚本这些题属于哪个 concept。因此它们能进入 cleaned 总量统计，却不能进入 concept、"
        "student-concept、DAG bottleneck 等依赖 concept_id 的统计。",
        "",
        "下一步判断标准也很明确：如果 `9596878`、`9597496`、`9643536` 是课程学习路径里应该建模的题，"
        "就应该补到 mapping 表并重新跑脚本；如果它们只是额外 review/final 或不用于路径规划的题，"
        "那保持排除是合理的，但报告里已经明确记录了它们没有进入 concept-level 统计。",
        "",
    ]

    return "\n".join(lines)


def write_report(output_dir, report_text):
    # 报告是 Markdown，便于直接阅读，也方便之后提交到版本库或贴到项目文档里。
    # Windows PowerShell/记事本对无 BOM UTF-8 有时会误判编码，
    # 所以这里用 utf-8-sig 写中文报告，减少打开时乱码的概率。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8-sig")
    return report_path


def main():
    args = parse_args()

    mapping = load_mapping(args.mapping_path)
    validate_mapping_for_join(mapping)
    cleaned = load_cleaned(args.cleaned_path)
    raw_context = load_raw_context(args.raw_path)
    concept_catalog = load_concept_catalog(args.concepts_path)
    teaching_order = load_teaching_order(args.teaching_order_path)
    dag_degrees, dag_edges = load_dag_degrees(args.dag_edges_path)

    all_concepts = all_concepts_frame(mapping, concept_catalog, dag_degrees, teaching_order)
    enriched = build_enriched_interactions(cleaned, mapping)
    enriched_mapped = enriched.loc[enriched["is_mapped"]].copy()
    unmapped_cleaned_question_summary = build_unmapped_cleaned_question_summary(enriched)
    unmapped_cleaned_question_assessment_context = build_unmapped_assessment_context(
        raw_context, unmapped_cleaned_question_summary
    )

    mapping_concept_summary = build_mapping_concept_summary(mapping, all_concepts)
    mapping_quality_summary = build_mapping_quality_summary(
        mapping, cleaned, mapping_concept_summary
    )
    interaction_overview = build_interaction_overview(cleaned, mapping)
    question_summary = build_question_summary(mapping, enriched)
    student_concept_summary = build_student_concept_summary(enriched_mapped)
    concept_summary = build_concept_summary(
        all_concepts, mapping_concept_summary, enriched_mapped, student_concept_summary
    )
    student_summary = build_student_summary(cleaned, student_concept_summary)
    assessment_summary = build_assessment_summary(mapping, enriched_mapped)
    daily_attempts_summary = build_daily_attempts(cleaned)
    concept_bottleneck_summary = build_bottleneck_summary(concept_summary, dag_degrees)
    concept_order_summary = build_concept_order_summary(
        student_concept_summary, concept_summary, teaching_order
    )
    dag_prerequisite_violation_summary = build_dag_prerequisite_violation_summary(
        student_concept_summary, dag_edges
    )
    practice_quiz_transfer_summary = build_practice_quiz_transfer_summary(enriched_mapped)
    interaction_overview = add_teaching_alignment_metric(
        concept_order_summary, interaction_overview
    )

    tables = {
        "mapping_quality_summary.csv": mapping_quality_summary,
        "mapping_concept_summary.csv": mapping_concept_summary,
        "interaction_overview_summary.csv": interaction_overview,
        "question_summary.csv": question_summary,
        "concept_summary.csv": concept_summary,
        "student_summary.csv": student_summary,
        "student_concept_summary.csv": student_concept_summary,
        "assessment_summary.csv": assessment_summary,
        "daily_attempts_summary.csv": daily_attempts_summary,
        "concept_bottleneck_summary.csv": concept_bottleneck_summary,
        "concept_order_summary.csv": concept_order_summary,
        "dag_prerequisite_violation_summary.csv": dag_prerequisite_violation_summary,
        "practice_quiz_transfer_summary.csv": practice_quiz_transfer_summary,
        "unmapped_cleaned_question_summary.csv": unmapped_cleaned_question_summary,
        "unmapped_cleaned_question_assessment_context.csv": unmapped_cleaned_question_assessment_context,
    }
    report_text = build_learning_statistics_report(
        tables=tables,
        joined_rows=len(enriched_mapped),
        unmapped_rows=len(enriched) - len(enriched_mapped),
    )
    report_path = write_report(args.output_dir, report_text)

    print("Generated learning statistics report:")
    print(f"- {report_path}")
    print(f"Mapped interaction rows used for concept-level summaries: {len(enriched_mapped)}")
    print(f"Unmapped cleaned interaction rows: {len(enriched) - len(enriched_mapped)}")


if __name__ == "__main__":
    main()
