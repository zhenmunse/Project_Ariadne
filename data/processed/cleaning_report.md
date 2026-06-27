# Data Cleaning Report

## What Was Done

The raw PrairieLearn export at `data/anonymized_submissions_ECS32A_sq26.csv`
was cleaned with `experiments/00_clean_raw_data.py` and written to
`data/processed/cleaned_interactions.csv`.

The output schema is exactly:

```text
user_id,item_id,is_correct,timestamp
```

Column mapping:

```text
anon_student_id / student_id -> user_id
question_id                  -> item_id
is_correct                   -> is_correct
timestamp                    -> timestamp
```

Cleaning operations applied:

1. Standardized column names by stripping surrounding spaces.
2. Used `anon_student_id` as `student_id` because the real export uses that
   column name.
3. Normalized student IDs as strings. Pure numeric IDs would have leading zeros
   removed; anonymized IDs such as `student_0030df3636f5` are preserved.
4. Dropped unfinished submissions where both `is_correct` and `score` were
   missing. These are likely PrairieLearn Save/autosave/page-close records, not
   completed attempts.
5. Logged dropped-row counts by student, question, and assessment title.
6. Recomputed `attempt_number` by `(student_id, question_id)` instead of
   trusting the PrairieLearn-exported value, because variant changes can reset
   the exported attempt number.
7. Deduplicated by `(student_id, question_id, timestamp)`.
8. Renamed columns to the format expected by `src/data_engine/preprocessor.py`.
9. Validated the cleaned output for duplicate rows, nulls, invalid timestamps,
   and invalid `is_correct` values.

## Validation Results

Raw rows:

```text
83739
```

Rows dropped because both `is_correct` and `score` were missing:

```text
12389
```

Drop rate:

```text
14.79%
```

Cleaned rows:

```text
71350
```

Final checks:

```text
columns: user_id,item_id,is_correct,timestamp
null values: 0
empty required fields: 0
duplicate user_id/item_id/timestamp rows: 0
bad timestamps: 0
is_correct values: 0 and 1 only
unique users: 296
unique items: 144
timestamp range: 2026-04-02T13:10:47-07:00 -> 2026-06-11T12:51:19-07:00
```

Content-level validation also passed: the cleaned CSV exactly matches the raw
export after applying the documented cleaning transformation.

## High Missing-Submission Assessments

Because the missing-submission drop rate is above 10%, the script reports the
top assessments contributing dropped rows. The largest contributors were:

```text
Practice Quiz 2    1196
Quiz 4             1099
Quiz 6             1070
Practice Quiz 5    1021
Practice Quiz 6     999
Quiz 5              974
Quiz 2              922
Review Quiz 1       906
Practice Quiz 1     852
Quiz 3              837
```

These rows are likely Save/autosave/page-close records, but the high rate is
worth mentioning when discussing PrairieLearn export behavior.

## Additional Fix

`src/data_engine/preprocessor.py` was updated so it no longer forces `user_id`
to `int`. Real PrairieLearn anonymized IDs are strings such as
`student_0030df3636f5`, and the preprocessor only needs `user_id` for grouping.

## Known Unresolved Issue

True orphan filtering is not fully possible with the current repository files.
The configured file `data/ecs32a_concepts_required_full_v1.csv` contains concept
metadata:

```text
node_id,concept_name,teaching_order,source_block,week_introduced,bloom_level
```

It does not contain `question_id` or `item_id`. Therefore, the script currently
prints a warning and temporarily treats all raw `question_id` values as valid so
that Week 1 question-level cleaning can complete.

To fully support concept-level sessions for the Oracle and Planner, the project
still needs a real mapping file such as:

```text
item_id,node_id
```

or:

```text
question_id,node_id
```

Without that mapping, `cleaned_interactions.csv` is valid question-level data,
but the final question-to-concept conversion is not scientifically complete.
