"""Build shared Zero/Full prompts from frozen target bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.llm.artifacts import sha256_file, value_hash


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments" / "llm" / "protocol.json"

SYSTEM_PROMPT = """You are an expert in introductory computer science education and instructional sequencing.

Solve the task using only the information in the current prompt and your pretrained knowledge. You do not have access to web search, external tools, code execution, files, retrieval systems, connectors, or prior conversation history.

Reason internally, then return only the requested JSON object. Do not include explanations, markdown, or additional fields."""

OBJECTIVE = "Choose the prerequisite-valid ordering expected to minimize the student's total number of learning attempts."
OUTPUT_CONTRACT = '{"sequence": ["opaque_id_1", "opaque_id_2", "..."]}'
TEMPLATE_VERSION = "llm-sequence-v1"


def shared_curriculum_block(bundle: dict[str, Any]) -> str:
    concept_lines = [
        f"{opaque}: {bundle['concept_names'][opaque]}"
        for opaque in bundle["concept_order"]
    ]
    edge_lines = [f"{source} -> {target}" for source, target in bundle["edge_order"]]
    return "\n".join([
        f"Target concept:\n{bundle['target_opaque_id']}: {bundle['concept_names'][bundle['target_opaque_id']]}",
        "",
        "Concepts:",
        *concept_lines,
        "",
        "Prerequisite edges:",
        *(edge_lines or ["(none)"]),
    ])


def _statistics_block(bundle: dict[str, Any], statistics: dict[str, Any]) -> str:
    by_node = {int(row["real_node_id"]): row for row in statistics["nodes"]}
    decimals = int(statistics["render_decimal_places"])
    lines = []
    for opaque in bundle["concept_order"]:
        real = int(bundle["opaque_to_real"][opaque])
        row = by_node[real]
        rate = "null" if row["success_rate"] is None else f"{row['success_rate']:.{decimals}f}"
        lines.append(
            f"{opaque}: attempt_count={int(row['attempt_count'])}, success_rate={rate}"
        )
    return "\n".join(lines)


def build_prompt(
    bundle: dict[str, Any],
    condition: str,
    statistics: dict[str, Any],
    *,
    manifest_hash: str,
    statistics_artifact_hash: str,
) -> dict[str, Any]:
    if condition not in {"zero", "full"}:
        raise ValueError("condition must be zero or full")
    shared = shared_curriculum_block(bundle)
    paragraphs = [
        "A new student is starting an introductory computer science curriculum from scratch.",
        "You are given the complete set of concepts in a target curriculum and a directed prerequisite graph. An edge A -> B means that A must be mastered before B.",
    ]
    if condition == "full":
        paragraphs.append(
            "You are also given aggregate training-student statistics. attempt_count is the number of canonical training concept sessions; success_rate is the fraction with session_score >= 0.8. A null rate means no training observations. Treat statistics supported by more observations as more reliable."
        )
    paragraphs.extend([
        "Produce a complete learning sequence containing every provided concept exactly once. Respect every prerequisite edge, use only the supplied opaque IDs, and place the target concept last. Do not add, omit, or repeat concepts.",
        OBJECTIVE,
        f"Return exactly this JSON shape and nothing else:\n{OUTPUT_CONTRACT}",
        shared,
    ])
    if condition == "full":
        paragraphs.extend([
            "Aggregate historical statistics:",
            _statistics_block(bundle, statistics),
        ])
    else:
        paragraphs.append("No historical student-performance data are available.")
    user_prompt = "\n\n".join(paragraphs)
    prompt_hash = value_hash({"system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt})
    return {
        "schema_version": 1,
        "condition": condition,
        "target_node": bundle["target_node"],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt_hash": prompt_hash,
        "system_prompt_hash": value_hash(SYSTEM_PROMPT),
        "template_version": TEMPLATE_VERSION,
        "template_hash": value_hash({
            "version": TEMPLATE_VERSION,
            "objective": OBJECTIVE,
            "output_contract": OUTPUT_CONTRACT,
            "condition": condition,
        }),
        "shared_curriculum_hash": value_hash(shared),
        "mapping_hash": bundle["mapping_hash"],
        "statistics_hash": statistics_artifact_hash if condition == "full" else None,
        "protocol_hash": sha256_file(PROTOCOL_PATH),
        "manifest_hash": manifest_hash,
        "generation_source_hash": sha256_file(Path(__file__)),
    }
