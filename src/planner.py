"""Normalize a JD and extract candidate requirements from it."""
import hashlib
import pathlib
import re
import unicodedata
import numpy as np

from src.schemas import Requirement, RequirementPlan
from src import llm, config, embeddings
from src.validate import is_source_substring

_PROMPT = (pathlib.Path(__file__).parent.parent / "prompts" / "planner.txt").read_text()
_CACHE: dict[str, RequirementPlan] = {}
_CACHE_VERSION = "source-span-semantic-query-v1"

_BULLETS = "•●▪◦‣⁃"
_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"


def normalize_job_description(job_description: str) -> str:
    """Return stable, readable text without changing the JD's meaning.

    Job descriptions are commonly pasted from PDFs, messaging apps, and job
    portals.  This normalizes their presentation noise while deliberately
    leaving semantic decisions (requirements versus logistics, AND/OR, and
    importance) to the planner prompt.
    """
    text = unicodedata.normalize("NFKC", job_description)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = text.translate({ord(char): None for char in _ZERO_WIDTH})

    # Put common Unicode bullets on their own lines. This also repairs text
    # copied from sites that collapses multiple visual list items into one line.
    text = re.sub(rf"[ \t]*([{re.escape(_BULLETS)}])[ \t]*", r"\n- ", text)

    lines: list[str] = []
    previous_blank = True
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False

    return "\n".join(lines).strip()


def _finalize_requirements(
    requirements: list[Requirement],
    normalized_jd: str,
) -> tuple[list[Requirement], int]:
    """Validate source spans, deduplicate, and assign stable IDs."""
    finalized: list[Requirement] = []
    dropped = 0
    seen: dict[str, int] = {}
    for requirement in requirements:
        if not is_source_substring(requirement.source_span, normalized_jd):
            dropped += 1
            continue
        text = " ".join(requirement.requirement.split()).strip()
        key = text.casefold()
        if not text:
            continue
        if key in seen:
            index = seen[key]
            finalized[index] = _merge_requirement(finalized[index], requirement)
            continue
        seen[key] = len(finalized)
        finalized.append(requirement.model_copy(update={
            "id": f"R{len(finalized) + 1}",
            "requirement": text,
        }))
    finalized = _semantic_deduplicate(finalized)
    finalized = [
        requirement.model_copy(update={"id": f"R{index + 1}"})
        for index, requirement in enumerate(finalized)
    ]
    return finalized, dropped


def _merge_requirement(existing: Requirement, duplicate: Requirement) -> Requirement:
    importance_rank = {"unknown": 0, "preferred": 1, "required": 2}
    updates = {}
    if importance_rank[duplicate.importance] > importance_rank[existing.importance]:
        updates["importance"] = duplicate.importance
    if existing.category == "unknown" and duplicate.category != "unknown":
        updates["category"] = duplicate.category
    if existing.kind == "scored" and duplicate.kind == "gate":
        updates["kind"] = "gate"
    merged_terms = list(dict.fromkeys([*existing.query_terms, *duplicate.query_terms]))[:5]
    if merged_terms != existing.query_terms:
        updates["query_terms"] = merged_terms
    return existing.model_copy(update=updates) if updates else existing


def _semantic_deduplicate(requirements: list[Requirement]) -> list[Requirement]:
    """Merge paraphrased requirements above the configured cosine threshold."""
    if len(requirements) < 2:
        return requirements
    vectors = np.asarray(
        embeddings.get_embeddings(
            [requirement.requirement for requirement in requirements],
            input_type="query",
        ),
        dtype="float32",
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(requirements) or not vectors.shape[1]:
        raise ValueError("Embedding provider returned invalid requirement embeddings.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)

    kept: list[Requirement] = []
    kept_indices: list[int] = []
    for index, requirement in enumerate(requirements):
        duplicate_index = next(
            (
                position
                for position, source_index in enumerate(kept_indices)
                if float(normalized[index] @ normalized[source_index])
                >= config.SEMANTIC_DEDUP_THRESHOLD
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(requirement)
            kept_indices.append(index)
        else:
            kept[duplicate_index] = _merge_requirement(
                kept[duplicate_index], requirement
            )
    return kept


def plan(job_description: str, model: str | None = None) -> RequirementPlan:
    """Extract requirements from a job description. Results are cached by normalized JD."""
    normalized_jd = normalize_job_description(job_description)
    if not normalized_jd:
        raise ValueError("Job description is empty.")

    cache_input = f"{_CACHE_VERSION}\0{normalized_jd}"
    jd_hash = hashlib.sha256(cache_input.encode()).hexdigest()
    if jd_hash in _CACHE:
        return _CACHE[jd_hash]

    model = model or config.PLANNER_MODEL
    prompt = _PROMPT.format(job_description=normalized_jd)
    try:
        result: RequirementPlan = llm.call(
            prompt, model, RequirementPlan, temperature=0.0
        )
    except ValueError:
        # JSON-mode providers do not all enforce the supplied schema. One retry
        # handles occasional malformed enum values or JSON without weakening the
        # validated contract used by the rest of the pipeline.
        result = llm.call(
            prompt,
            model,
            RequirementPlan,
            temperature=0.0,
            system=(
                "You are a strict job-requirement extraction engine. Follow the "
                "allowed enum values and output schema exactly. Return JSON only."
            ),
        )

    requirements, dropped = _finalize_requirements(
        result.requirements, normalized_jd
    )
    if not requirements:
        raise ValueError("Planner returned no requirements. Cannot proceed to scoring.")

    plan_result = RequirementPlan(
        requirements=requirements,
        dropped_requirement_count=dropped,
    )
    _CACHE[jd_hash] = plan_result
    return plan_result


def clear_cache() -> None:
    _CACHE.clear()
