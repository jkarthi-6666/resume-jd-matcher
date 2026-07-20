"""Normalize a JD and extract candidate requirements from it."""
import hashlib
import pathlib
import re
import unicodedata

from src.schemas import Requirement, RequirementPlan
from src import llm, config

_PROMPT = (pathlib.Path(__file__).parent.parent / "prompts" / "planner.txt").read_text()
_CACHE: dict[str, list[Requirement]] = {}

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


def _finalize_requirements(requirements: list[Requirement]) -> list[Requirement]:
    """Deduplicate model output and assign stable, unique requirement IDs."""
    finalized: list[Requirement] = []
    seen: dict[str, int] = {}
    importance_rank = {"unknown": 0, "preferred": 1, "required": 2}
    for requirement in requirements:
        text = " ".join(requirement.requirement.split()).strip()
        key = text.casefold()
        if not text:
            continue
        if key in seen:
            index = seen[key]
            existing = finalized[index]
            updates = {}
            if importance_rank[requirement.importance] > importance_rank[existing.importance]:
                updates["importance"] = requirement.importance
            if existing.category == "unknown" and requirement.category != "unknown":
                updates["category"] = requirement.category
            if updates:
                finalized[index] = existing.model_copy(update=updates)
            continue
        seen[key] = len(finalized)
        finalized.append(requirement.model_copy(update={
            "id": f"R{len(finalized) + 1}",
            "requirement": text,
        }))
    return finalized


def plan(job_description: str, model: str | None = None) -> list[Requirement]:
    """Extract requirements from a job description. Results are cached by normalized JD."""
    normalized_jd = normalize_job_description(job_description)
    if not normalized_jd:
        raise ValueError("Job description is empty.")

    jd_hash = hashlib.sha256(normalized_jd.encode()).hexdigest()
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

    requirements = _finalize_requirements(result.requirements)
    if not requirements:
        raise ValueError("Planner returned no requirements. Cannot proceed to scoring.")

    _CACHE[jd_hash] = requirements
    return requirements


def clear_cache() -> None:
    _CACHE.clear()
