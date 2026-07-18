"""JD requirement extraction with deterministic caching (keyed by JD hash)."""
import hashlib
import pathlib
from src.schemas import Requirement, RequirementPlan
from src import llm, config

_PROMPT = (pathlib.Path(__file__).parent.parent / "prompts" / "planner.txt").read_text()
_CACHE: dict[str, list[Requirement]] = {}


def plan(job_description: str, model: str | None = None) -> list[Requirement]:
    """Extract requirements from a job description. Results are cached by JD hash."""
    jd_hash = hashlib.sha256(job_description.encode()).hexdigest()
    if jd_hash in _CACHE:
        return _CACHE[jd_hash]

    model = model or config.PLANNER_MODEL
    prompt = _PROMPT.format(job_description=job_description)
    result: RequirementPlan = llm.call(prompt, model, RequirementPlan, temperature=0.0)

    if not result.requirements:
        raise ValueError("Planner returned no requirements. Cannot proceed to scoring.")

    _CACHE[jd_hash] = result.requirements
    return result.requirements


def clear_cache() -> None:
    _CACHE.clear()
