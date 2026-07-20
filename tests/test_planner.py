from unittest.mock import patch

from src.planner import clear_cache, normalize_job_description, plan
from src.schemas import Requirement, RequirementPlan


def test_normalize_job_description_repairs_common_copy_paste_noise():
    raw = (
        "\ufeffPosition:\xa0Gen AI Engineer•Location:\tChennai\r\n"
        "\r\n• Experience:  5–7 Years\u200b"
    )

    assert normalize_job_description(raw) == (
        "Position: Gen AI Engineer\n"
        "- Location: Chennai\n\n"
        "- Experience: 5–7 Years"
    )


def test_plan_normalizes_input_deduplicates_and_reassigns_ids():
    model_result = RequirementPlan(requirements=[
        Requirement(
            id="skill-7",
            requirement="  Python   programming ",
            category="unknown",
            importance="unknown",
            source_span="Python programming",
        ),
        Requirement(
            id="skill-7",
            requirement="python programming",
            category="technical_skill",
            importance="required",
            source_span="Python programming",
        ),
        Requirement(
            id="anything",
            requirement="Vector databases (Pinecone, Chroma, FAISS)",
            category="technical_skill",
            importance="required",
            source_span="Vector databases",
        ),
    ])
    clear_cache()

    with patch("src.planner.llm.call", return_value=model_result) as mock_call, \
         patch(
             "src.planner.embeddings.get_embeddings",
             return_value=[[1.0, 0.0], [0.0, 1.0]],
         ):
        plan_result = plan("Key Skills:• Python programming• Vector databases")
        requirements = plan_result.requirements

    sent_prompt = mock_call.call_args.args[0]
    assert "Key Skills:\n- Python programming\n- Vector databases" in sent_prompt
    assert [r.id for r in requirements] == ["R1", "R2"]
    assert [r.requirement for r in requirements] == [
        "Python programming",
        "Vector databases (Pinecone, Chroma, FAISS)",
    ]
    assert requirements[0].category == "technical_skill"
    assert requirements[0].importance == "required"


def test_plan_retries_one_schema_failure():
    valid = RequirementPlan(requirements=[
        Requirement(
            id="R1",
            requirement="Python programming",
            category="technical_skill",
            importance="required",
            source_span="Python programming",
        )
    ])
    clear_cache()

    with patch("src.planner.llm.call", side_effect=[ValueError("bad JSON"), valid]) as mock_call:
        requirements = plan("Python programming is required").requirements

    assert requirements[0].requirement == "Python programming"
    assert mock_call.call_count == 2
    assert "strict job-requirement extraction engine" in mock_call.call_args.kwargs["system"]


def test_plan_retries_when_every_requirement_omits_source_span():
    spanless = RequirementPlan(requirements=[
        Requirement(
            id="R1", requirement="Python programming",
            category="technical_skill", importance="required",
        )
    ])
    grounded = RequirementPlan(requirements=[
        Requirement(
            id="R1", requirement="Python programming",
            category="technical_skill", importance="required",
            source_span="Python programming",
        )
    ])
    clear_cache()

    with patch(
        "src.planner.llm.call", side_effect=[spanless, grounded]
    ) as mock_call:
        result = plan("Python programming is required")

    assert [r.requirement for r in result.requirements] == ["Python programming"]
    assert mock_call.call_count == 2
    assert "non-empty verbatim source_span" in mock_call.call_args.kwargs["system"]


def test_plan_drops_only_requirements_with_invalid_source_spans():
    model_result = RequirementPlan(requirements=[
        Requirement(
            id="R1", requirement="Python programming",
            category="technical_skill", importance="required",
            source_span="Python programming",
        ),
        Requirement(
            id="R2", requirement="AWS experience",
            category="technical_skill", importance="required",
            source_span="AWS experience",
        ),
    ])
    clear_cache()

    with patch("src.planner.llm.call", return_value=model_result) as mock_call:
        result = plan("Python programming is required")

    assert [r.requirement for r in result.requirements] == ["Python programming"]
    assert result.dropped_requirement_count == 1
    mock_call.assert_called_once()


def test_plan_merges_semantic_duplicates_using_embeddings():
    model_result = RequirementPlan(requirements=[
        Requirement(
            id="R1", requirement="5+ years Python",
            category="unknown", importance="preferred",
            source_span="5+ years Python",
        ),
        Requirement(
            id="R2", requirement="At least five years of Python experience",
            category="experience", importance="required",
            source_span="at least five years of Python experience",
        ),
    ])
    clear_cache()
    jd = "5+ years Python; at least five years of Python experience is required."

    with patch("src.planner.llm.call", return_value=model_result), \
         patch(
             "src.planner.embeddings.get_embeddings",
             return_value=[[1.0, 0.0], [0.95, 0.05]],
         ) as mock_embeddings:
        result = plan(jd)

    assert len(result.requirements) == 1
    assert result.requirements[0].importance == "required"
    assert result.requirements[0].category == "experience"
    mock_embeddings.assert_called_once()
