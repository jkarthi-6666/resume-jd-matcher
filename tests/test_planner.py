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
        ),
        Requirement(
            id="skill-7",
            requirement="python programming",
            category="technical_skill",
            importance="required",
        ),
        Requirement(
            id="anything",
            requirement="Vector databases (Pinecone, Chroma, FAISS)",
            category="technical_skill",
            importance="required",
        ),
    ])
    clear_cache()

    with patch("src.planner.llm.call", return_value=model_result) as mock_call:
        requirements = plan("Key Skills:• Python programming• Vector databases")

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
        )
    ])
    clear_cache()

    with patch("src.planner.llm.call", side_effect=[ValueError("bad JSON"), valid]) as mock_call:
        requirements = plan("Python programming is required")

    assert requirements[0].requirement == "Python programming"
    assert mock_call.call_count == 2
    assert "strict job-requirement extraction engine" in mock_call.call_args.kwargs["system"]
