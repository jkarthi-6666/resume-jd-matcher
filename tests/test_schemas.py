from src.schemas import Requirement, RequirementPlan


def test_requirement_plan_accepts_unknown_category():
    """An ambiguous model category must not abort an otherwise valid plan."""
    plan = RequirementPlan.model_validate({
        "requirements": [{
            "id": "R1",
            "requirement": "Five years of relevant experience",
            "category": "unknown",
            "importance": "required",
        }]
    })

    assert plan.requirements == [
        Requirement(
            id="R1",
            requirement="Five years of relevant experience",
            category="unknown",
            importance="required",
        )
    ]
