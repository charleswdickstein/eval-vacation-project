from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from property_content.contracts import Fact, PropertyFacts
from property_content.generation import DeterministicListingGenerator
from property_content.inspect_task import property_content_eval


async def _mock_outputs(messages, tools, tool_choice, config):
    prompt = messages[-1].text
    if prompt.startswith("Extract only atomic"):
        content = '{"facts": []}'
    elif prompt.startswith("Write structured vacation-rental"):
        fact_payload = json.loads(prompt.split("FACTS:\n", 1)[1])
        facts = PropertyFacts(
            property_id=0,
            normalized_description="",
            all_facts=[Fact.model_validate(item) for item in fact_payload],
        )
        content = (
            await DeterministicListingGenerator().generate(facts)
        ).raw_completion
    elif prompt.startswith("Determine whether EVERY"):
        content = json.dumps(
            {
                "verdict": "SUPPORTED",
                "unsupported_spans": [],
                "reason": "Mock control: supported.",
            }
        )
    elif prompt.startswith("Evaluate this ONE generated"):
        dimension = {
            "verdict": "PASS",
            "evidence": ["Loft"],
            "reason": "Mock control: editorial dimension passed.",
        }
        content = json.dumps(
            {
                "specificity": dimension,
                "section_differentiation": dimension,
                "clarity_and_concision": dimension,
                "guest_facing_language": dimension,
                "publishability": dimension,
            }
        )
    else:
        raise AssertionError(f"Unexpected mock prompt: {prompt[:100]}")
    return ModelOutput.from_content(model="mockllm/model", content=content)


def test_inspect_solver_and_named_scorer_run_end_to_end(tmp_path: Path) -> None:
    model = get_model("mockllm/model", custom_outputs=_mock_outputs)
    log = inspect_eval(
        property_content_eval(model=model, grader_model=model),
        log_dir=str(tmp_path),
        display="none",
        limit=1,
    )[0]

    assert log.status == "success"
    assert log.samples is not None
    score = log.samples[0].scores["pipeline_evaluation"]
    assert set(score.value) == {
        "description_fact_support_rate",
        "schema_valid",
        "structure_valid",
        "citation_valid",
        "numeric_consistency",
        "conflict_free",
        "non_repetitive",
        "grounded_statement_rate",
        "grounding_passed",
        "editorial_quality_passed",
    }
