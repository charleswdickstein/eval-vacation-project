from __future__ import annotations

import asyncio
import json

from property_content.contracts import GroundingVerdict
from property_content.datasets import PropertyFixture
from property_content.extraction_evaluation import (
    FakeDescriptionFactSupportGrader,
    LLMDescriptionFactSupportGrader,
    evaluate_description_fact_support,
)
from property_content.ingestion import FakeDescriptionFactExtractor, PropertyFactBuilder
from property_content.modeling import StaticTextModel


def _facts(fixture: PropertyFixture):
    return asyncio.run(
        PropertyFactBuilder(
            FakeDescriptionFactExtractor(fixture.expected_description_facts)
        ).build(fixture.property)
    )


def test_semantic_description_fact_support_target_is_one_hundred_percent(
    rich_fixture: PropertyFixture,
) -> None:
    facts = _facts(rich_fixture)

    support = asyncio.run(
        evaluate_description_fact_support(facts, FakeDescriptionFactSupportGrader())
    )

    assert support.support_rate == 1.0
    assert support.passed


def test_partial_description_support_fails(rich_fixture: PropertyFixture) -> None:
    facts = _facts(rich_fixture)
    support = asyncio.run(
        evaluate_description_fact_support(
            facts,
            FakeDescriptionFactSupportGrader([GroundingVerdict.PARTIAL]),
        )
    )

    assert support.support_rate < 1.0
    assert not support.passed


def test_llm_verifier_uses_full_description_and_allows_paraphrases(
    rich_fixture: PropertyFixture,
) -> None:
    facts = _facts(rich_fixture)
    fact = next(
        item
        for item in facts.all_facts
        if item.category == "feature.private_roof_terrace"
    )
    model = StaticTextModel(
        [
            json.dumps(
                {
                    "verdict": "SUPPORTED",
                    "unsupported_spans": [],
                    "reason": "The fact is a faithful paraphrase of the description.",
                }
            )
        ]
    )

    decision = asyncio.run(
        LLMDescriptionFactSupportGrader(model).grade(
            fact,
            facts.normalized_description,
        )
    )

    assert decision.verdict is GroundingVerdict.SUPPORTED
    assert "ORIGINAL_DESCRIPTION" in model.prompts[0]
    assert facts.normalized_description in model.prompts[0]
    assert "Exact word or substring overlap is not required" in model.prompts[0]
