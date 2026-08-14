from __future__ import annotations

import asyncio

from property_content.contracts import (
    GenerationResult,
    GroundedStatement,
    GroundingVerdict,
)
from property_content.datasets import PropertyFixture
from property_content.evaluation import (
    FakeSemanticGroundingGrader,
    evaluate_listing,
    evaluate_semantic_grounding,
)
from property_content.generation import DeterministicListingGenerator
from property_content.ingestion import FakeDescriptionFactExtractor, PropertyFactBuilder


def _pipeline(fixture: PropertyFixture):
    facts = asyncio.run(
        PropertyFactBuilder(
            FakeDescriptionFactExtractor(fixture.expected_description_facts)
        ).build(fixture.property)
    )
    result = asyncio.run(DeterministicListingGenerator().generate(facts))
    return facts, result


def test_grounded_deterministic_output_passes_every_mandatory_check(
    rich_fixture: PropertyFixture,
) -> None:
    facts, result = _pipeline(rich_fixture)

    report = evaluate_listing(result, facts)

    assert report.passed, {
        check.name: check.explanation for check in report.checks if not check.passed
    }
    assert "required_fact_coverage" not in {check.name for check in report.checks}


def test_wrong_number_fails_with_statement_details(
    rich_fixture: PropertyFixture,
) -> None:
    facts, result = _pipeline(rich_fixture)
    assert result.content is not None
    corrupted = result.content.model_copy(deep=True)
    corrupted.property_highlights[1] = GroundedStatement(
        text="Space for up to 9 guests across 2 bedrooms.",
        fact_ids=["rental.max_guests", "rental.bedrooms"],
    )

    report = evaluate_listing(
        GenerationResult(raw_completion=corrupted.model_dump_json(), content=corrupted),
        facts,
    )
    check = report.by_name("numeric_consistency")

    assert not check.passed
    assert check.details["failures"][0]["unsupported_numbers"] == ["9"]


def test_valid_citation_does_not_hide_partially_supported_claim(
    rich_fixture: PropertyFixture,
) -> None:
    facts, result = _pipeline(rich_fixture)
    assert result.content is not None
    corrupted = result.content.model_copy(deep=True)
    corrupted.property_highlights[0] = GroundedStatement(
        text="A loft with a sea view in Barcelona.",
        fact_ids=["property.type", "location.city"],
    )
    deterministic = evaluate_listing(
        GenerationResult(raw_completion=corrupted.model_dump_json(), content=corrupted),
        facts,
    )
    assert deterministic.passed

    statement_count = len(corrupted.iter_statements())
    verdicts = [GroundingVerdict.PARTIAL] + [GroundingVerdict.SUPPORTED] * (
        statement_count - 1
    )
    semantic = asyncio.run(
        evaluate_semantic_grounding(
            corrupted, facts, FakeSemanticGroundingGrader(verdicts)
        )
    )

    assert semantic.grounded_statement_rate < 1
    assert not semantic.passed


def test_fully_supported_target_is_exactly_one_hundred_percent(
    rich_fixture: PropertyFixture,
) -> None:
    facts, result = _pipeline(rich_fixture)
    assert result.content is not None

    semantic = asyncio.run(
        evaluate_semantic_grounding(
            result.content, facts, FakeSemanticGroundingGrader()
        )
    )

    assert semantic.grounded_statement_rate == 1.0
    assert semantic.passed
