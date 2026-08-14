from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from property_content.datasets import PropertyFixture
from property_content.evaluation import LLMEditorialQualityGrader
from property_content.generation import DeterministicListingGenerator
from property_content.ingestion import FakeDescriptionFactExtractor, PropertyFactBuilder


def _content(fixture: PropertyFixture):
    facts = asyncio.run(
        PropertyFactBuilder(
            FakeDescriptionFactExtractor(fixture.expected_description_facts)
        ).build(fixture.property)
    )
    result = asyncio.run(DeterministicListingGenerator().generate(facts))
    assert result.content is not None
    return result.content


def _assessment(*, failed_dimension: str | None = None) -> str:
    dimensions = {}
    for name in (
        "specificity",
        "section_differentiation",
        "clarity_and_concision",
        "guest_facing_language",
        "publishability",
    ):
        failed = name == failed_dimension
        dimensions[name] = {
            "verdict": "FAIL" if failed else "PASS",
            "evidence": ["Casa Llum"],
            "reason": "Concrete editorial reason.",
        }
    return json.dumps(dimensions)


def test_single_output_editorial_rubric_passes_only_when_every_dimension_passes(
    rich_fixture: PropertyFixture,
) -> None:
    model = Mock()
    model.complete = AsyncMock(return_value=_assessment())

    result = asyncio.run(
        LLMEditorialQualityGrader(model).evaluate(_content(rich_fixture))
    )

    assert result.passed
    assert result.overall_verdict == "PASS"
    prompt = model.complete.await_args.args[0]
    assert "Evaluate this ONE generated" in prompt
    assert "VERSION_A" not in prompt
    assert "baseline" not in prompt.casefold()
    assert model.complete.await_args.kwargs["response_schema"]


def test_one_failed_editorial_dimension_fails_the_overall_result(
    rich_fixture: PropertyFixture,
) -> None:
    model = Mock()
    model.complete = AsyncMock(
        return_value=_assessment(failed_dimension="section_differentiation")
    )

    result = asyncio.run(
        LLMEditorialQualityGrader(model).evaluate(_content(rich_fixture))
    )

    assert not result.passed
    assert result.overall_verdict == "FAIL"
    assert result.section_differentiation.verdict == "FAIL"


def test_editorial_decision_requires_quoted_evidence(
    rich_fixture: PropertyFixture,
) -> None:
    assessment = json.loads(_assessment())
    assessment["specificity"]["evidence"] = []
    model = Mock()
    model.complete = AsyncMock(return_value=json.dumps(assessment))

    with pytest.raises(ValidationError):
        asyncio.run(
            LLMEditorialQualityGrader(model).evaluate(_content(rich_fixture))
        )


def test_evidence_not_found_in_output_fails_the_dimension(
    rich_fixture: PropertyFixture,
) -> None:
    assessment = json.loads(_assessment())
    assessment["publishability"]["evidence"] = ["Text that is not in the listing"]
    model = Mock()
    model.complete = AsyncMock(return_value=json.dumps(assessment))

    result = asyncio.run(
        LLMEditorialQualityGrader(model).evaluate(_content(rich_fixture))
    )

    assert not result.passed
    assert result.publishability.verdict == "FAIL"
    assert "not found verbatim" in result.publishability.reason


def test_editorial_grader_sees_public_amenity_text_without_internal_codes(
    rich_fixture: PropertyFixture,
) -> None:
    assessment = json.loads(_assessment())
    assessment["guest_facing_language"]["evidence"] = [
        "The property includes High-speed internet."
    ]
    model = Mock()
    model.complete = AsyncMock(return_value=json.dumps(assessment))

    result = asyncio.run(
        LLMEditorialQualityGrader(model).evaluate(_content(rich_fixture))
    )

    prompt = model.complete.await_args.args[0]
    assert result.passed
    assert "amenity_code" not in prompt
