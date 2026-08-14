from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from property_content.contracts import GenerationResult
from property_content.datasets import PropertyFixture
from property_content.generation import (
    DeterministicListingGenerator,
    ListingGenerator,
    LLMListingGenerator,
    render_listing,
)
from property_content.ingestion import FakeDescriptionFactExtractor, PropertyFactBuilder


def _facts(fixture: PropertyFixture):
    return asyncio.run(
        PropertyFactBuilder(
            FakeDescriptionFactExtractor(fixture.expected_description_facts)
        ).build(fixture.property)
    )


def test_deterministic_generator_returns_all_four_fields(rich_fixture: PropertyFixture) -> None:
    facts = _facts(rich_fixture)
    result = asyncio.run(DeterministicListingGenerator().generate(facts))

    assert result.content is not None
    assert set(result.content.model_dump()) == {
        "hero_headline",
        "property_highlights",
        "about_this_place",
        "amenities_descriptions",
    }
    assert result.content.amenities_descriptions


def test_llm_generator_uses_dependency_injection_and_strict_parsing(
    rich_fixture: PropertyFixture,
) -> None:
    facts = _facts(rich_fixture)
    deterministic = asyncio.run(DeterministicListingGenerator().generate(facts))
    model = Mock()
    model.complete = AsyncMock(return_value=deterministic.raw_completion)
    generator = LLMListingGenerator(model)

    result = asyncio.run(generator.generate(facts))

    assert isinstance(generator, ListingGenerator)
    assert result.content == deterministic.content
    prompt = model.complete.await_args.args[0]
    assert "all four top-level fields" in prompt
    assert "You may use any subset of facts" in prompt
    assert "20-word minimum" in prompt
    assert "property-type noun" in prompt
    assert "capacity or bedroom facts do not support the type noun" in prompt
    assert model.complete.await_args.kwargs["response_schema"]


def test_llm_generator_preserves_parse_error(rich_fixture: PropertyFixture) -> None:
    facts = _facts(rich_fixture)
    model = Mock()
    model.complete = AsyncMock(return_value="```json\n{}\n```")

    result = asyncio.run(LLMListingGenerator(model).generate(facts))

    assert isinstance(result, GenerationResult)
    assert result.content is None
    assert result.parse_error


def test_renderer_removes_fact_ids_without_changing_copy(
    rich_fixture: PropertyFixture,
) -> None:
    facts = _facts(rich_fixture)
    result = asyncio.run(DeterministicListingGenerator().generate(facts))
    assert result.content is not None

    rendered = render_listing(result.content)

    assert rendered.hero_headline == result.content.hero_headline.text
    assert rendered.about_this_place == " ".join(
        statement.text for statement in result.content.about_this_place
    )
    assert "fact_ids" not in rendered.model_dump_json()
