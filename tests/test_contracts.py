from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from property_content.contracts import GenerationResult, ListingContent, PropertyData
from property_content.datasets import PropertyFixture


def test_all_four_property_fixtures_validate(
    property_fixtures: list[PropertyFixture],
) -> None:
    assert len(property_fixtures) == 4
    assert all(
        isinstance(fixture.property, PropertyData) for fixture in property_fixtures
    )
    assert sum(fixture.holdout for fixture in property_fixtures) == 1


def test_listing_content_requires_every_assignment_output() -> None:
    incomplete = {
        "hero_headline": {
            "text": "A grounded city apartment stay",
            "fact_ids": ["property.type"],
        },
        "property_highlights": [
            {"text": "Located in Barcelona.", "fact_ids": ["location.city"]}
        ],
        "about_this_place": [
            {"text": "The property is in Barcelona.", "fact_ids": ["location.city"]}
        ],
    }

    with pytest.raises(ValidationError, match="amenities_descriptions"):
        ListingContent.model_validate(incomplete)


def test_invalid_completion_is_not_silently_repaired() -> None:
    raw = json.dumps({"hero_headline": "not the declared schema"})
    result = GenerationResult.from_raw_completion(raw)

    assert result.content is None
    assert result.parse_error is not None
    assert result.raw_completion == raw
