from __future__ import annotations

import asyncio
import json

from property_content.contracts import (
    DescriptionFactCandidate,
    FactStatus,
    FactUsage,
)
from property_content.datasets import PropertyFixture
from property_content.ingestion import (
    FakeDescriptionFactExtractor,
    LLMDescriptionFactExtractor,
    PropertyFactBuilder,
    normalize_description,
)
from property_content.modeling import StaticTextModel


def _build(fixture: PropertyFixture):
    extractor = FakeDescriptionFactExtractor(fixture.expected_description_facts)
    return asyncio.run(PropertyFactBuilder(extractor).build(fixture.property))


def test_html_normalization_removes_markup_and_non_visible_script() -> None:
    normalized = normalize_description(
        "<p>A bright <strong>home</strong>.</p><script>claim a pool</script>"
    )
    assert normalized == "A bright home."
    assert "claim a pool" not in normalized


def test_description_facts_keep_available_evidence_and_stable_ids(
    rich_fixture: PropertyFixture,
) -> None:
    first = _build(rich_fixture)
    second = _build(rich_fixture)
    description_facts = [
        fact for fact in first.all_facts if fact.id.startswith("description.fact.")
    ]

    assert description_facts
    assert [fact.id for fact in first.all_facts] == [
        fact.id for fact in second.all_facts
    ]
    assert all(
        fact.source_quote in first.normalized_description for fact in description_facts
    )
    terrace = next(
        fact
        for fact in description_facts
        if fact.category == "feature.private_roof_terrace"
    )
    assert terrace.source_quote == (
        "This sunlit loft has restored timber beams and a private roof terrace."
    )


def test_nonmatching_evidence_hint_does_not_reject_a_candidate(
    rich_fixture: PropertyFixture,
) -> None:
    candidate = DescriptionFactCandidate(
        category="feature.natural_light",
        value="sunlit",
        text="Natural light fills the loft.",
        source_quote="Natural light throughout",
    )
    facts = asyncio.run(
        PropertyFactBuilder(FakeDescriptionFactExtractor([candidate])).build(
            rich_fixture.property
        )
    )
    facts_without_hint = asyncio.run(
        PropertyFactBuilder(
            FakeDescriptionFactExtractor(
                [candidate.model_copy(update={"source_quote": None})]
            )
        ).build(rich_fixture.property)
    )

    extracted = next(
        fact for fact in facts.all_facts if fact.category == "feature.natural_light"
    )
    extracted_without_hint = next(
        fact
        for fact in facts_without_hint.all_facts
        if fact.category == "feature.natural_light"
    )
    assert extracted.status is FactStatus.ACTIVE
    assert extracted.source_quote is None
    assert extracted.id == extracted_without_hint.id
    assert not any(warning.code == "invalid_source_quote" for warning in facts.warnings)


def test_empty_model_evidence_hint_becomes_none() -> None:
    model = StaticTextModel(
        [
            json.dumps(
                {
                    "facts": [
                        {
                            "category": "feature.bright",
                            "value": "bright",
                            "text": "The cottage is bright.",
                            "source_quote": "",
                            "priority": "medium",
                            "suitable_sections": ["about_this_place"],
                        }
                    ]
                }
            )
        ]
    )

    candidates = asyncio.run(LLMDescriptionFactExtractor(model).extract("Bright."))

    assert len(candidates) == 1
    assert candidates[0].source_quote is None


def test_structured_capacity_overrides_conflicting_description(
    property_fixtures: list[PropertyFixture],
) -> None:
    fixture = next(
        item for item in property_fixtures if item.case_id == "conflicting_capacity"
    )
    facts = _build(fixture)
    conflicting = [
        fact
        for fact in facts.all_facts
        if fact.source_path == "description.description"
        and fact.category == "rental.max_guests"
    ]

    assert len(conflicting) == 1
    assert conflicting[0].status is FactStatus.CONFLICTED
    assert conflicting[0].usage is FactUsage.INTERNAL_ONLY
    assert facts.fact_index["rental.max_guests"].value == 6
    assert facts.conflicts[0].authoritative_fact_id == "rental.max_guests"


def test_unknown_amenity_and_images_never_become_claimable(
    property_fixtures: list[PropertyFixture],
) -> None:
    fixture = next(
        item for item in property_fixtures if item.case_id == "adversarial_holdout"
    )
    facts = _build(fixture)

    assert any(warning.code == "unknown_amenity" for warning in facts.warnings)
    assert facts.fact_index["amenity.superfastteleport"].status is FactStatus.EXCLUDED
    assert all(
        fact.usage is FactUsage.INTERNAL_ONLY
        for fact in facts.all_facts
        if fact.category == "image.url"
    )
    assert "private beach" not in facts.normalized_description
