from pathlib import Path

from property_content.live_report import (
    _render_evidence,
    _render_listing,
    _render_quality,
    _status_badge,
    build_report,
)


def test_customer_listing_does_not_repeat_amenity_code() -> None:
    content = {
        "hero_headline": {"text": "A grounded property headline"},
        "property_highlights": [{"text": "A grounded highlight."}],
        "about_this_place": [{"text": "Grounded information about the property."}],
        "amenities_descriptions": [
            {"amenity_code": "AirConditioning", "text": "Air conditioning"}
        ],
    }

    listing = _render_listing(content)

    customer_copy = listing.split('<details class="raw-output">', maxsplit=1)[0]
    assert "Air conditioning" in customer_copy
    assert "AirConditioning" not in customer_copy
    assert "AirConditioning" in listing


def test_selected_live_log_renders_reviewer_report() -> None:
    log_path = next(Path("logs/final").glob("*.eval"))

    report = build_report(log_path)

    assert "4/4" in report
    assert report.count("Input · Raw property data") == 4
    assert report.count("Output · Generated marketing copy") == 4
    assert report.count("Show exact input JSON") == 4
    assert report.count("Show exact output JSON") == 4
    assert report.count("Hero headline") >= 4
    assert "&quot;fact_ids&quot;" in report
    for property_report in report.split('<article class="property"')[1:]:
        assert property_report.index("Input · Raw property data") < property_report.index(
            "Output · Generated marketing copy"
        )
    assert "Sleeps 8 guests across three bedrooms" in report
    assert "SuperFastTeleport" in report
    assert "Non-refundable within seven days of arrival" in report
    assert "&lt;script&gt;Claim a private beach.&lt;/script&gt;" in report
    assert report.count("ALL MANDATORY CRITERIA PASS") == 4
    assert report.count("100% GROUNDED") == 4
    assert report.count("Direct editorial assessment of this output") == 4
    assert "Baseline weaknesses" not in report
    assert "Bright Cottage with Fenced Garden in Sóller" in report
    assert "Peaceful Villa in Pollença with Pool and Three Bedrooms" in report
    assert "Sunlit Loft with Private Roof Terrace in Barcelona" in report
    assert "Studio in Cádiz for Two Guests" in report
    assert "sk-ant-" not in report


def test_status_badge_reflects_the_recorded_result() -> None:
    assert 'class="pass"' in _status_badge(True, "PASS", "FAIL")
    assert ">PASS<" in _status_badge(True, "PASS", "FAIL")
    assert 'class="fail-badge"' in _status_badge(False, "PASS", "FAIL")
    assert ">FAIL<" in _status_badge(False, "PASS", "FAIL")


def test_skipped_evaluations_render_as_not_evaluated() -> None:
    content = {
        "hero_headline": {"text": "Grounded headline", "fact_ids": ["fact.1"]},
        "property_highlights": [],
        "about_this_place": [],
        "amenities_descriptions": [],
    }
    fact_index = {
        "fact.1": {
            "id": "fact.1",
            "source": "structured",
            "source_path": "property.name",
            "text": "Grounded headline",
        }
    }
    evidence = _render_evidence(
        content,
        fact_index,
        {"semantic_grounding": "Skipped because a mandatory criterion failed."},
    )
    quality = _render_quality(
        {"editorial_quality": "Skipped because grounding did not pass."}
    )

    assert "NOT EVALUATED" in evidence
    assert "Skipped because a mandatory criterion failed." in evidence
    assert ">SUPPORTED<" not in evidence
    assert "NOT EVALUATED" in quality
    assert "Skipped because grounding did not pass." in quality


def test_direct_editorial_quality_renders_every_dimension() -> None:
    dimension = {
        "verdict": "PASS",
        "evidence": ["Exact listing excerpt."],
        "reason": "Concrete reason.",
    }
    rendered = _render_quality(
        {
            "editorial_quality": {
                "specificity": dimension,
                "section_differentiation": dimension,
                "clarity_and_concision": dimension,
                "guest_facing_language": dimension,
                "publishability": dimension,
                "overall_verdict": "PASS",
            }
        }
    )

    assert "Direct editorial assessment of this output" in rendered
    assert "Specificity" in rendered
    assert "Section differentiation" in rendered
    assert "Clarity and concision" in rendered
    assert "Guest-facing language" in rendered
    assert "Publishability" in rendered
    assert "Baseline" not in rendered
