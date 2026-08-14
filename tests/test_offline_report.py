from __future__ import annotations

import asyncio

from property_content.offline_report import build_offline_summary


def test_offline_report_exercises_all_fixtures_without_claiming_live_results() -> None:
    summary = asyncio.run(build_offline_summary())

    assert summary["run_type"] == "offline_control"
    assert summary["uses_llm"] is False
    assert summary["aggregate"]["fixture_count"] == 4
    assert summary["aggregate"]["mandatory_pass_count"] == 4
    assert summary["aggregate"]["minimum_grounded_statement_rate"] == 1.0
    assert summary["aggregate"]["minimum_description_fact_support_rate"] == 1.0
    assert all(
        set(sample["output"])
        == {
            "hero_headline",
            "property_highlights",
            "about_this_place",
            "amenities_descriptions",
        }
        for sample in summary["samples"]
    )
