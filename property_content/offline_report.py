"""Generate a network-free control report from fixtures and deterministic fakes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from property_content.datasets import load_property_fixtures
from property_content.evaluation import (
    FakeSemanticGroundingGrader,
    evaluate_listing,
    evaluate_semantic_grounding,
)
from property_content.extraction_evaluation import (
    FakeDescriptionFactSupportGrader,
    evaluate_description_fact_support,
)
from property_content.generation import (
    DeterministicListingGenerator,
    render_listing,
)
from property_content.ingestion import FakeDescriptionFactExtractor, PropertyFactBuilder

PROJECT_ROOT = Path(__file__).parents[1]


async def build_offline_summary() -> dict[str, Any]:
    """Exercise the complete control path without representing it as a live LLM run."""

    fixtures = load_property_fixtures(PROJECT_ROOT / "fixtures" / "properties.json")
    samples: list[dict[str, Any]] = []
    for fixture in fixtures:
        facts = await PropertyFactBuilder(
            FakeDescriptionFactExtractor(fixture.expected_description_facts)
        ).build(fixture.property)
        description_support = await evaluate_description_fact_support(
            facts,
            FakeDescriptionFactSupportGrader(),
        )
        generation = await DeterministicListingGenerator().generate(facts)
        assert generation.content is not None
        deterministic = evaluate_listing(generation, facts)
        semantic = await evaluate_semantic_grounding(
            generation.content,
            facts,
            FakeSemanticGroundingGrader(),
        )
        samples.append(
            {
                "case_id": fixture.case_id,
                "holdout": fixture.holdout,
                "fact_count": len(facts.all_facts),
                "claimable_fact_count": len(facts.claimable_facts),
                "warnings": [
                    warning.model_dump(mode="json") for warning in facts.warnings
                ],
                "conflicts": [
                    conflict.model_dump(mode="json") for conflict in facts.conflicts
                ],
                "output": render_listing(generation.content).model_dump(mode="json"),
                "grounded_output": generation.content.model_dump(mode="json"),
                "checks": {
                    check.name: {
                        "passed": check.passed,
                        "explanation": check.explanation,
                        "details": check.details,
                    }
                    for check in deterministic.checks
                },
                "mandatory_criteria_passed": (
                    description_support.passed
                    and deterministic.passed
                    and semantic.passed
                ),
                "description_fact_support_rate": description_support.support_rate,
                "grounded_statement_rate": semantic.grounded_statement_rate,
            }
        )

    return {
        "run_type": "offline_control",
        "uses_llm": False,
        "description": (
            "Network-free pipeline verification using fixture-authored extraction "
            "candidates, a deterministic listing generator, and a calibrated fake semantic grader."
        ),
        "samples": samples,
        "aggregate": {
            "fixture_count": len(samples),
            "mandatory_pass_count": sum(
                sample["mandatory_criteria_passed"] for sample in samples
            ),
            "minimum_grounded_statement_rate": min(
                sample["grounded_statement_rate"] for sample in samples
            ),
            "minimum_description_fact_support_rate": min(
                sample["description_fact_support_rate"] for sample in samples
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "offline-summary.json",
    )
    args = parser.parse_args()
    summary = asyncio.run(build_offline_summary())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
