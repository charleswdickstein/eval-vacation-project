"""Inspect AI task that runs the real extraction, generation, and evaluation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model, ResponseSchema, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from property_content.contracts import (
    FactStatus,
    FactUsage,
    GenerationResult,
    GroundingVerdict,
    ListingContent,
    PropertyData,
    PropertyFacts,
)
from property_content.datasets import load_property_fixtures
from property_content.evaluation import (
    LLMEditorialQualityGrader,
    LLMSemanticGroundingGrader,
    evaluate_listing,
    evaluate_semantic_grounding,
)
from property_content.extraction_evaluation import (
    LLMDescriptionFactSupportGrader,
    evaluate_description_fact_support,
)
from property_content.generation import (
    LLMListingGenerator,
)
from property_content.ingestion import LLMDescriptionFactExtractor, PropertyFactBuilder

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURES_PATH = PROJECT_ROOT / "fixtures" / "properties.json"
GENERATOR_MODEL = "anthropic/claude-sonnet-4-5"
GRADER_MODEL = "anthropic/claude-sonnet-4-5"


def _inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic $refs before Inspect coerces the schema to JSONSchema."""

    definitions = schema.get("$defs", {})

    def inline(node: Any, expanding: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, list):
            return [inline(item, expanding) for item in node]
        if not isinstance(node, dict):
            return node
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            key = reference.removeprefix("#/$defs/")
            if key in definitions and key not in expanding:
                resolved = inline(definitions[key], expanding | {key})
                overrides = {
                    name: inline(value, expanding)
                    for name, value in node.items()
                    if name != "$ref"
                }
                return resolved | overrides
        return {
            name: inline(value, expanding)
            for name, value in node.items()
            if name != "$defs"
        }

    return inline(schema)


class InspectTextModel:
    """Adapt an Inspect Model to the project's injectable model protocol."""

    def __init__(self, model: Model) -> None:
        self._model = model

    async def complete(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        schema = (
            ResponseSchema(
                name="structured_response",
                json_schema=_inline_json_schema_refs(response_schema),
            )
            if response_schema is not None
            else None
        )
        config = GenerateConfig(temperature=0, response_schema=schema)
        return (await self._model.generate(prompt, config=config)).completion


@solver
def build_facts_and_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        property_data = PropertyData.model_validate_json(state.input_text)
        generator_model = InspectTextModel(get_model())
        fact_builder = PropertyFactBuilder(LLMDescriptionFactExtractor(generator_model))
        facts = await fact_builder.build(property_data)

        default_model = get_model()
        verifier_model = InspectTextModel(
            get_model(role="grader", default=default_model)
        )
        candidate_support = await evaluate_description_fact_support(
            facts,
            LLMDescriptionFactSupportGrader(verifier_model),
        )
        rejected = [
            decision
            for decision in candidate_support.facts
            if decision.verdict is not GroundingVerdict.SUPPORTED
        ]
        rejected_by_id = {decision.fact_id: decision for decision in rejected}
        if rejected_by_id:
            facts = facts.model_copy(
                update={
                    "all_facts": [
                        fact.model_copy(
                            update={
                                "status": FactStatus.EXCLUDED,
                                "usage": FactUsage.INTERNAL_ONLY,
                                "exclusion_reason": (
                                    "Semantic verifier rejected candidate: "
                                    f"{rejected_by_id[fact.id].verdict.value}: "
                                    f"{rejected_by_id[fact.id].reason}"
                                ),
                            }
                        )
                        if fact.id in rejected_by_id
                        else fact
                        for fact in facts.all_facts
                    ]
                }
            )

        state.store.set("property_facts", facts.model_dump(mode="json"))
        state.store.set(
            "rejected_description_candidates",
            [decision.model_dump(mode="json") for decision in rejected],
        )
        state.store.set(
            "ingestion_warnings",
            [warning.model_dump(mode="json") for warning in facts.warnings],
        )
        state.store.set(
            "fact_conflicts",
            [conflict.model_dump(mode="json") for conflict in facts.conflicts],
        )
        state.user_prompt.text = LLMListingGenerator.build_prompt(facts)
        return await generate(
            state,
            response_schema=ResponseSchema(
                name="listing_content",
                json_schema=_inline_json_schema_refs(
                    ListingContent.model_json_schema()
                ),
            ),
        )

    return solve


SCORE_METRICS = {
    "description_fact_support_rate": [mean()],
    "schema_valid": [mean()],
    "structure_valid": [mean()],
    "citation_valid": [mean()],
    "numeric_consistency": [mean()],
    "conflict_free": [mean()],
    "non_repetitive": [mean()],
    "grounded_statement_rate": [mean()],
    "grounding_passed": [mean()],
    "editorial_quality_passed": [mean()],
}


@scorer(metrics=SCORE_METRICS)
def pipeline_evaluation():
    async def score(state: TaskState, target: Target) -> Score:
        facts = PropertyFacts.model_validate(state.store.get("property_facts"))
        completion = state.output.completion if state.output is not None else ""
        result = GenerationResult.from_raw_completion(completion)
        deterministic = evaluate_listing(result, facts)

        default_model = get_model()
        grader_model = InspectTextModel(get_model(role="grader", default=default_model))
        description_support = await evaluate_description_fact_support(
            facts,
            LLMDescriptionFactSupportGrader(grader_model),
        )

        values: dict[str, int | float | None] = {
            "description_fact_support_rate": description_support.support_rate,
            **{check.name: int(check.passed) for check in deterministic.checks},
        }
        explanation: dict[str, Any] = {
            "description_extraction": {
                "semantic_support": description_support.model_dump(mode="json"),
                "rejected_candidates": state.store.get(
                    "rejected_description_candidates"
                )
                or [],
            },
            "deterministic": {
                check.name: {
                    "passed": check.passed,
                    "explanation": check.explanation,
                    "details": check.details,
                }
                for check in deterministic.checks
            },
        }

        extraction_passed = description_support.passed
        if result.content is None or not deterministic.passed or not extraction_passed:
            values.update(
                grounded_statement_rate=0.0,
                grounding_passed=0,
                editorial_quality_passed=0,
            )
            explanation["semantic_grounding"] = (
                "Skipped because a mandatory extraction, schema, or deterministic criterion failed."
            )
            explanation["editorial_quality"] = "Skipped because grounding did not pass."
        else:
            semantic = await evaluate_semantic_grounding(
                result.content,
                facts,
                LLMSemanticGroundingGrader(grader_model),
            )
            values["grounded_statement_rate"] = semantic.grounded_statement_rate
            values["grounding_passed"] = int(semantic.passed)
            explanation["semantic_grounding"] = semantic.model_dump(mode="json")

            if semantic.passed:
                quality = await LLMEditorialQualityGrader(grader_model).evaluate(
                    result.content,
                )
                values["editorial_quality_passed"] = int(quality.passed)
                explanation["editorial_quality"] = quality.model_dump(mode="json")
            else:
                values["editorial_quality_passed"] = 0
                explanation["editorial_quality"] = (
                    "Skipped because grounded_statement_rate was below 100%."
                )

        return Score(
            value=values,
            answer=completion,
            explanation=json.dumps(explanation, ensure_ascii=False, indent=2),
            metadata={
                "property_facts": facts.model_dump(mode="json"),
                "generation_result": result.model_dump(mode="json"),
            },
        )

    return score


def property_dataset() -> MemoryDataset:
    fixtures = load_property_fixtures(FIXTURES_PATH)
    return MemoryDataset(
        name="synthetic_property_fixtures",
        samples=[
            Sample(
                id=fixture.case_id,
                input=fixture.property.model_dump_json(),
                target="grounded",
                metadata={
                    "case_id": fixture.case_id,
                    "holdout": fixture.holdout,
                },
            )
            for fixture in fixtures
        ],
    )


@task
def property_content_eval(
    model: str | Model | None = None,
    grader_model: str | Model | None = None,
) -> Task:
    """Use programmed live defaults while allowing offline model injection."""

    return Task(
        dataset=property_dataset(),
        solver=build_facts_and_generate(),
        scorer=pipeline_evaluation(),
        model=model if model is not None else GENERATOR_MODEL,
        model_roles={
            "grader": grader_model if grader_model is not None else GRADER_MODEL
        },
        config=GenerateConfig(temperature=0, max_tokens=2500),
    )
