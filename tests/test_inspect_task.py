from __future__ import annotations

from inspect_ai import Task
from inspect_ai.model import get_model

from property_content.contracts import ListingContent
from property_content.inspect_task import (
    GENERATOR_MODEL,
    GRADER_MODEL,
    _inline_json_schema_refs,
    property_content_eval,
    property_dataset,
)
from property_content.ingestion import _DescriptionExtraction


def test_inspect_schema_inlines_pydantic_references() -> None:
    schema = _inline_json_schema_refs(ListingContent.model_json_schema())

    assert "$defs" not in schema
    assert "$ref" not in str(schema)
    assert schema["properties"]["hero_headline"]["type"] == "object"


def test_description_extraction_schema_avoids_nullable_any_of() -> None:
    schema = _inline_json_schema_refs(_DescriptionExtraction.model_json_schema())

    assert "anyOf" not in str(schema)
    source_quote = schema["properties"]["facts"]["items"]["properties"][
        "source_quote"
    ]
    assert source_quote["type"] == "string"
    assert "source_quote" in schema["properties"]["facts"]["items"]["required"]


def test_inspect_dataset_contains_four_independent_samples() -> None:
    dataset = property_dataset()

    assert len(dataset) == 4
    assert len({sample.id for sample in dataset}) == 4
    assert sum(bool(sample.metadata["holdout"]) for sample in dataset) == 1
    assert all(set(sample.metadata) == {"case_id", "holdout"} for sample in dataset)


def test_live_model_defaults_are_programmed() -> None:
    assert GENERATOR_MODEL == "anthropic/claude-sonnet-4-5"
    assert GRADER_MODEL == "anthropic/claude-sonnet-4-5"


def test_inspect_task_builds_without_credentials_with_injected_models() -> None:
    model = get_model("mockllm/model")
    evaluation_task = property_content_eval(model=model, grader_model=model)

    assert isinstance(evaluation_task, Task)
    assert evaluation_task.dataset is not None
    assert str(evaluation_task.model) == "mockllm/model"
    assert evaluation_task.model_roles is not None
    assert str(evaluation_task.model_roles["grader"]) == "mockllm/model"
