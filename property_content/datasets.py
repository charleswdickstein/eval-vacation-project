"""Fixture loading with expectations kept separate from production input models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from property_content.contracts import DescriptionFactCandidate, PropertyData


class PropertyFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    holdout: bool
    property: PropertyData
    expected_description_facts: list[DescriptionFactCandidate]


def load_property_fixtures(path: str | Path) -> list[PropertyFixture]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PropertyFixture.model_validate(item) for item in data]
