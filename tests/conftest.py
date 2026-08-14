from __future__ import annotations

from pathlib import Path

import pytest

from property_content.datasets import PropertyFixture, load_property_fixtures

FIXTURES_PATH = Path(__file__).parents[1] / "fixtures" / "properties.json"


@pytest.fixture(scope="session")
def property_fixtures() -> list[PropertyFixture]:
    return load_property_fixtures(FIXTURES_PATH)


@pytest.fixture()
def rich_fixture(property_fixtures: list[PropertyFixture]) -> PropertyFixture:
    return next(
        item for item in property_fixtures if item.case_id == "description_rich"
    )
