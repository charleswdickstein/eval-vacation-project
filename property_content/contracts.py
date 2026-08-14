"""Typed contracts shared by ingestion, generation, and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Base model that rejects accidental or misspelled fields."""

    model_config = ConfigDict(extra="forbid")


class PropertyDescription(StrictModel):
    name: str
    headline: str
    description: str


class RentalInfo(StrictModel):
    max_guests: int = Field(ge=1)
    bedrooms: int = Field(ge=0)
    bathrooms: float = Field(ge=0)


class Location(StrictModel):
    city: str
    country: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Policies(StrictModel):
    cancellation_policy: str | None
    payment_schedule: str | None
    damage_deposit: str | None


class HouseRules(StrictModel):
    check_in_time: str
    check_out_time: str


class PropertyData(StrictModel):
    property_id: int
    property_name: str
    property_type: str
    description: PropertyDescription
    amenities: list[str]
    image_urls: list[AnyHttpUrl]
    reviews: list[str]
    num_of_reviews: int = Field(ge=0)
    average_review_score: float = Field(ge=0, le=5)
    rental_info: RentalInfo
    location: Location
    policies: Policies
    house_rules: HouseRules

    @field_validator("property_name", "property_type", mode="after")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class FactSource(StrEnum):
    STRUCTURED = "structured"
    DESCRIPTION = "description"
    REVIEW = "review"


class FactUsage(StrEnum):
    MARKETING = "marketing"
    ATTRIBUTED_ONLY = "attributed_only"
    INTERNAL_ONLY = "internal_only"


class FactStatus(StrEnum):
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    EXCLUDED = "excluded"


class FactPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ListingSection(StrEnum):
    HERO_HEADLINE = "hero_headline"
    PROPERTY_HIGHLIGHTS = "property_highlights"
    ABOUT_THIS_PLACE = "about_this_place"
    AMENITIES_DESCRIPTIONS = "amenities_descriptions"


FactValue = str | int | float | bool | None


class Fact(StrictModel):
    id: str
    category: str
    value: FactValue
    text: str
    source: FactSource
    source_path: str
    source_quote: str | None = None
    usage: FactUsage
    status: FactStatus
    priority: FactPriority
    suitable_sections: list[ListingSection] = Field(default_factory=list)
    exclusion_reason: str | None = None

    @property
    def is_claimable(self) -> bool:
        return (
            self.status is FactStatus.ACTIVE
            and self.usage is not FactUsage.INTERNAL_ONLY
        )


class FactConflict(StrictModel):
    fact_id: str
    authoritative_fact_id: str
    reason: str


class IngestionWarning(StrictModel):
    code: str
    source_path: str
    message: str


class DescriptionFactCandidate(StrictModel):
    category: str
    value: FactValue
    text: str
    source_quote: str | None = None
    priority: FactPriority = FactPriority.MEDIUM
    suitable_sections: list[ListingSection] = Field(default_factory=list)


class PropertyFacts(StrictModel):
    property_id: int
    normalized_description: str
    all_facts: list[Fact]
    warnings: list[IngestionWarning] = Field(default_factory=list)
    conflicts: list[FactConflict] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_fact_ids(self) -> PropertyFacts:
        fact_ids = [fact.id for fact in self.all_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("all_facts contains duplicate fact IDs")
        return self

    @property
    def fact_index(self) -> dict[str, Fact]:
        return {fact.id: fact for fact in self.all_facts}

    @property
    def claimable_facts(self) -> list[Fact]:
        priority_order = {
            FactPriority.HIGH: 0,
            FactPriority.MEDIUM: 1,
            FactPriority.LOW: 2,
        }
        return sorted(
            (fact for fact in self.all_facts if fact.is_claimable),
            key=lambda fact: (priority_order[fact.priority], fact.id),
        )


class GroundedStatement(StrictModel):
    text: str
    fact_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def statement_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("fact_ids")
    @classmethod
    def fact_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("fact_ids must be unique")
        return value


class AmenityDescription(GroundedStatement):
    amenity_code: str


class ListingContent(StrictModel):
    """The four mandatory generation outputs from the assignment."""

    hero_headline: GroundedStatement
    property_highlights: list[GroundedStatement] = Field(min_length=1)
    about_this_place: list[GroundedStatement] = Field(min_length=1)
    amenities_descriptions: list[AmenityDescription]

    def iter_statements(self) -> list[tuple[ListingSection, GroundedStatement]]:
        statements: list[tuple[ListingSection, GroundedStatement]] = [
            (ListingSection.HERO_HEADLINE, self.hero_headline)
        ]
        statements.extend(
            (ListingSection.PROPERTY_HIGHLIGHTS, statement)
            for statement in self.property_highlights
        )
        statements.extend(
            (ListingSection.ABOUT_THIS_PLACE, statement)
            for statement in self.about_this_place
        )
        statements.extend(
            (ListingSection.AMENITIES_DESCRIPTIONS, statement)
            for statement in self.amenities_descriptions
        )
        return statements


class GenerationResult(StrictModel):
    raw_completion: str
    content: ListingContent | None = None
    parse_error: str | None = None

    @model_validator(mode="after")
    def content_xor_error(self) -> GenerationResult:
        if (self.content is None) == (self.parse_error is None):
            raise ValueError("exactly one of content or parse_error must be set")
        return self

    @classmethod
    def from_raw_completion(cls, raw_completion: str) -> GenerationResult:
        try:
            content = ListingContent.model_validate_json(raw_completion)
        except ValidationError as exc:
            return cls(raw_completion=raw_completion, parse_error=str(exc))
        return cls(raw_completion=raw_completion, content=content)


class RenderedAmenity(StrictModel):
    amenity_code: str
    text: str


class RenderedListing(StrictModel):
    hero_headline: str
    property_highlights: list[str]
    about_this_place: str
    amenities_descriptions: list[RenderedAmenity]


class CheckResult(StrictModel):
    name: str
    passed: bool
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)


class DeterministicEvaluation(StrictModel):
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        mandatory_check_names = {
            "schema_valid",
            "structure_valid",
            "citation_valid",
            "numeric_consistency",
            "conflict_free",
        }
        return all(
            check.passed for check in self.checks if check.name in mandatory_check_names
        )

    def by_name(self, name: str) -> CheckResult:
        return next(check for check in self.checks if check.name == name)


class GroundingVerdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class StatementGrounding(StrictModel):
    section: ListingSection
    statement: str
    fact_ids: list[str]
    verdict: GroundingVerdict
    unsupported_spans: list[str] = Field(default_factory=list)
    reason: str


class SemanticGroundingEvaluation(StrictModel):
    statements: list[StatementGrounding]

    @property
    def grounded_statement_rate(self) -> float:
        if not self.statements:
            return 0.0
        supported = sum(
            result.verdict is GroundingVerdict.SUPPORTED for result in self.statements
        )
        return supported / len(self.statements)

    @property
    def passed(self) -> bool:
        return bool(self.statements) and self.grounded_statement_rate == 1.0
