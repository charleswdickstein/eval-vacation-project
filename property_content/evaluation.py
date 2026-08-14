"""Deterministic gates and injectable model-based grounding/quality graders."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from property_content.contracts import (
    CheckResult,
    DeterministicEvaluation,
    Fact,
    FactStatus,
    FactUsage,
    GenerationResult,
    GroundedStatement,
    GroundingVerdict,
    ListingContent,
    ListingSection,
    PropertyFacts,
    SemanticGroundingEvaluation,
    StatementGrounding,
)
from property_content.generation import render_listing
from property_content.modeling import AsyncTextModel

WORD_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _normalized_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def _content_words(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "at",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "this",
        "to",
        "with",
    }
    return {word for word in WORD_RE.findall(text.casefold()) if word not in stop_words}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _statements(
    content: ListingContent,
) -> list[tuple[ListingSection, GroundedStatement]]:
    return content.iter_statements()


def _schema_check(result: GenerationResult) -> CheckResult:
    if result.content is None:
        return CheckResult(
            name="schema_valid",
            passed=False,
            explanation="Model output did not parse as ListingContent with all four required fields.",
            details={"parse_error": result.parse_error},
        )
    return CheckResult(
        name="schema_valid",
        passed=True,
        explanation="Output parsed as ListingContent and contains all four required fields.",
    )


def _structure_check(content: ListingContent, facts: PropertyFacts) -> CheckResult:
    failures: list[str] = []
    headline_words = _word_count(content.hero_headline.text)
    if not 4 <= headline_words <= 14:
        failures.append(f"hero_headline has {headline_words} words; expected 4-14")

    highlight_count = len(content.property_highlights)
    if not 3 <= highlight_count <= 5:
        failures.append(
            f"property_highlights has {highlight_count} items; expected 3-5"
        )
    for index, statement in enumerate(content.property_highlights):
        words = _word_count(statement.text)
        if words > 30:
            failures.append(
                f"property_highlights[{index}] has {words} words; maximum is 30"
            )

    about_words = sum(
        _word_count(statement.text) for statement in content.about_this_place
    )
    if not 20 <= about_words <= 140:
        failures.append(f"about_this_place has {about_words} words; expected 20-140")

    amenity_count = len(content.amenities_descriptions)
    known_amenities = [
        fact for fact in facts.claimable_facts if fact.category == "amenity"
    ]
    if amenity_count > 6:
        failures.append(
            f"amenities_descriptions has {amenity_count} items; maximum is 6"
        )
    if known_amenities and amenity_count == 0:
        failures.append(
            "amenities_descriptions is empty despite recognized amenity facts"
        )

    return CheckResult(
        name="structure_valid",
        passed=not failures,
        explanation=(
            "All four output fields meet declared count and length rules."
            if not failures
            else "; ".join(failures)
        ),
        details={"failures": failures},
    )


def _citation_check(content: ListingContent, facts: PropertyFacts) -> CheckResult:
    fact_index = facts.fact_index
    failures: list[dict[str, object]] = []
    for section, statement in _statements(content):
        for fact_id in statement.fact_ids:
            fact = fact_index.get(fact_id)
            if fact is None:
                failures.append(
                    {
                        "section": section,
                        "statement": statement.text,
                        "fact_id": fact_id,
                        "reason": "unknown fact ID",
                    }
                )
            elif not fact.is_claimable:
                failures.append(
                    {
                        "section": section,
                        "statement": statement.text,
                        "fact_id": fact_id,
                        "reason": f"fact is {fact.status}/{fact.usage}",
                    }
                )

    return CheckResult(
        name="citation_valid",
        passed=not failures,
        explanation=(
            "Every statement cites only active facts permitted for generated copy."
            if not failures
            else f"Found {len(failures)} invalid fact reference(s)."
        ),
        details={"failures": failures},
    )


def _numeric_check(content: ListingContent, facts: PropertyFacts) -> CheckResult:
    fact_index = facts.fact_index
    failures: list[dict[str, object]] = []
    for section, statement in _statements(content):
        observed = set(NUMBER_RE.findall(statement.text))
        if not observed:
            continue
        supported: set[str] = set()
        for fact_id in statement.fact_ids:
            fact = fact_index.get(fact_id)
            if fact is not None:
                supported.update(NUMBER_RE.findall(str(fact.value)))
                supported.update(NUMBER_RE.findall(fact.text))
        unsupported = sorted(observed - supported)
        if unsupported:
            failures.append(
                {
                    "section": section,
                    "statement": statement.text,
                    "unsupported_numbers": unsupported,
                    "cited_fact_ids": statement.fact_ids,
                }
            )

    return CheckResult(
        name="numeric_consistency",
        passed=not failures,
        explanation=(
            "Every numeric claim appears in at least one cited fact."
            if not failures
            else f"Found unsupported numeric claims in {len(failures)} statement(s)."
        ),
        details={"failures": failures},
    )


def _conflict_check(content: ListingContent, facts: PropertyFacts) -> CheckResult:
    fact_index = facts.fact_index
    failures: list[dict[str, str]] = []
    for section, statement in _statements(content):
        for fact_id in statement.fact_ids:
            fact = fact_index.get(fact_id)
            if fact is not None and (
                fact.status is not FactStatus.ACTIVE
                or fact.usage is FactUsage.INTERNAL_ONLY
            ):
                failures.append(
                    {
                        "section": section,
                        "statement": statement.text,
                        "fact_id": fact_id,
                        "status": fact.status,
                    }
                )
    return CheckResult(
        name="conflict_free",
        passed=not failures,
        explanation=(
            "No statement cites a conflicted, excluded, or internal-only fact."
            if not failures
            else f"Found {len(failures)} prohibited fact use(s)."
        ),
        details={"failures": failures},
    )


def _repetition_check(content: ListingContent) -> CheckResult:
    statements = _statements(content)
    normalized = [_normalized_text(statement.text) for _, statement in statements]
    duplicate_text = sorted(
        text for text, count in Counter(normalized).items() if count > 1
    )

    fact_sections: dict[str, set[ListingSection]] = defaultdict(set)
    for section, statement in statements:
        for fact_id in statement.fact_ids:
            fact_sections[fact_id].add(section)
    overused_facts = {
        fact_id: sorted(section.value for section in sections)
        for fact_id, sections in fact_sections.items()
        if len(sections) > 2
    }

    similar_pairs: list[dict[str, object]] = []
    for left_index, (_, left) in enumerate(statements):
        for right_index in range(left_index + 1, len(statements)):
            right = statements[right_index][1]
            similarity = _jaccard(_content_words(left.text), _content_words(right.text))
            if similarity > 0.65:
                similar_pairs.append(
                    {
                        "left": left.text,
                        "right": right.text,
                        "similarity": round(similarity, 3),
                    }
                )

    passed = not duplicate_text and not overused_facts and not similar_pairs
    return CheckResult(
        name="non_repetitive",
        passed=passed,
        explanation=(
            "No exact, highly similar, or cross-section fact repetition was detected."
            if passed
            else "Detected repetition that should be reviewed as an editorial-quality issue."
        ),
        details={
            "duplicate_text": duplicate_text,
            "facts_used_in_more_than_two_sections": overused_facts,
            "similar_pairs": similar_pairs,
        },
    )


def evaluate_listing(
    result: GenerationResult,
    facts: PropertyFacts,
) -> DeterministicEvaluation:
    """Apply named, explainable checks without making a model call."""

    schema = _schema_check(result)
    if result.content is None:
        skipped = [
            CheckResult(
                name=name,
                passed=False,
                explanation="Not evaluated because schema_valid failed.",
            )
            for name in (
                "structure_valid",
                "citation_valid",
                "numeric_consistency",
                "conflict_free",
                "non_repetitive",
            )
        ]
        return DeterministicEvaluation(checks=[schema, *skipped])

    content = result.content
    return DeterministicEvaluation(
        checks=[
            schema,
            _structure_check(content, facts),
            _citation_check(content, facts),
            _numeric_check(content, facts),
            _conflict_check(content, facts),
            _repetition_check(content),
        ]
    )


class _GroundingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: GroundingVerdict
    unsupported_spans: list[str] = Field(default_factory=list)
    reason: str


class SemanticGroundingGrader(ABC):
    @abstractmethod
    async def grade(
        self, statement: str, cited_facts: Sequence[Fact]
    ) -> _GroundingDecision:
        """Judge whether all assertions follow from only the cited facts."""


class FakeSemanticGroundingGrader(SemanticGroundingGrader):
    def __init__(self, decisions: Sequence[GroundingVerdict] | None = None) -> None:
        self._decisions = iter(decisions or [])
        self.calls: list[tuple[str, list[Fact]]] = []

    async def grade(
        self, statement: str, cited_facts: Sequence[Fact]
    ) -> _GroundingDecision:
        self.calls.append((statement, list(cited_facts)))
        verdict = next(self._decisions, GroundingVerdict.SUPPORTED)
        return _GroundingDecision(
            verdict=verdict,
            unsupported_spans=[]
            if verdict is GroundingVerdict.SUPPORTED
            else [statement],
            reason=(
                "All factual assertions are supported by the cited facts."
                if verdict is GroundingVerdict.SUPPORTED
                else "Calibration response marks this statement as not fully supported."
            ),
        )


class LLMSemanticGroundingGrader(SemanticGroundingGrader):
    """Entailment grader that sees a statement and its cited facts only."""

    def __init__(self, model: AsyncTextModel) -> None:
        self._model = model

    async def grade(
        self, statement: str, cited_facts: Sequence[Fact]
    ) -> _GroundingDecision:
        prompt = self.build_prompt(statement, cited_facts)
        raw_completion = await self._model.complete(
            prompt,
            response_schema=_GroundingDecision.model_json_schema(),
        )
        try:
            return _GroundingDecision.model_validate_json(raw_completion)
        except ValidationError as exc:
            raise ValueError(f"semantic grader returned invalid JSON: {exc}") from exc

    @staticmethod
    def build_prompt(statement: str, cited_facts: Sequence[Fact]) -> str:
        evidence = [
            {
                "id": fact.id,
                "text": fact.text,
                "value": fact.value,
                "source_quote": fact.source_quote,
            }
            for fact in cited_facts
        ]
        return (
            "Determine whether EVERY factual assertion in STATEMENT follows from EVIDENCE. "
            "Use SUPPORTED only when the complete statement is entailed. Use PARTIAL when "
            "some but not all assertions are supported, and UNSUPPORTED when the central "
            "claim is unsupported. Marketing tone is allowed, but new views, quality, "
            "distance, location, capacity, or amenity claims are not. Return JSON only "
            "with verdict, unsupported_spans, and a concrete reason. EVIDENCE entries are "
            "authoritative source records for this property, not claims awaiting outside "
            "corroboration. Never call matching evidence circular or demand an independent "
            "source. Headline and highlight fragments still make ordinary factual claims: "
            "'1 bathroom' is SUPPORTED by evidence that the property has 1 bathroom. A "
            "definite reference such as 'the loft' is SUPPORTED when the evidence identifies "
            "this property's type as a loft. A faithful paraphrase is SUPPORTED when "
            "its cited facts cover every assertion; exact word overlap is unnecessary. An exact "
            "or faithful attribution such as 'A guest review says: X' is SUPPORTED when the "
            "cited review evidence contains X. By contrast, 'fully equipped kitchen' is "
            "PARTIAL or UNSUPPORTED when the evidence states only that kitchen facilities "
            "exist.\n\n"
            f"STATEMENT:\n{statement}\n\nEVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
        )


async def evaluate_semantic_grounding(
    content: ListingContent,
    facts: PropertyFacts,
    grader: SemanticGroundingGrader,
) -> SemanticGroundingEvaluation:
    fact_index = facts.fact_index
    results: list[StatementGrounding] = []
    for section, statement in _statements(content):
        missing = [
            fact_id for fact_id in statement.fact_ids if fact_id not in fact_index
        ]
        if missing:
            decision = _GroundingDecision(
                verdict=GroundingVerdict.UNSUPPORTED,
                unsupported_spans=[statement.text],
                reason=f"Unknown cited fact IDs: {', '.join(missing)}",
            )
        else:
            cited_facts = [fact_index[fact_id] for fact_id in statement.fact_ids]
            decision = await grader.grade(statement.text, cited_facts)
        results.append(
            StatementGrounding(
                section=section,
                statement=statement.text,
                fact_ids=statement.fact_ids,
                verdict=decision.verdict,
                unsupported_spans=decision.unsupported_spans,
                reason=decision.reason,
            )
        )
    return SemanticGroundingEvaluation(statements=results)


class EditorialVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class EditorialDimensionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: EditorialVerdict
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class _EditorialAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specificity: EditorialDimensionEvaluation
    section_differentiation: EditorialDimensionEvaluation
    clarity_and_concision: EditorialDimensionEvaluation
    guest_facing_language: EditorialDimensionEvaluation
    publishability: EditorialDimensionEvaluation

    @property
    def dimensions(self) -> tuple[EditorialDimensionEvaluation, ...]:
        return (
            self.specificity,
            self.section_differentiation,
            self.clarity_and_concision,
            self.guest_facing_language,
            self.publishability,
        )


class EditorialQualityEvaluation(_EditorialAssessment):
    overall_verdict: EditorialVerdict

    @property
    def passed(self) -> bool:
        return self.overall_verdict is EditorialVerdict.PASS


class EditorialQualityGrader(ABC):
    @abstractmethod
    async def evaluate(
        self,
        content: ListingContent,
    ) -> EditorialQualityEvaluation:
        """Evaluate one grounded output against an observable editorial rubric."""


class LLMEditorialQualityGrader(EditorialQualityGrader):
    def __init__(self, model: AsyncTextModel) -> None:
        self._model = model

    async def evaluate(
        self,
        content: ListingContent,
    ) -> EditorialQualityEvaluation:
        rendered = render_listing(content).model_dump(mode="json")
        public_copy = {
            "hero_headline": rendered["hero_headline"],
            "property_highlights": rendered["property_highlights"],
            "about_this_place": rendered["about_this_place"],
            "amenities_descriptions": [
                item["text"] for item in rendered["amenities_descriptions"]
            ],
        }
        prompt = (
            "Evaluate this ONE generated vacation-rental listing. Do not compare it with "
            "another output and do not reward mentioning more facts. Evaluate exactly five "
            "dimensions: (1) specificity: concrete property details rather than generic "
            "marketing phrases. Judge the detail that is present; do not demand absent fact "
            "categories such as dimensions, nearby attractions, or distances, and do not "
            "penalize factual restraint by itself. (2) section_differentiation: headline "
            "attracts, highlights are skimmable, about copy adds context, and amenities state "
            "what is available without needless repetition. Exact or lightly paraphrased "
            "cross-section reuse must fail this dimension. (3) clarity_and_concision: direct, "
            "readable language with no "
            "removable filler; (4) guest_facing_language: no raw codes, internal metadata, or "
            "awkward template language; and (5) publishability: professional, persuasive tone "
            "without unsupported hype. Factual grounding is evaluated separately. For every "
            "dimension return verdict PASS or FAIL, at least one exact excerpt from the listing "
            "as evidence, and a concrete reason. Evidence must be copied verbatim from a "
            "displayed text value, including its punctuation; do not quote JSON keys or "
            "internal structure. Return JSON only.\n\n"
            f"GENERATED_LISTING:\n{json.dumps(public_copy, ensure_ascii=False)}"
        )
        raw = await self._model.complete(
            prompt,
            response_schema=_EditorialAssessment.model_json_schema(),
        )
        data = _EditorialAssessment.model_validate_json(raw)
        source_texts = [
            public_copy["hero_headline"],
            *public_copy["property_highlights"],
            public_copy["about_this_place"],
            *public_copy["amenities_descriptions"],
        ]
        validated_dimensions = data.model_copy(deep=True)
        for name in (
            "specificity",
            "section_differentiation",
            "clarity_and_concision",
            "guest_facing_language",
            "publishability",
        ):
            decision = getattr(validated_dimensions, name)
            invalid_evidence = [
                excerpt
                for excerpt in decision.evidence
                if not any(excerpt in source_text for source_text in source_texts)
            ]
            if invalid_evidence:
                setattr(
                    validated_dimensions,
                    name,
                    decision.model_copy(
                        update={
                            "verdict": EditorialVerdict.FAIL,
                            "reason": (
                                "Evaluator evidence was not found verbatim in the generated "
                                f"listing: {invalid_evidence}"
                            ),
                        }
                    ),
                )
        overall_verdict = (
            EditorialVerdict.PASS
            if all(
                item.verdict is EditorialVerdict.PASS
                for item in validated_dimensions.dimensions
            )
            else EditorialVerdict.FAIL
        )
        return EditorialQualityEvaluation(
            **validated_dimensions.model_dump(),
            overall_verdict=overall_verdict,
        )
