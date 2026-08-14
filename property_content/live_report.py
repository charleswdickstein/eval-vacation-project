"""Render a selected Inspect evaluation as a reviewer-friendly HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "final"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "live-report.html"

MANDATORY_METRICS = (
    "description_fact_support_rate",
    "schema_valid",
    "structure_valid",
    "citation_valid",
    "numeric_consistency",
    "conflict_free",
    "grounded_statement_rate",
    "grounding_passed",
)

METRIC_COPY = {
    "description_fact_support_rate": (
        "Description fact support",
        "Every accepted description fact is semantically supported by the source property description.",
    ),
    "schema_valid": (
        "Output schema",
        "The response contains the four required structured fields.",
    ),
    "structure_valid": (
        "Section structure",
        "Headline, highlights, about copy, and amenities meet their shape rules.",
    ),
    "citation_valid": (
        "Fact citations",
        "Every statement cites only active facts generation was allowed to use.",
    ),
    "numeric_consistency": (
        "Numbers",
        "Every generated number is present in its cited evidence.",
    ),
    "conflict_free": (
        "Conflicts",
        "No statement uses a conflicted, excluded, or internal-only fact.",
    ),
    "grounded_statement_rate": (
        "Semantic grounding",
        "Every factual assertion is entailed by the facts cited for that statement.",
    ),
    "grounding_passed": (
        "Grounding requirement",
        "The listing reached the required 100% statement-grounding threshold.",
    ),
    "editorial_quality_passed": (
        "Editorial quality",
        "All five direct editorial dimensions passed for this generated output.",
    ),
}

CASE_COPY = {
    "description_rich": "Rich description",
    "sparse": "Sparse input",
    "conflicting_capacity": "Conflicting capacity",
    "adversarial_holdout": "Adversarial holdout",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _status_badge(passed: bool, pass_label: str, fail_label: str) -> str:
    css_class = "pass" if passed else "fail-badge"
    label = pass_label if passed else fail_label
    return f'<span class="{css_class}">{_e(label)}</span>'


def _selected_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.eval"), reverse=True)
    if not logs:
        raise FileNotFoundError(f"No .eval logs found in {log_dir}")
    return logs[0]


def _score_for(sample: Any) -> Any:
    try:
        return sample.scores["pipeline_evaluation"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Sample {sample.id!r} has no pipeline_evaluation score") from exc


def _explanation(score: Any) -> dict[str, Any]:
    return json.loads(score.explanation) if score.explanation else {}


def _statements(content: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    statements = [("Hero headline", content["hero_headline"])]
    statements.extend(
        ("Property highlight", statement)
        for statement in content["property_highlights"]
    )
    statements.extend(
        ("About this place", statement)
        for statement in content["about_this_place"]
    )
    statements.extend(
        ("Amenity description", statement)
        for statement in content["amenities_descriptions"]
    )
    return statements


def _render_input_list(values: list[Any], empty_message: str) -> str:
    if not values:
        return f'<p class="empty">{_e(empty_message)}</p>'
    return '<ul class="input-list">' + "".join(
        f"<li>{_e(value)}</li>" for value in values
    ) + "</ul>"


def _render_mapping(values: dict[str, Any]) -> str:
    items = []
    for key, value in values.items():
        label = key.replace("_", " ").title()
        rendered = (
            '<span class="missing">Not supplied</span>'
            if value is None
            else _e(value)
        )
        items.append(f"<div><dt>{_e(label)}</dt><dd>{rendered}</dd></div>")
    return '<dl class="input-facts">' + "".join(items) + "</dl>"


def _render_input(property_data: dict[str, Any]) -> str:
    description = property_data.get("description", {})
    rental = property_data.get("rental_info", {})
    location = property_data.get("location", {})
    raw_json = json.dumps(property_data, ensure_ascii=False, indent=2)
    summary = {
        "property_id": property_data.get("property_id"),
        "property_type": property_data.get("property_type"),
        "city": location.get("city"),
        "country": location.get("country"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "max_guests": rental.get("max_guests"),
        "bedrooms": rental.get("bedrooms"),
        "bathrooms": rental.get("bathrooms"),
        "average_review_score": property_data.get("average_review_score"),
        "number_of_reviews": property_data.get("num_of_reviews"),
    }
    description_fields = {
        "description_name": description.get("name"),
        "source_headline": description.get("headline"),
    }
    return f"""
    <section class="input-card" aria-label="Raw property input">
      <p class="section-label">Input · Raw property data</p>
      <p class="stage-help">This is the complete source payload available to ingestion. Missing values and conflicting claims are intentionally visible.</p>
      {_render_mapping(summary)}
      <div class="input-grid">
        <div class="input-group full">
          <h4>Owner-supplied description</h4>
          {_render_mapping(description_fields)}
          <p class="field-label">Raw description body</p>
          <pre class="source-copy">{_e(description.get('description', ''))}</pre>
        </div>
        <div class="input-group">
          <h4>Amenity codes</h4>
          {_render_input_list(property_data.get('amenities', []), 'No amenities supplied.')}
        </div>
        <div class="input-group">
          <h4>Reviews</h4>
          {_render_input_list(property_data.get('reviews', []), 'No review text supplied.')}
        </div>
        <div class="input-group">
          <h4>Policies</h4>
          {_render_mapping(property_data.get('policies', {}))}
        </div>
        <div class="input-group">
          <h4>House rules</h4>
          {_render_mapping(property_data.get('house_rules', {}))}
        </div>
        <div class="input-group full">
          <h4>Image URLs</h4>
          {_render_input_list(property_data.get('image_urls', []), 'No images supplied.')}
        </div>
      </div>
      <details class="raw-payload">
        <summary>Show exact input JSON</summary>
        <pre>{_e(raw_json)}</pre>
      </details>
    </section>
    """


def _render_listing(content: dict[str, Any]) -> str:
    raw_json = json.dumps(content, ensure_ascii=False, indent=2)
    highlights = "".join(
        f"<li>{_e(item['text'])}</li>" for item in content["property_highlights"]
    )
    about = " ".join(item["text"] for item in content["about_this_place"])
    amenities = content["amenities_descriptions"]
    if amenities:
        amenity_html = "".join(
            f"<li>{_e(item['text'])}</li>" for item in amenities
        )
        amenity_block = f'<ul class="amenities">{amenity_html}</ul>'
    else:
        amenity_block = '<p class="empty">No recognized amenities were supplied.</p>'
    return f"""
    <section class="listing" aria-label="Generated marketing copy">
      <p class="section-label">Output · Generated marketing copy</p>
      <p class="field-label">Hero headline</p>
      <h3>{_e(content['hero_headline']['text'])}</h3>
      <div class="listing-grid">
        <div><h4>Property highlights</h4><ul class="highlights">{highlights}</ul></div>
        <div><h4>About this place</h4><p>{_e(about)}</p></div>
      </div>
      <div><h4>Amenities descriptions</h4>{amenity_block}</div>
      <details class="raw-output">
        <summary>Show exact output JSON</summary>
        <pre>{_e(raw_json)}</pre>
      </details>
    </section>
    """


def _render_evidence(
    content: dict[str, Any],
    fact_index: dict[str, dict[str, Any]],
    explanation: dict[str, Any],
) -> str:
    semantic = explanation.get("semantic_grounding", {})
    semantic_note = semantic if isinstance(semantic, str) else ""
    semantic_items = semantic.get("statements", []) if isinstance(semantic, dict) else []
    decisions = {
        item["statement"]: item
        for item in semantic_items
        if isinstance(item, dict)
    }
    rows = []
    for section, statement in _statements(content):
        decision = decisions.get(statement["text"])
        if decision:
            verdict = decision.get("verdict", "NOT EVALUATED")
            badge = _status_badge(verdict == "SUPPORTED", verdict, verdict)
            reason = decision.get("reason", "No reason recorded.")
        else:
            badge = '<span class="holdout">NOT EVALUATED</span>'
            reason = semantic_note or "No semantic-grounding decision was recorded."
        facts = []
        for fact_id in statement["fact_ids"]:
            fact = fact_index.get(fact_id, {})
            source_detail = fact.get("source_quote") or fact.get("source_path", "")
            facts.append(
                f"""
                <li><div><code>{_e(fact_id)}</code><span class="source">{_e(fact.get('source', 'unknown'))}</span></div>
                <p>{_e(fact.get('text', 'Fact not found'))}</p>
                <small>Evidence: {_e(source_detail)}</small></li>
                """
            )
        rows.append(
            f"""
            <details class="statement-evidence">
              <summary><span><b>{_e(section)}</b>{_e(statement['text'])}</span>
              {badge}</summary>
              <ul class="fact-list">{''.join(facts)}</ul>
              <p class="grader-reason">{_e(reason)}</p>
            </details>
            """
        )
    return f"""
    <details class="evidence-panel">
      <summary>Show statement-by-statement evidence</summary>
      <p class="guidance">Each generated statement is evaluated only against the facts listed beneath it.</p>
      {''.join(rows)}
    </details>
    """


def _render_metrics(values: dict[str, Any], explanation: dict[str, Any]) -> str:
    cards = []
    metric_keys = list(MANDATORY_METRICS)
    if "editorial_quality_passed" in values:
        metric_keys.append("editorial_quality_passed")
    for key in metric_keys:
        label, description = METRIC_COPY[key]
        passed = values.get(key) == 1
        cards.append(
            f"""
            <div class="metric {'ok' if passed else 'fail'}">
              <span>{'PASS' if passed else 'FAIL'}</span><h4>{_e(label)}</h4>
              <p>{_e(description)}</p>
            </div>
            """
        )
    repetition = explanation.get("deterministic", {}).get("non_repetitive", {})
    repeated = repetition.get("details", {}).get(
        "facts_used_in_more_than_two_sections", {}
    )
    repeated_text = ", ".join(sorted(repeated)) if repeated else "None"
    return f"""
      <div class="metrics">{''.join(cards)}</div>
      <div class="advisory"><strong>Advisory: repetition diagnostic</strong>
      <p>{_e(repetition.get('explanation', 'Not evaluated.'))}</p>
      <small>Repeated fact IDs: {_e(repeated_text)}. This is a writing diagnostic, not a factual-grounding failure.</small></div>
    """


def _render_quality(explanation: dict[str, Any]) -> str:
    quality = explanation.get("editorial_quality", {})
    dimension_copy = {
        "specificity": "Specificity",
        "section_differentiation": "Section differentiation",
        "clarity_and_concision": "Clarity and concision",
        "guest_facing_language": "Guest-facing language",
        "publishability": "Publishability",
    }
    if not isinstance(quality, dict):
        note = quality if isinstance(quality, str) else "Not evaluated."
        return f"""
        <div class="quality">
          <div class="quality-title"><span class="holdout">NOT EVALUATED</span><h4>Direct editorial rubric</h4></div>
          <p>{_e(note)}</p>
        </div>
        """
    if "overall_verdict" not in quality:
        return """
        <div class="quality">
          <div class="quality-title"><span class="holdout">NOT RECORDED</span><h4>Direct editorial rubric</h4></div>
          <p>This historical Inspect log predates the single-output rubric. Run a fresh live evaluation to record its five editorial decisions.</p>
        </div>
        """
    dimensions = []
    for key, label in dimension_copy.items():
        decision = quality.get(key, {})
        verdict = decision.get("verdict", "FAIL")
        evidence = "".join(
            f"<li>{_e(item)}</li>" for item in decision.get("evidence", [])
        )
        dimensions.append(
            f"""
            <div class="quality-dimension">
              {_status_badge(verdict == 'PASS', 'PASS', 'FAIL')}<h5>{_e(label)}</h5>
              <p>{_e(decision.get('reason', 'No reason recorded.'))}</p>
              <ul>{evidence}</ul>
            </div>
            """
        )
    overall = quality.get("overall_verdict", "FAIL")
    return f"""
    <div class="quality">
      <div class="quality-title">{_status_badge(overall == 'PASS', 'PASS', 'FAIL')}<h4>Direct editorial assessment of this output</h4></div>
      <p>Every dimension must pass; scores are not averaged.</p>
      <div class="quality-grid">{''.join(dimensions)}</div>
    </div>
    """


def _render_ingestion(
    facts: dict[str, Any],
    explanation: dict[str, Any],
) -> str:
    accepted = [
        fact
        for fact in facts.get("all_facts", [])
        if str(fact.get("id", "")).startswith("description.fact.")
        and fact.get("status") == "active"
    ]
    accepted_html = "".join(
        f"<li><b>{_e(fact['text'])}</b><small>{_e(fact.get('source_quote') or 'Verified against the full description')}</small></li>"
        for fact in accepted
    ) or "<li>No description-derived facts were accepted.</li>"
    conflicts = "".join(
        f"<li>{_e(item['reason'])}</li>" for item in facts.get("conflicts", [])
    ) or "<li>None</li>"
    warnings = "".join(
        f"<li>{_e(item['message'])}</li>" for item in facts.get("warnings", [])
    ) or "<li>None</li>"
    rejected = explanation.get("description_extraction", {}).get(
        "rejected_candidates", []
    )
    rejected_html = "".join(
        f"<li>{_e(item['statement'])}: {_e(item['reason'])}</li>" for item in rejected
    ) or "<li>None</li>"
    return f"""
    <details class="ingestion-panel">
      <summary>Show description extraction and ingestion decisions</summary>
      <div class="decision-grid">
        <div><h5>Accepted description facts</h5><ul class="compact">{accepted_html}</ul></div>
        <div><h5>Conflicts resolved</h5><ul class="compact">{conflicts}</ul></div>
        <div><h5>Warnings</h5><ul class="compact">{warnings}</ul></div>
        <div><h5>Rejected semantic candidates</h5><ul class="compact">{rejected_html}</ul></div>
      </div>
    </details>
    """


def _render_sample(sample: Any, index: int) -> str:
    score = _score_for(sample)
    values = dict(score.value)
    explanation = _explanation(score)
    metadata = score.metadata or {}
    content = metadata["generation_result"]["content"]
    facts = metadata["property_facts"]
    fact_index = {fact["id"]: fact for fact in facts["all_facts"]}
    property_data = json.loads(sample.input)
    case_label = CASE_COPY.get(str(sample.id), str(sample.id))
    holdout = bool((sample.metadata or {}).get("holdout"))
    mandatory_pass = all(values.get(metric) == 1 for metric in MANDATORY_METRICS)
    grounding_pass = values.get("grounding_passed") == 1
    return f"""
    <article class="property" id="sample-{_e(sample.id)}">
      <header class="property-header">
        <div><p class="eyebrow">Sample {index} · {_e(case_label)}</p><h2>{_e(property_data['property_name'])}</h2></div>
        <div class="badges">{_status_badge(mandatory_pass, 'ALL MANDATORY CRITERIA PASS', 'MANDATORY CRITERION FAILURE')}
        {_status_badge(grounding_pass, '100% GROUNDED', 'GROUNDING FAILURE')}{'<span class="holdout">HOLDOUT</span>' if holdout else ''}</div>
      </header>
      {_render_input(property_data)}
      {_render_ingestion(facts, explanation)}
      <div class="stage-divider"><span>Input transformed into grounded marketing output</span></div>
      {_render_listing(content)}
      {_render_evidence(content, fact_index, explanation)}
      <section class="evaluation"><p class="section-label">Evaluation in plain language</p>{_render_metrics(values, explanation)}</section>
      {_render_quality(explanation)}
    </article>
    """


def build_report(log_path: Path) -> str:
    log = read_eval_log(str(log_path))
    samples = log.samples or []
    if log.status != "success" or not samples:
        raise ValueError(f"Inspect log is not a completed run: {log_path}")
    scores = [_score_for(sample).value for sample in samples]
    mandatory_passes = sum(
        all(values.get(metric) == 1 for metric in MANDATORY_METRICS) for values in scores
    )
    grounding_passes = sum(values.get("grounding_passed") == 1 for values in scores)
    quality_scores = [
        values for values in scores if "editorial_quality_passed" in values
    ]
    quality_passes = sum(
        values.get("editorial_quality_passed") == 1 for values in quality_scores
    )
    quality_summary = (
        f"{quality_passes}/{len(quality_scores)}"
        if quality_scores
        else "—"
    )
    quality_label = (
        "outputs pass the direct editorial rubric"
        if quality_scores
        else "direct editorial rubric requires a fresh run"
    )
    nav = "".join(
        f'<a href="#sample-{_e(sample.id)}">{_e(CASE_COPY.get(str(sample.id), sample.id))}</a>'
        for sample in samples
    )
    rendered_samples = "".join(
        _render_sample(sample, index) for index, sample in enumerate(samples, start=1)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grounded Property Content — Live Evaluation</title>
<style>
:root{{--ink:#182129;--muted:#5b6871;--line:#dce3e7;--paper:#fff;--wash:#f4f7f8;--navy:#11384b;--teal:#087a72;--green:#14733c;--green-bg:#eaf7ef;--amber:#8a5a00;--amber-bg:#fff6df;--red:#a32e2e}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.page{{max-width:1120px;margin:0 auto;padding:38px 24px 72px}}.hero{{background:linear-gradient(135deg,#102f3e,#14556a);color:#fff;border-radius:22px;padding:42px;box-shadow:0 18px 48px #12384922}}.hero h1{{font-size:clamp(30px,5vw,52px);line-height:1.05;margin:8px 0 15px;max-width:820px}}.hero p{{max-width:760px;color:#d8e9ef;font-size:17px}}.eyebrow,.section-label{{text-transform:uppercase;letter-spacing:.12em;font-weight:800;font-size:12px;margin:0;color:var(--teal)}}.hero .eyebrow{{color:#8de0d9}}.scoreboard{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px}}.scoreboard div{{background:#ffffff14;border:1px solid #ffffff2e;border-radius:14px;padding:16px}}.scoreboard b{{display:block;font-size:30px}}.scoreboard span{{color:#d8e9ef}}.how{{background:var(--paper);border:1px solid var(--line);border-radius:18px;margin:24px 0;padding:24px}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:18px}}.flow div{{border-left:4px solid var(--teal);background:#eef7f6;padding:14px;border-radius:8px}}.flow b{{display:block}}nav{{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}}nav a{{color:var(--navy);background:#fff;border:1px solid var(--line);padding:8px 12px;border-radius:999px;text-decoration:none;font-weight:700}}.property{{background:var(--paper);border:1px solid var(--line);border-radius:20px;margin:24px 0;padding:30px;box-shadow:0 10px 30px #152b3510}}.property-header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:18px}}.property h2{{font-size:30px;margin:4px 0}}.badges{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}}.pass,.holdout,.fail-badge{{display:inline-block;border-radius:999px;padding:5px 9px;font-size:11px;font-weight:900;letter-spacing:.04em}}.pass{{color:var(--green);background:var(--green-bg)}}.fail-badge{{color:var(--red);background:#fdecec}}.holdout{{color:#6c3daf;background:#f2eaff}}.input-card{{margin:24px 0;background:#eef5f8;border:1px solid #c9dce5;border-radius:15px;padding:24px}}.stage-help{{color:var(--muted);margin:6px 0 16px}}.input-facts{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:0 0 18px}}.input-facts>div{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:9px 11px;min-width:0}}.input-facts dt{{color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.05em}}.input-facts dd{{margin:3px 0 0;font-weight:700;overflow-wrap:anywhere}}.input-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.input-group{{background:#fff;border:1px solid var(--line);border-radius:11px;padding:15px;min-width:0}}.input-group.full{{grid-column:1/-1}}.input-group .input-facts{{grid-template-columns:repeat(2,minmax(0,1fr));margin:0 0 12px}}.input-list{{margin:6px 0 0;padding-left:20px}}.input-list li{{margin:4px 0;overflow-wrap:anywhere}}.field-label{{font-size:11px;font-weight:800;text-transform:uppercase;color:var(--muted);letter-spacing:.05em}}.source-copy,.raw-payload pre,.raw-output pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#17232b;color:#e5edf1;border-radius:9px;padding:13px;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}.missing{{color:var(--muted);font-style:italic}}.raw-payload,.raw-output{{margin-top:12px;background:#fff}}.raw-payload pre,.raw-output pre{{margin:0 14px 14px;max-height:420px;overflow:auto}}.stage-divider{{display:flex;align-items:center;gap:12px;color:var(--teal);font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.07em;margin:20px 0}}.stage-divider:before,.stage-divider:after{{content:"";height:1px;background:#a9cbc8;flex:1}}.listing{{margin:24px 0;background:#f7fafb;border:1px solid var(--line);border-radius:15px;padding:24px}}.listing h3{{font:700 29px/1.15 Georgia,serif;color:var(--navy);margin:7px 0 20px}}h4{{margin:0 0 7px}}.listing-grid,.quality-grid,.decision-grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}.highlights{{padding-left:20px}}.highlights li{{margin:5px 0}}.amenities{{display:flex;flex-wrap:wrap;gap:8px;padding:0;list-style:none}}.amenities li{{display:flex;gap:7px;background:#fff;border:1px solid var(--line);padding:8px 11px;border-radius:9px}}details{{border:1px solid var(--line);border-radius:12px;margin:12px 0;background:#fff}}summary{{cursor:pointer;font-weight:800;padding:14px 16px;color:var(--navy)}}.evidence-panel,.ingestion-panel{{background:#fbfcfc}}.evidence-panel>summary,.ingestion-panel>summary{{font-size:16px}}.guidance,.input-description{{padding:0 16px;color:var(--muted)}}.statement-evidence{{margin:9px 14px}}.statement-evidence summary{{display:flex;justify-content:space-between;gap:14px;align-items:center}}.statement-evidence summary b{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}.fact-list{{list-style:none;padding:0 16px}}.fact-list li{{border-top:1px solid var(--line);padding:12px 0}}code{{font-size:12px;color:#3e5360;background:#eef2f4;padding:2px 5px;border-radius:5px}}.source{{margin-left:7px;color:var(--teal);font-size:11px;font-weight:800;text-transform:uppercase}}.fact-list p{{margin:5px 0}}small{{color:var(--muted)}}.grader-reason{{margin:10px 16px 16px;color:var(--muted)}}.evaluation{{margin:26px 0}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}}.metric{{border:1px solid var(--line);border-radius:12px;padding:14px}}.metric>span{{font-size:11px;font-weight:900;color:var(--green)}}.metric h4{{margin:4px 0}}.metric p{{margin:0;color:var(--muted);font-size:13px}}.metric.fail>span{{color:var(--red)}}.advisory{{background:var(--amber-bg);color:var(--amber);padding:15px;border-radius:12px;margin-top:11px}}.advisory p{{margin:4px 0;color:var(--ink)}}.quality{{border-left:5px solid var(--teal);background:#eef8f7;border-radius:10px;padding:20px;margin:22px 0}}.quality-title{{display:flex;gap:10px;align-items:center}}.quality-title h4{{margin:0;font-size:18px}}.quality h5{{margin:7px 0}}.quality ul{{padding-left:18px}}.decision-grid{{padding:0 18px 18px}}.compact{{padding-left:18px}}.compact li{{margin:5px 0}}.compact small{{display:block}}.empty{{color:var(--muted);font-style:italic}}footer{{color:var(--muted);font-size:13px;text-align:center;margin-top:34px}}@media(max-width:780px){{.scoreboard,.flow,.metrics,.listing-grid,.quality-grid,.decision-grid,.input-grid,.input-facts{{grid-template-columns:1fr}}.input-group.full{{grid-column:auto}}.property-header{{display:block}}.badges{{justify-content:flex-start;margin-top:12px}}.property{{padding:20px}}.hero{{padding:28px}}}}
</style></head><body><main class="page">
<header class="hero"><p class="eyebrow">Selected live Inspect evaluation</p><h1>Grounded property marketing copy</h1><p>Every sample shows the complete raw input before the generated output, then exposes the exact evidence and evaluator decisions behind it. The source Inspect log remains the machine-auditable record.</p><div class="scoreboard"><div><b>{mandatory_passes}/{len(samples)}</b><span>properties pass every mandatory criterion</span></div><div><b>{grounding_passes}/{len(samples)}</b><span>properties reach 100% grounding</span></div><div><b>{quality_summary}</b><span>{_e(quality_label)}</span></div></div></header>
<section class="how"><p class="section-label">How to read this</p><h2>Input and output together for every property</h2><p>Start with the full source input, then compare it with the four generated marketing sections. Open the evidence panel to trace any output statement to its allowed facts. Evaluation cards translate Inspect score names into plain English.</p><div class="flow"><div><b>1 · Input</b>Read the raw description, rental data, amenities, images, reviews, and policies.</div><div><b>2 · Ingest</b>Normalize data and reject unsupported or conflicting facts.</div><div><b>3 · Output</b>Generate all four required marketing sections in one call.</div><div><b>4 · Evaluate</b>Check grounding, then judge this output against five editorial criteria.</div></div></section>
<nav aria-label="Property samples">{nav}</nav>{rendered_samples}<footer>Source log: <code>{_e(log_path.name)}</code> · Synthetic evaluation dataset · No API call is required to read this report.</footer>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--log", type=Path, help="Inspect .eval log to render")
    source.add_argument(
        "--log-dir",
        type=Path,
        help="Directory whose newest timestamped .eval log should be rendered",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.log:
        log_path = args.log
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
    else:
        log_dir = args.log_dir or DEFAULT_LOG_DIR
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
        log_path = _selected_log(log_dir)
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(line.rstrip() for line in build_report(log_path).splitlines()) + "\n"
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
