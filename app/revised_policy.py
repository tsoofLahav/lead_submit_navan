"""Deterministic interpretation for the revised policy's structured evidence."""

from datetime import date
from decimal import Decimal

from . import validation
from .ai.schemas import (
    ComputedLeadPredictionOutput, ComputedTravelPrediction, LeadAssessmentOutput, ScopedResearchOutput,
    ScopedSignalEvidence, Signal,
)
from .tier_eligibility import MIN_GBV_MONTHS, TIER_1_GBV, TIER_2_EMPLOYEES, TIER_2_GBV

ISOLATABLE = {
    "research.value_or_range", "research.blank_value", "research.nonfinite_value",
    "research.invalid_range", "research.unsourced_evidence", "research.invalid_source_url",
}
RANK = {"low": 0, "medium": 1, "high": 2}
BANDS = ("small", "medium", "large", "very_large")


def _gap(existing: str | None, notice: str) -> str:
    return " ".join(part for part in (existing, notice) if part)


def normalize_research(output: ScopedResearchOutput) -> ScopedResearchOutput:
    """Repair local defects without creating or changing claims or source identities."""
    findings = []
    for signal_index, finding in enumerate(output.signals):
        if finding.status == "missing" and finding.evidence:
            findings.append(finding)  # Contradictory status needs explicit correction.
            continue
        evidence = []
        notes = []
        for index, original in enumerate(finding.evidence):
            sources = []
            for source in original.sources:
                if source.publication_date is not None and not _iso_date(source.publication_date):
                    source = source.model_copy(update={"publication_date": None})
                    notes.append("An unverified publication date was omitted.")
                if source not in sources:
                    sources.append(source)
            item = original.model_copy(update={"sources": sources})
            try:
                validation.validate_research_evidence(item, f"signals[{signal_index}].evidence[{index}]")
            except validation.AIOutputValidationError as error:
                if error.diagnostic.rule_id in ISOLATABLE:
                    notes.append(f"An unsupported observation was excluded ({error.diagnostic.rule_id}).")
                    continue
                # Keep ambiguous metadata for bounded correction, alongside independent signal errors.
            if item not in evidence:
                evidence.append(item)
        status, gap = finding.status, finding.information_gap
        if not evidence:
            status = "missing"
            if notes or not validation._nonempty(gap):
                gap = _gap(gap, "No supported observation remains." if notes else "No supported observation was reported.")
        elif (
            finding.status == "conflicting"
            and len({(str(item.value), repr(item.estimated_range)) for item in evidence}) < 2
            and len(evidence) < len(finding.evidence)
        ):
            status = ("estimated" if all(item.approximation or item.evidence_type == "indirect"
                                         or item.estimated_range is not None for item in evidence) else "found")
        elif status == "found" and all(item.approximation or item.evidence_type == "indirect"
                                      or item.estimated_range is not None for item in evidence):
            status = "estimated"
        for note in dict.fromkeys(notes):
            gap = _gap(gap, note)
        findings.append(finding.model_copy(update={"evidence": evidence, "status": status, "information_gap": gap}))
    result = output.model_copy(update={"signals": findings})
    validation.validate_research(result)
    return result


def _iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _quality(item: ScopedSignalEvidence) -> bool:
    return item.applicable and RANK[item.reliability] >= RANK["medium"]


def _admissible(item: ScopedSignalEvidence) -> bool:
    return _quality(item) and item.treatment in {"usable", "preferred"}


def _bounds(item: ScopedSignalEvidence) -> tuple[int | None, int | None] | None:
    metric = item.employee_bounds
    if not _quality(item) or metric is None or metric.basis != "employees":
        return None
    lower, upper = metric.lower, metric.upper
    if (
        (lower is None and upper is None) or (lower is not None and lower < 1)
        or (upper is not None and upper < 1)
        or (lower is not None and upper is not None and lower > upper)
    ):
        return None
    if type(item.value) in (int, float) and (
        (lower is not None and item.value < lower) or (upper is not None and item.value > upper)
    ):
        return None
    if item.estimated_range is not None:
        interval = item.estimated_range
        if interval.unit.strip().casefold() in {"employee", "employees", "headcount"} and (
            lower != interval.minimum or upper != interval.maximum
        ):
            return None
    if metric.scope == "evaluated_population":
        return lower, upper
    if metric.scope == "verified_subset" and lower is not None:
        return lower, None
    return None


def _employee_candidates(research: ScopedResearchOutput) -> list[tuple[ScopedSignalEvidence, int | None, int | None]]:
    finding = next(row for row in research.signals if row.signal == Signal.COMPANY_SIZE)
    candidates = [(item, *bounds) for item in finding.evidence if (bounds := _bounds(item)) is not None]

    def superseded(item):
        if item.treatment != "weaker_or_outdated":
            return False
        for preferred, _, _ in candidates:
            if preferred.treatment != "preferred" or not preferred.sources:
                continue
            if preferred.employee_bounds.scope != item.employee_bounds.scope:
                continue
            better = RANK[preferred.reliability] > RANK[item.reliability]
            earlier = _iso_date(item.employee_bounds.as_of)
            later = _iso_date(preferred.employee_bounds.as_of)
            if better or (earlier and later and later > earlier and RANK[preferred.reliability] >= RANK[item.reliability]):
                return True
        return False

    return [row for row in candidates if not superseded(row[0])]


def _band(value: int | None, *, upper: bool) -> str:
    if value is None:
        return BANDS[-1] if upper else BANDS[0]
    return BANDS[0 if value < 50 else 1 if value < 150 else 2 if value < 500 else 3]


def company_scale(research: ScopedResearchOutput) -> str | None:
    bands = {_band(bound, upper=is_upper) for _, lower, upper in _employee_candidates(research)
             for bound, is_upper in ((lower, False), (upper, True))}
    return next(iter(bands)) if len(bands) == 1 else None


def _relevant_for_refs(findings, references):
    return {signal: [item for item in findings[signal].evidence if _admissible(item)] for signal in references}


def _unsupported(prediction, gap):
    gaps = list(dict.fromkeys([*prediction.information_gaps, gap]))
    return prediction.model_copy(update={
        "level": "unknown", "confidence": "insufficient", "information_gaps": gaps,
        "rationale": gap,  # Do not keep prose asserting the discarded result.
    })


def _partial_confidence(prediction, material):
    if material and prediction.confidence == "high":
        gap = "Material evidence disagreement limits confidence."
        return prediction.model_copy(update={
            "confidence": "medium", "information_gaps": list(dict.fromkeys([*prediction.information_gaps, gap])),
            "rationale": gap + " " + prediction.rationale,
        })
    return prediction


def compute_predictions(
    draft: LeadAssessmentOutput, research: ScopedResearchOutput,
) -> ComputedLeadPredictionOutput:
    findings = {item.signal: item for item in research.signals}
    travel = draft.travel_spend_potential
    growth, support = draft.growth_potential, draft.expected_support_needs
    def supported_references(item, field):
        validation.require(
            validation._nonempty(item.rationale), "Prediction must include a rationale.",
            "prediction.rationale", "3.5", f"{field}.rationale", "blank",
        )
        references = list(dict.fromkeys(item.supporting_signals))
        supported = [key for key in references if findings[key].evidence]
        gaps = item.information_gaps
        if supported != references:
            gaps = [*gaps, "A reference to unavailable evidence was excluded."]
        return item.model_copy(update={"supporting_signals": supported, "information_gaps": gaps})

    travel = supported_references(travel, "travel_spend_potential")
    growth = supported_references(growth, "growth_potential")
    support = supported_references(support, "expected_support_needs")

    scale = company_scale(research)
    admitted = {}
    rating_signals = set()
    for index, rating in enumerate(travel.intensity_ratings):
        evidence = findings[rating.signal].evidence
        if not evidence and rating.signal in validation.TRAVEL_WEIGHTS:
            continue
        validation.require(
            rating.signal in validation.TRAVEL_WEIGHTS and rating.signal not in rating_signals
            and rating.signal in travel.supporting_signals
            and all(0 <= pointer < len(evidence) for pointer in rating.evidence_indices),
            "Travel rating must reference distinct configured signals and valid evidence indices.",
            "prediction.travel_rating_references", "3.2", f"travel_spend_potential.intensity_ratings[{index}]",
            {"signal": rating.signal, "index_count": len(rating.evidence_indices)},
        )
        rating_signals.add(rating.signal)
        if any(_admissible(evidence[i]) for i in rating.evidence_indices):
            admitted[rating.signal] = rating
    weight = sum(validation.TRAVEL_WEIGHTS[key] for key in admitted)
    weighted = sum(validation.TRAVEL_WEIGHTS[key] * item.rating for key, item in admitted.items())
    level, intensity, score = "unknown", None, None
    core_count = len(admitted.keys() & validation.TRAVEL_CORE)
    if core_count and scale is not None:
        score = str(Decimal(weighted) / weight)
        intensity = "low" if 100 * weighted < 67 * weight else "high" if 100 * weighted > 133 * weight else "medium"
        level = validation.TRAVEL_MATRIX[scale][("low", "medium", "high").index(intensity)]
    travel_gaps = list(travel.information_gaps)
    if scale is None:
        travel_gaps.append("The applicable employee evidence does not establish one company scale.")
    if not core_count:
        travel_gaps.append("No supported core travel signal was established.")
    material = any(item.material_uncertainty for item in travel.intensity_ratings) or _material_disagreement(
        findings, admitted.keys() & validation.TRAVEL_CORE,
    )
    travel_confidence = travel.confidence
    if level == "unknown":
        travel_confidence = "insufficient"
    elif travel_confidence == "insufficient":
        level = "unknown"
        travel_gaps.append("The assessment reports insufficient confidence in a travel conclusion.")
    elif (core_count == 1 or material) and travel_confidence == "high":
        travel_confidence = "medium"
        travel_gaps.append("High confidence requires consistent evidence for both core travel signals.")
    travel_output = ComputedTravelPrediction(
        level=level, confidence=travel_confidence, supporting_signals=travel.supporting_signals,
        rationale=travel.rationale if level != "unknown" else "The available evidence does not establish travel potential.",
        information_gaps=list(dict.fromkeys(travel_gaps)), company_scale=scale,
        intensity_ratings=[admitted[rating.signal] for rating in travel.intensity_ratings if rating.signal in admitted],
        admitted_weight=weight, possible_weight=sum(validation.TRAVEL_WEIGHTS.values()),
        weighted_score=score, intensity=intensity,
    )

    growth_evidence = _relevant_for_refs(findings, growth.supporting_signals)
    direction = {"low": {"stable", "decreasing"}, "medium": {"increasing"}, "high": {"increasing"}}
    allowed_directions = direction.get(growth.level, set())
    primary = any(item.growth_direction in allowed_directions for key in validation.PRIMARY_GROWTH
                  for item in growth_evidence.get(key, ()))
    secondary = {}
    for key in validation.SECONDARY_GROWTH:
        for item in growth_evidence.get(key, ()):
            if item.observation_period and item.observation_key and item.growth_direction in allowed_directions:
                secondary.setdefault((item.observation_period, item.growth_direction), {}).setdefault(
                    item.observation_key, set(),
                ).add(key)
    secondary_ok = any(len(observations) >= 2 for observations in secondary.values())
    if growth.level == "unknown" or growth.confidence == "insufficient" or not (primary or secondary_ok):
        growth = _unsupported(growth, "Available evidence does not establish the proposed growth direction and strength.")
    else:
        growth = _partial_confidence(growth, _material_disagreement(findings, growth.supporting_signals))

    support_evidence = _relevant_for_refs(findings, support.supporting_signals)
    if (
        support.level == "unknown" or support.confidence == "insufficient"
        or not (support_evidence.get(Signal.COMPLEXITY) or (
            support_evidence.get(Signal.COMPANY_SIZE) and support_evidence.get(Signal.GEOGRAPHY)
        ))
    ):
        support = _unsupported(support, "Relevant, reliable organizational complexity and contextual evidence are required.")
    else:
        support = _partial_confidence(support, _material_disagreement(findings, support.supporting_signals))
    return ComputedLeadPredictionOutput(
        travel_spend_potential=travel_output, growth_potential=growth, expected_support_needs=support,
    )


def _material_disagreement(findings, references):
    return any(item.applicable and item.treatment == "unresolved" for key in references for item in findings[key].evidence)


def tier_alignment(level: str, tier: int) -> str:
    if level == "unknown":
        return "undetermined"
    if level in validation.UPGRADE_TRAVEL_LEVELS[tier]:
        return "potential_underclassification"
    if level in validation.DOWNGRADE_TRAVEL_LEVELS[tier]:
        return "potential_overclassification"
    return "consistent"


def lead_targets(lead, tier: int, research: ScopedResearchOutput, predictions) -> dict[int, list[str]]:
    """Forecast suitability guidance for the model, independent of approval."""
    travel = predictions.travel_spend_potential
    growth = predictions.growth_potential
    support = predictions.expected_support_needs
    sufficient = {"low", "medium", "high"}
    basis = {
        "low": (3,), "medium": (2,), "high": (1, 2), "very_high": (1,),
    }
    targets = {}
    if travel.confidence in sufficient:
        for target in basis.get(travel.level, ()):
            targets[target] = ["travel_spend_potential"]

    strong_growth = growth.level == "high" and growth.confidence in sufficient
    strong_support = support.level == "high" and support.confidence in sufficient
    refinements = []
    if strong_growth:
        refinements.append("growth_potential")
    if strong_support:
        refinements.append("expected_support_needs")
    if targets and refinements:
        refinements.sort(key=lambda key: getattr(predictions, key).confidence != "high")
        next_higher_touch = min(targets) - 1
        if next_higher_touch >= 1:
            targets[next_higher_touch] = ["travel_spend_potential", refinements[0]]
        if travel.level == "low" and len(refinements) == 2:
            targets[1] = ["travel_spend_potential", *refinements]
    elif travel.level == "unknown" and strong_growth and support.level in {"medium", "high"} and support.confidence in sufficient:
        targets[1 if strong_support else 2] = ["growth_potential", "expected_support_needs"]

    candidates = _employee_candidates(research)
    if tier == 3 and candidates and all(
        item.sources and lower is not None and lower >= TIER_2_EMPLOYEES
        and not (item.approximation and lower == upper)
        for item, lower, upper in candidates
    ):
        targets.setdefault(2, ["company_size"])
    return targets


def customer_tier_guidance(activity) -> dict:
    """Describe the current measurements without resolving unknown thresholds."""
    gbv = activity.total_gbv
    months = activity.months_since_signup
    headcount = activity.employee_count
    tier_1_gbv = None if gbv is None else gbv * 12 >= TIER_1_GBV * months
    tier_2_gbv = None if gbv is None else gbv * 12 >= TIER_2_GBV * months
    tier_2_headcount = None if headcount is None else headcount >= TIER_2_EMPLOYEES
    return {
        "indicative_run_rate_meets_tier_1_threshold": tier_1_gbv,
        "indicative_run_rate_meets_tier_2_threshold": tier_2_gbv,
        "headcount_meets_tier_2_threshold": tier_2_headcount,
        "short_observation_period": months < MIN_GBV_MONTHS,
    }
