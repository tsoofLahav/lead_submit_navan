import math
import re
from datetime import date
from urllib.parse import urlsplit
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from .ai.schemas import (
    ActivityEvidence, LeadPredictionOutput, ProductAssessmentOutput, RecommendationOutput, ResearchOutput, Signal,
)
from .tier_eligibility import MIN_GBV_MONTHS, possible_eligible_tiers


Trend = Literal["increasing", "stable", "decreasing", "unknown"]
TIERS = (1, 2, 3)
TRENDS = ("increasing", "stable", "decreasing", "unknown")
TREND_FIELDS = ("travel_spend_trend", "company_size_trend", "product_adoption_trend")
DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.ASCII)


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class LeadInput:
    company_name: str
    company_domain: str
    submitted_annual_gbv: Decimal
    employee_count: int | None = None


@dataclass(frozen=True)
class CustomerActivityInput:
    company_name: str
    company_domain: str
    current_tier: int
    months_since_signup: int
    total_gbv: Decimal | None = None
    avg_monthly_bookings: Decimal | None = None
    employee_count: int | None = None
    avg_monthly_active_users: Decimal | None = None
    avg_monthly_support_requests: Decimal | None = None
    travel_spend_trend: Trend = "unknown"
    company_size_trend: Trend = "unknown"
    product_adoption_trend: Trend = "unknown"


def display_name(value: str) -> str:
    if not isinstance(value, str) or not value.split() or "\x00" in value:
        raise ValidationError("company_name must be a non-empty name.")
    return " ".join(value.split())


def company_name_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def domain_key(value: str) -> str:
    return value.strip().lower().removesuffix(".")


def validate_domain(value: str) -> str:
    if not isinstance(value, str):
        raise ValidationError("company_domain must be a DNS domain.")
    domain = domain_key(value)
    labels = domain.split(".")
    if (
        not value.isascii() or len(domain) > 253 or len(labels) < 2 or labels[-1].isdigit()
        or not all(DOMAIN_LABEL.fullmatch(label) for label in labels)
    ):
        raise ValidationError("company_domain must be a DNS domain without a scheme, port or path.")
    return domain


def validate_positive_integer(field: str, value: int | None) -> None:
    if type(value) is not int or not 1 <= value <= 2_147_483_647:
        raise ValidationError(f"{field} must be a positive PostgreSQL integer.")


def validate_amount(field: str, value: Decimal, precision: int) -> None:
    if (
        not isinstance(value, Decimal) or not value.is_finite()
        or not 0 <= value < Decimal(10) ** (precision - 2)
        or value != value.quantize(Decimal("0.01"))
    ):
        raise ValidationError(
            f"{field} must be a non-negative Decimal with at most two significant decimal places "
            f"and fewer than {precision - 1} integer digits."
        )


def validate_lead(lead: LeadInput) -> LeadInput:
    validate_amount("submitted_annual_gbv", lead.submitted_annual_gbv, 15)
    if lead.employee_count is not None:
        validate_positive_integer("employee_count", lead.employee_count)
    return replace(
        lead, company_name=display_name(lead.company_name), company_domain=validate_domain(lead.company_domain)
    )


def validate_activity(activity: CustomerActivityInput) -> CustomerActivityInput:
    if type(activity.current_tier) is not int or activity.current_tier not in TIERS:
        raise ValidationError("current_tier must be 1, 2 or 3.")
    validate_positive_integer("months_since_signup", activity.months_since_signup)
    if activity.employee_count is not None:
        validate_positive_integer("employee_count", activity.employee_count)
    for field, precision in (
        ("total_gbv", 15),
        ("avg_monthly_bookings", 12),
        ("avg_monthly_active_users", 12),
        ("avg_monthly_support_requests", 12),
    ):
        value = getattr(activity, field)
        if value is not None:
            validate_amount(field, value, precision)
    for field in TREND_FIELDS:
        if getattr(activity, field) not in TRENDS:
            raise ValidationError(f"{field} must be increasing, stable, decreasing or unknown.")
    return replace(
        activity,
        company_name=display_name(activity.company_name),
        company_domain=validate_domain(activity.company_domain),
    )


@dataclass(frozen=True)
class ValidationDiagnostic:
    rule_id: str
    policy_section: str
    field_path: str
    observed: object
    message: str


class AIOutputValidationError(ValueError):
    """An AI response violates a deterministic business constraint."""

    def __init__(
        self, message: str, diagnostic: ValidationDiagnostic,
        additional_diagnostics: tuple[ValidationDiagnostic, ...] = (),
    ):
        super().__init__(message)
        self.diagnostic = diagnostic
        self.additional_diagnostics = additional_diagnostics


TRAVEL_WEIGHTS = {Signal.EXTERNAL_ACTIVITY: 2, Signal.GEOGRAPHY: 2, Signal.EVENTS: 1}
TRAVEL_CORE = {Signal.EXTERNAL_ACTIVITY, Signal.GEOGRAPHY}
TRAVEL_MATRIX = {
    "small": ("low", "low", "medium"),
    "medium": ("low", "medium", "high"),
    "large": ("medium", "high", "high"),
    "very_large": ("high", "high", "very_high"),
}
UPGRADE_TRAVEL_LEVELS = {1: set(), 2: {"very_high"}, 3: {"medium", "high", "very_high"}}
DOWNGRADE_TRAVEL_LEVELS = {1: {"low", "medium"}, 2: {"low"}, 3: set()}
PRIMARY_GROWTH = {Signal.GROWTH}
SECONDARY_GROWTH = {Signal.EXTERNAL_ACTIVITY, Signal.GEOGRAPHY}


def require(
    condition: bool, message: str, rule_id: str, policy_section: str,
    field_path: str, observed: object,
) -> None:
    if not condition:
        raise AIOutputValidationError(
            message, ValidationDiagnostic(rule_id, policy_section, field_path, observed, message)
        )


def _nonempty(text: str | None) -> bool:
    return bool(text and text.strip())


def validate_research(output: ResearchOutput) -> None:
    names = [finding.signal for finding in output.signals]
    require(len(names) == len(Signal) and set(names) == set(Signal), "Every policy signal must appear exactly once.",
            "research.signal_coverage", "2.1", "signals",
            {"count": len(names), "missing": sorted(set(Signal) - set(names)),
             "duplicates": [signal for signal in Signal if names.count(signal) > 1]})
    diagnostics = []
    for index, finding in enumerate(output.signals):
        try:
            _validate_research_signal(finding, index)
        except AIOutputValidationError as error:
            diagnostics.append(error.diagnostic)
            if len(diagnostics) == 5:
                break
    if diagnostics:
        first, *additional = diagnostics
        raise AIOutputValidationError(first.message, first, tuple(additional))


def _validate_research_signal(finding, index: int) -> None:
    path = f"signals[{index}]"
    if finding.status == "missing":
        message = "Missing signals require an explicit gap and no evidence."
        require(not finding.evidence, message, "research.missing_evidence", "2.1, 2.4",
                f"{path}.evidence", {"count": len(finding.evidence)})
        require(_nonempty(finding.information_gap), message, "research.missing_gap", "2.4",
                f"{path}.information_gap", None if finding.information_gap is None else "blank")
        return
    require(bool(finding.evidence), "A non-missing signal requires evidence.",
            "research.evidence_required", "2.4", f"{path}.evidence", {"count": len(finding.evidence)})
    if finding.status == "conflicting":
        message = "Conflicts require competing evidence and an explicit gap."
        require(len(finding.evidence) >= 2, message, "research.conflict_evidence", "2.4, 4.1",
                f"{path}.evidence", {"count": len(finding.evidence)})
        require(_nonempty(finding.information_gap), message, "research.conflict_gap", "2.4, 4.1",
                f"{path}.information_gap", None if finding.information_gap is None else "blank")
        values = {(str(item.value), repr(item.estimated_range)) for item in finding.evidence}
        require(len(values) >= 2, "Conflicting evidence must preserve distinct findings.",
                "research.conflict_distinct_values", "4.1", f"{path}.evidence", {"distinct_values": len(values)})
    if finding.status == "estimated":
        require(any(item.evidence_type == "indirect" or item.estimated_range is not None
                    or (getattr(item, "approximation", False) and bool(item.sources))
                    for item in finding.evidence),
                "An estimate requires indirect evidence, a supported range or a sourced approximation.",
                "research.estimate_representation", "2.4", f"{path}.evidence",
                {"evidence_types": sorted({item.evidence_type for item in finding.evidence}),
                 "range_count": sum(item.estimated_range is not None for item in finding.evidence)})
    for evidence_index, item in enumerate(finding.evidence):
        evidence_path = f"{path}.evidence[{evidence_index}]"
        validate_research_evidence(item, evidence_path)

def validate_research_evidence(item, evidence_path: str) -> None:
    require((item.value is None) != (item.estimated_range is None), "Evidence needs a value or a range.",
            "research.value_or_range", "2.4", evidence_path,
            {"value_present": item.value is not None, "range_present": item.estimated_range is not None})
    if isinstance(item.value, str):
        require(_nonempty(item.value), "An empty value is not evidence.",
                "research.blank_value", "2.4", f"{evidence_path}.value", "blank")
    if isinstance(item.value, float):
        require(math.isfinite(item.value), "Numeric evidence must be finite.",
                "research.nonfinite_value", "2.4", f"{evidence_path}.value", "non-finite")
    if item.estimated_range is not None:
        bounds = item.estimated_range
        require(
            math.isfinite(bounds.minimum) and math.isfinite(bounds.maximum)
            and bounds.minimum <= bounds.maximum and _nonempty(bounds.unit),
            "Invalid estimated range.", "research.invalid_range", "2.4", f"{evidence_path}.estimated_range",
            {"bounds_finite": math.isfinite(bounds.minimum) and math.isfinite(bounds.maximum),
             "bounds_ordered": bounds.minimum <= bounds.maximum, "unit_present": _nonempty(bounds.unit)},
        )
    if not item.sources:
        require(item.evidence_type == "indirect" and _nonempty(item.context),
                "Evidence without sources must be indirect and include supporting context.",
                "research.unsourced_evidence", "2.3, 2.4", f"{evidence_path}.sources",
                {"count": 0, "evidence_type": item.evidence_type, "context_present": _nonempty(item.context)})
    urls = set()
    for source_index, source in enumerate(item.sources):
        source_path = f"{evidence_path}.sources[{source_index}]"
        try:
            parsed = urlsplit(source.url)
            valid_url = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username
        except ValueError:
            valid_url = False
        valid_date = True
        try:
            if source.publication_date is not None:
                date.fromisoformat(source.publication_date)
        except ValueError:
            valid_date = False
        message = "Invalid or duplicate source metadata."
        require(valid_url, message, "research.invalid_source_url", "2.4", f"{source_path}.url", "invalid URL")
        require(valid_date, message, "research.invalid_source_date", "2.4",
                f"{source_path}.publication_date", "invalid ISO date")
        require(source.url not in urls, message, "research.duplicate_source_url", "4.1",
                f"{source_path}.url", "duplicate URL")
        urls.add(source.url)

def validate_product_assessment(output: ProductAssessmentOutput, activity, metrics, triggers) -> None:
    available = {
        key for key in ActivityEvidence
        if getattr(metrics, key, getattr(activity, key, None)) not in (None, "unknown")
    }
    require(_nonempty(output.rationale), "A product assessment needs a rationale.",
                "assessment.rationale", "6.5", "rationale", "blank")
    require(triggers.metrics == metrics, "Assessment metrics must match the deterministic gate result.",
                "assessment.gate_metrics", "6.1–6.2", "context.derived_metrics", "does not match deterministic result")
    for index, finding in enumerate(output.findings):
        refs = set(finding.evidence)
        require(bool(refs) and refs <= available and len(refs) == len(finding.evidence), "Product finding references missing or duplicate observations.",
                "assessment.evidence_references", "6.1", f"findings[{index}].evidence", {"count": len(finding.evidence), "distinct_count": len(refs), "unavailable": sorted(refs - available)})
        require(_nonempty(finding.interpretation), "Product finding needs an interpretation.",
                "assessment.interpretation", "6.5", f"findings[{index}].interpretation", "blank")
    if not output.findings:
        require(output.confidence == "insufficient" and any(map(_nonempty, output.information_gaps)),
                "An empty assessment requires insufficient confidence and an information gap.",
                "assessment.empty_confidence_gap", "6.4", "findings", {"count": 0, "confidence": output.confidence, "gap_present": any(map(_nonempty, output.information_gaps))})
    if output.confidence == "medium":
        require(activity.months_since_signup >= MIN_GBV_MONTHS,
                "Medium snapshot confidence requires at least three completed months.",
                "assessment.medium_tenure", "6.4", "confidence", output.confidence)


def lead_recommendation_evidence(
    assessment: LeadPredictionOutput, research: ResearchOutput, lead: LeadInput,
) -> set[str]:
    """Return the canonical citation keys permitted by the existing lead validator."""
    available = {
        key for key in ("travel_spend_potential", "growth_potential", "expected_support_needs")
        if getattr(assessment, key).level != "unknown"
    } | {finding.signal.value for finding in research.signals if finding.status != "missing"}
    available.add("submitted_annual_gbv")
    if lead.employee_count is not None:
        available.add("employee_count")
    return available


def product_recommendation_evidence(assessment: ProductAssessmentOutput) -> set[str]:
    """Return citation keys used by the validated activity assessment."""
    return {key.value for finding in assessment.findings for key in finding.evidence}


def validate_recommendation(
    output: RecommendationOutput, *, current_tier: int,
    assessment: LeadPredictionOutput | ProductAssessmentOutput,
    lead=None, research: ResearchOutput | None = None, activity=None, metrics=None, triggers=None,
    permissible_targets: dict[int, list[str]] | None = None,
) -> None:
    target = output.recommended_tier
    require(type(current_tier) is int and current_tier in (1, 2, 3), "Invalid current tier.",
                "recommendation.current_tier", "1, 5.3", "context.current_tier", "invalid tier")
    require(target is None or (type(target) is int and target in (1, 2, 3)), "Invalid recommended tier.",
                "recommendation.target_tier", "1, 5.3", "recommended_tier", "invalid tier")
    direction_matches = {
        "retain": target == current_tier,
        "undetermined": target is None,
        "upgrade": target is not None and target < current_tier,
        "downgrade": target is not None and target > current_tier,
    }
    require(direction_matches[output.outcome], "Recommendation direction disagrees with the current and proposed tiers.",
                "recommendation.direction", "5.3", "outcome", {"outcome": output.outcome, "current_tier": current_tier, "recommended_tier": target})
    require(_nonempty(output.rationale), "A recommendation needs a rationale.",
                "recommendation.rationale", "5.3", "rationale", "blank")
    changes_tier = output.outcome in {"upgrade", "downgrade"}

    if isinstance(assessment, LeadPredictionOutput):
        require(lead is not None and research is not None, "Lead recommendation requires the validated lead and research.",
                "recommendation.lead_context", "5.2", "context", {"lead_present": lead is not None, "research_present": research is not None})
        available = lead_recommendation_evidence(assessment, research, lead)
        if permissible_targets is None:
            from .revised_policy import lead_targets
            permissible_targets = lead_targets(lead, current_tier, research, assessment)
        if output.outcome == "retain":
            require(current_tier in permissible_targets, "The current support model is not established. Use Undetermined.",
                    "recommendation.retain_unsupported", "5.3", "outcome", output.outcome)
        if changes_tier:
            require(output.confidence in {"medium", "high"}, "Lead tier changes require at least Medium confidence.",
                    "recommendation.lead_confidence", "5.2", "confidence", output.confidence)
            require(not getattr(output, "material_suitability_uncertainty", False),
                    "Material uncertainty blocks an actionable lead recommendation.",
                    "recommendation.lead_uncertainty", "5.2", "material_suitability_uncertainty", True)
    else:
        require(activity is not None and metrics is not None and triggers is not None,
                "Re-evaluation recommendation requires the current activity and gate result.",
                "recommendation.activity_context", "6.5", "context", {"activity_present": activity is not None, "metrics_present": metrics is not None, "gate_present": triggers is not None})
        require(current_tier == activity.current_tier, "Current tier must match the submitted activity.",
                "recommendation.activity_tier", "6.5", "context.current_tier", current_tier)
        validate_product_assessment(assessment, activity, metrics, triggers)
        available = product_recommendation_evidence(assessment)
        if permissible_targets is not None:
            available |= {key for keys in permissible_targets.values() for key in keys}
        require(output.confidence != "high", "Summary snapshots cannot establish high confidence.",
                "recommendation.snapshot_high_confidence", "6.4", "confidence", output.confidence)
        if output.confidence == "medium":
            require(assessment.confidence == "medium" and bool(output.supporting_evidence),
                    "Medium confidence requires at least three completed months and supporting evidence.",
                "recommendation.snapshot_medium_confidence", "6.4", "confidence", {"assessment": assessment.confidence, "evidence_count": len(output.supporting_evidence)})
        if changes_tier:
            require(output.confidence == "medium", "A snapshot tier change needs sufficient supporting evidence.",
                    "recommendation.snapshot_change_confidence", "6.4", "confidence", output.confidence)
        if output.outcome == "retain":
            eligible = possible_eligible_tiers(activity.total_gbv, activity.months_since_signup, activity.employee_count)
            require(eligible == {current_tier} and not assessment.material_eligibility_uncertainty,
                    "The snapshot does not establish that retaining the current tier is supported. Use Undetermined.",
                    "recommendation.retain_unsupported", "5.3, 6.3", "outcome", output.outcome)
        if permissible_targets is None:
            from .revised_policy import customer_targets
            permissible_targets = customer_targets(activity, assessment)
            available |= {key for keys in permissible_targets.values() for key in keys}

    references = set(output.supporting_evidence)
    require(references <= available and len(references) == len(output.supporting_evidence),
            "Recommendation references unavailable or duplicate evidence.",
                "recommendation.evidence_references", "5.3, 6.5", "supporting_evidence", {"count": len(output.supporting_evidence), "distinct_count": len(references), "unavailable_count": len(references - available)})
    if changes_tier:
        require(bool(references), "A proposed tier change requires supporting evidence.",
                "recommendation.missing_evidence", "5.3", "supporting_evidence", [])
        require(target in permissible_targets, "The backend did not establish this tier target.",
                "recommendation.eligibility", "5.2, 6.3", "recommended_tier", target)
        qualifying = set(permissible_targets[target])
        all_required = isinstance(assessment, LeadPredictionOutput) or output.outcome == "downgrade"
        cited = qualifying <= references if all_required else bool(references & qualifying)
        if isinstance(assessment, LeadPredictionOutput) and output.confidence == "high":
            prediction_keys = qualifying & {"travel_spend_potential", "growth_potential", "expected_support_needs"}
            require(all(getattr(assessment, key).confidence == "high" for key in prediction_keys),
                    "High recommendation confidence requires High confidence in its forecast basis.",
                    "recommendation.lead_confidence_basis", "5.2", "confidence", output.confidence)
        require(cited, "Cite qualifying evidence for the proposed tier.",
                "recommendation.qualifying_evidence", "5.2, 6.3", "supporting_evidence",
                {"qualifying_fields": permissible_targets[target], "qualifying_references": 0})
