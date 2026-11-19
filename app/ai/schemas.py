from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class StructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Confidence = Literal["high", "medium", "low", "insufficient"]
Level = Literal["low", "medium", "high", "unknown"]
def _integer_only(value):
    if type(value) is not int:
        raise ValueError("Expected an integer.")
    return value


Tier = Annotated[Literal[1, 2, 3], BeforeValidator(_integer_only)]


class Signal(StrEnum):
    COMPANY_SIZE = "company_size"
    EXTERNAL_ACTIVITY = "external_business_activity"
    GEOGRAPHY = "geographic_distribution"
    EVENTS = "corporate_events"
    GROWTH = "business_growth_and_expansion"
    COMPLEXITY = "operational_complexity"


class ResearchSource(StructuredOutput):
    url: str
    title: str | None
    publication_date: str | None = Field(
        description="Source publication date as YYYY-MM-DD. Use null when the full date is unknown. "
        "Do not substitute a year, month, fiscal period or access date."
    )


class SignalRange(StructuredOutput):
    minimum: float
    maximum: float
    unit: str


class SignalEvidence(StructuredOutput):
    value: str | int | float | None
    estimated_range: SignalRange | None = Field(description="Provide either a value or a bounded range, with the other field null.")
    sources: list[ResearchSource] = Field(
        description="Cite at least one actual supporting source for every direct evidence item. "
        "An empty list is only valid for indirect evidence with nonblank supporting context. "
        "If no evidence can be supported, report the signal as missing. Never invent sources."
    )
    reliability: Literal["high", "medium", "low"]
    evidence_type: Literal["direct", "indirect"]
    context: str | None


class ResearchSignal(StructuredOutput):
    signal: Signal
    status: Literal["found", "estimated", "conflicting", "missing"] = Field(
        description="An estimated signal needs indirect evidence or a supported range. A conflicting signal needs at least two distinct observations."
    )
    evidence: list[SignalEvidence] = Field(description="Empty for missing signals. Preserve competing observations for conflicts.")
    information_gap: str | None = Field(description="Nonblank for missing or conflicting signals. Otherwise null when there is no gap.")


class ResearchOutput(StructuredOutput):
    signals: list[ResearchSignal]


class Prediction(StructuredOutput):
    level: Level
    confidence: Confidence
    supporting_signals: list[Signal]
    rationale: str
    information_gaps: list[str]


class IntensityRating(StructuredOutput):
    signal: Signal
    rating: Annotated[Literal[0, 1, 2], BeforeValidator(_integer_only)]


class TravelPrediction(StructuredOutput):
    level: Literal["low", "medium", "high", "very_high", "unknown"]
    confidence: Confidence
    supporting_signals: list[Signal]
    rationale: str
    information_gaps: list[str]
    company_scale: Literal["small", "medium", "large", "very_large"] | None
    intensity_ratings: list[IntensityRating]


class LeadPredictionOutput(StructuredOutput):
    travel_spend_potential: TravelPrediction
    growth_potential: Prediction
    expected_support_needs: Prediction


class ActivityEvidence(StrEnum):
    MONTHS = "months_since_signup"
    GBV = "total_gbv"
    BOOKINGS = "avg_monthly_bookings"
    EMPLOYEES = "employee_count"
    ACTIVE_USERS = "avg_monthly_active_users"
    SUPPORT = "avg_monthly_support_requests"
    TRAVEL_TREND = "travel_spend_trend"
    SIZE_TREND = "company_size_trend"
    ADOPTION_TREND = "product_adoption_trend"
    ANNUALIZED_GBV = "annualized_gbv"
    ADOPTION_RATE = "adoption_rate_percent"
    SUPPORT_RATE = "monthly_support_per_100_active_users"


class ActivityFinding(StructuredOutput):
    evidence: list[ActivityEvidence] = Field(
        description="References to known submitted observations or backend metrics. Do not repeat or recalculate the measurements."
    )
    interpretation: str


class ProductAssessmentOutput(StructuredOutput):
    confidence: Literal["medium", "low", "insufficient"]
    material_eligibility_uncertainty: bool = Field(
        description="True if unresolved eligibility evidence or snapshot reliability could change a tier decision. "
        "Missing adoption measurements or adoption trends alone do not count."
    )
    findings: list[ActivityFinding]
    rationale: str
    information_gaps: list[str]


class RecommendationOutput(StructuredOutput):
    outcome: Literal["retain", "upgrade", "downgrade", "undetermined"]
    recommended_tier: Tier | None
    confidence: Confidence
    key_findings: list[str]
    supporting_evidence: list[str] = Field(
        description="Use structured input keys for the cited evidence. For product assessments, use evidence keys cited by their findings."
    )
    rationale: str
    follow_up_actions: list[str]
    information_gaps: list[str]


def evaluation_output_fields(
    assessment: LeadPredictionOutput | ProductAssessmentOutput,
    recommendation: RecommendationOutput,
) -> dict:
    """Map validated outputs without generating any persistence metadata."""
    return {
        "predictions": {
            **assessment.model_dump(mode="json"),
            **({"tier_alignment": recommendation.tier_alignment} if isinstance(recommendation, ComputedRecommendation) else {}),
            "recommendation_evidence": {
                "outcome": recommendation.outcome,
                "key_findings": recommendation.key_findings,
                "supporting_evidence": recommendation.supporting_evidence,
            },
        },
        "recommended_tier": recommendation.recommended_tier,
        "confidence": recommendation.confidence,
        "rationale": recommendation.rationale,
        "follow_up_actions": recommendation.follow_up_actions,
        "information_gaps": recommendation.information_gaps,
    }


class EmployeeBounds(StructuredOutput):
    lower: int | None
    upper: int | None
    scope: Literal["evaluated_population", "verified_subset", "other", "unknown"]
    basis: Literal["employees", "other", "unknown"]
    as_of: str | None = Field(description="Verified observation date YYYY-MM-DD, distinct from publication date.")
    submitted_count_resolution: Literal["verified_replacement", "unresolved"] | None = Field(
        description="Verified replacement requires explicit source evidence about the submitted count's scope or obsolescence."
    )


class ScopedSignalEvidence(SignalEvidence):
    applicable: bool = Field(description="Relevant to the evaluated entity and the claim represented by this signal.")
    approximation: bool = Field(description="True for a reported approximate value or an inference.")
    treatment: Literal["usable", "preferred", "weaker_or_outdated", "unresolved"]
    employee_bounds: EmployeeBounds | None = Field(description="Only for company size. Inclusive integer bounds, null on an open side.")
    observation_key: str | None = Field(description="Same key for repeated reporting of one underlying observation.")
    observation_period: str | None = Field(description="Canonical comparable period for directional growth evidence, or null.")
    growth_direction: Literal["increasing", "stable", "decreasing"] | None


class ScopedResearchSignal(ResearchSignal):
    status: Literal["found", "estimated", "conflicting", "missing"] = Field(
        description="Estimated includes explicitly sourced approximate values. Conflicting preserves distinct unresolved observations."
    )
    evidence: list[ScopedSignalEvidence]


class ScopedResearchOutput(ResearchOutput):
    signals: list[ScopedResearchSignal]


class EvidenceRating(IntensityRating):
    evidence_indices: list[int] = Field(description="Zero-based indices of exact supporting observations within this signal.")
    material_uncertainty: bool = Field(description="True if unresolved evidence could materially change this rating.")


class TravelAssessment(StructuredOutput):
    confidence: Confidence
    supporting_signals: list[Signal]
    rationale: str
    information_gaps: list[str]
    intensity_ratings: list[EvidenceRating]


class LeadAssessmentOutput(StructuredOutput):
    travel_spend_potential: TravelAssessment
    growth_potential: Prediction
    expected_support_needs: Prediction


class ComputedTravelPrediction(TravelPrediction):
    admitted_weight: int
    possible_weight: int
    weighted_score: str | None
    intensity: Literal["low", "medium", "high"] | None


class ComputedLeadPredictionOutput(LeadPredictionOutput):
    travel_spend_potential: ComputedTravelPrediction


class ComputedRecommendation(RecommendationOutput):
    tier_alignment: Literal["consistent", "potential_underclassification", "potential_overclassification", "undetermined"] | None
