from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .tier_eligibility import MIN_GBV_MONTHS, possible_eligible_tiers
from .validation import CustomerActivityInput, TREND_FIELDS, validate_activity

SUPPORT_TRIGGER_RATE = 5

NO_TRIGGER_RESPONSE = (
    "Customer activity has been successfully updated. No re-evaluation triggers were detected "
    "based on the available information. The current tier remains unchanged."
)


class GateStatus(StrEnum):
    TRIGGERED = "TRIGGERED"
    NO_TRIGGER = "NO_TRIGGER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class ActivityMetrics:
    annualized_gbv: Decimal | None
    adoption_rate_percent: Decimal | None
    monthly_support_per_100_active_users: Decimal | None


@dataclass(frozen=True)
class TriggerResult:
    status: GateStatus
    triggered_conditions: tuple[str, ...]
    metrics: ActivityMetrics
    information_gaps: tuple[str, ...]
    explanation: str


class ReevaluationGate:
    @staticmethod
    def assess(activity: CustomerActivityInput) -> TriggerResult:
        """Assess the submitted snapshot using active policy Sections 6.1 and 6.2."""
        activity = validate_activity(activity)
        gaps = []
        triggered = []
        active_users = activity.avg_monthly_active_users
        employees = activity.employee_count
        support_requests = activity.avg_monthly_support_requests
        annualized = adoption = support = None

        if activity.total_gbv is None:
            gaps.append("Total GBV since signup is unavailable.")
        else:
            annualized = activity.total_gbv * 12 / activity.months_since_signup
        if activity.months_since_signup < MIN_GBV_MONTHS:
            gaps.append("GBV-based tier eligibility requires at least three completed months.")

        if employees is None:
            gaps.append("Current employee count is unavailable.")
        if active_users is None:
            gaps.append("Average monthly active users are unavailable.")
        if employees is not None and active_users is not None:
            adoption = 100 * active_users / employees
            if adoption > 100:
                gaps.append("Estimated adoption exceeds 100% using historical active users and current headcount.")

        if support_requests is None:
            gaps.append("Average monthly support requests are unavailable.")
        if active_users == 0:
            gaps.append("The monthly support rate requires a positive active-user count.")
        if support_requests is not None and active_users is not None and active_users > 0:
            support = 100 * support_requests / active_users
            if 100 * support_requests >= SUPPORT_TRIGGER_RATE * active_users:
                triggered.append("high_monthly_support_rate")

        eligible_tiers = possible_eligible_tiers(activity.total_gbv, activity.months_since_signup, employees)
        mismatch = activity.current_tier not in eligible_tiers
        if mismatch:
            triggered.append("potential_tier_mismatch")
        eligibility_assessable = mismatch or len(eligible_tiers) == 1

        known_trend = False
        for field in TREND_FIELDS:
            trend = getattr(activity, field)
            if trend == "unknown":
                gaps.append(f"{field.replace('_', ' ').capitalize()} is unknown.")
            else:
                known_trend = True
                if trend in ("increasing", "decreasing"):
                    triggered.append(f"{field}_{trend}")

        if triggered:
            status = GateStatus.TRIGGERED
            explanation = "Customer activity has been successfully updated. AI re-evaluation is required."
        elif eligibility_assessable or known_trend or support is not None:
            status = GateStatus.NO_TRIGGER
            explanation = NO_TRIGGER_RESPONSE
        else:
            status = GateStatus.INSUFFICIENT_DATA
            explanation = (
                "Customer activity has been successfully updated. Re-evaluation eligibility could not be "
                f"assessed. {' '.join(gaps)}"
            )

        return TriggerResult(
            status,
            tuple(triggered),
            ActivityMetrics(annualized, adoption, support),
            tuple(gaps),
            explanation,
        )
