from dataclasses import replace
from decimal import Decimal as D

import pytest

from app.reevaluation_gate import GateStatus, NO_TRIGGER_RESPONSE, ReevaluationGate
from app.tier_eligibility import initial_tier
from app.validation import CustomerActivityInput, LeadInput, ValidationError, validate_activity, validate_lead


@pytest.mark.parametrize(
    ("gbv", "employees", "expected"),
    [
        ("500000", None, 1), ("500000", 150, 1), ("499999.99", 150, 2),
        ("100000", None, 2), ("99999.99", 150, 2), ("99999.99", 149, 3),
        ("0", None, 3), ("0", 150, 2),
    ],
)
def test_initial_tier(gbv, employees, expected):
    assert initial_tier(D(gbv), employees) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("company_name", " \t "), ("company_name", "AC\x00ME"),
        ("company_domain", "https://example.com"), ("company_domain", "bad_.example"),
        ("submitted_annual_gbv", None), ("submitted_annual_gbv", D("NaN")),
        ("submitted_annual_gbv", D("0.001")), ("submitted_annual_gbv", D("10000000000000")),
        ("employee_count", 0), ("employee_count", True), ("employee_count", 2_147_483_648),
    ],
)
def test_invalid_lead(field, value):
    lead = LeadInput("Acme Inc", "example.com", D("0"))
    with pytest.raises(ValidationError, match=field):
        validate_lead(replace(lead, **{field: value}))


@pytest.fixture
def activity():
    return CustomerActivityInput("Acme Inc", "example.com", 3, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_tier", None), ("current_tier", 4), ("current_tier", True),
        ("months_since_signup", None), ("months_since_signup", 0),
        ("total_gbv", D("-1")), ("avg_monthly_bookings", D("-1")),
        ("employee_count", -1), ("avg_monthly_active_users", D("Infinity")),
        ("avg_monthly_support_requests", D("0.001")), ("travel_spend_trend", "flat"),
        ("company_size_trend", None), ("product_adoption_trend", "Increasing"),
    ],
)
def test_invalid_activity(activity, field, value):
    with pytest.raises(ValidationError, match=field):
        validate_activity(replace(activity, **{field: value}))


@pytest.mark.parametrize(
    ("months", "gbv", "employees", "tier", "status"),
    [
        (2, "900000", None, 3, GateStatus.INSUFFICIENT_DATA),
        (3, "125000", None, 2, GateStatus.TRIGGERED),
        (3, "124999.99", None, 2, GateStatus.NO_TRIGGER),
        (3, "25000", None, 3, GateStatus.TRIGGERED),
        (3, "24999.99", None, 3, GateStatus.INSUFFICIENT_DATA),
        (3, "24999.99", 149, 3, GateStatus.NO_TRIGGER),
        (3, "24999.99", 150, 3, GateStatus.TRIGGERED),
        (1, None, 150, 3, GateStatus.TRIGGERED),
        (3, None, 150, 1, GateStatus.INSUFFICIENT_DATA),
        (3, "0", None, 1, GateStatus.TRIGGERED),
    ],
)
def test_eligibility_boundaries(activity, months, gbv, employees, tier, status):
    result = ReevaluationGate.assess(
        replace(
            activity,
            months_since_signup=months,
            total_gbv=D(gbv) if gbv is not None else None,
            employee_count=employees,
            current_tier=tier,
        )
    )
    assert result.status is status
    if gbv is not None:
        assert result.metrics.annualized_gbv == D(gbv) * 12 / months


@pytest.mark.parametrize(("support", "status"), [("4.99", GateStatus.NO_TRIGGER), ("5", GateStatus.TRIGGERED)])
def test_support_and_adoption_metrics(activity, support, status):
    result = ReevaluationGate.assess(
        replace(activity, employee_count=200, avg_monthly_active_users=D("100"),
                avg_monthly_support_requests=D(support), current_tier=2)
    )
    assert result.status is status
    assert result.metrics.adoption_rate_percent == 50
    assert result.metrics.monthly_support_per_100_active_users == D(support)
    assert ("high_monthly_support_rate" in result.triggered_conditions) == (status is GateStatus.TRIGGERED)


def test_zero_and_unknown_values(activity):
    result = ReevaluationGate.assess(
        replace(activity, avg_monthly_active_users=D("0"), avg_monthly_support_requests=D("0"),
                avg_monthly_bookings=D("0"), employee_count=50)
    )
    assert result.status is GateStatus.INSUFFICIENT_DATA
    assert result.metrics.adoption_rate_percent == 0
    assert result.metrics.annualized_gbv is None
    assert result.metrics.monthly_support_per_100_active_users is None
    assert "positive active-user count" in result.explanation


@pytest.mark.parametrize("field", ["travel_spend_trend", "company_size_trend", "product_adoption_trend"])
@pytest.mark.parametrize("trend", ["increasing", "decreasing"])
def test_qualitative_triggers(activity, field, trend):
    result = ReevaluationGate.assess(replace(activity, **{field: trend}))
    assert result.status is GateStatus.TRIGGERED
    assert result.triggered_conditions == (f"{field}_{trend}",)


def test_stable_trend_is_sufficient_to_assess(activity):
    result = ReevaluationGate.assess(replace(activity, travel_spend_trend="stable"))
    assert result.status is GateStatus.NO_TRIGGER
    assert result.explanation == NO_TRIGGER_RESPONSE


def test_adoption_above_one_hundred_is_reported(activity):
    result = ReevaluationGate.assess(replace(activity, employee_count=50, avg_monthly_active_users=D("60")))
    assert result.metrics.adoption_rate_percent == 120
    assert any("exceeds 100%" in gap for gap in result.information_gaps)
