from decimal import Decimal

TIER_1_GBV = 500_000
TIER_2_GBV = 100_000
TIER_2_EMPLOYEES = 150
MIN_GBV_MONTHS = 3


def initial_tier(annual_gbv: Decimal, employee_count: int | None) -> int:
    """Apply the Section 1 lead thresholds in priority order."""
    if annual_gbv >= TIER_1_GBV:
        return 1
    if annual_gbv >= TIER_2_GBV or (employee_count is not None and employee_count >= TIER_2_EMPLOYEES):
        return 2
    return 3


def possible_eligible_tiers(
    gbv: Decimal | None, months: int, employee_count: int | None
) -> set[int]:
    """Enumerate tiers consistent with known Section 1 eligibility inputs."""
    if gbv is None or months < MIN_GBV_MONTHS:
        gbv_tiers = {1, 2, 3}
    elif gbv * 12 >= TIER_1_GBV * months:
        gbv_tiers = {1}
    elif gbv * 12 >= TIER_2_GBV * months:
        gbv_tiers = {2}
    else:
        gbv_tiers = {3}

    if employee_count is None and 3 in gbv_tiers:
        gbv_tiers.add(2)
    elif employee_count is not None and employee_count >= TIER_2_EMPLOYEES and 3 in gbv_tiers:
        gbv_tiers.remove(3)
        gbv_tiers.add(2)
    return gbv_tiers
