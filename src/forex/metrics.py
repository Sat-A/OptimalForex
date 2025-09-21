def calculate_gbp(amount_usd: float, rate_usd_to_gbp: float, fees: dict) -> float:
    pct = float(fees.get("percent", 0.0))
    fixed = float(fees.get("fixed", 0.0))
    fee_amt = (amount_usd * pct / 100.0) + fixed
    net_usd = amount_usd - fee_amt
    return net_usd * float(rate_usd_to_gbp)

def regret_vs_hindsight(gbp: float, gbp_star: float) -> float:
    return gbp - gbp_star

def pct_of_perfect(gbp: float, gbp_star: float) -> float:
    """Return how much of the perfect-foresight outcome was achieved (0..1)."""
    return float(gbp / gbp_star) if gbp_star else 0.0

def regret_pct(gbp: float, gbp_star: float) -> float:
    """Relative regret vs. perfect foresight (negative is worse)."""
    return float((gbp - gbp_star) / gbp_star) if gbp_star else 0.0
