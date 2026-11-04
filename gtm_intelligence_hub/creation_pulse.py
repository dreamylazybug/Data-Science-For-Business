"""Creation Pulse: point-in-time creation events and quarter pacing.

All bundled records are synthetic. The engine uses creation-date amounts only;
subsequent amount changes are deliberately excluded from gross creation.
"""
from datetime import date, timedelta
from collections import defaultdict
import random

def quarter_start(day):
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)

def next_quarter(day):
    start = quarter_start(day)
    return date(start.year + (start.month == 10), (start.month + 2) % 12 + 1, 1)

def selling_days(start, end):
    """Weekdays in inclusive date interval; holiday calendar not yet supplied."""
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end-start).days+1)
            if (start + timedelta(days=i)).weekday() < 5]

def demo_events(as_of=date(2026, 11, 4)):
    """Reproducible fictional events across the current and prior-year quarters."""
    rng = random.Random(1701)
    products = ["Cloud Security", "Identity Security", "Data Protection", "Network Security"]
    segments = ["Commercial", "Enterprise", "SMB"]
    records = []
    for year, multiplier in [(as_of.year-1, 0.92), (as_of.year, 1.0)]:
        start = quarter_start(as_of).replace(year=year)
        days = selling_days(start, next_quarter(start)-timedelta(days=1))
        for idx, day in enumerate(days):
            for j in range(rng.randint(1, 4)):
                records.append({"opportunity_id":f"SYN-{year}-{idx}-{j}",
                    "created_date":day.isoformat(), "created_amount":round(rng.randint(45, 380)*1000*multiplier, 2),
                    "product":products[(idx+j)%len(products)], "segment":segments[(idx+2*j)%len(segments)],
                    "motion": ["New logo", "Cross-sell", "Upsell"][(idx+j)%3],
                    "synthetic":True})
    return records

def pulse(events, as_of, quarterly_target):
    """Gross creation and equivalent selling-day prior-year comparison.

    Future-dated events are excluded. The prior year is aligned by selling-day
    ordinal rather than calendar date; holidays are not modeled.
    """
    start = quarter_start(as_of)
    end = next_quarter(as_of)-timedelta(days=1)
    cutoff = min(as_of, end)
    elapsed = selling_days(start, cutoff)
    full = selling_days(start, end)
    prior_start = start.replace(year=start.year-1)
    prior_full = selling_days(prior_start, next_quarter(prior_start)-timedelta(days=1))
    prior_elapsed = prior_full[:len(elapsed)]
    current_days = set(elapsed)
    prior_days = set(prior_elapsed)
    by_day = defaultdict(float)
    prior_by_day = defaultdict(float)
    products = defaultdict(float)
    segments = defaultdict(float)
    motions = defaultdict(float)
    count = 0
    for event in events:
        day = date.fromisoformat(str(event["created_date"]))
        amount = float(event["created_amount"])
        if amount < 0:
            raise ValueError("Created amount cannot be negative")
        if day in current_days:
            by_day[day] += amount
            products[event.get("product", "Unknown")] += amount
            segments[event.get("segment", "Unknown")] += amount
            motions[event.get("motion", "Unknown")] += amount
            count += 1
        elif day in prior_days:
            prior_by_day[day] += amount
    qtd = sum(by_day.values())
    prior_qtd = sum(prior_by_day.values())
    elapsed_count = len(elapsed)
    full_count = len(full)
    pace_target = quarterly_target * elapsed_count / full_count if full_count else 0
    # Simple pace estimate only, not a statistically calibrated forecast.
    projected = qtd * full_count / elapsed_count if elapsed_count else 0
    series = []
    current_running = prior_running = 0.0
    for i, day in enumerate(elapsed):
        current_running += by_day[day]
        if i < len(prior_elapsed):
            prior_running += prior_by_day[prior_elapsed[i]]
        series.append({"selling_day":i+1,"current":round(current_running,2),
                       "prior":round(prior_running,2),
                       "required":round(quarterly_target*(i+1)/full_count,2)})
    return {"qtd":qtd, "prior_qtd":prior_qtd,
            "yoy_pct":((qtd/prior_qtd)-1)*100 if prior_qtd else None,
            "required_to_date":pace_target, "pace_gap":qtd-pace_target,
            "projected":projected, "target":quarterly_target,
            "remaining_to_target":max(0,quarterly_target-qtd),
            "elapsed_selling_days":elapsed_count,"quarter_selling_days":full_count,
            "opportunity_count":count,"by_product":dict(products),
            "by_segment":dict(segments),"by_motion":dict(motions),
            "series":series, "quarter_start":start.isoformat(),
            "as_of":cutoff.isoformat()}

def separate_horizons():
    """Model contracts; no conversion predictions until calibrated inputs exist."""
    return {"current_quarter_bookings":{"status":"not_modelled",
             "requires":["open pipeline snapshots","close dates","stage history","conversion by cohort"]},
            "future_quarter_health":{"status":"not_modelled",
             "requires":["creation cohorts","historical time-to-close","win/loss outcomes","future bookings targets"]}}
