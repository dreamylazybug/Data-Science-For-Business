"""Explainable firmographic fit scoring."""
def parts(value):
    return {v.strip().lower() for v in str(value or "").replace("|", ",").split(",") if v.strip()}

def fit(entity, profile):
    weights = {"industry": 30, "segment": 20, "country": 20, "technologies": 20}
    matched = 0
    assessed = 0
    details = []
    for field, weight in weights.items():
        target = parts(profile.get(field, ""))
        if not target:
            continue
        actual = parts(entity.get(field, ""))
        if not actual:
            details.append((field, "unknown"))
            continue
        assessed += weight
        hit = bool(actual.intersection(target))
        if hit:
            matched += weight
        details.append((field, "match" if hit else "mismatch"))
    try:
        size = int(float(entity.get("employees", "")))
    except (TypeError, ValueError):
        size = None
    minimum = profile.get("min_employees")
    maximum = profile.get("max_employees")
    if minimum is not None or maximum is not None:
        if size is None:
            details.append(("employees", "unknown"))
        else:
            assessed += 10
            hit = (minimum is None or size >= minimum) and (maximum is None or size <= maximum)
            if hit:
                matched += 10
            details.append(("employees", "match" if hit else "mismatch"))
    return {"score": round(100 * matched / assessed) if assessed else None,
            "coverage": assessed, "details": details}
