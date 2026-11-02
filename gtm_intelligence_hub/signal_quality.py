"""Signal quality, account matching, and configurable ICP utilities."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from urllib.parse import urlparse

def normalize_company(value):
    value=re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    tokens=value.split()
    while tokens and tokens[-1] in {"inc","llc","ltd","corp","corporation","co","company","limited"}:
        tokens.pop()
    return " ".join(tokens)

def company_match(company, headline):
    """Conservative whole-name check. Reject ambiguous one-token short names."""
    name=normalize_company(company)
    title=normalize_company(headline)
    if not name or len(name)<5:
        return False
    return bool(re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",title))

def freshness_days(published, now=None):
    if not published:
        return None
    try:
        date=parsedate_to_datetime(published)
        if date.tzinfo is None:
            date=date.replace(tzinfo=timezone.utc)
        current=now or datetime.now(timezone.utc)
        return max(0,(current-date).days)
    except (ValueError, TypeError, OverflowError):
        return None

def deduplicate(signals):
    seen=set()
    result=[]
    for signal in signals:
        url=signal.get("url","").strip()
        title=re.sub(r"\W+"," ",signal.get("title","").lower()).strip()
        key=(normalize_company(signal.get("company","")),url or title)
        if key in seen:
            continue
        seen.add(key)
        result.append(signal)
    return result

def quality_flags(signal, max_age=180):
    flags=[]
    if signal.get("demo"):
        flags.append("Synthetic demonstration")
        return flags
    if not company_match(signal.get("company",""),signal.get("title","")):
        flags.append("Company name not confirmed in headline")
    age=freshness_days(signal.get("published",""))
    if age is None:
        flags.append("Publication date unknown")
    elif age>max_age:
        flags.append("Stale signal")
    if urlparse(signal.get("url","")).scheme not in {"http","https"}:
        flags.append("No source link")
    return flags

def filter_icp(signal, allowed_companies):
    """Exact normalized company match, never fuzzy-match unrelated companies."""
    if not allowed_companies:
        return True
    return normalize_company(signal.get("company","")) in {normalize_company(x) for x in allowed_companies}
