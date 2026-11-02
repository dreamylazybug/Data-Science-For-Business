"""Evidence-linked GTM signal discovery. Standard-library-only ingestion."""
from __future__ import annotations
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import re
import xml.etree.ElementTree as ET

PRODUCTS = {
    "Cloud Security": {"terms": ["cloud", "aws", "azure", "gcp", "datacenter", "data center"], "triggers": ["expansion", "migration", "launch", "acquisition", "investment"]},
    "Identity Security": {"terms": ["identity", "access", "workforce", "authentication", "zero trust"], "triggers": ["acquisition", "merger", "expansion", "compliance"]},
    "Data Protection": {"terms": ["data", "privacy", "database", "analytics", "ai"], "triggers": ["regulation", "compliance", "launch", "expansion", "investment"]},
    "Network Security": {"terms": ["network", "branch", "office", "infrastructure", "connectivity"], "triggers": ["expansion", "acquisition", "migration", "modernization"]},
}
EVENTS = {
    "M&A": ["acquisition", "acquires", "merger", "merges"],
    "Expansion": ["expansion", "expands", "new office", "new region", "opens"],
    "Funding": ["funding", "raises", "investment", "series a", "series b"],
    "Technology Change": ["migration", "migrates", "modernization", "cloud"],
    "Regulatory": ["regulation", "compliance", "mandate", "privacy law"],
}
def demo_signals():
    return [
        {"company":"Acme Cloud", "title":"Acme Cloud announces a new cloud region (fictional demo)", "summary":"Cloud expansion and infrastructure investment.", "url":"", "published":"", "source":"Synthetic demonstration", "demo":True},
        {"company":"Vertex Data", "title":"Vertex Data expands AI analytics operations (fictional demo)", "summary":"AI and data expansion creates potential data protection needs.", "url":"", "published":"", "source":"Synthetic demonstration", "demo":True},
        {"company":"Northstar Labs", "title":"Northstar Labs announces acquisition (fictional demo)", "summary":"Acquisition and identity integration may require security review.", "url":"", "published":"", "source":"Synthetic demonstration", "demo":True},
    ]
def fetch_news(company, limit=10, timeout=8):
    """Opt-in Google News RSS query. Returns source-linked headlines, not verified facts."""
    company = company.strip()
    if not company or len(company)>100 or any(c in company for c in "\r\n"):
        raise ValueError("Enter a company name of up to 100 characters.")
    query = f'"{company}" (expansion OR acquisition OR funding OR migration OR security OR compliance)'
    url = "https://news.google.com/rss/search?" + urlencode({"q":query, "hl":"en-US", "gl":"US", "ceid":"US:en"})
    request = Request(url, headers={"User-Agent":"AnalytiCultGTM/0.2 (+research prototype)"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(1_000_001)
    if len(payload)>1_000_000:
        raise ValueError("News feed exceeds size limit.")
    root = ET.fromstring(payload)
    result=[]
    for item in root.findall("./channel/item")[:limit]:
        title=item.findtext("title","").strip()
        link=item.findtext("link","").strip()
        if urlparse(link).scheme not in ("http","https"):
            continue
        source=item.find("source")
        result.append({"company":company,"title":title[:500],"summary":re.sub("<[^>]+>"," ",item.findtext("description",""))[:700],"url":link,"published":item.findtext("pubDate",""),"source":source.text if source is not None else "Google News RSS","demo":False})
    return result
def assess_signal(signal, product_catalog=None):
    """Transparent rule-based relevance, NOT calibrated propensity or a factual assertion."""
    catalog=PRODUCTS if product_catalog is None else product_catalog
    text=(signal["title"]+" "+signal.get("summary","")).lower()
    event_types=[name for name, terms in EVENTS.items() if any(term in text for term in terms)]
    matches=[]
    for product, rules in catalog.items():
        terms=[t for t in rules["terms"] if t in text]
        triggers=[t for t in rules["triggers"] if t in text]
        if terms and (triggers or event_types):
            score=min(100, 25+15*len(set(terms))+12*len(set(triggers))+8*len(event_types))
            matches.append({"product":product,"relevance":score,"evidence_terms":terms[:5],
                            "hypothesis":f"Explore whether {signal['company']} has a need for {product} given the reported {', '.join(event_types) or 'technology activity'}.",
                            "next_action":"Validate the reported event, map existing technology, and identify a relevant buyer before outreach."})
    return sorted(matches,key=lambda x:x["relevance"],reverse=True)
