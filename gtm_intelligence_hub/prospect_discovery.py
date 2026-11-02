"""Discover candidate companies from topical news; candidates require human verification."""
from __future__ import annotations
import re
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from html import unescape
from pipeline_engine import EVENTS
from signal_quality import deduplicate

TOPICS={
    "Cloud infrastructure expansion": '"cloud" ("expands" OR "new region" OR "data center")',
    "AI and data investments": '"AI" ("funding" OR "investment" OR "expands")',
    "Mergers and acquisitions": '"acquires" OR "acquisition" OR "merger"',
    "Cybersecurity modernization": '"security" ("migration" OR "modernization" OR "investment")',
}
# Require a named organization immediately preceding a concrete event verb.
PATTERN=re.compile(
    r"^([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4})"
    r"\s+(announces|announced|acquires|acquired|raises|raised|expands|expanded|opens|opened|launches|launched|invests|invested)\b",
    re.IGNORECASE,
)
BAD={"the","this","a","an","new","how","why","report","market","startup","company","companies","government","breaking","exclusive","analysts","researchers"}
def extract_candidate(title):
    """Return a conservative *candidate* name, never a verified entity."""
    # RSS commonly appends ' - Publisher'. Strip only the final separator.
    headline=title.rsplit(" - ",1)[0].strip()
    match=PATTERN.match(headline)
    if not match:
        return None
    name=match.group(1).strip(" .")
    if name.lower().split()[0] in BAD or len(name)<4 or len(name)>90:
        return None
    return name

def fetch_topic(topic, limit=35, timeout=10):
    if topic not in TOPICS:
        raise ValueError("Unsupported topic")
    url="https://news.google.com/rss/search?"+urlencode({"q":TOPICS[topic],"hl":"en-US","gl":"US","ceid":"US:en"})
    request=Request(url,headers={"User-Agent":"AnalytiCultGTM/0.4 research prototype"})
    with urlopen(request,timeout=timeout) as response:
        raw=response.read(1_000_001)
    if len(raw)>1_000_000:
        raise ValueError("Feed too large")
    root=ET.fromstring(raw)
    results=[]
    for item in root.findall("./channel/item")[:limit]:
        title=unescape(item.findtext("title","")).strip()[:500]
        company=extract_candidate(title)
        link=item.findtext("link","").strip()
        if not company or urlparse(link).scheme not in ("http","https"):
            continue
        source=item.find("source")
        summary=re.sub(r"<[^>]+>"," ",item.findtext("description",""))
        events=[event for event,terms in EVENTS.items() if any(re.search(r"\b"+re.escape(t)+r"\b",title.lower()) for t in terms)]
        results.append({"company":company,"title":title,"summary":unescape(summary)[:700],
                        "url":link,"published":item.findtext("pubDate",""),
                        "source":source.text if source is not None else "Google News RSS",
                        "demo":False,"discovery":True,"events":events,"entity_status":"Candidate — unverified"})
    return deduplicate(results)
