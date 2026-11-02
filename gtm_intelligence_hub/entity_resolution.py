"""Local company-registry resolution; no claims of third-party verification."""
import re
from urllib.parse import urlparse
from signal_quality import normalize_company

def _domain(value):
    value=str(value or "").strip().lower()
    if not value:
        return ""
    host=urlparse(value if "://" in value else "https://"+value).hostname or ""
    return host.removeprefix("www.")

def prepare_registry(records):
    """Input records: company, optional domain, aliases (pipe separated), industry, segment."""
    result=[]
    for idx,row in enumerate(records):
        company=str(row.get("company","")).strip()
        if not company or company.lower()=="nan":
            continue
        aliases=[a.strip() for a in str(row.get("aliases","") or "").split("|") if a.strip() and a.strip().lower()!="nan"]
        result.append({"id":idx,"company":company,"domain":_domain(row.get("domain","")),
                       "aliases":aliases,"industry":str(row.get("industry","") or ""),
                       "segment":str(row.get("segment","") or ""),
                       "country":str(row.get("country","") or ""),
                       "technologies":str(row.get("technologies","") or ""),
                       "employees":str(row.get("employees","") or "")})
    return result

def resolve(candidate, registry, domain_hint=""):
    """Exact normalized name/alias or supplied-domain matching; ambiguity blocks assignment."""
    name=normalize_company(candidate)
    hint=_domain(domain_hint)
    matches=[]
    for entity in registry:
        names=[entity["company"]]+entity["aliases"]
        name_hit=bool(name) and any(normalize_company(n)==name for n in names)
        domain_hit=bool(hint and entity["domain"] and hint==entity["domain"])
        if name_hit or domain_hit:
            matches.append((entity,name_hit,domain_hit))
    if not matches:
        return {"status":"Unresolved","entity":None,"candidates":[],"method":"No exact registry match"}
    # Domain can disambiguate name collisions only when supplied independently.
    if hint:
        domain_matches=[m for m in matches if m[2]]
        if domain_matches:
            matches=domain_matches
    if len(matches)>1:
        return {"status":"Ambiguous","entity":None,"candidates":[m[0]["company"] for m in matches],
                "method":"Multiple registry matches; manual review required"}
    entity,name_hit,domain_hit=matches[0]
    return {"status":"Registry match — unverified","entity":entity,"candidates":[entity["company"]],
            "method":"Exact domain" if domain_hit else "Exact name or alias"}

def registry_from_frame(frame):
    cols={str(c).strip().lower():c for c in frame.columns}
    if "company" not in cols:
        raise ValueError("Registry CSV must include a company column")
    selected=[]
    for _,row in frame.iterrows():
        selected.append({k:("" if k not in cols or row[cols[k]]!=row[cols[k]] else str(row[cols[k]]))
                         for k in ("company","domain","aliases","industry","segment","country","technologies","employees")})
    return prepare_registry(selected)
