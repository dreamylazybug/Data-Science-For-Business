"""Fictional, deterministic account contacts, engagement relationships and seller profiles."""
import random
from collections import defaultdict
from datetime import date, timedelta

FIRST = ["Avery","Jordan","Taylor","Morgan","Casey","Riley","Sam","Devin","Alex","Jamie","Cameron","Drew"]
LAST = ["Reed","Patel","Chen","Rivera","Kim","Johnson","Singh","Brooks","Garcia","Nguyen","Shah","Miller"]
ROLES = [
    ("CEO","C-Suite","Economic buyer"),("CFO","C-Suite","Financial approver"),
    ("CIO","C-Suite","Executive sponsor"),("VP Security","Decision Maker","Technical decision maker"),
    ("IT Director","Decision Maker","Technical decision maker"),
    ("Security Operations Manager","Operations","Operational champion"),
    ("Procurement Director","Operations","Commercial evaluator"),
    ("Security Analyst","Analyst","Technical evaluator"),
    ("Data Analyst","Analyst","Technical evaluator"),
]
CERTS = ["Cloud Security Professional","Zero Trust Architecture","Enterprise Negotiation",
         "Data Protection Specialist","Identity Security Foundations",
         "Network Security Advanced","Solution Value Selling","POV Design Practitioner"]
SPECIALTIES = ["Solution Architect","Product Specialist","Security Specialist"]

def people(seed=881):
    rng=random.Random(seed)
    names=[f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    sellers=[]
    for i in range(24):
        region=["West","East","Central","International"][i%4]
        segment=["SMB","Commercial","Enterprise"][i%3]
        certifications=sorted(rng.sample(CERTS,rng.randint(1,5)))
        sellers.append({"seller_id":f"SELL-{i+1:03d}","seller_name":names[i],
            "role":"Account Executive","sales_level_1":["Commercial Sales","Enterprise Sales","Inside Sales"][i%3],
            "sales_level_2":f"{region} - {segment}","sales_territory":region,
            "seller_tenure_years":round(rng.uniform(.5,12),1),
            "certification_count":len(certifications),"certifications":" | ".join(certifications),
            "training_modules_completed":rng.randint(4,55)})
    specialists=[]
    for i in range(12):
        region=["West","East","Central","International"][i%4]
        certifications=sorted(rng.sample(CERTS,rng.randint(2,6)))
        specialists.append({"specialist_id":f"SPEC-{i+1:03d}",
            "specialist_name":names[24+i],"role":SPECIALTIES[i%3],
            "sales_level_1":"Specialist Organization","sales_level_2":f"{region} - Specialists",
            "sales_territory":region,"seller_tenure_years":round(rng.uniform(1,14),1),
            "certification_count":len(certifications),"certifications":" | ".join(certifications),
            "training_modules_completed":rng.randint(10,65)})
    return sellers,specialists

def buyer_profiles(opportunity_rows,seed=928):
    """Stable contacts per account; contacts never synthesized from real people."""
    rng=random.Random(seed)
    accounts={}
    for row in opportunity_rows:
        accounts.setdefault(row["Account Code"],row["Account Name"])
    contacts=[]
    for code,name in sorted(accounts.items()):
        for i,(title,category,buying_role) in enumerate(ROLES):
            strength=rng.choice(["Strong","Developing","Weak","Unknown"])
            contacts.append({"contact_id":f"{code}-C{i+1:02d}","account_code":code,
                "account_name":name,"contact_name":f"{FIRST[(i+int(code[-3:]))%len(FIRST)]} {LAST[(i*3+int(code[-3:]))%len(LAST)]}",
                "title":title,"role_category":category,"buying_role":buying_role,
                "relationship_strength":strength,
                "relationship_score": {"Strong":85,"Developing":60,"Weak":25,"Unknown":None}[strength],
                "influence":rng.choice(["High","Medium","Low"]),
                "sentiment":rng.choice(["Supportive","Neutral","Skeptical","Unknown"]),
                "last_engagement_date":"","engagement_count":0})
    return contacts

def buyer_relationships(contacts):
    """Directed account-level influence links, not asserted real reporting lines."""
    by_account=defaultdict(list)
    for contact in contacts: by_account[contact["account_code"]].append(contact)
    edges=[]
    for code,group in by_account.items():
        for source,target,relation in [(3,2,"Influences"),(4,2,"Advises"),(5,3,"Champions"),
                                       (7,5,"Works with"),(8,4,"Works with"),
                                       (6,1,"Procurement approval"),(1,0,"Financial alignment")]:
            edges.append({"account_code":code,"source_contact_id":group[source]["contact_id"],
                          "target_contact_id":group[target]["contact_id"],
                          "relationship_type":relation,
                          "relationship_strength":group[source]["relationship_strength"]})
    return edges

def engagement_events(opportunity_rows,contacts,as_of=date(2026,11,4),seed=919):
    rng=random.Random(seed)
    by_account=defaultdict(list)
    for c in contacts: by_account[c["account_code"]].append(c)
    opps={r["Opportunity code"]:r for r in opportunity_rows}
    events=[]
    for opp_id,row in sorted(opps.items()):
        created=date.fromisoformat(row["opportunity created date"])
        if created>as_of: continue
        available=(as_of-created).days
        if available<0: continue
        for j in range(rng.randint(0,8)):
            contact=rng.choice(by_account[row["Account Code"]])
            day=created+timedelta(days=rng.randint(0,available))
            events.append({"engagement_id":f"ENG-{opp_id}-{j+1:02d}",
                "opportunity_code":opp_id,"account_code":row["Account Code"],
                "contact_id":contact["contact_id"],"seller_id":row["seller_id"],
                "engagement_date":day.isoformat(),
                "channel":rng.choice(["Email","Meeting","Call","Demo","Executive briefing"]),
                "direction":rng.choice(["Inbound","Outbound"]),
                "outcome":rng.choice(["Positive","Neutral","Objection","Follow-up needed"]),
                "decision_maker_present":contact["role_category"] in ("C-Suite","Decision Maker"),
                "synthetic":True})
    recent={}
    counts=defaultdict(int)
    for e in events:
        counts[e["contact_id"]]+=1
        recent[e["contact_id"]]=max(recent.get(e["contact_id"],""),e["engagement_date"])
    for c in contacts:
        c["engagement_count"]=counts[c["contact_id"]]
        c["last_engagement_date"]=recent.get(c["contact_id"],"")
    return events

def performance(opportunity_rows,people_rows):
    """Observed closed-only win rates by product1 and product2; no causal inference."""
    opps=defaultdict(list)
    for row in opportunity_rows: opps[row["Opportunity code"]].append(row)
    summary=defaultdict(lambda:{"wins":0,"losses":0,"won_tcv":0.0,"won_age_days":0,
                                "loss_reasons":defaultdict(int)})
    for lines in opps.values():
        row=lines[0]
        if row["stage"] not in ("Closed Won","Closed Lost"): continue
        for level in ("product_level1","product_level2"):
            key=(row["seller_id"],level,row[level])
            item=summary[key]
            if row["stage"]=="Closed Won":
                item["wins"]+=1
                item["won_tcv"]+=sum(float(r["TCV"]) for r in lines)
                item["won_age_days"]+=(date.fromisoformat(row["opportunity closed date"])-date.fromisoformat(row["opportunity created date"])).days
            else:
                item["losses"]+=1
                item["loss_reasons"][row["primary reason lost"]]+=1
    result=[]
    for person in people_rows:
        for (seller_id,level,product),v in summary.items():
            if seller_id!=person["seller_id"]: continue
            closed=v["wins"]+v["losses"]
            result.append({"seller_id":seller_id,"seller_name":person["seller_name"],
                "product_level":level,"product":product,"closed_opportunities":closed,
                "won":v["wins"],"lost":v["losses"],"win_rate":v["wins"]/closed if closed else None,
                "avg_won_deal_size":round(v["won_tcv"]/v["wins"],2) if v["wins"] else None,
                "avg_won_deal_age_days":round(v["won_age_days"]/v["wins"],1) if v["wins"] else None,
                "primary_lost_reason":max(v["loss_reasons"],key=v["loss_reasons"].get) if v["loss_reasons"] else "",
                "sales_level_1":person["sales_level_1"],"sales_level_2":person["sales_level_2"],
                "sales_territory":person["sales_territory"]})
    return result
