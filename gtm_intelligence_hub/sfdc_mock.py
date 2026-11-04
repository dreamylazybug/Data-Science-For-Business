"""Deterministic Salesforce-style synthetic GTM data. No real customer records."""
import random
from datetime import date, timedelta
from collections import defaultdict

PRODUCTS = {
    "Cloud Security": [("Cloud Posture","CSPM-CORE",0.34),("Workload Protection","CWPP-PRO",0.31)],
    "Identity Security": [("Identity Governance","IGA-ENT",0.27),("Access Management","IAM-ADV",0.30)],
    "Data Protection": [("Data Discovery","DSPM-PLUS",0.24),("DLP","DLP-ENT",0.28)],
    "Network Security": [("Next Gen Firewall","NGFW-500",0.37),("SASE","SASE-PRO",0.34)],
}
INDUSTRIES = ["Technology","Financial Services","Healthcare","Manufacturing","Retail","Education"]
SEGMENTS = ["SMB","Commercial","Enterprise"]
COMPETITORS = ["Vendor Alpha","Vendor Beta","Vendor Gamma","Vendor Delta"]
REASONS = ["Price","Feature gap","Incumbent preference","Timing","Procurement","No decision"]
FIELDS = ["Opportunity code","Name","Account Name","Account Code","product_level1",
          "product_level2","SKU","pipeline source","Route to Market","forecast category",
          "stage","comments","next step","specialist","specialist comments","POV status",
          "POV health","discount %","non standard discount %","account segment","industry",
          "primary reason lost","incumbent competitor","winning competitor","sales_level_1",
          "sales_level_2","sales_territory","TCV","Upsell TCV","Renew TCV","NetNew TCV",
          "opportunity term","opportunity created date","opportunity closed date",
          "line item","line item start date","line item end date","line item term"]

def generate(as_of=date(2026,11,4), years=3, seed=20261104):
    """One record per SKU line item; financial values are line-level, not repeated totals.

    An opportunity may have multiple lines. Product-level win propensity varies
    between 24% and 37%; realized conversion also depends on open opportunities.
    """
    rng = random.Random(seed)
    start = date(as_of.year-years,as_of.month,as_of.day)
    monday = start-timedelta(days=start.weekday())
    rows = []
    opportunities = []
    n = 0
    week = monday
    while week <= as_of:
        for _ in range(rng.choice([3,3,4,4])):
            created = week + timedelta(days=rng.randint(0,4))
            if created < start or created > as_of:
                continue
            n += 1
            account_id = rng.randint(1,115)
            account = f"Demo Account {account_id:03d}"
            level1 = rng.choice(list(PRODUCTS))
            sku_set = PRODUCTS[level1]
            product2,sku,win_rate = rng.choice(sku_set)
            term = rng.choice([12,12,24,36])
            segment = rng.choices(SEGMENTS,weights=[20,55,25])[0]
            industry = rng.choice(INDUSTRIES)
            competitor = rng.choice(COMPETITORS)
            source = rng.choice(["Inbound","Outbound","Partner","Marketing","Expansion"])
            route = rng.choice(["Direct","Channel","Marketplace"])
            region = rng.choice(["West","East","Central","International"])
            sales_l1 = rng.choice(["Commercial Sales","Enterprise Sales","Inside Sales"])
            sales_l2 = f"{region} - {segment}"
            discount = round(rng.uniform(0,28),1)
            nonstandard = round(max(0,discount-20),1)
            days_to_close = rng.randint(35,210)
            close = created+timedelta(days=days_to_close)
            is_closed = close <= as_of
            won = is_closed and rng.random() < win_rate
            lost = is_closed and not won
            stage = "Closed Won" if won else "Closed Lost" if lost else rng.choice(
                ["Qualification","Discovery","Solution Validation","Proposal","Negotiation"])
            forecast = "Closed" if is_closed else {"Qualification":"Pipeline","Discovery":"Pipeline",
                "Solution Validation":"Best Case","Proposal":"Best Case","Negotiation":"Commit"}[stage]
            pov_status = rng.choice(["Not required","Not started","In progress","Completed"])
            pov_health = "N/A" if pov_status=="Not required" else rng.choice(["Green","Amber","Red"])
            specialist = rng.choice(["","Solution Architect","Product Specialist","Security Specialist"])
            opp_id = f"OPP-{n:06d}"
            # One to three SKU lines per opportunity; unique line IDs prevent double counting.
            count = rng.choices([1,2,3],weights=[65,27,8])[0]
            allocations = [rng.uniform(.3,1) for _ in range(count)]
            total_alloc = sum(allocations)
            total = rng.randint(40,500)*1000
            opp_total = 0
            for li in range(count):
                p2,sku_line,_ = sku_set[(sku_set.index((product2,sku,win_rate))+li)%len(sku_set)]
                value = round(total*allocations[li]/total_alloc,2) if li<count-1 else round(total-opp_total,2)
                opp_total += value
                # Motion is assessed at product_level2 for this demonstration.
                motion = rng.choices(["NetNew","Upsell","Renew"],weights=[55,30,15])[0]
                line_start = close if won else created+timedelta(days=days_to_close)
                line_end = line_start+timedelta(days=round(term*30.4375))
                rows.append(dict(zip(FIELDS,[
                    opp_id,f"{account} - {level1} - {n}",account,f"ACC-{account_id:05d}",
                    level1,p2,sku_line,source,route,forecast,stage,
                    "Synthetic sales note; validate customer need and buyer engagement.",
                    "Confirm decision process and next customer milestone." if not is_closed else "Closed",
                    specialist,"Synthetic specialist assessment." if specialist else "",
                    pov_status,pov_health,discount,nonstandard,segment,industry,
                    rng.choice(REASONS) if lost else "",competitor,
                    competitor if lost and rng.random()<.7 else ("Other / unknown" if lost else ""),
                    sales_l1,sales_l2,region,value,
                    value if motion=="Upsell" else 0,
                    value if motion=="Renew" else 0,
                    value if motion=="NetNew" else 0,
                    term,created.isoformat(),close.isoformat() if is_closed else "",
                    f"{opp_id}-LI-{li+1:02d}",line_start.isoformat(),line_end.isoformat(),term
                ])))
            opportunities.append((opp_id,stage,level1))
        week += timedelta(days=7)
    return rows

def creation_events(rows):
    """Deduplicate at opportunity level; use sum of SKU-level created TCV."""
    grouped = {}
    for row in rows:
        key = row["Opportunity code"]
        if key not in grouped:
            grouped[key] = {"created_date":row["opportunity created date"],
                            "created_amount":0.0,"product":row["product_level1"],
                            "segment":row["account segment"],"motion":"Mixed"}
        grouped[key]["created_amount"] += float(row["TCV"])
    return list(grouped.values())

def conversion_summary(rows):
    """Closed-only historical conversion by product; open opportunities excluded."""
    unique = {}
    for row in rows:
        unique[row["Opportunity code"]] = (row["product_level1"],row["stage"])
    counts = defaultdict(lambda: {"won":0,"lost":0,"open":0})
    for product,stage in unique.values():
        key = "won" if stage=="Closed Won" else "lost" if stage=="Closed Lost" else "open"
        counts[product][key] += 1
    return {p:{**v,"closed_win_rate":v["won"]/(v["won"]+v["lost"]) if v["won"]+v["lost"] else None}
            for p,v in counts.items()}
