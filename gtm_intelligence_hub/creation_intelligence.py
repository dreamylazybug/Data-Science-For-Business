"""Evidence-linked Creation Intelligence demo. Descriptive heuristics, not trained predictions."""
from collections import defaultdict,Counter
from datetime import date
from statistics import mean
from sfdc_mock import conversion_summary

def opportunity_records(rows):
    groups=defaultdict(list)
    for r in rows: groups[r["Opportunity code"]].append(r)
    return [{**lines[0],"TCV":round(sum(float(x["TCV"]) for x in lines),2),
             "line_count":len(lines)} for lines in groups.values()]

def horizon_models(rows,as_of,bookings_target=12_000_000,future_target=12_000_000):
    """Separate open-pipeline current bookings and forward creation-cohort coverage.

    Product win rate estimated from historical closed outcomes, with a 30% prior
    shrunk by 20 pseudo-observations. Timing modeled with coarse sales-cycle
    heuristic, explicitly not calibrated or backtested.
    """
    opps=opportunity_records(rows)
    rates=conversion_summary(rows)
    open_opps=[o for o in opps if o["stage"] not in ("Closed Won","Closed Lost")]
    current_end=date(as_of.year+(as_of.month>9),((as_of.month-1)//3*3+3)%12+1,1)
    from datetime import timedelta
    current_end-=timedelta(days=1)
    current=0.0
    future=[0.0,0.0,0.0]
    for o in open_opps:
        r=rates.get(o["product_level1"],{})
        wins=r.get("won",0); losses=r.get("lost",0)
        prob=(wins+6)/(wins+losses+20)
        age=(as_of-date.fromisoformat(o["opportunity created date"])).days
        # Remaining sales-cycle uncertainty approximated from opportunity age.
        # The fractions are assumptions, NOT empirically calibrated probabilities.
        timing=([.18,.32,.30,.20] if age<45 else
                [.30,.32,.25,.13] if age<100 else [.42,.30,.20,.08])
        expected=float(o["TCV"])*prob
        current+=expected*timing[0]
        for i in range(3): future[i]+=expected*timing[i+1]
    # Historical booked this quarter is distinct from expected open-pipeline contribution.
    start=date(as_of.year,((as_of.month-1)//3)*3+1,1)
    booked=sum(o["TCV"] for o in opps if o["stage"]=="Closed Won" and
               o["opportunity closed date"] and start<=date.fromisoformat(o["opportunity closed date"])<=as_of)
    current_gap=max(0,bookings_target-booked-current)
    return {"current":{"booked_qtd":booked,"expected_open":current,
                       "target":bookings_target,"gap":current_gap,
                       "coverage":(booked+current)/bookings_target if bookings_target else None},
            "future":{"expected_by_quarter":future,"target_per_quarter":future_target,
                      "gaps":[max(0,future_target-v) for v in future]},
            "method":"Closed-cohort product win-rate smoothing + illustrative age-based timing assumptions. Not calibrated."}

def inspect(rows,contacts,engagements,sellers,performance,as_of):
    opps=opportunity_records(rows)
    by_account=defaultdict(list)
    for c in contacts: by_account[c["account_code"]].append(c)
    by_opp=defaultdict(list)
    for e in engagements: by_opp[e["opportunity_code"]].append(e)
    perf=defaultdict(list)
    for p in performance: perf[(p["product_level1"] if "product_level1" in p else p["product"],p["product_level"])].append(p)
    seller_by_id={s["seller_id"]:s for s in sellers}
    won=[o for o in opps if o["stage"]=="Closed Won"]
    output=[]
    quarter_start=date(as_of.year,((as_of.month-1)//3)*3+1,1)
    for o in opps:
        created=date.fromisoformat(o["opportunity created date"])
        if not quarter_start<=created<=as_of: continue
        buyers=by_account[o["Account Code"]]
        interactions=by_opp[o["Opportunity code"]]
        decision_contacts={e["contact_id"] for e in interactions if e["decision_maker_present"]}
        engaged={e["contact_id"] for e in interactions}
        seller=seller_by_id.get(o["seller_id"],{})
        peer_wins=[w for w in won if w["product_level1"]==o["product_level1"] and
                   w["account segment"]==o["account segment"] and w["Opportunity code"]!=o["Opportunity code"]]
        peer_wins.sort(key=lambda w:(w["industry"]!=o["industry"],w["sales_territory"]!=o["sales_territory"]))
        precedent=peer_wins[0] if peer_wins else None
        gaps=[]
        if not decision_contacts: gaps.append("No decision-maker engagement recorded")
        if len(engaged)<2: gaps.append("Single-threaded or unrecorded buyer engagement")
        if float(o["discount %"])>20: gaps.append("Discount above illustrative 20% review threshold")
        if o["POV status"] in ("Not started","In progress") and o["POV health"]=="Red":
            gaps.append("POV needs technical review")
        if not o["specialist_id"] and o["product_level1"] in ("Cloud Security","Data Protection"):
            gaps.append("Consider product specialist for technical validation")
        # Product fit cannot be verified without account entitlement/usage data.
        gaps.append("Next-best-product fit unverified: account entitlement history unavailable")
        actions=[]
        if not decision_contacts: actions.append("Map economic buyer and secure decision-maker meeting")
        if len(engaged)<2: actions.append("Develop multi-threaded buyer engagement plan")
        if float(o["discount %"])>20: actions.append("Review comparable pricing and discount approval")
        if "POV needs technical review" in gaps: actions.append("Schedule POV health review with specialist")
        if not o["specialist_id"]: actions.append("Assess specialist intervention")
        actions.append("Share relevant product case study, ROI worksheet and competitor battlecard (catalog not connected)")
        if precedent: actions.append("Request peer coaching from seller of "+precedent["Opportunity code"])
        output.append({"opportunity_code":o["Opportunity code"],"account":o["Account Name"],
            "product":o["product_level1"],"segment":o["account segment"],"industry":o["industry"],
            "tcv":o["TCV"],"stage":o["stage"],"seller":o["seller_name"],
            "seller_tenure_years":seller.get("seller_tenure_years"),
            "seller_certifications":seller.get("certifications",""),
            "buyer_contacts":len(buyers),"engaged_contacts":len(engaged),
            "decision_maker_contacts":len(decision_contacts),
            "engagements":len(interactions),"discount":o["discount %"],
            "pov_status":o["POV status"],"specialist":o["specialist_name"],
            "competitor":o["incumbent competitor"],
            "precedent_opportunity":precedent["Opportunity code"] if precedent else None,
            "precedent_seller":precedent["seller_name"] if precedent else None,
            "gaps":gaps,"actions":actions,"evidence_note":"Synthetic CRM, contact and engagement records; thresholds are illustrative."})
    return sorted(output,key=lambda x:(-len(x["gaps"]),-x["tcv"]))
