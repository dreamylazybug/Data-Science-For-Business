"""Creation Pulse Streamlit view: synthetic demonstration, no CRM integration."""
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from creation_pulse import pulse
from sfdc_mock import generate, creation_events, conversion_summary
from buyer_seller_mock import people, buyer_profiles, buyer_relationships, engagement_events, performance
from creation_intelligence import horizon_models, inspect

def dollars(value):
    return f"$"+f"{value/1_000_000:,.2f}M"

def render():
    st.subheader("Creation Pulse")
    st.caption("Quarter-to-date creation, prior-year linearity and target pace · synthetic demo data")
    as_of = st.date_input("Inspection date", value=date(2026,11,4),
                          min_value=date(2026,1,1), max_value=date(2026,12,31))
    target_m = st.number_input("Quarterly gross creation target ($M)", min_value=1.0,
                               value=35.0, step=1.0)
    rows = generate(as_of=as_of)
    sellers,specialists=people()
    contacts=buyer_profiles(rows)
    relationships=buyer_relationships(contacts)
    engagements=engagement_events(rows,contacts,as_of)
    seller_performance=performance(rows,sellers)
    events = creation_events(rows)
    result = pulse(events, as_of, target_m*1_000_000)
    a,b,c,d = st.columns(4)
    a.metric("Created QTD",dollars(result["qtd"]),
             f'{result["yoy_pct"]:+.1f}% vs prior year' if result["yoy_pct"] is not None else "Prior year unavailable")
    b.metric("Expected by today",dollars(result["required_to_date"]))
    c.metric("Pace variance",dollars(result["pace_gap"]))
    d.metric("Quarter-end pace estimate",dollars(result["projected"]))
    st.caption(f'Quarter starting {result["quarter_start"]} · {result["elapsed_selling_days"]} of '
               f'{result["quarter_selling_days"]} weekdays elapsed · '
               f'{result["opportunity_count"]} created opportunities. '
               "Weekday-only calendar; holidays and historical seasonality are not yet modeled.")
    df = pd.DataFrame(result["series"])
    if not df.empty:
        fig = go.Figure()
        for key, name in [("current","Current QTD"),("prior","Prior-year aligned"),
                          ("required","Required linear pace")]:
            fig.add_trace(go.Scatter(x=df["selling_day"],y=df[key]/1_000_000,
                                     mode="lines",name=name))
        fig.update_layout(title="Cumulative gross creation by elapsed selling day",
                          xaxis_title="Selling day in quarter",yaxis_title="Pipeline created ($M)",
                          legend_title_text="")
        st.plotly_chart(fig,use_container_width=True)
    with st.expander("Demo dataset · Salesforce-style opportunity line items", expanded=False):
        st.caption("Generated automatically: three years of fictional CRM records, 3–4 opportunities per week, SKU-level line items. No upload required.")
        st.metric("Unique opportunities",len({r["Opportunity code"] for r in rows}))
        st.metric("SKU line items",len(rows))
        st.dataframe(pd.DataFrame(rows).head(25),use_container_width=True,hide_index=True)
        st.download_button("Download synthetic SFDC CSV",
                           pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                           file_name="synthetic_sfdc_opportunity_line_items.csv",
                           mime="text/csv")
        rates = conversion_summary(rows)
        st.dataframe(pd.DataFrame([{"Product":p,**v} for p,v in sorted(rates.items())]),
                     use_container_width=True,hide_index=True)
        st.caption("Closed-win rates exclude open opportunities; line items are not counted as separate opportunities.")
    with st.expander("Buyer and seller intelligence · Synthetic supporting datasets", expanded=False):
        st.caption("Account contacts, relationship links, engagement events, named sellers and specialists. All records are fictional.")
        views = {
            "Buyer contacts":contacts,"Buyer relationships":relationships,
            "Buyer engagements":engagements,"Sellers":sellers,"Specialists":specialists,
            "Seller product performance":seller_performance,
        }
        chosen=st.selectbox("Inspect dataset",list(views),key="buyer_seller_dataset")
        frame=pd.DataFrame(views[chosen])
        st.metric("Records",len(frame))
        st.dataframe(frame.head(100),use_container_width=True,hide_index=True)
        st.download_button("Download selected CSV",frame.to_csv(index=False).encode("utf-8"),
                           file_name=chosen.lower().replace(" ","_")+".csv",mime="text/csv")
        st.caption("Relationship strength and sentiment are simulated, not inferred from emails or actual interactions. Seller performance uses closed opportunities only.")
    st.subheader("Creation composition")
    tab1,tab2,tab3=st.tabs(["Product","Segment","Sales motion"])
    for tab,group in [(tab1,"by_product"),(tab2,"by_segment"),(tab3,"by_motion")]:
        with tab:
            items=result[group]
            st.dataframe(pd.DataFrame([{"Category":k,"Created amount":v,
                                        "Share of QTD":v/result["qtd"] if result["qtd"] else 0}
                                       for k,v in sorted(items.items(),key=lambda x:-x[1])]),
                         use_container_width=True,hide_index=True,
                         column_config={"Created amount":st.column_config.NumberColumn(format="$%.0f"),
                                        "Share of QTD":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.1f")})
    st.subheader("Conversion-adjusted sufficiency · Two horizons")
    st.caption("Illustrative estimates from historical closed-opportunity win rates and assumed time-to-close distributions. Not validated forecasts.")
    left,right=st.columns(2)
    with left:
        current_target=st.number_input("Current-quarter bookings target ($M)",min_value=0.1,value=12.0,step=1.0)
    with right:
        future_target=st.number_input("Future bookings target per quarter ($M)",min_value=0.1,value=12.0,step=1.0)
    horizons=horizon_models(rows,as_of,current_target*1e6,future_target*1e6)
    current=horizons["current"]
    a,b,c=st.columns(3)
    a.metric("Booked this quarter",dollars(current["booked_qtd"]))
    b.metric("Expected from open pipeline",dollars(current["expected_open"]))
    c.metric("Current-quarter gap",dollars(current["gap"]))
    future=horizons["future"]
    st.bar_chart(pd.DataFrame({"Quarter":["Next Q","Q+2","Q+3"],
                              "Expected bookings ($M)":[v/1e6 for v in future["expected_by_quarter"]],
                              "Target ($M)":[future_target]*3}).set_index("Quarter"))
    st.caption(horizons["method"])
    st.subheader("Created-this-quarter opportunity inspection")
    inspected=inspect(rows,contacts,engagements,sellers,seller_performance,as_of)
    st.metric("Opportunities inspected",len(inspected))
    if inspected:
        st.dataframe(pd.DataFrame([{"Opportunity":o["opportunity_code"],"Account":o["account"],
            "Product":o["product"],"TCV":o["tcv"],"Seller":o["seller"],
            "Buyer engagements":o["engagements"],"Decision makers":o["decision_maker_contacts"],
            "Quality flags":len(o["gaps"])} for o in inspected]),
            use_container_width=True,hide_index=True)
        selected=st.selectbox("Inspect opportunity",options=[o["opportunity_code"] for o in inspected])
        detail=next(o for o in inspected if o["opportunity_code"]==selected)
        a,b,c=st.columns(3)
        a.metric("Engaged contacts",detail["engaged_contacts"])
        b.metric("Decision makers engaged",detail["decision_maker_contacts"])
        c.metric("Discount",f'{detail["discount"]:.1f}%')
        st.write("**Seller:**",detail["seller"]," · **Specialist:**",detail["specialist"] or "Unassigned")
        st.write("**Competitor:**",detail["competitor"])
        st.write("**Relevant seller precedent:**",detail["precedent_opportunity"] or "No matching won deal",
                 detail["precedent_seller"] or "")
        st.write("**Inspection findings**")
        for finding in detail["gaps"]: st.warning(finding)
        st.write("**Recommended next actions**")
        for action in detail["actions"]: st.write("•",action)
        st.caption(detail["evidence_note"])
    st.caption("All companies, buyers and sellers are fictional. Opportunity inspections use transparent demo rules; "
               "next-best-product fit, collateral availability and true engagement strength need additional evidence.")
