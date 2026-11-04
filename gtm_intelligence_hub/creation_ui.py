"""Creation Pulse Streamlit view: synthetic demonstration, no CRM integration."""
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from creation_pulse import demo_events, pulse, quarter_start, separate_horizons

def dollars(value):
    return f"$"+f"{value/1_000_000:,.2f}M"

def render():
    st.subheader("Creation Pulse")
    st.caption("Quarter-to-date creation, prior-year linearity and target pace · synthetic demo data")
    as_of = st.date_input("Inspection date", value=date(2026,11,4),
                          min_value=date(2026,1,1), max_value=date(2026,12,31))
    target_m = st.number_input("Quarterly gross creation target ($M)", min_value=1.0,
                               value=35.0, step=1.0)
    events = demo_events(as_of)
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
    st.subheader("Separate sufficiency horizons")
    left,right=st.columns(2)
    with left:
        st.markdown("**Current-quarter bookings sufficiency**")
        st.info("Model not yet calibrated. Requires open pipeline snapshots, stage history, expected close dates and cohort conversion.")
    with right:
        st.markdown("**Future-quarter pipeline health**")
        st.info("Model not yet calibrated. Requires creation cohorts, historical win/loss and time-to-close distributions.")
    st.caption("Creation target and pacing are not bookings forecasts. Values and opportunities are fictional; "
               "amount changes after creation are excluded. No live CRM or external enrichment is used.")
