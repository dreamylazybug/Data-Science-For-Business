"""Analyti-Cult GTM Intelligence Hub — synthetic-data MVP."""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analyti-Cult | GTM Intelligence Hub", layout="wide", page_icon="📈")
MODULES = [
    "GTM Command Center", "Creation Intelligence", "Pipeline Generation", "Pipeline Conversion",
    "Win/Loss Intelligence", "Pricing Intelligence", "Customer Expansion",
    "Renewal Intelligence", "Revenue Forecasting", "Seller Intelligence",
]
st.sidebar.title("Analyti-Cult")
st.sidebar.caption("AI-first GTM Intelligence Hub · v0.8")
module = st.sidebar.radio("Explore", MODULES)
st.sidebar.info("Synthetic data by default. Pipeline Generation supports opt-in live news headline research.")
st.title(module)
st.caption("Illustrative synthetic data only — no production AI predictions.")

accounts = pd.DataFrame([
    {"Account":"Acme Cloud","Segment":"Enterprise","Pipeline":480000,"Win probability":0.65,"Renewal risk":0.18,"Expansion potential":120000},
    {"Account":"Northstar Labs","Segment":"Commercial","Pipeline":210000,"Win probability":0.42,"Renewal risk":0.72,"Expansion potential":85000},
    {"Account":"Vertex Data","Segment":"Commercial","Pipeline":160000,"Win probability":0.81,"Renewal risk":0.22,"Expansion potential":70000},
    {"Account":"Summit Systems","Segment":"Enterprise","Pipeline":620000,"Win probability":0.38,"Renewal risk":0.55,"Expansion potential":200000},
    {"Account":"Blue Harbor","Segment":"SMB","Pipeline":95000,"Win probability":0.73,"Renewal risk":0.31,"Expansion potential":30000},
])
with st.sidebar.expander("Advanced · Data uploads", expanded=False):
    uploaded = st.file_uploader("Optional account CSV", type=["csv"])
if uploaded:
    try:
        incoming = pd.read_csv(uploaded)
        st.sidebar.success("CSV loaded for preview")
        with st.expander("Uploaded data preview"):
            st.dataframe(incoming.head(100), use_container_width=True)
        st.sidebar.caption("Demo calculations continue to use synthetic data until schema mapping is implemented.")
    except Exception as exc:
        st.sidebar.error(f"Unable to read CSV: {exc}")

if module == "GTM Command Center":
    a,b,c,d = st.columns(4)
    a.metric("Open pipeline", f"${accounts['Pipeline'].sum():,.0f}")
    b.metric("Weighted pipeline", f"${(accounts['Pipeline']*accounts['Win probability']).sum():,.0f}")
    c.metric("Expansion potential", f"${accounts['Expansion potential'].sum():,.0f}")
    d.metric("High renewal risk", int((accounts['Renewal risk']>=0.5).sum()))
    st.subheader("Illustrative AI briefing")
    st.info("Northstar Labs shows elevated renewal risk; Summit Systems has a large deal with lower win propensity. Investigate both before acting.")
    st.plotly_chart(px.bar(accounts, x="Account", y="Pipeline", color="Segment", title="Pipeline by account"), use_container_width=True)
elif module == "Creation Intelligence":
    from creation_ui import render
    render()
elif module == "Pipeline Generation":
    from pipeline_ui import render
    render()
elif module == "Pipeline Conversion":
    st.plotly_chart(px.scatter(accounts, x="Win probability", y="Pipeline", size="Pipeline", hover_name="Account", title="Opportunity risk overview"), use_container_width=True)
    st.dataframe(accounts[["Account","Pipeline","Win probability"]], use_container_width=True)
elif module == "Win/Loss Intelligence":
    st.info("Demo themes: pricing complexity, procurement delays, technical differentiation. These are placeholders, not extracted insights.")
    st.bar_chart(pd.Series({"Pricing complexity":12,"Procurement":8,"Feature gaps":6,"Competition":9}))
elif module == "Pricing Intelligence":
    discount = st.slider("Illustrative discount (%)", 0, 40, 10)
    list_price = st.number_input("List price ($)", 1000, 10000000, 300000, step=10000)
    st.metric("Illustrative net price", f"${list_price*(1-discount/100):,.0f}")
    st.caption("Scenario arithmetic only; no estimated elasticity or optimal price is claimed.")
elif module == "Customer Expansion":
    st.dataframe(accounts[["Account","Expansion potential","Segment"]].sort_values("Expansion potential",ascending=False), use_container_width=True)
    st.info("Next-best-product recommendations will require product catalog, entitlements, and consumption signals.")
elif module == "Renewal Intelligence":
    st.plotly_chart(px.bar(accounts, x="Account", y="Renewal risk", title="Synthetic renewal risk indicators"), use_container_width=True)
    st.caption("These are demo inputs, not calibrated churn probabilities.")
elif module == "Revenue Forecasting":
    st.metric("Illustrative weighted forecast", f"${(accounts['Pipeline']*accounts['Win probability']).sum():,.0f}")
    st.caption("Forecasting models, uncertainty ranges, and historical backtests are future milestones.")
elif module == "Seller Intelligence":
    st.info("Seller coaching will use consented activity evidence and context-adjusted measures; employee decisions require human review.")
    st.dataframe(pd.DataFrame([
        {"Seller":"Seller A","Pipeline":520000,"Deals":6},
        {"Seller":"Seller B","Pipeline":410000,"Deals":5},
        {"Seller":"Seller C","Pipeline":635000,"Deals":8},
    ]), use_container_width=True)
