"""Streamlit UI for signal discovery and human-reviewed opportunity hypotheses."""
import pandas as pd
import streamlit as st
from pipeline_engine import PRODUCTS, assess_signal, demo_signals, fetch_news

def render():
    st.subheader("AI Pipeline Generation | Signal-to-opportunity discovery")
    st.caption("Evidence-linked research prototype: rule-based matching, not a trained AI propensity model.")
    mode=st.radio("Signal source",["Synthetic demonstration","Live news headlines (opt-in)"],horizontal=True)
    if mode=="Synthetic demonstration":
        signals=demo_signals()
        st.info("All companies and events below are fictional demonstration examples.")
    else:
        company=st.text_input("Company to research",placeholder="Enter an actual company name")
        st.caption("Searches Google News RSS only when you click Discover. Headlines are unverified and may contain irrelevant matches.")
        if st.button("Discover external signals",type="primary"):
            try:
                with st.spinner("Retrieving headlines..."):
                    st.session_state["gtm_signals"]=fetch_news(company)
                    st.session_state["gtm_company"]=company
            except Exception as exc:
                st.error(f"Could not retrieve the feed: {exc}")
        signals=st.session_state.get("gtm_signals",[]) if st.session_state.get("gtm_company")==company else []
    products=st.multiselect("Product areas",list(PRODUCTS),default=list(PRODUCTS))
    rows=[]
    for signal in signals:
        for match in assess_signal(signal,{p:PRODUCTS[p] for p in products}):
            rows.append({**signal,**match})
    if not rows:
        st.info("No matching signals yet. Try another company or broaden the selected product areas.")
        return
    st.metric("Opportunity hypotheses",len(rows))
    st.metric("Unique signals matched",len({r["title"] for r in rows}))
    st.dataframe(pd.DataFrame([{"Company":r["company"],"Signal":r["title"],"Product":r["product"],"Relevance heuristic":r["relevance"],"Source":r["source"]} for r in rows]),use_container_width=True,hide_index=True)
    for i,row in enumerate(rows):
        with st.expander(f'{row["company"]} · {row["product"]} · heuristic {row["relevance"]}/100'):
            st.write("**Signal:**",row["title"])
            st.write("**Source:**",row["source"]," | **Published:**",row["published"] or "Not provided")
            if row["url"]:
                st.link_button("Open source headline",row["url"])
            st.write("**Evidence terms:**",", ".join(row["evidence_terms"]))
            st.write("**Opportunity hypothesis:**",row["hypothesis"])
            st.write("**Suggested validation:**",row["next_action"])
            if st.checkbox("Mark reviewed",key=f"review_{i}"):
                st.success("Marked for review in this session. No CRM action or external outreach performed.")
    st.caption("A high heuristic relevance score is not a win probability, purchase intent confirmation, or independently verified event.")
