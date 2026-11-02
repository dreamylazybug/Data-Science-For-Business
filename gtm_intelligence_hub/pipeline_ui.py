"""Pipeline Generation: ICP, quality checks and optional local LLM."""
import pandas as pd
import streamlit as st
from pipeline_engine import PRODUCTS, assess_signal, demo_signals, fetch_news
from signal_quality import company_match, deduplicate, filter_icp, freshness_days, quality_flags
from local_llm import analyze_local

def render():
    st.subheader("AI Pipeline Generation | Signal-to-opportunity discovery")
    st.caption("Live source research + transparent relevance rules + optional local Ollama reasoning.")
    mode=st.radio("Signal source",["Synthetic demonstration","Live news headlines (opt-in)"],horizontal=True)
    icp=st.file_uploader("Optional ICP account list (CSV with company column)",type=["csv"],key="pipeline_icp")
    allowed=[]
    if icp is not None:
        try:
            data=pd.read_csv(icp)
            col=next((c for c in data.columns if c.lower().strip() in {"company","account","account_name"}),None)
            if col is None:
                st.warning("ICP CSV needs a company, account, or account_name column.")
            else:
                allowed=data[col].dropna().astype(str).tolist()
                st.caption(f"ICP filter active: {len(allowed)} named companies")
        except Exception as exc:
            st.error(f"Invalid ICP CSV: {exc}")
    if mode=="Synthetic demonstration":
        signals=demo_signals()
        st.info("All synthetic companies and events are fictional demonstrations.")
    else:
        company=st.text_input("Company to research",placeholder="Enter a company name")
        st.caption("Google News RSS is queried only when Discover is clicked. Headline matches do not verify company identity.")
        if st.button("Discover external signals",type="primary"):
            try:
                with st.spinner("Retrieving news headlines..."):
                    st.session_state["gtm_signals"]=fetch_news(company)
                    st.session_state["gtm_company"]=company
            except Exception as exc:
                st.error(f"News feed unavailable: {exc}")
        signals=st.session_state.get("gtm_signals",[]) if st.session_state.get("gtm_company")==company else []
    signals=deduplicate([s for s in signals if filter_icp(s,allowed)])
    products=st.multiselect("Product areas",list(PRODUCTS),default=list(PRODUCTS))
    include_uncertain=st.checkbox("Show unconfirmed company matches",value=False)
    rows=[]
    for signal in signals:
        flags=quality_flags(signal)
        if not signal.get("demo") and not include_uncertain and "Company name not confirmed in headline" in flags:
            continue
        for match in assess_signal(signal,{p:PRODUCTS[p] for p in products}):
            rows.append({**signal,**match,"flags":flags,"age_days":freshness_days(signal.get("published",""))})
    if not rows:
        st.info("No matching hypotheses. Try another company, widen product areas, or inspect unconfirmed matches.")
        return
    a,b=st.columns(2)
    a.metric("Opportunity hypotheses",len(rows))
    b.metric("Distinct signals",len({(r["company"],r["title"]) for r in rows}))
    st.dataframe(pd.DataFrame([{"Company":r["company"],"Headline":r["title"],"Product":r["product"],"Relevance heuristic":r["relevance"],"Age (days)":r["age_days"],"Quality flags":", ".join(r["flags"])} for r in rows]),use_container_width=True,hide_index=True)
    with st.expander("Optional local AI interpretation"):
        st.caption("Uses a locally running Ollama model. Sends headline and summary to localhost only, when requested. Model output is unverified interpretation.")
        use_llm=st.checkbox("Enable local LLM interpretation")
        model=st.text_input("Ollama model name",value="llama3.2",disabled=not use_llm)
    for i,row in enumerate(rows):
        with st.expander(f'{row["company"]} · {row["product"]} · relevance {row["relevance"]}/100'):
            st.write("**Headline:**",row["title"])
            st.write("**Source:**",row["source"]," | **Published:**",row["published"] or "Unknown")
            if row["url"]:
                st.link_button("Read source headline",row["url"])
            st.write("**Quality flags:**",", ".join(row["flags"]) or "None detected")
            st.write("**Matched terms:**",", ".join(row["evidence_terms"]))
            st.write("**Rule-based hypothesis:**",row["hypothesis"])
            st.write("**Next action:**",row["next_action"])
            if use_llm and st.button("Analyze with local LLM",key=f"llm_{i}"):
                try:
                    with st.spinner("Analyzing locally..."):
                        result=analyze_local(row,products,model=model)
                    st.session_state[f"llm_result_{i}"]=result
                except Exception as exc:
                    st.error(f"Local analysis failed: {exc}")
            if use_llm and f"llm_result_{i}" in st.session_state:
                st.json(st.session_state[f"llm_result_{i}"])
            if st.checkbox("Reviewed by human",key=f"review_{i}"):
                st.success("Marked reviewed for this session; no CRM write or outreach.")
    st.caption("Scores are heuristic relevance rankings, not purchase intent or calibrated probabilities. Validate sources and company identity before use.")
