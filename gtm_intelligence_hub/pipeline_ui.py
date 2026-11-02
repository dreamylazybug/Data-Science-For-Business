"""Pipeline Generation: ICP, quality checks and optional local LLM."""
import pandas as pd
import streamlit as st
from pipeline_engine import PRODUCTS, assess_signal, demo_signals, fetch_news
from signal_quality import company_match, deduplicate, filter_icp, freshness_days, quality_flags
from local_llm import analyze_local
from prospect_discovery import TOPICS, fetch_topic
from entity_resolution import registry_from_frame, resolve
from icp_scoring import fit

def render():
    st.subheader("AI Pipeline Generation | Signal-to-opportunity discovery")
    st.caption("Live source research + transparent relevance rules + optional local Ollama reasoning.")
    mode=st.radio("Signal source",["Synthetic demonstration","Live news headlines (opt-in)","Discover new prospects (opt-in)"],horizontal=True)
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
    registry_upload=st.file_uploader("Optional company registry CSV (company, domain, aliases, industry, segment)",type=["csv"],key="company_registry")
    registry=[]
    if registry_upload is not None:
        try:
            registry=registry_from_frame(pd.read_csv(registry_upload))
            st.caption(f"Registry loaded: {len(registry)} entities. Matches are user-supplied, not independently verified.")
        except Exception as exc:
            st.error(f"Registry error: {exc}")
    with st.expander("Ideal customer profile (ICP) criteria"):
        industry=st.text_input("Target industries (comma-separated)",value="")
        segment=st.text_input("Target segments (comma-separated)",value="")
        country=st.text_input("Target countries (comma-separated)",value="")
        technology=st.text_input("Target technologies (comma-separated)",value="")
        employee_range=st.checkbox("Filter by employee count")
        min_employees=st.number_input("Minimum employees",min_value=0,value=100,disabled=not employee_range)
        max_employees=st.number_input("Maximum employees",min_value=0,value=5000,disabled=not employee_range)
    profile={"industry":industry,"segment":segment,"country":country,"technologies":technology,
             "min_employees":int(min_employees) if employee_range else None,
             "max_employees":int(max_employees) if employee_range else None}
    if mode=="Synthetic demonstration":
        signals=demo_signals()
        st.info("All synthetic companies and events are fictional demonstrations.")
    elif mode=="Discover new prospects (opt-in)":
        topic=st.selectbox("Market event to scan",list(TOPICS))
        st.caption("Searches public headlines for candidate company names. Extraction is conservative and does not establish corporate identity.")
        if st.button("Scan market for prospects",type="primary"):
            try:
                with st.spinner("Scanning public news..."):
                    st.session_state["prospect_signals"]=fetch_topic(topic)
                    st.session_state["prospect_topic"]=topic
            except Exception as exc:
                st.error(f"Prospect scan failed: {exc}")
        signals=st.session_state.get("prospect_signals",[]) if st.session_state.get("prospect_topic")==topic else []
        if signals:
            st.info(f"Found {len(set(s['company'] for s in signals))} unverified company candidates. Validate company identities before sales use.")
            st.dataframe(pd.DataFrame([{"Candidate":s["company"],"Headline":s["title"],"Source":s["source"],"Published":s["published"]} for s in signals]),use_container_width=True,hide_index=True)
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
    resolutions={i:resolve(s["company"],registry) for i,s in enumerate(signals)}
    if registry:
        st.subheader("Entity resolution review")
        st.dataframe(pd.DataFrame([{"Candidate":s["company"],"Status":resolutions[i]["status"],
                                   "Registry entity":resolutions[i]["entity"]["company"] if resolutions[i]["entity"] else "",
                                   "Industry":resolutions[i]["entity"]["industry"] if resolutions[i]["entity"] else "",
                                   "Segment":resolutions[i]["entity"]["segment"] if resolutions[i]["entity"] else "",
                                   "Match method":resolutions[i]["method"]}
                                  for i,s in enumerate(signals)]),use_container_width=True,hide_index=True)
    require_match=st.checkbox("Require unambiguous registry match for opportunities",value=bool(registry),disabled=not registry)
    products=st.multiselect("Product areas",list(PRODUCTS),default=list(PRODUCTS))
    include_uncertain=st.checkbox("Show unconfirmed company matches",value=False)
    rows=[]
    for signal_index,signal in enumerate(signals):
        entity_result=resolutions[signal_index]
        if require_match and entity_result["status"]!="Registry match — unverified":
            continue
        icp_fit=fit(entity_result["entity"],profile) if entity_result["entity"] else {"score":None,"coverage":0,"details":[]}
        flags=quality_flags(signal)
        if not signal.get("demo") and not signal.get("discovery") and not include_uncertain and "Company name not confirmed in headline" in flags:
            continue
        for match in assess_signal(signal,{p:PRODUCTS[p] for p in products}):
            rows.append({**signal,**match,"flags":flags,"age_days":freshness_days(signal.get("published","")),"resolution":entity_result,"icp_fit":icp_fit})
    if mode=="Discover new prospects (opt-in)" and signals and not rows:
        st.warning("Candidates were discovered, but none met the product relevance rules. Review the candidate table above or select additional product areas.")
    if not rows:
        st.info("No matching hypotheses. Try another company, widen product areas, or inspect unconfirmed matches.")
        return
    a,b=st.columns(2)
    a.metric("Opportunity hypotheses",len(rows))
    b.metric("Distinct signals",len({(r["company"],r["title"]) for r in rows}))
    st.dataframe(pd.DataFrame([{"Company":r["company"],"Headline":r["title"],"Product":r["product"],"Relevance heuristic":r["relevance"],"ICP fit":r["icp_fit"]["score"],"ICP evidence coverage":r["icp_fit"]["coverage"],"Age (days)":r["age_days"],"Quality flags":", ".join(r["flags"])} for r in rows]),use_container_width=True,hide_index=True)
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
            if row.get("discovery"):
                st.warning("Candidate company name extracted from headline; not verified against an entity registry.")
            st.write("**Entity resolution:**",row["resolution"]["status"]," — ",row["resolution"]["method"])
            if row["resolution"]["status"]=="Ambiguous":
                st.warning("Multiple registry entities match this name. No automatic assignment.")
            st.write("**ICP fit:**",row["icp_fit"]["score"] if row["icp_fit"]["score"] is not None else "Unknown",
                     " | **Evaluated weight:**",row["icp_fit"]["coverage"],"/ 100")
            if row["icp_fit"]["details"]:
                st.dataframe(pd.DataFrame(row["icp_fit"]["details"],columns=["Criterion","Result"]),hide_index=True)
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
