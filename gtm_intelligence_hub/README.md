# Analyti-Cult GTM Intelligence Hub v0.2

An isolated Streamlit GTM intelligence prototype. The existing general-purpose Analyti-Cult application on `master` is unchanged.

## Run

```bash
pip install -r gtm_intelligence_hub/requirements.txt
streamlit run gtm_intelligence_hub/app.py
```

## Pipeline Generation v1

Open **Pipeline Generation** in the sidebar. Choose between fictional synthetic signals or opt-in Google News RSS headlines. For live mode, enter a company name and click **Discover external signals**. Select product areas, inspect linked source headlines, and review suggested opportunity hypotheses.

The signal discovery engine:
- Retrieves publicly available RSS headlines only on explicit user action.
- Matches reported events to configurable product areas using transparent keyword rules.
- Displays source, headline, relevance heuristic, opportunity hypothesis, and validation steps.
- Requires human validation; it does not write to CRM or initiate outreach.

**Limitations:** News headlines are not independently verified. Entity disambiguation, robust date parsing, source deduplication, ICP filtering, trained propensity scores, customer-specific product catalogs, and LLM inference are not implemented. Relevance scores are heuristic rankings, **not** probabilities. Other GTM modules remain synthetic demonstrations.

## Next engineering milestones

1. Add a company registry and robust entity matching to reduce false positives.
2. Implement source deduplication, date normalization, and signal freshness.
3. Support uploaded ICP and product catalog mappings.
4. Add optional local LLM extraction with structured evidence and strict citation validation.
5. Train and calibrate outcome-based ranking only after sufficient labeled data exists.
6. Add automated tests and CI.
