# Analyti-Cult GTM Intelligence Hub v0.3

Streamlit GTM intelligence research prototype, developed in a dedicated branch. Existing `master` remains unchanged.

## Start the application

```bash
pip install -r gtm_intelligence_hub/requirements.txt
streamlit run gtm_intelligence_hub/app.py
```

## Pipeline Generation v2

Select **Pipeline Generation**. Choose fictional demonstration signals or opt-in live Google News RSS headline retrieval. Enter a company name and click **Discover external signals** to fetch live headlines.

Capabilities:
- Transparent product relevance heuristics for cloud, identity, data, and network security.
- Conservative headline company-name checks, duplicate removal, publication-age indicators, and quality flags.
- Optional ICP CSV upload with a `company`, `account`, or `account_name` column. Exact normalized account matching is used.
- Optional **local Ollama** interpretation. Install Ollama separately, pull a compatible model (e.g. `ollama pull llama3.2`), and start its local service. In the UI enable local LLM interpretation and click **Analyze with local LLM** for an individual signal.
- Model JSON is checked for an allowed product/event and a verbatim evidence substring. This is limited validation, not independent factual verification.
- Human review indicators only. No CRM updates or outreach.

## Test

```bash
python -m unittest discover -s gtm_intelligence_hub/tests -v
```

## Limitations and security

- Live RSS headlines are unverified, can be irrelevant, and may be incomplete. News is fetched only on user request. The app does not fetch full articles.
- Local LLM calls are restricted to localhost:11434 and are opt-in; ensure Ollama is running locally. No external hosted LLM is used.
- Source text is untrusted. The prompt tells the model not to follow instructions in news content, but model outputs still require human review.
- Product catalog is code-configured, not yet editable from the UI. ICP CSV filters named accounts but is not a full firmographic ICP engine.
- Scores are heuristic relevance indicators, not calibrated probabilities or verified buying intent.
- Company matching is conservative string matching, not a validated corporate-entity registry. Deduplication is basic.
- Tests are committed but have not been executed in CI. The Streamlit app has not been verified end-to-end in this session.
- Other GTM modules remain synthetic demonstrations.

## Next milestones

Introduce full entity resolution and source provenance, customizable product catalog, multi-source intelligence, stronger LLM output validation, and end-to-end CI tests.

## New prospect discovery (v0.4)

Open **Pipeline Generation → Discover new prospects (opt-in)**, select a market event, then click **Scan market for prospects**. The module queries a Google News RSS search and extracts candidate organization names only from headlines following a conservative `Company announces/raises/expands/...` pattern. It lists candidates even when no product match is found; qualifying candidates also flow into product relevance hypotheses.

This is **candidate extraction**, not verified entity resolution or comprehensive prospect discovery. The pattern misses many legitimate headlines and can misidentify publishers, product names, and other named entities. Results must be validated against authoritative company information before sales action. Live access requires outbound network connectivity and the feed's continued availability.

Run tests:
```bash
python -m unittest discover -s gtm_intelligence_hub/tests -v
```

## Entity resolution v1 (v0.5)

Upload an optional **company registry CSV** in Pipeline Generation. Required column: `company`. Optional: `domain`, `aliases` (pipe-separated names), `industry`, `segment`.

Example:
```csv
company,domain,aliases,industry,segment
Acme Cloud Inc,acme.example,Acme Cloud|Acme,Software,Commercial
Northstar Labs,northstar.example,Northstar,Healthcare,Enterprise
```

Exact normalized names and aliases are compared against the candidate names extracted from headlines. Duplicate matches are marked **Ambiguous**, not silently selected. The optional **Require unambiguous registry match for opportunities** toggle excludes ambiguous and unresolved names from product hypotheses. Uploaded domains can be used by the resolver when independently supplied as a hint; news URLs are **not** treated as corporate domains.

This is local matching against user-supplied data, not independent entity verification. Company registries must be curated; parent/subsidiary structures, ticker/LEI validation, and authoritative third-party enrichment remain future work.

Run tests:
```bash
python -m unittest discover -s gtm_intelligence_hub/tests -v
```

## Firmographic ICP fit (v0.6)

Upload a company registry with `company` and optional `industry`, `segment`, `country`, `technologies` (pipe-separated), and `employees`. Define target ICP attributes in the **Ideal customer profile (ICP) criteria** expander.

The system scores exact attribute overlaps with weights: industry 30, segment 20, country 20, technologies 20, employee count 10. Only available, configured attributes enter the denominator. A 100% fit with 30/100 evidence coverage is therefore **not** equivalent to a 100% fit with complete evidence. Unknown fields are shown separately. Scores are descriptive heuristics, not buying probabilities. No external firmographic data vendor is connected.

Example CSV:
```csv
company,domain,aliases,industry,segment,country,technologies,employees
Acme Cloud Inc,acme.example,Acme Cloud|Acme,Software,Commercial,US,Azure|Python,500
```

Run tests:
```bash
python -m unittest discover -s gtm_intelligence_hub/tests -v
```

## Continuous integration (v0.7)

GitHub Actions workflow: `.github/workflows/gtm-ci.yml`. Runs on commits to `gtm-intelligence-hub` and supports manual dispatch. It installs requirements, checks Python syntax, runs offline unit tests, and executes Streamlit AppTest smoke tests for the Command Center and Pipeline Generation demo mode.

Check the **Actions** tab for the latest result. CI does not validate external news availability, Ollama connectivity, or production deployment. Passing CI is a prerequisite for further release testing, not a production certification.

Local equivalent:
```bash
python -m pip install -r gtm_intelligence_hub/requirements.txt
python -m compileall -q gtm_intelligence_hub
python -m unittest discover -s gtm_intelligence_hub/tests -v
```

## Creation Intelligence — Phase 1: Creation Pulse

Open **Creation Intelligence** in the app sidebar. This phase uses deterministic fictional opportunity-creation events (no CRM connection) and shows QTD gross creation, aligned prior-year creation, linear pace against a configurable quarterly creation target, a simple run-rate quarter-end estimate, and breakdowns by product, segment and sales motion.

**Definitions:** Created pipeline is measured at the opportunity's initial creation amount. Subsequent value changes are not counted as new creation. Prior-year comparisons align the number of elapsed weekdays rather than calendar dates; holidays and fiscal calendars are not yet supported. The quarter-end estimate is a simple weekday run-rate extrapolation, not a trained forecast. Target is a *gross pipeline creation* target, not a bookings target.

**Separate models:** Current-quarter bookings sufficiency and future-quarter pipeline health have separate documented input contracts and are not yet implemented. No conversion rate, win probability or confidence interval is fabricated.

Tests: `python -m unittest discover -s gtm_intelligence_hub/tests -v`.

## Salesforce-style synthetic data

Creation Intelligence automatically uses `sfdc_mock.generate(as_of=...)`: a deterministic, fictional three-year rolling history with roughly 3–4 new opportunities each week. Records have all 38 requested fields, including opportunity and account codes, product hierarchy, SKU, route to market, forecast category, stage, specialist and POV details, discounts, segment/industry, competitors, sales hierarchy, TCV motion components and line-item dates. One row represents one SKU line item; `Opportunity code` identifies the opportunity, so dashboards must roll up line-item TCV and count unique opportunities. The demo defines NetNew/Upsell/Renew at product_level2; a real deployment needs configurable product-grain entitlement logic and historical snapshots.

Product-specific *generating probabilities* span 24–37%, with observed closed-only win rates varying due to random sampling. Open deals are excluded from the closed-win denominator. This is **not** an estimate from real Salesforce data. Opportunity creation values are held fixed in the generator. Opportunity stage and close outcomes are simulated as of the selected date, not reconstructed from historical snapshots. The default generator is deterministic for a given as-of date; changing the date regenerates the rolling history rather than querying a frozen historical database. Do not use it for backtesting or validated forecasts.

In Creation Intelligence, expand **Demo dataset · Salesforce-style opportunity line items** to inspect or download the CSV. No upload is required.
