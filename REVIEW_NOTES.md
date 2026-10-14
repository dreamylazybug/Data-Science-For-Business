# Analyti-Cult review and replacement

All original Python modules have full replacements. Added ModelUtils.py for shared train/test preprocessing.

## Important changes
- Local Mac file paths removed; default CSV read relative to MAIN.py.
- Pandas applymap and assignment to Arrow-backed string columns removed.
- Original data remains unchanged; no synthetic Index predictor.
- Train/test split happens before fitting imputers and one-hot encoding.
- Classification targets are label encoded and decoded in downloaded predictions.
- Fixed misaligned prediction DataFrames and incorrect XGBoost download.
- Clustering chooses a feasible k, scales numeric data and returns a downloadable CSV.
- Plots render in-memory; no saving to deployment filesystem.
- Optional ag-grid removed to reduce dependency conflicts.

## Caveats
- The modeling section is classification only, not continuous-target regression.
- High-cardinality categorical columns can make one-hot encoding expensive.
- Label encoding in Prepare_Data is for exploratory exports only; do not use it for supervised modeling.
- This is a static review, not a live Streamlit Cloud deployment test.

## Deploy
Upload all replacement .py files, requirements.txt, the CSV, samplebox.png, and .streamlit/config.toml to your repository. MAIN.py is the entry point.
