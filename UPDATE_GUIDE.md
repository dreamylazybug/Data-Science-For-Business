# Analyti-Cult — General-Purpose Prediction Studio Update

Upload the **contents** of this folder into the `analyti-cult-fixes` branch at repository root, not the enclosing folder. Files with matching names should be replaced. `PredictionStudio.py` is a new module. Keep `.streamlit/config.toml` in its subfolder.

Key changes: general-purpose classification/regression selector, task-appropriate algorithms, train-only preprocessing, holdout metrics, compact Plotly evaluation charts, filtered CSV prediction export with generated row IDs, and tabs for exploring/cleaning/visualizing/clustering.

Test with the default HR dataset and at least one unrelated CSV. A CSV does not guarantee a meaningful ML task; user must choose appropriate predictors and target. Current version handles mixed numeric and categorical data, but very high-cardinality categorical columns and large datasets can be slow. Datetime/text fields are treated as categories, not automatically featurized. Predictions shown are for holdout rows, not all rows or future records. Pseudonymous row IDs do not anonymize any upstream dataset.

This package was syntax-compiled, but full Streamlit Cloud execution must still be verified after deployment.
