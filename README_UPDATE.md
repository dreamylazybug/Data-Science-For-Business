# Analyti-Cult — Class balance and prediction diagnostics update

Replace only these files in your `analyti-cult-fixes` branch:
- `MAIN.py`
- `ModelUtils.py`
- `PredictionStudio.py`

Keep all other files, including the bundled `HR Employee Attrition.csv`, in place. The default demonstration dataset remains automatically loaded. Uploaded CSV files still override it.

New options:
- Balance class weights during training (classification only): sklearn class_weight for Logistic Regression / Random Forest; sample_weight for XGBoost.
- Binary decision threshold control on holdout probabilities.
- Per-class precision, recall, F1, support, class imbalance warning, and zero-recall warning.
- Algorithm-specific depth and learning-rate controls.

Note: threshold tuning on the held-out test set is exploratory, not a valid way to select a final threshold and report unbiased performance. For production, choose the threshold on separate validation data and evaluate once on untouched test data.

Smoke tested Logistic Regression and Random Forest with bundled HR dataset. XGBoost UI integration and deployed Streamlit runtime have not been fully tested.
