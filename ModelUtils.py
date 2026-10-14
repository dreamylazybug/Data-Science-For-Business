"""Shared leakage-safe model preprocessing and results."""
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def prepare_split(predictors, targets, testsz, scale=False):
    if predictors.empty or not len(predictors.columns):
        raise ValueError('Choose at least one predictor column.')
    if targets.isna().any():
        raise ValueError('Target has missing values; clean or exclude those rows first.')
    y_encoder = LabelEncoder()
    y = pd.Series(y_encoder.fit_transform(targets.astype(str)), index=targets.index, name=targets.name or 'Target')
    if y.nunique() < 2:
        raise ValueError('Target needs at least two classes.')
    if len(y) < 5:
        raise ValueError('At least five rows are required.')
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and int(len(y)*testsz) >= y.nunique() and (len(y)-int(len(y)*testsz)) >= y.nunique() else None
    X_train, X_test, y_train, y_test = train_test_split(predictors.copy(), y, test_size=testsz, random_state=42, stratify=stratify)
    numeric = predictors.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in predictors if c not in numeric]
    transformers = []
    if numeric:
        steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
        if scale: steps.append(('scale', StandardScaler()))
        transformers.append(('numeric', Pipeline(steps), numeric))
    if categorical:
        transformers.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='(missing)')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return X_train, X_test, y_train, y_test, y_encoder, preprocessor

def show_results(model, X_train, X_test, y_train, y_test, y_encoder):
    predictions = model.predict(X_test)
    st.write(f'Train accuracy: **{accuracy_score(y_train, model.predict(X_train)):.1%}**')
    st.write(f'Test accuracy: **{accuracy_score(y_test, predictions):.1%}**')
    st.text(classification_report(y_test, predictions, labels=list(range(len(y_encoder.classes_))), target_names=[str(x) for x in y_encoder.classes_], zero_division=0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
    st.pyplot(fig); plt.close(fig)
    result = X_test.copy()
    result['Actual Target'] = y_encoder.inverse_transform(y_test.to_numpy())
    result['Target Predicted'] = y_encoder.inverse_transform(predictions)
    return result
