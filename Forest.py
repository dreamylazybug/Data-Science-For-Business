"""Random forest classification model."""
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ModelUtils import prepare_split, show_results

def forest(predictors, targets, testsz, lr=None, maxdepth=5):
    st.subheader('Random Forest')
    X_train, X_test, y_train, y_test, labels, prep = prepare_split(predictors, targets, testsz)
    model = Pipeline([('preprocess', prep), ('classifier', RandomForestClassifier(n_estimators=200, max_depth=int(maxdepth), random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    return show_results(model, X_train, X_test, y_train, y_test, labels)
