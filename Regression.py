"""Logistic classification model."""
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ModelUtils import prepare_split, show_results

def regres(predictors, targets, testsz):
    st.subheader('Logistic Regression')
    X_train, X_test, y_train, y_test, labels, prep = prepare_split(predictors, targets, testsz, scale=True)
    model = Pipeline([('preprocess', prep), ('classifier', LogisticRegression(max_iter=2000, random_state=42))])
    model.fit(X_train, y_train)
    return show_results(model, X_train, X_test, y_train, y_test, labels)
