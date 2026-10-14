"""XGBoost classification model."""
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from ModelUtils import prepare_split, show_results

def xgboost(predictors, targets, testsz, lr, maxdepth):
    st.subheader('XGBoost')
    X_train, X_test, y_train, y_test, labels, prep = prepare_split(predictors, targets, testsz)
    n_classes = len(labels.classes_)
    config = {'objective':'binary:logistic'} if n_classes == 2 else {'objective':'multi:softprob', 'num_class':n_classes}
    model = Pipeline([('preprocess', prep), ('classifier', XGBClassifier(n_estimators=200, learning_rate=float(lr), max_depth=int(maxdepth), random_state=42, n_jobs=2, tree_method='hist', **config))])
    model.fit(X_train, y_train)
    return show_results(model, X_train, X_test, y_train, y_test, labels)
