"""Analyti-Cult: general-purpose exploratory analysis and prediction studio."""
from pathlib import Path
import pandas as pd
import streamlit as st
import Explore, Prep_Data, Correlations, Cluster, Custplots
from ModelUtils import suggest_task, fit_supervised
from PredictionStudio import render

st.set_page_config(page_title='Analyti-Cult', page_icon='📊', layout='wide')
st.title('Analyti-Cult')
st.caption('Explore any tabular dataset · Visualize trends · Prepare data · Train and evaluate models')
st.markdown('Find practice datasets on [Kaggle](https://www.kaggle.com/datasets).')

uploaded = st.sidebar.file_uploader('Upload CSV dataset', type=['csv'])
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        dataset_name = uploaded.name
    else:
        df = pd.read_csv(Path(__file__).resolve().parent / 'HR Employee Attrition.csv')
        dataset_name = 'HR Employee Attrition (example)'
except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
    st.error(f'Could not load dataset: {exc}')
    st.stop()
if df.empty or not df.columns.is_unique:
    st.error('Dataset is empty or has duplicate column names. Please upload a valid CSV.')
    st.stop()

st.caption(f'Loaded: **{dataset_name}**')
col1, col2, col3 = st.columns(3)
col1.metric('Rows', f'{len(df):,}')
col2.metric('Columns', f'{len(df.columns):,}')
col3.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')

explore_tab, prepare_tab, charts_tab, model_tab, cluster_tab = st.tabs([
    'Explore', 'Prepare', 'Visualize', 'Prediction Studio', 'Clustering'])
with explore_tab:
    Explore.Load_Data(df)
with prepare_tab:
    st.caption('This optional export is for exploration. Models preprocess raw predictors separately after train/test splitting.')
    if st.button('Prepare dataset for download'):
        try:
            cleaned = Prep_Data.Prepare_Data(df)
            st.download_button('Download prepared CSV', cleaned.to_csv(index=False).encode('utf-8'),
                               'prepared_data.csv', 'text/csv')
        except (ValueError, TypeError) as exc:
            st.error(f'Preparation failed: {exc}')
with charts_tab:
    plot_type = st.selectbox('Chart type', ['Distribution', 'Scatter', 'Linear Regression', 'Box Plot for Categorical Data', 'Correlations'])
    if plot_type == 'Correlations':
        Correlations.corell(df)
    else:
        Custplots.cstplt(df, plot_type)
with model_tab:
    st.subheader('Configure a model')
    st.caption('Select a target and predictors. Classification predicts categories; regression predicts numeric values.')
    target = st.selectbox('Target column', df.columns.tolist(), index=df.columns.get_loc('Attrition') if 'Attrition' in df else 0)
    available = [c for c in df.columns if c != target]
    selected = st.multiselect('Predictor columns', available, default=[])
    recommendation = suggest_task(df[target])
    task = st.radio('Prediction task', ['Classification', 'Regression'], index=0 if recommendation == 'Classification' else 1,
                    horizontal=True, help='Automatically suggested based on target type; you can override.')
    if task == 'Classification':
        algorithm = st.selectbox('Algorithm', ['Logistic Regression', 'Random Forest', 'XGBoost'])
    else:
        algorithm = st.selectbox('Algorithm', ['Ridge Regression', 'Random Forest', 'XGBoost'])
    a, b, c = st.columns(3)
    test_fraction = a.slider('Test fraction', 0.15, 0.4, 0.25, 0.05)
    depth = b.slider('Maximum tree depth', 2, 12, 5)
    learning_rate = c.slider('XGBoost learning rate', 0.01, 0.3, 0.1, 0.01)
    if st.button('Train and evaluate', type='primary'):
        try:
            with st.spinner('Training model and evaluating holdout predictions...'):
                st.session_state['model_result'] = fit_supervised(df, target, selected, task, algorithm,
                                        test_fraction=test_fraction, max_depth=depth, learning_rate=learning_rate)
            st.session_state['model_config'] = (dataset_name, target, tuple(selected), task, algorithm, test_fraction, depth, learning_rate)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state.pop('model_result', None)
            st.error(f'Model could not run: {exc}')
    current_config = (dataset_name, target, tuple(selected), task, algorithm, test_fraction, depth, learning_rate)
    if st.session_state.get('model_config') == current_config and 'model_result' in st.session_state:
        render(st.session_state['model_result'])
    elif 'model_result' in st.session_state:
        st.info('Model settings changed. Click Train and evaluate to refresh the results.')
with cluster_tab:
    st.caption('Unsupervised clustering uses selected numeric columns; no target column is required.')
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cluster_features = st.multiselect('Numeric columns', numeric_cols, default=numeric_cols[:min(3,len(numeric_cols))])
    if cluster_features:
        try:
            result = Cluster.cluster(df[cluster_features])
            if result is not None:
                st.download_button('Download clusters (CSV)', result.to_csv(index=False).encode('utf-8'),
                                   'clusters.csv', 'text/csv')
        except (ValueError, TypeError) as exc:
            st.error(f'Clustering could not run: {exc}')
    else:
        st.info('Select at least one numeric column to cluster.')
