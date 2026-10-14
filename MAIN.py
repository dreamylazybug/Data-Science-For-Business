"""Analyti-Cult: general-purpose exploratory analysis and prediction studio."""
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import Explore, Cluster, PivotStudio, VisualizeStudio
from ModelUtils import suggest_task, fit_supervised
from PredictionStudio import render

st.set_page_config(page_title='Analyti-Cult', page_icon='📊', layout='wide')
st.title('Analyti-Cult')
st.caption('Explore any tabular dataset · Clean data · Build pivots · Visualize trends · Train and evaluate models')
st.markdown('Find practice datasets on [Kaggle](https://www.kaggle.com/datasets).')

# A built-in demonstration dataset is loaded automatically on first visit.
# Uploading a CSV temporarily replaces the demo for this browser session.
uploaded = st.sidebar.file_uploader('Upload your own CSV (optional)', type=['csv'])
st.sidebar.caption('Leave this empty to explore the built-in HR Attrition demo.')

base_dir = Path(__file__).resolve().parent
demo_candidates = [
    base_dir / 'HR Employee Attrition.csv',
    base_dir / 'HR-Employee-Attrition.csv',
    base_dir / 'data' / 'HR Employee Attrition.csv',
]

def read_csv_checked(source):
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError('The CSV contains no rows.')
    if not data.columns.is_unique:
        raise ValueError('The CSV contains duplicate column names.')
    return data

if uploaded is not None:
    try:
        df = read_csv_checked(uploaded)
        dataset_name = uploaded.name
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f'Uploaded CSV could not be read: {exc}')
        st.info('Remove the uploaded file to return to the demonstration dataset.')
        st.stop()
else:
    demo_file = next((p for p in demo_candidates if p.is_file()), None)
    if demo_file is None:
        st.error('The built-in demonstration CSV is missing from the deployed repository.')
        st.info('Upload **HR Employee Attrition.csv** alongside **MAIN.py** in the same GitHub branch, or upload your own CSV using the sidebar.')
        st.stop()
    try:
        df = read_csv_checked(demo_file)
        dataset_name = 'HR Employee Attrition (demo)'
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f'The demonstration CSV could not be read: {exc}')
        st.stop()

# Keep a cleaned working copy only for the current dataset.
import hashlib
source_key = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes() + str(tuple(df.columns)).encode()).hexdigest()
if st.session_state.get('clean_source_key') != source_key:
    st.session_state['clean_source_key'] = source_key
    st.session_state.pop('cleaned_dataset', None)
    st.session_state.pop('model_result', None)
    st.session_state.pop('model_config', None)
    st.session_state.pop('pivot_result', None)
    st.session_state.pop('pivot_signature', None)
    st.session_state.pop('viz_requested_source', None)
    for key in list(st.session_state):
        if key.startswith(('viz_', 'pivot_')):
            st.session_state.pop(key, None)

st.sidebar.divider()
use_cleaned = st.sidebar.checkbox('Use cleaned data in analysis',
    value=False, disabled='cleaned_dataset' not in st.session_state,
    help='Applies the cleaned copy to Explore, Visualize, Prediction Studio and Clustering.')
raw_df = df
if st.session_state.get('pivot_cleaned_setting') != use_cleaned:
    st.session_state.pop('pivot_result', None)
    st.session_state.pop('pivot_signature', None)
    st.session_state['pivot_cleaned_setting'] = use_cleaned
if use_cleaned and 'cleaned_dataset' in st.session_state:
    df = st.session_state['cleaned_dataset'].copy()

st.caption(f'Loaded: **{dataset_name}**' + (' · cleaned copy active' if use_cleaned else ' · original data active'))
col1, col2, col3 = st.columns(3)
col1.metric('Rows', f'{len(df):,}')
col2.metric('Columns', f'{len(df.columns):,}')
col3.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')

explore_tab, clean_tab, pivot_tab, charts_tab, model_tab, cluster_tab = st.tabs([
    'Explore', 'Clean Data', 'Pivot Studio', 'Visualize', 'Prediction Studio', 'Clustering'])
with explore_tab:
    Explore.Load_Data(df)
with clean_tab:
    st.subheader('Clean Data')
    st.caption('Create a cleaned copy of the current dataset. The original is never overwritten.')
    st.write(f'Original: **{len(raw_df):,} rows**, **{raw_df.shape[1]:,} columns**, '
             f'**{int(raw_df.isna().sum().sum()):,} missing cells**, '
             f'**{int(raw_df.duplicated().sum()):,} duplicate rows**.')
    with st.form('cleaning_options'):
        drop_duplicates = st.checkbox('Remove duplicate rows', value=True)
        missing_method = st.selectbox('Handle missing values',
            ['Keep missing values', 'Drop rows containing missing values',
             'Fill numeric with median; categorical with mode',
             'Fill numeric with mean; categorical with mode'])
        drop_empty_cols = st.checkbox('Remove entirely empty columns', value=True)
        submitted = st.form_submit_button('Create cleaned copy', type='primary')
    if submitted:
        cleaned = raw_df.copy()
        if drop_empty_cols:
            cleaned = cleaned.dropna(axis=1, how='all')
        if drop_duplicates:
            cleaned = cleaned.drop_duplicates()
        if missing_method == 'Drop rows containing missing values':
            cleaned = cleaned.dropna()
        elif missing_method.startswith('Fill numeric'):
            for col in cleaned.columns:
                if not cleaned[col].isna().any():
                    continue
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    nonnull = cleaned[col].dropna()
                    if not nonnull.empty:
                        value = nonnull.mean() if 'mean' in missing_method else nonnull.median()
                        cleaned[col] = cleaned[col].fillna(value)
                else:
                    modes = cleaned[col].mode(dropna=True)
                    if not modes.empty:
                        cleaned[col] = cleaned[col].fillna(modes.iloc[0])
        if cleaned.empty or cleaned.shape[1] == 0:
            st.error('Cleaning produced an empty dataset. Choose less restrictive options.')
        else:
            st.session_state['cleaned_dataset'] = cleaned
            st.success('Cleaned copy created. Enable “Use cleaned data in analysis” in the sidebar to analyze it.')
    if 'cleaned_dataset' in st.session_state:
        cleaned = st.session_state['cleaned_dataset']
        c1, c2, c3 = st.columns(3)
        c1.metric('Cleaned rows', f'{len(cleaned):,}')
        c2.metric('Cleaned columns', f'{cleaned.shape[1]:,}')
        c3.metric('Remaining missing cells', f'{int(cleaned.isna().sum().sum()):,}')
        st.dataframe(cleaned.head(50), use_container_width=True)
        st.download_button('Download cleaned CSV', cleaned.to_csv(index=False).encode('utf-8'),
                           file_name='cleaned_data.csv', mime='text/csv')

with pivot_tab:
    PivotStudio.render(df, source_signature=(source_key, use_cleaned))
with charts_tab:
    VisualizeStudio.render(raw_df, st.session_state.get('cleaned_dataset'),
        st.session_state.get('pivot_result'), st.session_state.get('pivot_signature'))
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
    a, b = st.columns(2)
    test_fraction = a.slider('Test fraction', 0.15, 0.4, 0.20, 0.05)
    depth = b.slider('Maximum tree depth', 2, 12, 5) if algorithm in ('Random Forest', 'XGBoost') else 5
    learning_rate = st.slider('XGBoost learning rate', 0.01, 0.3, 0.1, 0.01) if algorithm == 'XGBoost' else 0.1
    class_balance = st.checkbox('Balance class weights during training', value=False,
        help='Useful when target classes are uneven. May improve minority-class recall at the expense of precision.') if task == 'Classification' else False
    if st.button('Train and evaluate', type='primary'):
        try:
            with st.spinner('Training model and evaluating holdout predictions...'):
                st.session_state['model_result'] = fit_supervised(df, target, selected, task, algorithm,
                                        test_fraction=test_fraction, max_depth=depth, learning_rate=learning_rate, class_balance=class_balance)
            st.session_state['model_config'] = (dataset_name, target, tuple(selected), task, algorithm, test_fraction, depth, learning_rate, class_balance, use_cleaned)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state.pop('model_result', None)
            st.error(f'Model could not run: {exc}')
    current_config = (dataset_name, target, tuple(selected), task, algorithm, test_fraction, depth, learning_rate, class_balance, use_cleaned)
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
