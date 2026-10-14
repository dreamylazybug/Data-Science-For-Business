"""Analyti-Cult: Streamlit data exploration and modeling."""
from pathlib import Path
import pandas as pd
import streamlit as st
import Explore, Prep_Data, Correlations, Regression, Forest, XGBoost, Cluster, Custplots
from Download import to_excel

st.set_page_config(page_title='Analyti-Cult', layout='wide')
st.title('Analyti-Cult')
st.write('1. Load data → 2. Clean (optional) → 3. Explore → 4. Train models')
st.markdown('Free datasets: [Kaggle](https://www.kaggle.com/datasets)')

uploaded = st.sidebar.file_uploader('Choose a CSV data file', type=['csv'])
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.caption(f'Uploaded dataset: {uploaded.name}')
    else:
        path = Path(__file__).resolve().parent / 'HR Employee Attrition.csv'
        df = pd.read_csv(path)
        st.caption('Default dataset: HR Employee Attrition')
except (OSError, pd.errors.ParserError, UnicodeError) as exc:
    st.error(f'Unable to load CSV: {exc}')
    st.stop()
if df.empty or len(df.columns) == 0:
    st.error('The dataset is empty.'); st.stop()
if df.columns.duplicated().any():
    st.error('Duplicate column names detected; rename them in the CSV.'); st.stop()

st.sidebar.header('Explore and prepare')
if st.sidebar.button('Explore Data'):
    Explore.Load_Data(df)
if st.sidebar.button('Clean and Prepare Data'):
    cleaned = Prep_Data.Prepare_Data(df)
    st.download_button('Download prepared CSV', cleaned.to_csv(index=False).encode('utf-8'), 'Prepared_Data.csv', 'text/csv')
plttype = st.sidebar.selectbox('Select plot type', ['None','Scatter','Linear Regression','Distribution','Box Plot for Categorical Data'])
if plttype != 'None':
    Custplots.cstplt(df, plttype)
if st.sidebar.button('Correlations within data'):
    Correlations.corell(df)

st.sidebar.header('Predictive modeling (classification)')
st.sidebar.caption('Choose a categorical target. Models train on the original data; preprocessing is fitted on training rows only.')
target = st.sidebar.selectbox('Target variable', df.columns, index=df.columns.get_loc('Attrition') if 'Attrition' in df else 0)
features = st.sidebar.multiselect('Predictor variables', [c for c in df.columns if c != target])
testsz = st.sidebar.slider('Test set fraction', 0.2, 0.7, 0.3, 0.05)
lr = st.sidebar.slider('XGBoost learning rate', 0.05, 0.3, 0.1, 0.05)
maxdepth = st.sidebar.slider('Tree maximum depth', 2, 10, 5)
model_choice = st.sidebar.selectbox('Model', ['Logistic Regression','Random Forest','XGBoost'])
if st.sidebar.button('Train selected model'):
    if not features:
        st.warning('Select at least one predictor.')
    elif df[target].nunique(dropna=True) > min(30, max(2, len(df)//5)):
        st.warning('This looks like a continuous or high-cardinality target. These are classification models; choose a categorical target.')
    else:
        try:
            with st.spinner('Training model...'):
                if model_choice == 'Logistic Regression':
                    result = Regression.regres(df[features], df[target], testsz)
                elif model_choice == 'Random Forest':
                    result = Forest.forest(df[features], df[target], testsz, lr, maxdepth)
                else:
                    result = XGBoost.xgboost(df[features], df[target], testsz, lr, maxdepth)
            st.dataframe(result.head(100), use_container_width=True)
            st.download_button('Download model predictions (Excel)', to_excel(result), f'{model_choice.replace(" ", "_")}_predictions.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except (ValueError, TypeError) as exc:
            st.error(f'Model could not run: {exc}')

st.sidebar.header('Clustering')
cluster_cols = st.sidebar.multiselect('Numeric columns to cluster', df.select_dtypes(include='number').columns.tolist())
if st.sidebar.button('k-Means'):
    result = Cluster.cluster(df[cluster_cols])
    if result is not None:
        st.download_button('Download clusters (CSV)', result.to_csv(index=False).encode('utf-8'), 'cluster_Data.csv', 'text/csv')
