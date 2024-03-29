#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM STREAMLIT
# AUTHOR: LAXMI MULLAPUDI 
# LAST UPDATED " 2022/10/20
########################################################################################



import pandas as pd
import streamlit as st
import Explore
import Prep_Data
import Correlations
import Clean
import Regression
import XGBoost
import Cluster
import Custplots
import Forest
import xlsxwriter
import io


st.title('Analyti-Cult')

st.write('Step 1. Browse and Load data  \n Step 2. Optional - Clean Data  \n Step 3. Explore and Visualize  \n  Step 4. Click to deploy Models - Regression, XGBoost, Clustering ')
st.write("For free datasets, check out [link](https://www.kaggle.com/datasets)")
st.write('Default data set is HR Attrition from Kaggle, please select your file for analyzing your data')

df = pd.read_csv('/Users/aaru/Documents/Product Management/Analytics_Data_Science/HR Employee Attrition.csv')

# Read input Data
uploaded_file = st.sidebar.file_uploader("Choose the CSV data file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else :
    df = pd.read_csv('/Users/aaru/Documents/Product Management/Analytics_Data_Science/HR Employee Attrition.csv')
    
if uploaded_file is not None:
    # Run Load & Prep Data for analysis
    
    if st.sidebar.button('Explore Data'):
        Explore.Load_Data(df)
    
    
    
    # Data Preparation
    # Replacing blanks from data & col names, separating numeric & categorical variables, removing outliers 4 std dev away, label encoding categorical variables, merging numeric & categorical dataframe finally
    
    if st.sidebar.button('Clean and Prepare Data'):
       df3 = Prep_Data.Prepare_Data(df)
       st.dataframe(df3)
    
    
    plttype = st.sidebar.selectbox('Select the plot type', ('None','Scatter', 'Linear Regression', 'Distribution', 'Box Plot for Categorical Data'))
    if plttype != 'None' :
            Custplots.cstplt(df,plttype)
    
    if st.sidebar.button('Correlations within data'):
        Correlations.corell(df)
        
    
    st.sidebar.write('Predictive Modeling Section')
    
    
    # Selecting Features
    
    df3 = Prep_Data.Prepare_Data(df)
    
    # Select Target Variable & Dimensions
    choices = df3.columns.values.tolist()
    optiony = st.sidebar.selectbox(
        'Select the y-variable or the target for regression?', choices)
    optionx = st.sidebar.multiselect(
        'Select the x-variables or the predictors for regression', choices)
    
    # Storing them as predictors & targets data frames
    predictors = df3[optionx]
    targets = df3[optiony]
    
    with st.sidebar:
    
        with st.form(key='Regression'):
            st.write('Parameter Tuning for XGBoost')
            testsz = st.number_input('Test Sample Size', min_value=0.2, max_value=0.7, value=0.3, step=0.05)
            lr = st.number_input('Learning rate', min_value=0.05, max_value=0.3, value=0.1, step=0.05)
            maxdepth = st.number_input('Maximum Depth', min_value=3, max_value=10, value=5, step=1)
            submit_button = st.form_submit_button(label='Submit parameters')
    
    if st.sidebar.button('Logistic Regression'):
        df_xlsx = Regression.regres(predictors, targets, testsz)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_xlsx.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            st.download_button(label="Download Excel worksheets", data=buffer, file_name="regression.xlsx",mime="application/vnd.ms-excel")
        
    if st.sidebar.button('Random Forest'):
        dforest = Forest.forest(predictors, targets, testsz, lr, maxdepth)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            dforest.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            st.download_button(label="Download Excel worksheets", data=buffer, file_name="RandomForest.xlsx",mime="application/vnd.ms-excel")
        
    if st.sidebar.button('Gradient Boosting - XGBOOST'):
        XGBoost.xgboost(predictors, targets, testsz, lr, maxdepth)
        dboost = Forest.forest(predictors, targets, testsz, lr, maxdepth)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            dboost.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            st.download_button(label="Download Excel worksheets", data=buffer, file_name="XGBoost.xlsx",mime="application/vnd.ms-excel")
        
    st.sidebar.write('Data Clustering')
    
    
    
    # Pick the parameters to cluster 
    choicesclst = df3.columns.values.tolist()
    optionclst = st.sidebar.multiselect(
        'Select the columns to cluster data points', choicesclst)
    
    # Data for Clustering
    dclst = df3[optionclst]
    
    if st.sidebar.button('k-Means'):
        Cluster.cluster(dclst)

