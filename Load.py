#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM STREAMLIT
# AUTHOR: LAXMI MULLAPUDI 
# LAST UPDATED " 2022/10/20
########################################################################################


import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
import xgboost as xgb
import math
#from apyori import apriori
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode



def Load_Data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # https://towardsdatascience.com/make-dataframes-interactive-in-streamlit-c3d0c4f84ccb
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
        gb.configure_side_bar()  # Add a sidebar
        # Enable multi-row selection
        gb.configure_selection('multiple', use_checkbox=True,
                               groupSelectsChildren="Group checkbox select children")
        gridOptions = gb.build()
    
        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT',
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=True,
            height=350,
            width='100%',
            reload_data=True
        )
        return(df) 
    else :
        st.error('Error-Try loading another file or change to .csv mode', icon="🚨")



  