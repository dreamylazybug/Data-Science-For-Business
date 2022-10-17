#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM TKINTER PYTHON
# AUTHOR: LAXMI MULLAPUDI (lmullapu@cisco.com)
# LAST UPDATED: 10/29/2020
# CREATED:10/1/2020
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





# --------------

def corell(df):

    # Preparing data - specific to corellogram
    # Separating numeric & Categorical columns
    st.empty()
    dtf = df._get_numeric_data()
    dtf = dtf.fillna(0)  # Filling NAs
    dtg = df.select_dtypes(include=['object'])

    # Plotting Corellogram
    corre = dtf.corr()
    f = plt.figure(figsize=(16, 14))
    a = f.add_subplot(111)
    colormap = plt.cm.viridis
    a = sns.heatmap(corre, vmax=1.0, center=0,
                    square=True, linewidths=2, cmap=colormap, linecolor='white', annot=False)
    plt.title('Correlogram', fontsize=25)  # title with fontsize 20

    results_path = 'results.png'

    # Saving files to Root folder
    plt.savefig(results_path)
    st.image(results_path, caption='Correlations between your data columns')
    # Success Message
    st.success('Your Correlogram is stored in CorrelHeatmap.png file', icon="✅")



