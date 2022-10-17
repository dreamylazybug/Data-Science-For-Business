#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM TKINTER PYTHON
# AUTHOR: LAXMI MULLAPUDI (lmullapu@cisco.com)
# LAST UPDATED: 10/29/2020
# CREATED:10/1/2020
########################################################################################


import seaborn as sns
import matplotlib.pylab as plt

import streamlit as st





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



