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

# ---------------------------------------------------------------------------
# Browsing & loading data from # Youtube Link: https://www.youtube.com/watch?v=PgLjwl6Br0k
# Changes made by lmullapu to the orientation of the frames to make them relative
# ----------------------------------------------------------------------------




def cluster(dclstmer):
    try:
        st.write('k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. k-means clustering minimizes within-cluster variances (squared Euclidean distances)')
        # defining the kmeans function with initialization as k-means++
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=12, init='k-means++')

        # fitting the k means algorithm on scaled data
        kmeans.fit(dclstmer)

        # fitting multiple k-means algorithms and storing the values in an empty list
        SSE = []
        for cluster in range(1, 20):
            kmeans = KMeans(n_clusters=cluster, init='k-means++')
            kmeans.fit(dclstmer)
            SSE.append(kmeans.inertia_)

        # converting the results into a dataframe and plotting them
        frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})

        # Normalizing frame
        from sklearn import preprocessing

        x = frame.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        framnorm = pd.DataFrame(x_scaled)

        # picking the cluster
        slope = []
        val = []
        m = len(framnorm)-1
        n = len(framnorm.columns)
        for i in range(0, m):
            y1 = framnorm.iloc[i, 1]
            y2 = framnorm.iloc[i+1, 1]
            x1 = framnorm.iloc[i, 0]
            x2 = framnorm.iloc[i+1, 0]
            s = (y2-y1)/(x2-x1)
            s = abs(s)
            s = round(s, 0)
            if s == 0:
                break
            slope.append(s)
        os = i+1

        # Plotting data for optimal # of clusters
        f = plt.figure(figsize=(5, 5))
        a = f.add_subplot(111)
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        a = plt.plot(frame['Cluster'], frame['SSE'], marker='o')
        a = plt.axvline(x=os, color='black', linestyle='-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Inertia with # of Clusters (blue) & Optimal # of Clusters (black))',
                  fontsize=12)  # title with fontsize 20

        res_path_3 = 'Optimal_Clusters.png'

       # Saving files to Root folder
        plt.savefig(res_path_3)
        st.image(res_path_3, caption='Optimal Number of Clusters')
        plt.savefig("Optimal_Clusters.png")

        st.success(
            'The optimal number of clusters image has been saved as Optimal_Clusters.png', icon="✅")

        # k means using optimal # of clusters clusters and k-means++ initialization
        kmeans = KMeans(n_clusters=os, init='k-means++')
        kmeans.fit(dclstmer)
        pred = kmeans.predict(dclstmer)

        # Look at value count
        framenew = pd.DataFrame(dclstmer)
        framenew['cluster'] = pred
        framenew['cluster'].value_counts()
        clstchar = framenew.groupby(['cluster']).mean() 
        st.write('Summarizing characteristics of clusters')
        st.table(clstchar)
        framenew.to_csv('cluster_Data.csv')
        st.success(
            'Success, Your clusters are saved in .csv file named cluster_Data', icon="✅")

    except ValueError:
        st.error('Try reselecting colymns', icon="🚨")
    return None
    return None

