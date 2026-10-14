"""K-means clustering with elbow chart and CSV results."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def cluster(df):
    if df.empty or df.shape[1] == 0:
        st.warning('Select at least one numeric column to cluster.'); return None
    if len(df) < 3:
        st.warning('At least three rows are required.'); return None
    data = df.select_dtypes(include='number')
    if data.shape[1] != df.shape[1]:
        st.warning('Clustering requires numeric columns.'); return None
    matrix = StandardScaler().fit_transform(SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(data))
    max_k = min(10, len(df)-1)
    ks = list(range(1, max_k+1))
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(matrix).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, inertia, marker='o')
    ax.set(xlabel='Number of clusters', ylabel='Inertia', title='Elbow chart')
    st.pyplot(fig); plt.close(fig)
    k = st.slider('Number of clusters', 2, max_k, min(3,max_k))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)
    result = df.copy(); result['cluster'] = labels
    st.dataframe(result.groupby('cluster').mean(numeric_only=True), use_container_width=True)
    return result
