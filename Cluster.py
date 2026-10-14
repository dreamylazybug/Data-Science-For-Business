"""K-means clustering with compact interactive elbow chart."""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from Custplots import polish, COLORS

def cluster(df):
    if df.empty or df.shape[1] == 0:
        st.warning('Select numeric columns to cluster.'); return None
    if len(df) < 3:
        st.warning('At least three rows are required.'); return None
    data = df.select_dtypes(include='number')
    if data.shape[1] != df.shape[1]:
        st.warning('Clustering requires numeric columns.'); return None
    if data.notna().sum().eq(0).any():
        st.warning('Remove columns containing only missing values.'); return None
    matrix = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(data))
    max_k = min(10, len(df)-1)
    ks = list(range(1, max_k + 1))
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(matrix).inertia_ for k in ks]
    st.subheader('Choosing the number of clusters')
    fig = px.line(x=ks, y=inertia, markers=True, labels={'x':'Number of clusters','y':'Inertia'},
                  color_discrete_sequence=COLORS)
    fig.update_traces(line_width=3, marker_size=8)
    st.plotly_chart(polish(fig, 320), use_container_width=True)
    k = st.slider('Number of clusters', 2, max_k, min(3, max_k))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)
    result = df.copy()
    result['cluster'] = labels
    st.dataframe(result.groupby('cluster').mean(numeric_only=True).round(2), use_container_width=True)
    return result
