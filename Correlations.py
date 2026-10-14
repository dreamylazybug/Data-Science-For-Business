"""Numeric correlation heatmap."""
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def corell(df):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        st.warning('Select data with at least two numeric columns.')
        return
    fig, ax = plt.subplots(figsize=(min(16, max(7, numeric.shape[1])), 7))
    sns.heatmap(numeric.corr(), cmap='viridis', center=0, ax=ax)
    ax.set_title('Numeric feature correlations')
    st.pyplot(fig)
    plt.close(fig)
