"""Interactive charts using the original (unencoded) dataset."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def cstplt(df, plttype):
    if df.empty or df.columns.empty:
        st.warning('No data available to plot.')
        return
    columns = df.columns.tolist()
    numeric = df.select_dtypes(include='number').columns.tolist()
    if plttype == 'Distribution':
        x = st.selectbox('Variable', columns, key='dist_x')
        if st.button('Start plotting', key='dist_go'):
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.histplot(data=df, x=x, ax=ax)
            ax.tick_params(axis='x', rotation=30)
            st.pyplot(fig); plt.close(fig)
    elif plttype in ('Scatter', 'Linear Regression'):
        if len(numeric) < 2:
            st.warning('Choose a dataset with at least two numeric columns.'); return
        x = st.selectbox('X variable', numeric, key='scatter_x')
        y = st.selectbox('Y variable', numeric, index=min(1,len(numeric)-1), key='scatter_y')
        hue = st.selectbox('Color grouping (optional)', ['None'] + columns, key='scatter_hue') if plttype == 'Scatter' else 'None'
        if st.button('Start plotting', key='scatter_go'):
            fig, ax = plt.subplots(figsize=(9, 5))
            if plttype == 'Linear Regression':
                sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={'alpha':0.5})
            else:
                sns.scatterplot(data=df, x=x, y=y, hue=None if hue=='None' else hue, ax=ax)
            st.pyplot(fig); plt.close(fig)
    elif plttype == 'Box Plot for Categorical Data':
        if not numeric:
            st.warning('A numeric Y column is required.'); return
        y = st.selectbox('Numeric Y variable', numeric, key='box_y')
        x = st.selectbox('Category X variable', columns, key='box_x')
        if st.button('Start plotting', key='box_go'):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df, x=x, y=y, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig); plt.close(fig)
            sample = Path(__file__).parent / 'samplebox.png'
            if sample.exists(): st.image(str(sample), caption='How to read a box plot')
