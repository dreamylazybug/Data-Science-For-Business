"""Compact, interactive numeric correlation heatmap."""
import plotly.express as px
import streamlit as st
from Custplots import polish

def corell(df):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        st.warning('At least two numeric columns are needed.'); return
    st.subheader('Feature correlations')
    if numeric.shape[1] > 25:
        chosen = st.multiselect('Columns to compare (up to 25)', numeric.columns.tolist(),
                                default=numeric.columns[:12].tolist(), max_selections=25)
        numeric = numeric[chosen]
        if numeric.shape[1] < 2:
            st.info('Select at least two columns.'); return
    corr = numeric.corr()
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                    aspect='auto', labels=dict(color='Correlation'))
    fig.update_traces(hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>')
    fig = polish(fig, max(350, min(680, 28 * len(corr) + 120)))
    fig.update_layout(coloraxis_colorbar=dict(len=.7, thickness=12))
    fig.update_xaxes(tickangle=-35, side='bottom')
    st.plotly_chart(fig, use_container_width=True)
