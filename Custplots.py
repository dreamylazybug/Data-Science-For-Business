"""Compact, interactive Plotly visualizations for Analyti-Cult."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

COLORS = ['#2563eb', '#14b8a6', '#f59e0b', '#8b5cf6', '#ef4444', '#64748b']

def polish(fig, height=340):
    fig.update_layout(
        template='plotly_white', height=height,
        margin=dict(l=35, r=25, t=35, b=35),
        font=dict(family='Arial, sans-serif', size=12, color='#334155'),
        title=None, legend_title_text='',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor='white', font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor='#e2e8f0', zeroline=False)
    fig.update_yaxes(gridcolor='#edf2f7', zeroline=False)
    return fig

def cstplt(df, plttype):
    if df.empty or df.columns.empty:
        st.warning('No data available to plot.')
        return
    numeric = df.select_dtypes(include='number').columns.tolist()
    columns = df.columns.tolist()
    st.subheader(plttype)
    if plttype == 'Distribution':
        x = st.selectbox('Variable', columns, key='dist_x')
        data = df[[x]].dropna().copy()
        if data.empty:
            st.info('No non-missing values to plot.'); return
        if pd.api.types.is_numeric_dtype(data[x]):
            fig = px.histogram(data, x=x, nbins=35, color_discrete_sequence=COLORS)
            fig.update_layout(yaxis_title='Count', bargap=0.05)
        else:
            counts = data[x].astype(str).value_counts().head(15).sort_values()
            fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                         labels={'x':'Count','y':x}, color_discrete_sequence=COLORS)
        st.plotly_chart(polish(fig), use_container_width=True)
    elif plttype in ('Scatter', 'Linear Regression'):
        if len(numeric) < 2:
            st.warning('At least two numeric columns are required.'); return
        a, b = st.columns(2)
        x = a.selectbox('X variable', numeric, key='scatter_x')
        y = b.selectbox('Y variable', numeric, index=min(1, len(numeric)-1), key='scatter_y')
        if x == y:
            st.info('Choose different X and Y columns.'); return
        color = None
        if plttype == 'Scatter':
            candidate = st.selectbox('Color grouping (optional)', ['None'] + columns, key='scatter_hue')
            if candidate != 'None' and df[candidate].nunique(dropna=True) <= 15:
                color = candidate
            elif candidate != 'None':
                st.caption('Too many groups for a readable legend; showing one color.')
        cols = list(dict.fromkeys([x, y] + ([color] if color else [])))
        data = df[cols].dropna().sample(n=min(3000, len(df[cols].dropna())), random_state=42)
        if data.empty:
            st.info('No complete data points to plot.'); return
        if plttype == 'Linear Regression':
            fig = px.scatter(data, x=x, y=y, opacity=.5, color_discrete_sequence=COLORS)
            if len(data) >= 2 and data[x].nunique() > 1:
                coefficients = np.polyfit(data[x].astype(float), data[y].astype(float), 1)
                xs = np.linspace(data[x].min(), data[x].max(), 100)
                fig.add_scatter(x=xs, y=np.polyval(coefficients, xs), mode='lines',
                                name='Linear trend', line=dict(color='#ef4444', width=3))
        else:
            fig = px.scatter(data, x=x, y=y, color=color, opacity=.65,
                             color_discrete_sequence=COLORS)
        st.plotly_chart(polish(fig, 380), use_container_width=True)
    elif plttype == 'Box Plot for Categorical Data':
        if not numeric:
            st.warning('A numeric variable is required.'); return
        a, b = st.columns(2)
        y = a.selectbox('Numeric variable', numeric, key='box_y')
        x = b.selectbox('Group by', columns, key='box_x')
        if df[x].nunique(dropna=True) > 20:
            st.warning('Choose a grouping column with 20 or fewer unique values.'); return
        data = df[[x, y]].dropna()
        if data.empty:
            st.info('No complete rows to plot.'); return
        fig = px.box(data, x=x, y=y, color_discrete_sequence=COLORS, points='outliers')
        st.plotly_chart(polish(fig, 360), use_container_width=True)
