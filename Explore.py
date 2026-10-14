"""Dataset exploration."""
import streamlit as st

def Load_Data(df):
    st.subheader('Data preview')
    st.write(f'{len(df):,} rows × {len(df.columns):,} columns')
    st.dataframe(df.head(500), use_container_width=True)
    st.subheader('Descriptive statistics')
    st.dataframe(df.describe(include='all').T.fillna(''), use_container_width=True)
    st.subheader('Missing values')
    st.dataframe(df.isna().sum().rename('Missing values').to_frame(), use_container_width=True)
    return df
