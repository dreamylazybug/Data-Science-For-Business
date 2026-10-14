"""Legacy data-loading entry point."""
import pandas as pd
import streamlit as st
from Explore import Load_Data as display_data

def Load_Data(uploaded_file):
    if uploaded_file is None:
        st.warning('Please select a CSV file.')
        return None
    df = pd.read_csv(uploaded_file)
    display_data(df)
    return df
