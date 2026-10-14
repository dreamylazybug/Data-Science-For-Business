"""Dataset preparation for exploration (not model fitting)."""
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


def Prepare_Data(df, show_details=True):
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip().str.replace(' ', '_', regex=False)
    if data.columns.duplicated().any():
        raise ValueError('Duplicate column names after replacing spaces; rename columns first.')
    categorical = data.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()
    numeric = data.select_dtypes(include='number').columns.tolist()
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())
        data[col] = data[col].fillna(0)
    mappings = []
    for col in categorical:
        values = data[col].astype('string').fillna('(missing)').astype(str)
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(values)
        mappings.extend({'Column': col, 'Original': str(value), 'Encoded': int(i)}
                        for i, value in enumerate(encoder.classes_))
    if show_details:
        st.write(f'Encoded {len(categorical)} categorical columns.')
        st.dataframe(data.head(100), use_container_width=True)
        if mappings:
            with st.expander('Categorical value mappings'):
                st.dataframe(pd.DataFrame(mappings), use_container_width=True)
    return data
