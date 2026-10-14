"""Optional PCA for already-split numeric feature matrices."""
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

def PCA(X_train, X_test):
    if st.selectbox('Apply PCA?', ('No','Yes')) == 'No':
        return X_train, X_test
    if X_train.shape[1] < 2 or len(X_train) < 2:
        st.warning('PCA needs at least two features and two training rows.')
        return X_train, X_test
    scaler = StandardScaler()
    train = scaler.fit_transform(X_train)
    test = scaler.transform(X_test)
    pca = SklearnPCA(n_components=2)
    return pca.fit_transform(train), pca.transform(test)
