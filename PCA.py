#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM TKINTER PYTHON
# AUTHOR: LAXMI MULLAPUDI (lmullapu@cisco.com)
# LAST UPDATED: 10/29/2020
# CREATED:10/1/2020
########################################################################################



import streamlit as st

# ---------------------------------------------------------------------------
# Browsing & loading data from # Youtube Link: https://www.youtube.com/watch?v=PgLjwl6Br0k
# Changes made by lmullapu to the orientation of the frames to make them relative
# ----------------------------------------------------------------------------



def PCA(X_train, X_test):
   
    
    option = st.selectbox('Do Principal Component Analysis - recommended when the number of predicting columns are more than 15',('Yes', 'No'))
    if option == 'Yes' :
        # performing preprocessing part
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Applying PCA function on training
        # and testing set of X component
        from sklearn.decomposition import PCA

        pca = PCA(n_components = 2)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)


