#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM TKINTER PYTHON
# AUTHOR: LAXMI MULLAPUDI (lmullapu@cisco.com)
# LAST UPDATED: 10/29/2020
# CREATED:10/1/2020
########################################################################################


import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
# ---------------------------------------------------------------------------
# Browsing & loading data from # Youtube Link: https://www.youtube.com/watch?v=PgLjwl6Br0k
# Changes made by lmullapu to the orientation of the frames to make them relative
# ----------------------------------------------------------------------------



def regres(predictors, targets, testsz):
    
    try:
        # Preparing data
        features = predictors.columns.tolist()
        trainsz = 1.0 - testsz
        X_train, X_test, y_train, y_test = train_test_split(
            predictors, targets, random_state=1, test_size=testsz, train_size=trainsz)
        ds = LogisticRegression(random_state=0)
        ds.fit(X_train, y_train)
        y_pred = ds.predict(X_test)

        from sklearn.metrics import confusion_matrix
        sc1 = ds.score(X_train, y_train)*100

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        sc2 = ds.score(X_test, y_test)*100

        dk = (cm.tolist())
        f = plt.figure(figsize=(10, 7))
        a = f.add_subplot(111)
        accdata = {'Data': ['Train Data', 'Test Data'], 'Accuracy': [sc1, sc2]}
        dct = pd.DataFrame(accdata, columns=['Data', 'Accuracy'])
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        a = sns.barplot(x="Data", y="Accuracy", palette="Blues",  data=dct)
        a.set(xlabel='Data Set', ylabel='% Accuracy')
        plt.title('Regression based Prediction',
                  fontsize=12)  # title with fontsize 20
        for p in a.patches:
            a.annotate(format(p.get_height(), '.1f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')

        res_path = 'confusion matrix.png'

        # Saving files to Root folder
        plt.savefig(res_path)
        st.image(res_path, caption='Corretion based confusion Matrix')

        # text_representation.to_csv('clusteroutput.csv')

        # Print predicted values
        dxtest = pd.DataFrame(data=X_test)
        dytest = pd.DataFrame(data=y_test)
        dpred = pd.DataFrame(data=y_pred)
        dpred.columns = ['Target Predicted']
        dpred.head()
        drest = pd.concat([dxtest, dytest], axis=1, sort=False)
        drest = pd.concat([drest, dpred], axis=1, sort=False)
        drest.to_excel("Regresseddata.xlsx", sheet_name='Predictions')

        st.success(
            'Regressed data was stored in Regresseddata.xlsx file', icon="✅")

    except ValueError:
        st.error(
            'This is an error, Please make sure you have col named Target that needs prediction', icon="🚨")
    return None
    return None


