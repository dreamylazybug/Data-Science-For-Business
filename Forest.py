#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM TKINTER PYTHON
# AUTHOR: LAXMI MULLAPUDI (lmullapu@cisco.com)
# LAST UPDATED: 10/29/2020
# CREATED:10/1/2020
########################################################################################


import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Browsing & loading data from # Youtube Link: https://www.youtube.com/watch?v=PgLjwl6Br0k
# Changes made by lmullapu to the orientation of the frames to make them relative
# ----------------------------------------------------------------------------



def forest(predictors, targets, testsz, lr, maxdepth):
   
    
   st.write('Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned')
   # Drop the columns with more than 1000 unique values for prediction
   for col in predictors.columns:
       if len(predictors[col].unique()) > 100000:
           predictors.drop(col, inplace=True, axis=1)
           st.warning(
               'A selected predictor has more than 1000 unique values * hence removed from analysis', icon="⚠️")

   try:

       # Sending alert if # of predicted samples is less than 500 or 5%
       count = len(targets.index)
       if count < 500:
           st.error(
               'Number of data points for prediction is less than 500', icon="🚨")

       # XGBoost Model
       X = predictors
       y = targets
       
       # split data into train and test sets
       trainsz = 1.0 - testsz
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsz, train_size=trainsz, stratify=y)
    
        # fit model on all training data
       
       classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
       classifier_rf.fit(X_train, y_train)
        
       # Checking the oob score -  computed as the number of correctly predicted rows from the out of bag sample.
       testaccuracy = classifier_rf.oob_score_
       st.write('The Random Forest Model before parameters tuning had an accuracy of', testaccuracy) 
       
       # Plotting a tree - sample
       # setting font sizeto 30
       plt.figure(figsize=(14,14))
       plot_tree(classifier_rf.estimators_[5], feature_names = X.columns,filled=True, fontsize=10)
       # setting font sizeto 30
       res_path_2 = 'SampleTree.png'

       # Saving files to Root folder
       plt.savefig(res_path_2)
       st.image(res_path_2, caption='Random Forest - Sample Tree')
       plt.savefig("Sampletree.png")
      
      
       # Feature Importances
       imp_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Imp": classifier_rf.feature_importances_
        })
       # Sorting by importance
  
       dtvar = imp_df.sort_values(by="Imp", ascending=False)
       fig = plt.figure(figsize=(12, 5))
       sns.barplot(data=dtvar, x="Feature", y="Imp")
       plt.xlabel("Feature")
       plt.ylabel("Importance")
       plt.title("Feature Importance Chart")
       st.pyplot(fig)
       
       
   
       # Saving files to Root folder
       res_path_3 = 'Features.png'
       plt.savefig(res_path_3)
       plt.savefig("Featureimportance.png")
      
        
       # Storing Prediction
       y_predicted = classifier_rf.predict(X_test)
       dxtest = pd.DataFrame(data=X_test)
       dytest = pd.DataFrame(data=y_test)
       dpred = pd.DataFrame(data=y_predicted)
       dpred.columns = ['Target Predicted']
      
       # Concatenating test X & Y
       drest = pd.concat([dxtest, dytest], axis=1, sort=False)
       
       st.success(
           'Success!', icon="✅")
    
   except ValueError:
       st.error(
           'This is an error, Please make sure you have col named Target that needs prediction', icon="🚨")
   return drest
   return None
       
      






