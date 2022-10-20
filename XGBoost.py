#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM TKINTER PYTHON
# AUTHOR: LAXMI MULLAPUDI (lmullapu@cisco.com)
# LAST UPDATED: 10/29/2020
# CREATED:10/1/2020
########################################################################################


import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel



# ---------------------------------------------------------------------------
# Browsing & loading data from # Youtube Link: https://www.youtube.com/watch?v=PgLjwl6Br0k
# Changes made by lmullapu to the orientation of the frames to make them relative
# ----------------------------------------------------------------------------



def xgboost(predictors, targets, testsz, lr, maxdepth):
    st.write('Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker decision tree models')
    # Drop the columns with more than 1000 unique values for prediction
    for col in predictors.columns:
        if len(predictors[col].unique()) > 1000:
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
        from sklearn.model_selection import train_test_split
        trainsz = 1.0 - testsz
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsz, train_size=trainsz, stratify=y)

       
        # fit model on all training data
        from xgboost import XGBClassifier
        xg_reg = XGBClassifier(learning_rate =lr,
        n_estimators=1000,
        max_depth=maxdepth,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
        eval_set = [(X_test, y_test)]
        xg_reg.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="aucpr", eval_set=eval_set, verbose=True)

        # make predictions for test data and evaluate
        preds = xg_reg.predict(X_test)
        
        # Test & train accuracy
        trsc = xg_reg.score(X_train, y_train)*100
        tssc = xg_reg.score(X_test, y_test)*100
        
        # Fit model using each importance as a threshold
        thresholds = xg_reg.feature_importances_
        for thresh in thresholds:
        	# select features using threshold
        	selection = SelectFromModel(xg_reg, threshold=thresh, prefit=True)
        	select_X_train = selection.transform(X_train)
        	# train model
        	selection_model = XGBClassifier()
        	selection_model.fit(select_X_train, y_train)
        	# eval model
        	select_X_test = selection.transform(X_test)
        	predictions = selection_model.predict(select_X_test)
        	accuracy = accuracy_score(y_test, predictions)
        	st.write('Thresh=%.3f, n=%d, Accuracy: %.2f%%' % (thresh, select_X_train.shape[1], accuracy*100.0))

        from sklearn.metrics import confusion_matrix
        cmxg = confusion_matrix(y_test, preds)

        params = {"objective": "reg:linear", 'colsample_bytree': 0.3,
                  'learning_rate': 0.1, 'max_depth': 8, 'alpha': 10}
        
        
        f = plt.figure(figsize=(10, 7))
        a = f.add_subplot(111)
        accdata = {'Data': ['Train Data', 'Test Data'],
                   'Accuracy': [trsc, tssc]}
        dct = pd.DataFrame(accdata, columns=['Data', 'Accuracy'])
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        a = sns.barplot(x="Data", y="Accuracy", palette="Blues",  data=dct)
        a.set(xlabel='Data Set', ylabel='% Accuracy')
        plt.ylim(0, 100)
        sns.set_context("paper", font_scale=1.5)
        plt.title('Gradient Boosting Prediction',
                  fontsize=14)  # title with fontsize 20
        for p in a.patches:
            a.annotate(format(p.get_height(), '.1f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')

        res_path_1 = 'confusion matrix.png'

       # Saving files to Root folder
        plt.savefig(res_path_1)
        st.image(res_path_1, caption='Correlation based confusion Matrix')
        plt.savefig("xgboost.png")

        # Print predicted values
        dxtest = pd.DataFrame(data=X_test)
        dytest = pd.DataFrame(data=y_test)
        dpred = pd.DataFrame(data=preds)

        dpred.columns = ['Target Predicted']
        dpred.head()

        # Concatenating test X & Y
        drest = pd.concat([dxtest, dytest], axis=1, sort=False)
        st.download_button('BoostingPredictions', drest)
    

        # Print Parameters
        data_dmatrix = xgb.DMatrix(data=X, label=y)
        xg_reg = xgb.train(
            params=params, dtrain=data_dmatrix, num_boost_round=10)
        ax = xgb.plot_importance(xg_reg)
        fig4 = ax.figure
        fig4.set_size_inches(10, 10)
        res_path_2 = 'Parameters.png'

       # Saving files to Root folder
        plt.savefig(res_path_2)
        st.image(res_path_2, caption='Parameters Feature Importance')
        st.success(
            'Success", "Your predictions & parameters are saved in .csv file named Boostingpredictions & params.png', icon="✅")
    except ValueError:
        st.error(
            'This is an error, Please make sure you have col named Target that needs prediction', icon="🚨")
    return None
    return None





