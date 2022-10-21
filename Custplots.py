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
import  Semiclean




# --------------

def cstplt(df, plttype):
    df3 = Semiclean.Clean(df)
    if plttype == 'Linear Regression' :
        # Select Target Variable & Dimensions
        choices = df3.columns.values.tolist()
        optiony = st.selectbox(
            'Select the y-variable for scatter plot to run Linear Regression model line', choices)
        optionx = st.selectbox(
            'Select the x-variables for scatter plot to run Linear Regression model line', choices)
        if st.button('Start plotting'): 
            st.write('In the simplest invocation, both functions draw a scatterplot of two variables, x and y, and then fit the regression model y ~ x and plot the resulting regression line and a 95% confidence interval for that regression')
            fig = plt.figure(figsize=(20, 10))
            sns.regplot(data=df3, x= optionx, y= optiony)
            #plt.title('Scatter Plot y vs x', fontsize=25)  # title with fontsize 20
            st.pyplot(fig)
            
    
    if plttype == 'Distribution' :
        # Select Target Variable & Dimensions
        choices = df3.columns.values.tolist()
        optionx1 = st.selectbox(
            'Select the variable for plotting distribution', choices)
        if st.button('Start plotting'): 
            st.write('A histogram is a bar plot where the axis representing the data variable is divided into a set of discrete bins and the count of observations falling within each bin is shown using the height of the corresponding bar')
            fig1 = plt.figure(figsize=(20, 10))
            sns.displot(data=df3, x= optionx1)
            plt.savefig('Dist.png')
            st.image('Dist.png',caption = 'Distribution') 
           

    if plttype == 'Scatter' :
         # Select Target Variable & Dimensions
         choices = df3.columns.values.tolist()
         optiony2 = st.selectbox(
             'Select the y-variable for scatter plot', choices)
         optionx2 = st.selectbox(
             'Select the x-variables for scatter plot', choices)
         h = st.selectbox(
             'Select the hue/color variation for scatter plot', choices)
         if st.button('Start plotting'): 
             st.write('The scatter plot is a mainstay of statistical visualization. It depicts the joint distribution of two variables using a cloud of points, where each point represents an observation in the dataset. This depiction allows the eye to infer a substantial amount of information about whether there is any meaningful relationship between them')
             fig2 = plt.figure(figsize=(20, 10))
             sns.relplot(data=df3, x= optionx2, y= optiony2, hue = h)
             plt.savefig('scatter.png')
             st.image('scatter.png',caption = 'Scatter Plot y vs x') 
         
    if plttype == 'Box Plot for Categorical Data' :
         # Select Target Variable & Dimensions
         choices = df3.columns.values.tolist()
         optiony3 = st.selectbox(
             'Select the y-variable for box plot', choices)
         st.write('Please select a categorical variable for x')
         optionx3 = st.selectbox(
             'Select the x-variables for box plot', choices)
         h1 = st.selectbox(
             'Select the hue/color variation for scatter plot', choices)
         if st.button('Start plotting'):
             st.write('This kind of plot shows the three quartile values of the distribution along with extreme values. The “whiskers” extend to points that lie within 1.5 IQRs of the lower and upper quartile, and then observations that fall outside this range are displayed independently.')
             fig3 = plt.figure(figsize=(20, 10))
             sns.catplot(data=df3, x= optionx3, y= optiony3, hue = h1, kind = 'box')
             plt.savefig('box.png')
             st.image('box.png',caption = 'Box Plot y vs x') 
             st.write('How to read the box plot ?')
             from PIL import Image
             image = Image.open('./samplebox.png')
             st.image(image)
             
