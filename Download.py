#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM STREAMLIT
# AUTHOR: LAXMI MULLAPUDI 
# LAST UPDATED " 2022/10/20
########################################################################################



import numpy as np
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


# Data Preparation
# Replacing blanks from data & col names, separating numeric & categorical variables, Fill NA with 0 in numeric data, label encoding categorical variables, merging numeric & categorical dataframe finally

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

  




