#######################################################################################
# DATA SCIENCE MODELS POWERED BY VISUALIZATION FROM STREAMLIT
# AUTHOR: LAXMI MULLAPUDI 
# LAST UPDATED " 2022/10/20
########################################################################################



import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode



def Load_Data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # https://towardsdatascience.com/make-dataframes-interactive-in-streamlit-c3d0c4f84ccb
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
        gb.configure_side_bar()  # Add a sidebar
        # Enable multi-row selection
        gb.configure_selection('multiple', use_checkbox=True,
                               groupSelectsChildren="Group checkbox select children")
        gridOptions = gb.build()
    
        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT',
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=True,
            height=350,
            width='100%',
            reload_data=True
        )
        return(df) 
    else :
        st.error('Error-Try loading another file or change to .csv mode', icon="🚨")



  