import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
import shap
import lime
import lime.lime_tabular

def set_full_page_gradient(color1, color2):
    # Set the full page background to a gradient
    st.markdown(f"""
        <style>
        body {{
            color: #fff;  # Sets the text color to white, change as necessary
            background-image: linear-gradient(to right, {color1}, {color2});
        }}
        .reportview-container .main .block-container{{
            padding-top: 5rem;
            padding-bottom: 5rem;
            color: #fff;  # Sets the text color for the content in the container to white, change as necessary
            background-color: transparent;
        }}
        </style>
        """, unsafe_allow_html=True)

def main():
    # Set page config
    st.set_page_config(page_title='Wildfire Damage Analysis', layout='wide')
    set_full_page_gradient('#6a0dad', '#9932cc')
    
    # Introduction section
    st.markdown('<div class="big-font">Wildfire Damage Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
        ## Introduction
        This dashboard presents an analysis of the economic impacts of wildfires, developed in collaboration with Deloitte's sustainability arm.
        The project aims to understand and predict the financial damages caused by wildfires, leveraging data on various environmental and economic factors.
        The predictive modeling was done using an XGBoost regression model, enhanced with SHAP and LIME for interpretability.
    """)

    # Rest of the app here
    # Display data, interactive widgets, visualizations, etc.

    #Load in data
    data = pd.read_excel('climateprojdata_final.xlsx')

    #Drop/rename columns (XGBoost doesn't accept special characters)
    data['El Nino'] = data['El Nino'].replace({'El Nino': 1, 'La Nina': 0})
    data = data.drop(columns=['Unnamed: 0', 'COUNTY', 'Total Fires', 'Large Fires', 'Total Acres Burned'])
    data = data.rename(columns={'.25 acres or <':'.25 acres or less', '5000 acres or >':'5000 acres or more'})

    st.write(data)


if __name__ == "__main__":
    main()
