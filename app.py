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


def main():
    # Set page config
    st.set_page_config(page_title='Wildfire Damage Analysis', layout='wide')

    # Custom CSS to inject larger fonts and a purple background.
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #a333c8;
            color: #ffffff;
        }
        .markdown-text-container {
            font-family: 'Helvetica';
            font-size: 20px;
            color: #ffffff;
        }
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

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
