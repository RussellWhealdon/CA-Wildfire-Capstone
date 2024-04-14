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

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://source.unsplash.com/a-close-up-of-a-white-wall-with-wavy-lines-75xPHEQBmvA");
background-size: cover;
}
</style>
"""

st.markdown(f"<h1 style='text-align: center;'>California Wildfire Damage Analysis</h1>", unsafe_allow_html=True)
    
# Introduction section
st.markdown(page_bg_img, unsafe_allow_html=True)
st.write("Introduction - This dashboard presents an analysis of the economic impacts of wildfires, developed in collaboration with Deloitte's sustainability arm. The project aims to understand and predict the financial damages caused by wildfires, leveraging data on various environmental and economic factors. The predictive modeling was done using an XGBoost regression model, enhanced with SHAP and LIME for interpretability.")

# Rest of the app here
# Display data, interactive widgets, visualizations, etc.

#Load in data
data = pd.read_excel('climateprojdata_final.xlsx')

#Drop/rename columns (XGBoost doesn't accept special characters)
data['El Nino'] = data['El Nino'].replace({'El Nino': 1, 'La Nina': 0})
data = data.drop(columns=['Unnamed: 0', 'COUNTY', 'Total Fires', 'Large Fires', 'Total Acres Burned'])
data = data.rename(columns={'.25 acres or <':'.25 acres or less', '5000 acres or >':'5000 acres or more'})

st.markdown(f"<h2 style='text-align: center;'>Overview of Data</h2>", unsafe_allow_html=True)
st.write("The data provided comes from the CA gov website that reports various reported characteristics of fires from around the state of California as well risk index metrics from the SOVI, NRI, and BRIC datasets.")
st.write(data)

st.markdown(f"<h2 style='text-align: center;'>Data Transformations</h2>", unsafe_allow_html=True)

### Create Model for Total Dollar Damage
TDD_log = np.log1p(data['Total Dollar Damage'])
TDD = data['Total Dollar Damage']

# Create a figure with two subplots
plt.figure(figsize=(20, 6))

# Subplot for Normal Data
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.hist(TDD, bins=100, color='red', alpha=0.7)
plt.title('Distribution of Total Dollar Damage')
plt.xlabel('Total Dollar Damage')
plt.ylabel('Frequency')
plt.grid(True)

# Subplot Transformed Data
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.hist(TDD_log, bins=100, color='green', alpha=0.7)
plt.title('Distribution of Log of Total Dollar Damage')
plt.xlabel('Log of Total Dollar Damage')
plt.ylabel('Frequency')
plt.grid(True)

# Display subplots
st.pyplot(plt)

st.markdown(f"<h2 style='text-align: center;'>Sampling Techniques</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.image("https://i.stack.imgur.com/nxgwe.png")
