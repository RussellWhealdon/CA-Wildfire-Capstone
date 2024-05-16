import streamlit as st
import pandas as pd

st.set_page_config(page_title= f"CA Wildfire Dash",page_icon="🧑‍🚀",layout="wide")

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
st.subheader("Introduction")
st.write("This dashboard presents an analysis of the economic impacts of wildfires, developed in collaboration with Deloitte's sustainability arm. The project aims to understand and predict the financial damages caused by wildfires, leveraging data on various environmental and economic factors. The predictive modeling was done using an XGBoost regression model, enhanced with SHAP and LIME for model interpretability.")

# Rest of the app here
# Display data, interactive widgets, visualizations, etc.

#Load in data
raw_data = pd.read_csv('Data/ClimateProjData.csv')
model_data = pd.read_excel('Data/climateprojdata_final.xlsx')
data_dictionary = pd.read_csv("Data/ClimateProjData - Dictionary.csv")



