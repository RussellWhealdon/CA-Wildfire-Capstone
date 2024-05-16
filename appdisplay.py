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
st.markdown(f"<h3 style='text-align: center;'>Introduction</h3>", unsafe_allow_html=True)
st.write("This dashboard presents an analysis of the economic impacts of wildfires, developed in collaboration with Deloitte's sustainability arm. The project aims to understand and predict the financial damages caused by wildfires, leveraging data on various environmental and economic factors. The predictive modeling was done using an XGBoost regression model, enhanced with SHAP and LIME for model interpretability.")

# Rest of the app here
# Display data, interactive widgets, visualizations, etc.

#Load in data
raw_data = pd.read_csv('Data/ClimateProjData.csv')
model_data = pd.read_excel('Data/climateprojdata_final.xlsx')
data_dictionary = pd.read_csv("Data/ClimateProjData - Dictionary.csv")


st.markdown(f"<h3 style='text-align: center;'>Overview of Data</h3>", unsafe_allow_html=True)
st.write("The data provided shows the impact of wildfires in counties across California aggregated by year, as well as charactersitcs related to each county including size, climate, and risk metrics.")
st.write("Sources include:")
st.markdown("- CA Gov - State of California, for Detailed wildfire data including causes, size of fires, and damages")
st.markdown("- NRI - FEMA’s National Risk Index, for general overview of the areas at risk of wildfires relative to the entire nation")
st.markdown("- NCEI & NOAA, for environmental data including rainfall, temperature, and weather patterns.")
st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)
with st.expander("See Data Preview"):
    st.write(raw_data)
with st.expander("See Data Dictionary"):
    st.write(data_dictionary)

st.markdown(f"<h3 style='text-align: center;'>Exploratory Analysis</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("Information about this image I'm about to show")

with col2:
    st.image("images/TargetVarDist..png")





