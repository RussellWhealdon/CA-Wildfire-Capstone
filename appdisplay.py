import streamlit as st
import pandas as pd

### Set Page config
st.set_page_config(page_title= f"CA Wildfire Dash",page_icon="🧑‍🚀",layout="wide")

### Set Background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://source.unsplash.com/a-close-up-of-a-white-wall-with-wavy-lines-75xPHEQBmvA");
background-size: cover;
}
</style>
"""
### Big title
st.markdown(f"<h1 style='text-align: center;'>California Wildfire Damage Analysis</h1>", unsafe_allow_html=True)
    
### Introduction section
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center;'>Introduction</h2>", unsafe_allow_html=True)
st.write("This dashboard presents an analysis of the economic impacts of wildfires, developed in collaboration with Deloitte's sustainability arm. The project aims to understand and predict the financial damages caused by wildfires, leveraging data on various environmental and economic factors. The predictive modeling was done using an XGBoost regression model, enhanced with SHAP and LIME for model interpretability.")

#### Load in data
raw_data = pd.read_csv('Data/ClimateProjData.csv')
model_data = pd.read_excel('Data/climateprojdata_final.xlsx')
data_dictionary = pd.read_csv("Data/ClimateProjData - Dictionary.csv")

### Data Overview
st.markdown(f"<h2 style='text-align: center;'>Overview of Data</h2>", unsafe_allow_html=True)
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


### Exploratory Analysis
st.markdown(f"<h2 style='text-align: center;'>Exploratory Analysis</h2>", unsafe_allow_html=True)


### Dollar Damage, Acres Burned, and Total Fires summed across Counties
col1, col2 = st.columns(2)
with col1:
    st.subheader("Key Variables Summed Across Counties")
    st.markdown("This plot shows the total Dollar Damage, Acres Burned, and Number of Fires for each county for the entire time period of our dataset.")
    st.markdown("""
    - Reflects the sporadic nature of wildfire impact across counties
    - Some counties have large values for acres burned but little to no economic impact 
    - Same goes for the total number of fires
    - Economic impact largely depends on the location of the fire and nature of the burn
    - Hoping to draw insight from more descrete variables such as weather patterns and fire cause to find trends
    - To highlight a specific example, Nevada county tallied the second highest total damage of all counties in California yet shows a fairly small value for total acres burned within the time frame of our data. In this specific instance two deadly 2017 wildfires did almost 2 billion dollars worth of damage to Nevada county. The fires were relatively small, burning roughly 300 acres of land yet destroyed key structures and commercial buildings. Understanding situations like these are key to understanding the nature of wildfire impact.
    """)
with col2:
    st.image("images/TargetVarDist..png")


### Correlation Score for TDD and Correlation percentile analyis

col3, col4 = st.columns(2)
with col3:
    st.subheader("Key Variables Summed Across Counties")
    st.markdown("This plot shows the total Dollar Damage, Acres Burned, and Number of Fires for each county for the entire time period of our dataset.")
with col2:
    st.image("images/CorrelationScore.png")
    st.image("images/CorrelationPercentiles.png")




