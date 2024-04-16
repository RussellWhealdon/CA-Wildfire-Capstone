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
from joblib import load

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

#Drop/rename columns (XGBoost doesn't accept special characters)
model_data['El Nino'] = model_data['El Nino'].replace({'El Nino': 1, 'La Nina': 0})
model_data = model_data.drop(columns=['Unnamed: 0', 'COUNTY', 'Total Fires', 'Large Fires', 'Total Acres Burned'])
model_data = model_data.rename(columns={'.25 acres or <':'.25 acres or less', '5000 acres or >':'5000 acres or more'})
raw_data["Year"] = raw_data["Year"].astype(str)

st.subheader("Overview of Data")
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


st.markdown(f"<h2 style='text-align: center;'>Making the Model</h2>", unsafe_allow_html=True)
st.subheader("Data Transformations")

### Create Model for Total Dollar Damage
TDD_log = np.log1p(model_data['Total Dollar Damage'])
TDD = model_data['Total Dollar Damage']

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

st.subheader("Sampling Techniques")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Walk forward Validation")
    st.write("Walk forward validation is a model validation technique used primarily in time series forecasting to ensure that the model is robust and performs well on unseen data. Unlike other cross-validation techniques that randomly shuffle model_data into training and testing sets, walk forward validation respects the chronological order of observations. This method involves incrementally 'walking' the cutoff point between the training and testing datasets forward in time, training the model on a fixed or expanding window of past data, and then testing it on the following model_data points. It mimics the real-world process of predicting future events based on past data, making it highly relevant for tasks such as financial forecasting, where the temporal sequence of model_data points is crucial. By using walk forward validation, one can more accurately gauge the predictive power of a time series model in practical scenarios, enhancing its reliability and effectiveness when deployed in dynamic environments.")
    st.image("https://i.stack.imgur.com/nxgwe.png")

with col2:
    code = '''tscv = TimeSeriesSplit(n_splits=4)

    #Move Year to the index column so I can use it in future analysis but not include it as a feature in the model (inappropriate for this type of modeling)
    model_data = model_data.sort_values(by='Year')
    model_data = model_data.set_index('Year')

    ## Model Training and sampling process
    # Sample model_data through TimeSeriesSplit (Implementation of Walk forward validation)
    # Train model w/ Grid Search for best params
    # Save the best model

    # Iterate through each split in the time series
    for train_index, test_index in tscv.split(model_data):
    #Rest of training process
    ...'''
    st.subheader("Implementation")
    st.code(code, language='python')

#Set lists/col names for future use
data_copy = model_data
data_copy = model_data.drop(columns = ['Total Dollar Damage', 'Year'])
valid_feature_names = data_copy.columns

all_transformed_predictions_TD = []
all_actual_values_TD = []
all_years = []

#Initialize Timeseries Split
tscv = TimeSeriesSplit(n_splits=4)

model_data = model_data.sort_values(by='Year')
model_data = model_data.set_index('Year')

#Getting last train test split for display purposes
count = 1
for train_index, test_index in tscv.split(model_data):
  count +=1
  if count == 4:
    training_index = train_index
    testing_index = test_index

st.subheader("Training the Model")
train_data, test_data = model_data.iloc[train_index], model_data.iloc[test_index]

# Extract features and target variable from training data
X_train = train_data.drop('Total Dollar Damage', axis=1)
y_train = train_data['Total Dollar Damage']

# Apply the log transformation to the target variable
y_train_log = np.log1p(train_data['Total Dollar Damage'])

# Extract features and target variable from test data
X_test = test_data.drop('Total Dollar Damage', axis=1)
y_test = test_data['Total Dollar Damage']

# Assign valid feature names to X_train and X_test DataFrames
X_train.columns = valid_feature_names
X_test.columns = valid_feature_names

# Initialize the XGBoost model
final_model_TDLog = load('final_modelTDLog.joblib')

# Make predictions on the test data
log_predictions = final_model_TDLog.predict(X_test)

# Transform the predictions back to their original scale
transformed_predictions = np.expm1(log_predictions)

# Get the 'Year' for each entry in the test set
years_test = model_data.iloc[test_index].index


###Evaluate error statistics and residual plots
# Store the transformed predictions and actual values for evaluation
all_transformed_predictions_TD.extend(transformed_predictions)
all_actual_values_TD.extend(y_test)
all_years.extend(years_test)

# Calculate the absolute differences
absolute_differences = [abs(all_actual_values_TD - all_transformed_predictions_TD) for all_actual_values_TD, all_transformed_predictions_TD in zip(all_actual_values_TD, all_transformed_predictions_TD)]

# Calculate the median absolute error
median_absolute_error = sorted(absolute_differences)[len(absolute_differences) // 2]

col3, col4, col5 = st.columns(3)

with col3: 
    # Show the result
    st.subheader("Error Stats")
    st.write("Median Absolute Error:", median_absolute_error)
    st.write("Decided to use median absolute error due to large values of error stats. More traditional metrics like MSE were very large which made it hard to see improvements in the model")

#Looking at Residuals and and actuals together
residuals_TD = [actual - predicted for actual, predicted in zip(all_actual_values_TD, all_transformed_predictions_TD)]
residual_TDdf = pd.DataFrame({'Year': all_years, 'Actual': all_actual_values_TD, 'Predicted': all_transformed_predictions_TD, 'Residual': residuals_TD})

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(all_transformed_predictions_TD, residuals_TD, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.axhline(y=0, color='red', linestyle='--')  # Line at 0 to indicate no error

with col4: 
    st.subheader("Residual Plot: Actual over Residuals")
    st.pyplot(plt)

residual_TDdf.groupby('Year')['Residual'].mean().plot(kind='line', title='Mean Residuals Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Residual')

with col5:
    st.subheader("Residual Plot: Residuals YoY")
    st.pyplot(plt)
