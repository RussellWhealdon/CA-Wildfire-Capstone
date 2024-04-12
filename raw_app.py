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

#Load in data
data = pd.read_excel('climateprojdata_v2 (1).xlsx')

#Drop/rename columns (XGBoost doesn't accept special characters)
data['El Nino'] = data['El Nino'].replace({'El Nino': 1, 'La Nina': 0})
data = data.drop(columns=['Unnamed: 0', 'COUNTY', 'Total Fires', 'Large Fires', 'Total Acres Burned'])
data = data.rename(columns={'.25 acres or <':'.25 acres or less', '5000 acres or >':'5000 acres or more'})

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
plt.show()


# Initialize the XGBoost model
model_TDLog = XGBRegressor()

#Get col names for later steps
data_copy = data.drop(columns = ['Total Dollar Damage', 'Year'])
valid_feature_names = data_copy.columns

#Make list for predictions and actuals
all_transformed_predictions_TD = []
all_actual_values_TD = []
all_years = []

#Create Param grid
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [3, 4, 5],       
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
}

##Use walk forward validation for sampling method
# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=4)

#Move Year to the index column so I can use it in future analysis but not include it as a feature in the model (inappropriate for this type of modeling)
data = data.sort_values(by='Year')
data = data.set_index('Year')

## Model Training and sampling process
# Sample data through TimeSeriesSplit (Implementation of Walk forward validation)
# Train model w/ Grid Search for best params
# Save the best model

# Iterate through each split in the time series
for train_index, test_index in tscv.split(data):
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]

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
    xgb_model_TDLog = xgb.XGBRegressor()

    # Initialize GridSearchCV with the model and parameter grid
    grid_search_TDLog = GridSearchCV(estimator=xgb_model_TDLog, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error')

    # Fit the grid search to the training data
    grid_search_TDLog.fit(X_train, y_train)

    # Get the best hyperparameters and best estimator
    best_params = grid_search_TDLog.best_params_
    best_estimator = grid_search_TDLog.best_estimator_

    # Train the final model with the best hyperparameters on the entire training dataset
    final_model_TDLog = best_estimator
    final_model_TDLog.fit(X_train, y_train_log)

    # Make predictions on the test data
    log_predictions = final_model_TDLog.predict(X_test)

    # Transform the predictions back to their original scale
    transformed_predictions = np.expm1(log_predictions)

    # Get the 'Year' for each entry in the test set
    years_test = data.iloc[test_index].index

    # Store the transformed predictions and actual values for evaluation
    all_transformed_predictions_TD.extend(transformed_predictions)
    all_actual_values_TD.extend(y_test)
    all_years.extend(years_test)


## Calculate Error Statistics
mae = mean_absolute_error(all_actual_values_TD, all_transformed_predictions_TD)
print(f'Mean Absolute Error: {round(mae,2)}$')

# Calculate the absolute differences
absolute_differences = [abs(all_actual_values_TD - all_transformed_predictions_TD) for all_actual_values_TD, all_transformed_predictions_TD in zip(all_actual_values_TD, all_transformed_predictions_TD)]

# Calculate the median absolute error
median_absolute_error = sorted(absolute_differences)[len(absolute_differences) // 2]

# Print the result
print("Median Absolute Error:", median_absolute_error)

## Gather residuals and actuals to look at resudual plots
residuals_TD = [actual - predicted for actual, predicted in zip(all_actual_values_TD, all_transformed_predictions_TD)]
residual_TDdf = pd.DataFrame({'Year': all_years, 'Actual': all_actual_values_TD, 'Predicted': all_transformed_predictions_TD, 'Residual': residuals_TD})

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(all_transformed_predictions_TD, residuals_TD, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.axhline(y=0, color='red', linestyle='--')  # Line at 0 to indicate no error
plt.show()

# Looking at residuals by Year
residual_TDdf.groupby('Year')['Residual'].mean().plot(kind='line', title='Mean Residuals Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Residual')

## Feature Importance Plots 
# F-Score: How often is each variable used in the tree
xgb.plot_importance(final_model_TDLog, max_num_features=30)
plt.show()

# Create a SHAP explainer
explainer = shap.Explainer(final_model_TDLog)
# Compute SHAP values - this might take some time depending on the size of your data
shap_values = explainer(X_train)

shap.summary_plot(shap_values, X_train)

#Finding indicies for highest predictions (in hopes of finding out why they were so high
all_transformed_predictions_TD_array = np.array(all_transformed_predictions_TD)
top_5_indices = np.argsort(all_transformed_predictions_TD_array)[::-1][:5]

# For the first prediction in the test set
shap.waterfall_plot(shap_values[140])

## Looking at Lime Explainer values
# Initialize a LIME explainer
# Note: The training data (X_train) should be a numpy array for the LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=['Target'],
    mode='regression'
)

# Select an instance to explain
# For example, explaining the prediction for the first instance in the test set
instance = X_test.iloc[77]

# Generate the LIME explanation
exp = lime_explainer.explain_instance(instance.values, final_model_TDLog.predict)

# Show the explanation
exp.show_in_notebook(show_table=True)

