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
st.markdown("")

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


### Correlation Score for TDD
st.markdown("")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Correlation Analysis")
    st.markdown("This plot shows the correlation scores between our feature and target variables.")
    st.markdown("""
    - Majority of correlation scores are low, further emphasizing the complex nature of this dataset
    - Can see relevance of fire causes and some environmental factors
    - Number of fires of the size 5000 acres or greater seems to be more relevant than the total number of fires.
    - Although some variables did not show high values of correlation, this is only one type of relationship with others being exponential, temporal and geographic which we will evaluate in further sections.
    """)
with col4:
    minicol1, minicol2 = st.columns(2)
    with minicol1:
        st.image("images/CorrelationScore.png")

### Correlation percentile analyis
col5, col6 = st.columns(2)
with col5:
    st.subheader("Distribution of Total Dollar Damage")
    st.markdown("The boxplot dispalys the distribution of Total Dollar Damage at Log scale")
    st.markdown("""
    - The distribution of Total Dollar Damage is quite large and ranges from 0 all the way to roughly 4 billion dollars
    - The greater majority of data points are 0-1 million dollars
    - Traditionally, we would remove outliers before modeling but in this case the outliers are key points of information 
    - Understanding them is what will drive the value of this analysis.
    """)

with col6: 
    st.image("images/BoxPlotTDD.png")

### Correlation percentile analyis
col5, col6 = st.columns(2)
with col5:
    st.subheader("Percentile Correlation Analysis")
    st.markdown("The plot shows a preview of correlation scores between our target and feature variables across different percetiles of Total Dollar Damage")
    st.markdown("""
    - The 99% percentile would be fires with the most damage where as the 10% would be the smaller end
    - Figured relationships may differ at lower or higher ends of the distribution
    - The majority of the values are on the lower end of the percentiles with large jumps existing between the 90th and 95th percentiles. 
    - We can see large jumps in certain variables like “Elec. Power” and “Debris Burning” meaning that they may be more prevalent in the high-damage events
    - Increase in the building and agricultural values as we move into the higher percentiles. This may indicate that regions that stand more to lose are inherently more vulnerable to large damages
    """)

with col6: 
    st.image("images/CorrelationPercentiles.png")

### Evaluation of Fire Causes
col7, col8 = st.columns(2)
with col7:
    st.subheader("Fire Cause Analysis")
    st.markdown("The plot shows the totals for categories of fire causes over the timeframe of our dataset")
    st.markdown("""
    - From this we can see that Man-made fires has seen a steady uptick in recent years
    - Natural causes have suprisingly declined 
    - This shows the need for governemnt intervention as a means of lower fire totals
    """)
with col8:
    st.image("images/FireCauses_OT.png")


### Modeling 
st.markdown(f"<h2 style='text-align: center;'>Modeling</h2>", unsafe_allow_html=True)
st.markdown("""
In working with our sponsors they mentioned they wanted to incorporate some sort of Mahine Learning methodologies with the hopes of understanding the impact of different 
features in this dataset on Total Dollar Damage. Prior effforts had been made to use methodolgies like Linear Regression but those lacked the complexity 
to deal with a dataset such as this. Some things we knew we need to account for were the aspect of time and the distribution of the target variable making our sampling methods
and transformations important as we began to build things out. It's important to note that our intentions here are to model for the purpose of explainability not necessarily 
predictability. Wildfires are so sporadic that trying to predict their damage accurately would be unresonable. However, we do feel value can be gained from builing a model
tailered to showing how features are impacting the degree of damage. 
""")

### Transformation
col7, col8 = st.columns(2)
with col7:
    st.subheader("Target Variable Log Transformation")
    st.markdown("""
    Although the XGBoost Regressor model does not assume that the data be normally distributed, the variance of our target variable was causing unstable predictions that did 
    not capture the true nature of the dataset. In order to account for this, we used log transformation to lessen the range of our target variables. A visual representation 
    of this can be seen in the appendix. The log transformation takes the log form of each target variable before the model gets trained. Once we have our predictions for 
    the model, we then transform the predictions back to their original scale to evaluate the model's performance. We saw more accurate and stable predictions for our models 
    which we believe to be because of the change in distribution.""")

with col8:
    st.markdown("")
    st.image("images/TransDist..png")


### Sampling Methods
col9, col10 = st.columns(2)
with col9:
    st.subheader("Sampling Methodology - Walk Forward Validation")
    st.markdown("""
    In order to ensure that our model was not shown data that it otherwise wouldn’t have in a real world setting we needed to control for the element of time. To do this 
    we used a method of sampling called walk-forward validation. This is where you split the data based on the time column, putting the newer values in the testing set and 
    the older values in the training set. The same basic principles apply by putting the majority of data in the training set and vice versa for testing. The other issue we 
    had with our data was that we had a relatively small sample size of about 750 rows. This was due to the fact that we were only able to collect data at the County level 
    rather than at each instance of wildfire. This reflects the grander issue that wildfires pose. They are hard to track from a damage perspective and the cause and true 
    size are often hard to evaluate. Nonetheless, to address this issue we used cross validation to give our model a better chance to pick up on trends by seeing different 
    groupings of rows through multiple iterations.""")

with col10:
    st.image("images/WalkForwardValidationDiagram.png")

### Model Training
st.subheader("Model Training - XG Boost Regressor")
st.markdown("""
An XGBoost regressor is a powerful machine learning model for regression tasks, using gradient boosting techniques. It builds on an ensemble of decision trees 
sequentially, each correcting its predecessor, and incorporates regularization to prevent overfitting. XGBoost is known for its high performance and efficiency with 
large datasets. We thought it would be effective here becuase of it's ability to handle complex data but still offers more interpretability than models like Nueral 
Networks or SVMs.
""")

### Model Output
col11, col12 = st.columns(2)
with col11:
    st.subheader("Model Output - Shapley Values (Global)")
    st.markdown("""  
    A Shapley value summary plot is a visualization tool used in machine learning to interpret the output of models, particularly those involving complex algorithms like 
    tree ensembles or neural networks. It is based on Shapley values, a concept from cooperative game theory that allocates payouts (in this case, prediction impact) fairly 
    among contributors (features).
    """)
    st.markdown("""
    Interpretation:
    - Feature Importance: Features at the top of the plot are generally more important to the model’s output than those at the bottom.
    - Positive or Negative Impact: If most dots for a feature are to the right of the center, this feature generally pushes predictions higher. Conversely, dots to the left suggest a tendency to lower predictions.
    - Value Effect: The color of the dots can help you understand if higher or lower values of the feature increase or decrease the output. For example, if high values of a feature (indicated by one color) are mostly on the right, high values increase the prediction value.
    - Consistency: A tightly clustered group of dots indicates that a feature has a consistent effect on the prediction across different data points, while a widely spread group indicates variability in its impact.
    """)

with col12:
    st.image("images/ShapleyValueOutput.png")

col13, col14 = st.columns(2)
with col13:
    st.subheader("Model Output - Shapley Values (Global)")
    st.markdown("""  
    A Shapley value summary plot is a visualization tool used in machine learning to interpret the output of models, particularly those involving complex algorithms like 
    tree ensembles or neural networks. It is based on Shapley values, a concept from cooperative game theory that allocates payouts (in this case, prediction impact) fairly 
    among contributors (features).
    """)
    st.markdown("""
    Interpretation:
    - Feature Importance: Features at the top of the plot are generally more important to the model’s output than those at the bottom.
    - Positive or Negative Impact: If most dots for a feature are to the right of the center, this feature generally pushes predictions higher. Conversely, dots to the left suggest a tendency to lower predictions.
    - Value Effect: The color of the dots can help you understand if higher or lower values of the feature increase or decrease the output. For example, if high values of a feature (indicated by one color) are mostly on the right, high values increase the prediction value.
    - Consistency: A tightly clustered group of dots indicates that a feature has a consistent effect on the prediction across different data points, while a widely spread group indicates variability in its impact.
    """)
with col14:
    st.image("images/LIMEExpl..png")


