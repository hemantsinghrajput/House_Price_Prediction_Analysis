import pandas as pd
import zipfile
from sklearn.impute import SimpleImputer
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Step 1: Extract the data from the provided zip file
zip_path = 'archive (1).zip'
extract_path = 'extracted_data/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 2: Load each dataset
cpi_df = pd.read_csv(extract_path + 'Consumer-Price-Index.csv')
fed_funds_df = pd.read_csv(extract_path + 'FedFunds.csv')
gdp_df = pd.read_csv(extract_path + 'GDP.csv')
home_price_df = pd.read_csv(extract_path + 'Home-Price-Index.csv')
mortgage_df = pd.read_csv(extract_path + 'Mortgage.csv')
population_growth_df = pd.read_csv(extract_path + 'Population-Growth.csv') 
unemployment_rate_df = pd.read_csv(extract_path + 'Unemployment-Rate.csv')

# Step 3: Merge the datasets based on the common 'DATE' column
merged_df = cpi_df.merge(fed_funds_df, on='DATE', how='left')
merged_df = merged_df.merge(gdp_df, on='DATE', how='left')
merged_df = merged_df.merge(home_price_df, on='DATE', how='left')
merged_df = merged_df.merge(mortgage_df, on='DATE', how='left')
merged_df = merged_df.merge(population_growth_df, on='DATE', how='left')
merged_df = merged_df.merge(unemployment_rate_df, on='DATE', how='left')

# Step 4: Filter the data based on the date (from 1987-01-01)
cut_off_date = '1987-01-01'
merged_df = merged_df[merged_df['DATE'] >= cut_off_date]

# Step 5: Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' as well
merged_df.iloc[:, 1:] = imputer.fit_transform(merged_df.iloc[:, 1:])

# Display the updated dataframe
print(merged_df.head())
# Step 5: Handle missing values
features = ['FEDFUNDS', 'GDP', 'MORTGAGE30US', 'SPPOPGROWUSA', 'UNRATE']
target = 'CSUSHPISA'

# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
merged_df[features] = imputer.fit_transform(merged_df[features])

# Step 6: Build a predictive model (Linear Regression for simplicity)
train_data, test_data = train_test_split(merged_df, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(train_data[features], train_data[target])
# Step 7: Make predictions on the test set
test_predictions = model.predict(test_data[features])

# Step 8: Evaluate the model performance
mse = mean_squared_error(test_data[target], test_predictions)
print(f'Mean Squared Error (MSE): {mse}')

# Step 9: Continue with Streamlit app and visualization
# You can use the trained model to make predictions and visualize the results using Streamlit.

# Example Streamlit code:
# (Note: Make sure to install Streamlit using pip install streamlit if you haven't already)
# Save this code in a file named app.py and run it with the command streamlit run app.py




import streamlit as st
import numpy as np

# Function to make predictions using the trained model
# Function to make predictions using the trained model
# Function to make predictions using the trained model
def make_predictions(features_input):
    # Convert the input features to a NumPy array
    features_input_array = np.array(features_input)
    
    # Reshape the input features for prediction
    features_input_reshaped = features_input_array.reshape(1, -1)
    
    # Make predictions using the model
    prediction = model.predict(features_input_reshaped)
    return prediction[0]

# Streamlit UI
st.title('Home Price Prediction App')

# User input for features
fed_funds = st.slider('Federal Funds Rate', float(merged_df['FEDFUNDS'].min()), float(merged_df['FEDFUNDS'].max()))
gdp = st.slider('GDP', float(merged_df['GDP'].min()), float(merged_df['GDP'].max()))
mortgage_rate = st.slider('30-Year Mortgage Rate', float(merged_df['MORTGAGE30US'].min()), float(merged_df['MORTGAGE30US'].max()))
population_growth = st.slider('Population Growth', float(merged_df['SPPOPGROWUSA'].min()), float(merged_df['SPPOPGROWUSA'].max()))
unemployment_rate = st.slider('Unemployment Rate', float(merged_df['UNRATE'].min()), float(merged_df['UNRATE'].max()))

# Make predictions based on user input
user_input_features = [fed_funds, gdp, mortgage_rate, population_growth, unemployment_rate]
prediction_result = make_predictions(user_input_features)

# Display prediction result
st.subheader('Predicted Home Price:')
st.write(f'${prediction_result:.2f}')

df = merged_df

import pandas as pd
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ... (previous code remains the same)

# Streamlit UI for Analysis
st.title('US Home Price Prediction and Analysis')
st.subheader('Model Performance')

# Display model performance
st.write(f'Mean Squared Error (MSE): {mse}')

# Sidebar
st.sidebar.title('Analysis Settings')

# Display distribution of key factors using histograms
st.sidebar.subheader('Distribution of Key Factors')
for feature in features:
    st.sidebar.subheader(feature)
    fig, ax = plt.subplots()
    ax.hist(merged_df[feature].dropna())
    st.sidebar.pyplot(fig)

# Display correlation matrix
st.sidebar.subheader('Correlation Matrix')
corr_matrix = merged_df[features + [target]].corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.sidebar.pyplot(fig)

# Display scatter plots for each factor against the target
st.sidebar.subheader('Scatter Plots')
for feature in features:
    st.sidebar.subheader(f'{feature} vs {target}')
    fig, ax = plt.subplots()
    ax.scatter(merged_df[feature], merged_df[target])
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    st.sidebar.pyplot(fig)



# Display predictions vs actual values
st.subheader('Predictions vs Actual Values')
fig, ax = plt.subplots(figsize=(20, 8))  # Adjust the figure size as needed
ax.plot(test_data['DATE'], test_data[target], label='Actual')
ax.plot(test_data['DATE'], test_predictions, label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel(target)
ax.legend()
ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
st.pyplot(fig)




# Display individual factor analysis
selected_factor = st.sidebar.selectbox('Select a factor for analysis', features)
st.subheader(f'Analysis of {selected_factor}')
st.line_chart(merged_df[[selected_factor, target]])

# Additional analysis and visualizations can be added as needed.
# Display a dropdown to select features
selected_feature = st.selectbox('Select a feature:', features)

# Plot line chart for the selected feature against the target
st.subheader(f'Line Chart: {selected_feature} vs {target}')
fig, ax = plt.subplots(figsize=(12, 6))

# Check if the selected feature is the target itself
if selected_feature != target:
    ax.plot(test_data['DATE'], test_data[selected_feature], label=selected_feature)
    
ax.plot(test_data['DATE'], test_data[target], label='Actual')
ax.set_xlabel('Date')
ax.set_ylabel(selected_feature if selected_feature != target else 'Target')
ax.legend()
ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
st.pyplot(fig)
