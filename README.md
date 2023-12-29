# US Home Price Prediction and Analysis
-> Overview
-This code provides a comprehensive solution for predicting and analyzing US home prices based on various economic indicators. It includes data extraction, cleaning, merging, imputation, model training, and a Streamlit app for user interaction.
- Run file in commond promp using "<path> streamlit run app.py"
  
-> Code Structure
-Data Extraction and Merging
The script starts by extracting data from a provided zip file containing multiple CSV files. It loads each dataset into Pandas DataFrames and merges them based on the common 'DATE' column.

-Data Cleaning
The data is filtered based on a specified date ('1987-01-01') to include records from that date onwards. Missing values are handled using the SimpleImputer with a mean strategy.

-Model Training
A linear regression model is built using the scikit-learn library. The model is trained on features such as Federal Funds Rate, GDP, Mortgage Rate, Population Growth, and Unemployment Rate.

-Streamlit App
The Streamlit app allows users to input values for key economic indicators, and the trained model predicts the corresponding home price. The app also provides an analysis section with visualizations of model performance, feature distributions, a correlation matrix, and scatter plots.

-Analysis Settings Sidebar
The Streamlit app includes a sidebar with settings for analyzing the distribution of key factors, a correlation matrix, and scatter plots.

-Visualization
The code generates various visualizations using Matplotlib and Seaborn. It displays the model's predictions against actual values, distribution histograms, correlation matrices, and scatter plots.

-Additional Features
User-friendly input sliders in the Streamlit app for predicting home prices.
A dropdown to select features for individual analysis.
Line charts for visualizing the relationship between selected features and the target variable.

-📊 Visualizations: Histograms, correlation matrix, scatter plots, and line charts.
-📈 Model Performance: Display of Mean Squared Error (MSE).
-⚙️ Settings Sidebar: Analysis settings for distributions and correlations.
-🔄 Data Processing: Data extraction, merging, and cleaning.
-🚀 Streamlit App: User-friendly interface for predictions and analysis.
