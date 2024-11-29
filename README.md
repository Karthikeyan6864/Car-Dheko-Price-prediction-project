# Car Price Prediction: Streamlit App & Machine Learning Model
## Overview
This project predicts the price of used cars in Indian Rupees (₹) based on historical data. By leveraging machine learning, the application estimates car prices based on features such as Fuel Type, Engine Type, Model, Mileage, Kilometers Driven, Max Power, and others.

The web application, built using Streamlit, provides an intuitive interface for car buyers, sellers, and enthusiasts to predict used car prices efficiently. The backend model is powered by Ridge Regression, ensuring accurate predictions even with multicollinear data.

## Project Highlights
### Machine Learning Model: Utilized Ridge Regression for handling multicollinearity and continuous prediction tasks effectively.
Web Application: Developed with Streamlit for a user-friendly experience to input car details and receive price predictions.
### Prediction Inputs: Supports both categorical (e.g., Fuel Type, Model) and numerical (e.g., Mileage, Max Power) inputs.
### Output: Displays the predicted price in a clear and intuitive format with proper currency representation (₹).
## Data Processing and Model Development
### Data Preprocessing
### Data Sources:

Collected car data from six major Indian cities: bangalore_cars.xlsx, chennai_cars.xlsx, etc.
Unified all city datasets into a single DataFrame for analysis and modeling.
### Data Cleaning:

Removed irrelevant and redundant columns (e.g., Ownership and ft).
Standardized categorical values and cleaned unknown entries.
### Addressed missing values:
Dropped columns with over 15% missing data.
Applied mean imputation for numerical features and mode imputation for categorical features.
### Feature Engineering:

Parsed fields like Max Power and Torque into numeric formats using regular expressions.
Standardized price values (e.g., lakh, crore) into numeric formats.
Encoded categorical features using target encoding and label encoding.
### Outlier Detection:

Removed outliers using the Z-score method for columns like Mileage and Max Power.
## Exploratory Data Analysis (EDA)
## Descriptive Statistics:

Used histograms and boxplots to analyze price distribution and detect anomalies.
### Correlation Analysis:

Visualized feature relationships using heatmaps and scatter plots (e.g., Mileage vs. Price).
### Categorical Diversity:

Assessed the impact of categorical features like Fuel Type and Color on price.
## Model Development
### Model Selection:

Chose Ridge Regression for its ability to handle multicollinearity.
### Data Splitting:

Split the dataset into training and test sets for evaluation.
### Performance Metrics:

### R² Score: Evaluated explained variance.
Mean Squared Error (MSE) and Mean Absolute Error (MAE) to measure prediction accuracy.
### Pipeline Creation:

Built a pipeline incorporating StandardScaler for normalization and Ridge Regression for prediction.
## Streamlit Application
## App Features
### Interactive Inputs:

Input car details (e.g., Fuel Type, Model, Mileage, Max Power) via dropdowns and sliders.
### Real-time Predictions:

Dynamically predicts car prices based on user inputs.
### User-Friendly Output:

Displays the predicted price with proper formatting in Indian Rupees (₹).

## Deployment
The app can be deployed to Streamlit Cloud, Heroku, or other hosting platforms for public access.
## Conclusion
This project integrates machine learning and web development to create a practical solution for predicting car prices. The app offers:

-A robust and interactive platform for car buyers, sellers, and dealers.
-Reliable price predictions backed by rigorous data preprocessing and analysis.
-Whether you're a car enthusiast, dealer, or buyer, this tool helps make informed decisions in the used car market.
