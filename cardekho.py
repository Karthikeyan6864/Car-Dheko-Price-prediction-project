import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import streamlit as st


file_path = r"C:\Users\Karthikeyan\Downloads\dfupdate.xlsx"
df = pd.read_excel(file_path)


print(df.info())

# Function to find outliers and their count using Z-Score
def find_outliers_zscore_with_count(dataframe, columns, threshold=3):
    outliers_indices = set()
    for column in columns:
        z_scores = zscore(dataframe[column])
        outliers = dataframe[abs(z_scores) > threshold]
        outliers_indices.update(outliers.index)  
    return outliers_indices

# Remove outliers from the dataset
columns_to_check = ['Seating Capacity', 'modelYear', 'km']
outlier_indices = find_outliers_zscore_with_count(df, columns_to_check)
df_cleaned = df.drop(index=outlier_indices)


features = ['km', 'ownerNo', 'Max Power', 'modelYear', 'Seating Capacity', 'Mileage',
            'Fuel Type_Encoded', 'Engine Type_Encoded', 'city_Encoded', 'model_Encoded',
            'bt_Encoded', 'Color_Encoded']
target = 'price'


X = df_cleaned[features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R² Score: {cv_scores.mean() * 100:.2f}%")

pipeline.fit(X_train, y_train)

train_r2 = r2_score(y_train, pipeline.predict(X_train))
test_r2 = r2_score(y_test, pipeline.predict(X_test))

print(f"Training R²: {train_r2 * 100:.2f}%")
print(f"Testing R²: {test_r2 * 100:.2f}%")

model_filename = "ridge_regression_model.pkl"
joblib.dump(pipeline, model_filename)
print(f"Model saved as {model_filename}")

# Streamlit app
st.title("Car Price Prediction App")


categorical_columns = ['Fuel Type', 'Engine Type', 'city', 'model', 'bt', 'Color']
encoded_columns = [f"{col}_Encoded" for col in categorical_columns]

# Create dropdowns for categorical features
inputs = {}
for non_encoded, encoded in zip(categorical_columns, encoded_columns):
    options = df_cleaned[non_encoded].unique()  
    selected = st.sidebar.selectbox(non_encoded, options)  
    encoded_value = df_cleaned.loc[df_cleaned[non_encoded] == selected, encoded].iloc[0]
    inputs[encoded] = encoded_value  

# Inputs for numerical features
inputs['km'] = st.sidebar.number_input("Kilometers Driven", min_value=0, value=10000)
inputs['ownerNo'] = st.sidebar.number_input("Number of Owners", min_value=0, value=1)
inputs['Max Power'] = st.sidebar.number_input("Max Power (in bhp)", min_value=0.0, value=100.0)
inputs['modelYear'] = st.sidebar.number_input("Model Year", min_value=2000, value=2020)
inputs['Seating Capacity'] = st.sidebar.number_input("Seating Capacity", min_value=2, value=5)
inputs['Mileage'] = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, value=20.0)


user_data = pd.DataFrame([inputs])


user_data = user_data[features] 

# Predict the price
if st.button("Predict"):
    
    model = joblib.load(model_filename)
    
    predicted_price = model.predict(user_data)[0]
    
    st.subheader(f"Predicted Price: ₹{predicted_price:,.2f}")

