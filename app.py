import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. SETUP & MODEL TRAINING (Cached) ---
# We use @st.cache_resource so the model trains only once when the app starts.
# This avoids the "AttributeError" from loading incompatible .pkl files.
@st.cache_resource
def load_and_train_model():
    # 1. Load Data
    try:
        df = pd.read_csv("car_data.csv")
    except FileNotFoundError:
        st.error("Error: 'car_data.csv' not found. Please upload it to your GitHub repository.")
        return None, None, None

    # 2. Preprocessing (Matching your Notebook Logic)
    # Ensure column names are clean
    # Check if 'CarName' exists (CamelCase) or 'carname' (lowercase)
    if 'CarName' in df.columns:
        car_col = 'CarName'
    else:
        car_col = 'carname' # Fallback
        
    # Extract Brand
    df['brand'] = df[car_col].apply(lambda x: x.split(' ')[0].lower())
    
    # Fix Spelling (from your notebook)
    brand_corrections = {
        'maxda': 'mazda', 'porcshce': 'porsche', 'toyouta': 'toyota', 
        'vokswagen': 'volkswagen', 'vw': 'volkswagen'
    }
    df['brand'] = df['brand'].replace(brand_corrections)

    # 3. Define Features
    features = ['enginesize', 'curbweight', 'horsepower', 'citympg', 'highwaympg', 'brand']
    target = 'price'
    
    # Handle capitalization differences in CSV
    # Rename columns to lowercase to ensure matching
    df.columns = [c.lower() for c in df.columns]
    
    X = df[features]
    y = df[target]

    # 4. Build Pipeline
    numeric_features = ['enginesize', 'curbweight', 'horsepower', 'citympg', 'highwaympg']
    categorical_features = ['brand']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 5. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # Calculate Metrics
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return pipeline, df, (r2, rmse)

# Load the model and data
pipeline, df, metrics = load_and_train_model()

# --- 2. INTRODUCTION ---
st.title("Car Price Prediction Project")
st.subheader("Student Name: Husnain Raza")
st.write("""
**Project Overview:** This project analyzes a dataset of car features to predict their market price. 
We explore how factors like engine size, horsepower, and brand influence the cost of a vehicle.
""")

# --- 3. EDA SECTION ---
if df is not None and st.checkbox("Show Data Analysis (EDA)"):
    st.subheader("Exploratory Data Analysis")
    st.write("Here is how different features relate to Price:")
    
    # Plot 1: Distribution of Prices
    st.write("**1. Distribution of Car Prices**")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax1)
    st.pyplot(fig1)
    
    # Plot 2: Price vs Horsepower
    st.write("**2. Price vs Horsepower**")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df['horsepower'], y=df['price'], ax=ax2)
    st.pyplot(fig2)

# --- 4. MODEL PREDICTION ---
st.subheader("Predict Car Price")
st.write("Enter the car details below to get a price estimate:")

# Input fields
col1, col2 = st.columns(2)
with col1:
    enginesize = st.number_input("Engine Size", 50, 400, 120)
    curbweight = st.number_input("Curb Weight", 1000, 5000, 2500)
    horsepower = st.number_input("Horsepower", 40, 500, 150)
with col2:
    citympg = st.number_input("City MPG", 5, 60, 25)
    highwaympg = st.number_input("Highway MPG", 5, 60, 30)
    brand = st.text_input("Brand (e.g. toyota, bmw, audi)", "toyota")

# Create a dataframe for the model
input_df = pd.DataFrame([{
    "enginesize": enginesize,
    "curbweight": curbweight,
    "horsepower": horsepower,
    "citympg": citympg,
    "highwaympg": highwaympg,
    "brand": brand.lower() # Convert to lowercase to match model
}])

if st.button("Predict Price"):
    if pipeline is not None:
        try:
            prediction = pipeline.predict(input_df)[0]
            st.success(f"Estimated Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model failed to load. Check dataset.")

# --- 5. MODEL METRICS ---
if metrics:
    st.subheader("Model Performance")
    st.write(f"**Model Accuracy (R² Score):** {metrics[0]*100:.1f}%")
    st.write(f"**RMSE (Error):** ${metrics[1]:,.0f}")

st.subheader("Conclusion")
st.write("""
In conclusion, our model performs reliably on the test set. 
The analysis shows that **Horsepower** and **Engine Size** are the strongest predictors of a car's price. 
Premium brands like BMW and Porsche naturally cost more, regardless of size.
""")
