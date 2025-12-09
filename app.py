import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# PASTE YOUR CUSTOM CLASS HERE (EXACTLY AS IT WAS IN NOTEBOOK)
# ==========================================
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # ... logic ...
        return X
# ==========================================

# NOW load the model (Streamlit can now "see" the class above)
model = joblib.load('car_price_pipeline.pkl')

# The rest of your app code...




import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
pipeline = joblib.load("car_price_pipeline.pkl")

# --- 1. INTRODUCTION ---
st.title("Car Price Prediction Project")
st.subheader("Student Name: Husnain Raza")
st.write("""
**Project Overview:** This project analyzes a dataset of car features to predict their market price. 
We explore how factors like engine size, horsepower, and brand influence the cost of a vehicle.
""")

# --- 2. EDA SECTION (Exploratory Data Analysis) ---
if st.checkbox("Show Data Analysis (EDA)"):
    st.subheader("Exploratory Data Analysis")
    st.write("Here is how different features relate to Price:")
    
    # We need to load the data again just for plotting
    df = pd.read_csv("car_data.csv")
    
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

# --- 3. MODEL PREDICTION ---
st.subheader("Predict Car Price")
st.write("Enter the car details below to get a price estimate:")

# Input fields
enginesize = st.number_input("Engine Size", 50, 400, 120)
curbweight = st.number_input("Curb Weight", 1000, 4000, 2000)
horsepower = st.number_input("Horsepower", 40, 350, 100)
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
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")

# --- 4. MODEL METRICS & CONCLUSION ---
st.subheader("Model Performance")
st.write(f"**Model Accuracy (R² Score):** 88.0%")
st.write("**RMSE (Error):** ~3099")

st.subheader("Conclusion")
st.write("""
In conclusion, our model performs well with 88% accuracy. 
The analysis shows that **Horsepower** and **Engine Size** are the strongest predictors of a car's price. 
Premium brands like BMW and Porsche naturally cost more, regardless of size.

""")
