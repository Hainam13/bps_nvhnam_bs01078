# app.py - Milk Tea Sales Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# 1. Load Data
# ==========================
st.title("🍹 Milk Tea Sales Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("milk_tea_sales_sample.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

st.subheader("📊 Sample Data")
st.dataframe(df.head())

# ==========================
# 2. Data Overview
# ==========================
st.subheader("🔍 Data Information")
st.write("Số dòng trùng lặp:", df.duplicated().sum())
st.write("Missing values:")
st.write(df.isnull().sum())

# ==========================
# 3. Visualizations
# ==========================
st.subheader("📈 Visualizations")

# Time series
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['date'], df['sales'], label='Sales')
ax.set_title("Milk Tea Sales Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)

# Boxplots
for col in ['is_weekend', 'is_holiday', 'promotion']:
    fig, ax = plt.subplots()
    sns.boxplot(x=col, y="sales", data=df, ax=ax)
    ax.set_title(f"Sales vs {col.capitalize()}")
    st.pyplot(fig)

# Scatterplots
for col in ['temperature', 'rain_mm']:
    fig, ax = plt.subplots()
    sns.scatterplot(x=col, y="sales", data=df, ax=ax)
    ax.set_title(f"{col.capitalize()} vs Sales")
    st.pyplot(fig)

# Average sales by day of week
df['day_of_week'] = df['date'].dt.dayofweek
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df['day_name'] = df['day_of_week'].apply(lambda x: day_names[x])
avg_sales_by_day = df.groupby('day_name')['sales'].mean().reindex(day_names)

fig, ax = plt.subplots()
sns.barplot(x=avg_sales_by_day.index, y=avg_sales_by_day.values, palette="viridis", ax=ax)
ax.set_title("Average Sales by Day of the Week")
st.pyplot(fig)

# ==========================
# 4. Machine Learning Model
# ==========================
st.subheader("🤖 Linear Regression Model")

features = ['temperature', 'rain_mm', 'is_weekend', 'is_holiday', 'promotion']
target = 'sales'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"🔎 Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"🔎 Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"🔎 R-squared (R²): {r2:.2f}")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(y_test.values, label="Actual Sales", marker='o')
ax.plot(y_pred, label="Predicted Sales", marker='x')
ax.set_title("Actual vs Predicted Sales")
ax.set_xlabel("Test Sample Index")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.success("✅ App đã chạy thành công!")
