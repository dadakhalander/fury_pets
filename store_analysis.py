import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO

# Page configuration
st.set_page_config(page_title="🐾 Fury Friends Dashboard", layout="wide")
st.title("🐾 Fury Friends Store Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("new_data_3.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    
    # Add manager full name column
    if 'Managers First Name' in df.columns and 'Managers Surname' in df.columns:
        df['Manager Full Name'] = df['Managers First Name'] + ' ' + df['Managers Surname']
    else:
        st.error("Missing 'Managers First Name' or 'Managers Surname' column.")
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
selected_area = st.sidebar.selectbox("📍 Select Store Location", sorted(df['Area'].unique()))
selected_pets = st.sidebar.multiselect("🐶 Select Pet Types", sorted(df['Pet'].unique()), default=sorted(df['Pet'].unique()))
selected_months = st.sidebar.multiselect("🗓️ Select Months", sorted(df['Month'].unique()), default=sorted(df['Month'].unique()))

# Filter data
filtered_df = df[
    (df['Area'] == selected_area) &
    (df['Pet'].isin(selected_pets)) &
    (df['Month'].isin(selected_months))
]

st.subheader(f"📌 Analyzing: {selected_area}")
st.markdown(f"**Pet Types:** {', '.join(selected_pets)} | **Months:** {', '.join(selected_months)}")

# --- Metrics ---
total_profit = filtered_df['Profit'].sum()
st.metric(label="💰 Total Profit", value=f"${total_profit:,.2f}")

# --- Profit by Pet Type ---
st.subheader("📊 Profit by Pet Type")
pet_profit = filtered_df.groupby('Pet')['Profit'].sum().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=pet_profit.index, y=pet_profit.values, palette='Set2', ax=ax1)
    ax1.set_title("Profit by Pet")
    ax1.set_ylabel("Profit")
    ax1.set_xlabel("Pet Type")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    pet_profit.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax2, colors=sns.color_palette("Set2"))
    ax2.set_ylabel("")
    ax2.set_title("Profit Share by Pet Type")
    st.pyplot(fig2)

# --- Top-Selling Pet ---
top_pet = pet_profit.idxmax()
top_pet_value = pet_profit.max()
st.success(f"🏆 Top-Selling Pet: **{top_pet}** (${top_pet_value:,.2f})")

# --- Machine Learning Model for Store Prediction ---
st.sidebar.header("🤖 Optional: Machine Learning Prediction Model")

use_ml_model = st.sidebar.checkbox("Enable Machine Learning Predictions for Future Profit")

if use_ml_model:
    # Store selection for prediction
    store_options = sorted(df['Area'].unique())
    selected_store = st.sidebar.selectbox("Select Store for Profit Prediction", store_options)

    # Filter the dataset for the selected store
    store_df = df[df['Area'] == selected_store]

    st.subheader(f"🤖 Profit Prediction for {selected_store}")

    # Define features and target for training the model
    features = ['Area', 'Pet', 'Units Sld', 'Manager Full Name', 'Month']
    target = 'Profit'

    # Preprocessing: OneHotEncode categorical features and handle numeric features
    X = store_df[features]
    y = store_df[target]

    # Create a column transformer for one-hot encoding categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Area', 'Pet', 'Manager Full Name', 'Month']),
            ('num', 'passthrough', ['Units Sld'])
        ])

    # Define RandomForest model within a pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Train-test split (we train on historical data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict future profits based on the input data
    predictions = model.predict(X_test)

    # Display prediction results for the selected store
    st.write("### Model Prediction Results")
    predicted_profits = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    st.write(predicted_profits.head())

    # --- Visualizing the Prediction Errors ---
    st.subheader("📊 Prediction Error Analysis")
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    sns.regplot(x=y_test, y=predictions, ax=ax7, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
    ax7.set_xlabel("Actual Profit")
    ax7.set_ylabel("Predicted Profit")
    ax7.set_title("Actual vs Predicted Profit")

    # Explanation below the graph
    st.markdown("""
        - **Actual Profit**: The real profit values from the data.
        - **Predicted Profit**: The values predicted by the machine learning model based on the training data.
        - The **red line** represents the ideal prediction where the predicted value matches the actual value.
        - The closer the points are to this **red line**, the better the model's predictions are.
        - **Discrepancies** from the line indicate areas where the model is either overestimating or underestimating the profit.
    """)

    st.pyplot(fig7)
