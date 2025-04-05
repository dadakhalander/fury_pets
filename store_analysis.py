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

# --- Profit by Manager ---
st.subheader("🧑‍💼 Profit by Manager")
manager_profit = filtered_df.groupby('Manager Full Name')['Profit'].sum().sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 4))
manager_profit.plot(kind='bar', color='lightcoral', edgecolor='black', ax=ax3)
ax3.set_ylabel("Profit")
ax3.set_title("Manager Profit Breakdown")
plt.xticks(rotation=45)
st.pyplot(fig3)

# --- Monthly Profit Trend ---
st.subheader("📈 Monthly Profit Trend")
monthly_profit = filtered_df.groupby(filtered_df['Date'].dt.to_period("M"))['Profit'].sum()

fig4, ax4 = plt.subplots(figsize=(10, 4))
monthly_profit.plot(marker='o', linestyle='-', color='blue', ax=ax4)
ax4.set_ylabel("Profit")
ax4.set_xlabel("Month")
ax4.set_title("Profit Over Time")
plt.xticks(rotation=45)
st.pyplot(fig4)

# --- Profit per Unit Sold ---
st.subheader("📦 Profit per Unit Sold")
efficiency = filtered_df.groupby('Pet').apply(lambda x: x['Profit'].sum() / x['Units Sld'].sum())

fig5, ax5 = plt.subplots(figsize=(8, 4))
efficiency.sort_values().plot(kind='bar', color='skyblue', ax=ax5)
ax5.set_ylabel("Profit per Unit")
ax5.set_title("Efficiency by Pet Type")
st.pyplot(fig5)

# --- Recommendations ---
st.subheader("📌 Smart Suggestions")

if len(monthly_profit) > 1 and monthly_profit.pct_change().iloc[-1] < 0:
    st.warning("📉 Monthly profit is declining. Consider reviewing promotions or pricing.")
if efficiency.max() > 100 and efficiency.idxmax() != top_pet:
    st.info(f"🔍 Consider promoting **{efficiency.idxmax()}** — it yields high profit per unit!")

# --- Heatmap: Manager vs Pet Profit ---
st.subheader("📐 Profit Heatmap: Manager vs Pet")
heatmap_data = filtered_df.pivot_table(index='Manager Full Name', columns='Pet', values='Profit', aggfunc='sum', fill_value=0)

fig6, ax6 = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax6)
st.pyplot(fig6)

# --- Data Table ---
st.subheader("📋 Preview of Filtered Data")
st.dataframe(filtered_df.head(50))

# --- Download Button ---
st.subheader("📥 Export Filtered Data")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{selected_area}_filtered_data.csv",
    mime='text/csv'
)

# --- Machine Learning Model ---
st.sidebar.header("🤖 Optional: Machine Learning Prediction Model")

use_ml_model = st.sidebar.checkbox("Enable Machine Learning Predictions for Future Profit")

if use_ml_model:
    model_type = st.radio("Choose Model", ("Basic Model", "Tuned Model"))

    if model_type == "Basic Model":
        st.subheader("🤖 Basic Machine Learning Predictions for Future Profit")

        # Define features and target for training the model
        features = ['Area', 'Pet', 'Units Sld', 'Manager Full Name', 'Month']
        target = 'Profit'

        # Preprocessing: OneHotEncode categorical features and handle numeric features
        X = filtered_df[features]
        y = filtered_df[target]

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

        # Display prediction results
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
            - **Actual Profit**: The real profit values from the test data.
            - **Predicted Profit**: The values predicted by the machine learning model based on the training data.
            - The red line represents the ideal prediction where the predicted value matches the actual value.
            - The closer the points are to this red line, the better the model's predictions.
        """)

        st.pyplot(fig7)

    if model_type == "Tuned Model":
        st.subheader("🤖 Tuned Machine Learning Predictions for Future Profit")

        # Define features and target for training the model
        features = ['Area', 'Pet', 'Units Sld', 'Manager Full Name', 'Month']
        target = 'Profit'

        # Preprocessing: OneHotEncode categorical features and handle numeric features
        X = filtered_df[features]
        y = filtered_df[target]

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

        # Hyperparameter grid for tuning
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }

        # Perform Grid Search for best parameters
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        # Best parameters and best model
        st.write(f"Best Parameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # Train-test split (we train on historical data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict future profits based on the input data
        predictions = best_model.predict(X_test)

        # Display prediction results
        st.write("### Tuned Model Prediction Results")
        predicted_profits = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        st.write(predicted_profits.head())

        # --- Visualizing the Prediction Errors ---
        st.subheader("📊 Tuned Model Prediction Error Analysis")
        fig8, ax8 = plt.subplots(figsize=(10, 5))
        sns.regplot(x=y_test, y=predictions, ax=ax8, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        ax8.set_xlabel("Actual Profit")
        ax8.set_ylabel("Predicted Profit")
        ax8.set_title("Actual vs Predicted Profit (Tuned Model)")

        # Explanation below the graph
        st.markdown("""
            - **Actual Profit**: The real profit values from the test data.
            - **Predicted Profit**: The values predicted by the tuned machine learning model based on the training data.
            - The red line represents the ideal prediction where the predicted value matches the actual value.
            - The closer the points are to this red line, the better the model's predictions.
        """)

        st.pyplot(fig8)
