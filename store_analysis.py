import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Page config
st.set_page_config(page_title="Fury Friends Dashboard", layout="wide")
st.title("🐾 Fury Friends Store Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("new_data_3.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

area_options = df['Area'].unique()
selected_area = st.sidebar.selectbox("Select Store Location", sorted(area_options))

pet_options = df['Pet'].unique()
selected_pets = st.sidebar.multiselect("Select Pet Types", options=sorted(pet_options), default=sorted(pet_options))

month_options = df['Month'].unique()
selected_months = st.sidebar.multiselect("Select Months", options=month_options, default=month_options)

# Filter data
filtered_df = df[
    (df['Area'] == selected_area) &
    (df['Pet'].isin(selected_pets)) &
    (df['Month'].isin(selected_months))
]

st.subheader(f"📍 Store Area: {selected_area}")
st.write(f"Filters applied: Pet Types = {', '.join(selected_pets)} | Months = {', '.join(selected_months)}")

# Display total profit
total_profit = filtered_df['Profit'].sum()
st.metric(label="💰 Total Profit", value=f"${total_profit:,.2f}")

# Profit by Pet Type
st.subheader("🐕 Profit by Pet Type")
pet_profit = filtered_df.groupby('Pet')['Profit'].sum().sort_values(ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=pet_profit.index, y=pet_profit.values, palette='Set2', ax=ax1)
    ax1.set_title("Profit by Pet Type")
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

# Profit by Manager
st.subheader("👨‍💼 Profit by Store Manager")
manager_profit = filtered_df.groupby('Managers First Name')['Profit'].sum().sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 5))
manager_profit.plot(kind='bar', color='lightcoral', edgecolor='black', ax=ax3)
ax3.set_title("Profit by Manager")
ax3.set_ylabel("Profit")
ax3.set_xlabel("Manager")
plt.xticks(rotation=45)
st.pyplot(fig3)

# Data table preview
st.subheader("📋 Filtered Data Preview")
st.dataframe(filtered_df.head(50))

# Download filtered data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(filtered_df)

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv_data,
    file_name=f"filtered_data_{selected_area}.csv",
    mime='text/csv'
)
