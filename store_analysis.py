import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("new_data_3.csv")

# Preprocess date column if available
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Set Streamlit page config
st.set_page_config(page_title="Store Performance Dashboard", layout="wide")
st.title("🐾 Store Performance Dashboard")

# Sidebar - Location selection
store_locations = df['Area'].unique()
selected_location = st.sidebar.selectbox("Select Store Location", store_locations)

# Filter data by selected location
filtered_df = df[df['Area'] == selected_location]

# --- Display Total Profit ---
total_profit = filtered_df['Profit'].sum()
st.subheader(f"Total Profit for {selected_location}")
st.metric(label="Total Profit ($)", value=f"{total_profit:,.2f}")

# --- Profit by Pet Type ---
st.subheader(f"Profit by Pet Type in {selected_location}")
pet_profit = filtered_df.groupby('Pet')['Profit'].sum().sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x=pet_profit.index, y=pet_profit.values, palette="Set2", ax=ax1)
ax1.set_xlabel("Pet Type")
ax1.set_ylabel("Profit")
ax1.set_title("Profit by Pet Type")
st.pyplot(fig1)

# --- Profit by Manager ---
st.subheader(f"Profit by Manager in {selected_location}")
manager_profit = filtered_df.groupby('Managers First Name')['Profit'].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=manager_profit.index, y=manager_profit.values, palette="Set3", ax=ax2)
ax2.set_xlabel("Manager")
ax2.set_ylabel("Profit")
ax2.set_title("Profit by Manager")
st.pyplot(fig2)

# --- Optional Pie Chart for Pet Profit Distribution ---
st.subheader(f"Pet Profit Share in {selected_location}")
fig3, ax3 = plt.subplots(figsize=(7, 7))
pet_profit.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax3, colors=sns.color_palette("Set2"))
ax3.set_ylabel('')
ax3.set_title("Profit Distribution by Pet Type")
st.pyplot(fig3)

# End of the app
st.markdown("---")
st.caption("Dashboard powered by Streamlit | Data from Fury Friends 4376")
