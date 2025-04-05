import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Page configuration
st.set_page_config(page_title="🐾 Fury Friends Dashboard", layout="wide")
st.title("🐾 Fury Friends Store Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("new_data_3.csv")
    
    # Debug: Print out the column names to verify
    st.write("Columns in the data:", df.columns)
    
    # Check if necessary columns exist
    if 'Managers First Name' in df.columns and 'Managers Surname' in df.columns:
        df['Manager Full Name'] = df['Managers First Name'] + ' ' + df['Managers Surname']
    else:
        st.error("Missing required columns: 'Managers First Name' or 'Managers Surname'. Please check your data.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
selected_area = st.sidebar.selectbox("📍 Select Store Location", sorted(df['Area'].unique()))
selected_pets = st.sidebar.multiselect("🐶 Select Pet Types", sorted(df['Pet'].unique()), default=sorted(df['Pet'].unique()))
selected_months = st.sidebar.multiselect("🗓️ Select Months", sorted(df['Month'].unique()), default=sorted(df['Month'].unique()))

# Filtered data
filtered_df = df[
    (df['Area'] == selected_area) &
    (df['Pet'].isin(selected_pets)) &
    (df['Month'].isin(selected_months))
]

st.subheader(f"📌 Analyzing: {selected_area}")
st.markdown(f"**Pet Types:** {', '.join(selected_pets)} | **Months:** {', '.join(selected_months)}")

# --- METRICS ---
total_profit = filtered_df['Profit'].sum()
st.metric(label="💰 Total Profit", value=f"${total_profit:,.2f}")

# --- PROFIT BY PET TYPE ---
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

# --- TOP-SELLING PET ---
top_pet = pet_profit.idxmax()
top_pet_value = pet_profit.max()
st.success(f"🏆 Top-Selling Pet: **{top_pet}** (${top_pet_value:,.2f})")

# --- PROFIT BY MANAGER ---
st.subheader("🧑‍💼 Profit by Manager")
manager_profit = filtered_df.groupby('Manager Full Name')['Profit'].sum().sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 4))
manager_profit.plot(kind='bar', color='lightcoral', edgecolor='black', ax=ax3)
ax3.set_ylabel("Profit")
ax3.set_title("Manager Profit Breakdown")
plt.xticks(rotation=45)
st.pyplot(fig3)

# --- MONTHLY PROFIT TREND ---
st.subheader("📈 Monthly Profit Trend")
monthly_profit = filtered_df.groupby(filtered_df['Date'].dt.to_period("M"))['Profit'].sum()

fig4, ax4 = plt.subplots(figsize=(10, 4))
monthly_profit.plot(marker='o', linestyle='-', color='blue', ax=ax4)
ax4.set_ylabel("Profit")
ax4.set_xlabel("Month")
ax4.set_title("Profit Over Time")
plt.xticks(rotation=45)
st.pyplot(fig4)

# --- EFFICIENCY: PROFIT PER UNIT SOLD ---
st.subheader("📦 Profit per Unit Sold (Efficiency)")
efficiency = filtered_df.groupby('Pet').apply(lambda x: x['Profit'].sum() / x['Units Sld'].sum())

fig5, ax5 = plt.subplots(figsize=(8, 4))
efficiency.sort_values().plot(kind='bar', color='skyblue', ax=ax5)
ax5.set_ylabel("Profit per Unit")
ax5.set_title("Efficiency by Pet Type")
st.pyplot(fig5)

# --- SMART RECOMMENDATIONS ---
st.subheader("📌 Smart Suggestions")
if len(monthly_profit) > 1 and monthly_profit.pct_change().iloc[-1] < 0:
    st.warning("📉 Monthly profit is declining. Consider reviewing promotions or pricing.")
if efficiency.max() > 100 and efficiency.idxmax() != top_pet:
    st.info(f"🔍 Consider promoting **{efficiency.idxmax()}** — it yields high profit per unit!")

# --- HEATMAP: MANAGER vs PET PROFIT ---
st.subheader("📐 Profit Heatmap: Manager vs Pet")
heatmap_data = filtered_df.pivot_table(index='Manager Full Name', columns='Pet', values='Profit', aggfunc='sum', fill_value=0)

fig6, ax6 = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax6)
st.pyplot(fig6)

# --- DATA TABLE ---
st.subheader("📋 Preview of Filtered Data")
st.dataframe(filtered_df[['Date', 'Area', 'Pet', 'Units Sld', 'Revenue', 'Profit', 'Manager Full Name']].head(50))

# --- DOWNLOAD BUTTON ---
st.subheader("📥 Export Filtered Data")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{selected_area}_filtered_data.csv",
    mime='text/csv'
)
