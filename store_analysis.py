import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page config
st.set_page_config(page_title="🐾 Fury Friends Dashboard", layout="wide")
st.title("🐾 Fury Friends Store Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("new_data_3.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Month'] = df['Date'].dt.month_name()

    if {'Managers First Name', 'Managers Surname'}.issubset(df.columns):
        df['Manager Full Name'] = df['Managers First Name'] + ' ' + df['Managers Surname']
    else:
        df['Manager Full Name'] = df.get('Manager Full Name', 'Unknown')

    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
selected_area = st.sidebar.selectbox(" Select Store Location", sorted(df['Area'].dropna().unique()))
selected_pets = st.sidebar.multiselect(" Select Pet Types", sorted(df['Pet'].dropna().unique()), default=sorted(df['Pet'].dropna().unique()))
selected_months = st.sidebar.multiselect(" Select Months", sorted(df['Month'].dropna().unique()), default=sorted(df['Month'].dropna().unique()))

# Filtered Data
filtered_df = df[
    (df['Area'] == selected_area) &
    (df['Pet'].isin(selected_pets)) &
    (df['Month'].isin(selected_months))
]

st.subheader(f" Analyzing: {selected_area}")
st.markdown(f"**Pet Types:** {', '.join(selected_pets)} | **Months:** {', '.join(selected_months)}")

# Total Profit Metric
total_profit = filtered_df['Profit'].sum()
st.metric(label=" Total Profit", value=f"£{total_profit:,.2f}")

# Profit by Pet Type
st.subheader(" Profit by Pet Type")
pet_profit = filtered_df.groupby('Pet')['Profit'].sum().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.barplot(x=pet_profit.index, y=pet_profit.values, palette='Set2', ax=ax1)
    ax1.set_title("Profit by Pet")
    ax1.set_ylabel("Profit")
    ax1.set_xlabel("Pet Type")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    pet_profit.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax2, colors=sns.color_palette("Set2"))
    ax2.set_ylabel("")
    ax2.set_title("Profit Share by Pet Type")
    st.pyplot(fig2)

# Top-Selling Pet
if not pet_profit.empty:
    top_pet = pet_profit.idxmax()
    top_pet_value = pet_profit.max()
    st.success(f" Top-Selling Pet: **{top_pet}** (£{top_pet_value:,.2f})")

# Profit by Manager
st.subheader(" Profit by Manager")
manager_profit = filtered_df.groupby('Manager Full Name')['Profit'].sum().sort_values(ascending=False)

fig3, ax3 = plt.subplots()
manager_profit.plot(kind='bar', color='lightcoral', edgecolor='black', ax=ax3)
ax3.set_ylabel("Profit")
ax3.set_title("Manager Profit Breakdown")
plt.xticks(rotation=45)
st.pyplot(fig3)

# Monthly Profit Trend
st.subheader(" Monthly Profit Trend")
monthly_profit = filtered_df.groupby(filtered_df['Date'].dt.to_period("M"))['Profit'].sum()

fig4, ax4 = plt.subplots()
monthly_profit.plot(marker='o', linestyle='-', color='blue', ax=ax4)
ax4.set_ylabel("Profit")
ax4.set_xlabel("Month")
ax4.set_title("Profit Over Time")
plt.xticks(rotation=45)
st.pyplot(fig4)

# Profit per Unit Sold
st.subheader(" Profit per Unit Sold")
efficiency = filtered_df.groupby('Pet').apply(lambda x: x['Profit'].sum() / x['Units Sld'].sum())

fig5, ax5 = plt.subplots()
efficiency.sort_values().plot(kind='bar', color='skyblue', ax=ax5)
ax5.set_ylabel("Profit per Unit")
ax5.set_title("Efficiency by Pet Type")
st.pyplot(fig5)

# Smart Suggestions
st.subheader(" Smart Suggestions")
if len(monthly_profit) > 1 and monthly_profit.pct_change().iloc[-1] < 0:
    st.warning(" Monthly profit is declining. Consider reviewing promotions or pricing.")
if efficiency.max() > 100 and efficiency.idxmax() != pet_profit.idxmax():
    st.info(f" Consider promoting **{efficiency.idxmax()}** — it yields high profit per unit!")

# Heatmap: Manager vs Pet
st.subheader(" Profit Heatmap: Manager vs Pet")
heatmap_data = filtered_df.pivot_table(index='Manager Full Name', columns='Pet', values='Profit', aggfunc='sum', fill_value=0)

fig6, ax6 = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax6)
st.pyplot(fig6)

# Store Comparison
st.header(" Store Comparison Dashboard")
store_comparison_df = df[(df['Pet'].isin(selected_pets)) & (df['Month'].isin(selected_months))]

st.subheader(" Total Profit by Store Location")
area_profit = store_comparison_df.groupby('Area')['Profit'].sum().sort_values(ascending=False)

fig7, ax7 = plt.subplots()
sns.barplot(x=area_profit.index, y=area_profit.values, palette='pastel', ax=ax7)
ax7.set_ylabel("Total Profit (£)")
ax7.set_title("Profit by Store")
plt.xticks(rotation=45)
st.pyplot(fig7)

# Store-wise Monthly Trend
st.subheader(" Monthly Profit Trend by Store")
store_trend = store_comparison_df.copy()
store_trend['Month_Year'] = store_trend['Date'].dt.to_period("M")

fig8, ax8 = plt.subplots(figsize=(12, 5))
for area in store_trend['Area'].dropna().unique():
    area_data = store_trend[store_trend['Area'] == area].groupby('Month_Year')['Profit'].sum()
    area_data.plot(ax=ax8, label=area)

ax8.set_ylabel("Monthly Profit")
ax8.set_xlabel("Month")
ax8.set_title("Store-wise Monthly Profit Trend")
ax8.legend(title="Store")
st.pyplot(fig8)

# Store Insights
st.subheader(" Store Performance Insights")
top_store = area_profit.idxmax()
st.success(f" **Top Performing Store:** {top_store} (£{area_profit.max():,.2f})")

# Preview Data
st.subheader(" Preview of Filtered Data")
st.dataframe(filtered_df.head(50))

# Export Data
st.subheader(" Export Filtered Data")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{selected_area}_filtered_data.csv",
    mime='text/csv'
)

# Optional Map (only if lat/long exists)
if {'lat', 'long'}.issubset(df.columns):
    st.subheader(" Store Locations Map")
    map_df = df.dropna(subset=['lat', 'long'])
    st.map(map_df[['lat', 'long']])
else:
    st.info("📍 Latitude and Longitude data not available. Map view is disabled.")
