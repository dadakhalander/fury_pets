import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Load data
df = pd.read_csv("new_data_3.csv")

# Preprocess date and month
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
month_dict = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
df['Month'] = df['Month'].map(month_dict)

# Sidebar selection
store_areas = ["All"] + sorted(df['Area'].unique())
selected_area = st.sidebar.selectbox("Select Store Area", store_areas)

# Filter data by selected area
filtered_df = df if selected_area == "All" else df[df['Area'] == selected_area]

st.title("🐾 Fury Friends Store Dashboard")
st.markdown(f"### Insights for {'All Areas' if selected_area == 'All' else selected_area}")

# Total Profit by Store Area
st.subheader("Total Profit Across Stores")
store_profit_stats = df.groupby('Area')['Profit'].sum()
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x=store_profit_stats.index, y=store_profit_stats.values, palette='viridis', ax=ax1)
ax1.set_title("Total Profit Across Stores")
ax1.set_xlabel("Store Area")
ax1.set_ylabel("Total Profit")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Units Sold by Pet Type
st.subheader("Sales Distribution by Pet Type")
pet_sales = filtered_df.groupby('Pet')['Units Sld'].sum()
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=pet_sales.index, y=pet_sales.values, palette='viridis', ax=ax2)
ax2.set_title("Units Sold by Pet Type")
ax2.set_xlabel("Pet Type")
ax2.set_ylabel("Total Units Sold")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Units Sold by Area
st.subheader("Sales Distribution by Store Location")
location_sales = df.groupby('Area')['Units Sld'].sum()
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.barplot(x=location_sales.index, y=location_sales.values, palette='coolwarm', ax=ax3)
ax3.set_title("Total Sales by Store Area")
ax3.set_xlabel("Area")
ax3.set_ylabel("Units Sold")
plt.xticks(rotation=45)
st.pyplot(fig3)

# Sales by Pet Type and Area
st.subheader("Sales by Pet Type and Store Location")
grouped_sales = df.groupby(['Area', 'Pet'])['Units Sld'].sum().unstack()
fig4 = grouped_sales.plot(kind='bar', figsize=(14, 8), colormap='Set1')
plt.title("Sales by Pet Type and Store Location")
plt.xlabel("Store Location")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.legend(title='Pet Type', bbox_to_anchor=(1.05, 1))
st.pyplot(plt.gcf())

# Correlation of Profit with Area_encoded
st.subheader("Correlation of Profit with Location by Pet Type")
if 'Area_encoded' in df.columns:
    grouped_by_pet = df.groupby('Pet')[['Area_encoded', 'Profit']].corr().iloc[0::2, -1]
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    grouped_by_pet.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax5)
    ax5.set_title("Correlation of Profit with Location (encoded) by Pet")
    ax5.set_ylabel("Correlation Coefficient")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

# Profitability by Manager
st.subheader("Profitability per Store Manager")
manager_profit = filtered_df.groupby('Managers First Name')['Profit'].sum()
fig6, ax6 = plt.subplots(figsize=(12, 6))
manager_profit.plot(kind='bar', color='lightcoral', edgecolor='black', ax=ax6)
plt.title("Profitability per Store Manager")
plt.xlabel("Manager")
plt.ylabel("Total Profit")
plt.xticks(rotation=45)
st.pyplot(fig6)

# Profitability by Manager and Pet Type
st.subheader("Profitability by Manager and Pet Type")
manager_pet_profit = filtered_df.groupby(['Managers First Name', 'Pet'])['Profit'].sum()
fig7 = manager_pet_profit.unstack().plot(kind='bar', stacked=True, figsize=(12, 7), colormap='Set2')
plt.title("Profitability by Manager and Pet Type")
plt.xlabel("Manager")
plt.ylabel("Total Profit")
plt.legend(title="Pet Type", bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Pie charts for each manager
st.subheader("Profit Share by Pet Type (Per Manager)")
for manager in manager_pet_profit.index.levels[0]:
    if manager in manager_pet_profit.index:
        manager_data = manager_pet_profit.loc[manager]
        if isinstance(manager_data, pd.Series):
            fig, ax = plt.subplots(figsize=(7, 7))
            manager_data.plot.pie(autopct='%1.1f%%', startangle=90,
                                  colors=sns.color_palette("Set2", len(manager_data)), ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Profit Share for {manager}")
            st.pyplot(fig)

# Revenue share by pet type
st.subheader("Revenue by Pet Type")
pet_revenue = filtered_df.groupby('Pet')['Revenue'].sum()
fig8, ax8 = plt.subplots(figsize=(8, 8))
pet_revenue.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                 colors=sns.color_palette("Set2", len(pet_revenue)), ax=ax8)
ax8.set_ylabel("")
ax8.set_title("Revenue Share by Pet Type")
st.pyplot(fig8)

# Monthly Profit Trends
st.subheader("Monthly Profit Trends")
monthly_profit = df.groupby('Month')['Profit'].sum().reindex(month_dict.values())
fig9, ax9 = plt.subplots(figsize=(12, 6))
ax9.plot(monthly_profit.index, monthly_profit.values, marker='o', color='b', linestyle='-')
ax9.set_title("Profitability Trends Over Time")
ax9.set_xlabel("Month")
ax9.set_ylabel("Total Profit")
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig9)
