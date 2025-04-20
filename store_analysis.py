import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="🐾 Fury Friends Dashboard", layout="wide")
st.title("🐾 Fury Friends Store Dashboard")

# --- Load Data ---
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

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Options")
selected_area = st.sidebar.selectbox("📍 Select Store Location", sorted(df['Area'].dropna().unique()))
selected_pets = st.sidebar.multiselect("🐶 Select Pet Types", sorted(df['Pet'].dropna().unique()), default=sorted(df['Pet'].dropna().unique()))
selected_months = st.sidebar.multiselect("🗓️ Select Months", sorted(df['Month'].dropna().unique()), default=sorted(df['Month'].dropna().unique()))

# --- Filtered Data ---
filtered_df = df[
    (df['Area'] == selected_area) &
    (df['Pet'].isin(selected_pets)) &
    (df['Month'].isin(selected_months))
]

# --- Overview ---
st.markdown(f"### 📌 Analyzing: **{selected_area}**")
st.markdown(f"**Pet Types:** {', '.join(selected_pets)} | **Months:** {', '.join(selected_months)}")

total_profit = filtered_df['Profit'].sum()
st.metric(label="💰 Total Profit", value=f"£{total_profit:,.2f}")

# --- Profit by Pet Type ---
with st.expander("📊 Profit by Pet Type", expanded=True):
    pet_profit = filtered_df.groupby('Pet')['Profit'].sum().sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(pet_profit, x=pet_profit.index, y=pet_profit.values,
                     labels={'x': 'Pet Type', 'y': 'Total Profit (£)'},
                     title="Profit by Pet Type",
                     color=pet_profit.index, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(pet_profit, values=pet_profit.values, names=pet_profit.index,
                     title='Profit Share by Pet Type',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    if not pet_profit.empty:
        top_pet = pet_profit.idxmax()
        top_pet_value = pet_profit.max()
        st.success(f"🏆 Top-Selling Pet: **{top_pet}** (£{top_pet_value:,.2f})")

# --- Profit by Manager ---
with st.expander("🧑‍💼 Profit by Manager"):
    manager_profit = filtered_df.groupby('Manager Full Name')['Profit'].sum().sort_values(ascending=False)
    fig = px.bar(manager_profit, x=manager_profit.index, y=manager_profit.values,
                 labels={'x': 'Manager', 'y': 'Total Profit (£)'},
                 title="Manager Profit Breakdown",
                 color_discrete_sequence=['indianred'])
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# --- Monthly Profit Trend ---
with st.expander("📈 Monthly Profit Trend"):
    monthly_profit = filtered_df.groupby(filtered_df['Date'].dt.to_period("M"))['Profit'].sum().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_profit.index.astype(str),
        y=monthly_profit.values,
        mode='lines+markers',
        name='Profit',
        line=dict(color='dodgerblue')
    ))
    fig.update_layout(
        title="Monthly Profit Trend",
        xaxis_title="Month",
        yaxis_title="Profit (£)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Efficiency by Pet Type ---
with st.expander("📦 Profit per Unit Sold"):
    efficiency = filtered_df.groupby('Pet').apply(lambda x: x['Profit'].sum() / x['Units Sld'].sum()).sort_values()
    fig = px.bar(efficiency, x=efficiency.index, y=efficiency.values,
                 labels={'x': 'Pet Type', 'y': 'Profit per Unit'},
                 title="Efficiency by Pet Type",
                 color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig, use_container_width=True)

# --- Recommendations ---
with st.expander("📌 Smart Suggestions"):
    if len(monthly_profit) > 1 and monthly_profit.pct_change().iloc[-1] < 0:
        st.warning("📉 Monthly profit is declining. Consider reviewing promotions or pricing.")
    if efficiency.max() > 100 and efficiency.idxmax() != pet_profit.idxmax():
        st.info(f"🔍 Consider promoting **{efficiency.idxmax()}** — it yields high profit per unit!")

# --- Heatmap: Manager vs Pet ---
with st.expander("📐 Heatmap: Manager vs Pet Profit"):
    heatmap_data = filtered_df.pivot_table(index='Manager Full Name', columns='Pet', values='Profit', aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# --- Store Comparison Section ---
st.header("📍 Store Comparison Dashboard")

# Profit by Store
with st.expander("💰 Profit by Store"):
    store_comparison_df = df[(df['Pet'].isin(selected_pets)) & (df['Month'].isin(selected_months))]
    area_profit = store_comparison_df.groupby('Area')['Profit'].sum().sort_values(ascending=False)
    fig = px.bar(area_profit, x=area_profit.index, y=area_profit.values,
                 labels={'x': 'Area', 'y': 'Profit (£)'},
                 title="Total Profit by Store",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"🏆 **Top Performing Store:** {area_profit.idxmax()} (£{area_profit.max():,.2f})")

# Monthly Trend by Store
with st.expander("📈 Monthly Profit Trend by Store"):
    store_trend = store_comparison_df.copy()
    store_trend['Month_Year'] = store_trend['Date'].dt.to_period("M")
    fig = go.Figure()

    for area in store_trend['Area'].dropna().unique():
        area_data = store_trend[store_trend['Area'] == area].groupby('Month_Year')['Profit'].sum()
        fig.add_trace(go.Scatter(
            x=area_data.index.astype(str),
            y=area_data.values,
            mode='lines+markers',
            name=area
        ))

    fig.update_layout(
        title="Monthly Profit Trend by Store",
        xaxis_title="Month",
        yaxis_title="Profit (£)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Data Preview ---
with st.expander("📋 Preview Filtered Data"):
    st.dataframe(filtered_df, use_container_width=True)

# --- Download Button ---
st.subheader("📥 Export Filtered Data")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📁 Download Filtered CSV",
    data=csv_data,
    file_name=f"{selected_area}_filtered_data.csv",
    mime='text/csv'
)
