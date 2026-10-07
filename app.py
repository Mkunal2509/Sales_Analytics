import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import subprocess
import io
import os
import glob
from datetime import datetime, timedelta

st.set_page_config(page_title="Retail Product & Sales Analytics", layout="wide")

# --- 1. R PATH DETECTION ---
def find_rscript_path():
    windows_candidates = glob.glob(r"C:\Program Files\R\R-*\bin\x64\Rscript.exe") + \
                         glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
    if windows_candidates:
        return windows_candidates[-1]
    
    for path in ["/usr/local/bin/Rscript", "/usr/bin/Rscript", "/opt/homebrew/bin/Rscript"]:
        if os.path.exists(path):
            return path
            
    return "Rscript"

# --- 2. PRODUCT CATEGORY ASSIGNMENT ---
def assign_product_category(desc):
    """Categorizes text descriptions using retail keywords."""
    if not isinstance(desc, str):
        return "Other / General Giftware"
    
    d = desc.upper()
    if any(k in d for k in ["BAG", "LUNCH BAG", "TOTE", "BOX", "STORAGE", "CONTAINER"]):
        return "Bags & Storage"
    elif any(k in d for k in ["HEART", "LIGHT", "CANDLE", "HANGER", "CLOCK", "MIRROR", "CUSHION", "FRAME", "SIGN"]):
        return "Home & Décor"
    elif any(k in d for k in ["CAKE", "BAKING", "TEAPOT", "MUG", "CUP", "BOWL", "PLATE", "CUTLERY", "PANTRY", "BOTTLE"]):
        return "Kitchen & Dining"
    elif any(k in d for k in ["CARD", "PAPER", "PEN", "PENCIL", "NOTEBOOK", "STICKER", "RIBBON", "WRAP"]):
        return "Stationery & Crafts"
    elif any(k in d for k in ["CHRISTMAS", "XMAS", "PARTY", "BALLOON", "BUNTING", "ORNAMENT", "GARLAND"]):
        return "Seasonal & Party"
    else:
        return "Other / General Giftware"

# --- 3. SYNTHETIC SAMPLE DATA GENERATOR ---
@st.cache_data
def generate_sample_dataset(n_rows=15000):
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    
    products = [
        ("WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
        ("JUMBO BAG RED RETROSPOT", 1.95),
        ("REGENCY CAKESTAND 3 TIER", 12.75),
        ("ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("PACK OF 72 RETROSPOT CAKE CASES", 0.55),
        ("NATURAL SLATE HEART CHALKBOARD", 2.95),
        ("HEART OF WICKER SMALL", 1.45),
        ("VINTAGE SNAP CARDS", 0.85),
        ("LUNCH BAG SPACEBOY DESIGN", 1.95),
        ("PAPER CHAIN KIT 50'S CHRISTMAS", 2.55)
    ]
    
    dates = [start_date + timedelta(days=int(np.random.uniform(0, 365))) for _ in range(n_rows)]
    cust_ids = [f"CUST-{np.random.randint(1001, 1500)}" for _ in range(n_rows)]
    chosen_idx = np.random.choice(len(products), size=n_rows)
    
    descs = [products[i][0] for i in chosen_idx]
    prices = [products[i][1] for i in chosen_idx]
    quantities = np.random.geometric(p=0.2, size=n_rows)
    
    df = pd.DataFrame({
        "Invoice": [f"INV-{100000 + i}" for i in range(n_rows)],
        "CustomerID": cust_ids,
        "Description": descs,
        "Quantity": quantities,
        "UnitPrice": prices,
        "InvoiceDate": dates
    })
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.date
    df["ProductCategory"] = df["Description"].apply(assign_product_category)
    return df

# --- 4. CSV INGESTION & CLEANING ---
@st.cache_data
def load_and_clean_data(filepath="online_retail_II.csv"):
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df.columns = [c.strip() for c in df.columns]
    
    col_map = {
        "Customer ID": "CustomerID",
        "InvoiceNo": "Invoice",
        "Price": "UnitPrice"
    }
    df = df.rename(columns=col_map)
    
    # Drop records with missing CustomerID or cancelled orders
    df = df.dropna(subset=["CustomerID", "Description"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    
    # Positive purchases only
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    
    # Compute metrics & classifications
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = df["InvoiceDate"].dt.date
    df["ProductCategory"] = df["Description"].apply(assign_product_category)
    
    return df

# --- 5. RFM COMPUTATION ---
@st.cache_data
def compute_rfm(df_subset, n_clusters=4):
    ref_date = pd.to_datetime(df_subset["InvoiceDate"]).max() + timedelta(days=1)
    
    rfm = df_subset.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - pd.to_datetime(x.max())).days,
        "Invoice": "nunique",
        "Revenue": "sum"
    }).rename(columns={"InvoiceDate": "Recency", "Invoice": "Frequency", "Revenue": "Monetary"})
    
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaled = StandardScaler().fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    rfm["Cluster"] = kmeans.fit_predict(scaled).astype(str)
    return rfm

# --- 6. R ARIMA RUNNER ---
def run_r_arima(daily_df, rscript_bin, horizon=30):
    daily_df.to_csv("temp_daily_sales.csv", index=False)
    try:
        cmd = [rscript_bin, "forecast.R", "temp_daily_sales.csv", str(horizon)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            st.error(f"R error: {stderr}")
            return None
        return pd.read_csv(io.StringIO(stdout))
    except Exception as e:
        st.error(f"Failed to execute Rscript: {e}")
        return None

# --- SIDEBAR: SOURCE & FILTERS ---
st.sidebar.title("Filters & Config")

# Choose Data Source
source = st.sidebar.radio(
    "Data Source:",
    ["Use 'online_retail_II.csv' (Local)", "Built-in Sample (Fast)"]
)

if source == "Use 'online_retail_II.csv' (Local)":
    raw_df = load_and_clean_data("online_retail_II.csv")
    if raw_df is None:
        st.warning("`online_retail_II.csv` was not found. Using built-in sample.")
        raw_df = generate_sample_dataset()
else:
    raw_df = generate_sample_dataset()

# R Binary Path
detected_r = find_rscript_path()
rscript_path = st.sidebar.text_input("Rscript Executable Path", value=detected_r)

# Filter 1: Product Types / Categories
category_options = sorted(raw_df["ProductCategory"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select Product Types:",
    options=category_options,
    default=category_options
)

# Filter 2: Optional Keyword Search
search_term = st.sidebar.text_input("Search Product Description:", "")

# Filter 3: Date Range
min_d = raw_df["Date"].min()
max_d = raw_df["Date"].max()
date_range = st.sidebar.date_input("Date Range:", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Apply Filters
filtered_df = raw_df[raw_df["ProductCategory"].isin(selected_categories)]

if search_term.strip():
    filtered_df = filtered_df[filtered_df["Description"].str.contains(search_term.strip(), case=False, na=False)]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[(filtered_df["Date"] >= start_d) & (filtered_df["Date"] <= end_d)]

# Sidebar Export Button
st.sidebar.divider()
csv_export = filtered_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download Filtered CSV",
    data=csv_export,
    file_name="filtered_retail_data.csv",
    mime="text/csv"
)

# --- MAIN DASHBOARD ---
st.title("Giftware & Retail Sales Analytics")
st.caption("Categorized by Product Type | Python (RFM Segmentation) + R (ARIMA Forecasting)")

# KPI Summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{filtered_df['Revenue'].sum():,.2f}")
col2.metric("Total Orders", f"{filtered_df['Invoice'].nunique():,}")
col3.metric("Items Sold", f"{int(filtered_df['Quantity'].sum()):,}")
avg_order_val = filtered_df.groupby("Invoice")["Revenue"].sum().mean() if len(filtered_df) > 0 else 0
col4.metric("Avg Order Value", f"£{avg_order_val:,.2f}")

st.divider()

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["Product Performance", "Customer RFM Clusters (Python)", "Time Series Forecast (R)"])

# TAB 1: PRODUCT PERFORMANCE BY TYPE
with tab1:
    st.subheader("Revenue Breakdown by Product Type")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Category Donut Chart
        cat_rev = filtered_df.groupby("ProductCategory")["Revenue"].sum().reset_index()
        fig_pie = px.pie(
            cat_rev,
            names="ProductCategory",
            values="Revenue",
            hole=0.4,
            title="Revenue Distribution by Category",
            template="plotly_white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        # Top 10 Best Sellers within Selected Types
        top_items = (
            filtered_df.groupby(["Description", "ProductCategory"])["Revenue"]
            .sum()
            .reset_index()
            .sort_values("Revenue", ascending=False)
            .head(10)
        )
        fig_bar = px.bar(
            top_items,
            x="Revenue",
            y="Description",
            color="ProductCategory",
            orientation="h",
            title="Top 10 Selling Products",
            template="plotly_white",
            labels={"Revenue": "Sales (£)", "Description": "Product Name"}
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

# TAB 2: RFM CLUSTERING
with tab2:
    st.subheader("Customer Value Segmentation (RFM)")
    k_clusters = st.slider("Select Number of Customer Clusters ($K$)", 2, 5, 4)
    
    if len(filtered_df["CustomerID"].unique()) >= k_clusters:
        rfm_df = compute_rfm(filtered_df, n_clusters=k_clusters)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_cluster = px.scatter(
                rfm_df.reset_index(),
                x="Recency",
                y="Monetary",
                size="Frequency",
                color="Cluster",
                log_y=True,
                title="Customer Distribution (Recency vs. Spend)",
                labels={"Recency": "Days Since Last Order", "Monetary": "Total Spend (£)"},
                template="plotly_white"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        with c2:
            st.write("Cluster Profiles (Averages):")
            summary = rfm_df.groupby("Cluster").agg({
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean"
            }).round(1)
            st.dataframe(summary, use_container_width=True)
    else:
        st.warning("Not enough distinct customers in the current filtered selection to create clusters.")

# TAB 3: TIME SERIES FORECAST (R ENGINE)
with tab3:
    st.subheader("Daily Sales Forecast with R Auto-ARIMA")
    horizon = st.slider("Prediction Horizon (Days Ahead)", 7, 60, 30)
    
    daily_sales = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
    daily_sales = daily_sales.sort_values("Date")
    
    if st.button("Generate R Forecast"):
        if len(daily_sales) < 14:
            st.error("At least 14 days of sales history are required to train the ARIMA model. Broaden your category or date selections.")
        else:
            with st.spinner("Invoking R ARIMA engine..."):
                forecast_df = run_r_arima(daily_sales, rscript_bin=rscript_path, horizon=horizon)
                
                if forecast_df is not None:
                    last_date = daily_sales["Date"].max()
                    forecast_df["Date"] = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
                    
                    fig = go.Figure()
                    
                    # Show recent history for visual readability
                    recent = daily_sales.tail(90)
                    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Revenue"], mode="lines", name="Actual Daily Revenue"))
                    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines+markers", line=dict(dash="dash", color="red"), name="R ARIMA Forecast"))
                    
                    # Confidence Band
                    fig.add_trace(go.Scatter(
                        x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
                        y=forecast_df["Hi95"].tolist() + forecast_df["Lo95"].tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(255,0,0,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% Confidence Interval"
                    ))
                    
                    fig.update_layout(title="Daily Demand Trajectory", xaxis_title="Date", yaxis_title="Revenue (£)", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)