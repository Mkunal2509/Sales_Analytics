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
import shutil
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retail Sales Analytics & Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENSURE R PACKAGES & RSCRIPT PATH ---
def find_rscript_path():
    # 1. System PATH (Default on Linux / Streamlit Community Cloud)
    system_r = shutil.which("Rscript")
    if system_r:
        return system_r

    # 2. Check standard Windows installations
    windows_candidates = glob.glob(r"C:\Program Files\R\R-*\bin\x64\Rscript.exe") + \
                         glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
    if windows_candidates:
        return windows_candidates[-1]

    # 3. Standard Unix fallback locations
    for p in ["/usr/bin/Rscript", "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript"]:
        if os.path.exists(p):
            return p

    return "Rscript"

# --- PRODUCT CLASSIFICATION ENGINE ---
def classify_product(desc):
    if not isinstance(desc, str):
        return "Other / General Giftware"
    d = desc.upper()
    if any(k in d for k in ["HEART", "LIGHT", "CANDLE", "HOLDER", "LANTERN", "FRAME", "CLOCK", "MIRROR", "CUSHION"]):
        return "Home & Décor"
    elif any(k in d for k in ["MUG", "CUP", "PLATE", "BOWL", "TEASPOON", "BOTTLE", "GLASS", "KITCHEN", "BAKING", "CUTLERY", "COFFEE"]):
        return "Kitchen & Dining"
    elif any(k in d for k in ["BAG", "BOX", "TIN", "STORAGE", "POCKET", "CASE", "PURSE", "BASKET", "LUNCH"]):
        return "Bags & Storage"
    elif any(k in d for k in ["CARD", "NOTEBOOK", "PENCIL", "PEN", "STICKER", "PAPER", "WRAP", "TAPE", "RIBBON"]):
        return "Stationery & Crafts"
    elif any(k in d for k in ["CHRISTMAS", "XMAS", "TREE", "EASTER", "PARTY", "BALLOON", "BUNTING", "DECORATION"]):
        return "Seasonal & Party"
    else:
        return "Other / General Giftware"

# --- DATA INGESTION & CACHED PREPROCESSING ---
@st.cache_data
def load_and_clean_data():
    csv_file = "online_retail_II.csv"
    parquet_file = "online_retail_II.parquet"

    dtypes = {
        'Invoice': 'str',
        'StockCode': 'str',
        'Description': 'str',
        'Quantity': 'int32',
        'Price': 'float32',
        'Customer ID': 'float32',
        'Country': 'category'
    }

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype=dtypes, encoding="ISO-8859-1")
    else:
        st.error(f"Dataset file '{csv_file}' or '{parquet_file}' not found in the root repository folder.")
        st.stop()

    # Drop missing customer IDs & clean cancellations
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['Price']
    df['Category'] = df['Description'].apply(classify_product)
    df['Customer ID'] = df['Customer ID'].astype(int).astype(str)

    return df

# Load cleaned transactional records
with st.spinner("Loading and preprocessing transaction data..."):
    df_raw = load_and_clean_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🔍 Analytics Controls")

# Product Category Filter
all_categories = sorted(df_raw['Category'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    options=all_categories,
    default=all_categories
)

# Text Search Filter
search_query = st.sidebar.text_input("Search Item Description", value="").strip().upper()

# Date Range Filter
min_date = df_raw['InvoiceDate'].min().date()
max_date = df_raw['InvoiceDate'].max().date()

selected_dates = st.sidebar.date_input(
    "Select Transaction Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply dynamic filters
df_filtered = df_raw.copy()
if selected_categories:
    df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]

if search_query:
    df_filtered = df_filtered[df_filtered['Description'].str.upper().str.contains(search_query, na=False)]

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    df_filtered = df_filtered[
        (df_filtered['InvoiceDate'].dt.date >= start_date) & 
        (df_filtered['InvoiceDate'].dt.date <= end_date)
    ]

# --- MAIN DASHBOARD LAYOUT ---
st.title("📊 Retail Sales Analytics & Demand Forecasting Dashboard")
st.markdown("Polyglot Big Data Analytics Architecture combining **Python** (ETL, Unsupervised ML, Streamlit) and **R** (Auto-ARIMA Time Series Forecasting).")

# High-Level KPIs
total_revenue = float(df_filtered['Revenue'].sum())
total_orders = int(df_filtered['Invoice'].nunique())
total_units = int(df_filtered['Quantity'].sum())
unique_customers = int(df_filtered['Customer ID'].nunique())
aov = total_revenue / total_orders if total_orders > 0 else 0.0

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
kpi_col1.metric("Total Revenue", f"£{total_revenue:,.2f}")
kpi_col2.metric("Total Orders", f"{total_orders:,}")
kpi_col3.metric("Units Sold", f"{total_units:,}")
kpi_col4.metric("Active Customers", f"{unique_customers:,}")
kpi_col5.metric("Avg Order Value", f"£{aov:.2f}")

st.markdown("---")

# Dashboard Tabs
tab_sales, tab_rfm, tab_forecast = st.tabs([
    "📦 Product & Category Analytics",
    "👥 Customer RFM Segmentation",
    "🔮 Demand Forecasting (R Engine)"
])

# ==============================================================================
# TAB 1: PRODUCT & CATEGORY ANALYTICS
# ==============================================================================
with tab_sales:
    st.subheader("Product Performance & Category Distribution")
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        cat_rev = df_filtered.groupby('Category', observed=True)['Revenue'].sum().reset_index()
        fig_cat = px.pie(
            cat_rev,
            names='Category',
            values='Revenue',
            title="Revenue Distribution by Category",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_cat.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_chart2:
        top_items = df_filtered.groupby('Description', observed=True)['Revenue'].sum().reset_index()
        top_items = top_items.sort_values(by='Revenue', ascending=False).head(10)
        fig_top = px.bar(
            top_items,
            x='Revenue',
            y='Description',
            orientation='h',
            title="Top 10 Products by Total Spend",
            color='Revenue',
            color_continuous_scale="Viridis"
        )
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Filtered Transaction Table")
    st.dataframe(
        df_filtered[['Invoice', 'StockCode', 'Description', 'Quantity', 'Price', 'Revenue', 'Customer ID', 'InvoiceDate', 'Category']].head(100),
        use_container_width=True
    )
    
    csv_buffer = df_filtered.head(5000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Transactions (CSV)",
        data=csv_buffer,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )

# ==============================================================================
# TAB 2: CUSTOMER RFM SEGMENTATION
# ==============================================================================
with tab_rfm:
    st.subheader("RFM (Recency, Frequency, Monetary) Customer Clustering")
    st.markdown("Groups customers into behavioral cohorts using log-transformed features and **$K$-Means Clustering**.")

    if df_filtered.empty or df_filtered['Customer ID'].nunique() < 5:
        st.warning("Insufficient data available in the current filter range to compute customer clusters.")
    else:
        ref_date = df_filtered['InvoiceDate'].max() + pd.Timedelta(days=1)

        rfm = df_filtered.groupby('Customer ID').agg({
            'InvoiceDate': lambda d: (ref_date - d.max()).days,
            'Invoice': 'nunique',
            'Revenue': 'sum'
        }).reset_index()

        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
        rfm = rfm[rfm['Monetary'] > 0]

        # Log transform to manage extreme skewness
        rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        # Train K-Means (k=3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        # Name clusters by average spend and recency
        cluster_profile = rfm.groupby('Cluster').agg({'Recency': 'mean', 'Monetary': 'mean'}).sort_values('Monetary')
        cluster_map = {
            cluster_profile.index[0]: "Low-Value / Occasional",
            cluster_profile.index[1]: "Moderate / Steady",
            cluster_profile.index[2]: "Champions / High Spenders"
        }
        rfm['Segment'] = rfm['Cluster'].map(cluster_map)

        rfm_col1, rfm_col2 = st.columns([2, 1])

        with rfm_col1:
            fig_rfm = px.scatter(
                rfm,
                x='Recency',
                y='Monetary',
                size='Frequency',
                color='Segment',
                hover_data=['Customer ID', 'Frequency'],
                log_y=True,
                title="Customer Segments: Recency vs. Spend (Log Scale)",
                color_discrete_map={
                    "Low-Value / Occasional": "#E74C3C",
                    "Moderate / Steady": "#F39C12",
                    "Champions / High Spenders": "#2ECC71"
                }
            )
            st.plotly_chart(fig_rfm, use_container_width=True)

        with rfm_col2:
            st.markdown("#### Segment Breakdown")
            seg_summary = rfm.groupby('Segment').agg(
                Customers=('Customer ID', 'count'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Frequency=('Frequency', 'mean'),
                Avg_Monetary=('Monetary', 'mean')
            ).reset_index()
            st.dataframe(seg_summary.style.format({
                'Avg_Recency': '{:.1f} days',
                'Avg_Frequency': '{:.1f}',
                'Avg_Monetary': '£{:,.2f}'
            }), use_container_width=True)

# ==============================================================================
# TAB 3: DEMAND FORECASTING VIA HEADLESS R ENGINE
# ==============================================================================
with tab_forecast:
    st.subheader("Time Series Sales Forecasting with R (`auto.arima`)")
    st.markdown("Daily revenues are aggregated and passed to **`forecast.R`** via an IPC subprocess to calculate predictions with 80% and 95% confidence intervals.")

    daily_sales = df_filtered.groupby(df_filtered['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    daily_sales.columns = ['Date', 'Revenue']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date')

    # Re-index to ensure a continuous daily sequence (fill missing dates with 0)
    if not daily_sales.empty:
        idx = pd.date_range(daily_sales['Date'].min(), daily_sales['Date'].max())
        daily_sales = daily_sales.set_index('Date').reindex(idx, fill_value=0.0).reset_index()
        daily_sales.columns = ['Date', 'Revenue']

    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        forecast_horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=7)
        rscript_exec = find_rscript_path()
        run_forecast_btn = st.button("🚀 Run R Auto-ARIMA Forecast", type="primary")

    with f_col2:
        if run_forecast_btn:
            if len(daily_sales) < 14:
                st.error("Insufficient historical days in the filtered range to fit an ARIMA model (requires at least 14 days).")
            else:
                with st.spinner("Invoking headless R engine (fitting Auto-ARIMA with weekly seasonality)..."):
                    # Save temporary daily sales sequence
                    temp_csv = "temp_daily_sales.csv"
                    daily_sales.to_csv(temp_csv, index=False)

                    try:
                        cmd = [rscript_exec, "forecast.R", temp_csv, str(forecast_horizon)]
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                        # Clean output to isolate CSV data
                        raw_stdout = result.stdout.strip()
                        lines = [l for l in raw_stdout.splitlines() if "," in l and not l.startswith("Registered") and not l.startswith("Attaching")]
                        csv_payload = "\n".join(lines)

                        forecast_df = pd.read_csv(io.StringIO(csv_payload))

                        # Build continuous forward dates
                        last_date = daily_sales['Date'].max()
                        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
                        forecast_df['Date'] = future_dates

                        # Construct Forecast Plot with Plotly
                        fig_fc = go.Figure()

                        # Historical actual sales
                        fig_fc.add_trace(go.Scatter(
                            x=daily_sales['Date'],
                            y=daily_sales['Revenue'],
                            mode='lines',
                            name='Historical Daily Sales',
                            line=dict(color='#2980B9', width=2)
                        ))

                        # 95% Confidence Band (Shaded)
                        fig_fc.add_trace(go.Scatter(
                            x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                            y=list(forecast_df['Hi95']) + list(forecast_df['Lo95'][::-1]),
                            fill='toself',
                            fillcolor='rgba(46, 204, 113, 0.15)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name='95% Confidence Interval'
                        ))

                        # 80% Confidence Band (Shaded)
                        fig_fc.add_trace(go.Scatter(
                            x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                            y=list(forecast_df['Hi80']) + list(forecast_df['Lo80'][::-1]),
                            fill='toself',
                            fillcolor='rgba(46, 204, 113, 0.3)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name='80% Confidence Interval'
                        ))

                        # Projected Mean Line
                        fig_fc.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Forecast'],
                            mode='lines+markers',
                            name='Auto-ARIMA Projected Mean',
                            line=dict(color='#27AE60', width=2.5, dash='dash')
                        ))

                        fig_fc.update_layout(
                            title=f"Sales Forecast over next {forecast_horizon} Days (R forecast Engine)",
                            xaxis_title="Date",
                            yaxis_title="Revenue (£)",
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig_fc, use_container_width=True)

                        # Display Forecast Data Summary Table
                        st.markdown("#### Projected Values with Statistical Bounds")
                        st.dataframe(
                            forecast_df[['Date', 'Forecast', 'Lo80', 'Hi80', 'Lo95', 'Hi95']].style.format({
                                'Forecast': '£{:,.2f}',
                                'Lo80': '£{:,.2f}',
                                'Hi80': '£{:,.2f}',
                                'Lo95': '£{:,.2f}',
                                'Hi95': '£{:,.2f}'
                            }),
                            use_container_width=True
                        )

                    except subprocess.CalledProcessError as e:
                        st.error(f"R script execution failed:\n\n{e.stderr}")
                    except Exception as ex:
                        st.error(f"Error parsing forecast output: {ex}")
                    finally:
                        if os.path.exists(temp_csv):
                            os.remove(temp_csv)
        else:
            # Display historical trend chart by default
            fig_hist = px.line(
                daily_sales,
                x='Date',
                y='Revenue',
                title="Historical Daily Sales Revenue Trend",
                color_discrete_sequence=['#2980B9']
            )
            fig_hist.update_layout(hovermode="x unified")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.info("Click **'Run R Auto-ARIMA Forecast'** in the left panel to execute time-series modeling.")