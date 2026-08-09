import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv(
    "BACKEND_URL",
     "https://brain-tumor-detection-system-bzzb.onrender.com"
)

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Admin Dashboard")

# Check login
if "token" not in st.session_state:
    st.warning("Please login first.")
    st.stop()

headers = {
    "Authorization": f"Bearer {st.session_state['token']}"
}

# ----------------------
# Fetch history
# ----------------------
dashboard_response = requests.get(
    f"{BACKEND_URL}/dashboard",
    headers=headers,
    timeout = 180
    
)

if dashboard_response.status_code != 200:
    st.error("Unable to load dashboard.")
    st.stop()

dashboard = dashboard_response.json()

history_response = requests.get(
    f"{BACKEND_URL}/history",
    headers=headers,
    timeout = 180
    
)

if history_response.status_code != 200:
    st.error("Unable to load prediction history.")
    st.stop()

history = history_response.json()

# Convert to DataFrame
df = pd.DataFrame(history)

# ----------------------
# Statistics
# ----------------------
st.subheader("Overall Statistics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Total Predictions",
        dashboard["total_predictions"]
    )

with col2:
    st.metric(
        "Total Users",
        dashboard["total_users"]
    )

with col3:
    st.metric(
        "Average Confidence (%)",
        dashboard["average_confidence"]
    )

# ----------------------
# Tumor Distribution
# ----------------------
st.divider()

st.subheader("Tumor Class Distribution")

class_counts = df["predicted_class"].value_counts()

st.bar_chart(class_counts)

# ----------------------
# Recent Predictions
# ----------------------
st.divider()

st.subheader("Recent Predictions")

display_df = df[
    [
        "filename",
        "predicted_class",
        "confidence",
        "prediction_time"
    ]
]

st.dataframe(
    display_df,
    use_container_width=True
)

# ----------------------
# Download CSV
# ----------------------
st.divider()

csv_data = display_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇ Download Prediction Data",
    data=csv_data,
    file_name="prediction_history.csv",
    mime="text/csv",
    key="admin_csv_download"
)

# ----------------------
# Refresh Button
# ----------------------
if st.button("Refresh"):
    st.rerun()