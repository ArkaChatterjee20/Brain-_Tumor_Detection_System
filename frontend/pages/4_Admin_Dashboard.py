import streamlit as st
import requests
import pandas as pd

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
response = requests.get(
    "http://127.0.0.1:8000/history",
    headers=headers
)

if response.status_code != 200:
    st.error("Unable to load data.")
    st.stop()

history = response.json()

if len(history) == 0:
    st.info("No prediction records found.")
    st.stop()

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
        len(df)
    )

with col2:
    st.metric(
        "Total Patients",
        df["filename"].nunique()
    )

with col3:
    avg_conf = round(df["confidence"].mean(), 2)
    st.metric(
        "Average Confidence (%)",
        avg_conf
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

csv = display_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Prediction Data",
    data=csv,
    file_name="prediction_history.csv",
    mime="text/csv"
)

# ----------------------
# Refresh Button
# ----------------------
if st.button("Refresh"):
    st.rerun()