import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# ----------------------------------------------------
# Load environment variables
# ----------------------------------------------------

load_dotenv()

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://brain-tumor-detection-system-bzzb.onrender.com"
).rstrip("/")


# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Admin Dashboard")


# ----------------------------------------------------
# Check Login
# ----------------------------------------------------

token = st.session_state.get("token")

if not token:
    st.warning("Please login first.")
    st.stop()


headers = {
    "Authorization": f"Bearer {token}"
}


# ----------------------------------------------------
# Helper function for backend requests
# ----------------------------------------------------

def backend_get(endpoint, timeout=(10, 120)):
    try:
        response = requests.get(
            f"{BACKEND_URL}{endpoint}",
            headers=headers,
            timeout=timeout
        )

        return response

    except requests.exceptions.Timeout:
        st.error(
            "The backend is taking too long to respond. "
            "Please wait a few seconds and try again."
        )
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the backend server. "
            "Please make sure the backend is running."
        )
        return None

    except requests.exceptions.RequestException as e:
        st.error(
            f"Backend request failed: {e}"
        )
        return None


# ----------------------------------------------------
# Load Dashboard Statistics
# ----------------------------------------------------

with st.spinner("Loading dashboard..."):

    dashboard_response = backend_get(
        "/dashboard",
        timeout=(10, 120)
    )


if dashboard_response is None:
    st.stop()


# ----------------------------------------------------
# Authentication / Authorization errors
# ----------------------------------------------------

if dashboard_response.status_code == 401:

    st.error(
        "Your login session has expired. "
        "Please login again."
    )

    # Remove invalid token
    st.session_state.pop("token", None)

    st.stop()


if dashboard_response.status_code == 403:

    st.error(
        "You do not have permission to access the admin dashboard."
    )

    st.stop()


if dashboard_response.status_code != 200:

    st.error(
        f"Unable to load dashboard. "
        f"Backend returned status {dashboard_response.status_code}."
    )

    st.stop()


# ----------------------------------------------------
# Parse dashboard
# ----------------------------------------------------

try:

    dashboard = dashboard_response.json()

except ValueError:

    st.error(
        "The backend returned an invalid dashboard response."
    )

    st.stop()


# ----------------------------------------------------
# Load Prediction History
# ----------------------------------------------------

with st.spinner("Loading prediction history..."):

    history_response = backend_get(
        "/history",
        timeout=(10, 120)
    )


if history_response is None:
    st.stop()


# ----------------------------------------------------
# History Authentication errors
# ----------------------------------------------------

if history_response.status_code == 401:

    st.error(
        "Your login session has expired. "
        "Please login again."
    )

    st.session_state.pop("token", None)

    st.stop()


if history_response.status_code == 403:

    st.error(
        "You do not have permission to access prediction history."
    )

    st.stop()


if history_response.status_code != 200:

    st.error(
        f"Unable to load prediction history. "
        f"Backend returned status {history_response.status_code}."
    )

    st.stop()


# ----------------------------------------------------
# Parse history
# ----------------------------------------------------

try:

    history = history_response.json()

except ValueError:

    st.error(
        "The backend returned an invalid history response."
    )

    st.stop()


# ----------------------------------------------------
# Convert to DataFrame
# ----------------------------------------------------

df = pd.DataFrame(history)


# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

st.subheader("Overall Statistics")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Predictions",
        dashboard.get("total_predictions", 0)
    )


with col2:

    st.metric(
        "Total Users",
        dashboard.get("total_users", 0)
    )


with col3:

    st.metric(
        "Average Confidence (%)",
        dashboard.get("average_confidence", 0)
    )


# ----------------------------------------------------
# Tumor Distribution
# ----------------------------------------------------

st.divider()

st.subheader("Tumor Class Distribution")


if not df.empty and "predicted_class" in df.columns:

    class_counts = df["predicted_class"].value_counts()

    st.bar_chart(class_counts)

else:

    st.info(
        "No prediction data available."
    )


# ----------------------------------------------------
# Recent Predictions
# ----------------------------------------------------

st.divider()

st.subheader("Recent Predictions")


if not df.empty:

    required_columns = [
        "filename",
        "predicted_class",
        "confidence",
        "prediction_time"
    ]

    available_columns = [
        column
        for column in required_columns
        if column in df.columns
    ]

    display_df = df[available_columns].copy()

    st.dataframe(
        display_df,
        use_container_width=True
    )

else:

    display_df = pd.DataFrame(
        columns=[
            "filename",
            "predicted_class",
            "confidence",
            "prediction_time"
        ]
    )

    st.info(
        "No prediction history available."
    )


# ----------------------------------------------------
# Download CSV
# ----------------------------------------------------

st.divider()

st.subheader("Download Prediction Data")


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


# ----------------------------------------------------
# Refresh Button
# ----------------------------------------------------

st.divider()

if st.button(
    "🔄 Refresh Dashboard",
    key="admin_refresh"
):

    st.rerun()