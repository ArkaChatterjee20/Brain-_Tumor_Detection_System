import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://brain-tumor-detection-system-bzzb.onrender.com"
).rstrip("/")


# --------------------------------------------------
# Page
# --------------------------------------------------

st.title("📊 Dashboard Analytics")


# --------------------------------------------------
# Authentication
# --------------------------------------------------

token = st.session_state.get("token")

if not token:

    st.warning(
        "Please login first."
    )

    st.stop()


headers = {
    "Authorization": f"Bearer {token}"
}


# --------------------------------------------------
# Dashboard request
# --------------------------------------------------

try:

    response = requests.get(
        f"{BACKEND_URL}/dashboard",
        headers=headers,
        timeout=(10, 120)
    )

except requests.exceptions.Timeout:

    st.error(
        "Dashboard request timed out. "
        "Please try again."
    )

    st.stop()

except requests.exceptions.ConnectionError:

    st.error(
        "Unable to connect to the backend."
    )

    st.stop()

except requests.exceptions.RequestException as e:

    st.error(
        f"Dashboard request failed: {e}"
    )

    st.stop()


# --------------------------------------------------
# Authentication error
# --------------------------------------------------

if response.status_code == 401:

    st.error(
        "Your login session has expired. "
        "Please login again."
    )

    st.stop()


# --------------------------------------------------
# Other errors
# --------------------------------------------------

if response.status_code != 200:

    st.error(
        f"Unable to load dashboard. "
        f"Backend status: {response.status_code}"
    )

    st.stop()


# --------------------------------------------------
# Data
# --------------------------------------------------

data = response.json()


# --------------------------------------------------
# Top metrics
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Total Users",
        data["total_users"]
    )


with col2:

    st.metric(
        "Total Predictions",
        data["total_predictions"]
    )


st.divider()


# --------------------------------------------------
# Tumor statistics
# --------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Glioma",
        data["glioma"]
    )

    st.metric(
        "Pituitary",
        data["pituitary"]
    )


with col2:

    st.metric(
        "Meningioma",
        data["meningioma"]
    )

    st.metric(
        "No Tumor",
        data["notumor"]
    )


st.divider()


# --------------------------------------------------
# Chart
# --------------------------------------------------

chart_data = pd.DataFrame({

    "Tumor Type": [
        "Glioma",
        "Meningioma",
        "Pituitary",
        "No Tumor"
    ],

    "Count": [
        data["glioma"],
        data["meningioma"],
        data["pituitary"],
        data["notumor"]
    ]

})


st.subheader(
    "Prediction Distribution"
)


st.bar_chart(
    chart_data.set_index(
        "Tumor Type"
    )
)