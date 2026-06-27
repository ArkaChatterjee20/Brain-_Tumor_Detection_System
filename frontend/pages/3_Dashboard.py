import streamlit as st
import requests
import pandas as pd

st.title("📊 Dashboard Analytics")

if "token" not in st.session_state:

    st.warning("Please login first")

    st.stop()

headers = {
    "Authorization":
    f"Bearer {st.session_state['token']}"
}

response = requests.get(
    "http://127.0.0.1:8000/dashboard",
    headers=headers
)

if response.status_code == 200:

    data = response.json()

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

    st.subheader("Prediction Distribution")

    st.bar_chart(
        chart_data.set_index(
            "Tumor Type"
        )
    )

else:

    st.error(
        "Unable to load dashboard"
    )