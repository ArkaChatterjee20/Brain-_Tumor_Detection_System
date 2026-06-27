import streamlit as st
import requests
from PIL import Image
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------
if "token" not in st.session_state:
    st.session_state.token = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "gradcam" not in st.session_state:
    st.session_state.gradcam = None

if "report_path" not in st.session_state:
    st.session_state.report_path = None

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Brain Tumor Detection System")

# -----------------------------
# Login
# -----------------------------
if st.session_state.token is None:

    st.subheader("🔐 User Login")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        response = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={
                "email": email,
                "password": password
            }
        )

        if response.status_code == 200:

            token = response.json()["access_token"]

            st.session_state.token = token

            st.success("Login Successful")

            st.code(token)

            st.rerun()

        else:

            try:
                st.error(response.json()["detail"])
            except:
                st.error(response.text)

    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

st.sidebar.success("Logged In")

st.sidebar.write("Backend")

st.sidebar.code(BACKEND_URL)

if st.sidebar.button("Logout"):

    st.session_state.token = None
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.gradcam = None
    st.session_state.report_path = None

    st.rerun()

st.sidebar.divider()

st.sidebar.markdown("### Features")

st.sidebar.markdown("✅ CNN Prediction")
st.sidebar.markdown("✅ Grad-CAM")
st.sidebar.markdown("✅ PDF Report")
st.sidebar.markdown("✅ Prediction History")
st.sidebar.markdown("✅ Dashboard")
# ==========================================================
# Upload MRI Image
# ==========================================================

st.header("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded MRI",
            use_container_width=True
        )

    with col2:

        st.info(
            """
            Supported Formats

            • JPG

            • JPEG

            • PNG
            """
        )

        if st.button(
            "🧠 Predict",
            use_container_width=True
        ):

            headers = {
                "Authorization":
                f"Bearer {st.session_state.token}"
            }

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            # -----------------------------
            # DEBUG
            # -----------------------------

            with st.expander("JWT Debug"):

                st.write("Current Token")

                st.code(
                    st.session_state.token
                )

                st.write("Headers")

                st.write(headers)

            with st.spinner(
                "Running CNN Model..."
            ):

                response = requests.post(

                    f"{BACKEND_URL}/predict",

                    files=files,

                    headers=headers

                )

            # -----------------------------
            # DEBUG RESPONSE
            # -----------------------------

            with st.expander("Backend Response"):

                st.write(
                    "Status Code:",
                    response.status_code
                )

                try:

                    st.json(
                        response.json()
                    )

                except:

                    st.write(
                        response.text
                    )

            # -----------------------------
            # Prediction Success
            # -----------------------------

            if response.status_code == 200:

                result = response.json()

                st.session_state.prediction = result.get(
                    "prediction"
                )

                st.session_state.confidence = result.get(
                    "confidence"
                )

                st.session_state.gradcam = result.get(
                    "gradcam_image"
                )

                st.session_state.report_path = result.get(
                    "report_path"
                )

                st.success(
                    "Prediction Completed Successfully"
                )

            else:

                try:

                    st.error(
                        response.json()["detail"]
                    )

                except:

                    st.error(
                        response.text
                    )


# ==========================================================
# Prediction Result
# ==========================================================

if st.session_state.prediction is not None:

    st.divider()

    st.header("🧠 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Prediction",

            st.session_state.prediction

        )

        st.metric(

            "Confidence",

            f"{st.session_state.confidence}%"

        )

        st.progress(

            min(
                float(
                    st.session_state.confidence
                ) / 100,
                1.0
            )

        )

    with col2:

        try:

            gradcam = Image.open(

                st.session_state.gradcam

            )

            st.image(

                gradcam,

                caption="Grad-CAM",

                use_container_width=True

            )

        except:

            st.warning(

                "Grad-CAM image unavailable."

            )


# ==========================================================
# PDF Report
# ==========================================================

if st.session_state.report_path:

    st.divider()

    st.header("📄 AI Report")

    try:

        with open(

            st.session_state.report_path,

            "rb"

        ) as pdf:

            st.download_button(

                "📥 Download PDF Report",

                pdf,

                file_name="Brain_Tumor_Report.pdf",

                mime="application/pdf",

                use_container_width=True

            )

    except Exception as e:

        st.warning(

            f"Unable to load report\n\n{e}"

        )
# ==========================================================
# Prediction History
# ==========================================================

st.divider()

st.header("📜 Prediction History")

if st.button(
    "🔄 Load Prediction History",
    use_container_width=True
):

    headers = {
        "Authorization":
        f"Bearer {st.session_state.token}"
    }

    with st.spinner("Loading History..."):

        response = requests.get(
            f"{BACKEND_URL}/history",
            headers=headers
        )

    if response.status_code == 200:

        history = response.json()

        if len(history) == 0:

            st.info(
                "No Prediction History Available."
            )

        else:

            history_df = pd.DataFrame(history)

            st.dataframe(
                history_df,
                use_container_width=True
            )

            csv = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "⬇ Download History CSV",

                csv,

                "prediction_history.csv",

                "text/csv",

                use_container_width=True

            )

            st.divider()

            for item in history:

                with st.container():

                    st.subheader(item["filename"])

                    col1, col2 = st.columns([2,1])

                    with col1:

                        st.write(
                            "Prediction :",
                            item["predicted_class"]
                        )

                        st.write(
                            "Confidence :",
                            f"{item['confidence']} %"
                        )

                        st.write(
                            "Prediction Time :",
                            item["prediction_time"]
                        )

                    with col2:

                        try:

                            gradcam = Image.open(
                                item["gradcam_path"]
                            )

                            st.image(
                                gradcam,
                                caption="Grad-CAM",
                                use_container_width=True
                            )

                        except:

                            st.warning(
                                "Grad-CAM unavailable."
                            )

                st.divider()

    else:

        st.error(
            "Unable to load history."
        )


# ==========================================================
# Dashboard
# ==========================================================

st.divider()

st.header("📊 Dashboard")

if st.button(
    "📈 Load Dashboard",
    use_container_width=True
):

    headers = {
        "Authorization":
        f"Bearer {st.session_state.token}"
    }

    response = requests.get(
        f"{BACKEND_URL}/dashboard",
        headers=headers
    )

    if response.status_code == 200:

        data = response.json()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Predictions",
                data["total_predictions"]
            )

            st.metric(
                "Tumor Cases",
                data["tumor_predictions"]
            )

        with col2:

            st.metric(
                "No Tumor Cases",
                data["no_tumor_predictions"]
            )

            st.metric(
                "Average Confidence",
                f"{data['average_confidence']} %"
            )

        chart = pd.DataFrame({

            "Category":[
                "Tumor",
                "No Tumor"
            ],

            "Count":[
                data["tumor_predictions"],
                data["no_tumor_predictions"]
            ]

        })

        st.bar_chart(
            chart.set_index("Category")
        )

    else:

        st.error(
            "Unable to load dashboard."
        )


# ==========================================================
# About Project
# ==========================================================

st.divider()

st.header("ℹ About Project")

st.success(
    """
CNN-based Brain Tumor Detection System

✅ FastAPI REST API

✅ JWT Authentication

✅ MySQL Database

✅ CNN Classification

✅ Grad-CAM Explainability

✅ PDF Report Generation

✅ Prediction History

✅ Dashboard

✅ Admin Dashboard

✅ Streamlit Frontend
"""
)

st.divider()

st.caption(
    "Brain Tumor Detection System | Final Year Project"
)