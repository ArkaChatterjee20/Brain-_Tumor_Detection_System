import streamlit as st
import requests
from PIL import Image
import pandas as pd


import os

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
).rstrip("/")
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

if "gradcam_url" not in st.session_state:
    st.session_state.gradcam_url = None

if "report_url" not in st.session_state:
    st.session_state.report_url = None

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Brain Tumor Detection System")

# -----------------------------
# Login
# -----------------------------
if not st.session_state.get("token"):
    
    page = st.radio(
        "Choose",
        ["Login", "Register"],
        horizontal=True
    )

    if page == "Login":

        st.subheader("🔐 Login")

        email = st.text_input("Email")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            response = requests.post(
                f"{BACKEND_URL}/auth/login",
                data={
                    "username": email,
                    "password": password
                }
            )

            if response.status_code == 200:

                token = response.json()["access_token"]

                st.session_state.token = token

                st.success("Login Successful")

                st.rerun()

            else:

                try:
                    st.error(response.json()["detail"])
                except:
                    st.error(response.text)

    else:

        st.subheader("📝 Register")

        username = st.text_input("Username")

        email = st.text_input("Email ")

        password = st.text_input(
            "Password ",
            type="password"
        )

        if st.button("Register"):

            response = requests.post(
                f"{BACKEND_URL}/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                }
            )

            if response.status_code == 200:

                st.success(
                    "Registration Successful. Please Login."
                )

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
    st.session_state.gradcam_url = None
    st.session_state.report_url = None
    if "prediction_id" in st.session_state:
        del st.session_state["prediction_id"]

    st.rerun()

st.sidebar.divider()

st.sidebar.markdown("### Features")

st.sidebar.markdown("✅ CNN Prediction")
st.sidebar.markdown("✅ Grad-CAM")
st.sidebar.markdown("✅ PDF Report")
st.sidebar.markdown("✅ Prediction History")
st.sidebar.markdown("✅ Dashboard")
# Authentication Check
# ----------------------------------------------------

if not st.session_state.get("token"):
    st.warning("Please login first.")
    st.stop()
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

            # Get current token
            token = st.session_state.get("token")

            if not token:
                st.warning("Please login first.")
                st.stop()

            headers = {
                "Authorization": f"Bearer {token}"
            }

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:

                with st.spinner(
                    "Running CNN Model..."
                ):

                    response = requests.post(
                        f"{BACKEND_URL}/predict",
                        headers=headers,
                        files=files,
                        timeout=180
                    )

                # ------------------------------------------------
                # TOKEN EXPIRED / INVALID
                # ------------------------------------------------

                if response.status_code == 401:

                    st.session_state["token"] = None

                    st.warning(
                        "Your login session has expired. "
                        "Please login again."
                    )

                    st.rerun()


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    st.session_state["prediction"] = (
                        result["prediction"]
                    )

                    st.session_state["confidence"] = (
                        result["confidence"]
                    )

                    st.session_state["prediction_id"] = (
                        result["prediction_id"]
                    )

                    st.session_state["gradcam_url"] = (
                        result["gradcam_url"]
                    )

                    st.session_state["report_url"] = (
                        result["report_url"]
                    )

                    st.success(
                        "Prediction Completed Successfully"
                    )

                else:

                    try:

                        error_detail = response.json().get(
                            "detail",
                            "Prediction failed."
                        )

                    except Exception:

                        error_detail = response.text

                    st.error(
                        f"Prediction failed: {error_detail}"
                    )


            except requests.exceptions.Timeout:

                st.error(
                    "The backend took too long to respond. "
                    "Please try again."
                )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the backend. "
                    "Please check that the backend is running."
                )


            except requests.exceptions.RequestException as e:

                st.error(
                    f"Request failed: {e}"
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
                float(st.session_state.confidence) / 100,
                1.0
            )
        )

    with col2:

        try:

              headers = {
                      "Authorization": f"Bearer {st.session_state.token}"
                    }

              response = requests.get(
                          st.session_state.gradcam_url,
                          headers=headers,
                          timeout = (10,120)
                        )

              if response.status_code == 200:

                from io import BytesIO

                image = Image.open(BytesIO(response.content))

                st.image(
                  image,
                  caption="Grad-CAM",
                  use_container_width=True
                )

              else:

                st.warning("Grad-CAM unavailable.")

        except Exception as e:

                st.warning(f"Grad-CAM unavailable.\n\n{e}")

# ==========================================================
# PDF Report
# ==========================================================

if "prediction_id" in st.session_state:
    
    st.divider()

    st.header("📄 AI Report")
    

    headers = {
        "Authorization": f"Bearer {st.session_state.token}"
    }
    

    try:

        response = requests.get(
          st.session_state.report_url,
          headers=headers,
          timeout=(10, 120)
    )

    except requests.exceptions.RequestException as e:

        st.warning(
          f"Unable to download report: {e}"
    )

        response = None
    
    

    if response is not None and response.status_code == 200:

        st.download_button(
            "📥 Download PDF Report",
            response.content,
            file_name="Brain_Tumor_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    else:
        st.warning("Unable to download report.")

    
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

        try:

             response = requests.get(
              f"{BACKEND_URL}/history",
              headers=headers,
              timeout=(10, 120)
            )

        except requests.exceptions.Timeout:

         st.error(
        "Prediction history request timed out. "
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
        f"Prediction history request failed: {e}"
    )

         st.stop()
    if response.status_code == 401:
    
      st.session_state["token"] = None

      st.warning(
        "Your login session has expired. "
        "Please login again."
    )

      st.rerun()

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

            csv_data = history_df.to_csv(
             index=False
             ).encode("utf-8")

            st.download_button(
              label="⬇ Download Prediction History CSV",
              data=csv_data,
              file_name="prediction_history.csv",
              mime="text/csv",
              key="history_csv_download"
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

                                 headers = {
                                    "Authorization": f"Bearer {st.session_state.token}"
                                }

                                 response = requests.get(
                                     item["gradcam_url"],
                                     headers=headers,
                                     timeout = (10,120)
                                )

                                 if response.status_code == 200:

                                    from io import BytesIO

                                    image = Image.open(
                                       BytesIO(response.content)
                                    )

                                    st.image(
                                        image,
                                        
                                        use_container_width=True
                                    )

                                 else:

                                    st.warning(
                                      "Grad-CAM unavailable."
                                    )

                             except Exception as e:

                                st.warning(
                                  f"Unable to load Grad-CAM\n\n{e}"
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

    try:

         response = requests.get(
           f"{BACKEND_URL}/dashboard",
           headers=headers,
           timeout=(10, 120)
    )

    except requests.exceptions.Timeout:

      st.error(
        "Dashboard request timed out. Please try again."
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
    if response.status_code == 401:
    
     st.session_state["token"] = None

     st.warning(
        "Your login session has expired. "
        "Please login again."
    )

     st.rerun()

    if response.status_code == 200:

        data = response.json()

        # ---------------------------------------
        # Total Tumor Cases
        # ---------------------------------------

        tumor_cases = (
            data["glioma"] +
            data["meningioma"] +
            data["pituitary"]
        )

        # ---------------------------------------
        # Metrics
        # ---------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Predictions",
                data["total_predictions"]
            )

            st.metric(
                "Tumor Cases",
                tumor_cases
            )

            st.metric(
                "Glioma",
                data["glioma"]
            )

            st.metric(
                "Meningioma",
                data["meningioma"]
            )

        with col2:

            st.metric(
                "No Tumor Cases",
                data["notumor"]
            )

            st.metric(
                "Pituitary",
                data["pituitary"]
            )

            st.metric(
                "Total Users",
                data["total_users"]
            )

            st.metric(
                "Average Confidence",
                f"{data['average_confidence']} %"
            )

        # ---------------------------------------
        # Bar Chart
        # ---------------------------------------

        chart = pd.DataFrame({

            "Category": [

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

        st.subheader("📊 Prediction Distribution")

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