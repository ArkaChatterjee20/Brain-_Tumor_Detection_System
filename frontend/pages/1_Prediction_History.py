import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://brain-tumor-detection-system-bzzb.onrender.com"
).rstrip("/")


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("📜 Prediction History")


# --------------------------------------------------
# Authentication check
# --------------------------------------------------

token = st.session_state.get("token")

if not token:
    st.warning("Please login first.")
    st.stop()


# --------------------------------------------------
# Headers
# --------------------------------------------------

headers = {
    "Authorization": f"Bearer {token}"
}


# --------------------------------------------------
# Get prediction history
# --------------------------------------------------

try:

    response = requests.get(
        f"{BACKEND_URL}/history",
        headers=headers,
        timeout=(10, 120)
    )

except requests.exceptions.Timeout:

    st.error(
        "The backend took too long to respond. "
        "Please try again."
    )

    st.stop()

except requests.exceptions.ConnectionError:

    st.error(
        "Unable to connect to the backend. "
        "Please try again in a few seconds."
    )

    st.stop()

except requests.exceptions.RequestException as e:

    st.error(
        f"Unable to fetch prediction history: {e}"
    )

    st.stop()


# --------------------------------------------------
# Handle authentication failure
# --------------------------------------------------

if response.status_code == 401:

    st.error(
        "Your login session has expired. "
        "Please login again."
    )

    st.stop()


# --------------------------------------------------
# Handle other errors
# --------------------------------------------------

if response.status_code != 200:

    st.error(
        f"Unable to fetch history. "
        f"Backend status: {response.status_code}"
    )

    st.stop()


# --------------------------------------------------
# Read history
# --------------------------------------------------

history = response.json()


if not history:

    st.info("No prediction history found.")

    st.stop()


# --------------------------------------------------
# Display history
# --------------------------------------------------

for item in history:

    with st.container():

        st.subheader(
            item["filename"]
        )

        st.write(
            "Prediction:",
            item["predicted_class"]
        )

        st.write(
            "Confidence:",
            f"{item['confidence']} %"
        )

        st.write(
            "Prediction Time:",
            item["prediction_time"]
        )


        # --------------------------------------------------
        # Grad-CAM
        # --------------------------------------------------

        if st.button(
            f"View Grad-CAM - {item['filename']}",
            key=f"gradcam_{item['id']}"
        ):

            try:

                gradcam_response = requests.get(
                    item["gradcam_url"],
                    headers=headers,
                    timeout=(10, 120)
                )

                if gradcam_response.status_code == 200:

                    image = Image.open(
                        BytesIO(
                            gradcam_response.content
                        )
                    )

                    st.image(
                        image,
                        caption="Grad-CAM",
                        width=300
                    )

                elif gradcam_response.status_code == 401:

                    st.error(
                        "Your login session has expired."
                    )

                elif gradcam_response.status_code == 404:

                    st.warning(
                        "Grad-CAM is unavailable for this prediction."
                    )

                else:

                    st.warning(
                        f"Grad-CAM request failed "
                        f"({gradcam_response.status_code})."
                    )

            except requests.exceptions.Timeout:

                st.warning(
                    "Grad-CAM request timed out. "
                    "Please try again."
                )

            except requests.exceptions.ConnectionError:

                st.warning(
                    "Could not connect to the backend."
                )

            except requests.exceptions.RequestException as e:

                st.warning(
                    f"Unable to load Grad-CAM: {e}"
                )


        # --------------------------------------------------
        # Download report
        # --------------------------------------------------

        if st.button(
            f"Download Report - {item['filename']}",
            key=f"report_{item['id']}"
        ):

            try:

                report_response = requests.get(
                    item["report_url"],
                    headers=headers,
                    timeout=(10, 120)
                )

                if report_response.status_code == 200:

                    st.download_button(
                        label="📄 Download PDF",
                        data=report_response.content,
                        file_name=f"report_{item['id']}.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_{item['id']}"
                    )

                elif report_response.status_code == 401:

                    st.error(
                        "Your login session has expired."
                    )

                elif report_response.status_code == 404:

                    st.warning(
                        "Report is not available."
                    )

                else:

                    st.warning(
                        f"Report request failed "
                        f"({report_response.status_code})."
                    )

            except requests.exceptions.Timeout:

                st.warning(
                    "Report generation/download timed out."
                )

            except requests.exceptions.ConnectionError:

                st.warning(
                    "Could not connect to the backend."
                )

            except requests.exceptions.RequestException as e:

                st.warning(
                    f"Unable to download report: {e}"
                )


        st.divider()