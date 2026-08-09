import streamlit as st
import requests
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL",  "https://brain-tumor-detection-system-bzzb.onrender.com")

st.title("📜 Prediction History")

if "token" not in st.session_state:

    st.warning(
        "Please login first."
    )

    st.stop()

headers = {

    "Authorization":
    f"Bearer {st.session_state['token']}"

}

response = requests.get(

    f"{BACKEND_URL}/history",

    headers=headers,
    timeout=180

)

if response.status_code == 200:

    history = response.json()

    if len(history) == 0:

        st.info(
            "No prediction history found."
        )

    else:

        for item in history:
    
         with st.container():

           st.subheader(item["filename"])

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

           if st.button(
            f"View Grad-CAM - {item['filename']}",
            key=f"gradcam_{item['id']}"
        ):

            try:

                gradcam_response = requests.get(
                    item["gradcam_url"],
                    headers=headers,
                    timeout=180
                )

                if gradcam_response.status_code == 200:

                    from io import BytesIO

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

                else:

                    st.warning(
                        "Grad-CAM unavailable."
                    )

            except requests.exceptions.RequestException as e:

                st.warning(
                    f"Unable to load Grad-CAM: {e}"
                )

        st.divider()
else:

    st.error(

        "Unable to fetch history"

    )