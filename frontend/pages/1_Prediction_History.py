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

    headers=headers

)

if response.status_code == 200:

    history = response.json()

    if len(history) == 0:

        st.info(
            "No prediction history found."
        )

    else:

        for item in history:
    
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
             "Time:",
             item["prediction_time"]
          )

          try:

              response = requests.get(
                 item["gradcam_url"],
                 headers=headers
              )

              if response.status_code == 200:

                from io import BytesIO

                image = Image.open(
                    BytesIO(response.content)
                )

                st.image(
                  image,
                  width=300,
                  caption="Grad-CAM"
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

        "Unable to fetch history"

    )