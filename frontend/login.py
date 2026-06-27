
import streamlit as st
import requests

st.set_page_config(
    page_title="User Login",
    page_icon="🔐"
)

st.title("🔐 User Login")

email = st.text_input(
    "Email"
)

password = st.text_input(
    "Password",
    type="password"
)

if st.button("Login"):

    response = requests.post(

        "http://127.0.0.1:8000/auth/login",

        json={

            "email": email,

            "password": password

        }

    )

    if response.status_code == 200:

        data = response.json()

        st.session_state["token"] = data[
            "access_token"
        ]

        st.success(
            "Login Successful"
        )

        # DEBUGGING
        st.subheader(
            "Generated JWT Token"
        )

        st.write(
            st.session_state["token"]
        )

        st.write(
            st.session_state
        )

    else:

        st.error(
            "Invalid Email or Password"
        )

        try:

            st.write(
                response.json()
            )

        except:

            st.write(
                response.text
            )

