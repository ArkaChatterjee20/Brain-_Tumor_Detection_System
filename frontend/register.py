import streamlit as st
import requests

st.set_page_config(
    page_title="Register",
    page_icon="📝"
)

st.title("📝 User Registration")

username = st.text_input(
    "Username"
)

email = st.text_input(
    "Email"
)

password = st.text_input(
    "Password",
    type="password"
)

if st.button("Register"):

    data = {

        "username": username,

        "email": email,

        "password": password

    }

    response = requests.post(
        "http://127.0.0.1:8000/auth/register",
        json=data
    )

    if response.status_code == 200:

        st.success(
            "Registration Successful!"
        )

    else:

        st.error(
            response.json()["detail"]
        )