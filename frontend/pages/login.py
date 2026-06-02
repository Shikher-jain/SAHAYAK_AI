"""Authentication page — login and registration for Sahayak AI."""
from __future__ import annotations

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def _call_auth(method: str, path: str, **kwargs):
    url = f"{st.session_state.get('backend_url', BACKEND_URL)}{path}"
    try:
        resp = requests.request(method, url, timeout=30, **kwargs)
        return resp
    except Exception as exc:
        return None


def show_auth_page():
    """Display login/register forms."""
    st.set_page_config(page_title="Sahayak - Login", page_icon="image.png", layout="centered")
    st.markdown(
        '<div style="text-align:center;font-size:2.5rem;font-weight:700;">Sahayak AI</div>'
        '<div style="text-align:center;color:#666;">Your AI Learning Companion</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab_login, tab_register = st.tabs(["Login", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted and username and password:
                resp = _call_auth("post", "/auth/login", json={"username": username, "password": password})
                if resp and resp.status_code == 200:
                    data = resp.json()
                    st.session_state.auth_token = data["access_token"]
                    st.session_state.auth_user = data["username"]
                    st.session_state.auth_role = data["role"]
                    st.success(f"Welcome back, {data['username']}!")
                    st.switch_page("app.py")        
                    # st.success(f"Welcome back, {data['username']}!")
                    # st.rerun()
                else:
                    detail = resp.json().get("detail", "Login failed") if resp else "Server unreachable"
                    st.error(detail)

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_user")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            new_fullname = st.text_input("Full Name (optional)", key="reg_name")
            role = st.selectbox("Role", ["student", "teacher", "admin"], key="reg_role")
            reg_submitted = st.form_submit_button("Create Account")
            if reg_submitted and new_username and new_email and new_password:
                resp = _call_auth("post", "/auth/register", json={
                    "username": new_username,
                    "email": new_email,
                    "password": new_password,
                    "role": role,
                    "full_name": new_fullname,
                })
                if resp and resp.status_code == 201:
                    st.success("Account created! Please login.")
                else:
                    detail = resp.json().get("detail", "Registration failed") if resp else "Server unreachable"
                    st.error(detail)


def is_authenticated() -> bool:
    return bool(st.session_state.get("auth_token"))

show_auth_page()