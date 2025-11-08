# auth.py
import streamlit as st
from sqlalchemy.orm import Session
from app.models import User   # ✅ ensure full package path
from app.db import get_session
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------
# Core auth operations
# -----------------------
def register_user(name: str, email: str, password: str, role: str = "employee") -> str:
    session: Session = get_session()
    try:
        existing = session.query(User).filter_by(email=email).first()
        if existing:
            return "❌ Email already registered."

        hashed_pw = generate_password_hash(password)
        new_user = User(name=name, email=email, password_hash=hashed_pw, role=role)
        session.add(new_user)
        session.commit()
        return "✅ Registration successful!"
    except Exception as e:
        session.rollback()
        return f"⚠️ Error: {e}"
    finally:
        session.close()

def login_user(email: str, password: str) -> str:
    session: Session = get_session()
    try:
        user = session.query(User).filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return "❌ Invalid email or password."

        st.session_state["logged_in"] = True
        st.session_state["user_id"] = user.id
        st.session_state["name"] = user.name
        st.session_state["email"] = user.email
        st.session_state["role"] = user.role
        return f"✅ Welcome back, {user.name}!"
    except Exception as e:
        return f"⚠️ Error: {e}"
    finally:
        session.close()

def logout_user():
    for key in ["logged_in", "user_id", "name", "email", "role"]:
        st.session_state.pop(key, None)

# -----------------------
# Streamlit UI widgets
# -----------------------
def show_login_page():
    st.subheader("🔐 Login or Register")

    t1, t2 = st.tabs(["Login", "Register"])

    with t1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            msg = login_user(email, password)
            st.info(msg)
            if "Welcome back" in msg:
                st.rerun()

    with t2:
        name = st.text_input("Full Name", key="reg_name")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register"):
            if not name or not email or not password:
                st.warning("Please fill all fields.")
            else:
                msg = register_user(name, email, password, role="employee")
                st.info(msg)

def show_logout_button():
    if st.button("Logout"):
        logout_user()
        st.success("You have been logged out.")
        st.rerun()
