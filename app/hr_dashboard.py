# app/hr_dashboard.py
import streamlit as st
from app.db import get_session
from app.models import User, MoodLog
import pandas as pd
import plotly.express as px

def show_hr_dashboard():
    st.title("📊 HR Dashboard — Employee Mood Overview")

    session = get_session()

    users = session.query(User).all()
    logs = session.query(MoodLog).all()

    if not logs:
        st.info("No mood logs available yet.")
        return

    df = pd.DataFrame([
        {
            "Employee": log.user.name,
            "Email": log.user.email,
            "Emotion": log.dominant_emotion,
            "Confidence": log.confidence,
            "Timestamp": log.timestamp,
        }
        for log in logs
    ])

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    emp_filter = st.sidebar.selectbox("Select Employee", ["All"] + sorted(df["Employee"].unique()))
    emo_filter = st.sidebar.selectbox("Filter by Emotion", ["All"] + sorted(df["Emotion"].unique()))

    if emp_filter != "All":
        df = df[df["Employee"] == emp_filter]
    if emo_filter != "All":
        df = df[df["Emotion"] == emo_filter]

    # Table
    st.subheader("📄 Mood Log Records")
    st.dataframe(df, use_container_width=True)

    # Graph 1: Emotion Distribution
    st.subheader("📊 Emotion Distribution (Overall / Filtered)")
    fig = px.histogram(df, x="Emotion", color="Emotion", title="Emotion Counts by Type")
    st.plotly_chart(fig, use_container_width=True)

    # Graph 2: Mood Trends Over Time
    st.subheader("📈 Mood Trends Over Time")
    df_sorted = df.sort_values("Timestamp")
    fig2 = px.scatter(df_sorted, x="Timestamp", y="Emotion", color="Employee",
                      title="Mood Timeline by Employee")
    st.plotly_chart(fig2, use_container_width=True)

    session.close()
