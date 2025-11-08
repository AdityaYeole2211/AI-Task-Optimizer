# app/utils/email_alerts.py
import json
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from typing import Dict
from app.db import get_session
from app.models import MoodLog, User


NEGATIVE_EMOTIONS = {"sadness", "anger", "fear"}  # Customize if needed


def save_mood_log(user_id: int, merged_probs: Dict[str, float]):
    """Save the latest mood log to database."""
    session = get_session()
    try:
        dominant = max(merged_probs, key=merged_probs.get)
        confidence = float(merged_probs[dominant])
        mood_log = MoodLog(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            dominant_emotion=dominant,
            confidence=confidence,
            emotion_probs=json.dumps(merged_probs),
        )
        session.add(mood_log)
        session.commit()
        session.refresh(mood_log)
        return mood_log
    finally:
        session.close()


def last_n_logs(user_id: int, n: int = 2):
    """Fetch last n mood logs for given user."""
    session = get_session()
    try:
        logs = (
            session.query(MoodLog)
            .filter_by(user_id=user_id)
            .order_by(MoodLog.timestamp.desc())
            .limit(n)
            .all()
        )
        return logs
    finally:
        session.close()


def should_alert_recent_negative(user_id: int, n: int = 2) -> bool:
    """Return True if the last n moods are all negative emotions."""
    logs = last_n_logs(user_id, n)
    if len(logs) < n:
        return False
    return all(log.dominant_emotion.lower() in NEGATIVE_EMOTIONS for log in logs)


def get_hr_emails():
    """Fetch all HR user emails."""
    session = get_session()
    try:
        hrs = session.query(User).filter_by(role="hr").all()
        return [h.email for h in hrs if h.email]
    finally:
        session.close()


def send_email_alert(to_emails, subject, body):
    """Send an email alert via SMTP."""
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    from_email = os.getenv("FROM_EMAIL", smtp_user)

    if not smtp_host or not smtp_user or not smtp_pass:
        raise ValueError("SMTP credentials missing from .env")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
