# # utils/filters.py
# from sqlalchemy.orm import Session
# from sqlalchemy import desc
# from models import MoodLog

# def last_n_moods(session: Session, user_id: int, n: int = 2):
#     return (
#         session.query(MoodLog)
#         .filter(MoodLog.user_id == user_id)
#         .order_by(desc(MoodLog.timestamp))
#         .limit(n)
#         .all()
#     )

# def two_consecutive_sad(session: Session, user_id: int) -> bool:
#     logs = last_n_moods(session, user_id, n=2)
#     if len(logs) < 2:
#         return False
#     return all((log.dominant_emotion or "").lower() == "sadness" for log in logs)
