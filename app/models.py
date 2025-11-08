# models.py
import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Business-friendly: email is login, name is display
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="employee")  # employee | hr | admin
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    mood_logs = relationship("MoodLog", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"

class MoodLog(Base):
    __tablename__ = "mood_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    dominant_emotion = Column(String(50))
    confidence = Column(Float)

    # store emotion probabilities (JSON string) for analysis
    emotion_probs = Column(String)  # JSON-serialized dict

    user = relationship("User", back_populates="mood_logs")

    def __repr__(self):
        return (
            f"<MoodLog(id={self.id}, user_id={self.user_id}, "
            f"emotion='{self.dominant_emotion}', confidence={self.confidence})>"
        )

def init_db(db_url="sqlite:///mood_tracker.db"):
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine

if __name__ == "__main__":
    engine = init_db()
    print("✅ Database initialized and tables created successfully!")
