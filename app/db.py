# db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.models import Base, init_db 

DATABASE_URL = "sqlite:///mood_tracker.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_session():
    return SessionLocal()

def reset_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("🔄 Database reset complete!")

if __name__ == "__main__":
    print("✅ Database connection successful!")
