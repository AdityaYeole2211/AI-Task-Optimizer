# create_hr_user.py
from app.db import get_session
from app.models import User
from werkzeug.security import generate_password_hash
import getpass

def create_hr_user(name, email, password):
    """Create an HR user manually (admin-only)."""
    session = get_session()
    try:
        if session.query(User).filter_by(email=email).first():
            print("❌ HR with this email already exists.")
            return
        hashed_pw = generate_password_hash(password)
        hr = User(name=name, email=email, password_hash=hashed_pw, role="hr")
        session.add(hr)
        session.commit()
        print(f"✅ HR user '{email}' created successfully!")
    except Exception as e:
        print("⚠️ Error creating HR:", e)
    finally:
        session.close()

if __name__ == "__main__":
    print("---- HR User Creation ----")
    name = input("Full name: ")
    email = input("Email: ")
    password = getpass.getpass("Password: ")
    create_hr_user(name, email, password)
