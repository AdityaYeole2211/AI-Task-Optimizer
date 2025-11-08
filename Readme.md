🧠 AI Task Optimizer
🎭 Multi-Modal Emotion Detection + Task Recommendation + HR Dashboard
📘 Overview

AI Task Optimizer is a Streamlit-based system that analyzes both text and facial expressions to detect an employee’s current emotion and recommend suitable tasks.
It includes a role-based login system with two types of users:

🧑‍💻 Employee: Enters text and uploads a face image to get emotion analysis and an AI-recommended task.

🧑‍💼 HR/Admin: Views historical mood logs of all employees, filters them by employee/date, and receives automatic alerts if an employee shows consecutive negative emotions.

The project combines:

Deep learning models for text and image emotion recognition

A Streamlit UI for interaction and visualization

An SQLite database for user authentication and mood logging

Automated HR email alerts for employee well-being tracking

⚙️ Features
Feature	Description
🧑‍💻 Employee Login	Employees can register, log in, and record their mood via text + image.
🎭 Multi-Modal Emotion Detection	Combines text and image emotions using weighted averaging (0.6 image + 0.4 text).
🤖 AI Task Recommendation	Gemini-powered LLM suggests context-appropriate tasks based on detected emotion.
💾 Local Database (SQLite)	Stores users, roles, and mood logs persistently.
📊 HR Dashboard	Displays employee mood history, graphs over time, and filters by employee/date.
📧 Email Alerts	Automatically notifies HR if an employee shows consecutive negative emotions.
🏗️ Project Structure:

```bash
AI-Task-Optimizer/
│
├── app/
│   ├── app.py                 # Main Streamlit application
│   ├── auth.py                # Login/Register logic
│   ├── db.py                  # Database session setup
│   ├── models.py              # SQLAlchemy models (User, MoodLog)
│   ├── hr_dashboard.py        # HR dashboard and analytics
│   ├── create_hr_user.py      # Script for admin to create HR users
│   ├── utils/
│   │   ├── email_alerts.py    # Email alert system
│   │   ├── filters.py         # Mood filtering utilities
│   │   └── __init__.py
│   ├── text_utils.py          # Text emotion model loader & predictor
│   ├── image_utils.py         # Image emotion model loader & predictor
│   └── __init__.py
│
├── models/
│   ├── emotion_lstm_model.h5  # Pretrained text emotion model
│   └── tokenizer.pkl          # Tokenizer for text model
│
├── mood_tracker.db            # SQLite database
├── .env                       # API keys & SMTP credentials (not tracked)
├── .gitignore
├── requirements.txt
└── README.md
```
🧩 Tech Stack

Frontend/UI: Streamlit

Backend: Python 3.10+, SQLite (via SQLAlchemy ORM)

AI Models:

Keras LSTM model for text emotion detection

CNN model for image emotion detection

LLM Integration: Gemini (Google Generative AI)

Visualization: Matplotlib / Plotly (for HR graphs)

Email Service: SMTP (Gmail)

🚀 Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/AdityaYeole2211/AI-Task-Optimizer.git
cd AI-Task-Optimizer

2️⃣ Create and activate a virtual environment
python -m venv amdvenv
# On Windows:
amdvenv\Scripts\activate
# On macOS/Linux:
source amdvenv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Configure environment variables

Create a .env file in the project root with:

GEMINI_API_KEY=your_gemini_api_key

```ini
# Email SMTP settings (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
FROM_EMAIL=your_email@gmail.com
```
5️⃣ Initialize the database (first run only)
python -m app.models

6️⃣ Create an HR user (admin-only)
python -m app.create_hr_user

7️⃣ Run the Streamlit app
streamlit run app/app.py

👥 Roles
Employee

Registers via UI

Logs emotions (text + image)

Receives task recommendations

Mood history automatically stored in DB

HR

Added by admin via create_hr_user.py

Logs in via Streamlit app

Views mood history dashboard

Gets email notifications for consecutive negative moods

📊 HR Dashboard Features

Filter by employee name or ID

Filter by date range

Visualize trends (e.g., “Employee Mood Over Time”)

Export mood logs as CSV

🧠 Task Recommendation Logic

If emotion is negative (sadness, fear, anger, disgust):

Suggest light or easy tasks to reduce cognitive load.

If emotion is positive (joy, surprise, neutral):

Suggest tasks requiring more focus and effort.

📬 Email Alerts

If an employee records two consecutive negative emotions,
an alert email is automatically sent to all HR emails listed in the database.

☁️ Deployment (Streamlit Cloud)

Push your repository to GitHub.

Go to https://share.streamlit.io
.

Connect your repo.

Add .env variables under Secrets in Streamlit Cloud:

GEMINI_API_KEY=your_key
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
FROM_EMAIL=your_email@gmail.com


Deploy 🚀