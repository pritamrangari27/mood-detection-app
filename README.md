---
title: Mood Detection App
emoji: 🎭
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
---

# Mood Detection App

A Flask-based web application that detects your mood from facial expressions using AI and recommends Spotify songs based on your emotion.

## Features
- 🎭 Real-time facial emotion detection using webcam
- 🎵 Spotify song recommendations based on detected mood
- 💬 User chat functionality
- 👤 User authentication with JWT
- 📊 SQLite database for persistent storage

## Tech Stack
- **Backend:** Flask, SQLAlchemy, JWT
- **ML:** TensorFlow Lite (FER2013 emotion model)
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **Music API:** Spotify Web API

## How to Use
1. Register a new account or login
2. Click "Detect Mood" on the dashboard
3. Allow camera access and capture your face
4. Get song recommendations based on your detected emotion!

## Environment Variables
- `FLASK_SECRET_KEY` — Flask session secret
- `JWT_SECRET_KEY` — JWT signing secret
- `SPOTIFY_CLIENT_ID` — Spotify API client ID
- `SPOTIFY_CLIENT_SECRET` — Spotify API secret

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate on Windows
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`

## License
MIT
