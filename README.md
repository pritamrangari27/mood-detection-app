# Mood Detection App

Short guide to run and deploy this Flask app locally, with Docker, or to a PaaS.

Prerequisites
- Python 3.10+ (for local venv) or Docker Desktop (recommended)
- `emotion_model.h5` (place in repo root or mount at runtime)

Environment variables
- `FLASK_SECRET_KEY` — Flask secret
- `JWT_SECRET_KEY` — JWT signing secret
- `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` — optional for music features

1) Run locally (quick, development)

PowerShell commands:
```powershell
cd 'C:\Users\LENOVO\Desktop\Project ML2'
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
set FLASK_SECRET_KEY=changeit
set JWT_SECRET_KEY=changeit
python app.py
```

- The app listens on port 5000 by default. Use `http://localhost:5000`.
- If `tensorflow` install is slow/fails, use Docker (recommended).

2) Run with Gunicorn (production-like, Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FLASK_SECRET_KEY=changeit
export JWT_SECRET_KEY=changeit
gunicorn --bind 0.0.0.0:5000 app:app --workers 3 --threads 2
```

3) Docker (recommended for reproducible deploy)

- Build image (mount model at runtime to avoid baking large `.h5` into the image):

```bash
docker build -t mood-app:latest .
docker run -d -p 5000:5000 \
  -e FLASK_SECRET_KEY=changeit -e JWT_SECRET_KEY=changeit \
  -v "$(pwd)/emotion_model.h5:/app/emotion_model.h5:ro" \
  --name mood-app-container mood-app:latest
```

- On Windows PowerShell, replace `$(pwd)` with `%CD%` or full path.
- If installing `tensorflow` in the slim image fails, use a TF base image instead (example in `Dockerfile`).

4) Deploy to a PaaS (Render, Railway, Azure Web App for Containers)

- Option A (Render with Docker): Connect repo, choose Dockerfile, set env vars (`FLASK_SECRET_KEY`, `JWT_SECRET_KEY`, `SPOTIFY_*`).
- Option B (Heroku-like with buildpack): Use `Procfile` (already added). Note: Heroku slug size limits may block large TF models — prefer Docker or host model externally.

Procfile (already present):

```
web: gunicorn --bind 0.0.0.0:$PORT app:app
```

5) Model storage options
- Bake into image: copy `emotion_model.h5` into image (increase image size).
- Mount at runtime (recommended for development): use `-v` to mount local file.
- External storage: upload model to S3/GCS and download on startup — implement a small startup script to fetch if missing.

6) Troubleshooting
- TensorFlow install fails: try an official `tensorflow/tensorflow` base image or use a machine with compatible wheel support.
- App can't find model: confirm `emotion_model.h5` exists in repository root or the mounted path `/app/emotion_model.h5`.
- Ports/Firewall: ensure port 5000 is open on server/VM and platform routing is configured.

7) Next steps I can do for you
- Add a startup script to download the model from cloud storage.
- Add CI/CD steps to build and push Docker images to a registry.

Enjoy — tell me which next step you'd like me to implement.
