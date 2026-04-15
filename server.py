"""
Compatibility wrapper for Render deployment.
This file exists because Render may have cached 'gunicorn server:app' as the start command.
It simply imports the Flask app from app.py.
"""
from app import app

if __name__ == "__main__":
    app.run()
