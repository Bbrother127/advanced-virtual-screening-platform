#!/bin/bash
# Build script for SnapDeploy
set -e

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Starting application with gunicorn..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120