# Use Python 3.12 base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire app and checkpoint
COPY . .

# Create checkpoint directory if it doesn't exist
RUN mkdir -p /app/tinyllama-chatdoctor-checkpoint

# Copy checkpoint if it exists
COPY ./tinyllama-chatdoctor-checkpoint /app/tinyllama-chatdoctor-checkpoint

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the app with Gunicorn with startup-friendly settings
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 7200 --keep-alive 2 --max-requests 1000 --max-requests-jitter 50 --preload app:app
