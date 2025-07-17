# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Ensure the checkpoint directory exists
RUN mkdir -p /app/tinyllama-chatdoctor-checkpoint

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run using Gunicorn (NO --preload)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "7200", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "50", "app:app"]
