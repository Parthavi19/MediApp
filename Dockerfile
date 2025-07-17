# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app and checkpoint
COPY . .
COPY ./tinyllama-chatdoctor-checkpoint /app/tinyllama-chatdoctor-checkpoint

# Expose port
EXPOSE 8080

# Run the app with Gunicorn for Cloud Run compatibility
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
