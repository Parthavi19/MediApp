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
COPY ./tinyllama-chatdoctor-checkpoint /app/tinyllama-chatdoctor-checkpoint

# Expose port
EXPOSE 8080

# Run the app with Gunicorn, binding explicitly to PORT
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8080", "app:app"]
