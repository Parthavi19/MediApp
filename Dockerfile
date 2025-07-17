# Use slim base image with Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV CHECKPOINT_DIR=/app/tinyllama-chatdoctor-checkpoint

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Use Gunicorn to run Flask (faster, safer, recommended)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
