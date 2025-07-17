FROM python:3.10-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CHECKPOINT_DIR=/app/tinyllama-chatdoctor-checkpoint

# Set working directory
WORKDIR /app

# Copy all project files (make sure templates/, static/ etc. are included)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask/Gunicorn will run on
EXPOSE 8080

# Run the app using Gunicorn (1 worker)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
