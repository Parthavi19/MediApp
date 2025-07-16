# Use Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements (if you use requirements.txt)
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Install dependencies directly
RUN pip install flask "transformers[torch]" datasets accelerate

# Copy the entire app
COPY . .

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
