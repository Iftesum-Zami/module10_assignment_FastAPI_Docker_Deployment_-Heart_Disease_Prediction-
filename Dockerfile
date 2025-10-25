# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model code
COPY app/ ./app/
COPY model/ ./model/

# Add PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app

# Expose port (if your app uses one, e.g. Flask)
EXPOSE 5000

# Update entrypoint to use uvicorn for ASGI apps
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
