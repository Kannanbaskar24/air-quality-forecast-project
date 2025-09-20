# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI with reload (optional in production)
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
