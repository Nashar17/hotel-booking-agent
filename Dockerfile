# Base image — Python 3.11 slim (lighter than full Python image)
# We use 3.11 instead of 3.13 for maximum library compatibility
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker layer caching — faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Set environment variable so Streamlit doesn't open a browser inside the container
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the app
CMD ["python", "-m", "streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]