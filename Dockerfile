# Use official Python runtime as parent image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app folder and other necessary files
COPY app ./app
COPY models ./models
COPY utils ./utils
COPY metrics ./metrics

# Expose port for Streamlit or your app (default 8501 for Streamlit)
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]