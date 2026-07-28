# Use a Python base image with a specific version
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
# This step is done first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code only.
# Model weights and CSV are pulled from Hugging Face Hub at container
# startup (see app.py) — no need to bake them into the image.
COPY app.py .

# Expose the port the app listens on internally
EXPOSE 7860

# Command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
