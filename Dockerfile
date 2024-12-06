# Use the latest stable Python version
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime


# Copy the project files
COPY ./app/ /app

# Copy pyproject.toml and poetry.lock (if exists) to /app
COPY pyproject.toml poetry.lock* /app

# Set the working directory
WORKDIR /app
RUN pip install poetry

# Default command to run the application
CMD ["python", "main.py"]
