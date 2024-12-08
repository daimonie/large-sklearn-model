# Use the latest stable Python version
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime


# Copy the project files
COPY ./app/ /app
 
# Set the working directory
WORKDIR /app
RUN pip install click==8.1.7 gutenbergpy==0.3.5
# Default command to run the application
CMD ["python", "main.py"]
