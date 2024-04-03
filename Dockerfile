# Use an official Python runtime as a parent image
FROM python:3.9.6-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY . /app
RUN ls -l
RUN cat requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
#EXPOSE 5001

# Define environment variable
#ENV FLASK_APP=app.py
#ENV FLASK_RUN_PORT=5001

CMD ["python", "app.py", "server"]