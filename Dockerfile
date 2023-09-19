# Docker file content
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

ENTRYPOINT ["python"] # can pass another command to run, when running the container
CMD ["test.py"]
