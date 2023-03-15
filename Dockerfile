FROM python:3.11

# Update
RUN apt-get -y update

# Create user instead of using root
RUN groupadd -r user && useradd -r -g user user
USER user

# Define workdir
WORKDIR /home/user/app

# Install project
COPY src/ .
COPY requirements.txt .
RUN pip install -r requirements.txt

# Enjoy

