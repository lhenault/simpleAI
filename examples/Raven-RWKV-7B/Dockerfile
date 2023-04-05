FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Update, install
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip ninja-build git

RUN pip install --upgrade pip setuptools wheel

# Create user instead of using root
ENV USER='user'
RUN groupadd -r user && useradd -r -g $USER $USER
RUN mkdir /home/$USER
RUN chown $USER:$USER /home/$USER
USER $USER

# Define workdir
WORKDIR /home/$USER/app

# Install project
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
# Get model weights and tokenizer
RUN python3 get_models.py
# Publish port
EXPOSE 50051:50051

# Enjoy
ENTRYPOINT ["python3", "server.py"]
CMD ["--address", "[::]:50051"]