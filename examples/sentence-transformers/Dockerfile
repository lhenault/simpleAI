FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Update, install
ENV PY_VERSION='3.9'
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y build-essential python${PY_VERSION} git
RUN apt install -y python3-pip python3-distutils
RUN python${PY_VERSION} -m pip install --upgrade pip

# Create user instead of using root
ENV USER='user'
RUN groupadd -r user && useradd -r -g $USER $USER
USER $USER

# Define workdir
WORKDIR /home/$USER/app

# Install project
COPY . .
RUN python${PY_VERSION} -m pip install -r requirements.txt

# Get model weights and tokenizer
RUN python${PY_VERSION} get_models.py

# Publish port
EXPOSE 50051:50051

# Enjoy
ENTRYPOINT python${PY_VERSION} server.py
CMD ["--address", "[::]:50051"]
