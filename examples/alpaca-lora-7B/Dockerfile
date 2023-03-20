FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Update, install
RUN apt update && \
    apt install -y build-essential python3 python3-pip git

RUN pip install --upgrade pip setuptools wheel

# Create user instead of using root
ENV USER='user'
RUN groupadd -r user && useradd -r -g $USER $USER
USER $USER

# Define workdir
WORKDIR /home/$USER/app

# Install project
COPY . .
RUN pip install -r requirements.txt

# Fix: OSError (non writable)
RUN mkdir -p /home/$USER/.cache/huggingface/hub/models--tloen--alpaca-lora-7b
RUN chmod a+rwx -R /home/$USER/.cache/huggingface/hub/models--tloen--alpaca-lora-7b

# Get model weights and tokenizer
RUN python3 get_models.py

# Publish port
EXPOSE 50051:50051

# Enjoy
ENTRYPOINT ["python3", "server.py"]
CMD ["--address", "[::]:50051"]
