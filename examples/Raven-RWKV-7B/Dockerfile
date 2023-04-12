FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Update, install
RUN apt-get update && \
    apt-get install -y build-essential ninja-build git wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda create -y --name py39 python=3.9 && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/py39/bin:$PATH

RUN pip install --upgrade pip setuptools wheel

# Create user instead of using root
ENV USER='user'
RUN groupadd -r user && useradd -r -g $USER $USER
RUN mkdir -p /home/$USER/app
RUN chown -R $USER:$USER /home/$USER
USER $USER

# Define workdir
WORKDIR /home/$USER/app

# Install project
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY get_models.py .

# Get model weights and tokenizer
RUN python3 get_models.py

# Copy rest
COPY . .

# Publish port
EXPOSE 50051:50051

# Enjoy
ENTRYPOINT ["python3", "server.py"]
CMD ["--address", "[::]:50051"]