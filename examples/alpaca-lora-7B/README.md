# Alpaca-7B service

## Description

This project use the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) (Low-Rank LLaMA Instruct-Tuning) model and turns it into an gRPC service through [SimpleAI](https://github.com/lhenault/simpleAI).

## Build

```bash
docker build . -t alpaca-7b-service:latest
```

## Start service

```bash
docker run -it --rm -p 50051:50051 --gpus all alpaca-7b-service:latest
```
