# RavenRWKV service

## Description

This project uses the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) model and turns it into an gRPC service that can be used through [SimpleAI](https://github.com/lhenault/simpleAI).

RWKV is an RNN with Transformer-level language model performance that can be trained like a GPT transformer and is 100% attention-free. It combines the best of RNN and transformer, providing great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding.

## Build

```bash
docker build . -t raven-rwkv-service:latest
```

## Start service

```bash
docker run -it --rm --p 50051:50051 --gpus all raven-rwkv-service:latest
```


## Credits

Heavily borrowed from lhenault & BlinkDL

https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B