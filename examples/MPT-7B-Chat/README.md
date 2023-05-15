# MosaicML's MPT-7B-Chat

## Description

This example shows how to use [MPT-7B-Chat model](https://www.mosaicml.com/blog/mpt-7b), with a [SimpleAI](https://github.com/lhenault/simpleAI) server.

It implements `chat` methods for both streaming and non-streaming.

## Setup

First build the image with:

```bash
docker build . -t mpt-7b-chat:0.1
```

Then declare your model in your *SimpleAI* configuration file `models.toml`:

```toml
[mpt-7b-chat]
    [mpt-7b-chat.metadata]
        owned_by    = 'MosaicML'
        permission  = []
        description = 'MPT-7B-Chat is a chatbot-like model for dialogue generation. Built by finetuning MPT-7B on the ShareGPT-Vicuna, HC3, Alpaca, Helpful and Harmless, and Evol-Instruct datasets.'
    [mpt-7b-chat.network]
        type = 'gRPC'
        url = 'localhost:50051'
```

## Start service

Just start your container with:

```bash
docker run -it --rm -p 50051:50051 --gpus all mpt-7b-chat:0.1
```

And start your *SimpleAI* instance, for instance with:

```bash
simple_ai serve [--host 127.0.0.1] [--port 8080]
```
