# StableLM - Open Assistant

## Description

This example shows how to use the [Open-Assistant StableLM-7B SFT-7 model](https://huggingface.co/OpenAssistant/stablelm-7b-sft-v7-epoch-3), based [StableLM](https://github.com/Stability-AI/StableLM) by [Stability.AI](https://stability.ai/) and fine-tuned on human demonstrations as part of the [Open-Assistant project](https://github.com/LAION-AI/Open-Assistant), with a [SimpleAI](https://github.com/lhenault/simpleAI) server.

It implements `chat` methods for both streaming and non-streaming.

## Setup

First build the image with:

```bash
docker build . -t stablechat:0.1
```

Then declare your model in your *SimpleAI* configuration file `models.toml`:

```toml
[stablelm-open-assistant]
    [stablelm-open-assistant.metadata]
        owned_by    = 'OpenAssistant'
        permission  = []
        description = 'This is the 7th iteration English supervised-fine-tuning (SFT) model of the Open-Assistant project. It is based on a StableLM 7B that was fine-tuned on human demonstrations of assistant conversations collected through the https://open-assistant.io/ human feedback web app before April 12, 2023.'
    [stablelm-open-assistant.network]
        type = 'gRPC'
        url = 'localhost:50051'
```

## Start service

Just start your container with:

```bash
docker run -it --rm -p 50051:50051 --gpus all stablechat:0.1
```

And start your *SimpleAI* instance, for instance with:

```bash
simple_ai serve [--host 127.0.0.1] [--port 8080]
```
