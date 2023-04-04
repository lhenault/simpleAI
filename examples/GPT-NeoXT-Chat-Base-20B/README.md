# OpenChatKit service

## Description

This project use the [OpenChatKit](https://github.com/togethercomputer/OpenChatKit) model and turns it into an gRPC service that can be used through [SimpleAI](https://github.com/lhenault/simpleAI).

To quote the project:

>
    OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. The kit includes an instruction-tuned 20 billion parameter language model, a 6 billion parameter moderation model, and an extensible retrieval system for including up-to-date responses from custom repositories. It was trained on the OIG-43M training dataset, which was a collaboration between Together, LAION, and Ontocord.ai. Much more than a model release, this is the beginning of an open source project. We are releasing a set of tools and processes for ongoing improvement with community contributions.

## Build

```bash
docker build . -t openchatkit-service:latest
```

## Start service

```bash
docker run -it --rm -p 50051:50051 --gpus all openchatkit-service:latest
```
