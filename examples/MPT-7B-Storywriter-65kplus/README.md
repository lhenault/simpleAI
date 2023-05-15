# MosaicML's MPT-7B-StoryWriter-65k+ (4-bit quantization)

## Description

This example shows how to use the [MPT-7B-StoryWriter-65k+ model](https://huggingface.co/mosaicml/mpt-7b-storywriter), with a [SimpleAI](https://github.com/lhenault/simpleAI) server.

It implements `complete` method.

## Setup

First build the image with:

```bash
docker build . -t mpt-7b-storywriter:0.1
```

Then declare your model in your *SimpleAI* configuration file `models.toml`:

```toml
[mpt-7b-storywriter]
    [mpt-7b-storywriter.metadata]
        owned_by    = 'MosaicML'
        permission  = []
        description = 'MPT-7B-StoryWriter-65k+ is a model designed to read and write fictional stories with super long context lengths. It was built by finetuning MPT-7B with a context length of 65k tokens on a filtered fiction subset of the books3 dataset. At inference time, thanks to ALiBi, MPT-7B-StoryWriter-65k+ can extrapolate even beyond 65k tokens.'
    [mpt-7b-storywriter.network]
        type = 'gRPC'
        url = 'localhost:50051'
```

## Start service

Just start your container with:

```bash
docker run -it --rm -p 50051:50051 --gpus all mpt-7b-storywriter:0.1
```

And start your *SimpleAI* instance, for instance with:

```bash
simple_ai serve [--host 127.0.0.1] [--port 8080]
```
