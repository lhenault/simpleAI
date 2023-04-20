# RavenRWKV service

## Description

This project uses the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) model and turns it into an gRPC service that can be used through [SimpleAI](https://github.com/lhenault/simpleAI).

RWKV is an RNN with Transformer-level language model performance that can be trained like a GPT transformer and is 100% attention-free. It combines the best of RNN and transformer, providing great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding.

## Usage

Edit the `MODEL` variable in `get_models.py` to choose the model size and context.

Edit the `STRATEGY`  variable in `lib_raven.py` to decide how the weights will be loaded, play with this to optimise the throughput for your system. See below for a graphic explanation or checkout [ChatRWKV](https://github.com/BlinkDL/ChatRWKV) for more information.

![Strategies as of 20 Apr 2023](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/536b4b3bf87fbd999798141f409b151ca91a76c7/ChatRWKV-strategy.png)

## Build

```bash
docker build . -t raven-rwkv-service:latest
```

## Start service

```bash
docker run -it --rm -p 50051:50051 --gpus all raven-rwkv-service:latest
```

## Add to model.toml

```
```toml
[raven]
    [raven.metadata]
        owned_by    = 'BlinkDL'
        permission  = []
        description = 'RWKV fine tuned for instruction answering'
    [raven.network]
        url = 'localhost:50051'
```

```

## Credits

Heavily borrowed from lhenault & BlinkDL

https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B