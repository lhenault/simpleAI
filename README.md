# SimpleAI

A self-hosted alternative to the not-so-open AI API. It mostly focus on replicating the main endpoints for LLM:

- Text completion
- Chat
- Edits

While it is not an exact replication of the original endpoints, it should be relatively to switch from one to the other, allowing you to experiment with competing approaches quickly and easily.

## Installation

On a machine with Python 3.11+, simply navigate in the root directory and use:

```bash
pip install -r requirements.txt
```

There is alternatively a Dockerfile you can use to build an image and execute within a container.

## Setup

Copy the template `models.toml.template` and create `models.toml` : that will be where you declare your different models.

Once this is done, you can start adding your models, but in the meantime you can already start a server locally with:

```bash
uvicorn app:app --reload
```

## Integrating and declaring a model

Models are queried through [gRPC](https://grpc.io/), in order to separate the API itself from the model inference, and to support several languages beyond Python through this protocol.

To add a model, you first need to deploy a gRPC service (using the provided `.proto` file and / or the templates provided in `src/serve`). Once your model is live, you only have to add it to the `models.toml` configuration file. For instance, let's say you've locally deployed a [llama.cpp] model available on port 50051, just add:

```toml
[llama-7B-4b]
    [llama-7B-4b.metadata]
        owned_by    = 'Meta'
        permission  = []
        description = 'C++ implementation of Meta's LlaMA model, 7B parameters, 4-bit quantization'
    [llama-7B-4b.network]
        url = 'localhost:50051'
```

## Contribute

This is very much work in progress and ugly, so let me know if you want to help. PR, issues, documentation, cool logo, all the usual candidates are welcome.

## License

MIT License

Copyright (c) 2022 Louis HENAULT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
