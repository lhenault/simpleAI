# SimpleAI

A self-hosted alternative to the not-so-open AI API. It is focused on replicating the main endpoints for LLM:

- [x] Text completion (`/completions/`)
    - ✔️ Non `stream` responses
    - ✔️ `stream` responses
- [x] Chat (`/chat/completions/`) [ [example](/examples/GPT-NeoXT-Chat-Base-20B/) ]
    - ✔️ Non `stream` responses
    - ✔️ `stream` responses
- [x] Edits (`/edits/`) [ [example](/examples/alpaca-lora-7B/) ]
- [x] Embeddings (`/embeddings/`) [ [example](/examples/sentence-transformers) ]
- [ ] Not supported (yet): `images`, `audio`, `files`, `fine-tunes`, `moderations`

It allows you to experiment with competing approaches quickly and easily.

![Overview](/assets/overview.jpg)

*Why this project?*

Well first of all it's a fun little project, and perhaps a better use of my time than watching some random dog videos on Reddit or YouTube. I also believe it can be a great way to:

- experiment with new models and not be too dependent on a specific API provider,
- create benchmarks to decide which approach works best for you,
- handle some specific use cases where you cannot fully rely on an external service, without the need of re-writing everything

If you find interesting use cases, feel free to share your experience.

## Installation

On a machine with Python 3.9+:

- [Latest] From source:

```bash
pip install git+https://github.com/lhenault/simpleAI 
```

- From Pypi:

```bash
pip install simple_ai_server
```

## Setup

Start by creating a configuration file to declare your models:

```bash
simple_ai init
```

It should create `models.toml`, where you declare your different models (see how below). Then start the server with:

```bash
simple_ai serve [--host 127.0.0.1] [--port 8080]
```

You can then see the docs and try it [there](http://127.0.0.1:8080/docs#/).

## Integrating and declaring a model

### Model integration

Models are queried through [gRPC](https://grpc.io/), in order to separate the API itself from the model inference, and to support several languages beyond Python through this protocol.

To expose for instance an embedding model in Python, you simply have to import a few things, and implements the `.embed()` method of your `EmbeddingModel` class:

```python
import logging
from dataclasses import dataclass

from simple_ai.api.grpc.embedding.server import serve, LanguageModelServicer

@dataclass(unsafe_hash=True)
class EmbeddingModel:
    def embed(self, 
        inputs: list=[],
    ) -> list:
        # TODO : implements the embed method
        return [[]]

if __name__ == '__main__':   
    model_servicer = LanguageModelServicer(model=EmbeddingModel())
    serve(address='[::]:50051', model_servicer=model_servicer)
```

For a completion task, follow the same logic, but import `from simple_ai.api.grpc.completion.server` instead, and implements a `complete` method.

### Declaring a model

To add a model, you first need to deploy a gRPC service (using the provided `.proto` file and / or the tools provided in `src/api/`). Once your model is live, you only have to add it to the `models.toml` configuration file. For instance, let's say you've locally deployed a [llama.cpp](https://github.com/ggerganov/llama.cpp) model available on port 50051, just add:

```toml
[llama-7B-4b]
    [llama-7B-4b.metadata]
        owned_by    = 'Meta / ggerganov'
        permission  = []
        description = 'C++ implementation of LlaMA model, 7B parameters, 4-bit quantization'
    [llama-7B-4b.network]
        url = 'localhost:50051'
        type = 'gRPC'
```

You can see see and try of the provided examples in `examples/` directory (might require GPU).

## Usage

Thanks to [the Swagger UI](https://github.com/swagger-api/swagger-ui), you can see and try the different endpoints [here](http://127.0.0.1:8080/docs#/):

![Example query with cUrl](/assets/docs-example.jpg)

Or you can directly use the API with the tool of your choice.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8080/edits/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "alpaca-lora-7B",
  "instruction": "Make this message nicer and more formal",
  "input": "This meeting was useless and should have been a bloody email",
  "top_p": 1,
  "n": 1,
  "temperature": 1,
  "max_tokens": 256
}'
```

It's also compatible with [OpenAI python client](https://github.com/openai/openai-python):

```python
import openai

# Put anything you want in `API key`
openai.api_key = 'Free the models'

# Point to your own url
openai.api_base = "http://127.0.0.1:8080"

# Do your usual things, for instance a completion query:
print(openai.Model.list())
completion = openai.Completion.create(model="llama-7B", prompt="Hello everyone this is")
```

## Contribute

This is very much work in progress and far from being perfect, so let me know if you want to help. PR, issues, documentation, cool logo, all the usual candidates are welcome.

### Development Environment

In order for the following steps to work it is required to have make and poetry installed on your system.

To install the development environment run:
```
make install-dev 
```

This will install all dev dependencies as well as configure your pre-commit helpers.

