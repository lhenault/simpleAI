# Python server for simpleAI gRPC service

## Integrate your model

If you want to deploy your custom model with Python, you can:

1. Copy this directory somewhere
2. Use the provided `LanguageModelServer` class in `model.py`, and implement the available methods.
3. Follow the instructions to start the server and declare your model in the API (see documentation).

## Start service

Just launch the server with `python server.py`.

## Build gRPC from proto (optional)

To (re)generate the needed files from the `.proto`, use:

```bash
python -m grpc_tools.protoc -I../../../protos --python_out=. --pyi_out=. --grpc_python_out=. ../../../protos/llm_embed.proto
```
