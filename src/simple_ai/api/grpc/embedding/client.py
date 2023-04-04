"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

from typing import List

import grpc
from google.protobuf.json_format import MessageToDict
from . import llm_embed_pb2
from . import llm_embed_pb2_grpc


def get_embeddings(stub, sentences):
    response = stub.Embed(sentences)
    results = []
    for message in response.embedding:
        results.append(MessageToDict(message).get("feature"))
    return results


def run(
    url: str = "localhost:50051",
    inputs: List[str] = "",
):
    with grpc.insecure_channel(url) as channel:
        stub = llm_embed_pb2_grpc.LanguageModelStub(channel)
        sentences = llm_embed_pb2.Sentences(
            inputs=inputs,
        )
        return get_embeddings(stub, sentences)
