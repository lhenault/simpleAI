"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

from typing import Union

import grpc
from google.protobuf.json_format import MessageToDict


from . import llm_pb2
from . import llm_pb2_grpc


def get_one_completion(stub, message):
    response = stub.Complete(message)
    return response.reply


def get_completion(stub, message):
    return get_one_completion(stub=stub, message=message)


def run(
    url: str = "localhost:50051",
    prompt: str = "<|endoftext|>",
    suffix: str = "",
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stream: bool = False,
    logprobs: int = 0,
    echo: bool = False,
    stop: Union[str, list] = "",
    presence_penalty: float = 0.0,
    frequence_penalty: float = 0.0,
    best_of: int = 0,
    logit_bias: dict = {},
):
    with grpc.insecure_channel(url) as channel:
        stub = llm_pb2_grpc.LanguageModelStub(channel)
        message = llm_pb2.Message(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            logprobs=logprobs,
            echo=echo,
            stop=str(stop),
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            best_of=best_of,
            logit_bias=str(logit_bias),
        )
        return [get_completion(stub, message)]


def stream_completions(stub, message):
    responses = stub.StreamComplete(message)
    try:
        ## TODO x.reply should be a list of strings. wrapping in list here for now
        yield from map(lambda x: [x.reply], responses)
    finally:
        responses.cancel()


def run_stream(
    url: str = "localhost:50051",
    prompt: str = "<|endoftext|>",
    suffix: str = "",
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stream: bool = False,
    logprobs: int = 0,
    echo: bool = False,
    stop: Union[str, list] = "",
    presence_penalty: float = 0.0,
    frequence_penalty: float = 0.0,
    best_of: int = 0,
    logit_bias: dict = {},
):
    with grpc.insecure_channel(url) as channel:
        stub = llm_pb2_grpc.LanguageModelStub(channel)
        message = llm_pb2.Message(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            logprobs=logprobs,
            echo=echo,
            stop=str(stop),
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            best_of=best_of,
            logit_bias=str(logit_bias),
        )
        yield from stream_completions(stub, message)
