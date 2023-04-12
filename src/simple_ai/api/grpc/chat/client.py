"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

from typing import List, Union

import grpc
from google.protobuf.json_format import MessageToDict

from . import llm_chat_pb2
from . import llm_chat_pb2_grpc


def get_chatlog(stub, chatlog):
    response = stub.Chat(chatlog)
    results = []
    for message in response.messages:
        results.append(MessageToDict(message))
    return results


def run(
    url: str = "localhost:50051",
    messages: List[List[str]] = [],
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stream: bool = False,
    stop: Union[str, list] = "",
    presence_penalty: float = 0.0,
    frequence_penalty: float = 0.0,
    logit_bias: dict = {},
):
    with grpc.insecure_channel(url) as channel:
        stub = llm_chat_pb2_grpc.LanguageModelStub(channel)
        grpc_chatlog = llm_chat_pb2.ChatLogInput(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=str(stop),
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            logit_bias=str(logit_bias),
        )
        for role, content in messages:
            grpc_chat = llm_chat_pb2.Chat(role=role, content=content)
            grpc_chatlog.messages.append(grpc_chat)
        return get_chatlog(stub, grpc_chatlog)


def stream_chatlog(stub, chatlog):
    responses = stub.Stream(chatlog)

    try:
        yield from map(lambda x: [MessageToDict(x_i) for x_i in x.messages], responses)
    finally:
        responses.cancel()


def run_stream(
    url: str = "localhost:50051",
    messages: List[List[str]] = [],
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stream: bool = False,
    stop: Union[str, list] = "",
    presence_penalty: float = 0.0,
    frequence_penalty: float = 0.0,
    logit_bias: dict = {},
):
    with grpc.insecure_channel(url) as channel:
        stub = llm_chat_pb2_grpc.LanguageModelStub(channel)
        grpc_chatlog = llm_chat_pb2.ChatLogInput(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=str(stop),
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            logit_bias=str(logit_bias),
        )

        for role, content in messages:
            grpc_chat = llm_chat_pb2.Chat(role=role, content=content)
            grpc_chatlog.messages.append(grpc_chat)

        yield from stream_chatlog(stub, grpc_chatlog)


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="[::]:50051")
    args = parser.parse_args()

    res = run(messages=[["user", "hello"] for _ in range(5)])
    print(res)
