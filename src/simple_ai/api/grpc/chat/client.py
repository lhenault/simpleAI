"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

from typing import List, Union

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Value, ListValue, Struct

from . import llm_chat_pb2
from . import llm_chat_pb2_grpc


def dict_to_struct(d):
    """Convert a dictionary to Struct."""
    fields = {}
    for k, v in d.items():
        if isinstance(v, dict):
            fields[k] = Value(struct_value=dict_to_struct(v))
        elif isinstance(v, list):
            list_values = []
            for item in v:
                if isinstance(item, dict):
                    list_values.append(Value(struct_value=dict_to_struct(item)))
                elif isinstance(item, str):
                    list_values.append(Value(string_value=item))
                elif isinstance(item, float) or isinstance(item, int):
                    list_values.append(Value(number_value=item))
                elif isinstance(item, bool):
                    list_values.append(Value(bool_value=item))
                else:
                    raise ValueError(f"Unsupported type in list: {type(item)} for key: {k}")
            fields[k] = Value(list_value=ListValue(values=list_values))
        elif isinstance(v, str):
            fields[k] = Value(string_value=v)
        elif isinstance(v, float) or isinstance(v, int):
            fields[k] = Value(number_value=v)
        elif isinstance(v, bool):
            fields[k] = Value(bool_value=v)
        elif v is None:
            fields[k] = Value(null_value=0)
        else:
            raise ValueError(f"Unsupported type: {type(v)} for key: {k}")
    return Struct(fields=fields)


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
            if isinstance(content, str):
                grpc_chat = llm_chat_pb2.Chat(role=role, content=Value(string_value=content))
            else:
                list_value = ListValue(
                    values=[Value(struct_value=dict_to_struct(item)) for item in content]
                )
                grpc_chat = llm_chat_pb2.Chat(role=role, content=Value(list_value=list_value))
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
            if isinstance(content, str):
                grpc_chat = llm_chat_pb2.Chat(role=role, content=Value(string_value=content))
            else:
                list_value = ListValue(
                    values=[Value(struct_value=dict_to_struct(item)) for item in content]
                )
                grpc_chat = llm_chat_pb2.Chat(role=role, content=Value(list_value=list_value))
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
