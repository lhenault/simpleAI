"""The Python implementation of the gRPC server."""

from concurrent import futures

import grpc
from google.protobuf.json_format import MessageToDict

from . import llm_chat_pb2
from . import llm_chat_pb2_grpc
from .model import LanguageModel


class LanguageModelServicer(llm_chat_pb2_grpc.LanguageModelServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self, model=LanguageModel()) -> None:
        super().__init__()
        self.model = model

    def Chat(self, request, context):
        output = self.model.chat(
            chatlog=[MessageToDict(message=message) for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequence_penalty=request.frequence_penalty,
            logit_bias=request.logit_bias,
        )

        grpc_chatlog = llm_chat_pb2.ChatLogOutput()
        for chat in output:
            grpc_chat = llm_chat_pb2.Chat(role=chat.get("role"), content=chat.get("content"))
            grpc_chatlog.messages.append(grpc_chat)
        return grpc_chatlog

    def Stream(self, request, context):
        output = self.model.stream(
            chatlog=[MessageToDict(message=message) for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequence_penalty=request.frequence_penalty,
            logit_bias=request.logit_bias,
        )

        for chat in output:
            grpc_chatlog = llm_chat_pb2.ChatLogOutput()
            for message in chat:
                grpc_chat = llm_chat_pb2.Chat(
                    role=message.get("role"), content=message.get("content")
                )
                grpc_chatlog.messages.append(grpc_chat)
            yield grpc_chatlog


def serve(address="[::]:50051", model_servicer=LanguageModelServicer(), max_workers=10):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    llm_chat_pb2_grpc.add_LanguageModelServicer_to_server(model_servicer, server)
    server.add_insecure_port(address=address)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="[::]:50051")
    args = parser.parse_args()

    serve(address=args.address)
