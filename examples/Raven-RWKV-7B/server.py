import logging
from concurrent import futures

import grpc
from model import RavenRWKVModel as Model
from simple_ai.api.grpc.chat.server import (
    LanguageModelServicer as ChatServicer,
    llm_chat_pb2_grpc,
)
from simple_ai.api.grpc.completion.server import (
    LanguageModelServicer as CompletionServicer,
    llm_pb2_grpc,
)
from simple_ai.api.grpc.embedding.server import (
    LanguageModelServicer as EmbeddingServicer,
    llm_embed_pb2_grpc,
)


def serve(
    address="[::]:50051",
    chat_servicer=None,
    embedding_servicer=None,
    completion_servicer=None,
    max_workers=10,
):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    llm_chat_pb2_grpc.add_LanguageModelServicer_to_server(chat_servicer, server)
    llm_embed_pb2_grpc.add_LanguageModelServicer_to_server(embedding_servicer, server)
    llm_pb2_grpc.add_LanguageModelServicer_to_server(completion_servicer, server)
    server.add_insecure_port(address=address)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", type=str, default="[::]:50051")
    args = parser.parse_args()

    logging.info(f"Starting gRPC server on {args.address}")
    model = Model()
    chat_servicer = ChatServicer(model=Model())
    embedding_servicer = EmbeddingServicer(model=Model())
    completion_servicer = CompletionServicer(model=Model())
    serve(
        address=args.address,
        chat_servicer=chat_servicer,
        embedding_servicer=embedding_servicer,
        completion_servicer=completion_servicer,
    )
