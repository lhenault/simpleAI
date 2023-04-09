"""The Python implementation of the gRPC server."""

from concurrent import futures
import logging

import grpc
from . import llm_pb2
from . import llm_pb2_grpc

from .model import LanguageModel


class LanguageModelServicer(llm_pb2_grpc.LanguageModelServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self, model=LanguageModel()) -> None:
        super().__init__()
        self.model = model

    def Complete(self, request, context):
        predicted = self.model.complete(
            prompt=request.prompt,
            suffix=request.suffix,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            logprobs=request.logprobs,
            echo=request.echo,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequence_penalty=request.frequence_penalty,
            best_of=request.best_of,
            logit_bias=request.logit_bias,
        )
        return llm_pb2.Completions(reply=predicted)

    def StreamComplete(self, request, context):
        predicted_stream = self.model.stream_complete(
            prompt=request.prompt,
            suffix=request.suffix,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            logprobs=request.logprobs,
            echo=request.echo,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequence_penalty=request.frequence_penalty,
            best_of=request.best_of,
            logit_bias=request.logit_bias,
        )
        yield from map(lambda predicted: llm_pb2.Completions(reply=predicted), predicted_stream)


def serve(address="[::]:50051", model_servicer=LanguageModelServicer(), max_workers=10):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    llm_pb2_grpc.add_LanguageModelServicer_to_server(model_servicer, server)
    server.add_insecure_port(address=address)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("address", type=str, default="[::]:50051")
    args = parser.parse_args()

    serve(address=args.address)
