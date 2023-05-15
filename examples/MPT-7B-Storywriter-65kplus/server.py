import logging

from model import CompletionModel as Model
from simple_ai.api.grpc.completion.server import (
    LanguageModelServicer as CompletionServicer,
    serve,
)

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", type=str, default="[::]:50051")
    args = parser.parse_args()

    logging.info(f"Starting gRPC server on {args.address}")
    completion_servicer = CompletionServicer(model=Model())

    serve(
        address=args.address,
        model_servicer=completion_servicer,
    )
