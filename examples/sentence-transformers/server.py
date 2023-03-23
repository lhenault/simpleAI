import logging

from simple_ai.serve.python.embedding.server import serve, LanguageModelServicer

from model import SentenceTransformerModel as Model


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', type=str, default='[::]:50051')
    args = parser.parse_args()

    logging.info(f'Starting gRPC server on {args.address}')
    
    model_servicer = LanguageModelServicer(model=Model())
    serve(address=args.address, model_servicer=model_servicer)
