import argparse
import shutil
from pathlib import Path

import uvicorn


def serve_app(host="127.0.0.1", port=8080, **kwargs):
    from . import server

    uvicorn.run(app=server.app, host=host, port=port)


def init_app(path="./", **kwargs):
    shutil.copy(
        src=Path(Path(__file__).parent.absolute(), "models.toml.template"),
        dst=Path(path, "models.toml"),
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Init config args
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--path", default="./", type=str)
    init_parser.set_defaults(func=init_app)

    # Serving args
    serving_parser = subparsers.add_parser("serve")
    serving_parser.add_argument("--host", default="127.0.0.1", type=str)
    serving_parser.add_argument("--port", default=8080, type=int)
    serving_parser.set_defaults(func=serve_app)

    # Parse, call the appropriate function
    args = parser.parse_args()
    args.func(**args.__dict__)


if __name__ == "__main__":
    main()
