from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

MODEL = "rwkv-4-pile-169m"

TOKENIZER_PATH = Path(__file__).parent / "20B_tokenizer.json"
models = {
    "raven-14b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-14B-v8-Eng-20230408-ctx4096",
    },
    "raven-7b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-7B-v7-Eng-20230404-ctx4096",
    },
    "raven-7b-ctx1024": {
        "repo_id": "BlinkDL/rwkv-4-pile-7b",
        "title": "RWKV-4-Pile-7B-Instruct-test4-20230326",
    },
    "rwkv-4-pile-169m": {
        "repo_id": "BlinkDL/rwkv-4-pile-169m",
        "title": "RWKV-4-Pile-169M-20220807-8023",
    },
}


def fetch_tokenizer(tokenizer_path: Path):
    url = "https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B/raw/main/20B_tokenizer.json"
    tokenizer_path.parent.mkdir(exist_ok=True)

    response = requests.get(url)
    tokenizer_path.write_bytes(response.content)


def get_model_path(model="rwkv-4-pile-169m"):
    tokenizer_path = Path(__file__).parent / "20B_tokenizer.json"
    if not tokenizer_path.exists():
        fetch_tokenizer(tokenizer_path)

    model_params = models[model]

    model_path = hf_hub_download(
        repo_id=model_params["repo_id"], filename=f"{model_params['title']}.pth"
    )

    return model_path


if __name__ == "__main__":
    get_model_path(MODEL)
