import logging

MODEL_ID = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    repo_id = MODEL_ID
    try:
        snapshot_download(repo_id)
    except Exception as ex:
        logging.exception(f"Could not retrieve {repo_id}: {ex}")

    try:
        move_cache()
    except Exception as ex:
        logging.exception(f"Could not migrate cache: {ex}")
