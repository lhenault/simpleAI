import logging

MODEL_ID = "togethercomputer/GPT-NeoXT-Chat-Base-20B"

if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    repo_id = MODEL_ID
    try:
        snapshot_download(repo_id)
    except Exception as ex:
        logging.exception(f"Could not retrieve {repo_id}: {ex}")
