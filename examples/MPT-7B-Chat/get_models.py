import logging

MODEL_ID = "mosaicml/mpt-7b-chat"

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    for repo_id in (MODEL_ID,):
        try:
            snapshot_download(repo_id)
        except Exception as ex:
            logging.exception(f"Could not retrieve {repo_id}: {ex}")

    try:
        move_cache()
    except Exception as ex:
        logging.exception(f"Could not migrate cache: {ex}")
