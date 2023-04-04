import logging

TOKENIZER_ID = "decapoda-research/llama-7b-hf"
LLAMA_ID = "decapoda-research/llama-7b-hf"
ALPACA_ID = "tloen/alpaca-lora-7b"

if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    for repo_id in (TOKENIZER_ID, LLAMA_ID, ALPACA_ID):
        try:
            snapshot_download(repo_id)
        except Exception as ex:
            logging.exception(f"Could not retrieve {repo_id}: {ex}")
