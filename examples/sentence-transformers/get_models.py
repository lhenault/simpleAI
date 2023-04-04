import logging

from sentence_transformers import SentenceTransformer

MODEL_ID = "all-MiniLM-L6-v2"

if __name__ == "__main__":
    try:
        model = SentenceTransformer(MODEL_ID)
    except Exception as ex:
        logging.exception(f"Could not retrieve {MODEL_ID}: {ex}")
