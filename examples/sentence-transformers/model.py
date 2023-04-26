import logging
from dataclasses import dataclass
from typing import Union

from get_models import MODEL_ID
from sentence_transformers import SentenceTransformer


@dataclass(unsafe_hash=True)
class SentenceTransformerModel:
    model = SentenceTransformer(MODEL_ID)

    def embed(
        self,
        inputs: Union[list, tuple] = (),
    ) -> Union[list, tuple]:
        logging.info(f"Processing inputs : {inputs}")
        embeddings = self.model.encode(inputs)
        logging.info(
            f"Successfully computed embeddings (shape : {embeddings.shape}) for inputs : {inputs}"
        )
        return embeddings.tolist()
