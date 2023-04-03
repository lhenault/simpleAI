import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from get_models import MODEL_ID

@dataclass(unsafe_hash=True)
class SentenceTransformerModel:
    model = SentenceTransformer(MODEL_ID)
    def embed(self, 
        inputs: list=[],
    ) -> list:
        logging.info(f'Processing inputs : {inputs}')
        embeddings = self.model.encode(inputs)
        logging.info(f'Successfully computed embeddings (shape : {embeddings.shape}) for inputs : {inputs}')
        return embeddings.tolist()
