from typing import List
from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class LanguageModel:
    def embed(
        self,
        inputs: list = [],
    ) -> List[list]:
        # TODO : implement method for your LLM
        return [[]]
