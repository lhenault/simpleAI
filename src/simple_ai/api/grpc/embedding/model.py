from typing import List
from dataclasses import dataclass

from ....utils import TupleOrList

@dataclass(unsafe_hash=True)
class LanguageModel:
    def embed(
        self,
        inputs: TupleOrList = (),
    ) -> TupleOrList[TupleOrList]:
        raise NotImplementedError
