from dataclasses import dataclass

from ....utils import TupleOrList


@dataclass(unsafe_hash=True)
class LanguageModel:
    def chat(self, chatlog: TupleOrList[TupleOrList[str]] = (), *args, **kwargs) -> list:
        raise NotImplementedError

    def stream(self, chatlog: TupleOrList[TupleOrList[str]] = (), *args, **kwargs) -> list:
        raise NotImplementedError
