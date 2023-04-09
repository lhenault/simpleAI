from typing import Union
from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class LanguageModel:
    def complete(
        self,
        prompt: str = "<|endoftext|>",
        suffix: str = "",
        max_tokens: int = 7,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        logprobs: int = 0,
        echo: bool = False,
        stop: Union[str, list] = "",
        presence_penalty: float = 0.0,
        frequence_penalty: float = 0.0,
        best_of: int = 0,
        logit_bias: dict = {},
    ) -> str:
        # TODO : implement method for your LLM
        return "QWERTYUIOP\nASDFGHJKL\nZXCVBNM"

    def stream_complete(
        self,
        prompt: str = "<|endoftext|>",
        suffix: str = "",
        max_tokens: int = 7,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        logprobs: int = 0,
        echo: bool = False,
        stop: Union[str, list] = "",
        presence_penalty: float = 0.0,
        frequence_penalty: float = 0.0,
        best_of: int = 0,
        logit_bias: dict = {},
    ) -> str:
        # TODO : implement method for your LLM
        return "QWERTYUIOP\nASDFGHJKL\nZXCVBNM".split("\n")
