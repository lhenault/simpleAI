from typing import List, Optional, Union

from pydantic import BaseModel


class EmbeddingInput(BaseModel):
    model: str
    input: Union[str, list]
    user: str = ""


class CompletionInput(BaseModel):
    model: str
    prompt: Union[str, List[str]] = "<|endoftext|>"
    suffix: str = ""
    max_tokens: int = 7
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: int = 0
    echo: bool = False
    stop: Optional[Union[str, list]] = ""
    presence_penalty: float = 0.0
    frequence_penalty: float = 0.0
    best_of: int = 0
    logit_bias: dict = {}
    user: str = ""


class ChatCompletionInput(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, list]] = ""
    max_tokens: int = 7
    presence_penalty: float = 0.0
    frequence_penalty: float = 0.0
    logit_bias: Optional[dict] = {}
    user: str = ""


class InstructionInput(BaseModel):
    model: str
    instruction: str
    input: str = ""
    top_p: float = 1.0
    n: int = 1
    temperature: float = 1.0
    max_tokens: int = 256
