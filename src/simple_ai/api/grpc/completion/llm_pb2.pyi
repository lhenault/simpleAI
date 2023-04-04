from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Completions(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: str
    def __init__(self, reply: _Optional[str] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = [
        "best_of",
        "echo",
        "frequence_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "prompt",
        "stop",
        "stream",
        "suffix",
        "temperature",
        "top_p",
    ]
    BEST_OF_FIELD_NUMBER: _ClassVar[int]
    ECHO_FIELD_NUMBER: _ClassVar[int]
    FREQUENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    best_of: int
    echo: bool
    frequence_penalty: float
    logit_bias: str
    logprobs: int
    max_tokens: int
    n: int
    presence_penalty: float
    prompt: str
    stop: str
    stream: bool
    suffix: str
    temperature: float
    top_p: float
    def __init__(
        self,
        prompt: _Optional[str] = ...,
        suffix: _Optional[str] = ...,
        max_tokens: _Optional[int] = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        n: _Optional[int] = ...,
        stream: bool = ...,
        logprobs: _Optional[int] = ...,
        echo: bool = ...,
        stop: _Optional[str] = ...,
        presence_penalty: _Optional[float] = ...,
        frequence_penalty: _Optional[float] = ...,
        best_of: _Optional[int] = ...,
        logit_bias: _Optional[str] = ...,
    ) -> None: ...
