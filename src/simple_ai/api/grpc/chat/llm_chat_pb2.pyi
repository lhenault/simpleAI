from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class Chat(_message.Message):
    __slots__ = ["content", "role"]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    content: str
    role: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class ChatLogInput(_message.Message):
    __slots__ = [
        "frequence_penalty",
        "logit_bias",
        "max_tokens",
        "messages",
        "n",
        "presence_penalty",
        "stop",
        "stream",
        "temperature",
        "top_p",
    ]
    FREQUENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    frequence_penalty: float
    logit_bias: str
    max_tokens: int
    messages: _containers.RepeatedCompositeFieldContainer[Chat]
    n: int
    presence_penalty: float
    stop: str
    stream: bool
    temperature: float
    top_p: float
    def __init__(
        self,
        messages: _Optional[_Iterable[_Union[Chat, _Mapping]]] = ...,
        max_tokens: _Optional[int] = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        n: _Optional[int] = ...,
        stream: bool = ...,
        stop: _Optional[str] = ...,
        presence_penalty: _Optional[float] = ...,
        frequence_penalty: _Optional[float] = ...,
        logit_bias: _Optional[str] = ...,
    ) -> None: ...

class ChatLogOutput(_message.Message):
    __slots__ = ["messages"]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[Chat]
    def __init__(self, messages: _Optional[_Iterable[_Union[Chat, _Mapping]]] = ...) -> None: ...
