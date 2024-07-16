from google.protobuf import struct_pb2 as _struct_pb2
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

class ChatLogInput(_message.Message):
    __slots__ = (
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "presence_penalty",
        "frequence_penalty",
        "logit_bias",
    )
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    FREQUENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[Chat]
    max_tokens: int
    temperature: float
    top_p: float
    n: int
    stream: bool
    stop: str
    presence_penalty: float
    frequence_penalty: float
    logit_bias: str
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
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[Chat]
    def __init__(self, messages: _Optional[_Iterable[_Union[Chat, _Mapping]]] = ...) -> None: ...

class Chat(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: _struct_pb2.Value
    def __init__(
        self,
        role: _Optional[str] = ...,
        content: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...,
    ) -> None: ...
