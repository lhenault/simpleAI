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

class Embedding(_message.Message):
    __slots__ = ["feature"]
    FEATURE_FIELD_NUMBER: _ClassVar[int]
    feature: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, feature: _Optional[_Iterable[float]] = ...) -> None: ...

class ListOfEmbeddings(_message.Message):
    __slots__ = ["embedding"]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedCompositeFieldContainer[Embedding]
    def __init__(
        self, embedding: _Optional[_Iterable[_Union[Embedding, _Mapping]]] = ...
    ) -> None: ...

class Sentences(_message.Message):
    __slots__ = ["inputs"]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, inputs: _Optional[_Iterable[str]] = ...) -> None: ...
