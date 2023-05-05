import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .api.grpc.chat import client as chat_client
from .api.grpc.completion import client as lm_client
from .api.grpc.embedding import client as embed_client
from .api_models import ModelConfig, ModelInterfaceTypes, ModelTaskTypes

path = pathlib.Path(os.environ.get("SIMPLEAI_CONFIG_PATH", "models.toml"))
with path.open(mode="rb") as fp:
    MODELS_ZOO = tomllib.load(fp)


@dataclass(unsafe_hash=True)
class RpcCompletionLanguageModel:
    name: str
    url: str

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
        return lm_client.run(
            url=self.url,
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            best_of=best_of,
            logit_bias=logit_bias,
        )

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
        yield from lm_client.run_stream(
            url=self.url,
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            best_of=best_of,
            logit_bias=logit_bias,
        )


@dataclass(unsafe_hash=True)
class RpcEmbeddingLanguageModel:
    name: str
    url: str

    def embed(
        self,
        inputs: Union[str, list] = "",
    ) -> str:
        return embed_client.run(url=self.url, inputs=inputs)


@dataclass(unsafe_hash=True)
class RpcChatLanguageModel:
    name: str
    url: str

    def chat(
        self,
        messages: list[list[str]] = [],
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Union[str, list] = "",
        presence_penalty: float = 0.0,
        frequence_penalty: float = 0.0,
        logit_bias: dict = {},
    ) -> str:
        return chat_client.run(
            url=self.url,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            logit_bias=logit_bias,
        )

    def stream(
        self,
        messages: list[list[str]] = [],
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Union[str, list] = "",
        presence_penalty: float = 0.0,
        frequence_penalty: float = 0.0,
        logit_bias: dict = {},
    ) -> str:
        yield from chat_client.run_stream(
            url=self.url,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            presence_penalty=presence_penalty,
            frequence_penalty=frequence_penalty,
            logit_bias=logit_bias,
        )


def select_model_type(model_interface: str, task: str):
    if model_interface == "gRPC":
        if task == "embed":
            return RpcEmbeddingLanguageModel
        if task == "chat":
            return RpcChatLanguageModel
        if task == "complete":
            return RpcCompletionLanguageModel
        raise ValueError(f"`task` value must be in {ModelTaskTypes.list()}, got `{task}` instead`.")
    return ValueError(
        f"`model_interface` value must be in {ModelInterfaceTypes.list()} `gRPC`, got"
        f" `{model_interface}` instead."
    )


def get_model(model_id: str, task: ModelTaskTypes, metadata: dict = MODELS_ZOO):
    if model_id in metadata.keys():
        model_config = ModelConfig(**metadata.get(model_id)).network
        return select_model_type(model_config.type, task)(name=model_id, url=model_config.url)
    else:
        raise ValueError(f"Cannot find model named `{model_id}` in configuration.")


def list_models(metadata: dict = MODELS_ZOO) -> list:
    return dict(
        data=[
            {"id": key, **meta.get("metadata"), "object": "model"} for key, meta in metadata.items()
        ],
        object="list",
    )


def get_model_infos(model_id, metadata: dict = MODELS_ZOO) -> list:
    if model_id in metadata.keys():
        return {"id": model_id, **metadata.get(model_id).get("metadata")}
    return {}
