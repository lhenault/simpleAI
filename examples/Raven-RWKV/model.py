import json
import logging
import re
from dataclasses import dataclass

import lib_raven
import torch
from simple_ai.api.grpc.chat.server import LanguageModel


def format_chat_log(chat: list[dict[str, str]] = dict()) -> str:
    raw_chat_text = ""
    for item in chat:
        if item["role"] not in ("user", "assistant"):
            continue
        role = "Bob" if item.get("role") == "user" else "Alice"
        content = item.get("content").strip()
        content = re.sub("\n+", "\n", content)

        raw_chat_text += f"{role}: {content}\n\n"
    return raw_chat_text + "Alice:"


@dataclass(unsafe_hash=True)
class RavenRWKVModel(LanguageModel):
    gpu_id: int = 0
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device("cpu")
    model, pipeline = lib_raven.get_model()

    def chat(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        presencePenalty: int = 0.4,
        countPenalty: int = 0.4,
        *args,
        **kwargs,
    ) -> str:
        prompt = format_chat_log(chatlog)
        output = lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            prompt=None,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        )

        output = "".join(output)

        return [{"role": "raven", "content": output}]

    def complete(
        self,
        *args,
        **kwargs,
    ) -> str:
        output = self.stream_complete(*args, **kwargs)
        output = "".join(output)

        return output

    def stream_complete(
        self,
        prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        presencePenalty: int = 0.4,
        countPenalty: int = 0.4,
        stop=None,
        # *args,
        **kwargs,
    ) -> str:
        stop = json.loads(stop)
        output = lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
            stop_words=stop,
        )
        yield from output

    def stream(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        presencePenalty: int = 0.4,
        countPenalty: int = 0.4,
        *args,
        **kwargs,
    ):
        yield [{"role": "raven"}]

        stop_words = ["\n\nBob:", "\n\nAlice:"]

        prompt = format_chat_log(chatlog)
        first = True
        for delta in lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            prompt=None,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
            stop_words=stop_words,
        ):
            clean_delta = delta
            if first:
                clean_delta = delta[1:]  ## remove leading whitespace in completion
                first = False
            yield [{"content": clean_delta}]

    def embed(
        self,
        inputs: list = [],
    ) -> list:
        logging.info(f"Processing inputs : {inputs}")
        embeddings = lib_raven.embedding(inputs, self.model, self.pipeline)
        logging.info(
            f"Successfully computed embeddings (shape : {embeddings.shape}) for inputs : {inputs}"
        )
        return embeddings.tolist()
