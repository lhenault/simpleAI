import logging
from dataclasses import dataclass

import lib_raven
import torch
from simple_ai.api.grpc.chat.server import LanguageModel
from simple_ai.utils import format_chat_log


def endOverlap(a, b):
    for i in range(1, len(a) + 1):
        if b.startswith(a[-i:]):
            return i
    return 0


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
        output = lib_raven.chat(
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
        *args,
        **kwargs,
    ) -> str:
        output = lib_raven.complete(
            prompt,
            self.model,
            self.pipeline,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
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

        stop_words = set([f"{message['role']}:" for message in chatlog])

        prompt = format_chat_log(chatlog)
        chunk = ""
        for delta in lib_raven.chat(
            prompt,
            self.model,
            self.pipeline,
            prompt=None,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        ):
            chunk = chunk + delta
            longest_stopword = max(map(len, stop_words))

            if start_idx := max(map(lambda stop_word: endOverlap(chunk, stop_word), stop_words)):
                if start_idx > longest_stopword:
                    start_idx = longest_stopword  # can no longer be a stopword so cut it down
                good, chunk = chunk[:-start_idx], chunk[-start_idx:]

                if good:
                    yield [{"content": good}]

                if any(map(lambda stop_word: chunk.startswith(stop_word), stop_words)):
                    return
                continue

            # if start_idx:=max(map(lambda stop_word: endOverlap(stop_word, chunk), stop_words))>0:

            yield [{"content": chunk}]
            chunk = ""

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
