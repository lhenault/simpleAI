from dataclasses import dataclass

import lib_raven
import torch
from simple_ai.api.grpc.chat.server import LanguageModel
from simple_ai.utils import format_chat_log


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

        if len(chatlog) == 1:
            instruction = chatlog[0]["content"]
        if len(chatlog) == 2:
            instruction = chatlog[0]["content"]
            prompt = chatlog[1]["content"]
        else:
            raise ValueError("Instruct tuned only")

        output = lib_raven.chat(
            instruction,
            self.model,
            self.pipeline,
            prompt=prompt,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        )

        output = "".join(output)

        return [{"role": "raven", "content": output}]

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
        yield [{"role": "raven"} for i in chatlog]

        prompt = None
        if len(chatlog) == 1:
            instruction = chatlog[0]["content"]
        elif len(chatlog) == 2:
            instruction = chatlog[0]["content"]
            prompt = chatlog[1]["content"]
        else:
            raise ValueError("Instruct tuned only")

        for delta in lib_raven.chat(
            instruction,
            self.model,
            self.pipeline,
            prompt=prompt,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        ):
            yield [{"content": delta}]
