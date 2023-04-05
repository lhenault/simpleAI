from dataclasses import dataclass

import torch
from lib_raven import chat, get_model
from simple_ai.api.grpc.chat.server import LanguageModel
from simple_ai.utils import format_chat_log


@dataclass(unsafe_hash=True)
class RavenRWKVModel(LanguageModel):
    gpu_id: int = 0
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device("cpu")
    model, pipeline = get_model()

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

        output = chat(
            prompt,
            self.model,
            self.pipeline,
            token_count=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        )

        output = "".join(output)

        return [{"role": "raven", "content": output}]
