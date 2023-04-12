from dataclasses import dataclass

import torch
from get_models import MODEL_ID
from simple_ai.api.grpc.completion.server import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(unsafe_hash=True)
class OpenChatModel(LanguageModel):
    gpu_id: int = 0
    device = torch.device("cuda", gpu_id)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).half()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def __post_init__(self):
        self.model.to(self.device)

    def complete(
        self,
        prompt: str = "<|endoftext|>",
        max_tokens: int = 512,
        temperature: float = 0.6,
        *args,
        **kwargs,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=40,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output = self.tokenizer.batch_decode(outputs)[0]

        # remove the context from the output
        output = output[len(prompt) :]

        return output
