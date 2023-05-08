import logging
from dataclasses import dataclass
from threading import Thread

import torch
from get_models import MODEL_ID
from simple_ai.api.grpc.chat.server import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def preprocess_role(role: str):
    if role == "user":
        return "prompter"
    if role == "system":
        return "assistant"
    return role


def format_chat_log(chat: list[dict[str, str]] = dict()) -> str:
    raw_chat_text = ""
    for item in chat:
        raw_chat_text += (
            f"<|{preprocess_role(item.get('role'))}|>{item.get('content')}<|endoftext|>"
        )
    return raw_chat_text + "<|assistant|>"


@dataclass(unsafe_hash=True)
class OpenAssistantModel(LanguageModel):
    gpu_id: int = 0
    device = torch.device("cuda", gpu_id)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto").half()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def chat(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        role: str = "system",
        *args,
        **kwargs,
    ) -> str:
        logging.info(f"Preprocessing chatlog:\n{chatlog}")
        prompt = format_chat_log(chatlog)

        logging.info(f"Input prompt:\n{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output = self.tokenizer.batch_decode(outputs)[0]
        logging.info(f"Model output:\n{output}")

        # Remove the context from the output
        output = output[len(prompt) :]

        # Stop at "<|endoftext|>"
        if "<|endoftext|>" in output:
            output = output.split("<|endoftext|>")[0]
        return [{"role": role, "content": output}]

    def stream(
        self,
        chatlog: list[list[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.9,
        top_p: int = 0.5,
        role: str = "system",
        *args,
        **kwargs,
    ):
        yield [{"role": role}]

        logging.info(f"Preprocessing chatlog:\n{chatlog}")
        prompt = format_chat_log(chatlog)

        logging.info(f"Input prompt:\n{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate stream, yield delta
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        logging.info("Output:")
        for delta in streamer:
            if delta:
                if "<|endoftext|>" in delta:
                    logging.info(delta)
                    yield [{"content": delta.split("<|endoftext|>")[0]}]
                    break
                logging.info(delta)
                yield [{"content": delta}]
