import logging
from dataclasses import dataclass
from typing import Union

from get_models import ALPACA_ID, LLAMA_ID, TOKENIZER_ID
from peft import PeftModel
from simple_ai.api.grpc.completion.server import LanguageModel
from transformers import GenerationConfig, LLaMAForCausalLM, LLaMATokenizer


@dataclass(unsafe_hash=True)
class AlpacaModel(LanguageModel):
    try:
        tokenizer = LLaMATokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as ex:
        logging.exception(f"Could not load tokenizer: {ex}")
        tokenizer = None
    try:
        model = LLaMAForCausalLM.from_pretrained(
            LLAMA_ID,
            load_in_8bit=True,
            device_map="auto",
        )
    except Exception as ex:
        logging.exception(f"Could not load pretrained LlaMa model: {ex}")
        model = None
    try:
        model = PeftModel.from_pretrained(model, ALPACA_ID)
    except Exception as ex:
        logging.exception(f"Could not load pretrained Peft model: {ex}")
        model = None

    def complete(
        self,
        prompt: str = "<|endoftext|>",
        suffix: str = "",
        max_tokens: int = 512,
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
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            num_beams=4,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_ids = input_ids.cuda()

        output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_tokens,
        )
        results = []
        for sequence in output.sequences:
            results.append(self.tokenizer.decode(sequence).split("### Response:")[1].strip())
        return results[0]
