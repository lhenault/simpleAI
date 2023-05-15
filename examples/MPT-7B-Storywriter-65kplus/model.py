import logging
from dataclasses import dataclass
from threading import Event, Thread
from typing import Optional, Union

import torch
from get_models import MODEL_ID
from simple_ai.api.grpc.chat.server import LanguageModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


@dataclass
class StopOnTokens(StoppingCriteria):
    stop_token_ids: Optional[Union[list, tuple]]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# Model:
# - Has been trained with a sequence length of 2048
# - Fine tuned with a sequence length of 65536 (thanks to ALiBi)
# - Can use a higher sequence length at inferernce (thanks to ALiBi)
# Note: ALiBi is only implemented with torch and triton attention.
MAX_SEQUENCE_LENGTH = 65536


@dataclass(unsafe_hash=True)
class CompletionModel(LanguageModel):
    gpu_id: int = 0
    device = torch.device("cuda", gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, model_max_length=MAX_SEQUENCE_LENGTH, truncation_side="left"
    )
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Attention: either "torch" (default), "flash" or "triton"
    config.attn_config["attn_impl"] = "torch"
    config.update({"max_seq_len": MAX_SEQUENCE_LENGTH})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    def complete(
        self,
        prompt: str = "<|endoftext|>",
        max_tokens: int = 512,
        temperature: float = 0.6,
        end_of_text: Union[str, list, tuple] = ("<|endoftext|>",),
        *args,
        **kwargs,
    ) -> str:
        try:
            if isinstance(end_of_text, str):
                end_of_text = (end_of_text,)

            logging.info(f"Input prompt:\n{prompt}")
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2
            ).to(self.model.device)

            # Use Torch's Flash attention
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                output = self.tokenizer.batch_decode(outputs)[0]
                logging.info(f"Model output:\n{output}")

                # Remove the context from the output
                output = output[len(prompt) :]

                # Stop if end of text
                for item in end_of_text:
                    if item in output:
                        output = output.split(item)[0]
                    break

                # Avoid issues with GPU vRAM
                del inputs
                torch.cuda.empty_cache()

                return output
        except Exception as ex:
            logging.exception(ex)

        return ""

    def stream_complete(
        self,
        prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.6,
        end_of_text: Union[str, list, tuple] = ("<|endoftext|>",),
        *args,
        **kwargs,
    ) -> str:
        try:
            if isinstance(end_of_text, str):
                end_of_text = (end_of_text,)

            stop = StopOnTokens(stop_token_ids=self.tokenizer.convert_tokens_to_ids(end_of_text))

            if isinstance(end_of_text, str):
                end_of_text = (end_of_text,)

            logging.info(f"Input prompt:\n{prompt}")

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2
            ).to(self.model.device)

            # Use Torch's Flash attention
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                # Generate stream, yield delta
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([stop]),
                )

                stream_complete = Event()

                def generate_and_signal_complete(generation_kwargs=generation_kwargs):
                    self.model.generate(**generation_kwargs)
                    stream_complete.set()

                thread = Thread(target=generate_and_signal_complete)
                thread.start()

                logging.info("[Completions/Streaming] Output:")
                ended = False
                for delta in streamer:
                    if delta:
                        for item in end_of_text:
                            if item in delta:
                                logging.info(delta)
                                yield delta.split(item)[0]
                                ended = True
                                break
                        if ended:
                            break
                        logging.info(delta)
                        yield delta

                thread.join(timeout=60)

                # Avoid issues with GPU vRAM
                del streamer
                del inputs
                torch.cuda.empty_cache()

        except Exception as ex:
            logging.exception(ex)

        return
