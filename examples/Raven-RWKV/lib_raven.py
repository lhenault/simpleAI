import gc
import logging
import os
from typing import List

import torch
from get_models import MODEL, TOKENIZER_PATH, get_model_path
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

# if RWKV_CUDA_ON='1' then use CUDA kernel for seq mode (much faster)
# these settings must be configured before attempting to import rwkv
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
from rwkv.model import RWKV  # noqa: E402
from rwkv.utils import PIPELINE, PIPELINE_ARGS  # noqa: E402

STRATEGIES = {
    "streaming": "cuda fp16i8 *40+ -> cpu fp32 *1",  # Quite slow, take ~3gb VRAM
    "fp16i8": "cuda fp16i8 *40 -> cpu fp32 *1",  # fits the 14b on a T4, quite fast
    "cpu": "cpu fp32 *1",  # requires a lot of RAM
}

STRATEGY = STRATEGIES["streaming"]

logger = logging.getLogger(__file__)

nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 4096


def get_model():
    model_path = get_model_path(MODEL)

    model = RWKV(
        model=model_path, strategy="cuda fp16i8 *40 -> cuda fp16i8 *0+ -> cpu fp32 *1"
    )  # stream mode w/some static

    pipeline = PIPELINE(model, str(TOKENIZER_PATH))

    return model, pipeline


def generate_prompt(instruction, prompt=None):
    if prompt:
        return f"""Below is an instruction that describes a task, paired with an input"\
        " that provides further context. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Input:
{prompt}

# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that "\
                    "appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""


def chat(
    instruction,
    model,
    pipeline,
    prompt="",
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty=0.1,
    countPenalty=0.1,
):
    args = PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0],
    )  # stop generation whenever you see any token here

    ctx = instruction

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    logger.debug(f"vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}")

    all_tokens = []
    out_last = 0
    out_str = ""
    occurrence = {}
    state = None
    token = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tmp = pipeline.decode(all_tokens[out_last:])
        if "\ufffd" not in tmp:
            out_str += tmp
            yield tmp
            out_last = i + 1
    gc.collect()
    torch.cuda.empty_cache()


def embedding(
    inputs: List[str],
    model,
    pipeline,
    temperature=1.0,  # TODO remove
    top_p=0.7,
    presencePenalty=0.1,
    countPenalty=0.1,
):
    PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0],
    )  # stop generation whenever you see any token here

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    logger.debug(f"vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}")

    context = [pipeline.encode(ctx)[-ctx_limit:] for ctx in inputs]
    _, state = model.forward(context[0], None)
    *_, embedding = state

    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)
    return embedding


def complete(
    prompt,
    model,
    pipeline,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty=0.1,
    countPenalty=0.1,
):
    args = PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0],
    )  # stop generation whenever you see any token here

    ctx = prompt

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    logger.debug(f"vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}")

    all_tokens = []
    out_last = 0
    out_str = ""
    occurrence = {}
    state = None
    token = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tmp = pipeline.decode(all_tokens[out_last:])
        if "\ufffd" not in tmp:
            out_str += tmp
            yield tmp
            out_last = i + 1
    gc.collect()
    torch.cuda.empty_cache()
