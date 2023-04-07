import gc
import logging
import os
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

# if RWKV_CUDA_ON='1' then use CUDA kernel for seq mode (much faster)
# these settings must be configured before attempting to import rwkv
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
from rwkv.model import RWKV  # noqa: E402
from rwkv.utils import PIPELINE, PIPELINE_ARGS  # noqa: E402

logger = logging.getLogger(__file__)

nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 4096

models = {
    "raven-14b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-14B-v6-Eng-20230401-ctx4096",
    },
    "raven-7b-ctx4096": {
        "repo_id": "BlinkDL/rwkv-4-raven",
        "title": "RWKV-4-Raven-7B-v7-Eng-20230404-ctx4096",
    },
    "raven-7b-ctx1024": {
        "repo_id": "BlinkDL/rwkv-4-pile-7b",
        "title": "RWKV-4-Pile-7B-Instruct-test4-20230326",
    },
    "rwkv-4-pile-169m": {
        "repo_id": "BlinkDL/rwkv-4-pile-169m",
        "title": "RWKV-4-Pile-169M-20220807-8023",
    },
}

model = "raven-14b-ctx4096"
# model = "raven-7b-ctx1024"
# model = "raven-7b-ctx4096"
# model = "rwkv-4-pile-169m"
model_params = models[model]


def fetch_tokenizer(tokenizer_path: Path):
    url = "https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B/raw/main/20B_tokenizer.json"
    tokenizer_path.parent.mkdir(exist_ok=True)

    response = requests.get(url)
    tokenizer_path.write_bytes(response.content)


def get_model():
    tokenizer_path = Path(__file__).parent / "20B_tokenizer.json"
    if not tokenizer_path.exists():
        fetch_tokenizer(tokenizer_path)

    model_path = hf_hub_download(
        repo_id=model_params["repo_id"], filename=f"{model_params['title']}.pth"
    )

    model = RWKV(model=model_path, strategy="cuda fp16i8 *0+ -> cpu fp32 *1")  # stream mode
    # model = RWKV(model=model_path, strategy="cuda fp16i8 *10 -> cuda fp16")

    pipeline = PIPELINE(model, str(tokenizer_path))

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

    ctx = generate_prompt(instruction, prompt)

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
    inputs: list[str],
    model,
    pipeline,
    prompt="",
    token_count=200,
    temperature=1.0,
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

    raise state
