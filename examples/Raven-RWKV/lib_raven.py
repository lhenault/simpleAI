import logging
from typing import List

from get_models import MODEL, TOKENIZER_PATH, get_model_path

# if RWKV_CUDA_ON='1' then use CUDA kernel for seq mode (much faster)
# these settings must be configured before attempting to import rwkv
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

STRATEGIES = {
    "streaming": "cuda fp16i8 *40+ -> cpu fp32 *1",  # Quite slow, take ~3gb VRAM
    "fp16i8": "cuda fp16i8 *40 -> cpu fp32 *1",  # fits the 14b on a T4, quite fast
    "cpu": "cpu fp32 *1",  # requires a lot of RAM
}

STRATEGY = STRATEGIES["cpu"]

logger = logging.getLogger(__file__)

ctx_limit = 4096


def get_model():
    model_path = get_model_path(MODEL)

    model = RWKV(model=model_path, strategy=STRATEGY)  # stream mode w/some static

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


def complete(
    instruction,
    model,
    pipeline: PIPELINE,
    prompt="",
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty=0.1,
    countPenalty=0.1,
    stop_words=None,
):
    args = PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0],
        stop_words=stop_words,
    )  # stop generation whenever you see any token here

    for delta in pipeline.igenerate(ctx=instruction, token_count=token_count, args=args):
        yield delta


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

    context = [pipeline.encode(ctx)[-ctx_limit:] for ctx in inputs]
    _, state = model.forward(context[0], None)
    *_, embedding = state

    if len(embedding.shape) == 1:
        embedding = embedding.unsqueeze(0)
    return embedding
