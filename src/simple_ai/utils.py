import json
import uuid
from datetime import datetime as dt

from .dummy import dummy_usage


def format_autocompletion_response(model_name, predictions, usage=dummy_usage) -> dict:
    response_id = uuid.uuid4()
    current_timestamp = int(dt.now().timestamp())

    return {
        "id": response_id,
        "object": "text_completion",
        "created": current_timestamp,
        "model": model_name,
        "choices": [
            {"text": text, "index": idx, "logprobs": None, "finish_reason": ""}
            for idx, text in enumerate(predictions)
        ],
        "usage": usage,
    }


def format_autocompletion_stream_response(
    current_timestamp, response_id, model_name, predictions
) -> dict:
    data = {
        "id": response_id,
        "object": "text_completion",
        "created": current_timestamp,
        "model": model_name,
        "choices": [
            {"text": text, "index": idx, "logprobs": None, "finish_reason": None}
            for idx, text in enumerate(predictions)
        ],
    }

    data = f"DATA: {data}\n\n"

    return data


def format_edits_response(model_name, predictions, usage=dummy_usage) -> dict:
    response_id = uuid.uuid4()
    current_timestamp = int(dt.now().timestamp())

    return {
        "id": response_id,
        "object": "edit",
        "created": current_timestamp,
        "model": model_name,
        "choices": [
            {
                "text": text,
                "index": idx,
            }
            for idx, text in enumerate(predictions)
        ],
        "usage": usage,
    }


def format_chat_response(model_name: str, predictions, usage=dummy_usage) -> dict:
    response_id = uuid.uuid4()
    current_timestamp = int(dt.now().timestamp())

    return {
        "id": response_id,
        "model": model_name,
        "object": "chat.completion",
        "created": current_timestamp,
        "choices": [
            {
                "index": idx,
                "message": message,
                "finish_reason": "stop",
            }
            for idx, message in enumerate(predictions)
        ],
        "usage": usage,
    }


def format_chat_delta_response_helper(
    current_timestamp, response_id, model_name: str, predictions, finish_reason=None
) -> dict:
    data = {
        "id": response_id,
        "model": model_name,
        "object": "chat.completion.chunk",
        "created": current_timestamp,
        "choices": [
            {
                "index": idx,
                "delta": message,
                "finish_reason": finish_reason,
            }
            for idx, message in enumerate(predictions)
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


def format_chat_delta_response(
    current_timestamp, response_id, model_name: str, predictions
) -> dict:
    data = format_chat_delta_response_helper(
        current_timestamp, response_id, model_name, predictions, finish_reason=None
    )

    return data


def format_embeddings_results(model_name: str, embeddings: list, usage: dict = dummy_usage) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding, "index": idx}
            for idx, embedding in enumerate(embeddings)
        ],
        "model": model_name,
        "usage": usage,
    }


def add_instructions(instructions: str, text: str) -> str:
    prompt = f"### Instruction:\n{instructions}.\n\n"
    if text:
        prompt += f"### Input:\n{text}.\n\n"
    prompt += "### Response:"
    return prompt


def format_chat_log(chat: list[dict[str, str]] = dict()) -> str:
    raw_chat_text = ""
    for item in chat:
        raw_chat_text += f"{item.get('role')}: {item.get('content')}\n\n"
    return raw_chat_text + "assistant: "
