dummy_models = [
    {
        "data": [
            {
                "id": "model-id-0",
                "object": "model",
                "owned_by": "organization-owner",
                "permission": [],
            },
            {
                "id": "model-id-1",
                "object": "model",
                "owned_by": "organization-owner",
                "permission": [],
            },
            {"id": "model-id-2", "object": "model", "owned_by": "openai", "permission": []},
        ],
        "object": "list",
    }
]

dummy_model = {
    "id": "model-id-0",
    "object": "model",
    "owned_by": "organization-owner",
    "permission": [],
}

dummy_complete = {
    "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
    "object": "text_completion",
    "created": 1589478378,
    "model": "text-davinci-003",
    "choices": [
        {
            "text": "\n\nThis is indeed a test",
            "index": 0,
            "logprobs": None,
            "finish_reason": "length",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}

dummy_chat = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\n\nHello there, how may I assist you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}

dummy_edit = {
    "object": "edit",
    "created": 1589478378,
    "choices": [
        {
            "text": "What day of the week is it?",
            "index": 0,
        }
    ],
    "usage": {"prompt_tokens": 25, "completion_tokens": 32, "total_tokens": 57},
}

dummy_embedding = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                0.0023064255,
                -0.009327292,
                -0.0028842222,
            ],
            "index": 0,
        }
    ],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}

dummy_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
