from typing import Annotated

from fastapi import Body, FastAPI

from .api_models import ChatCompletionInput, CompletionInput, EmbeddingInput, InstructionInput
from .dummy import dummy_chat, dummy_complete, dummy_edit, dummy_embedding
from .models import get_model, get_model_infos, list_models
from .utils import (
    add_instructions,
    format_autocompletion_response,
    format_chat_response,
    format_edits_response,
    format_embeddings_results,
)

app = FastAPI(
    title="SimpleAI",
    description="A self-hosted alternative API to the not so Open one",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Lhenault",
        "url": "https://github.com/lhenault",
    },
)


# Models
@app.get("/models/")
async def show_models():
    return list_models()


@app.get("/models/{model_id}")
async def show_model(model_id: str):
    return get_model_infos(model_id)


# Completions
@app.post("/completions/")
async def complete(body: Annotated[CompletionInput, Body(example=dummy_complete)]):
    assert body.logprobs <= 5

    llm = get_model(model_id=body.model)
    predictions = llm.complete(
        prompt=body.prompt,
        suffix=body.suffix,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        n=body.n,
        stream=body.stream,
        logprobs=body.logprobs,
        echo=body.echo,
        stop=body.stop,
        presence_penalty=body.presence_penalty,
        frequence_penalty=body.frequence_penalty,
        best_of=body.best_of,
        logit_bias=body.logit_bias,
    )
    output = format_autocompletion_response(model_name=llm.name, predictions=predictions)
    return output


# Chat / completions
@app.post("/chat/completions/")
async def chat_complete(body: Annotated[ChatCompletionInput, Body(example=dummy_chat)]):
    llm = get_model(model_id=body.mode, task="chat")
    predictions = llm.chat(
        messages=[
            [message.get("role", ""), message.get("content", "")] for message in body.messages
        ],
        temperature=body.temperature,
        top_p=body.top_p,
        n=body.n,
        stream=body.stream,
        max_tokens=body.max_tokens,
        stop=body.stop,
        presence_penalty=body.presence_penalty,
        frequence_penalty=body.frequence_penalty,
        logit_bias=body.logit_bias,
    )
    output = format_chat_response(model_name=llm.name, predictions=predictions)
    return output


# Edits
@app.post("/edits/")
async def edit(body: Annotated[InstructionInput, Body(example=dummy_edit)]):
    llm = get_model(model_id=body.model)
    input_text = add_instructions(instructions=body.instruction, text=input)

    predictions = llm.complete(
        prompt=input_text,
        temperature=body.temperature,
        top_p=body.top_p,
        n=body.n,
        max_tokens=body.max_tokens,
    )
    output = format_edits_response(model_name=llm.name, predictions=predictions)
    return output


# Embeddings
@app.post("/embeddings/")
async def embed(body: Annotated[EmbeddingInput, Body(example=dummy_embedding)]):
    llm = get_model(model_id=body.model, task="embed")
    if isinstance(body.input, str):
        body.input = [body.input]

    results = llm.embed(inputs=body.input)

    output = format_embeddings_results(model_name=llm.name, embeddings=results)
    return output
