from typing import Union

from fastapi import FastAPI

from .models import get_model, list_models, get_model_infos
from .utils import (
    format_autocompletion_response, 
    add_instructions, format_edits_response, 
    format_chat_log, format_chat_response
)
from .dummy import dummy_embedding

app = FastAPI(
    title='SimpleAI',
    description='A self-hosted alternative API to the not so Open one',
    version='0.0.1',
    terms_of_service='http://example.com/terms/',
    contact={
        'name': 'Lhenault',
        'url': 'https://github.com/lhenault',
    }
)

# Models
@app.get('/models/')
async def show_models():
    return list_models()

@app.get('/models/{model_id}')
async def show_model(model_id: str):
    return get_model_infos(model_id)

# Completions
@app.post('/completions/')
async def complete(
    model:              str, 
    prompt:             str='<|endoftext|>',
    suffix:             str='',
    max_tokens:         int=7, 
    temperature:        float=1.,
    top_p:              float=1.,
    n:                  int=1,
    stream:             bool=False,
    logprobs:           int=0,
    echo:               bool=False,
    stop:               Union[str, list]='',
    presence_penalty:   float=0.,
    frequence_penalty:  float=0.,
    best_of:            int=0,
    logit_bias:         dict={},
    user:               str=''
):
    assert logprobs <= 5
    
    llm = get_model(model_id=model)
    predictions = llm.complete(
        prompt=prompt,
        suffix=suffix,
        max_tokens=max_tokens, 
        temperature=temperature,
        top_p=top_p,
        n=n,
        stream=stream,
        logprobs=logprobs,
        echo=echo,
        stop=stop,
        presence_penalty=presence_penalty,
        frequence_penalty=frequence_penalty,
        best_of=best_of,
        logit_bias=logit_bias
    )
    output = format_autocompletion_response(model_name=llm.name, predictions=predictions)
    return output

# Chat / completions
@app.post('/chat/completions/')
async def chat_complete(
    model:              str,
    messages:           list[dict],
    temperature:        float=1.,
    top_p:              float=1.,
    n:                  int=1,
    stream:             bool=False,
    stop:               str|list='',
    max_tokens:         int=7, 
    presence_penalty:   float=0.,
    frequence_penalty:  float=0.,
    logit_bias:         dict={},
    user:               str=''
):
    llm = get_model(model_id=model)
    input_text = format_chat_log(chat=messages)
    predictions = llm.complete(
        prompt=input_text,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stream=stream,
        max_tokens=max_tokens,
        stop=stop,
        presence_penalty=presence_penalty,
        frequence_penalty=frequence_penalty,
        logit_bias=logit_bias
    )
    output = format_chat_response(model_name=llm.name, predictions=predictions)
    return output

# Edits
@app.post('/edits/')
async def edit(
    model:              str,
    instruction:        str,
    input:              str='',
    top_p:              float=1.,
    n:                  int=1,
    temperature:        float=1.,
    max_tokens:         int=256
):
    llm = get_model(model_id=model)
    input_text = add_instructions(instructions=instruction, text=input)
    
    predictions = llm.complete(
        prompt=input_text,
        temperature=temperature,
        top_p=top_p,
        n=n,
        max_tokens=max_tokens
    )
    output = format_edits_response(model_name=llm.name, predictions=predictions)
    return output

# Embeddings
@app.post('/embed/')
async def embed(
    model:              str,
    input:              str,
    user:               str=''
):
    llm = get_model(model_id=model)
    return dummy_embedding