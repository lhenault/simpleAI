from typing import Optional, Union
from pydantic import BaseModel

class EmbeddingInput(BaseModel):
    model: str
    input: Union[str, list]
    user:  str=''
    
class CompletionInput(BaseModel):
    model:              str 
    prompt:             str='<|endoftext|>'
    suffix:             str=''
    max_tokens:         int=7 
    temperature:        float=1.
    top_p:              float=1.
    n:                  int=1
    stream:             bool=False
    logprobs:           int=0
    echo:               bool=False
    stop:               Optional[Union[str, list]]=''
    presence_penalty:   float=0.
    frequence_penalty:  float=0.
    best_of:            int=0
    logit_bias:         dict={}
    user:               str=''
    
class ChatCompletionInput(BaseModel):
    model:              str
    messages:           list[dict]
    temperature:        float=1.
    top_p:              float=1.
    n:                  int=1
    stream:             bool=False
    stop:               Optional[Union[str, list]]=''
    max_tokens:         int=7 
    presence_penalty:   float=0.
    frequence_penalty:  float=0.
    logit_bias:         dict={}
    user:               str=''

class InstructionInput(BaseModel):
    model:              str
    instruction:        str
    input:              str=''
    top_p:              float=1.
    n:                  int=1
    temperature:        float=1.
    max_tokens:         int=256