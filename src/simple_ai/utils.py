from datetime import datetime as dt
import uuid


def format_autocompletion_response(model_name, predictions) -> dict:
    response_id         = uuid.uuid4()
    current_timestamp   = int(dt.now().timestamp())
 
    return {
        'id': response_id,
        'object': 'text_completion',
        'created': current_timestamp,
        'model': model_name,
        'choices': [
            {
            'text': text,
            'index': idx,
            'logprobs': None,
            'finish_reason': ''
            }
            
            for idx, text in enumerate(predictions)
        ]
    }


def format_edits_response(model_name, predictions) -> dict:
    response_id         = uuid.uuid4()
    current_timestamp   = int(dt.now().timestamp())
 
    return {
        'id': response_id,
        'object': 'edit',
        'created': current_timestamp,
        'model': model_name,
        'choices': [
            {
            'text': text,
            'index': idx,
            }
            
            for idx, text in enumerate(predictions)
        ]
    }

 
def format_chat_response(model_name: str, predictions) -> dict:
    response_id         = uuid.uuid4()
    current_timestamp   = int(dt.now().timestamp())
    
    return {
        'id': response_id,
        'model': model_name,
        'object': 'chat.completion',
        'created': current_timestamp,
        'choices': [
            {
                'role': 'assistant',
                'index': idx,
                'message': {
                    'role': 'assistant',
                    'content': text
                },
                'finish_reason': 'stop'
            }
            
            for idx, text in enumerate(predictions)
        ]
    }

def format_embeddings_results(model_name: str, embeddings: list) -> dict:
    return {
        'object': 'list',
        'data': [
            {
                'object': 'embedding',
                'embedding': embedding,
                'index': idx
            }
            for idx, embedding in enumerate(embeddings)
        ],
        'model': model_name,
    }
    
    
def add_instructions(instructions: str,  text: str) -> str:
    prompt = f'### Instruction:\n{instructions}.\n\n'
    if input:
        prompt += f'### Input:\n{text}.\n\n'
    prompt += f'### Response:'
    return prompt


def format_chat_log(chat: list[dict[str, str]]=dict()) -> str:
    raw_chat_text = ''
    for item in chat:
        raw_chat_text += f"{item.get('role')}: {item.get('content')}\n"
    return raw_chat_text