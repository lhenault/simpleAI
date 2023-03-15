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

 
def format_chat_response(model_name, predictions) -> dict:
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

    
def add_instructions(instructions: str,  text: str) -> str:
    return f'Instructions: {instructions}. \n Use the following text: "{text}". \nResults: '

def format_chat_log(chat: list[dict[str, str]]=dict()) -> str:
    raw_chat_text = ''
    for item in chat:
        raw_chat_text += f"{item.get('role')}: {item.get('role')}\n"
    return raw_chat_text