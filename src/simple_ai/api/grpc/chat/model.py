from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class LanguageModel:
    def chat(self, chatlog: list[list[str]] = [], *args, **kwargs) -> list:
        return [
            {"role": message.get("role"), "content": message.get("content")[::-1]}
            for message in chatlog
        ]

    def stream(self, chatlog: list[list[str]] = [], *args, **kwargs) -> list:
        raise NotImplementedError()
