from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class LanguageModel:
    def chat(self, chatlog: list[list[dict]] = [], *args, **kwargs) -> list:
        def reverse_content(content):
            if isinstance(content, str):
                return content[::-1]
            elif isinstance(content, list):
                return [
                    {
                        "type": item.get("type"),
                        "text": (
                            item.get("text")[::-1]
                            if item.get("type") == "text"
                            else item.get("image_url")
                        ),
                    }
                    for item in content
                ]
            return content

        return [
            {"role": message.get("role"), "content": reverse_content(message.get("content"))}
            for message in chatlog
        ]

    def stream(self, chatlog: list[list[str]] = [], *args, **kwargs) -> list:
        raise NotImplementedError()
