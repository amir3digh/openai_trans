import json
from enum import Enum
from typing import List, Literal
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

__version__ = "0.1.0"

load_dotenv()

class GPTTranslation(BaseModel):
    translate_from: str
    translate_to: str
    text_format: Literal["text", "markdown", "json", "html", "xml", "other"]
    is_formal: bool
    result: str


class SystemPrompt(Enum):
    DEFAULT = """
        You are a highly intelligent and reliable AI-based language translator.  
        Your task is to accurately and naturally translate text between any two given languages.  
        - Users may specify the source and target languages using tags like `[from:en][to:fr]`.  
        - If no tags are provided, automatically detect the source language and translate the content to **Persian** (`fa`).  
        - Preserve the **structure and formatting** of any input, including **JSON, HTML, and Markdown**. Only translate text content — do not alter code, keys, tags, or syntax.  
        - Maintain the meaning, tone, and cultural nuance of the original message.  
        - Adapt idioms and expressions appropriately for the target language and culture.  
        - Respect the formality or informality of the original text.  
        If needed, explain ambiguities or offer alternatives when multiple interpretations exist.  
        Only return the translated result unless explicitly asked to explain.
    """

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


class AITranslator(OpenAI):
    parser_class = GPTTranslation
    system_prompt = SystemPrompt.DEFAULT
    model = None
    final_prompt = None
    parsed_completion: ParsedChatCompletion = None
    token_usage: dict = None
    result = None

    def prepare_prompt(self, t_text, t_from=None, t_to=None):
        assert t_text is not None, "t_text is required. (the main text to be translated)"
        assert isinstance(t_text, str), """t_text must be of type str."""
        if not t_from:
            if not t_to:
                self.final_prompt = t_text
            else:
                assert isinstance(t_from, str), """t_from must be of type str."""
                self.final_prompt = f"""
                    [to:{t_to}],
                    {t_text}
                """
        elif not t_to:
            assert isinstance(t_from, str), """t_from must be of type str."""
            self.final_prompt = f"""
                [from:{t_from}],
                {t_text}
            """
        else:
            assert isinstance(t_from, str) and isinstance(t_to, str), """t_from and t_to must be of type str."""
            self.final_prompt = f"""
                [from:{t_from}][to:{t_to}],
                {t_text}
            """
        return self.final_prompt

    def translate(self, t_text, model="gpt-4o-mini", t_to=None, t_from=None):
        """
            Main method to start translation.
            Agent will detect input language automatically and translates it to persian (fa).
            If you want to auto-detect input language and use custom target translation pass t_to in any way you prefer.
            You can optionally specify gpt model and input language.
        """
        self.prepare_prompt(t_text, t_from, t_to)
        self.model = model
        self.parsed_completion = self.beta.chat.completions.parse(
            model=self.model,
            response_format=self.parser_class,
            messages=[
                {"role": "system", "content": self.system_prompt.value},
                {"role": "user", "content": self.final_prompt}
            ]
        )
        return self.done()

    def stream_translate(self):
        i = 1
        with self.beta.chat.completions.stream(
                model=self.model,
                response_format=self.parser_class,
                messages=[
                    {"role": "system", "content": self.system_prompt.value},
                    {"role": "user", "content": self.final_prompt}
                ]
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    if event.parsed is not None:
                        json_event = json.dumps(event.parsed)
                        yield f"id: {i}\ndata: {json_event}\n\n"
                        i += 1
                    else:
                        json_error = json.dumps({"errors": ["event could not be parsed"]})
                        yield f"id: {i}\ndata: {json_error}\n\n"
                        i += 1
                elif event.type == "error":
                    print(f"error: {event.type}: {event.error}")
                    json_error = json.dumps({"errors": ["error building project"]})
                    return f"id: {i}\ndata: {json_error}"

        self.parsed_completion = stream.get_final_completion()
        self.done()

    def done(self):
        assert self.parsed_completion is not None, (
            "No translation was performed. Call translate() or stream_translate() first."
        )
        self.result = self.parsed_completion.choices[0].message.parsed
        self.count_tokens()
        return self.result

    def count_tokens(self):
        assert self.parsed_completion is not None, """
            No completion was performed. Call translate() or stream_translate() first.
        """
        completion = self.parsed_completion
        try:
            self.token_usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
            return self.token_usage
        except Exception as e:
            assert self.result is not None, "Parsed result is not set"
            prompt_tokens = count_tokens(self.system_prompt.value + self.final_prompt, self.model)
            completion_tokens = count_tokens(self.result.model_dump_json())
            total_tokens = prompt_tokens + completion_tokens
            self.token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            return self.token_usage


ai_translator = AITranslator()