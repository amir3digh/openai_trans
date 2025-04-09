import json
from enum import Enum
from typing import List, Literal, Optional, Dict, Any, AsyncGenerator
import tiktoken
from openai import OpenAI
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

__version__ = "0.1.1"

class GPTTranslation(BaseModel):
    """Model representing a translation result from GPT."""
    translate_from: str
    translate_to: str
    text_format: Literal["text", "markdown", "json", "html", "xml", "other"]
    is_formal: bool
    result: str


class SystemPrompt(Enum):
    """System prompts for the translation service."""
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


class Translator(OpenAI):
    """
    A translator class that uses OpenAI's GPT models for text translation.
    
    This class extends OpenAI's client and provides methods for translating text
    between languages using GPT models. It supports both synchronous and streaming
    translation modes.
    """
    parser_class = GPTTranslation
    system_prompt = SystemPrompt.DEFAULT
    model: Optional[str] = None
    final_prompt: Optional[str] = None
    parsed_completion: Optional[ParsedChatCompletion] = None
    token_usage: Optional[Dict[str, int]] = None
    result: Optional[GPTTranslation] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the translator with OpenAI client configuration."""
        super().__init__(*args, **kwargs)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide the API key when initializing "
                "the Translator or set it in your environment variables."
            )

    def prepare_prompt(self, t_text: str, t_from: Optional[str] = None, t_to: Optional[str] = None) -> str:
        """
        Prepare the translation prompt with language tags.
        
        Args:
            t_text: The text to be translated
            t_from: Source language code (optional)
            t_to: Target language code (optional)
            
        Returns:
            str: The formatted prompt with language tags
        """
        assert t_text is not None, "t_text is required. (the main text to be translated)"
        assert isinstance(t_text, str), """t_text must be of type str."""
        if t_from is None:
            if t_to is None:
                self.final_prompt = t_text
            else:
                self.final_prompt = f"""
                    [to:{t_to}],
                    {t_text}
                """
        elif t_to is None:
            self.final_prompt = f"""
                [from:{t_from}],
                {t_text}
            """
        else:
            self.final_prompt = f"""
                [from:{t_from}][to:{t_to}],
                {t_text}
            """
        return self.final_prompt

    def translate(self, t_text: str, model: str = "gpt-4o-mini", t_to: Optional[str] = None, t_from: Optional[str] = None) -> GPTTranslation:
        """
        Translate text synchronously.
        
        Args:
            t_text: The text to be translated
            model: The GPT model to use (default: "gpt-4o-mini")
            t_to: Target language code (optional)
            t_from: Source language code (optional)
            
        Returns:
            GPTTranslation: The translation result
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

    def done(self) -> GPTTranslation:
        """
        Process the completion and return the translation result.
        
        Returns:
            GPTTranslation: The parsed translation result
            
        Raises:
            AssertionError: If no translation was performed
        """
        assert self.parsed_completion is not None, (
            "No translation was performed. Call translate() or stream_translate() first."
        )
        self.result = self.parsed_completion.choices[0].message.parsed
        self._count_tokens()
        return self.result

    def _count_tokens(self) -> Dict[str, int]:
        """
        Count tokens used in the translation.
        
        Returns:
            Dict[str, int]: Token usage statistics
            
        Raises:
            AssertionError: If no completion was performed
        """
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
            prompt_tokens = self.count_tokens(self.system_prompt.value + self.final_prompt, self.model)
            completion_tokens = self.count_tokens(self.result.model_dump_json(), self.model)
            total_tokens = prompt_tokens + completion_tokens
            self.token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            return self.token_usage

    @classmethod
    def count_tokens(cls, text: str, model: str = "gpt-4o-mini") -> int:
        """
        Count tokens in the given text using the specified model.
        
        Args:
            text: The text to count tokens for
            model: The GPT model to use (default: "gpt-4o-mini")
            
        Returns:
            int: Number of tokens in the text
        """
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    async def stream_translate(self, t_text: str, model: str = "gpt-4o-mini", t_to: Optional[str] = None, t_from: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream translation results as they become available.
        
        Args:
            t_text: The text to be translated
            model: The GPT model to use (default: "gpt-4o-mini")
            t_to: Target language code (optional)
            t_from: Source language code (optional)
            
        Yields:
            str: Server-Sent Events (SSE) formatted translation updates
        """
        self.prepare_prompt(t_text, t_from, t_to)
        self.model = model
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
                    yield f"id: {i}\ndata: {json_error}\n\n"
                    return

        self.parsed_completion = stream.get_final_completion()
        result = self.done()
        json_result = json.dumps(result.model_dump())
        yield f"id: {i}\ndata: {json_result}\n\n"


translator = Translator()