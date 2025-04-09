# OpenAI Translator

A powerful AI-based text translator using ChatGPT API. This package provides an easy-to-use interface for translating text between different languages while preserving formatting and structure.

## Features

- Automatic language detection
- Support for multiple languages
- Preserves text formatting (Markdown, HTML, JSON, etc.)
- Streaming translation support
- Token usage tracking
- Configurable formality level
- Easy-to-use Python interface

## Installation

```bash
pip install openai-trans
```

## Usage

First, set up your OpenAI API key in your environment:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file in your project root:

```
OPENAI_API_KEY=your-api-key-here
```

### Basic Usage

```python
from openai_trans import ai_translator

# Simple translation (auto-detect source language to Persian)
result = ai_translator.translate("Hello, how are you?")
print(result.result)

# Specify source and target languages
result = ai_translator.translate(
    "Hello, how are you?",
    t_from="en",
    t_to="fr"
)
print(result.result)

# Streaming translation
for chunk in ai_translator.stream_translate():
    print(chunk)
```

### Advanced Usage

```python
# Translate with specific formatting
result = ai_translator.translate(
    "Hello, how are you?",
    t_from="en",
    t_to="fr",
    text_format="markdown"
)

# Get token usage information
print(ai_translator.token_usage)
```

## API Reference

### `AITranslator` Class

The main class for translation operations.

#### Methods

- `translate(t_text, model="gpt-4o-mini", t_to=None, t_from=None)`: Translate text
- `stream_translate()`: Stream translation results
- `count_tokens()`: Get token usage information

#### Parameters

- `t_text`: Text to translate
- `model`: OpenAI model to use (default: "gpt-4o-mini")
- `t_to`: Target language code
- `t_from`: Source language code

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 