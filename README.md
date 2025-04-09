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
pip install openaitrans
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
from openaitrans import translator

# Simple translation (auto-detect source language to Persian)
result = translator.translate("Hello, how are you?")
print(result.result)

# Specify source and target languages
result = translator.translate(
    "Hello, how are you?",
    t_from="en",
    t_to="fr"
)
print(result.result)

# Streaming translation
for chunk in translator.stream_translate():
    print(chunk)
```

### Advanced Usage

```python
from openaitrans import Translator

# Create a custom translator instance with specific settings
custom_translator = Translator(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Translate with specific formatting
result = custom_translator.translate(
    "Hello, how are you?",
    t_from="en",
    t_to="fr",
    text_format="markdown"
)

# Get token usage information
print(custom_translator.token_usage)
```

## API Reference

### `Translator` Class

The main class for translation operations. Use this when you need custom settings.

#### Methods

- `translate(t_text, model="gpt-4o-mini", t_to=None, t_from=None)`: Translate text
- `stream_translate()`: Stream translation results
- `count_tokens()`: Get token usage information

#### Parameters

- `t_text`: Text to translate
- `model`: OpenAI model to use (default: "gpt-4o-mini")
- `t_to`: Target language code
- `t_from`: Source language code

### Default Instance

The package provides a default `translator` instance for quick and easy use:

```python
from openaitrans import translator

# Use the default instance
result = translator.translate("Hello, how are you?")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 