import pytest
from openaitrans import Translator, translator

def test_translator_initialization():
    """Test if the translator can be initialized properly"""
    translator_instance = Translator()
    assert translator_instance is not None
    assert translator is not None

def test_simple_translation():
    """Test basic translation functionality"""
    # Test with instance
    result = translator.translate("Hello, how are you?", t_to="fa")
    assert result is not None
    assert isinstance(result.result, str)
    assert len(result.result) > 0

    # Test with class
    translator_instance = Translator()
    result = translator_instance.translate("Hello, how are you?", t_to="fa")
    assert result is not None
    assert isinstance(result.result, str)
    assert len(result.result) > 0

def test_language_detection():
    """Test automatic language detection"""
    # Test with instance
    result = translator.translate("Bonjour", t_to="en")
    assert result is not None
    assert isinstance(result.result, str)
    assert len(result.result) > 0

    # Test with class
    translator_instance = Translator()
    result = translator_instance.translate("Bonjour", t_to="en")
    assert result is not None
    assert isinstance(result.result, str)
    assert len(result.result) > 0

def test_token_counting():
    """Test token counting functionality"""
    # Test with instance
    text = "Hello, this is a test"
    token_count = translator.count_tokens(text)
    assert isinstance(token_count, int)
    assert token_count > 0

    # Test with class
    translator_instance = Translator()
    token_count = translator_instance.count_tokens(text)
    assert isinstance(token_count, int)
    assert token_count > 0

@pytest.mark.asyncio
async def test_streaming_translation():
    """Test streaming translation functionality"""
    # Test with instance
    text = "Hello, this is a test"
    async for chunk in translator.stream_translate(text, t_to="fa"):
        assert isinstance(chunk, str)
        assert len(chunk) > 0

    # Test with class
    translator_instance = Translator()
    async for chunk in translator_instance.stream_translate(text, t_to="fa"):
        assert isinstance(chunk, str)
        assert len(chunk) > 0 