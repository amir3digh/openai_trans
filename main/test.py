from openai_trans import ai_translator

def test_translation():
    # Test basic translation
    result = ai_translator.translate("Hello, how are you?")
    assert result.result is not None
    assert isinstance(result.result, str)

def test_token_counting():
    # Test token counting
    result = ai_translator.translate("Hello, how are you?")
    assert ai_translator.token_usage is not None
    assert "total_tokens" in ai_translator.token_usage

if __name__ == "__main__":
    test_translation()
    test_token_counting()
    print("All tests passed!")