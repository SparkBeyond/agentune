"""Tests for utility functions in insightful_text_generator."""

import json
from unittest.mock import Mock, patch

import pytest

from agentune.analyze.feature.gen.insightful_text_generator.util import (
    estimate_tokens,
    extract_json_from_response,
    get_max_input_context_window,
    parse_json_response_field,
)


class TestExtractJsonFromResponse:
    """Test extract_json_from_response function."""
    
    def test_valid_json_with_newlines(self) -> None:
        """Test extracting valid JSON with newlines in markdown format."""
        response = '''
        Here's the analysis:
        
        ```json
        {
            "result": "success",
            "data": [1, 2, 3],
            "nested": {"key": "value"}
        }
        ```
        
        That's the response.
        '''
        
        result = extract_json_from_response(response)
        expected = {
            'result': 'success',
            'data': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        assert result == expected
    
    def test_valid_json_without_newlines(self) -> None:
        """Test extracting valid JSON without newlines."""
        response = '```json{"key": "value", "number": 42}```'
        
        result = extract_json_from_response(response)
        expected = {'key': 'value', 'number': 42}
        assert result == expected
    
    def test_valid_json_with_extra_whitespace(self) -> None:
        """Test extracting JSON with extra whitespace."""
        response = '''
        ```json
        
        {"status": "ok", "count": 5}
        
        ```
        '''
        
        result = extract_json_from_response(response)
        expected = {'status': 'ok', 'count': 5}
        assert result == expected
    
    def test_complex_json_object(self) -> None:
        """Test extracting complex JSON with various data types."""
        response = '''
        ```json
        {
            "string": "hello world",
            "number": 123,
            "float": 45.67,
            "boolean": true,
            "null_value": null,
            "array": ["a", "b", "c"],
            "nested_object": {
                "inner_key": "inner_value",
                "inner_array": [1, 2, 3]
            }
        }
        ```
        '''
        
        result = extract_json_from_response(response)
        expected = {
            'string': 'hello world',
            'number': 123,
            'float': 45.67,
            'boolean': True,
            'null_value': None,
            'array': ['a', 'b', 'c'],
            'nested_object': {
                'inner_key': 'inner_value',
                'inner_array': [1, 2, 3]
            }
        }
        assert result == expected
    
    def test_no_json_found(self) -> None:
        """Test error when no JSON code block is found."""
        response = 'This is just a regular response with no JSON.'
        
        with pytest.raises(ValueError, match='No JSON found in response'):
            extract_json_from_response(response)
    
    def test_no_json_with_other_code_blocks(self) -> None:
        """Test error when other code blocks exist but no JSON."""
        response = '''
        Here's some code:
        
        ```python
        print("hello world")
        ```
        
        And some text:
        ```
        just text
        ```
        '''
        
        with pytest.raises(ValueError, match='No JSON found in response'):
            extract_json_from_response(response)
    
    def test_multiple_json_sections_error(self) -> None:
        """Test error when multiple JSON sections are found."""
        response = '''
        First JSON:
        ```json
        {"first": "json"}
        ```
        
        Second JSON:
        ```json
        {"second": "json"}
        ```
        '''
        
        with pytest.raises(ValueError, match='Multiple JSON sections found in response \\(2 sections\\)'):
            extract_json_from_response(response)
    
    def test_invalid_json_syntax(self) -> None:
        """Test error when JSON syntax is invalid."""
        response = '''
        ```json
        {
            "invalid": json,
            "missing": "quotes"
        }
        ```
        '''
        
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_response(response)
    
    def test_empty_json_block(self) -> None:
        """Test behavior with empty JSON block."""
        response = '```json```'
        
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_response(response)
    
    def test_json_with_surrounding_text(self) -> None:
        """Test extracting JSON when surrounded by other text and code blocks."""
        response = '''
        Here's some analysis. First, let me show you some Python:
        
        ```python
        def hello():
            return "world"
        ```
        
        Now here's the JSON result:
        
        ```json
        {
            "analysis": "complete",
            "confidence": 0.95
        }
        ```
        
        And here's some more text after the JSON.
        '''
        
        result = extract_json_from_response(response)
        expected = {'analysis': 'complete', 'confidence': 0.95}
        assert result == expected


class TestParseJsonResponseField:
    """Test parse_json_response_field function."""
    
    def test_extract_existing_field(self) -> None:
        """Test extracting an existing field from JSON response."""
        response = '''
        ```json
        {
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        }
        ```
        '''
        
        result = parse_json_response_field(response, 'name')
        assert result == 'John Doe'
    
    def test_extract_numeric_field_as_string(self) -> None:
        """Test that numeric fields are converted to strings."""
        response = '''
        ```json
        {
            "count": 42,
            "score": 95.5,
            "active": true
        }
        ```
        '''
        
        assert parse_json_response_field(response, 'count') == '42'
        assert parse_json_response_field(response, 'score') == '95.5'
        assert parse_json_response_field(response, 'active') == 'True'
    
    def test_extract_nonexistent_field(self) -> None:
        """Test extracting a field that doesn't exist returns empty string."""
        response = '''
        ```json
        {
            "existing": "value"
        }
        ```
        '''
        
        result = parse_json_response_field(response, 'nonexistent')
        assert result == ''
    
    def test_extract_null_field(self) -> None:
        """Test extracting a null field returns string 'None'."""
        response = '''
        ```json
        {
            "null_field": null,
            "empty_string": ""
        }
        ```
        '''
        
        assert parse_json_response_field(response, 'null_field') == 'None'
        assert parse_json_response_field(response, 'empty_string') == ''
    
    def test_extract_nested_object_as_string(self) -> None:
        """Test extracting nested objects gets converted to string representation."""
        response = '''
        ```json
        {
            "nested": {
                "inner": "value"
            },
            "array": [1, 2, 3]
        }
        ```
        '''
        
        nested_result = parse_json_response_field(response, 'nested')
        array_result = parse_json_response_field(response, 'array')
        
        # Should convert to string representation
        assert nested_result is not None
        assert array_result is not None
        assert 'inner' in nested_result
        assert 'value' in nested_result
        assert '[1, 2, 3]' in array_result
    
    def test_invalid_json_returns_none(self) -> None:
        """Test that invalid JSON returns None."""
        response = '''
        ```json
        {
            invalid: json
        }
        ```
        '''
        
        result = parse_json_response_field(response, 'any_key')
        assert result is None
    
    def test_no_json_returns_none(self) -> None:
        """Test that response with no JSON returns None."""
        response = 'This is just text with no JSON.'
        
        result = parse_json_response_field(response, 'any_key')
        assert result is None
    
    def test_multiple_json_sections_returns_none(self) -> None:
        """Test that multiple JSON sections returns None."""
        response = '''
        ```json
        {"first": "json"}
        ```
        ```json
        {"second": "json"}
        ```
        '''
        
        result = parse_json_response_field(response, 'first')
        assert result is None
    
    def test_empty_response_returns_none(self) -> None:
        """Test that empty response returns None."""
        result = parse_json_response_field('', 'any_key')
        assert result is None

    def test_logging_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that errors are logged appropriately."""
        response = 'Invalid response'
        
        result = parse_json_response_field(response, 'test_key')
        
        assert result is None
        assert 'Failed to parse JSON response field "test_key"' in caplog.text
        assert 'No JSON found in response' in caplog.text


class TestEstimateTokens:
    """Test estimate_tokens function with various inputs and models."""
    
    def test_estimate_tokens_gpt4(self) -> None:
        """Test token estimation for GPT-4 model."""
        # Create mock LLMWithSpec
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'gpt-4'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        # Test with simple text
        text = 'Hello, world!'
        result = estimate_tokens(text, mock_llm_with_spec)
        
        # Should return a reasonable token count (tiktoken should give around 3-4 tokens)
        assert isinstance(result, int)
        assert result > 0
        assert result < 10  # Simple text shouldn't be too many tokens
    
    def test_estimate_tokens_gpt4o_mini(self) -> None:
        """Test token estimation for GPT-4o-mini model."""
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'gpt-4o-mini'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        text = 'This is a longer text that should have more tokens than the simple hello world example.'
        result = estimate_tokens(text, mock_llm_with_spec)
        
        assert isinstance(result, int)
        assert result > 10  # Longer text should have more tokens
        assert result < 50  # But not too many
    
    def test_estimate_tokens_unknown_model_fallback(self) -> None:
        """Test that unknown models fall back to cl100k_base encoding."""
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'unknown-model-name'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        text = 'Test text for unknown model'
        
        # Should not raise an error and should return a reasonable count
        result = estimate_tokens(text, mock_llm_with_spec)
        assert isinstance(result, int)
        assert result > 0
    
    def test_estimate_tokens_empty_string(self) -> None:
        """Test token estimation for empty string."""
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'gpt-4o'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        result = estimate_tokens('', mock_llm_with_spec)
        assert result == 0
    
    def test_estimate_tokens_very_long_text(self) -> None:
        """Test token estimation for very long text."""
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'gpt-4o'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        # Create a long text (about 1000 words)
        long_text = ' '.join(['word'] * 1000)
        result = estimate_tokens(long_text, mock_llm_with_spec)
        
        assert isinstance(result, int)
        assert result > 500  # Should be a substantial number of tokens
        assert result < 2000  # But not unreasonably high
    
    def test_estimate_tokens_special_characters(self) -> None:
        """Test token estimation with special characters and unicode."""
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'gpt-4o'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        text = 'Special chars: @#$%^&*()_+ Unicode: café, naïve, résumé 🚀🎉'
        result = estimate_tokens(text, mock_llm_with_spec)
        
        assert isinstance(result, int)
        assert result > 0
    
    @patch('agentune.analyze.feature.gen.insightful_text_generator.util.tiktoken.encoding_for_model')
    def test_estimate_tokens_tiktoken_error_fallback(self, mock_encoding_for_model: Mock) -> None:
        """Test fallback behavior when tiktoken raises KeyError."""
        mock_llm_spec = Mock()
        mock_llm_spec.model_name = 'problematic-model'
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.spec = mock_llm_spec
        
        # Make tiktoken.encoding_for_model raise KeyError
        mock_encoding_for_model.side_effect = KeyError('Unknown model')
        
        with patch('agentune.analyze.feature.gen.insightful_text_generator.util.tiktoken.get_encoding') as mock_get_encoding:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoding.return_value = mock_encoder
            
            result = estimate_tokens('test text', mock_llm_with_spec)
            
            # Should fall back to o200k_base
            mock_get_encoding.assert_called_once_with('o200k_base')
            assert result == 5


class TestGetMaxContextWindow:
    """Test get_max_context_window function with various LLM configurations."""
    
    def test_get_max_context_window_standard_model(self) -> None:
        """Test context window calculation for standard model."""
        mock_llm = Mock()
        mock_llm.metadata.context_window = 4096
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.llm = mock_llm
        
        result = get_max_input_context_window(mock_llm_with_spec)
        
        # Should return 75% of the context window (3072)
        expected = 4096 * 3 // 4
        assert result == expected
        assert result == 3072
    
    def test_get_max_context_window_large_model(self) -> None:
        """Test context window calculation for large model."""
        mock_llm = Mock()
        mock_llm.metadata.context_window = 32768  # Large model like GPT-4 Turbo
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.llm = mock_llm
        
        result = get_max_input_context_window(mock_llm_with_spec)
        
        # Should return 75% of the context window
        expected = 32768 * 3 // 4
        assert result == expected
        assert result == 24576
    
    def test_get_max_context_window_small_model(self) -> None:
        """Test context window calculation for small model."""
        mock_llm = Mock()
        mock_llm.metadata.context_window = 2048
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.llm = mock_llm
        
        result = get_max_input_context_window(mock_llm_with_spec)
        
        # Should return 75% of the context window
        expected = 2048 * 3 // 4
        assert result == expected
        assert result == 1536
    
    def test_get_max_context_window_very_large_model(self) -> None:
        """Test context window calculation for very large model."""
        mock_llm = Mock()
        mock_llm.metadata.context_window = 128000  # Very large model
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.llm = mock_llm
        
        result = get_max_input_context_window(mock_llm_with_spec)
        
        # Should return 75% of the context window
        expected = 128000 * 3 // 4
        assert result == expected
        assert result == 96000
    
    def test_get_max_context_window_odd_numbers(self) -> None:
        """Test context window calculation with odd numbers (integer division)."""
        mock_llm = Mock()
        mock_llm.metadata.context_window = 4097  # Odd number
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.llm = mock_llm
        
        result = get_max_input_context_window(mock_llm_with_spec)
        
        # Should handle integer division correctly
        expected = 4097 * 3 // 4
        assert result == expected
        assert result == 3072  # 4097 * 3 // 4 = 3072
    
    def test_get_max_context_window_minimum_case(self) -> None:
        """Test context window calculation for minimum viable case."""
        mock_llm = Mock()
        mock_llm.metadata.context_window = 100  # Very small
        
        mock_llm_with_spec = Mock()
        mock_llm_with_spec.llm = mock_llm
        
        result = get_max_input_context_window(mock_llm_with_spec)
        
        # Should return 75% even for small numbers
        expected = 100 * 3 // 4
        assert result == expected
        assert result == 75
