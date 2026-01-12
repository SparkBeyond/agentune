"""Tests for table formatting."""

import polars as pl

from agentune.core.dataset import Dataset
from agentune.core.formatter.tables import MarkdownTableFormatter
from agentune.core.schema import Field, Schema
from agentune.core.types import float64, int32, string


class TestMarkdownTableFormatter:
    """Test MarkdownTableFormatter functionality."""
    
    def test_string_truncation(self) -> None:
        """Test that long string values are truncated with '...'."""
        formatter = MarkdownTableFormatter(max_str=20)
        
        # Create test data with a long string
        data = pl.DataFrame({
            'id': [1, 2, 3],
            'short': ['abc', 'def', 'ghi'],
            'long': [
                'this is a very long string that should be truncated',
                'another extremely long text value here',
                'short'
            ],
        })
        
        schema = Schema((
            Field('id', int32),
            Field('short', string),
            Field('long', string),
        ))
        
        dataset = Dataset(schema=schema, data=data)
        
        result = formatter.format_table(dataset)
        lines = result.split('### Sample Data:')[1].strip().split('\n')
        
        # Verify truncated strings in correct column
        assert lines[1].split(',')[2] == 'this is a very long ...'
        assert lines[2].split(',')[2] == 'another extremely lo...'
        assert lines[3].split(',')[2] == 'short'
        
        # Verify short strings preserved in correct column
        assert lines[1].split(',')[1] == 'abc'
    
    def test_non_string_columns_preserved(self) -> None:
        """Test that non-string columns are not truncated."""
        formatter = MarkdownTableFormatter(max_str=10)
        
        # Create test data with various types
        data = pl.DataFrame({
            'id': [1, 2, 3],
            'value': [123456789012345, 987654321098765, 111222333444555],
            'price': [99.99999999, 12345.6789, 0.123456789],
            'name': ['this is a very long name', 'short', 'another long name here'],
        })
        
        schema = Schema((
            Field('id', int32),
            Field('value', int32),
            Field('price', float64),
            Field('name', string),
        ))
        
        dataset = Dataset(schema=schema, data=data)
        
        result = formatter.format_table(dataset)
        lines = result.split('### Sample Data:')[1].strip().split('\n')
        
        # Verify columns: id preserved, value preserved, price preserved, name truncated
        assert lines[1].split(',')[0] == '1'
        assert lines[1].split(',')[1] == '123456789012345'
        assert lines[1].split(',')[3] == 'this is a ...'
        assert lines[2].split(',')[3] == 'short'
    
    def test_custom_max_str_length(self) -> None:
        """Test that max_str parameter controls truncation length."""
        data = pl.DataFrame({
            'text': ['12345678901234567890', 'short'],
        })
        
        schema = Schema((Field('text', string),))
        dataset = Dataset(schema=schema, data=data)
        
        # Test with max_str=10
        result_10 = MarkdownTableFormatter(max_str=10).format_table(dataset)
        assert result_10.split('### Sample Data:')[1].strip().split('\n')[1] == '1234567890...'
        
        # Test with max_str=100 (no truncation)
        result_100 = MarkdownTableFormatter(max_str=100).format_table(dataset)
        assert result_100.split('### Sample Data:')[1].strip().split('\n')[1] == '12345678901234567890'
    
    def test_empty_dataset(self) -> None:
        """Test formatting an empty dataset."""
        formatter = MarkdownTableFormatter(max_str=20)
        
        data = pl.DataFrame({
            'id': pl.Series([], dtype=pl.Int32),
            'name': pl.Series([], dtype=pl.Utf8),
        })
        
        schema = Schema((
            Field('id', int32),
            Field('name', string),
        ))
        
        dataset = Dataset(schema=schema, data=data)
        
        result = formatter.format_table(dataset)
        csv_lines = result.split('### Sample Data:')[1].strip().split('\n')
        
        # Should have only header, no data rows
        assert len(csv_lines) == 1
        assert csv_lines[0] == 'id,name'
