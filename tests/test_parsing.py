"""Tests for parsing utilities."""

import pytest

from rlm_langgraph.utils.parsing import (
    extract_code_blocks,
    find_final_in_code,
    format_execution_result,
    truncate_context_preview,
)


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function."""

    def test_single_repl_block(self):
        """Test extracting a single repl block."""
        response = '''Here is some code:

```repl
print("hello")
x = 10
```

That's the code.'''
        blocks = extract_code_blocks(response)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0].code
        assert "x = 10" in blocks[0].code

    def test_multiple_blocks(self):
        """Test extracting multiple code blocks."""
        response = '''First block:

```repl
x = 1
```

Second block:

```repl
y = 2
```'''
        blocks = extract_code_blocks(response)
        assert len(blocks) == 2
        assert "x = 1" in blocks[0].code
        assert "y = 2" in blocks[1].code

    def test_python_fallback(self):
        """Test that python blocks are also extracted."""
        response = '''```python
print("hello")
```'''
        blocks = extract_code_blocks(response, language="repl")
        assert len(blocks) == 1

    def test_no_blocks(self):
        """Test response with no code blocks."""
        response = "This is just plain text without any code."
        blocks = extract_code_blocks(response)
        assert len(blocks) == 0

    def test_empty_block(self):
        """Test that empty blocks are skipped."""
        response = '''```repl
```'''
        blocks = extract_code_blocks(response)
        assert len(blocks) == 0


class TestFindFinalInCode:
    """Tests for find_final_in_code function."""

    def test_final_with_string(self):
        """Test FINAL with string argument."""
        code = 'FINAL("The answer is 42")'
        answer, var_name = find_final_in_code(code)
        assert answer == "The answer is 42"
        assert var_name is None

    def test_final_with_single_quotes(self):
        """Test FINAL with single quotes."""
        code = "FINAL('The answer')"
        answer, var_name = find_final_in_code(code)
        assert answer == "The answer"

    def test_final_var(self):
        """Test FINAL_VAR detection."""
        code = 'FINAL_VAR("result")'
        answer, var_name = find_final_in_code(code)
        assert var_name == "result"

    def test_no_final(self):
        """Test code without FINAL."""
        code = "x = 10\nprint(x)"
        answer, var_name = find_final_in_code(code)
        assert answer is None
        assert var_name is None


class TestFormatExecutionResult:
    """Tests for format_execution_result function."""

    def test_basic_formatting(self):
        """Test basic result formatting."""
        result = format_execution_result(
            code="print('hello')",
            stdout="hello\n",
            stderr="",
            error=None,
            locals_snapshot={"x": 10},
        )
        assert "print('hello')" in result
        assert "hello" in result
        assert "x = 10" in result

    def test_error_formatting(self):
        """Test error is included."""
        result = format_execution_result(
            code="1/0",
            stdout="",
            stderr="",
            error="ZeroDivisionError: division by zero",
            locals_snapshot={},
        )
        assert "Error" in result
        assert "ZeroDivisionError" in result


class TestTruncateContextPreview:
    """Tests for truncate_context_preview function."""

    def test_short_string(self):
        """Test short string is not truncated."""
        result = truncate_context_preview("hello", max_length=100)
        assert result == "hello"

    def test_long_string(self):
        """Test long string is truncated."""
        long_text = "x" * 1000
        result = truncate_context_preview(long_text, max_length=100)
        assert len(result) < 200
        assert "more chars" in result

    def test_dict_context(self):
        """Test dict context formatting."""
        context = {"key": "value"}
        result = truncate_context_preview(context, max_length=100)
        assert "key" in result
        assert "value" in result
