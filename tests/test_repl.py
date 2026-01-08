"""Tests for REPL environment."""

import pytest

from rlm_langgraph import REPLEnvironment


class TestREPLEnvironment:
    """Tests for REPLEnvironment class."""

    def test_context_available(self):
        """Test that context is available in REPL."""
        repl = REPLEnvironment("test context")
        result = repl.execute("print(context)")
        assert result.stdout.strip() == "test context"
        assert result.error is None

    def test_basic_execution(self):
        """Test basic code execution."""
        repl = REPLEnvironment("hello")
        result = repl.execute("x = len(context)\nprint(x)")
        assert result.stdout.strip() == "5"
        assert result.error is None

    def test_variable_persistence(self):
        """Test that variables persist between executions."""
        repl = REPLEnvironment("test")
        repl.execute("x = 10")
        result = repl.execute("print(x * 2)")
        assert result.stdout.strip() == "20"

    def test_safe_builtins(self):
        """Test that safe builtins are available."""
        repl = REPLEnvironment([1, 2, 3])
        result = repl.execute("print(len(context), sum(context), max(context))")
        assert result.stdout.strip() == "3 6 3"

    def test_blocked_builtins(self):
        """Test that dangerous builtins are blocked."""
        repl = REPLEnvironment("test")
        result = repl.execute("open('test.txt')")
        assert result.error is not None
        assert "NameError" in result.error

    def test_safe_imports(self):
        """Test that safe imports work."""
        repl = REPLEnvironment('{"key": "value"}')
        result = repl.execute("import json\ndata = json.loads(context)\nprint(data['key'])")
        assert result.stdout.strip() == "value"
        assert result.error is None

    def test_blocked_imports(self):
        """Test that unsafe imports are blocked."""
        repl = REPLEnvironment("test")
        result = repl.execute("import os")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_llm_query_placeholder(self):
        """Test that llm_query returns placeholder."""
        repl = REPLEnvironment("test")
        result = repl.execute("response = llm_query('What is 2+2?')")
        assert result.error is None
        assert len(result.sub_llm_calls) == 1
        assert result.sub_llm_calls[0].prompt == "What is 2+2?"

    def test_llm_query_batched(self):
        """Test that llm_query_batched returns placeholders."""
        repl = REPLEnvironment("test")
        result = repl.execute("responses = llm_query_batched(['Q1', 'Q2', 'Q3'])")
        assert result.error is None
        assert len(result.sub_llm_calls) == 3

    def test_final_answer(self):
        """Test FINAL() function."""
        repl = REPLEnvironment("test")
        result = repl.execute("FINAL('The answer is 42')")
        assert result.final_answer == "The answer is 42"

    def test_final_var(self):
        """Test FINAL_VAR() function."""
        repl = REPLEnvironment("test")
        repl.execute("answer = 42")
        result = repl.execute("FINAL_VAR('answer')")
        assert result.final_var_name == "answer"
        assert repl.get_final_answer() == "42"

    def test_error_handling(self):
        """Test that errors are captured properly."""
        repl = REPLEnvironment("test")
        result = repl.execute("1/0")
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_stdout_capture(self):
        """Test that stdout is captured."""
        repl = REPLEnvironment("test")
        result = repl.execute("print('hello')\nprint('world')")
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_json_context(self):
        """Test with JSON context."""
        context = {"name": "Alice", "age": 30}
        repl = REPLEnvironment(context)
        result = repl.execute("print(context['name'])")
        assert result.stdout.strip() == "Alice"

    def test_list_context(self):
        """Test with list context."""
        context = ["apple", "banana", "cherry"]
        repl = REPLEnvironment(context)
        result = repl.execute("print(len(context), context[1])")
        assert result.stdout.strip() == "3 banana"

    def test_reset(self):
        """Test environment reset."""
        repl = REPLEnvironment("test")
        repl.execute("x = 100")
        repl.reset()
        result = repl.execute("print(x)")
        assert result.error is not None
        assert "NameError" in result.error
