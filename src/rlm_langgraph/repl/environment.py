"""REPL Environment for executing LLM-generated code."""

import io
import json
import sys
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .safe_builtins import SAFE_BUILTINS, create_restricted_import, get_safe_module


@dataclass
class SubLLMCallRecord:
    """Record of an llm_query call made during code execution."""

    call_id: str
    prompt: str
    model: str | None = None
    response: str | None = None


@dataclass
class REPLResult:
    """Result from executing code in the REPL environment."""

    code: str
    stdout: str
    stderr: str
    error: str | None
    sub_llm_calls: list[SubLLMCallRecord] = field(default_factory=list)
    locals_snapshot: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    final_var_name: str | None = None


class REPLEnvironment:
    """
    A sandboxed Python REPL environment for RLM code execution.

    This environment:
    - Stores context as a variable accessible to LLM-generated code
    - Provides llm_query() and llm_query_batched() functions
    - Captures stdout/stderr
    - Tracks sub-LLM calls for later processing
    - Uses restricted builtins for safety
    """

    def __init__(self, context: str | dict | list):
        """
        Initialize the REPL environment.

        Args:
            context: The context to make available as a variable
        """
        self.context = context
        self.locals: dict[str, Any] = {}
        self.pending_llm_calls: list[SubLLMCallRecord] = []
        self._final_answer: str | None = None
        self._final_var_name: str | None = None

        # Initialize the namespace
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        """Set up the initial namespace with context and helper functions."""
        # Start with safe builtins
        self.locals = {"__builtins__": SAFE_BUILTINS.copy()}

        # Add restricted import
        self.locals["__builtins__"]["__import__"] = create_restricted_import()

        # Add context variable
        self.locals["context"] = self.context

        # Add safe modules (pre-import commonly used ones)
        self.locals["re"] = get_safe_module("re")
        self.locals["json"] = get_safe_module("json")
        self.locals["math"] = get_safe_module("math")

        # Add LLM query functions
        self.locals["llm_query"] = self._create_llm_query()
        self.locals["llm_query_batched"] = self._create_llm_query_batched()

        # Add FINAL functions
        self.locals["FINAL"] = self._create_final()
        self.locals["FINAL_VAR"] = self._create_final_var()

    def _create_llm_query(self) -> Callable[[str, str | None], str]:
        """
        Create the llm_query function for the REPL.

        Returns a placeholder that records calls for later processing.
        """

        def llm_query(prompt: str, model: str | None = None) -> str:
            """
            Query an LLM with the given prompt.

            Args:
                prompt: The prompt to send to the LLM
                model: Optional model override (uses recursive_model by default)

            Returns:
                Placeholder string (will be replaced with actual response)
            """
            call_id = f"llm_call_{uuid.uuid4().hex[:8]}"
            record = SubLLMCallRecord(
                call_id=call_id, prompt=prompt, model=model, response=None
            )
            self.pending_llm_calls.append(record)

            # Return placeholder that will be replaced
            return f"<<PENDING_LLM_CALL:{call_id}>>"

        return llm_query

    def _create_llm_query_batched(self) -> Callable[[list[str], str | None], list[str]]:
        """
        Create the llm_query_batched function for the REPL.

        Returns placeholders for batch calls.
        """

        def llm_query_batched(
            prompts: list[str], model: str | None = None
        ) -> list[str]:
            """
            Query an LLM with multiple prompts in batch.

            Args:
                prompts: List of prompts to send
                model: Optional model override

            Returns:
                List of placeholder strings
            """
            results = []
            for prompt in prompts:
                call_id = f"llm_call_{uuid.uuid4().hex[:8]}"
                record = SubLLMCallRecord(
                    call_id=call_id, prompt=prompt, model=model, response=None
                )
                self.pending_llm_calls.append(record)
                results.append(f"<<PENDING_LLM_CALL:{call_id}>>")

            return results

        return llm_query_batched

    def _create_final(self) -> Callable[[str], None]:
        """Create the FINAL function for returning answers."""

        def final(answer: str) -> None:
            """
            Signal that this is the final answer.

            Args:
                answer: The final answer string
            """
            self._final_answer = str(answer)
            print(f"FINAL ANSWER: {answer}")

        return final

    def _create_final_var(self) -> Callable[[str], None]:
        """Create the FINAL_VAR function for returning variable values."""

        def final_var(var_name: str) -> None:
            """
            Signal that a variable contains the final answer.

            Args:
                var_name: Name of the variable containing the answer
            """
            self._final_var_name = var_name
            if var_name in self.locals:
                value = self.locals[var_name]
                print(f"FINAL ANSWER (from {var_name}): {value}")
            else:
                print(f"WARNING: Variable '{var_name}' not found")

        return final_var

    @contextmanager
    def _capture_output(self):
        """Context manager to capture stdout and stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        new_stdout = io.StringIO()
        new_stderr = io.StringIO()

        try:
            sys.stdout = new_stdout
            sys.stderr = new_stderr
            yield new_stdout, new_stderr
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def execute(self, code: str) -> REPLResult:
        """
        Execute code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with stdout, stderr, errors, and sub-LLM calls
        """
        # Reset state for this execution
        self.pending_llm_calls = []
        self._final_answer = None
        self._final_var_name = None

        error = None
        stdout_content = ""
        stderr_content = ""

        with self._capture_output() as (stdout_capture, stderr_capture):
            try:
                # Execute the code
                exec(code, self.locals, self.locals)
            except Exception as e:
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()

        # Create snapshot of relevant locals (excluding functions and modules)
        locals_snapshot = {}
        for key, value in self.locals.items():
            if key.startswith("_") or key == "__builtins__":
                continue
            if callable(value) and key in (
                "llm_query",
                "llm_query_batched",
                "FINAL",
                "FINAL_VAR",
            ):
                continue
            if hasattr(value, "__module__"):  # Skip module objects
                continue

            try:
                # Try to make it JSON-serializable for inspection
                json.dumps(value)
                locals_snapshot[key] = value
            except (TypeError, ValueError):
                # Store a string representation for non-serializable objects
                locals_snapshot[key] = f"<{type(value).__name__}: {repr(value)[:100]}>"

        return REPLResult(
            code=code,
            stdout=stdout_content,
            stderr=stderr_content,
            error=error,
            sub_llm_calls=self.pending_llm_calls.copy(),
            locals_snapshot=locals_snapshot,
            final_answer=self._final_answer,
            final_var_name=self._final_var_name,
        )

    def inject_llm_responses(self, responses: dict[str, str]) -> None:
        """
        Inject LLM responses back into the namespace.

        This replaces placeholder values with actual responses.

        Args:
            responses: Mapping of call_id to response string
        """
        # Update any variables that contain placeholders
        for key, value in list(self.locals.items()):
            if isinstance(value, str):
                for call_id, response in responses.items():
                    placeholder = f"<<PENDING_LLM_CALL:{call_id}>>"
                    if placeholder in value:
                        self.locals[key] = value.replace(placeholder, response)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, str):
                        new_item = item
                        for call_id, response in responses.items():
                            placeholder = f"<<PENDING_LLM_CALL:{call_id}>>"
                            if placeholder in new_item:
                                new_item = new_item.replace(placeholder, response)
                        new_list.append(new_item)
                    else:
                        new_list.append(item)
                self.locals[key] = new_list

    def get_final_answer(self) -> str | None:
        """
        Get the final answer if one was provided.

        Returns:
            The final answer string, or None if not yet provided
        """
        if self._final_answer is not None:
            return self._final_answer

        if self._final_var_name is not None and self._final_var_name in self.locals:
            value = self.locals[self._final_var_name]
            return str(value)

        return None

    def reset(self) -> None:
        """Reset the REPL environment to initial state."""
        self._setup_namespace()
        self.pending_llm_calls = []
        self._final_answer = None
        self._final_var_name = None
