"""Safe builtins for REPL code execution."""

import builtins
from typing import Any

# Allowed builtin functions for safe code execution
ALLOWED_BUILTINS = [
    # Type constructors
    "bool",
    "bytes",
    "bytearray",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "set",
    "str",
    "tuple",
    # Iteration
    "enumerate",
    "filter",
    "iter",
    "map",
    "next",
    "range",
    "reversed",
    "sorted",
    "zip",
    # Math
    "abs",
    "divmod",
    "max",
    "min",
    "pow",
    "round",
    "sum",
    # String/repr
    "ascii",
    "bin",
    "chr",
    "format",
    "hex",
    "oct",
    "ord",
    "repr",
    # Type checking
    "callable",
    "hasattr",
    "isinstance",
    "issubclass",
    "type",
    # Attribute access
    "getattr",
    "setattr",
    "delattr",
    # Length/membership
    "len",
    "all",
    "any",
    # Printing (will be captured)
    "print",
    # Object creation
    "object",
    "slice",
    "staticmethod",
    "classmethod",
    "property",
    # Misc
    "id",
    "hash",
    "dir",
    "vars",
    # Exceptions (needed for try/except)
    "Exception",
    "BaseException",
    "TypeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "StopIteration",
    "ZeroDivisionError",
    "NameError",
    "ImportError",
    "AssertionError",
    "NotImplementedError",
    # Boolean constants
    "True",
    "False",
    "None",
]

# Explicitly blocked builtins (dangerous)
BLOCKED_BUILTINS = [
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "globals",
    "locals",
    "breakpoint",
    "memoryview",
    "help",
    "credits",
    "license",
    "copyright",
    "quit",
    "exit",
]


def create_safe_builtins() -> dict[str, Any]:
    """
    Create a dictionary of safe builtins for REPL execution.

    Returns:
        Dictionary of allowed builtin names to their values
    """
    safe = {}

    for name in ALLOWED_BUILTINS:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
        elif name in ("True", "False", "None"):
            # These are keywords, not builtins, but we include them
            safe[name] = {"True": True, "False": False, "None": None}[name]

    return safe


# Pre-built safe builtins dictionary
SAFE_BUILTINS = create_safe_builtins()


# Safe modules that can be imported
SAFE_MODULES = {
    "re": None,  # Will be imported lazily
    "json": None,
    "math": None,
    "datetime": None,
    "collections": None,
    "itertools": None,
    "functools": None,
    "operator": None,
    "string": None,
    "textwrap": None,
    "difflib": None,
    "statistics": None,
    "random": None,  # Useful for sampling
    "copy": None,
    "pprint": None,
}


def get_safe_module(name: str) -> Any:
    """
    Get a safe module by name.

    Args:
        name: Module name (must be in SAFE_MODULES)

    Returns:
        The imported module

    Raises:
        ImportError: If module is not in safe list
    """
    if name not in SAFE_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in REPL environment")

    if SAFE_MODULES[name] is None:
        import importlib

        SAFE_MODULES[name] = importlib.import_module(name)

    return SAFE_MODULES[name]


def create_restricted_import() -> Any:
    """
    Create a restricted __import__ function that only allows safe modules.

    Returns:
        Restricted import function
    """

    def restricted_import(
        name: str,
        _globals: dict | None = None,
        _locals: dict | None = None,
        _fromlist: tuple = (),
        _level: int = 0,
    ) -> Any:
        """Restricted import that only allows safe modules."""
        # Handle "from X import Y" syntax
        base_module = name.split(".")[0]

        if base_module not in SAFE_MODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(SAFE_MODULES.keys()))}"
            )

        return get_safe_module(base_module)

    return restricted_import
