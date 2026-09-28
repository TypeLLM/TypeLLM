"""Public compatibility facade for the TypeLLM prototype.

The implementation is split by responsibility; imports from this module remain
stable for existing callers, and ``python -m typellm`` launches the demo.
"""

from .benchmark import benchmark_prefix_cache
from .runtime import (
    Choice,
    Generation,
    TypeLLMClient,
    candidate_softmax,
    run_schema,
)
from .schema import (
    MAX_ENUM_CHOICES,
    Decision,
    SchemaError,
    compile_json_schema,
)
from .sglang import (
    GenerationCancelled,
    GenerationTimeout,
    SGLangClient,
    SGLangError,
    Usage,
    extract_candidate_logprobs,
)
from .cli import example_schema, main

__all__ = [
    "Choice",
    "Decision",
    "Generation",
    "GenerationCancelled",
    "GenerationTimeout",
    "MAX_ENUM_CHOICES",
    "SGLangClient",
    "SGLangError",
    "SchemaError",
    "TypeLLMClient",
    "Usage",
    "benchmark_prefix_cache",
    "candidate_softmax",
    "compile_json_schema",
    "example_schema",
    "run_schema",
]
