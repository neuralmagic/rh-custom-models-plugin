"""vLLM model definitions not carried upstream.

Each model family is a subpackage with a ``register()`` function and an
``ARCHITECTURES`` map, exposed as its own ``vllm.general_plugins`` entry
point. Importing this package registers nothing.
"""

__version__ = "0.1.0"
