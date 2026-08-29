"""CacheLLM - a local-first caching proxy for OpenAI-compatible LLM APIs.

Layers (see README for the full architecture):

    proxy/API layer      cachellm.server
    cache engine         cachellm.cache
    policy engine        cachellm.policy
    database             cachellm.db
    provider adapter     cachellm.providers
    cost calculator      cachellm.pricing
    statistics           cachellm.stats
    configuration        cachellm.config
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
