"""LLM factory — returns a ChatOpenAI client pointed at LM Studio."""

from langchain_openai import ChatOpenAI

from georeasoner.config import settings


def get_llm(**kwargs: object) -> ChatOpenAI:
    """Return a ChatOpenAI instance wired to the local LM Studio endpoint."""
    return ChatOpenAI(
        model=settings.lm_studio_model,
        base_url=settings.lm_studio_url,
        api_key=settings.lm_studio_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        **kwargs,
    )
