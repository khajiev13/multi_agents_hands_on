from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import TutorSettings


def build_model(settings: TutorSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.lab_tutor_llm_model,
        base_url=settings.lab_tutor_llm_base_url,
        api_key=settings.lab_tutor_llm_api_key,
        temperature=0,
        max_retries=2,
        timeout=120,
    )


def build_ocr_model(settings: TutorSettings) -> ChatOpenAI:
    config = settings.require_ocr()
    return ChatOpenAI(
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=0,
        max_tokens=32000,
        max_retries=2,
        timeout=180,
    )
