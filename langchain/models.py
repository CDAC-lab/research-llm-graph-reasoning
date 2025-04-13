from typing import Literal
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def get_llm(llm_type: Literal["llama3", "openai"],
            llm_model: str,
            api_key: str|SecretStr,
            temperature: int = 0):
    if llm_type == "llama3":
        return ChatGroq(temperature=temperature, model=llm_model, api_key=api_key)
    elif llm_type == "openai":
        return ChatOpenAI(model=llm_model, api_key=api_key)
    else:
        raise ValueError("Invalid LLM type")