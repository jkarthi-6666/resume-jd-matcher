"""Thin LLM provider wrapper. Every model call goes through call()."""
import json
from pydantic import BaseModel
from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,
    NVIDIA_API_KEY, NVIDIA_BASE_URL, EMBED_PROVIDER, EMBED_MODEL,
)


def call(
    prompt: str,
    model: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
    system: str = "You are a helpful assistant. Return valid JSON only.",
) -> BaseModel:
    raw = _call_raw(prompt, model, temperature, system)
    return _parse(raw, schema)


def call_raw(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    system: str = "You are a helpful assistant.",
) -> str:
    return _call_raw(prompt, model, temperature, system)


def _call_raw(prompt: str, model: str, temperature: float, system: str) -> str:
    if LLM_PROVIDER == "nvidia":
        return _nvidia(prompt, model, temperature, system)
    if model.startswith("gemini") or LLM_PROVIDER == "gemini":
        return _gemini(prompt, model, temperature, system)
    if model.startswith("gpt") or LLM_PROVIDER == "openai":
        return _openai(prompt, model, temperature, system)
    if model.startswith("claude") or LLM_PROVIDER == "anthropic":
        return _anthropic(prompt, model, temperature, system)
    return _nvidia(prompt, model, temperature, system)


def _nvidia(prompt: str, model: str, temperature: float, system: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _gemini(prompt: str, model: str, temperature: float, system: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system + " Return valid JSON only.",
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def _openai(prompt: str, model: str, temperature: float, system: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _anthropic(prompt: str, model: str, temperature: float, system: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system + " Return valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _parse(raw: str, schema: type[BaseModel]) -> BaseModel:
    try:
        data = json.loads(raw)
        return schema.model_validate(data)
    except Exception as e:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(cleaned)
            return schema.model_validate(data)
        except Exception:
            raise ValueError(f"Could not parse model response: {e}\nRaw: {raw[:500]}")


def get_embeddings(
    texts: list[str],
    model: str | None = None,
    input_type: str = "passage",
) -> list[list[float]]:
    """
    input_type: "passage" for documents being indexed, "query" for search queries.
    Only used by asymmetric NVIDIA models; ignored by other providers.
    """
    model = model or EMBED_MODEL
    if EMBED_PROVIDER == "nvidia":
        return _nvidia_embeddings(texts, model, input_type)
    if EMBED_PROVIDER == "gemini":
        return _gemini_embeddings(texts, model)
    return _openai_embeddings(texts, model)


def _nvidia_embeddings(texts: list[str], model: str, input_type: str = "passage") -> list[list[float]]:
    """NVIDIA NIM embeddings — OpenAI-compatible, supports batching."""
    from openai import OpenAI
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    response = client.embeddings.create(
        model=model,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def _gemini_embeddings(texts: list[str], model: str) -> list[list[float]]:
    from google import genai
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = genai.Client(api_key=GEMINI_API_KEY)

    def _embed_one(idx_text: tuple[int, str]) -> tuple[int, list[float]]:
        idx, text = idx_text
        response = client.models.embed_content(model=model, contents=text)
        embedding = response.embeddings[0]
        values = embedding.values if hasattr(embedding, "values") else list(embedding)
        return idx, values

    results: list[list[float]] = [[] for _ in texts]
    with ThreadPoolExecutor(max_workers=min(len(texts), 10)) as pool:
        futures = {pool.submit(_embed_one, (i, t)): i for i, t in enumerate(texts)}
        for future in as_completed(futures):
            idx, values = future.result()
            results[idx] = values
    return results


def _openai_embeddings(texts: list[str], model: str) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
