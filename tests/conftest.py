"""Shared test fixtures.

src/llm.py::_call_raw dispatches on LLM_PROVIDER, so patching a single
provider function (e.g. src.llm._openai) does NOT stop a call when
LLM_PROVIDER points somewhere else — the request silently goes out to the
network for real. Tests patch the provider-agnostic seams instead:

    src.llm.call                  — every structured model call
    src.embeddings.get_embeddings — every embedding call

_block_network is the backstop: anything that slips past those seams fails
loudly instead of billing a live API.
"""
import hashlib

import numpy as np
import pytest
from unittest.mock import patch

EMBED_DIM = 16


def fake_embeddings(texts, model=None, input_type="passage"):
    """Deterministic vector per text, one per input.

    Returning exactly len(texts) vectors matters: VectorIndex.build() adds the
    result to FAISS but keeps its own self._chunks list, so a count mismatch
    desyncs the two and query() hits an out-of-range index.
    """
    out = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out.append([float(x) for x in rng.random(EMBED_DIM)])
    return out


@pytest.fixture(autouse=True)
def _block_network():
    """Fail loudly if a test reaches a real provider."""

    def _boom(name):
        def _raise(*args, **kwargs):
            raise AssertionError(
                f"Test attempted a real {name} API call. Patch src.llm.call or "
                f"src.embeddings.get_embeddings instead of a provider function."
            )

        return _raise

    with patch("src.llm._nvidia", side_effect=_boom("nvidia")), \
         patch("src.llm._openai", side_effect=_boom("openai")), \
         patch("src.llm._anthropic", side_effect=_boom("anthropic")), \
         patch("src.llm._gemini", side_effect=_boom("gemini")), \
         patch("src.llm._nvidia_embeddings", side_effect=_boom("nvidia embeddings")), \
         patch("src.llm._openai_embeddings", side_effect=_boom("openai embeddings")), \
         patch("src.llm._gemini_embeddings", side_effect=_boom("gemini embeddings")):
        yield
