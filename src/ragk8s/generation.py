"""Answer generation with mandatory [n] citations (Ollama).

The retrieved chunks are numbered in the prompt; the model may only cite
those numbers, and the number -> (title, URL) mapping is attached afterwards,
so a cited source cannot be invented.

Note: the prompt below is the original one (French instructions) and is kept
unchanged in this refactoring step; it is revised in the next step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import config
from .retrieval import Hit

SYSTEM_PROMPT = """Tu es un assistant technique qui répond à des questions sur Kubernetes \
en te basant UNIQUEMENT sur les extraits de documentation fournis ci-dessous.

RÈGLES STRICTES :
1. N'utilise que les informations présentes dans les extraits fournis. \
Si la réponse ne s'y trouve pas, dis clairement "Je ne trouve pas cette information \
dans la documentation fournie."
2. Après CHAQUE affirmation factuelle, indique le numéro de l'extrait qui la \
justifie, entre crochets, ex: [1] ou [2][3] si plusieurs sources la confirment.
3. Ne mélange jamais tes connaissances générales avec le contenu des extraits : \
si ce n'est pas dans les extraits, ne l'affirme pas.
4. Réponds de façon concise et structurée.
"""

USER_TEMPLATE = """Extraits de documentation :

{context}

Question : {question}

Réponds en respectant strictement les règles données, en citant les numéros \
d'extraits entre crochets après chaque affirmation."""


def build_context(hits: Sequence[Hit]) -> str:
    return "\n".join(
        f"[Extrait {i}] (source: {h.chunk['title']})\n{h.chunk['text']}\n" for i, h in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=build_context(hits), question=question)},
    ]


def sources(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    return [{"n": i, "title": h.chunk["title"], "url": h.chunk["source_url"]} for i, h in enumerate(hits, start=1)]


def generate(question: str, hits: Sequence[Hit], model: str = config.OLLAMA_MODEL) -> str:
    import ollama

    response = ollama.chat(model=model, messages=build_messages(question, hits))
    return str(response["message"]["content"])
