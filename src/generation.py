"""
Étape 5 — Génération avec citations obligatoires.

Principe : le LLM reçoit les chunks récupérés (après hybride + reranking),
NUMÉROTÉS. Le prompt lui impose de citer le numéro du chunk après chaque
affirmation ([1], [2]...). On affiche ensuite la correspondance numéro ->
(titre, URL réelle) séparément : impossible pour le modèle d'inventer une
source, puisqu'il ne peut citer que les numéros qu'on lui a donnés.

Ce module réutilise retrieval.py pour aller chercher le contexte, donc il
doit être lancé depuis la racine du projet (comme les scripts précédents).
"""

import sys
from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder

from retrieval import StrategyIndex, search_hybrid, rerank, get_chunks, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

OLLAMA_MODEL = "qwen2.5:7b-instruct"

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


def build_context_block(chunks_with_scores):
    """Construit le bloc de contexte numéroté à injecter dans le prompt."""
    lines = []
    for i, item in enumerate(chunks_with_scores, start=1):
        chunk = item["chunk"]
        lines.append(f"[Extrait {i}] (source: {chunk['title']})\n{chunk['text']}\n")
    return "\n".join(lines)


def call_llm(query: str, top_chunks: list):
    """Appelle le LLM avec un contexte déjà récupéré (peu importe la méthode utilisée)."""
    context_block = build_context_block(top_chunks)

    user_prompt = f"""Extraits de documentation :

{context_block}

Question : {query}

Réponds en respectant strictement les règles données, en citant les numéros \
d'extraits entre crochets après chaque affirmation."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response["message"]["content"]

    sources = [
        {
            "n": i + 1,
            "title": c["chunk"]["title"],
            "url": c["chunk"]["source_url"],
        }
        for i, c in enumerate(top_chunks)
    ]
    return answer, sources


def generate_answer(query: str, strategy: str = "structure", top_k_final: int = 5):
    index = StrategyIndex(strategy)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

    top_chunks = get_chunks("hybrid_rerank", query, index, embed_model, cross_encoder, top_k=top_k_final)
    answer, sources = call_llm(query, top_chunks)

    return {"query": query, "answer": answer, "sources": sources}


def print_result(result):
    print("\n" + "=" * 70)
    print(f"QUESTION : {result['query']}")
    print("=" * 70)
    print(f"\n{result['answer']}\n")
    print("-" * 70)
    print("SOURCES :")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['title']}")
        print(f"      {s['url']}")


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "structure"
    query = sys.argv[2] if len(sys.argv) > 2 else "How do I create a Kubernetes Deployment?"

    result = generate_answer(query, strategy=strategy)
    print_result(result)
