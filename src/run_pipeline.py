"""
Étape 7a — Exécution du pipeline complet sur le jeu de questions annoté,
pour UNE combinaison (stratégie de chunking, méthode de retrieval).

Produit results/run_<strategy>_<method>.jsonl avec, pour chaque question :
la réponse générée, les contextes utilisés (textes des chunks), la vraie
réponse (ground_truth) et la catégorie (factual/impossible).

Ce fichier sera ensuite passé à evaluate_ragas.py pour calculer les métriques.

Usage :
  python src/run_pipeline.py <strategy> <method> [limit]

  strategy : fixed | structure | semantic
  method   : bm25 | dense | hybrid | hybrid_rerank
  limit    : (optionnel) ne traiter que les N premières questions,
             utile pour un premier test rapide avant de lancer sur les 60.
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from retrieval import StrategyIndex, get_chunks, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
from generation import call_llm

EVAL_DIR = Path("eval")
RESULTS_DIR = Path("results")


def load_questions(limit_factual=None):
    path = EVAL_DIR / "questions.jsonl"
    factual, impossible = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            (factual if q.get("category", "factual") == "factual" else impossible).append(q)

    if limit_factual:
        factual = factual[:limit_factual]

    return factual + impossible


def run(strategy, method, top_k=5, limit_factual=None):
    print(f"=== Pipeline : strategy={strategy}, method={method} ===")
    questions = load_questions(limit_factual=limit_factual)
    print(f"{len(questions)} questions à traiter "
          f"({sum(1 for q in questions if q.get('category') == 'factual')} factual + "
          f"{sum(1 for q in questions if q.get('category') == 'impossible')} impossible)")

    index = StrategyIndex(strategy)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Le cross-encoder n'est chargé que si on en a besoin (économise du temps sinon)
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME) if method == "hybrid_rerank" else None

    results = []
    for i, q in enumerate(questions):
        top_chunks = get_chunks(method, q["question"], index, embed_model, cross_encoder, top_k=top_k)
        answer, sources = call_llm(q["question"], top_chunks)

        results.append({
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "category": q.get("category", "factual"),
            "answer": answer,
            "contexts": [c["chunk"]["text"] for c in top_chunks],
            "sources": sources,
        })
        print(f"  [{i + 1}/{len(questions)}] traité")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"run_{strategy}_{method}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(results)} réponses générées -> {out_path}")


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "structure"
    method = sys.argv[2] if len(sys.argv) > 2 else "hybrid_rerank"
    limit_factual = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run(strategy, method, limit_factual=limit_factual)