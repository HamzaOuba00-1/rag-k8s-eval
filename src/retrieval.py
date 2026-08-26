"""
Étape 4 — Retrieval + reranking.

Ce module expose 4 fonctions de recherche sur un index donné (une stratégie
de chunking : fixed / structure / semantic) :

  - search_bm25    : recherche mot-clé pure
  - search_dense   : recherche sémantique pure
  - search_hybrid  : fusion des deux classements via Reciprocal Rank Fusion
  - rerank         : reclassement fin des meilleurs candidats via cross-encoder

Lancé directement (python src/retrieval.py), il fait une démo comparative
sur une question donnée pour UNE stratégie, histoire de voir concrètement
la différence entre les méthodes avant de passer à la génération.
"""

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_DIR = Path("data/index")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Modèle spécialisé reranking : plus lent, mais bien plus précis sur un
# petit lot de candidats. Il prend la question ET le chunk ensemble en
# entrée (contrairement aux embeddings, qui les encodent séparément) —
# c'est ce qui le rend plus précis mais aussi plus coûteux.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# --- Chargement d'un index ---------------------------------------------------

class StrategyIndex:
    """Regroupe tout ce qu'il faut pour chercher dans UNE stratégie de chunking."""

    def __init__(self, strategy: str):
        strat_dir = INDEX_DIR / strategy
        self.strategy = strategy

        self.chunks_meta = json.loads((strat_dir / "chunks_meta.json").read_text(encoding="utf-8"))

        with open(strat_dir / "bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]

        self.faiss_index = faiss.read_index(str(strat_dir / "dense.faiss"))


# --- Recherche BM25 -----------------------------------------------------------

def search_bm25(query: str, index: StrategyIndex, top_k: int = 10):
    scores = index.bm25.get_scores(tokenize(query))
    top_ids = np.argsort(scores)[::-1][:top_k]
    return [
        {"chunk": index.chunks_meta[i], "score": float(scores[i]), "rank": rank + 1}
        for rank, i in enumerate(top_ids)
    ]


# --- Recherche dense -----------------------------------------------------------

def search_dense(query: str, index: StrategyIndex, embed_model: SentenceTransformer, top_k: int = 10):
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.faiss_index.search(query_vec, top_k)
    return [
        {"chunk": index.chunks_meta[i], "score": float(s), "rank": rank + 1}
        for rank, (s, i) in enumerate(zip(scores[0], ids[0]))
    ]


# --- Fusion hybride (Reciprocal Rank Fusion) -----------------------------------

def search_hybrid(query: str, index: StrategyIndex, embed_model: SentenceTransformer,
                   top_k: int = 10, k_rrf: int = 60, candidate_pool: int = 30):
    """
    RRF : pour chaque document, on somme 1 / (k_rrf + rang) sur chaque
    méthode de recherche où il apparaît. k_rrf (typiquement 60) amortit
    l'impact des tout premiers rangs pour ne pas laisser une seule méthode
    dominer complètement le classement final.
    """
    bm25_results = search_bm25(query, index, top_k=candidate_pool)
    dense_results = search_dense(query, index, embed_model, top_k=candidate_pool)

    rrf_scores = {}
    chunk_lookup = {}

    for results in (bm25_results, dense_results):
        for r in results:
            chunk_id = r["chunk"]["chunk_id"]
            chunk_lookup[chunk_id] = r["chunk"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + r["rank"])

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"chunk": chunk_lookup[chunk_id], "score": score, "rank": rank + 1}
        for rank, (chunk_id, score) in enumerate(ranked)
    ]


# --- Reranking -----------------------------------------------------------------

def rerank(query: str, candidates: list, cross_encoder: CrossEncoder, top_k: int = 5):
    pairs = [(query, c["chunk"]["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    for rank, c in enumerate(reranked):
        c["rank"] = rank + 1
    return reranked


# --- Démo comparative ------------------------------------------------------------

def print_results(title, results):
    print(f"\n### {title} ###")
    for r in results[:5]:
        chunk = r["chunk"]
        score = r.get("rerank_score", r["score"])
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"  [{r['rank']}] score={score:.4f} | {chunk['title']}")
        print(f"      {preview}...")


def get_chunks(method: str, query: str, index: StrategyIndex, embed_model: SentenceTransformer,
                cross_encoder: CrossEncoder = None, top_k: int = 5):
    """
    Dispatch vers la bonne méthode de retrieval selon son nom.
    method: "bm25" | "dense" | "hybrid" | "hybrid_rerank"
    """
    if method == "bm25":
        return search_bm25(query, index, top_k=top_k)
    elif method == "dense":
        return search_dense(query, index, embed_model, top_k=top_k)
    elif method == "hybrid":
        return search_hybrid(query, index, embed_model, top_k=top_k)
    elif method == "hybrid_rerank":
        candidates = search_hybrid(query, index, embed_model, top_k=20)
        return rerank(query, candidates, cross_encoder, top_k=top_k)
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def main():
    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "structure"
    query = sys.argv[2] if len(sys.argv) > 2 else "How do I create a Kubernetes Deployment?"

    print(f"Stratégie: {strategy}")
    print(f"Question : {query}")

    index = StrategyIndex(strategy)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

    bm25_res = search_bm25(query, index, top_k=10)
    dense_res = search_dense(query, index, embed_model, top_k=10)
    hybrid_res = search_hybrid(query, index, embed_model, top_k=10)
    reranked_res = rerank(query, search_hybrid(query, index, embed_model, top_k=20), cross_encoder, top_k=5)

    print_results("BM25", bm25_res)
    print_results("Dense", dense_res)
    print_results("Hybride (RRF)", hybrid_res)
    print_results("Hybride + Reranking", reranked_res)


if __name__ == "__main__":
    main()
