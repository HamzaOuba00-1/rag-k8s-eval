"""
Étape 3 — Indexation.

Pour chacune des 3 stratégies de chunking (fixed, structure, semantic), on
construit DEUX index séparés :

1. BM25   : index "mot-clé", basé sur la fréquence des termes (rank_bm25).
            Pas d'IA ici, juste des statistiques. Fort quand la question
            contient les mots exacts de la doc (ex: noms de ressources K8s).
2. Dense  : chaque chunk est transformé en vecteur (embedding) avec un modèle
            de sentence-transformers, stocké dans un index FAISS pour une
            recherche par similarité cosinus. Fort pour capter le sens même
            sans les mots exacts.

L'index "hybride" ne se construit PAS ici : il se calcule à la volée à
l'étape retrieval, en combinant les scores BM25 + dense des deux index
qu'on construit maintenant. Pas besoin d'un 3e index séparé pour ça.

Structure de sortie :
  data/index/<strategie>/bm25.pkl          (index BM25 + corpus tokenisé)
  data/index/<strategie>/dense.faiss       (index vectoriel FAISS)
  data/index/<strategie>/chunks_meta.json  (métadonnées, MÊME ORDRE que les index)

Point important : l'ordre des chunks dans chunks_meta.json doit rester
strictement identique à l'ordre des vecteurs dans FAISS et des documents
dans BM25 — c'est ce qui permet de retrouver le texte/l'URL source à partir
d'un simple indice numérique renvoyé par la recherche.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")

STRATEGIES = ["fixed", "structure", "semantic"]

# Même modèle que pour le semantic chunking : cohérent, déjà en cache,
# rapide, tourne bien sur un GPU laptop comme le tien.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(strategy):
    path = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def tokenize(text):
    """Tokenisation simple pour BM25 : minuscule + mots alphanumériques."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(chunks, out_dir):
    print("  Construction de l'index BM25...")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)

    print(f"  BM25 sauvegardé -> {out_dir / 'bm25.pkl'}")


def build_dense(chunks, out_dir, model):
    print("  Construction de l'index dense (embeddings + FAISS)...")
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=64,
        normalize_embeddings=True,  # important : permet d'utiliser le produit
                                     # scalaire (Inner Product) comme mesure
                                     # de similarité cosinus dans FAISS
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product = cosinus (vecteurs normalisés)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "dense.faiss"))
    print(f"  FAISS sauvegardé -> {out_dir / 'dense.faiss'} ({index.ntotal} vecteurs)")


def main():
    print("Chargement du modèle d'embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Petite vérification utile : si CUDA n'est pas détecté, l'encodage
    # tournera sur CPU (toujours faisable pour ce volume, juste plus lent).
    import torch
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Device utilisé pour les embeddings : {device}")

    for strategy in STRATEGIES:
        print(f"\n=== Stratégie: {strategy} ===")
        out_dir = INDEX_DIR / strategy
        out_dir.mkdir(parents=True, exist_ok=True)

        chunks = load_chunks(strategy)
        print(f"  {len(chunks)} chunks chargés")

        # Métadonnées alignées : la ligne i correspond au vecteur i dans FAISS
        # et à l'entrée i de tokenized_corpus dans BM25 -> l'ORDRE compte.
        meta_path = out_dir / "chunks_meta.json"
        meta_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

        build_bm25(chunks, out_dir)
        build_dense(chunks, out_dir, model)

    print("\nIndexation terminée pour les 3 stratégies.")


if __name__ == "__main__":
    main()
