"""
Étape 2 — Chunking : 3 stratégies à comparer.

1. fixed     : découpage brut par nombre de caractères + overlap (baseline naïf)
2. structure : découpage qui respecte les titres Markdown (## , ###)
3. semantic  : découpage par similarité sémantique entre phrases consécutives

Chaque stratégie produit un fichier data/processed/chunks_<strategie>.jsonl
où chaque ligne est un chunk avec ses métadonnées (doc_id, source_url, texte).

Pourquoi JSONL (JSON Lines) et pas un gros JSON ? Parce qu'on va avoir des
milliers de chunks, et JSONL permet de lire/écrire ligne par ligne sans
charger tout le fichier en mémoire — bonne pratique pour des données qui
grossissent.
"""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Modèle d'embeddings utilisé UNIQUEMENT pour le semantic chunking ici
# (l'indexation dense à l'étape 3 aura son propre choix de modèle, qui peut
# être le même ou un autre — on garde les deux découplés).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # petit, rapide, tourne bien en local


# --- Chargement du corpus ---------------------------------------------------

def load_corpus():
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    docs = []
    for entry in manifest:
        text = (RAW_DIR / f"{entry['doc_id']}.md").read_text(encoding="utf-8")
        docs.append({**entry, "text": text})
    return docs


# --- Stratégie 1 : fixed-size (baseline naïf) -------------------------------

def chunk_fixed(text: str, chunk_size: int = 1000, overlap: int = 150):
    """
    Découpe brute par nombre de caractères, avec chevauchement.
    L'overlap sert à éviter de perdre le contexte pile à la frontière
    entre deux chunks (une phrase coupée en 2 aura une chance d'être
    entière dans au moins un des deux chunks).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --- Stratégie 2 : structure-aware (respecte les titres Markdown) ----------

def chunk_structure(text: str, max_chunk_size: int = 1500):
    """
    Découpe d'abord par titres Markdown (## ou ###), pour que chaque chunk
    corresponde à une vraie section logique du document. Si une section est
    trop longue, on la re-découpe en sous-morceaux (fixed-size) pour rester
    sous max_chunk_size — mais on ne mélange jamais deux sections différentes
    dans le même chunk.
    """
    # on découpe sur les lignes qui commencent par ## ou ### (titres de section)
    sections = re.split(r"(?m)^(#{2,3}\s+.+)$", text)
    # re.split avec groupe capturant garde les séparateurs dans la liste résultat

    chunks = []
    current_header = ""
    for part in sections:
        if not part.strip():
            continue
        if re.match(r"^#{2,3}\s+", part):
            current_header = part.strip()
            continue

        section_text = (current_header + "\n" + part).strip() if current_header else part.strip()

        if len(section_text) <= max_chunk_size:
            chunks.append(section_text)
        else:
            # section trop longue : on la re-découpe, en gardant le header
            # en préfixe de chaque sous-chunk pour ne pas perdre le contexte
            sub_chunks = chunk_fixed(part.strip(), chunk_size=max_chunk_size, overlap=100)
            for sc in sub_chunks:
                prefixed = f"{current_header}\n{sc}" if current_header else sc
                chunks.append(prefixed)

    return chunks if chunks else [text]  # fallback si le doc n'a aucun titre


# --- Stratégie 3 : semantic chunking ----------------------------------------

def split_sentences(text: str):
    """Découpage simple en phrases (suffisant pour de la doc technique)."""
    # on protège d'abord les blocs de code pour ne pas les couper n'importe où
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_semantic(text: str, model: SentenceTransformer,
                    similarity_threshold: float = 0.55, max_chunk_size: int = 1500):
    """
    1. Découpe le texte en phrases.
    2. Calcule l'embedding de chaque phrase.
    3. Parcourt les phrases dans l'ordre : tant que la similarité cosinus
       avec la phrase précédente reste au-dessus du seuil, on regroupe dans
       le même chunk (on est sur le même sujet). Dès que la similarité chute,
       on considère que le sujet change : on démarre un nouveau chunk.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    embeddings = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)

    chunks = []
    current_chunk = []
    current_len = 0

    def flush():
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    for i, sentence in enumerate(sentences):
        # Cas piège : un bloc de code / YAML sans ponctuation devient une
        # seule "phrase" géante. On la force à être re-découpée plutôt que
        # de laisser un chunk énorme passer à travers les mailles.
        if len(sentence) > max_chunk_size:
            flush()
            current_chunk, current_len = [], 0
            for piece in chunk_fixed(sentence, chunk_size=max_chunk_size, overlap=100):
                chunks.append(piece)
            continue

        if i == 0 or not current_chunk:
            current_chunk = [sentence]
            current_len = len(sentence)
            continue

        sim = float(np.dot(embeddings[i], embeddings[i - 1]))  # cosinus (vecteurs normalisés)
        would_exceed = current_len + len(sentence) > max_chunk_size

        if sim >= similarity_threshold and not would_exceed:
            current_chunk.append(sentence)
            current_len += len(sentence)
        else:
            flush()
            current_chunk = [sentence]
            current_len = len(sentence)

    flush()

    return chunks


# --- Orchestration -----------------------------------------------------------

def run_strategy(name, docs, chunk_fn):
    print(f"\n--- Stratégie: {name} ---")
    all_chunks = []
    chunk_counter = 0

    for doc in tqdm(docs):
        pieces = chunk_fn(doc["text"])
        for piece in pieces:
            if len(piece.strip()) < 50:
                continue  # chunk trop court pour être utile, on l'ignore
            all_chunks.append({
                "chunk_id": f"{name}_{chunk_counter:05d}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_url": doc["source_url"],
                "strategy": name,
                "text": piece.strip(),
            })
            chunk_counter += 1

    out_path = PROCESSED_DIR / f"chunks_{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [len(c["text"]) for c in all_chunks]
    print(f"{len(all_chunks)} chunks générés -> {out_path}")
    print(f"Longueur moyenne: {np.mean(lengths):.0f} caractères "
          f"(min {min(lengths)}, max {max(lengths)})")

    return all_chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_corpus()
    print(f"{len(docs)} documents chargés depuis {RAW_DIR}")

    run_strategy("fixed", docs, lambda t: chunk_fixed(t))
    run_strategy("structure", docs, lambda t: chunk_structure(t))

    print("\nChargement du modèle d'embeddings pour le semantic chunking...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    run_strategy("semantic", docs, lambda t: chunk_semantic(t, model))

    print("\nTerminé. 3 fichiers générés dans data/processed/:")
    print("  - chunks_fixed.jsonl")
    print("  - chunks_structure.jsonl")
    print("  - chunks_semantic.jsonl")


if __name__ == "__main__":
    main()
