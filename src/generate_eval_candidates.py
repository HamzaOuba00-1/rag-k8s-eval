"""
Étape 6a — Génération de candidats pour le jeu de questions annoté.

Ce script NE PRODUIT PAS le dataset final. Il produit des CANDIDATS
(eval/questions_candidates.jsonl) que tu dois relire et corriger toi-même
avant de les valider dans eval/questions.jsonl (le vrai fichier utilisé
pour l'évaluation RAGAS).

Pourquoi ce détour ? Écrire 50-100 questions à la main prend des heures.
Ici, on tire des chunks au hasard dans le corpus, et on demande au LLM
local de proposer une question + une réponse basée SEULEMENT sur ce chunk.
Comme le chunk est connu et fixe, le LLM a très peu de marge pour halluciner
— mais il peut quand même se tromper ou mal formuler, d'où la relecture
humaine obligatoire après coup.
"""

import json
import random
from pathlib import Path

import ollama

PROCESSED_DIR = Path("data/processed")
EVAL_DIR = Path("eval")
OLLAMA_MODEL = "qwen2.5:7b-instruct"

N_CANDIDATES = 60  # vise large, tu en élimineras une partie à la relecture
STRATEGY_FOR_SAMPLING = "structure"  # chunks les plus "propres" pour générer depuis

GENERATION_PROMPT = """Voici un extrait de la documentation officielle Kubernetes :

---
{chunk_text}
---

Génère UNE question précise en anglais qu'un utilisateur pourrait poser, \
et dont la réponse se trouve ENTIÈREMENT dans cet extrait. Donne aussi la \
réponse correspondante, courte et factuelle, basée uniquement sur l'extrait.

Réponds STRICTEMENT au format JSON suivant, sans aucun texte autour :
{{"question": "...", "ground_truth": "..."}}
"""


def load_chunks():
    path = PROCESSED_DIR / f"chunks_{STRATEGY_FOR_SAMPLING}.jsonl"
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def generate_candidate(chunk):
    prompt = GENERATION_PROMPT.format(chunk_text=chunk["text"][:1200])
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",  # force une sortie JSON valide
    )
    try:
        parsed = json.loads(response["message"]["content"])
        return {
            "question": parsed["question"].strip(),
            "ground_truth": parsed["ground_truth"].strip(),
            "source_doc_id": chunk["doc_id"],
            "source_title": chunk["title"],
            "source_url": chunk["source_url"],
            "source_chunk_text": chunk["text"],  # pour te faciliter la relecture
            "category": "factual",
            "reviewed": False,  # <- tu passes à true une fois relu/corrigé
        }
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    EVAL_DIR.mkdir(exist_ok=True)
    chunks = load_chunks()

    # on ne garde qu'un chunk par document pour maximiser la diversité
    # (éviter 3 questions sur le même doc pendant que d'autres n'en ont aucune)
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    sampled_docs = random.sample(list(by_doc.keys()), min(N_CANDIDATES, len(by_doc)))
    sampled_chunks = [random.choice(by_doc[doc_id]) for doc_id in sampled_docs]

    print(f"Génération de {len(sampled_chunks)} candidats question/réponse...")

    candidates = []
    for i, chunk in enumerate(sampled_chunks):
        result = generate_candidate(chunk)
        if result:
            candidates.append(result)
        print(f"  [{i+1}/{len(sampled_chunks)}] {'OK' if result else 'ÉCHEC (ignoré)'}")

    out_path = EVAL_DIR / "questions_candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{len(candidates)} candidats générés -> {out_path}")
    print("\n>>> PROCHAINE ÉTAPE (manuelle) : ouvre ce fichier, relis chaque question,")
    print(">>> corrige/rejette ce qui ne va pas, puis enregistre le résultat final")
    print(">>> dans eval/questions.jsonl (voir instructions détaillées à suivre).")


if __name__ == "__main__":
    main()
