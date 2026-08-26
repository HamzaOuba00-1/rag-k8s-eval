# RAG Production-Grade — Documentation Kubernetes

Projet personnel. Objectif : construire un système RAG sur la
documentation officielle Kubernetes, avec une couche d'évaluation rigoureuse
(fidélité, hallucination, comparaison de stratégies de retrieval).

## Pourquoi ce projet est différenciant

La plupart des projets RAG s'arrêtent à "ça marche, regardez la démo". Celui-ci
prouve autre chose : **la capacité à mesurer si un système RAG dit la vérité**,
et à comparer objectivement plusieurs approches de retrieval avec des métriques
chiffrées plutôt qu'à l'œil.

## Corpus

Documentation officielle Kubernetes (Markdown), sections :
- `content/en/docs/concepts/`
- `content/en/docs/tasks/`
- `content/en/docs/tutorials/`

Source : https://github.com/kubernetes/website (licence CC BY 4.0)

## Stack technique

| Brique | Outil |
|---|---|
| Langage | Python 3.11+ |
| Embeddings denses | sentence-transformers (local, gratuit) |
| Recherche BM25 | rank_bm25 |
| Base vectorielle | FAISS |
| Reranker | cross-encoder sentence-transformers |
| LLM (génération + juge RAGAS) | Ollama local — `qwen2.5:7b-instruct` |
| Évaluation | RAGAS |

**Note sur le choix "local"** : un modèle 7-8B quantifié est plus faible qu'une
API type Claude/GPT, en particulier pour le rôle de "juge" dans RAGAS (évaluer
la fidélité d'une réponse est une tâche de raisonnement fine). On en tiendra
compte à l'étape 7 : on croisera les scores RAGAS avec une vérification manuelle
sur un échantillon, pour ne pas faire une confiance aveugle au juge local. C'est
d'ailleurs un point intéressant à mentionner dans le rapport final.

## Roadmap du projet

- [ ] **Étape 1 — Ingestion** : cloner et nettoyer le corpus Markdown (`src/ingestion.py`)
- [ ] **Étape 2 — Chunking** : découper en 3 stratégies différentes à comparer
- [ ] **Étape 3 — Indexation** : BM25, dense, hybride
- [ ] **Étape 4 — Retrieval + reranking**
- [ ] **Étape 5 — Génération** avec citations obligatoires vers la source
- [ ] **Étape 6 — Jeu de questions annoté** (dataset d'évaluation maison, ~50-100 questions)
- [ ] **Étape 7 — Évaluation RAGAS** (fidélité, pertinence, taux d'hallucination)
- [ ] **Étape 8 — Comparaison chiffrée** des 3 stratégies (tableau + graphes)
- [ ] **Étape 9 — Packaging** : rapport, README

## Setup

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
cp .env.example .env
```

Installer Ollama et récupérer le modèle (une seule fois) :

```bash
# Installer Ollama : https://ollama.com/download
ollama pull qwen2.5:7b-instruct

# Vérifier que ça marche :
ollama run qwen2.5:7b-instruct "Dis bonjour en une phrase"
```

## Étape 1 : récupérer le corpus

```bash
python src/ingestion.py
```

Ça va cloner (en sparse-checkout, donc rapide et léger) uniquement les dossiers
Markdown utiles du repo `kubernetes/website`, et les copier nettoyés dans
`data/raw/`.
