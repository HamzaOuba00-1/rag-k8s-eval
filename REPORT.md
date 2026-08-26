# RAG Production-Grade sur la documentation Kubernetes — Rapport

**Auteur :** Hamza Ouba
**Contexte :** Projet personnel — PFE Data & IA, 5e année cycle ingénieur

---

## 1. Objectif du projet

Construire un système RAG (Retrieval-Augmented Generation) sur la documentation
officielle Kubernetes, avec un focus sur ce que la plupart des projets RAG
n'implémentent pas : une **couche d'évaluation rigoureuse**, capable de
mesurer objectivement la fidélité des réponses (hallucination), et de
comparer plusieurs stratégies de chunking et de retrieval avec des métriques
chiffrées plutôt qu'à l'œil.

## 2. Corpus

- Documentation officielle Kubernetes (`content/en/docs/concepts`, `tasks`,
  `tutorials`), récupérée depuis https://github.com/kubernetes/website
- 408 documents Markdown, ~840 000 mots
- Chaque document conserve son URL source réelle (`https://kubernetes.io/docs/...`),
  utilisée pour les citations dans les réponses générées

## 3. Architecture du pipeline

| Étape | Composant | Détail |
|---|---|---|
| Chunking | 3 stratégies comparées | fixed-size (baseline), structure-aware (titres Markdown), semantic (similarité entre phrases) |
| Indexation | BM25 + dense (FAISS) | par stratégie de chunking, `all-MiniLM-L6-v2` pour les embeddings |
| Retrieval | 4 méthodes comparées | BM25 seul, dense seul, hybride (Reciprocal Rank Fusion), hybride + reranking (cross-encoder) |
| Génération | Ollama local, `qwen2.5:7b-instruct` | citations `[n]` obligatoires vers les chunks utilisés, refus explicite si l'info n'est pas dans le contexte |
| Évaluation | RAGAS | faithfulness, answer_relevancy, context_precision, context_recall + taux de refus sur questions hors-corpus |

## 4. Jeu de questions d'évaluation

- **~52 questions factuelles**, générées semi-automatiquement (LLM local +
  relecture manuelle systématique — voir méthodologie ci-dessous) à partir
  d'un échantillon de chunks du corpus
- **8 questions "impossible"**, hors du corpus (autres technologies, champs
  API inexistants, prémisses fausses), pour mesurer la capacité du système à
  dire "je ne sais pas" plutôt que d'halluciner

### Méthodologie de génération/validation
Les paires question/réponse ont été générées par un LLM local à partir de
chunks individuels, puis **relues et corrigées manuellement** avant d'être
retenues comme gold standard — un contrôle indispensable pour la crédibilité
de l'évaluation (voir section Limites).

## 5. Analyse qualitative complémentaire (observée sur `src/retrieval.py`)
- BM25 seul ramène du bruit lexical (documents partageant des mots-clés
  sans rapport sémantique réel)
- Le retrieval dense pur donne les résultats les plus propres sur des
  requêtes formulées naturellement
- La fusion hybride (RRF) réintroduit parfois du bruit BM25 que le
  reranking corrige partiellement


## 6. Pistes d'amélioration

- Faire tourner l'évaluation complète (58 questions × 4 métriques × 6
  combinaisons) sur une infrastructure plus puissante ou une API payante
  pour confirmer les résultats à plus grande échelle
- Ajouter une vérification manuelle croisée sur un sous-échantillon, pour
  quantifier l'écart entre le juge local et un jugement humain
- Tester d'autres valeurs de seuil pour le semantic chunking (le seuil
  actuel de 0.55 produit des chunks nettement plus petits que les 2 autres
  stratégies — un des facteurs à investiguer si cette stratégie sous-performe)

## 7. Stack technique

Python 3.12, Ollama (qwen2.5:7b-instruct), sentence-transformers, FAISS,
rank_bm25, RAGAS, LangChain (wrappers Ollama/HuggingFace)
