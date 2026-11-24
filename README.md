# 👁️‍🗨️ Devoir Projet 3 : RAG Multimodal & Agentique

Ce projet implémente un système RAG multimodal capable d'extraire, d'indexer et d'interroger le texte et les figures d'un document PDF scientifique.

---

## 📂 Contenu du Dossier

- **`Attention_is_all_you_need.pdf`** : L'article de recherche original (Vaswani et al., 2017) téléchargé depuis arXiv.
- **`pdfs/attention is all you need.pdf`** : Copie de compatibilité pour le chargement du document.
- **`rag_multimodal_agentique.ipynb`** : Le notebook complet décrivant les 7 étapes du RAG multimodal.
- **`.gitignore`** : Fichier ignorant la base vectorielle ChromaDB et les variables d'environnement.

---

## 📝 Description des 7 Étapes

### 1. 📄 Extraction Multimodale
- Découpe le PDF page par page et extrait séparément le texte et les images avec PyMuPDF.

### 2. 🌐 Embeddings Multimodaux CLIP
- Projette le texte et les images dans le même espace vectoriel grâce au modèle CLIP ViT-B-32.
- Normalise les vecteurs pour permettre le calcul direct de la similarité cosinus.

### 3. 🗄️ Indexation Vectorielle dans ChromaDB
- Stocke les représentations vectorielles avec identifiant SHA-256 et métadonnées (numéro de page, chemin de l'image).

### 4. 🔍 Recherche par Similarité Cosinus
- Recherche les passages ou figures les plus pertinents à partir d'une requête textuelle ou visuelle.

### 5. 🛠️ Outil Agentique (`consulter_pdf`)
- Encapsule le retriever dans une fonction outillée utilisable par un agent intelligent.

### 6. 🤖 Agent ReAct (MIND + BODY)
- Orchestre un cycle Réflexion $\to$ Action $\to$ Observation avec le LLM local Ollama (`maternion/mimo-v2.6:9b-instruct`).

### 7. 🏆 Tests & Validation
- Interroge l'agent sur des questions textuelles (ex: attention multi-têtes) et demande la localisation de figures (Figure 2).

---

## 🚀 Comment Exécuter

1. Installez les dépendances :
   ```bash
   pip install pymupdf open_clip_torch chromadb langchain langchain-ollama torch
   ```
2. Démarrez votre modèle local Ollama :
   ```bash
   ollama run maternion/mimo-v2.6:9b-instruct
   ```
3. Exécutez le notebook `rag_multimodal_agentique.ipynb`.
