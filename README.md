# Devoir Projet 3 : RAG Multimodal et Agentique

Ce projet implemente un systeme de generation augmentee par recuperation (RAG) multimodal capable d'extraire, d'indexer et d'interroger le texte et les figures d'un document PDF scientifique.

---

## Contenu du Dossier

- **Attention_is_all_you_need.pdf** : Article scientifique fondateur (Vaswani et al., 2017) telecharge depuis arXiv.
- **pdfs/attention is all you need.pdf** : Copie de compatibilite pour le chemin relatif d'importation.
- **resultats_pdf/** : Dossier contenant les textes extraits page par page et les figures decoupees du document.
- **architecture_transformer.png** : Figure de l'architecture Transformer extraite pour les requetes multimodales.
- **rag_multimodal_agentique.ipynb** : Notebook execute avec succes articulant les 7 etapes de la construction du RAG.
- **.gitignore** : Configuration ignorant la base de donnees vectorielle locale et les caches.

---

## Description des 7 Etapes

### 1. Extraction Multimodale
- Decoupage du PDF page par page et separation du texte et des images integrees avec PyMuPDF.

### 2. Embeddings Multimodaux CLIP
- Projection conjointe du texte et des images dans l'espace vectoriel commun CLIP ViT-B-32.
- Normalisation L2 des vecteurs pour permettre la recherche par similarite cosinus directe.

### 3. Indexation Vectorielle dans ChromaDB
- Stockage des vecteurs d'embeddings avec hachage SHA-256 et metadonnees (page, chemin du fichier).

### 4. Recherche par Similarite Cosinus
- Recherche des k meilleurs resultats correspondants a partir d'une requete sous forme de texte ou d'image.

### 5. Outil Agentique (consulter_pdf)
- Encapsulation de la recherche vectorielle dans une fonction outillee destinee a un agent intelligent.

### 6. Construction de l'Agent ReAct
- Orchestration d'une boucle Reflexion -> Action -> Observation avec le modele local Ollama.

### 7. Tests et Validation
- Evaluation de l'agent sur des requetes documentaires complexes et localisation des figures de l'article.

---

## Comment Executer

1. Installer les dependances :
   ```bash
   pip install pymupdf open_clip_torch chromadb langchain langchain-ollama torch openai ipython
   ```
2. Demarrer le serveur local Ollama :
   ```bash
   ollama serve
   ```
3. Executer les cellules du notebook rag_multimodal_agentique.ipynb.
