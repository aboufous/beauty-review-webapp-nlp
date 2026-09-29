# GlowMate — Application web NLP de recommandation (Streamlit)

Application web qui permet de parcourir un catalogue de cosmétiques, de rédiger des avis et d'obtenir des recommandations de produits similaires. Un modèle de classification prédit si l'auteur d'un nouvel avis est un acheteur vérifié.

> Projet en équipe de 4 — RMIT University (2026)
> **Ma contribution : module de recherche floue** (`src/search.py`, page *Browse & Search*)

## Ma contribution : recherche floue

- Recherche par mots-clés tolérante aux fautes de frappe avec **rapidfuzz**
- Champ de recherche construit à partir de la marque, du titre et des tags produit (le jeu de données ne contient pas de description libre ; les tags portent les mots-clés de catégorie et d'ingrédients)
- Intégration dans la page Streamlit *Browse & Search*

## Fonctionnalités de l'application

| Page | Fonction |
|---|---|
| Browse & Search | Recherche floue dans le catalogue |
| Product Detail | Formulaire d'avis + prédiction du modèle |
| Product Detail | Recommandation hybride de produits similaires (TF-IDF) |
| Admin Dashboard | File de modération, KPIs, suivi du modèle |

## Modèle de classification

Fusion de trois modèles entraînés indépendamment, combinés par une régression logistique (stacking sur prédictions out-of-fold) :

1. **Texte** : TF-IDF (unigrammes + bigrammes) + régression logistique
2. **Métadonnées** : HistGradientBoosting (note, prix, longueur du texte, marque…)
3. **Prior** : probabilité lissée par produit et par marque

| Modèle | Macro-F1 | Accuracy |
|---|---|---|
| Texte | 0,627 | 0,706 |
| Métadonnées | 0,715 | 0,834 |
| Prior | 0,676 | 0,818 |
| **Fusion** | **0,741** | **0,841** |

## Lancer le projet

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

python scripts/build_catalog.py  # génère data/products.csv et data/reviews.csv
python scripts/train.py          # entraîne les modèles (~30 s)
streamlit run app.py             # http://localhost:8501
```

## Organisation du code

```
app.py              Page d'accueil
pages/              Pages Streamlit
src/
  search.py         Recherche floue (ma contribution)
  preprocess.py     Tokenisation
  data.py           Lecture du catalogue et des avis
  similarity.py     Produits similaires
  models/           Modèles texte, métadonnées, prior et fusion
scripts/            Construction du catalogue et entraînement
data/               Données
docs/               Documentation du modèle
```

## Technologies

Python · Streamlit · scikit-learn · rapidfuzz · pandas