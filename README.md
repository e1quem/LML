# Letterboxd Machine Learning (LML)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellow.svg)

Don't know which movie to pick in your Letterboxd watchlist for tonight? Don't worry, use this web-scraping machine learning repo to lose time artificially simulating your test to pick your perfect statistical match! No passion, only numbers.

Movies have a lot of varied features. A machine learning model will struggle to learn on a small sample. For good performance, consistent rating of 1000+ movies is recommended. In this example, we obtain high performances using [Karsten's Letterboxd profile](https://letterboxd.com/Kurstboy/) (2225 rated movies).

tst


## Pipeline de données
1. **Extraction des listes**
   - `user_scraper.py` récupère les films notés par un utilisateur (`/films/`), y compris la note et le `like` éventuel.
   - `WL_scraper.py` récupère la watchlist (`/watchlist/`), sans notes.
   - Ces scripts écrivent leurs résultats dans `out/movies_<pseudo>.csv` et `out/watchlist_<pseudo>.csv`.
2. **Enrichissement film par film**
   - `movie_scraper.py` reprend la CSV de films, charge chaque fiche Letterboxd et collecte vues, likes, histogramme de notation, pays, genres, équipes et durée.
   - Tous les champs calculés (écart-type, ratios, etc.) sont générés via `utils.compute_stats`.
   - Les fichiers enrichis s'appellent `out/enriched_<nom_du_fichier>.csv`.
3. **Modèles**
   - `ML_stack.py` entraîne trois modèles CatBoost (prédiction absolue, delta par rapport à la moyenne, probabilité de like) sur les films déjà notés.
   - `ML_watchlist.py` reprend ces modèles pour faire des prédictions sur la watchlist, puis trie les titres selon la note prédite et la probabilité de like.
   - Les deux scripts utilisent un encodage cible sécurisé, traitent textes libres (acteurs, studios, etc.) et empilent les trois modèles via une régression Ridge.
   - Les logs de CatBoost (courbes d'entraînement, erreurs) se trouvent dans `catboost_info/`.
4. **Analyse**
   - `overview.py` charge `out/enriched_movies_<pseudo>.csv` pour produire des graphiques de synthèse : nombre de films/likes par année, moyenne par décennie, camemberts top pays/genres, distribution des notes et métriques statistiques.
   - Les visuels sont enregistrés dans `out/art/` sous forme de `.svg`/`.png`.

## Installation et dépendances
```bash
python -m pip install -r requirements.txt
```
Si le fichier `requirements.txt` n'existe pas, installez manuellement : `pandas numpy matplotlib seaborn scikit-learn catboost shap tqdm selenium beautifulsoup4 undetected-chromedriver`.

### Pré-requis spécifiques
- Google Chrome 146 est attendu ; réglez `browser_executable_path` dans `utils.get_driver()` si nécessaire.
- `undetected_chromedriver` exige un driver compatible ; le script force l'utilisation IPv4 pour éviter certains blocages.
- Un fichier `out/` et `out/art/` doit exister (créez-les si besoin : `mkdir -p out/art`).

## Utilisation conseillée
1. **Scraper les films déjà notés**
   ```bash
   python user_scraper.py
   ```
   Répondez au prompt avec votre pseudo Letterboxd. 1 fichier `out/movies_<pseudo>.csv` sera généré.
2. **Enrichir chaque film**
   ```bash
   python movie_scraper.py
   ```
   Indiquez `movies_<pseudo>` (sans extension) pour enrichir les métadonnées et obtenir `out/enriched_movies_<pseudo>.csv`.
3. **Scraper la watchlist (optionnel)**
   ```bash
   python WL_scraper.py
   ```
   Idem : le fichier `out/watchlist_<pseudo>.csv` est produit puis enrichi par `ML_watchlist.py`.
4. **Lancer les modèles**
   - `python ML_stack.py` pour entraîner et évaluer les modèles sur vos films.
   - `python ML_watchlist.py` pour obtenir des prédictions sur la watchlist (nécessite `out/enriched_watchlist_<pseudo>.csv`).
5. **Produire un aperçu visuel**
   ```bash
   python overview.py
   ```
   Il génère `out/art/overview.svg` et affiche le temps de visionnage total.

## Sorties clefs
- `out/enriched_movies_<pseudo>.csv`, `out/enriched_watchlist_<pseudo>.csv` : données tabulaires utilisées par les modèles.
- `out/art/` : graphiques `overview.svg`, `missing_features.png`, `SHAPabsolute.png`, `SHAPdelta.png`, `ObservedEstimated*.svg`, dépendances SHAP des trois meilleures caractéristiques.
- `catboost_info/` : journaux d'entraînement, erreurs et fichiers `learn` / `test`.
- Predicts watchlist triée (`title`, `Predicted_Rating`, `Like_Probability`) imprimée par `ML_watchlist.py`.

## Conseils
- Les scripts demandent souvent un pseudo ; utilisez le même pour tous afin de garder la cohérence.
- Vérifiez les fichiers `.csv` générés après chaque étape avant de passer à la suivante.
- Pour réutiliser les modèles entraînés, copiez les artefacts CatBoost (les fichiers `catboost_info/learn*`).

N'hésitez pas à adapter les paramètres CatBoost (profondeur, learning rate) ou la logique de stacking selon vos préférences personnelles.
