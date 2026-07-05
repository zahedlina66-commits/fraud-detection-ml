# Détection de Fraude Bancaire par Carte de Crédit

## Contexte et objectif

Ce projet vise à développer un modèle de machine learning capable de détecter des transactions frauduleuses par carte de crédit. La fraude bancaire représente un enjeu financier majeur pour les établissements bancaires : au-delà de la perte directe liée à chaque transaction frauduleuse, elle engendre des coûts de traitement, un risque réputationnel et une perte de confiance client.

Ce projet illustre une problématique centrale en machine learning appliqué : la détection d'événements rares au sein d'un jeu de données très déséquilibré, où les métriques d'évaluation classiques (accuracy) sont trompeuses.

## Données

- **Source** : dataset "Credit Card Fraud Detection" (Kaggle), transactions de cartes bancaires européennes
- **Volume** : 284 807 transactions
- **Variables** : `Time` (temps écoulé depuis la première transaction), `Amount` (montant), `V1` à `V28` (composantes issues d'une Analyse en Composantes Principales, anonymisées pour protéger la confidentialité des données bancaires réelles), et la variable cible `Class` (0 = transaction normale, 1 = fraude)
- **Répartition de la cible** : 492 transactions frauduleuses sur 284 807, soit un taux de fraude de **0,17 %** — dataset extrêmement déséquilibré

> **Note** : le fichier `creditcard.csv` (144 Mo) n'est pas inclus dans ce dépôt car il dépasse la limite de taille de GitHub. Pour reproduire ce projet, téléchargez-le depuis [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et placez-le à la racine du projet avant d'exécuter `01_setup_db.py`.

## Architecture du projet

```
Données brutes (CSV)
   → Base de données SQLite (01_setup_db.py)
   → Nettoyage, rééquilibrage, modélisation ML (02_train_model.py)
   → Dashboard interactif (app.py)
```

### Fichiers du projet

| Fichier | Rôle |
|---|---|
| `creditcard.csv` | Données brutes sources |
| `01_setup_db.py` | Chargement des données, création de la base SQLite |
| `02_train_model.py` | Préparation des features, rééquilibrage, entraînement et évaluation des modèles |
| `app.py` | Dashboard Streamlit (visualisation + simulateur de prédiction) |
| `requirements.txt` | Liste des librairies nécessaires |
| `model.pkl`, `scaler.pkl`, `columns.pkl` | Modèle entraîné et objets nécessaires aux prédictions |

## Démarche technique

### 1. Le problème du déséquilibre extrême
Avec seulement 0,17 % de fraude, un modèle qui prédirait systématiquement "transaction normale" atteindrait déjà 99,8 % d'accuracy tout en étant totalement inutile. Deux enjeux ont donc guidé ce projet : le choix des métriques d'évaluation, et le traitement du déséquilibre des classes.

### 2. Séparation train/test avant tout traitement du déséquilibre
Le jeu de données a été séparé en 80 % entraînement / 20 % test **avant** toute technique de rééquilibrage, afin d'évaluer le modèle sur une distribution représentative de la réalité, et d'éviter toute fuite d'information entre les deux ensembles.

### 3. Deux approches du déséquilibre comparées
- **SMOTE** (Synthetic Minority Over-sampling Technique) appliqué à la régression logistique : génère des exemples synthétiques de fraude par interpolation entre cas existants, pour rééquilibrer artificiellement le jeu d'entraînement (rééquilibré ici à 227 451 / 227 451)
- **Pondération de classe** (`class_weight="balanced"`) appliquée au Random Forest : alternative à SMOTE qui pénalise davantage les erreurs sur la classe minoritaire, sans dupliquer de données

### 4. Métriques adaptées au déséquilibre
L'accuracy a été écartée au profit de :
- **Recall** sur la classe fraude : capacité du modèle à détecter les fraudes réelles (métrique prioritaire, car une fraude non détectée coûte cher)
- **Precision** sur la classe fraude : proportion d'alertes réellement fondées
- **AUC-ROC** : pouvoir de discrimination global du modèle
- **AUC-PR (average precision)** : métrique particulièrement adaptée aux classes très déséquilibrées, plus informative que l'AUC-ROC dans ce contexte

**Résultats obtenus :**

| Modèle | Precision (fraude) | Recall (fraude) | F1-score (fraude) | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Régression logistique + SMOTE | 0,06 | 0,92 | 0,11 | 0,971 | 0,724 |
| Random Forest (class_weight balanced) | 0,67 | 0,86 | 0,75 | 0,980 | 0,822 |

**Interprétation du compromis precision/recall** : le Random Forest surpasse nettement la régression logistique sur l'ensemble des métriques. Il conserve un recall élevé (86 % des fraudes détectées, contre 92 % pour la régression logistique) tout en améliorant considérablement la precision (67 % contre seulement 6 %) : les alertes générées sont donc bien plus fiables, avec un nombre de fausses alertes très réduit. Ce résultat illustre les limites de SMOTE dans ce contexte : la génération d'exemples synthétiques de fraude par interpolation peut créer des points artificiels peu représentatifs de la frontière de décision réelle, poussant le modèle à sur-classer des transactions normales comme frauduleuses. La pondération de classe (`class_weight="balanced"`), qui ajuste directement la fonction de coût sans dupliquer de données, se révèle ici plus performante. Le Random Forest est donc retenu comme modèle final de ce projet.

D'un point de vue métier, ce compromis reste à ajuster selon le contexte réel : un recall de 86 % avec une precision de 67 % signifie qu'environ une alerte sur trois est une fausse alerte, ce qui est largement gérable avec une vérification humaine en aval, tout en détectant la grande majorité des fraudes réelles.

### 5. Interprétabilité
D'après le Random Forest, les variables les plus déterminantes dans la prédiction de la fraude sont, par ordre d'importance : V14 (0,190), V10 (0,116), V4 (0,113), V12 (0,098) et V17 (0,097). Ces cinq variables concentrent à elles seules plus de 60 % de l'importance totale. Étant issues d'une Analyse en Composantes Principales appliquée aux données bancaires originales (anonymisées pour des raisons de confidentialité), leur interprétation métier directe n'est pas possible ; en contexte professionnel réel, un échange avec l'équipe fraude permettrait de relier ces composantes aux variables d'origine (localisation, type de commerçant, historique du porteur de carte, etc.) pour affiner l'analyse.

## Dashboard

Le dashboard Streamlit propose trois volets :
1. **Vue d'ensemble** : taux de fraude, nombre de transactions, montant total en jeu
2. **Analyse détaillée** : distribution des montants pour les transactions normales vs frauduleuses
3. **Simulateur de prédiction** : saisie du montant et du temps d'une transaction, calcul en temps réel de la probabilité de fraude

## Limites et pistes d'amélioration

- Les variables V1 à V28 étant issues d'une ACP, elles ne sont pas interprétables individuellement d'un point de vue métier — un travail avec les équipes fraude sur des variables métier brutes (localisation, historique client, type de commerçant) enrichirait l'analyse
- Le seuil de décision par défaut (0,5) n'a pas été optimisé ; une calibration selon le coût réel métier des faux négatifs et faux positifs améliorerait le compromis precision/recall
- D'autres techniques de gestion du déséquilibre pourraient être testées et comparées (undersampling, combinaison SMOTE + Tomek links, XGBoost avec `scale_pos_weight`)
- Le dataset ne couvre qu'une courte période ; un système de détection en production nécessiterait un réentraînement régulier pour s'adapter à l'évolution des schémas de fraude

## Stack technique

Python (pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn), SQL (SQLite), Streamlit
