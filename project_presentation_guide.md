# Guide de Présentation du Projet : CardioCheck AI 🫀

Ce document fournit une structure détaillée, diapositive par diapositive, et des explications approfondies sur le projet **CardioCheck AI** pour vous aider à concevoir une présentation académique ou professionnelle percutante.

---

## Sommaire de la Présentation
1. **Diapositive 1 : Introduction & Contexte du Projet**
2. **Diapositive 2 : Problématique Clinique**
3. **Diapositive 3 : Les Données (Dataset)**
4. **Diapositive 4 : Nettoyage et Préparation des Données**
5. **Diapositive 5 : Architecture du Pipeline de Machine Learning**
6. **Diapositive 6 : Modélisation et Optimisation (Hyperparamètres)**
7. **Diapositive 7 : Résultats et Comparaison des Modèles**
8. **Diapositive 8 : L'Application CardioCheck AI (Interface Streamlit)**
9. **Diapositive 9 : Déploiement et Reproductibilité**
10. **Diapositive 10 : Conclusion et Perspectives**

---

### 🎴 Diapositive 1 : Introduction & Contexte
* **Titre suggéré** : CardioCheck AI : Système Intelligent d'Évaluation du Risque Cardiaque
* **Objectif de la diapositive** : Poser le cadre du projet, présenter l'équipe et la vision.
* **Points clés à aborder** :
  - L'avancement des technologies d'Intelligence Artificielle au service de la santé.
  - Le développement d'une application web de support à la décision clinique.
  - Alliance entre la rigueur scientifique (modèle prédictif performant) et l'ergonomie d'utilisation (interface médecin/soignant).

---

### 🎴 Diapositive 2 : Problématique Clinique
* **Titre suggéré** : Les Maladies Cardiovasculaires et le Rôle du Diagnostic Précoce
* **Objectif de la diapositive** : Expliquer *pourquoi* ce projet est crucial.
* **Points clés à aborder** :
  - Les cardiopathies coronariennes sont l'une des premières causes de mortalité dans le monde.
  - Un diagnostic précoce peut sauver des vies, mais l'analyse croisée de dizaines de paramètres physiologiques prend du temps.
  - **Le but du projet** : Fournir une estimation instantanée de la probabilité de maladie cardiaque à partir d'examens standards (ECG, taux de cholestérol, tension, etc.) pour aider au triage clinique.

---

### 🎴 Diapositive 3 : Les Données (Dataset)
* **Titre suggéré** : Exploration de nos Données Cliniques
* **Objectif de la diapositive** : Présenter la source et la nature des données.
* **Points clés à aborder** :
  - **Source** : Dataset `benafialoua/heartdisease` issu de Kaggle.
  - **Volume** : Plus de 1 000 enregistrements de patients.
  - **Variables explicatives clés** :
    - *Physiologiques* : Âge, Sexe (sex), Tension artérielle au repos (trestbps), Cholestérol (chol), Glycémie à jeun (fbs).
    - *Tests d'effort / Diagnostic* : Type de douleur thoracique (cp), Fréquence cardiaque max (thalach), Angine induite par l'effort (exang), Dépression du segment ST (oldpeak), Pente du segment ST (slope), Nombre de vaisseaux colorés (ca), Thalassémie (thal).
  - **Cible (Target)** : Indication binaire de la présence ou de l'absence d'une cardiopathie (`target_binary`).

---

### 🎴 Diapositive 4 : Nettoyage et Préparation des Données
* **Titre suggéré** : Data Cleaning & Feature Engineering
* **Objectif de la diapositive** : Montrer la rigueur appliquée au traitement des données avant l'apprentissage.
* **Points clés à aborder** :
  - **Gestion des valeurs manquantes** : Filtrage strict sur les variables physiologiques majeures (âge, trestbps, chol, thalach).
  - **Validation des plages de données cohérentes** :
    - Cholestérol limité entre 100 et 600 mg/dl.
    - Pression artérielle limitée entre 80 et 200 mm Hg.
    - Dépression ST limitée entre 0.0 et 6.0.
  - **Conversion des types** : Standardisation des valeurs numériques en entiers ou flottants.

---

### 🎴 Diapositive 5 : Architecture du Pipeline de Machine Learning
* **Titre suggéré** : Prétraitement et Préparation des Features
* **Objectif de la diapositive** : Expliquer comment les données brutes sont adaptées pour les modèles mathématiques.
* **Points clés à aborder** :
  - Utilisation d'un `ColumnTransformer` scikit-learn pour automatiser le traitement :
    - **Variables numériques** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) ➡️ **StandardScaler** : Normalisation pour obtenir une moyenne de 0 et une variance de 1 (crucial pour la Régression Logistique).
    - **Variables catégorielles** (`cp`, `restecg`, `slope`, `ca`, `thal`, `sex`, `fbs`, `exang`) ➡️ **OneHotEncoder** : Encodage en vecteurs binaires (en évitant le piège de la colinéarité grâce à `drop='first'`).
  - Division robuste : **80% Entraînement / 20% Test** avec stratification (`stratify=y`) pour conserver la proportion de malades dans chaque échantillon.

---

### 🎴 Diapositive 6 : Modélisation et Optimisation (Hyperparamètres)
* **Titre suggéré** : Sélection et Entraînement des Algorithmes
* **Objectif de la diapositive** : Présenter la compétition entre les deux algorithmes testés.
* **Points clés à aborder** :
  - **Métrique métier prioritaire** : **Le Rappel (Recall)**. 
    - *Pourquoi ?* En médecine, un faux négatif (ne pas détecter un malade) est bien plus dangereux qu'un faux positif (investiguer un patient sain). Nous voulons maximiser le nombre de malades identifiés.
  - **Algorithme 1 : Régression Logistique**
    - Optimisation par grille (`GridSearchCV`) sur la pénalité (L1/L2) et la force de régularisation `C`.
  - **Algorithme 2 : XGBoost Classifier**
    - Optimisation par recherche aléatoire (`RandomizedSearchCV`) sur la profondeur des arbres (`max_depth`), le taux d'apprentissage (`learning_rate`), le nombre d'estimateurs (`n_estimators`) et le poids des classes (`scale_pos_weight`).

---

### 🎴 Diapositive 7 : Résultats et Comparaison des Modèles
* **Titre suggéré** : Performances Comparées sur l'Échantillon Test
* **Objectif de la diapositive** : Démontrer l'efficacité scientifique de la solution retenue.
* **Points clés à aborder** :
  - Présentation des scores de Rappel (Recall) obtenus sur les données de test.
  - Sélection automatique du meilleur modèle (le script `train_local.py` compare dynamiquement les performances et retient le plus robuste).
  - Ré-entraînement final du modèle sélectionné sur l'intégralité du dataset pour capturer un maximum de patterns cliniques avant le déploiement.

---

### 🎴 Diapositive 8 : L'Application CardioCheck AI (Interface Streamlit)
* **Titre suggéré** : Déploiement : L'Interface d'Aide au Diagnostic
* **Objectif de la diapositive** : Présenter l'interface utilisateur web et ses fonctionnalités premium.
* **Points clés à aborder** :
  - **Design Premium et Ergonomique** : 
    - Style *Glassmorphic* moderne avec un thème sombre et des cartes translucides.
    - Utilisation de la police professionnelle *Outfit* de Google Fonts.
  - **Saisie simplifiée par onglets** : Séparation logique entre les *Données Cliniques* standards et les *Examens & Diagnostics* spécialisés.
  - **Visualisation d'Impact** : Un jauge de risque dynamique (SVG) codée par couleur (Vert = Faible, Jaune = Modéré, Rouge = Élevé).
  - **Exportabilité** : Génération et téléchargement instantané d'un rapport clinique au format texte brut `.txt` pour intégration directe dans le dossier médical du patient.

---

### 🎴 Diapositive 9 : Déploiement et Reproductibilité
* **Titre suggéré** : Architecture Technique du Projet
* **Objectif de la diapositive** : Démontrer la facilité de déploiement et la portabilité du code.
* **Points clés à aborder** :
  - **Script de démarrage automatisé** (`run_local.sh`) :
    - Gestion de l'environnement virtuel Python (`venv`).
    - Installation des dépendances via `requirements.txt`.
    - Entraînement automatique si les fichiers `.pkl` du modèle sont manquants.
    - Lancement automatique de l'interface Streamlit locale.
  - **Sérialisation sécurisée** : Sauvegarde des pipelines de prétraitement et des modèles via `joblib` pour un chargement instantané à chaud sans temps de calcul.

---

### 🎴 Diapositive 10 : Conclusion et Perspectives
* **Titre suggéré** : Synthèse et Évolutions Futures
* **Objectif de la diapositive** : Ouvrir sur l'avenir du projet.
* **Points clés à aborder** :
  - **Bilan** : Une solution complète, de l'exploration des données brutes à l'application interactive prête à l'emploi.
  - **Perspectives d'amélioration** :
    - Intégration d'explications de prédictions locales (par exemple via les valeurs SHAP) pour rendre l'IA transparente pour le médecin.
    - Connexion directe à une API de base de données d'hôpitaux via FastAPI (les routes étant déjà prévues dans les dépendances).
    - Validation clinique externe avec de nouveaux centres hospitaliers.
