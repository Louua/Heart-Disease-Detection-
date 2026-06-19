# CardioCheck AI - Système Intelligent d'Évaluation du Risque Cardiaque

**CardioCheck AI** est un projet de Machine Learning End-to-End développé dans un cadre académique. Son but est de fournir aux professionnels de santé un outil d'aide à la décision clinique capable de prédire instantanément la probabilité de présence d'une cardiopathie chez un patient à partir de paramètres physiologiques et de résultats d'examens standards.

---

## Table des Matières
1. [Contexte et Problématique](#contexte-et-problématique)
2. [Jeu de Données (Dataset)](#jeu-de-données-dataset)
3. [Architecture du Prétraitement](#architecture-du-prétraitement)
4. [Modélisation et Optimisation](#modélisation-et-optimisation)
5. [Interface Application Web (Streamlit)](#interface-application-web-streamlit)
6. [Structure du Projet](#structure-du-projet)
7. [Installation et Lancement Local](#installation-et-lancement-local)
8. [Perspectives](#perspectives)

---

## Contexte et Problématique

Les cardiopathies coronariennes représentent l'une des causes majeures de mortalité dans le monde. Un diagnostic rapide et précis est essentiel pour orienter au mieux le parcours de soins d'un patient.

**CardioCheck AI** répond à cette problématique en proposant une interface web ergonomique connectée à un modèle d'intelligence artificielle optimisé. Le système évalue le risque cardiaque et fournit un diagnostic probabiliste en maximisant le **Rappel (Recall)** afin de minimiser au maximum le risque de Faux Négatifs (patients malades non détectés).

---

## Jeu de Données (Dataset)

Le modèle s'appuie sur le dataset clinique `heartdisease` (disponible sur Kaggle). Il comporte plus de **1 000 enregistrements de patients** avec 14 variables clés :

* **Données démographiques & physiologiques** : Âge, Sexe, Tension artérielle au repos (`trestbps`), Cholestérol sérique (`chol`), Glycémie à jeun (`fbs`).

* **Données d'examens cardiologiques** : Type de douleur thoracique (`cp`), Électrocardiogramme au repos (`restecg`), Fréquence cardiaque maximale atteinte (`thalach`), Angine de poitrine induite par l'effort (`exang`), Dépression du segment ST (`oldpeak`), Pente du segment ST d'effort (`slope`), Nombre de vaisseaux colorés par fluoroscopie (`ca`), Résultat de la scintigraphie cardiaque (`thal`).

* **Variable Cible (Target)** : `target_binary` (0 = Pas de maladie, 1 = Présence d'une anomalie cardiaque).

---

## Architecture du Prétraitement

Pour assurer la stabilité et la convergence des modèles mathématiques, les données subissent une préparation automatisée via un pipeline Scikit-Learn (`ColumnTransformer`) :

```mermaid
graph TD
    A[Données Brutes du Patient] --> B{Division des colonnes}
    B -->|Variables Numériques| C[StandardScaler]
    B -->|Variables Catégorielles| D[OneHotEncoder]
    C --> E[Moyenne = 0, Variance = 1]
    D --> F[Variables Binaires / Dummy]
    E --> G[Pipeline de Prétraitement Concaténé]
    F --> G
    G --> H[Modèle Prédictif]
