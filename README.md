# MlFlow_medical_screen_CNN
![banniere_one](img/banniere_readme.png)

Voici un README mis à jour pour ton projet utilisant **MobileNetV3Large** et **MLflow** :

---

# 🫁 Classification de Pneumonie par Radiographie avec MobileNetV2 et MLflow

## 📌 Objectif

Ce projet a pour but de développer un modèle de **classification binaire** (PNEUMONIA vs NORMAL) à partir de radios thoraciques, en utilisant le transfert de learning via **MobileNetV3Large** et le suivi d'expérience via **MLflow**.

---

## 🗃️ Dataset

Le jeu de données utilisé provient de Kaggle : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
Il est organisé selon la structure suivante :

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## ⚙️ Environnement

* Python = 3.10
* TensorFlow ≥ 2.11
* OpenCV, NumPy, scikit-learn,seaborn, 
* MLflow (serveur local : `http://localhost:5000`)

---

## 🔍 Étapes du pipeline

### 1. 📥 Préparation des données

* Chargement des images RGB.
* Redimensionnement à 224×224.
* Normalisation (valeurs entre 0 et 1).
* Encodage des étiquettes : `NORMAL = 0`, `PNEUMONIA = 1`.

### 2. 🧠 Modèle

* Base : `MobileNetV3Large (weights='imagenet')`, avec la tête personnalisée :

  * GlobalAveragePooling
  * Dropout
  * Dense(1, sigmoid)
* Fine-tuning des 20 dernières couches.

### 3. 🏋️‍♂️ Entraînement

* Optimiseur : Adam
* Perte : Binary Crossentropy
* EarlyStopping sur validation

### 4. 📊 Évaluation

* Précision sur l’ensemble test
* Matrice de confusion
* Rapport de classification
* Courbes :

  * ROC
  * Précision-rappel

### 5. 📦 Suivi via MLflow

* Paramètres (batch\_size, epochs, learning\_rate…)
* Métriques (acc, loss, auc…)
* Artéfacts (graphiques, modèle)
* Signature et input\_example du modèle

---

## ▶️ Exécution

Lancer le script principal (dans un notebook ou `.py`) après avoir démarré MLflow avec :

```bash
mlflow ui
```

puis accéder à :
📍 `http://localhost:5000`

---

## 📁 Résultats

* Test accuracy : \~XX% (variable selon l'entraînement)
* Courbes et matrices accessibles via MLflow
* Modèle enregistré : `mlruns/.../artifacts/model`



