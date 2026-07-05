import pandas as pd
import sqlite3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE

# Charger les données
conn = sqlite3.connect("fraud.db")
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

X = df.drop(columns=["Class"])
y = df["Class"]

# Split AVANT le rééquilibrage (important : on ne rééquilibre jamais le test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation (surtout utile pour Amount et Time, pas déjà normalisées)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Avant rééquilibrage :", y_train.value_counts().to_dict())

# SMOTE : rééquilibrage par sur-échantillonnage synthétique
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("Après rééquilibrage :", pd.Series(y_train_balanced).value_counts().to_dict())

# Modèle 1 : Régression logistique
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_balanced, y_train_balanced)
pred_log = log_reg.predict(X_test_scaled)
proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Régression Logistique (avec SMOTE) ===")
print(classification_report(y_test, pred_log))
print("AUC-ROC:", roc_auc_score(y_test, proba_log))
print("AUC-PR (average precision):", average_precision_score(y_test, proba_log))

# Modèle 2 : Random Forest avec pondération de classe (alternative à SMOTE)
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)  # Random Forest gère le déséquilibre via class_weight, pas besoin de SMOTE ici
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest (class_weight=balanced) ===")
print(classification_report(y_test, pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, proba_rf))
print("AUC-PR (average precision):", average_precision_score(y_test, proba_rf))

# Importance des variables
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 variables les plus importantes :")
print(importances.head(10))

# Sauvegarde
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModèle sauvegardé")