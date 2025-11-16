import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
import lightgbm as lgb
import joblib

# 1. Cargar datos
df = pd.read_csv("dataset.csv")  # cambia el nombre si hace falta

TARGET_COL = "target_variable"
ID_COL = "ID"

FEATURE_COLS = [
    "product_A_sold_in_the_past",
    "product_B_sold_in_the_past",
    "product_A_recommended",
    "product_A",
    "product_C",
    "product_D",
    "cust_hitrate",
    "cust_interactions",
    "cust_contracts",
    "opp_month",
    "opp_old",
    "competitor_Z",
    "competitor_X",
    "competitor_Y",
]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# 2. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # importante en clasificación binaria
)

print(X_test)

# 3. Dataset de LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. Hiperparámetros básicos de LightGBM
params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    # Si la clase 1 está desbalanceada:
    # "is_unbalance": True,
}

# 5. Entrenar modelo
model = lgb.train(
    params,
    train_set=train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# 6. Evaluar en test
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "lgbm_classifier.joblib")
