
import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelBinarizer

import xgboost as xgb

import time



RESULTS_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/results"



# ---- Load preprocessed data ----

X = pd.read_csv(f"{RESULTS_DIR}/franzosa_preprocessed.csv", index_col=0)

meta = pd.read_csv(f"{RESULTS_DIR}/franzosa_metadata_clean.csv")

meta = meta.set_index("Sample").loc[X.index]  # align order

y = meta["Study.Group"]



print("X shape:", X.shape)

print("y distribution:\n", y.value_counts())



X = X.fillna(0).values

y_binarized = LabelBinarizer().fit_transform(y)  # one-vs-rest for multiclass AUC



# ---- Model + hyperparameter grid definitions ----

models = {

    "RandomForest": (

        RandomForestClassifier(random_state=42),

        {"n_estimators": [100, 300], "max_depth": [None, 10]}

    ),

    "XGBoost": (

        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"),

        {"n_estimators": [100, 300], "max_depth": [3, 6]}

    ),

    "LASSO": (

        LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=42),

        {"C": [0.01, 0.1, 1.0]}

    ),

}



outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# Encode y as integers for stratification/model fitting

y_encoded = pd.factorize(y)[0]



results = {name: [] for name in models}



start = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded)):

    print(f"\n=== Outer fold {fold_idx + 1}/10 ===")

    X_train, X_test = X[train_idx], X[test_idx]

    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]



    for name, (model, param_grid) in models.items():

        grid = GridSearchCV(model, param_grid, cv=inner_cv, scoring="roc_auc_ovr", n_jobs=-1)

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_



        y_proba = best_model.predict_proba(X_test)

        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        results[name].append(auc)

        print(f"{name}: best params={grid.best_params_}, outer fold AUC={auc:.4f}")



elapsed = time.time() - start

print(f"\nTotal training time: {elapsed/60:.1f} minutes")



# ---- Summarize and save ----

summary_rows = []

for name, aucs in results.items():

    mean_auc = np.mean(aucs)

    std_auc = np.std(aucs)

    print(f"\n{name}: mean AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    summary_rows.append({"model": name, "mean_auc": mean_auc, "std_auc": std_auc, "fold_aucs": aucs})



summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(f"{RESULTS_DIR}/nested_cv_summary.csv", index=False)

print(f"\nSaved summary to {RESULTS_DIR}/nested_cv_summary.csv")

