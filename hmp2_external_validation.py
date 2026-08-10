
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb



FRANZOSA_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/FRANZOSA_IBD_2019"

HMP2_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/iHMP_IBDMDB_2019"

RESULTS_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/results"



# ---- Load the feature mapping we built ----

mapping = pd.read_csv(f"{RESULTS_DIR}/franzosa_hmp2_feature_mapping.csv")

print(f"Using {len(mapping)} shared metabolite features")



franzosa_cols_needed = ["Sample"] + mapping["franzosa_col"].tolist()

hmp2_cols_needed = ["Sample"] + mapping["hmp2_col"].tolist()



# ---- Load only the needed columns (much faster than loading everything) ----

franzosa_mtb = pd.read_csv(f"{FRANZOSA_DIR}/mtb.tsv", sep="\t", usecols=franzosa_cols_needed, index_col=0)

hmp2_mtb = pd.read_csv(f"{HMP2_DIR}/mtb.tsv/mtb.tsv", sep="\t", usecols=hmp2_cols_needed, index_col=0)



franzosa_meta = pd.read_csv(f"{FRANZOSA_DIR}/metadata.tsv", sep="\t").set_index("Sample")

hmp2_meta = pd.read_csv(f"{HMP2_DIR}/metadata.tsv", sep="\t").set_index("Sample")



# Rename columns to the shared metabolite name so both datasets align

franzosa_rename = dict(zip(mapping["franzosa_col"], mapping["metabolite_name"]))

hmp2_rename = dict(zip(mapping["hmp2_col"], mapping["metabolite_name"]))

franzosa_mtb = franzosa_mtb.rename(columns=franzosa_rename)

hmp2_mtb = hmp2_mtb.rename(columns=hmp2_rename)



# Align column order

shared_order = mapping["metabolite_name"].tolist()

franzosa_mtb = franzosa_mtb[shared_order]

hmp2_mtb = hmp2_mtb[shared_order]



print("Franzosa (train) shape:", franzosa_mtb.shape)

print("HMP2 (external test) shape:", hmp2_mtb.shape)



# ---- Preprocessing: same pipeline as before (CLR + z-score), fit on train, applied to test ----

def clr_transform(df):

    pseudo = df.fillna(0) + 1

    log_df = np.log(pseudo)

    return log_df.sub(log_df.mean(axis=1), axis=0)



franzosa_clr = clr_transform(franzosa_mtb)

hmp2_clr = clr_transform(hmp2_mtb)



train_mean = franzosa_clr.mean(axis=0)

train_std = franzosa_clr.std(axis=0)

X_train = (franzosa_clr - train_mean) / train_std

X_test = (hmp2_clr - train_mean) / train_std  # use TRAIN stats on test - no data leakage



X_train = X_train.fillna(0).values

X_test = X_test.fillna(0).values



y_train_raw = franzosa_meta.loc[franzosa_mtb.index, "Study.Group"].replace({"Control": "Control"})

y_test_raw = hmp2_meta.loc[hmp2_mtb.index, "Study.Group"].replace({"nonIBD": "Control"})



le = LabelEncoder()

le.fit(y_train_raw)

y_train = le.transform(y_train_raw)

y_test = le.transform(y_test_raw)



print("\nTrain class distribution:\n", y_train_raw.value_counts())

print("\nTest (HMP2) class distribution:\n", y_test_raw.value_counts())



# ---- Train final models on FULL Franzosa, test on HMP2 ----

models = {

    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),

    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, use_label_encoder=False, eval_metric="mlogloss"),

    "LASSO": LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000, random_state=42),

}



results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)

    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print(f"{name}: external validation AUC (HMP2) = {auc:.4f}")

    results.append({"model": name, "external_auc_hmp2": auc})



results_df = pd.DataFrame(results)

results_df.to_csv(f"{RESULTS_DIR}/hmp2_external_validation_results.csv", index=False)

print(f"\nSaved external validation results to {RESULTS_DIR}/hmp2_external_validation_results.csv")

