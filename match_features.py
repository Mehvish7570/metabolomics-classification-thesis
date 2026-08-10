
import pandas as pd

import re



FRANZOSA_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/FRANZOSA_IBD_2019"

HMP2_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/iHMP_IBDMDB_2019"



# Load just the headers (first row) of each metabolite table - no need to

# load full data yet, just column names, since these files are large

franzosa_cols = pd.read_csv(f"{FRANZOSA_DIR}/mtb.tsv", sep="\t", nrows=0).columns.tolist()

hmp2_cols = pd.read_csv(f"{HMP2_DIR}/mtb.tsv/mtb.tsv", sep="\t", nrows=0).columns.tolist()



print(f"Franzosa has {len(franzosa_cols)} feature columns")

print(f"HMP2 has {len(hmp2_cols)} feature columns")

print("\nSample Franzosa columns:", franzosa_cols[1:4])

print("Sample HMP2 columns:", hmp2_cols[1:4])



def extract_name_franzosa(col):

    # Format: "C18-neg_Cluster_0001: 4-hydroxystyrene" -> "4-hydroxystyrene"

    if ":" in col:

        name = col.split(":", 1)[1].strip()

        return name.lower() if name != "NA" else None

    return None



def extract_name_hmp2(col):

    # Format: "C18n_QI06__12.13-diHOME" -> "12.13-diHOME"

    if "__" in col:

        name = col.split("__", 1)[1].strip()

        return name.lower()

    return None



franzosa_names = {extract_name_franzosa(c): c for c in franzosa_cols[1:] if extract_name_franzosa(c)}

hmp2_names = {extract_name_hmp2(c): c for c in hmp2_cols[1:] if extract_name_hmp2(c)}



print(f"\nFranzosa named (non-NA) metabolites: {len(franzosa_names)}")

print(f"HMP2 named metabolites: {len(hmp2_names)}")



shared_names = set(franzosa_names.keys()) & set(hmp2_names.keys())

print(f"\nShared metabolite names between datasets: {len(shared_names)}")

print("Sample shared names:", list(shared_names)[:10])



# Save the mapping for use in the validation script

mapping_df = pd.DataFrame([

    {"metabolite_name": name, "franzosa_col": franzosa_names[name], "hmp2_col": hmp2_names[name]}

    for name in shared_names

])

mapping_df.to_csv("/rds/projects/g/guellili-slr-automation/mehvish/thesis/results/franzosa_hmp2_feature_mapping.csv", index=False)

print("\nSaved feature mapping to results/franzosa_hmp2_feature_mapping.csv")

