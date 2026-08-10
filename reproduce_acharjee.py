
import pandas as pd

import numpy as np

from sklearn.decomposition import PCA



DATA_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/FRANZOSA_IBD_2019"

RESULTS_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/results"



mtb = pd.read_csv(f"{DATA_DIR}/mtb.tsv", sep="\t", index_col=0)

meta = pd.read_csv(f"{DATA_DIR}/metadata.tsv", sep="\t")



print("Metabolite table shape:", mtb.shape)

print("Metadata shape:", meta.shape)

print(meta["Study.Group"].value_counts())



# ---- STEP 1: Sparsity filter ----

presence_frac = (mtb > 0).sum(axis=0) / mtb.shape[0]

mtb_filtered = mtb.loc[:, presence_frac >= 0.20]

print("\nAfter sparsity filter (>=20% presence):", mtb_filtered.shape)



# ---- STEP 2: CLR transformation ----

pseudo = mtb_filtered + 1

log_mtb = np.log(pseudo)

clr_mtb = log_mtb.sub(log_mtb.mean(axis=1), axis=0)

print("After CLR transform:", clr_mtb.shape)



# ---- STEP 3: Z-score normalisation ----

zscore_mtb = (clr_mtb - clr_mtb.mean(axis=0)) / clr_mtb.std(axis=0)

print("After z-score normalisation:", zscore_mtb.shape)



# ---- STEP 4: Outlier detection via PCoA (approximated with PCA on this

# already-transformed data) ----

# Flag samples whose PC1/PC2 position is > 3 standard deviations from

# the centroid as outliers.

pca = PCA(n_components=2)

pcs = pca.fit_transform(zscore_mtb.fillna(0))

pc_df = pd.DataFrame(pcs, index=zscore_mtb.index, columns=["PC1", "PC2"])



dist_from_centre = np.sqrt(pc_df["PC1"]**2 + pc_df["PC2"]**2)

threshold = dist_from_centre.mean() + 3 * dist_from_centre.std()

outliers = dist_from_centre[dist_from_centre > threshold].index.tolist()



print(f"\nOutlier threshold (mean + 3*std): {threshold:.2f}")

print(f"Detected {len(outliers)} outlier sample(s): {outliers}")



zscore_mtb_clean = zscore_mtb.drop(index=outliers)

meta_clean = meta[~meta["Sample"].isin(outliers)]

print(f"Shape after outlier removal: {zscore_mtb_clean.shape}")



# Save cleaned preprocessed data + matching metadata

zscore_mtb_clean.to_csv(f"{RESULTS_DIR}/franzosa_preprocessed.csv")

meta_clean.to_csv(f"{RESULTS_DIR}/franzosa_metadata_clean.csv", index=False)

print("\nSaved cleaned preprocessed data and metadata.")

