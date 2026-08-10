
import pandas as pd

import numpy as np

from scipy.stats import mannwhitneyu



DATA_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/FRANZOSA_IBD_2019"

RESULTS_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/results"



mtb = pd.read_csv(f"{DATA_DIR}/mtb.tsv", sep="\t", index_col=0)

meta = pd.read_csv(f"{DATA_DIR}/metadata.tsv", sep="\t").set_index("Sample")

mapping = pd.read_csv(f"{DATA_DIR}/mtb.map.tsv", sep="\t")



meta = meta.loc[mtb.index]

mapping_lookup = mapping.set_index("Compound")[["m.z", "Retention.Time"]]



def get_ion_mode(feature_name):

    if "-pos" in feature_name:

        return "positive"

    elif "-neg" in feature_name:

        return "negative"

    return None



comparisons = {

    "CD_vs_Control": ("CD", "Control"),

    "UC_vs_Control": ("UC", "Control"),

    "CD_vs_UC": ("CD", "UC"),

}



for comp_name, (group_a, group_b) in comparisons.items():

    print(f"\n=== Running {comp_name} ===")

    mask_a = meta["Study.Group"] == group_a

    mask_b = meta["Study.Group"] == group_b

    samples_a = meta.index[mask_a]

    samples_b = meta.index[mask_b]

    print(f"{group_a}: n={len(samples_a)}, {group_b}: n={len(samples_b)}")



    rows = []

    for feature in mtb.columns:

        vals_a = mtb.loc[samples_a, feature].dropna()

        vals_b = mtb.loc[samples_b, feature].dropna()

        if len(vals_a) < 3 or len(vals_b) < 3:

            continue

        try:

            stat, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")

        except ValueError:

            continue

        rows.append({"feature": feature, "pvalue": p, "statistic": stat})



    result_df = pd.DataFrame(rows)

    result_df = result_df.merge(mapping_lookup, left_on="feature", right_index=True, how="left")

    result_df = result_df.dropna(subset=["pvalue", "m.z", "Retention.Time"])

    result_df = result_df.rename(columns={"m.z": "mz", "Retention.Time": "rtime"})

    result_df["ion_mode"] = result_df["feature"].apply(get_ion_mode)



    for mode in ["positive", "negative"]:

        subset = result_df[result_df["ion_mode"] == mode]

        mummichog_input = subset[["mz", "rtime", "pvalue", "statistic"]]

        outpath = f"{RESULTS_DIR}/mummichog_input_{comp_name}_{mode}.txt"

        mummichog_input.to_csv(outpath, sep="\t", index=False)

        print(f"  {mode} mode: {len(mummichog_input)} features -> {outpath}")



print("\nAll comparisons and ion modes complete.")

