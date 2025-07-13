import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


if __name__ == "__main__":
    fn1 = "classifier_results/chunks/suffixesnoblocksbin_blockbench-blocksbin_chunkXX.csv"
    fn2 = "classifier_results/chunks/suffixesnoblocksbin_blockbench-noblocksbin_chunkXX.csv"

    blocks = pd.read_csv(fn1).rename(columns={"score": "blocksscore"})
    noblocks = pd.read_csv(fn2).rename(columns={"score": "noblocksscore"})

    df = blocks.merge(noblocks, on="id")
    df = df[~df["id"].isna()]

    # for meeus, the method purports to predict in-sample data
    # it assumes you have in-sample and out-of-sample data.
    # - we are going to assume that the noblocked data is in-sample
    # - we are going to assume that the blocked data is out-of-sample
    # we would expect to get lower predictions for blocked data on the blocks model than the noblocks model
    # so the ATE is ....expected negative, so we multiply by -1 to make it positive

    df["ate"] = df["blocksscore"] - df["noblocksscore"]  

    df["ate"] = -1 * df["ate"]

    print(np.mean(df["ate"]))

    # Assume `ate_values` is a 1D array of ATE estimates per unit/block/cluster
    stat, p = wilcoxon(df["ate"], alternative="greater")

    print(f"Statistic: {stat}, p-value: {p}")

    print(df)
