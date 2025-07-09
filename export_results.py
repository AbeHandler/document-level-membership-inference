import pandas as pd
from scipy.stats import wilcoxon


if __name__ == "__main__":
    fn1 = "classifier_results/chunks/minhashblocksample_targetsonly_doc_level_mia_blockbench-blocksbin_chunkXX.csv"
    fn2 = "classifier_results/chunks/minhashblocksample_targetsonly_doc_level_mia_blockbench-noblocksbin_chunkXX.csv"

    blocks = pd.read_csv(fn1).rename(columns={"score": "noblocksscore"})
    noblocks = pd.read_csv(fn2).rename(columns={"score": "blocksscore"})

    df = blocks.merge(noblocks, on="id")
    df = df[~df["id"].isna()]

    # for meeus, the method purports to predict in-sample data
    # it assumes you have in-sample and out-of-sample data.
    # so we are going to assume that the blocked data is in-sample
    # we would expect to get lower predictions for blocked data on the blocks model than the noblocks model
    # so the ATE is ....expected negative, so we multiply by -1 to make it positive

    df["ate"] = df["blocksscore"] - df["noblocksscore"]  

    print(-1 * df["ate"].mean())

    # Assume `ate_values` is a 1D array of ATE estimates per unit/block/cluster
    stat, p = wilcoxon(-1 * df["ate"], alternative="greater")

    print(f"Statistic: {stat}, p-value: {p}")