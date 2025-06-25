import pandas as pd


# TODO this drop dupes could be a little tighter. I am not 100% of the source
blocks = pd.read_csv("classifier_results/chunks/copywritetraps_blockbench-blocksbin_chunkXX.csv").rename(columns={"score": "score_blocks"}).drop_duplicates(subset="id", keep="first")
noblocks = pd.read_csv("classifier_results/chunks/copywritetraps_blockbench-noblocksbin_chunkXX.csv").rename(columns={"score": "score_noblocks"}).drop_duplicates(subset="id", keep="first")


D = blocks.merge(noblocks,on="id")

# returns a prediction for binary membership of input document
# 4.4 Meta-classifier. So it should be more confident in no blocks
D["delta"] = D["score_noblocks"] - D["score_blocks"]

print(D["delta"].mean())

D.to_csv("copyrighttraps.csv")