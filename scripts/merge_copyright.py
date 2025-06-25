import pandas as pd

blocks = pd.read_csv("classifier_results/chunks/copywritetraps_blockbench-blocksbin_chunkXX.csv").rename(columns={"score": "score_blocks"})
noblocks = pd.read_csv("classifier_results/chunks/copywritetraps_blockbench-noblocksbin_chunkXX.csv").rename(columns={"score": "score_noblocks"})

print(len(blocks))
import pdb; pdb.set_trace()
D = blocks.merge(noblocks,on="id")

print(len(D))
# returns a prediction for binary membership of input document
# 4.4 Meta-classifier 
D["delta"] = D["score_noblocks"] - D["score_blocks"]

D.to_csv("copyrighttraps.csv")