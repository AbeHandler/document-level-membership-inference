import pandas as pd

blocks = pd.read_csv("classifier_results/chunks/copywritetraps_blockbench-blocksbin_chunkXX.csv").rename(columns={"score": "score_blocks"})
noblocks = pd.read_csv("classifier_results/chunks/copywritetraps_blockbench-noblocksbin_chunkXX.csv").rename(columns={"score": "score_noblocks"})
D = blocks.merge(noblocks,on="id")

D.to_csv("copyrighttraps.csv")