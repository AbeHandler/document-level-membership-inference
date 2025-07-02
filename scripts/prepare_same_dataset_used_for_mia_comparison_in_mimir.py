from pathlib import Path

import os
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm as tqdm

ds = load_dataset("abehandlerorg/minhashblocksample", streaming=True, split="train")

targets = set([o.strip("\n") for o in open("targets.txt")])

df = []
for _ in tqdm(ds):
    if _["url"] in targets:
        _id = _["url"]
        _text = _["text"]
        df.append({'text': _text, "id": _id})

ds = pd.DataFrame(df)

user = os.environ["USER"]
home_dir = Path.home()

pathto = pathto = home_dir / ".cache" / "huggingface" / "token_write"
with open(pathto, "r") as inf:
    hf_token = inf.read().strip("\n")

Dataset.from_pandas(ds).push_to_hub(
    "abehandlerorg/minhashblocksample_targetsonly_doc_level_mia",
    private=False,  # True if you want it unlisted
    max_shard_size="5GB",  # Arrow shard size (default is 5GB)
    token=hf_token,
)
