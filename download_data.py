import os
from datasets import load_dataset
from datasets import Dataset
from pathlib import Path
from datasets import concatenate_datasets

# in slurm => export HF_DATASETS_CACHE=/scratch/alpine/$USER/hfdatasets

dataset_path = "datacache"

def warm_cache(dspath):
    try:
        ds_streaming = load_dataset(dspath, split="train", cache_dir="datacache", streaming=True)
        ds = ds_streaming.take(50000)
        ds_fixed = Dataset.from_generator(lambda: ds)
        dsname = dspath.split("/").pop()
        ds_fixed.save_to_disk(f"data/{dsname}")
    except UnicodeDecodeError as e:
        pass

if __name__ == "__main__":
    datasets = [o.strip('\n') for o in open("datasets.txt")]
    for dspath in datasets:
        warm_cache(dspath)

    # for suffix array one we only want when blocksbin == 0
    from datasets import load_dataset
    ds = load_dataset("abehandlerorg/suffixesnoblocksbin", split="train")
    ds = ds.filter(lambda x: x["blocksbin"] == 0)
    ds = ds.map(lambda x: {"text": x["sequence"]}, remove_columns=["sequence"])
    ds = ds.map(lambda x: {"id": x["text"]})
    ds.save_to_disk(f"data/suffixesnoblocksbin")