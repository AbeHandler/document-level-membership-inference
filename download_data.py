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

def prep_for_save():
    save_path = Path("data/guttenberg")
    save_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    datasets = [o.strip('\n') for o in open("datasets.txt")]
    for dspath in datasets:
        warm_cache(dspath)
