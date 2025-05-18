import os
from datasets import load_dataset
from datasets import Dataset
from pathlib import Path
from datasets import concatenate_datasets

# in slurm => export HF_DATASETS_CACHE=/scratch/alpine/$USER/hfdatasets

dataset_path = "datacache"

def load_guttenberg_from_files():
    ds = load_dataset("imperial-cpg/project-gutenberg-extended")
    arrow_files = [
            f"{dataset_path}/parquet-train-00000-of-00004.arrow",
            f"{dataset_path}/parquet-train-00001-of-00004.arrow",
            f"{dataset_path}/parquet-train-00002-of-00004.arrow",
            f"{dataset_path}/parquet-train-00003-of-00004.arrow"
        ]

    # Load each Arrow file and concatenate
    datasets = [Dataset.from_file(file) for file in arrow_files]
    return concatenate_datasets(datasets)

def warm_cache(dspath):
    try:
        ds = load_dataset(dspath, split='train[:1M]', cache_dir="datacache")
    except Exception as e:
        pass

def prep_for_save():
    save_path = Path("data/guttenberg")
    save_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for dspath in ['imperial-cpg/project-gutenberg-extended', 'deepmind/pg19']:
        warm_cache(dspath)

    #gut = load_guttenberg_from_files()
    prep_for_save()
    os.system("cp datacache/imperial-cpg___project-gutenberg-extended/default/0.0.0/ca09993467e976b1f5d88dd58b2f00a039596871/project-gutenberg-extended-train-0000*-of-00005.arrow data/guttenberg/")
