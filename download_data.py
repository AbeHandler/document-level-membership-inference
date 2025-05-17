from datasets import load_dataset
from datasets import Dataset
from pathlib import Path
from datasets import concatenate_datasets

# in slurm => export HF_DATASETS_CACHE=/scratch/alpine/$USER/hfdatasets

dataset_path = "/scratch/alpine/abha4861/hfdatasets/imperial-cpg___parquet/imperial-cpg--project-gutenberg-extended-51580d88358a5c2a/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec/"

def load_guttenberg_from_files():
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
        ds = load_dataset(dspath, split='train[:1M]')
    except Exception as e:
        pass

def prep_for_save():
    save_path = Path("data/guttenberg")
    save_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for dspath in ['imperial-cpg/project-gutenberg-extended', 'deepmind/pg19']:
        warm_cache(dspath)

    gut = load_guttenberg_from_files()
    prep_for_save()
    gut.save_to_disk("data/guttenberg")
