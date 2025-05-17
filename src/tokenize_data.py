## this is a slight modification of the following. I hit errors loading from HF cache so am loading from files
## https://github.com/computationalprivacy/document-level-membership-inference/blob/main/src/tokenize_data.py

from transformers import LlamaTokenizer
from datasets import Dataset
from datasets import concatenate_datasets
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--path_to_tokenizer", type=str, required=True)
parser.add_argument("--path_to_dataset", required=False, type=str)
parser.add_argument("--nb_workers", type=int, default=10)
parser.add_argument("--max_shard_size", type=str, default="4GB")
args = parser.parse_args()


def load_from_files(dataset):

    gutenberg = "/scratch/alpine/abha4861/hfdatasets/imperial-cpg___parquet/imperial-cpg--project-gutenberg-extended-51580d88358a5c2a/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec/"
    pg19 = "/scratch/alpine/abha4861/hfdatasets/deepmind___pg19/default/0.1.0/fb74320038a3c19e3cc87375222fc75ed3c8dc5a739b3e8dc835736388a7a882/"

    gutenberg_files = [
        f"{gutenberg}/parquet-train-00000-of-00004.arrow",
        f"{gutenberg}/parquet-train-00001-of-00004.arrow",
        f"{gutenberg}/parquet-train-00002-of-00004.arrow",
        f"{gutenberg}/parquet-train-00003-of-00004.arrow",
    ]

    pg19_files = [
        f"{pg19}/pg19-train-00000-of-00015.arrow",
        f"{pg19}/pg19-train-00001-of-00015.arrow",
        f"{pg19}/pg19-train-00002-of-00015.arrow",
        f"{pg19}/pg19-train-00003-of-00015.arrow",
    ]

    arrow_files = None
    if dataset == "pg19":
        arrow_files = pg19_files
    elif dataset == "gutenberg":
        arrow_files = gutenberg_files
    else:
        raise ValueError("bad dataset")

    # Load each Arrow file and concatenate
    datasets = [Dataset.from_file(file) for file in arrow_files]
    return concatenate_datasets(datasets)


def main():
    PATH_TO_DATASET = args.path_to_dataset
    DATASET_NAME = PATH_TO_DATASET.split("/")[-1]
    PATH_TO_TOKENIZER = args.path_to_tokenizer
    TOKENIZER_NAME = PATH_TO_TOKENIZER.split("/")[-1]

    tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_TOKENIZER)

    print(f"Loading {DATASET_NAME}...")

    if "pg19" in PATH_TO_DATASET:
        datasetname = "pg19"
    elif "gutenberg" in PATH_TO_DATASET:
        datasetname = "gutenberg"
    else:
        raise ValueError("unk dataset")

    dataset = load_from_files(datasetname)

    print(f"Starting tokenization {datasetname}...")
    tokenized_dataset = dataset.map(
        lambda samples: tokenizer(samples["text"]),
        batched=False,
        num_proc=args.nb_workers,
        remove_columns=["text"],
        load_from_cache_file=False,
        desc=f"Running {TOKENIZER_NAME} tokenizer on {DATASET_NAME}",
    )

    print(f"Tokenization done for {datasetname}...")

    tokenized_output_dir = f"tokenized/{TOKENIZER_NAME}/{datasetname}"
    Path(tokenized_output_dir).mkdir(exist_ok=True, parents=True)

    tokenized_dataset.save_to_disk(
        tokenized_output_dir,
        max_shard_size=args.max_shard_size,
        num_proc=args.nb_workers,
    )

    print(f"Tokenized dataset saved for {datasetname}")


if __name__ == "__main__":
    main()
