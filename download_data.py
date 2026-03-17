import argparse
import os
import shutil
from pathlib import Path
from datasets import load_dataset, Dataset


def load_url_set(filepath: str) -> set:
    """
    Load URLs from a text file into a set.

    Args:
        filepath: Path to text file with one URL per line

    Returns:
        Set of URLs with whitespace stripped
    """
    with open(filepath) as f:
        return set(line.strip() for line in f)


bothbins_file = "data/interim/R2/cleaning/verified_bothbins_urls.txt"
excluded_file = "data/interim/R2/cleaning/verified_excluded_urls.txt"

if not Path(bothbins_file).exists() or not Path(excluded_file).exists():
    # Create directory if needed
    Path("data/interim/R2/cleaning").mkdir(parents=True, exist_ok=True)

    cmd1 = '''cp /Users/abha4861/dolma/data/interim/R2/cleaning/verified_bothbins_urls.txt data/interim/R2/cleaning/verified_bothbins_urls.txt'''
    cmd2 = '''cp /Users/abha4861/dolma/data/interim/R2/cleaning/verified_excluded_urls.txt data/interim/R2/cleaning/verified_excluded_urls.txt'''
    os.system(cmd1)
    os.system(cmd2)

bothbins_urls = load_url_set("data/interim/R2/cleaning/verified_bothbins_urls.txt")
excluded_urls = load_url_set("data/interim/R2/cleaning/verified_excluded_urls.txt")

def dataset_exists(output_path):
    """Check if dataset already exists on disk."""
    path = Path(output_path)
    return path.exists() and path.is_dir() and any(path.iterdir())


def get_dataset_name(dspath):
    """Extract dataset name from path."""
    return dspath.split("/").pop()


def download_standard_dataset(dspath, data_dir="data", cache_dir="datacache", skip_existing=True):
    """Download a standard dataset from HuggingFace."""
    dsname = get_dataset_name(dspath)
    output_path = Path(data_dir) / dsname

    if skip_existing and dataset_exists(output_path):
        print(f"Dataset {dsname} already exists at {output_path}, skipping download")
        return

    print(f"Downloading {dsname} to {output_path}")
    try:
        ds_streaming = load_dataset(dspath, split="train", cache_dir=cache_dir, streaming=True)
        ds = ds_streaming.take(50000)
        ds_fixed = Dataset.from_generator(lambda: ds)
        ds_fixed.save_to_disk(str(output_path))
        print(f"Successfully downloaded {dsname}")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError while downloading {dsname}: {e}")


def download_standard_dataset_complete(dspath, data_dir="data", cache_dir="datacache", skip_existing=True):
    """Download a complete standard dataset from HuggingFace (no streaming, full dataset)."""
    dsname = get_dataset_name(dspath)
    output_path = Path(data_dir) / dsname

    if skip_existing and dataset_exists(output_path):
        print(f"Dataset {dsname} already exists at {output_path}, skipping download")
        return

    print(f"Downloading {dsname} to {output_path}")
    try:
        ds = load_dataset(dspath, split="train", cache_dir=cache_dir)

        # Filter based on URL membership
        if dsname.endswith("bothbins"):
            print(f"Filtering {dsname} to only include URLs in bothbins_urls")
            ds = ds.filter(lambda x: x["url"] in bothbins_urls)
        elif dsname.endswith("excluded-docs"):
            print(f"Filtering {dsname} to only include URLs in excluded_urls")
            ds = ds.filter(lambda x: x["url"] in excluded_urls)

        ds.save_to_disk(str(output_path))
        print(f"Successfully downloaded {dsname}")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError while downloading {dsname}: {e}")


def download_suffixes_dataset(data_dir="data", skip_existing=True):
    """Download and process the suffixesnoblocksbin dataset."""
    output_path = Path(data_dir) / "suffixesnoblocksbin"

    if skip_existing and dataset_exists(output_path):
        print(f"Dataset suffixesnoblocksbin already exists at {output_path}, skipping download")
        return

    print(f"Downloading suffixesnoblocksbin to {output_path}")
    ds = load_dataset("abehandlerorg/suffixesnoblocksbin", split="train")
    ds = ds.filter(lambda x: x["blocksbin"] == 0)
    ds = ds.map(lambda x: {"text": x["sequence"]}, remove_columns=["sequence"])
    ds = ds.map(lambda x: {"id": x["text"]})
    ds.save_to_disk(str(output_path))
    print(f"Successfully downloaded suffixesnoblocksbin")


def clear_cache(data_dir="data", cache_dir="datacache"):
    """Clear all downloaded data and cache."""
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)

    if data_path.exists():
        print(f"Removing data directory: {data_path}")
        shutil.rmtree(data_path)

    if cache_path.exists():
        print(f"Removing cache directory: {cache_path}")
        shutil.rmtree(cache_path)

    print("Cache cleared successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download datasets for document-level membership inference"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all downloaded data and cache before exiting"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if data already exists"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/storage/abe",
        help="Directory to save downloaded datasets (default: data)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="datacache",
        help="Directory for HuggingFace cache (default: datacache)"
    )
    parser.add_argument(
        "--datasets-file",
        type=str,
        default="datasets.txt",
        help="File containing list of datasets to download (default: datasets.txt)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Clear cache if requested (and exit)
    if args.clear_cache:
        clear_cache(data_dir=args.data_dir, cache_dir=args.cache_dir)
        return

    # Create data directory if it doesn't exist
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    # Read datasets list
    with open(args.datasets_file) as f:
        datasets = [line.strip() for line in f if line.strip()]

    # Download standard datasets
    skip_existing = not args.force
    for dspath in datasets:
        download_standard_dataset(
            dspath,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            skip_existing=skip_existing
        )

    # Download and process suffixes dataset
    download_suffixes_dataset(data_dir=args.data_dir, skip_existing=skip_existing)

    print("All downloads complete")


if __name__ == "__main__":
    main()