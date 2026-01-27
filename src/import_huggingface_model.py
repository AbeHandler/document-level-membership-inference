# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, default="openlm-research/open_llama_7b")
parser.add_argument( "--write_dir", type=str, required=True)
parser.add_argument("--max_shard_size", type=str, default="4GB")

args = parser.parse_args()

MODEL_NAME = args.model
WRITE_DIR = args.write_dir
MAX_SHARD_SIZE = args.max_shard_size

def main():
    model_short_name = MODEL_NAME.split('/')[1]
    tokenizer_path = f"{WRITE_DIR}/tokenizers/{model_short_name}"
    model_path = f"{WRITE_DIR}/models/{model_short_name}"

    # load and write the tokenizer
    if os.path.exists(tokenizer_path):
        print(f'Tokenizer already exists at {tokenizer_path}, skipping download.')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(tokenizer_path)
        print(f'The tokenizer for {MODEL_NAME} has been saved successfully.')

    # load and write the model
    if os.path.exists(model_path):
        print(f'Model already exists at {model_path}, skipping download.')
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.save_pretrained(model_path, max_shard_size=MAX_SHARD_SIZE)
        print(f'The model {MODEL_NAME} has been saved successfully.')

if __name__ == "__main__":
    main()
