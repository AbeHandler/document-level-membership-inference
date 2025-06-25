#!/bin/bash

set -e  # stop if any command fails
eval "$(conda shell.bash hook)"
conda activate doc_membership
huggingface-cli login --token $(cat  ~/.cache/huggingface/token_read)

export CUDA_VISIBLE_DEVICES=1

HF_MODEL=${1:-"dobolyilab/MISQSIPressPublic-bl1-124M"}  # default if not provided
MODEL_ID=$(basename "$HF_MODEL")  # gets 'open_llama_3b' from full model name
MEMBER_DATASET_NAME=${2:-"blockeddocs"}
N_POS_CHUNK=${3:-200}
NON_MEMBER_DATASET_NAME=$MEMBER_DATASET_NAME
CHUNK_PREFIX=$MEMBER_DATASET_NAME
TOKENIZER_PATH="./pretrained/tokenizers/$MODEL_ID"
TOKENIZED_MEMBER_PATH="data/tokenized/$MODEL_ID/$MEMBER_DATASET_NAME"
TOKENIZED_NON_MEMBER_PATH="data/tokenized/$MODEL_ID/$NON_MEMBER_DATASET_NAME"
N_CHUNKS=5
CHUNK_ID="XX"
NB_SAMPLES=400
MAX_LEN=128
STRIDE=127
SEED=42

RAW_DATA_PATH="data/final_chunks/${CHUNK_PREFIX}_${CHUNK_ID}_min_tokens100_seed42"
LABELS_PATH="data/final_chunks/${CHUNK_PREFIX}_${CHUNK_ID}_labels.pickle"
PPL_PATH="perplexity_results/perplexity_${MODEL_ID}_${MODEL_ID}_${CHUNK_PREFIX}_${CHUNK_ID}_min_tokens100_seed42__${NB_SAMPLES}_${MAX_LEN}_${STRIDE}_seed${SEED}.pickle"
NORM_PATH="data/final_chunks/general_proba/general_proba_${CHUNK_PREFIX}_${CHUNK_ID}_${MAX_LEN}.pickle"
EXP_NAME="${CHUNK_PREFIX}_${MODEL_ID}_chunk${CHUNK_ID}"
OUTDIR="./classifier_results/chunks"

python src/import_huggingface_model.py --model="$HF_MODEL" --write_dir="pretrained"

# 👀 Clear cache before main run!
python download_data.py  # will download everything from datasets.txt, be sure to clear cache

python src/tokenize_data.py --data_dir="./data" --hfpath="$HF_MODEL" --path_to_dataset="data/$NON_MEMBER_DATASET_NAME" --nb_workers=4 --path_to_tokenizer="$TOKENIZER_PATH"
python src/tokenize_data.py --data_dir="./data" --hfpath="$HF_MODEL" --path_to_dataset="data/$MEMBER_DATASET_NAME" --nb_workers=4 --path_to_tokenizer="$TOKENIZER_PATH"

python src/split_chunks.py -c config/split_chunks.ini \
  --path_to_member_data="$TOKENIZED_MEMBER_PATH" \
  --path_to_non_member_data="$TOKENIZED_NON_MEMBER_PATH" \
  --prefix="$CHUNK_PREFIX" \
  --n_pos_chunk=$N_POS_CHUNK


for chunk in $(seq 0 $((N_CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=3 python src/compute_perplexity.py --data_dir='./data' \
        --path_to_tokenizer="./pretrained/tokenizers/$MODEL_ID" \
        --path_to_model="./pretrained/models/$MODEL_ID" \
        --path_to_dataset="./data/final_chunks/${CHUNK_PREFIX}_${chunk}_min_tokens100_seed42" \
        --results_dir='./perplexity_results' --nb_samples=400 --stride=127 --max_length=128 \
        --top_probas=10 --shuffle=0 \
        --general_proba_path="./data/final_chunks/general_proba/general_proba_${CHUNK_PREFIX}_${chunk}_128.pickle" \
        --token_freq_path="./data/final_chunks/token_freq/token_freq_${CHUNK_PREFIX}_${chunk}.pickle"
done


python main.py \
  --experiment_name "$EXP_NAME" \
  --output_dir "$OUTDIR" \
  --n_chunks $N_CHUNKS \
  --path_to_raw_data "$RAW_DATA_PATH" \
  --path_to_labels "$LABELS_PATH" \
  --path_to_perplexity_results "$PPL_PATH" \
  --path_to_normalization_dict "$NORM_PATH" \
  --norm_type "ratio" \
  --feat_extraction_type "hist_1000" \
  --models "logistic_regression,random_forest" \
  --seed "$SEED" \
  --nb_samples "$NB_SAMPLES"
