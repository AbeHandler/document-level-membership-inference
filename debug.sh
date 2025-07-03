#!/bin/bash

set -e  # stop if any command fails
eval "$(conda shell.bash hook)"
conda activate doc_membership
huggingface-cli login --token $(cat  ~/.cache/huggingface/token_read)

GPU=1
export CUDA_VISIBLE_DEVICES=$GPU

HF_MODEL=${1:-"dobolyilab/MISQSIPressPublic-bl1-124M"}  # default if not provided
MODEL_ID=$(basename "$HF_MODEL")  # gets 'open_llama_3b' from full model name
MEMBER_DATASET_NAME=${2:-"blockeddocs"}
N_POS_CHUNK=${3:-200}
NB_SAMPLES=${4:-400}
N_CHUNKS=${5:-5}
echo $N_POS_CHUNK
echo $NB_SAMPLES
echo $N_CHUNKS
NON_MEMBER_DATASET_NAME='project-gutenberg-extended'
CHUNK_PREFIX=$MEMBER_DATASET_NAME
TOKENIZER_PATH="./pretrained/tokenizers/$MODEL_ID"
TOKENIZED_MEMBER_PATH="data/tokenized/$MODEL_ID/$MEMBER_DATASET_NAME"
TOKENIZED_NON_MEMBER_PATH="data/tokenized/$MODEL_ID/$NON_MEMBER_DATASET_NAME"
CHUNK_ID="XX"
MAX_LEN=128
STRIDE=127
SEED=42

RAW_DATA_PATH="data/final_chunks/${CHUNK_PREFIX}_${CHUNK_ID}_min_tokens100_seed42"
LABELS_PATH="data/final_chunks/${CHUNK_PREFIX}_${CHUNK_ID}_labels.pickle"
PPL_PATH="perplexity_results/perplexity_${MODEL_ID}_${MODEL_ID}_${CHUNK_PREFIX}_${CHUNK_ID}_min_tokens100_seed42__${NB_SAMPLES}_${MAX_LEN}_${STRIDE}_seed${SEED}.pickle"
NORM_PATH="data/final_chunks/general_proba/general_proba_${CHUNK_PREFIX}_${CHUNK_ID}_${MAX_LEN}.pickle"
EXP_NAME="${CHUNK_PREFIX}_${MODEL_ID}_chunk${CHUNK_ID}"
OUTDIR="./classifier_results/chunks"

echo $PPL_PATH


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
