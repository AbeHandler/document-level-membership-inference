
#python src/import_huggingface_model.py --model="openlm-research/open_llama_3b" --write_dir="pretrained"
#python download_data.py

ds=$(find datacache/ -type d| grep deepmind___pg19 | grep default/0.1.0/ | head  -1)
python src/tokenize_data.py --data_dir="./data" --path_to_tokenizer="./pretrained/tokenizers/open_llama_3b" --path_to_dataset=$ds --nb_workers=4

