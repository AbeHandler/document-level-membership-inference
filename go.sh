
#python src/import_huggingface_model.py --model="openlm-research/open_llama_3b" --write_dir="pretrained"
#python download_data.py


python src/tokenize_data.py --data_dir="./data" --path_to_tokenizer="./pretrained/tokenizers/open_llama_3b" --path_to_dataset=data/blockeddocs --nb_workers=4
python src/tokenize_data.py --data_dir="./data" --path_to_tokenizer="./pretrained/tokenizers/open_llama_3b" --path_to_dataset=data/project-gutenberg-extended --nb_workers=4
