
python src/import_huggingface_model.py --model="openlm-research/open_llama_3b" --write_dir="pretrained"
python download_data.py

mkdir -p datacache/gutenberg/

cp datacache/imperial-cpg___parquet/imperial-cpg--project-gutenberg-extended-51580d88358a5c2a/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec/*arrow datacache/gutenberg/
