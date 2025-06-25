
rule analysis:
    output:
        ".snake.init"
    shell:
        r"""
        # 1. Create the env (if it doesn’t already exist)
        conda create --name doc_membership python=3.9 -y

        # 2. “Hook” conda into this shell session
        eval "$(conda shell.bash hook)"

        # 3. Now you can activate
        conda activate doc_membership

        # 4. Install your requirements
        pip install -r requirements.txt

        # 5. Wipe the cache
        rm -rf data

        # 6. Mark as completed
        touch {output}
        """

rule copyright:
    shell:
        "./go.sh dobolyilab/blockbench-noblocksbin copywritetraps 100 200"

# I think this is the input but not 100% sure
# 👀 $ snakemake classifier_results/chunks/blockeddocs_MISQSIPressPublic-bl1-124M_chunkXX.csv -j 1
# snakemake classifier_results/chunks/blockeddocs_MISQSIPressPublic-bl1-124M_chunk00.csv --dag | dot -Tpng > dag.png && open dag.png
rule run_model:
    input:
        ".snake.init"
    output:
        "classifier_results/chunks/{dataset}_{model_id}_chunk{chunk}.csv"
    params:
        hf_model=lambda wildcards: f"dobolyilab/{wildcards.model_id}",
        dataset=lambda wildcards: wildcards.dataset
    shell:
        "./go.sh {params.hf_model} {params.dataset}"
