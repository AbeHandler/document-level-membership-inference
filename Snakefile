rule init:
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


rule miacomparison:
    output:
        '.snake.mia_comparison'
    shell:
        """
        $ python scripts/prepare_same_dataset_used_for_mia_comparison_in_mimir.py 
        ./go.sh dobolyilab/blockbench-noblocksbin abehandlerorg/minhashblocksample_targetsonly_doc_level_mia
        ./go.sh dobolyilab/blockbench-blocksbin abehandlerorg/minhashblocksample_targetsonly_doc_level_mia
        touch {output}
        """

rule copyright:
    input:
        ".snake.init"
    output:
        'copyrighttraps.csv'
    shell:
        """
        ./go.sh dobolyilab/blockbench-noblocksbin copywritetraps 500 500 1
        ./go.sh dobolyilab/blockbench-blocksbin copywritetraps 500 500 1
        python scripts/merge_copyright.py
        touch {output}
        """