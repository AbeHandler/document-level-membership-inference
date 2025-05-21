
rule analysis:
    output:
        ".meeus.init"
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
