import os

def fix_fasta(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('>'):
                f_out.write(line)
            else:
                f_out.write(line.replace('U', 'C'))
    print(f"Fixed FASTA saved to {output_path}")

fix_fasta('results/github_baselines/proteinprompt_human_same_go/tmp/human_proteinprompt_input.fasta', 
          'results/github_baselines/proteinprompt_human_same_go/tmp/human_proteinprompt_input_fixed.fasta')
