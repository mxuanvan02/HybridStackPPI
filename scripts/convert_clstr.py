import sys
import re

def parse_clstr_to_csv(clstr_path, csv_path):
    print(f"Parsing {clstr_path} to {csv_path}...")
    with open(clstr_path, 'r') as f_in, open(csv_path, 'w') as f_out:
        current_cluster = ""
        for line in f_in:
            if line.startswith(">Cluster"):
                current_cluster = line.strip().replace(">", "")
            else:
                match = re.search(r">(.+?)\.\.\.", line)
                if match:
                    protein_id = match.group(1)
                    f_out.write(f"{protein_id},{current_cluster}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_clstr.py <input.clstr> <output.csv>")
    else:
        parse_clstr_to_csv(sys.argv[1], sys.argv[2])
