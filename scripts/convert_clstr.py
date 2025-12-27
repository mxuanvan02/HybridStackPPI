import sys
import re

def parse_clstr_to_csv(clstr_path, csv_path):
    print(f"Parsing {clstr_path} to {csv_path}...")
    clusters = []
    current_cluster_members = []
    representative = None

    with open(clstr_path, 'r') as f_in:
        for line in f_in:
            if line.startswith(">Cluster"):
                if current_cluster_members:
                    clusters.append((current_cluster_members, representative))
                current_cluster_members = []
                representative = None
            else:
                # Format: 0	123aa, >ProteinID... *
                match = re.search(r">(.+?)\.\.\.", line)
                if match:
                    protein_id = match.group(1)
                    current_cluster_members.append(protein_id)
                    if "*" in line:
                        representative = protein_id
        
        # Add last cluster
        if current_cluster_members:
            clusters.append((current_cluster_members, representative))

    with open(csv_path, 'w') as f_out:
        for members, rep in clusters:
            if not rep:
                # Should not happen with CD-HIT output, but for safety:
                rep = members[0]
            for m in members:
                f_out.write(f"{m},{rep}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_clstr.py <input.clstr> <output.csv>")
    else:
        parse_clstr_to_csv(sys.argv[1], sys.argv[2])
