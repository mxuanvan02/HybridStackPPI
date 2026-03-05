#!/usr/bin/env python3
"""
filter_homology.py
==================
Chuẩn bị bộ dữ liệu Positive chuẩn (Golden Standard Positive Dataset)
như quy trình được mô tả bởi Pan et al. (được áp dụng trong DIP, BioGRID benchmarks).

Hai bước thực thi:
1. Lọc độ dài: Bỏ qua các chuỗi protein có độ dài < 50 amino acids (quá ngắn để tạo thành cấu trúc ổn định).
2. Lọc độ tương đồng (Homology): Chạy CD-HIT ở ngưỡng 40% sequence identity để
   loại bỏ rò rỉ dữ liệu (data leakage) giữa các protein paralogs.

Yêu cầu:
- Phải cài đặt `cd-hit` trong hệ thống.
  (Cài đặt qua conda: `conda install -c bioconda cd-hit`
   hoặc macOS: `brew install cd-hit`)

Kết quả đầu ra:
Tạo ra `_filtered.fasta` và `_pairs_filtered.tsv` chứa bộ dữ liệu lõi tinh khiết.
"""

import argparse
import os
import subprocess
import sys
import pandas as pd
from Bio import SeqIO
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter PPI dataset by length <50aa and CD-HIT sequence similarity <40%."
    )
    parser.add_argument(
        "--fasta", required=True, type=str,
        help="Path to the original dataset FASTA file (e.g., human_dict.fasta)"
    )
    parser.add_argument(
        "--pairs", required=True, type=str,
        help="Path to the original pairs TSV file (e.g., human_pairs.tsv)"
    )
    parser.add_argument(
        "--output-dir", required=True, type=str,
        help="Directory to save the filtered dataset."
    )
    parser.add_argument(
        "--sim", type=float, default=0.4,
        help="Sequence identity threshold for CD-HIT (default 0.4 = 40%)."
    )
    parser.add_argument(
        "--min-len", type=int, default=50,
        help="Minimum sequence length to keep (default 50)."
    )
    return parser.parse_args()


def check_cd_hit_installed():
    """Kiểm tra xem cd-hit đã được cài đặt và cấu hình trong PATH chưa."""
    try:
        subprocess.run(["cd-hit", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ Lỗi: `cd-hit` chưa được cài đặt hoặc chưa thêm vào PATH!")
        print("💡 Hướng dẫn cài đặt:")
        print("   - Trên macOS: brew install cd-hit")
        print("   - Hoặc Conda: conda install -c bioconda cd-hit")
        sys.exit(1)


def filter_by_length(fasta_in: str, fasta_out: str, min_len: int) -> set:
    """Đọc FASTA, lọc bỏ chuỗi ngắn và trả về set chứa các Protein ID thỏa mãn."""
    print(f"[*] Step 1: Lọc các chuỗi < {min_len} amino acids...")
    
    valid_ids = set()
    records_to_keep = []
    
    for record in SeqIO.parse(fasta_in, "fasta"):
        # Chỉ lấy word đầu tiên của ID (phù hợp với quy chuẩn xử lý fasta của HybridStackPPI)
        protein_id = record.id.split()[0]
        
        if len(record.seq) >= min_len:
            # Ghi đè ID để đảm bảo tính nhất quán giữa file Pairs và file Fasta
            record.id = protein_id
            record.description = ""
            records_to_keep.append(record)
            valid_ids.add(protein_id)
            
    SeqIO.write(records_to_keep, fasta_out, "fasta")
    print(f"    -> Đã giữ lại {len(valid_ids)}/{len(records_to_keep)} proteins (Độ dài >= {min_len}).")
    return valid_ids


def run_cd_hit(fasta_in: str, fasta_out: str, sim_threshold: float):
    """Thực thi CD-HIT thông qua bash command."""
    print(f"[*] Step 2: Chạy CD-HIT để loại bỏ Homologous proteins (Sim > {sim_threshold*100}%)...")
    
    # CD-HIT yêu cầu word size (Tham số -n) phải nhỏ hơn khi threshold thấp.
    # 0.4 -> 0.5 cần -n 2
    # 0.5 -> 0.6 cần -n 3
    # 0.6 -> 0.7 cần -n 4
    if sim_threshold < 0.5:
        word_size = 2
    elif sim_threshold < 0.6:
        word_size = 3
    elif sim_threshold < 0.7:
        word_size = 4
    else:
        word_size = 5

    # Lệnh chuẩn của tác giả CD-HIT được recommend cho độ tương đồng thấp:
    # -c similarity -n word_size -M memory(MB) -T threads
    cmd = [
        "cd-hit",
        "-i", fasta_in,
        "-o", fasta_out,
        "-c", str(sim_threshold),
        "-n", str(word_size),
        "-M", "0",   # 0 = Use unlimited memory
        "-T", "0",   # 0 = Use all CPU cores
        "-d", "0"    # Use full sequence name as ID
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("    -> CD-HIT Clustering hoàn tất thành công.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy CD-HIT:\n{e.stderr.decode('utf-8')}")
        sys.exit(1)


def parse_cdhit_clusters(fasta_out: str) -> set:
    """Đọc file đầu ra của CD-HIT để trích xuất danh sách Protein ID 'Đại diện' (Representatives)."""
    representative_ids = set()
    for record in SeqIO.parse(fasta_out, "fasta"):
        # Ghi chú: cd-hit thường chèn thêm thông tin vào description, 
        # ta cần rã chuỗi để lấy đúng ID gốc.
        rep_id = record.id.split()[0]
        representative_ids.add(rep_id)
    return representative_ids


def filter_pairs_tsv(pairs_in: str, pairs_out: str, valid_ids: set):
    """
    Giữ lại các cặp tương tác mà *CẢ HAI* Protein đều nằm trong tập `valid_ids`.
    Nếu 1 trong 2 protein bị CD-HIT gom cụm (tương đồng > 40%) hoặc < 50aa -> Loại bỏ cặp đó.
    Đây là cách khắt khe nhất để bảo đảm tập dữ liệu sạch (Clean Benchmark Test).
    """
    print(f"[*] Step 3: Lọc (Filter) tập Positive Pairs ({pairs_in})...")
    
    # Read pairs and drop duplicates/NA (if any)
    df = pd.read_csv(pairs_in, sep="\t", header=None, names=["protein1", "protein2", "label"])
    initial_pairs_count = len(df)
    
    # Chỉ giữ cặp nào mà cả p1 và p2 đều là representative (độc lập sinh học)
    df_filtered = df[df["protein1"].isin(valid_ids) & df["protein2"].isin(valid_ids)]
    
    final_pairs_count = len(df_filtered)
    
    # Write to disk
    df_filtered.to_csv(pairs_out, sep="\t", header=False, index=False)
    
    print(f"    -> Tổng số cặp ban đầu: {initial_pairs_count:,}")
    print(f"    -> Tổng số cặp tinh khiết (Sau bộ lọc vàng): {final_pairs_count:,}")
    print(f"    -> Đã loại bỏ {initial_pairs_count - final_pairs_count:,} cặp gây nhiễu/Rò rỉ tương đồng.")
    print(f"[*] Đã xuất file thành công tới:\n  - {pairs_out}")


def main():
    args = parse_args()
    check_cd_hit_installed()
    
    input_fasta = Path(args.fasta)
    input_pairs = Path(args.pairs)
    output_dir = Path(args.output_dir)
    
    if not input_fasta.exists():
        print(f"❌ Lỗi: Không tìm thấy file {input_fasta}")
        sys.exit(1)
    if not input_pairs.exists():
        print(f"❌ Lỗi: Không tìm thấy file {input_pairs}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Naming standards
    prefix = input_fasta.stem.replace("_dict", "") # e.g. human_dict -> human
    temp_len_fasta = output_dir / f"{prefix}_temp_len.fasta"
    final_fasta = output_dir / f"{prefix}_filtered_dict.fasta"
    final_pairs = output_dir / f"{prefix}_filtered_pairs.tsv"
    
    print("=" * 70)
    print(f"BỘ QUY TẮC LỌC DỮ LIỆU CHUẨN PPI (THEO PAN ET AL.)")
    print(f" - Min length = {args.min_len}")
    print(f" - Sequence Maximum Identity = {args.sim * 100}%")
    print("=" * 70)
    
    # 1. Bọn chuỗi quá ngắn <50aa
    len_filtered_ids = filter_by_length(str(input_fasta), str(temp_len_fasta), args.min_len)
    
    # 2. Xóa bỏ Homology Leakage (Dùng CD-HIT)
    run_cd_hit(str(temp_len_fasta), str(final_fasta), args.sim)
    
    # 3. Trích xuất ID đại diện (Cluster representatives)
    representative_ids = parse_cdhit_clusters(str(final_fasta))
    print(f"    -> Phát hiện {len(representative_ids):,} protein clusters (các chuỗi độc lập sinh học).")
    
    # 4. Filter tập Pair khắt khe
    filter_pairs_tsv(str(input_pairs), str(final_pairs), representative_ids)
    
    # Dọn dẹp Clean up temp file
    if temp_len_fasta.exists():
        temp_len_fasta.unlink()
        
    # Xoá cả file thông tin mảng rác do CD-HIT tự sinh
    clstr_file = Path(str(final_fasta) + ".clstr")
    if clstr_file.exists():
        clstr_file.unlink()

    print("=" * 70)
    print("✅ Hoàn tất! Bạn có thể dùng bộ dữ liệu '_filtered' mới này để sinh Negative Sampler.")


if __name__ == "__main__":
    main()
