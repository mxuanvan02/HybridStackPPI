#!/usr/bin/env python3
"""
Automated Report Generator for SOTA Benchmarks
==============================================
Generate LaTeX tables and comparison charts from benchmark results.

Outputs:
- LaTeX tables for inclusion in papers
- Bar charts for visual comparison
- Statistical summaries
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReportGenerator:
    """Generate publication-ready comparison reports."""
    
    def __init__(self, results_file: str, output_dir: str = "reports"):
        self.results_file = results_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        with open(results_file, "r") as f:
            self.results = json.load(f)
        
        # Metric display names
        self.metric_display_names = {
            "accuracy": "Acc",
            "precision": "Prec",
            "recall": "Recall",
            "f1": "F1",
            "mcc": "MCC",
            "auc_roc": "AUC-ROC",
            "auc_pr": "AUC-PR",
        }
    
    def generate_latex_table(
        self,
        dataset_name: str,
        metrics_to_include: List[str] = None,
    ) -> str:
        """
        Generate LaTeX table code for a dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., 'human', 'yeast')
            metrics_to_include: List of metrics to include (default: all)
        
        Returns:
            LaTeX table code as string
        """
        if dataset_name not in self.results:
            print(f"⚠️  Dataset '{dataset_name}' not found in results")
            return ""
        
        dataset_results = self.results[dataset_name]
        
        # Default metrics if not specified
        if metrics_to_include is None:
            metrics_to_include = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"]
        
        # Methods in order (put HybridStackPPI last for emphasis)
        method_order = []
        for method in ["pipr", "rapppid", "deeptrio", "lpbert"]:
            if method in dataset_results:
                method_order.append(method)
        if "hybridstack" in dataset_results:
            method_order.append("hybridstack")
        
        # Start LaTeX table
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Performance Comparison on BioGrid {dataset_name.capitalize()} Dataset}}")
        latex.append(f"\\label{{tab:comparison_{dataset_name}}}")
        
        # Column specification
        n_cols = len(metrics_to_include) + 1  # +1 for method name
        col_spec = "l" + "c" * len(metrics_to_include)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        # Header row
        header = ["Method"]
        for metric in metrics_to_include:
            display_name = self.metric_display_names.get(metric, metric.upper())
            header.append(display_name)
        latex.append(" & ".join(header) + " \\\\")
        latex.append("\\midrule")
        
        # Data rows
        for method in method_order:
            row = []
            
            # Method name
            method_display = {
                "hybridstack": "\\textbf{HybridStackPPI (Ours)}",
                "pipr": "PIPR",
                "rapppid": "RAPPPID",
                "deeptrio": "DeepTrio",
                "lpbert": "LPBERT",
            }.get(method, method.upper())
            row.append(method_display)
            
            # Metric values
            metrics = dataset_results[method]
            for metric in metrics_to_include:
                value = metrics.get(metric, None)
                if value is not None:
                    # Bold if HybridStackPPI and best value
                    formatted = f"{value:.3f}"
                    if method == "hybridstack":
                        formatted = f"\\textbf{{{formatted}}}"
                    row.append(formatted)
                else:
                    row.append("---")
            
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_comparison_chart(
        self,
        dataset_name: str,
        metrics_to_plot: List[str] = None,
    ) -> str:
        """
        Generate bar chart comparing methods.
        
        Args:
            dataset_name: Name of dataset
            metrics_to_plot: List of metrics to visualize
        
        Returns:
            Path to saved PNG file
        """
        if dataset_name not in self.results:
            print(f"⚠️  Dataset '{dataset_name}' not found in results")
            return ""
        
        dataset_results = self.results[dataset_name]
        
        # Default metrics
        if metrics_to_plot is None:
            metrics_to_plot = ["accuracy", "f1", "auc_roc"]
        
        # Extract data for plotting
        methods = []
        data_by_metric = {metric: [] for metric in metrics_to_plot}
        
        for method in ["pipr", "rapppid", "deeptrio", "lpbert", "hybridstack"]:
            if method not in dataset_results:
                continue
            
            methods.append(method.upper() if method != "hybridstack" else "HybridStack\n(Ours)")
            
            for metric in metrics_to_plot:
                value = dataset_results[method].get(metric, 0.0)
                data_by_metric[metric].append(value)
        
        # Create grouped bar chart
        x = np.arange(len(methods))
        width = 0.25  # Width of bars
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
        
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot)/2 + 0.5) * width
            display_name = self.metric_display_names.get(metric, metric.upper())
            bars = ax.bar(x + offset, data_by_metric[metric], width, 
                          label=display_name, color=colors[i % len(colors)],
                          alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Comparison on BioGrid {dataset_name.capitalize()}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight HybridStackPPI
        if "hybridstack" in [m.lower() for m in methods]:
            hybrid_idx = len(methods) - 1
            ax.axvspan(hybrid_idx - 0.5, hybrid_idx + 0.5, alpha=0.1, color='gold')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"comparison_{dataset_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Chart saved: {output_path}")
        return output_path
    
    def calculate_statistics(self) -> pd.DataFrame:
        """Calculate summary statistics across all datasets."""
        stats = []
        
        for dataset_name, dataset_results in self.results.items():
            for method, metrics in dataset_results.items():
                for metric_name, value in metrics.items():
                    stats.append({
                        "Dataset": dataset_name,
                        "Method": method,
                        "Metric": metric_name,
                        "Value": value,
                    })
        
        df = pd.DataFrame(stats)
        return df
    
    def export_full_report(self) -> None:
        """Generate complete report with all tables and charts."""
        print("\n" + "=" * 70)
        print("GENERATING FULL REPORT")
        print("=" * 70)
        
        # Generate LaTeX tables for each dataset
        print("\n📝 Generating LaTeX tables...")
        for dataset_name in self.results.keys():
            latex_table = self.generate_latex_table(dataset_name)
            
            if latex_table:
                output_file = os.path.join(self.output_dir, f"table_{dataset_name}.tex")
                with open(output_file, "w") as f:
                    f.write(latex_table)
                print(f"   ✅ {output_file}")
        
        # Generate comparison charts
        print("\n📊 Generating comparison charts...")
        for dataset_name in self.results.keys():
            self.generate_comparison_chart(dataset_name)
        
        # Generate summary statistics
        print("\n📈 Generating summary statistics...")
        stats_df = self.calculate_statistics()
        stats_file = os.path.join(self.output_dir, "summary_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"   ✅ {stats_file}")
        
        # Generate summary pivot table
        if not stats_df.empty:
            pivot = stats_df.pivot_table(
                values="Value",
                index=["Method", "Metric"],
                columns="Dataset",
                aggfunc="mean"
            )
            pivot_file = os.path.join(self.output_dir, "pivot_table.csv")
            pivot.to_csv(pivot_file)
            print(f"   ✅ {pivot_file}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("✅ REPORT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated files in {self.output_dir}/:")
        print("  - LaTeX tables: table_*.tex")
        print("  - Comparison charts: comparison_*.png")
        print("  - Summary statistics: summary_statistics.csv")
        print("  - Pivot table: pivot_table.csv")
        print("\nNext steps:")
        print("  1. Include LaTeX tables in your paper")
        print("  2. Use comparison charts in presentations")
        print("  3. Review summary statistics for insights")
        print("=" * 70)


def main():
    """Main report generator."""
    parser = argparse.ArgumentParser(description="Generate SOTA comparison reports")
    parser.add_argument(
        "--input",
        default="results/benchmark_summary.json",
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names or 'all'"
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["latex", "chart", "stats", "all"],
        help="Output format to generate"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Results file not found: {args.input}")
        print(f"   Run benchmarks first: python benchmarks/runner.py")
        return 1
    
    # Initialize generator
    generator = ReportGenerator(args.input, args.output)
    
    # Generate full report
    if args.format == "all":
        generator.export_full_report()
    else:
        # Generate specific format
        if args.format == "latex":
            for dataset in generator.results.keys():
                latex = generator.generate_latex_table(dataset)
                print(f"\n{latex}\n")
        elif args.format == "chart":
            for dataset in generator.results.keys():
                generator.generate_comparison_chart(dataset)
        elif args.format == "stats":
            stats = generator.calculate_statistics()
            print(stats)
    
    return 0


if __name__ == "__main__":
    exit(main())
