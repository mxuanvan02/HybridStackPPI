#!/usr/bin/env python3
"""
Artifact Manager for HybridStackPPI
===================================
Automated management of experimental results, plots, tables, and logs.

Features:
- Auto-creates timestamped result directories
- Standardized saving of plots (PNG 300dpi + PDF vector)
- LaTeX table generation from DataFrames
- Organized folder structure
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


class ArtifactManager:
    """
    Manages experimental artifacts with automatic organization.
    
    Creates directory structure:
        results/{timestamp}/
        ├── plots/
        ├── tables/
        ├── logs/
        └── models/
    
    Example:
        >>> manager = ArtifactManager(experiment_name="5fold_cv")
        >>> manager.save_plot(fig, "roc_curve")
        # Saves: results/20231203_120000_5fold_cv/plots/roc_curve.{png,pdf}
        
        >>> manager.save_table(df, "metrics")
        # Saves: results/20231203_120000_5fold_cv/tables/metrics.{csv,tex}
    """
    
    def __init__(
        self,
        base_dir: str = "results",
        experiment_name: Optional[str] = None,
        timestamp: bool = True,
    ):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for results
            experiment_name: Optional experiment identifier
            timestamp: Whether to add timestamp to directory name
        """
        # Generate directory name
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{ts}_{experiment_name}" if experiment_name else ts
        else:
            dir_name = experiment_name if experiment_name else "latest"
        
        self.root_dir = Path(base_dir) / dir_name
        
        # Create subdirectories
        self.plots_dir = self.root_dir / "plots"
        self.tables_dir = self.root_dir / "tables"
        self.logs_dir = self.root_dir / "logs"
        self.models_dir = self.root_dir / "models"
        
        for directory in [self.plots_dir, self.tables_dir, self.logs_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Artifact Manager initialized: {self.root_dir}")
    
    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        dpi: int = 300,
        formats: list = None,
    ) -> dict:
        """
        Save matplotlib figure in multiple formats.
        
        Args:
            fig: Matplotlib figure object
            name: Base filename (without extension)
            dpi: Resolution for raster formats
            formats: List of formats ['png', 'pdf', 'svg']
        
        Returns:
            Dictionary mapping format -> filepath
        """
        if formats is None:
            formats = ['png', 'pdf']  # Default: raster + vector
        
        saved_files = {}
        
        for fmt in formats:
            filepath = self.plots_dir / f"{name}.{fmt}"
            
            if fmt == 'png':
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(filepath, bbox_inches='tight')
            
            saved_files[fmt] = str(filepath)
        
        print(f"   ✅ Saved plot: {name} ({', '.join(formats)})")
        return saved_files
    
    def save_table(
        self,
        df: pd.DataFrame,
        name: str,
        formats: list = None,
        latex_caption: Optional[str] = None,
        latex_label: Optional[str] = None,
    ) -> dict:
        """
        Save DataFrame as CSV and optionally LaTeX table.
        
        Args:
            df: Pandas DataFrame
            name: Base filename
            formats: List of formats ['csv', 'tex', 'excel']
            latex_caption: Caption for LaTeX table
            latex_label: Label for LaTeX table
        
        Returns:
            Dictionary mapping format -> filepath
        """
        if formats is None:
            formats = ['csv', 'tex']
        
        saved_files = {}
        
        # CSV
        if 'csv' in formats:
            csv_path = self.tables_dir / f"{name}.csv"
            df.to_csv(csv_path, index=True)
            saved_files['csv'] = str(csv_path)
        
        # LaTeX
        if 'tex' in formats:
            tex_path = self.tables_dir / f"{name}.tex"
            
            # Generate LaTeX code
            latex_str = df.to_latex(
                float_format="%.4f",
                caption=latex_caption or f"Table: {name}",
                label=latex_label or f"tab:{name}",
            )
            
            with open(tex_path, 'w') as f:
                f.write(latex_str)
            
            saved_files['tex'] = str(tex_path)
        
        # Excel
        if 'excel' in formats:
            excel_path = self.tables_dir / f"{name}.xlsx"
            df.to_excel(excel_path, index=True)
            saved_files['excel'] = str(excel_path)
        
        print(f"   ✅ Saved table: {name} ({', '.join(formats)})")
        return saved_files
    
    def save_text(self, content: str, name: str, subdir: str = "logs") -> str:
        """
        Save text content to file.
        
        Args:
            content: Text to save
            name: Filename
            subdir: Subdirectory ('logs', 'tables', etc.)
        
        Returns:
            Path to saved file
        """
        if subdir == "logs":
            filepath = self.logs_dir / name
        elif subdir == "tables":
            filepath = self.tables_dir / name
        else:
            filepath = self.root_dir / subdir / name
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"   ✅ Saved text: {name}")
        return str(filepath)
    
    def save_model(self, model, name: str) -> str:
        """
        Save trained model (placeholder for pickle/joblib).
        
        Args:
            model: Model object
            name: Model filename
        
        Returns:
            Path to saved model
        """
        import joblib
        
        filepath = self.models_dir / f"{name}.pkl"
        joblib.dump(model, filepath)
        
        print(f"   ✅ Saved model: {name}")
        return str(filepath)
    
    def get_path(self, artifact_type: str, name: str) -> Path:
        """
        Get path for artifact without saving.
        
        Args:
            artifact_type: 'plot', 'table', 'log', 'model'
            name: Artifact name
        
        Returns:
            Path object
        """
        type_map = {
            'plot': self.plots_dir,
            'table': self.tables_dir,
            'log': self.logs_dir,
            'model': self.models_dir,
        }
        
        base_dir = type_map.get(artifact_type, self.root_dir)
        return base_dir / name
    
    def create_summary(self) -> str:
        """
        Create summary report of all artifacts.
        
        Returns:
            Summary text
        """
        summary = []
        summary.append("=" * 70)
        summary.append("ARTIFACT SUMMARY")
        summary.append("=" * 70)
        summary.append(f"\nRoot Directory: {self.root_dir}")
        summary.append("")
        
        for subdir_name, subdir_path in [
            ("Plots", self.plots_dir),
            ("Tables", self.tables_dir),
            ("Logs", self.logs_dir),
            ("Models", self.models_dir),
        ]:
            files = list(subdir_path.glob("*"))
            summary.append(f"{subdir_name}: {len(files)} files")
            for f in sorted(files)[:10]:  # Show first 10
                size_kb = f.stat().st_size / 1024
                summary.append(f"  - {f.name} ({size_kb:.1f} KB)")
            if len(files) > 10:
                summary.append(f"  ... and {len(files) - 10} more")
            summary.append("")
        
        summary.append("=" * 70)
        
        summary_text = "\n".join(summary)
        
        # Save summary
        summary_path = self.root_dir / "SUMMARY.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        return summary_text
    
    def archive(self, archive_name: Optional[str] = None) -> str:
        """
        Create zip archive of all artifacts.
        
        Args:
            archive_name: Optional custom archive name
        
        Returns:
            Path to archive file
        """
        if archive_name is None:
            archive_name = f"{self.root_dir.name}_archive"
        
        archive_path = self.root_dir.parent / archive_name
        
        shutil.make_archive(str(archive_path), 'zip', self.root_dir)
        
        print(f"📦 Created archive: {archive_path}.zip")
        return f"{archive_path}.zip"


# Convenience function
def create_artifact_manager(experiment_name: str = None, **kwargs) -> ArtifactManager:
    """
    Factory function to create ArtifactManager.
    
    Args:
        experiment_name: Experiment identifier
        **kwargs: Additional arguments for ArtifactManager
    
    Returns:
        ArtifactManager instance
    """
    return ArtifactManager(experiment_name=experiment_name, **kwargs)
