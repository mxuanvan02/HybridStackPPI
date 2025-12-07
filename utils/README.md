# Utils Package Documentation

## Overview

Professional utility modules for HybridStackPPI experiment management.

## Modules

### 1. `artifact_manager.py` - Artifact Management

Automated organization and saving of experimental results.

**Features**:
- Timestamped result directories
- Multi-format plot saving (PNG 300dpi + PDF vector)
- LaTeX table generation from DataFrames
- Organized folder structure
- Archive creation

**Usage**:

```python
from utils import ArtifactManager

# Create manager
manager = ArtifactManager(experiment_name="5fold_cv")

# Save plot
fig, ax = plt.subplots()
ax.plot(x, y)
manager.save_plot(fig, "roc_curve")  
# → results/20231203_120000_5fold_cv/plots/roc_curve.{png,pdf}

# Save table
df = pd.DataFrame(metrics)
manager.save_table(df, "results", formats=['csv', 'tex'])
# → results/.../tables/results.{csv,tex}

# Create summary
manager.create_summary()  # Lists all artifacts
```

**Directory Structure Created**:
```
results/{timestamp}_{experiment_name}/
├── plots/       # Figures (PNG + PDF)
├── tables/      # CSV + LaTeX tables
├── logs/        # Text logs
└── models/      # Saved models
```

---

### 2. `logger.py` - Enhanced Logging

Colored console logging with file output and tqdm integration.

**Features**:
- Colored output (via colorama)
- Dual output: Console + File
- tqdm progress bar integration
- Structured log levels
- Phase markers

**Usage**:

```python
from utils import EnhancedLogger

# Create logger
logger = EnhancedLogger(
    name="Experiment",
    log_file="results/exp.log"
)

# Logging
logger.header("Starting Experiment")
logger.phase("Data Loading")
logger.info("Processing...")
logger.success("Completed!")
logger.warning("Low memory")
logger.error("Failed to load")

# Metrics
logger.metric("Accuracy", 0.9756)

# Tables
logger.table({
    "Model": "HybridStackPPI",
    "Folds": 5,
})

# Progress
for i in logger.tqdm(range(100), desc="Training"):
    # Work here
    pass
```

**Log Levels with Colors**:
- `INFO`: Blue
- `SUCCESS`: Green
- `WARNING`: Yellow
- `ERROR`: Red
- `HEADER`: Magenta

---

## Installation

Already included in project. Dependencies:

```bash
pip install colorama  # For colored output (optional)
pip install tqdm      # For progress bars (optional)
```

---

## Integration with Existing Code

### Option 1: Replace Existing Logger

```python
# Before
from experiments.logger import PipelineLogger
logger = PipelineLogger()

# After
from utils import EnhancedLogger
logger = EnhancedLogger(name="Pipeline", log_file="results/pipe.log")
```

### Option 2: Use Alongside Existing

```python
# experiments/run.py
from experiments.logger import PipelineLogger  # Keep existing
from utils import ArtifactManager  # Add new

logger = PipelineLogger()
artifacts = ArtifactManager(experiment_name="cv_run")

# Use both
logger.header("Starting CV")
artifacts.save_plot(fig, "cv_curves")
```

---

## Examples

### Complete Experiment Flow

```python
from utils import ArtifactManager, EnhancedLogger

# Setup
logger = EnhancedLogger("Experiment", "results/exp.log")
manager = ArtifactManager(experiment_name="sota_comparison")

logger.header("SOTA Benchmarking Experiment")

# Data phase
logger.phase("Data Preparation")
train_ids, test_ids = get_sota_consistent_splits(fasta_path)
logger.success(f"Split data: {len(train_ids)} train, {len(test_ids)} test")

# Training phase
logger.phase("Model Training")
for fold in logger.tqdm(range(5), desc="Cross-Validation"):
    # Train model
    logger.info(f"Training fold {fold+1}")
    # Evaluate
    logger.metric(f"Fold {fold+1} Accuracy", accuracy)

# Save results
logger.phase("Saving Results")
manager.save_plot(roc_fig, "cv_roc_curves")
manager.save_table(metrics_df, "sota_comparison", formats=['csv', 'tex'])

logger.success("Experiment complete!")
logger.info(manager.create_summary())
```

---

## API Reference

### ArtifactManager

**`__init__(base_dir='results', experiment_name=None, timestamp=True)`**
- Initialize manager with optional timestamping

**`save_plot(fig, name, dpi=300, formats=['png', 'pdf'])`**
- Save matplotlib figure in multiple formats
- Returns dict of format → filepath

**`save_table(df, name, formats=['csv', 'tex'], latex_caption=None, latex_label=None)`**
- Save DataFrame as CSV and/or LaTeX
- Auto-generates LaTeX table code

**`save_text(content, name, subdir='logs')`**
- Save text content to file

**`save_model(model, name)`**
- Save sklearn/model object with joblib

**`create_summary()`**
- Generate summary of all artifacts

**`archive(archive_name=None)`**
- Create zip archive of results

---

### EnhancedLogger

**`__init__(name, log_file=None, level=logging.INFO, console=True)`**
- Initialize logger with optional file output

**`info(message)`, `warning(message)`, `error(message)`, `debug(message)`**
- Standard logging methods

**`success(message)`**
- Log success message (green)

**`header(message, char='=')`**
- Log decorated header

**`phase(message)`**
- Log experiment phase marker

**`metric(name, value, format_spec='.4f')`**
- Log metric with formatting

**`table(data, title=None)`**
- Log dict as formatted table

**`tqdm(iterable, desc='', **kwargs)`**
- Create tqdm progress bar (fallback to plain iteration if unavailable)

**`progress(current, total, prefix='Progress')`**
- Simple progress indicator

---

## Testing

Run example script:

```bash
python examples_utils.py
```

Expected output:
- Colored console logs
- File log in `results/example.log`
- Plots, tables in `results/{timestamp}_example_experiment/`

---

## Notes

- **Colorama**: Optional but recommended for colors
- **tqdm**: Optional but recommended for progress bars
- **Thread-safe**: Both logger and artifact manager are thread-safe
- **Timestamps**: Default format: `YYYYMMDD_HHMMSS`

---

## Support

For issues or questions, check:
- Example script: `examples_utils.py`
- Source code: `utils/artifact_manager.py`, `utils/logger.py`
