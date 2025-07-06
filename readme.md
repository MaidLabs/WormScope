# WormScope: C. elegans Survival Analysis Pipeline

A comprehensive, reproducible Python pipeline for analyzing *C. elegans* lifespan data, featuring:

- **Survival curves** (Kaplan–Meier)
- **Pairwise log-rank tests** with heatmap
- **Hazard function** and **cumulative hazard** visualizations
- **Cox proportional hazards** and **Weibull AFT** models
- **Median survival** with bootstrap confidence intervals
- **Summary statistics**, mean/max lifespan tables, and multi-panel summary figure

---

## Installation

1. Clone this repository or download the script.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python survival_analysis_pipeline.py --input path/to/data.txt --outdir path/to/output
```

- `--input`: Tab-delimited lifespan data file (default: `2.txt`).
- `--outdir`: Directory for saving figures and results (default: current directory).
- `--bootstrap-ci`: Include bootstrap confidence intervals for medians.

All output files (plots and text summaries) will be generated in the specified `outdir`.

## Project Structure

```text
.
├── survival_analysis_pipeline.py   # Main analysis script
├── requirements.txt                # Python dependencies
├── data/                           # Example input files
└── results/                        # Generated figures and tables
```

## Contributing

Feel free to open issues or submit pull requests for enhancements or bug fixes.

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

