# fNIRS_FullCap_OD
A Python pipeline for preprocessing and analyzing functional Near-Infrared Spectroscopy (fNIRS) data using Octamon & Brite devices (Artinis Medical Systems).

---

## Overview
This package processes OxySoft optical density (OD) `.txt` exports and takes them through to concentration data and batch-level statistics (OD -> concentration -> stats). Steps include:

- **Channel quality control (SQI, SCI, PSP)**
  Channels are scored with the scalp quality index (SQI), scalp coupling index (SCI), and peak spectral power (PSP). Each file is processed **once** under the exclusion criterion you select (`--exclusion-method`, e.g. `sci_psp`, or `none` for no exclusion). A cross-method exclusion comparison (SQI / SCI / PSP / SCI+PSP) is **always** written to the summary directory, whichever criterion actually ran, so you can revisit that choice without reprocessing.

Luca Pollonini, Heather Bortfeld, and John S. Oghalai, *"PHOEBE: a method for real time mapping of optodes-scalp coupling in functional near-infrared spectroscopy,"* Biomed. Opt. Express 7, 5104-5119 (2016)

  M. Sofía Sappia, Naser Hakimi, Willy N. J. M. Colier, and Jörn M. Horschig, *"Signal quality index: an algorithm for quantitative assessment of functional near infrared spectroscopy signal quality,"* Biomed. Opt. Express 11, 6732-6754 (2020)

- **Hard SQI floor**
  An optional floor that discards channels at or below a set SQI value regardless of the selected criterion. Adjust it with `--hard-sqi-discard` or turn it off with `--disable-hard-sqi`.
- **Gate A (cardiac SD gate)**
  An additional channel gate based on median cardiac SD. It is inert unless you supply dataset and/or per-channel median cardiac SD values (see [Activating Gate A](#activating-gate-a)).
- **Motion artifact correction (TDDR)**
  Fishburn, F.A., Ludlum, R.S., Vaidya, C.J., & Medvedev, A.V. (2019).
  *Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS.*
  NeuroImage, 184, 171-179. https://doi.org/10.1016/j.neuroimage.2018.09.025
- **Short-channel regression (SCR)** for superficial noise removal
- **Band-pass filtering** using a Butterworth filter
- **Region-averaged hemodynamic response** calculation across long channels (grouped by anatomical regions)
- **Visualizations and statistics** export for further analysis and interpretation

**Note:** This pipeline is highly tailored to our lab's walking tasks and file naming conventions.

---

## Requirements
Python 3.6 or higher

All dependencies are specified in `setup.py`. To install the package and all required libraries:
```bash
pip install -e .
```

---

## Usage

### Run from the command line:
```bash
python -m cli.main /path/to/input /path/to/output
```

### Examples
```bash
# Default run
python -m cli.main input/ output/

# Exclude channels by SCI + PSP
python -m cli.main input/ output/ --exclusion-method sci_psp

# Quality metrics only (skip full pipeline)
python -m cli.main input/ output/ --quality-only

# Full pipeline with no channel exclusion
python -m cli.main input/ output/ --exclusion-method none

# Change or disable the hard SQI floor
python -m cli.main input/ output/ --hard-sqi-discard 1.0
python -m cli.main input/ output/ --disable-hard-sqi
```

### Command-line options
| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing OxySoft OD `.txt` exports |
| `output_dir` | Directory for processed CSVs, plots, and summaries |
| `--exclusion-method` | Criterion used to exclude channels during processing (e.g. `sci_psp`, `none`) |
| `--quality-only` | Compute and report channel quality only; skip the full pipeline |
| `--hard-sqi-discard` | Discard channels at or below this SQI value |
| `--disable-hard-sqi` | Turn the hard SQI floor off |
| `--dataset-median-cardiac-sd` | Dataset-wide median cardiac SD, used to activate Gate A |
| `--channel-median-cardiac-sd-file` | CSV of per-channel median cardiac SD, used to activate Gate A |
| `--sqi-threshold`, `--sci-threshold`, `--psp-threshold`, `--combined-sci-threshold` | Thresholds used by the quality metrics and the cross-method comparison |
| `--plot-channels` | Plot channel quality |
| `--skip-plots` | Skip plot generation |
| `--skip-stats` | Skip batch-level statistics aggregation |

Defaults and the full list of pipeline options are defined in `add_pipeline_args` in `processing/batch.py`.

### Activating Gate A
Gate A needs `--dataset-median-cardiac-sd` and/or `--channel-median-cardiac-sd-file`; without them it's inactive and the pipeline behaves as if Gate A didn't exist. The channel file is not built by hand. It comes from a prior run's `cardiac_sd_by_channel.csv` summary sheet:

1. Run the pipeline once without the Gate A flags.
2. The statistics step writes `summary/cardiac_sd_by_channel.csv`.
3. Rerun with the flags pointing at that file:
   ```bash
   python -m cli.main input/ output/ \
       --dataset-median-cardiac-sd 0.004023 \
       --channel-median-cardiac-sd-file output/summary/cardiac_sd_by_channel.csv
   ```

### Outputs
- Processed CSVs and plots in `output_dir`
- Batch-level summary sheets in `output_dir/summary/`, including the cross-method exclusion comparison and `cardiac_sd_by_channel.csv`
