# fNIRS_FullCap_2025
A Python pipeline for preprocessing and analyzing functional Near-Infrared Spectroscopy (fNIRS) data using Octamon devices (Artinis Medical Systems).

---

##  Overview
This package processes `.txt` fNIRS files exported from Octamon systems, using a two-pass pipeline that includes:
- **Scalp Quality Index filtering**  
  Channels are assessed using scalp quality index. Two pass processing: One with automatic channel discarding based on a customized SQI value (auto set to 2). SQI values are logged for user review based on user input of SQI cutoff.
- **Motion artifact correction (TDDR)**  
  Fishburn, F.A., Ludlum, R.S., Vaidya, C.J., & Medvedev, A.V. (2019).  
  *Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS.*  
  NeuroImage, 184, 171–179. https://doi.org/10.1016/j.neuroimage.2018.09.025
- **Short-channel regression (SCR)** for superficial noise removal
- **Band-pass filtering** using FIR filters
- **Region-averaged hemodynamic response** calculation across long channels (grouped by anatomical regions)
- **Visualizations and statistics** export for further analysis and interpretation

 **Note:** This pipeline is highly tailored to our lab's specific walking tasks and file naming conventions.  
If you're adapting this for your own dataset, you may need to modify logic within the `fnirs_FullCap_OD/processing/` module — particularly functions for file selection, event parsing, and batch processing.

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
python -m fnirs_FullCap_OD.main /path/to/input /path/to/output
```

### Command-line options
| Argument         | Description                                               | Default |
|------------------|-----------------------------------------------------------|---------|
| `input_dir`      | Directory containing raw `.txt` fNIRS files               | —       |
| `output_dir`     | Directory where processed files and figures will be saved | —       |
| `--fs`           | Sampling frequency in Hz                                  | 50.0    |
| `--sci_threshold`   | SCI (Scalp Coupling Index) threshold for logging purposes | 0.7     |
| `--log_level`    | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)   | INFO    |
