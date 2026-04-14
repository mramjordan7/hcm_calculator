# HCM Calculator

A Python pipeline for calculating hot cracking susceptibility indexes from alloy composition data using ThermoCalc Scheil solidification simulations.

---

## Prerequisites

### ThermoCalc
ThermoCalc must be installed and licensed on the machine before the pipeline can run. The `tc_python` package is distributed with the ThermoCalc software installation and cannot be installed via pip.

1. Obtain a ThermoCalc license through your institution.
2. Install ThermoCalc following the vendor instructions for your operating system.
3. Verify `tc_python` is accessible:
```bash
python -c "import tc_python; print('tc_python OK')"
```
If this returns an error, `tc_python` is not on your Python path. Consult your ThermoCalc installation documentation.

### Python Dependencies
Python 3.7 or later is required. Install the required packages:
```bash
pip install pandas numpy scipy openpyxl
```

---

## Installation

```bash
git clone <repository_url>
cd hcm_calculator
```

Or download the ZIP from the GitHub repository page and extract it.

---

## Preparing Your Input File

### Create Your Alloy File
1. Open `Alloy Template.csv` included in the repository.
2. Make a copy. Do not modify the original template.
3. Replace the example rows with your alloy data.

> **All column headers must remain present even if cells are empty.** The pipeline references columns by name and will fail if any are missing or renamed.

### Alloy Family Values
The `Alloy Family` column controls which ThermoCalc database is assigned. Values must exactly match the strings below, including capitalization. Any other value silently falls back to `SSOL8` with no warning.

| Alloy Family | Database |
|---|---|
| `Aluminum` | TCAL9 |
| `Ni-Superalloy` | TCNI12 |
| `Ni alloy` | TCNI12 |
| `HEA` | TCHEA7 |
| `Steel` | TCFE13 |
| `Stainless Steel` | TCFE13 |
| `Titanium` | TCTI5 |
| `Ti-Al` | TCTI5 |
| `Fe-Ni-Cr` | TCFE13 |

### Scanning Velocity
The `Scanning Velo (mm/s)` column should be populated for every row. If left blank, the pipeline fills missing values with the mean scan speed of that alloy family group. If the entire alloy family group is blank, it defaults to 1000 mm/s. This value directly affects the solute trapping Scheil calculation.

### Crack Data Columns
The crack measurement columns (`Crack Y or N`, `Max Crack Length`, `Average Crack Length`, `# of Cracks`, `Area`, `Crack Density`) may be left empty if data is not available. They are dropped internally and do not affect calculations.

---

## Usage

Navigate to the project folder and run `main.py` using one of the following methods:

**Command-line argument (recommended):**
```bash
python main.py "path/to/your_alloy_file.csv"
```

**Interactive prompt:**
```bash
python main.py
```

### Pipeline Steps
The pipeline runs five sequential steps with progress printed to the terminal.

| Step | Description |
|---|---|
| 1 — Process input data | Reads your CSV, maps databases, prepares the working file |
| 2 — Load ThermoCalc databases | Queries each database for available elements |
| 3 — Run ThermoCalc calculations | Runs classic and solute trapping Scheil for each alloy as an isolated subprocess. This is the longest step |
| 4 — Calculate hot cracking indexes | Computes CSI, CSC, and HCS from the Scheil output |
| 5 — Prepare analysis dataset | Merges all results into a final Excel file |

---

## Outputs

All outputs are written to a timestamped subfolder inside `results/`, for example:
```
results/2026-04-08_14-32-01/
```

| File | Description |
|---|---|
| `classic_scheil/` | One CSV and one JSON metadata file per alloy for the classic Scheil calculation |
| `solute_trapping_scheil/` | One CSV and one JSON metadata file per alloy for the solute trapping calculation |
| `alloy_index_results.xlsx` | CSI, CSC, HCS, and solidification range for each alloy, one sheet per calculation mode |
| `hcm_analysis_data.xlsx` | Final merged dataset combining alloy compositions, processing parameters, and index results. Primary output for analysis |
| `overall_calculation_metadata.json` | Error log and run summary including failed alloys and error types |

---

## Troubleshooting

**`tc_python` import error**
ThermoCalc is not installed or `tc_python` is not on the Python path. Verify the ThermoCalc installation.

**`KeyError` on a column name**
A required column has been renamed or deleted from the input CSV. Restore the original column headers from the template.

**Alloy falls back to SSOL8 unexpectedly**
Check that the `Alloy Family` value in your CSV exactly matches one of the accepted strings in the table above.

**Individual alloy failures**
Check `overall_calculation_metadata.json` in the timestamped results folder for the specific error type and alloy name.
