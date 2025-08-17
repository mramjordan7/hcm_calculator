from pathlib import Path

import importlib.util

# Check if package is installed
if importlib.util.find_spec("tc_python") is None:
    raise ImportError(
        "tc_python must be installed separately via ThermoCalc license. "
        "Please contact ThermoCalc for licensing information."
    )

from process_data import process_data
from run_tcpython import run_tcpython
from index_calculations import calculate_indexes
from prepare_analysis_dataset import prepare_data_for_analysis


# Process input data
print("\n" + "="*50)
print("Processing input data...")
input_file = "../Alloy Master Crack Data.csv"
print(f"---Using the input file: '{input_file}' ---")
print("\n" + "="*50)
process_data(input_file=input_file)

# Run Scheil calculations
print("\n" + "="*50)
print("Starting ThermoCalc session...")
print("\n" + "="*50)
cache_dir = Path.cwd() / "tc_cache"
run_tcpython(cache_dir)

# Calculate hot cracking indexes from each alloys Scheil curve
print("\n" + "="*50)
print("Calculation hot cracking model indexes...")
print("\n" + "="*50)
calculate_indexes()

# Prepare final analysis dataset
print("\n" + "="*50)
print("Preparing analysis dataset...")
print("\n" + "="*50)
prepare_data_for_analysis()
