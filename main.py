"""
Hot Cracking Analysis Pipeline - Main Orchestration Script.

Runs the complete hot cracking analysis workflow from raw alloy data to final
analysis datasets. Executes all pipeline steps in sequence with error handling
and progress reporting.
"""

import importlib.util

# Check if package is installed
if importlib.util.find_spec("tc_python") is None:
    raise ImportError(
        "tc_python must be installed separately via ThermoCalc license. "
        "Please contact ThermoCalc for licensing information."
    )

from process_data import process_data
from load_databases import load_databases_and_elements
from run_all_alloys import run_all_alloys
from index_calculations import calculate_indexes
from prepare_analysis_dataset import prepare_data_for_analysis


def main():
    """Run complete hot cracking analysis pipeline."""
    # Step 1: Process input data
    print("\n" + "="*50)
    print("Step 1: Processing input data...")
    input_file = "../Alloy Master Crack Data.csv"
    print(f"Using input file: '{input_file}'")
    print("="*50)
    process_data(input_file=input_file)

    # Step 2: Load ThermoCalc databases
    print("\n" + "="*50)
    print("Step 2: Loading ThermoCalc databases...")
    print("="*50)
    load_databases_and_elements()

    # Step 3: Run Scheil calculations for all alloys
    print("\n" + "="*50)
    print("Step 3: Running ThermoCalc calculations...")
    print("="*50)
    run_all_alloys()

    # Step 4: Calculate hot cracking indexes
    print("\n" + "="*50)
    print("Step 4: Calculating hot cracking indexes...")
    print("="*50)
    calculate_indexes()

    # Step 5: Prepare final analysis dataset
    print("\n" + "="*50)
    print("Step 5: Preparing analysis dataset...")
    print("="*50)
    prepare_data_for_analysis()

    print("\n" + "="*50)
    print("HOT CRACKING ANALYSIS PIPELINE COMPLETE")
    print("="*50)
    print("Check the latest timestamped folder in 'results/' for outputs")


if __name__ == "__main__":
    main()
