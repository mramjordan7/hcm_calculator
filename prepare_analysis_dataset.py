"""
Analysis Dataset Preparation Tool.

Combines original alloy data, processing metadata, and calculated hot cracking indexes 
into unified datasets ready for statistical analysis. Merges input alloy compositions, 
processing parameters, and HCM results from both classic and solute trapping calculations. 
Exports combined datasets to Excel format for machine learning and correlation analysis.
"""
import pandas as pd
from pathlib import Path


def prepare_data_for_analysis():
    """Prepare combined analysis dataset from latest calculation results."""
    # Find the most recent timestamped results directory
    base_results_dir = Path('results')
    if not base_results_dir.exists():
        print("Error: No results directory found")
        return

    # Get most recent timestamped directory
    timestamped_dirs = [d for d in base_results_dir.iterdir() if d.is_dir()]
    if not timestamped_dirs:
        print("Error: No timestamped result directories found")
        return

    latest_results_dir = max(timestamped_dirs, key=lambda x: x.name)
    print(f"Using results from: {latest_results_dir}")
    # Load and clean input file data
    df_load_input = pd.read_csv(r"../Alloy Master Crack Data.csv")
    df_input = df_load_input.copy()

    unecessary_columns = [
        'Ref #', 'Source', 'Max Crack Length (µm)', '# of Cracks',
        'Area (mm^2)', 'Process', 'P (W)', 'Scanning Velo (mm/s)', 'h(µm)',
        't (µm)', 'E (J/mm^3)', 'Notes'
        ]
    df_input.drop(columns=unecessary_columns, inplace=True)

    # Load and clean processed data file from working_files
    working_files_dir = Path('working_files')
    processed_data_file = working_files_dir / 'processed_data.pkl'

    if not processed_data_file.exists():
        print(f"Error: {processed_data_file} not found")
        return

    # Load and clean processed data file
    df_load_processed_data = pd.read_pickle(processed_data_file)
    df_processed_data = df_load_processed_data.copy()

    processed_data = df_processed_data[['database', 'scan_speed_mps']]

    # Load alloy HCM results from latest timestamped directory
    hcm_results_file = latest_results_dir / 'alloy_index_results.xlsx'

    if not hcm_results_file.exists():
        print(f"Error: {hcm_results_file} not found")
        print("Run index_calculations.py first to generate HCM results")
        return

    print(f"Loading HCM results from: {hcm_results_file}")

    df_load_cl_hcm_results = pd.read_excel(hcm_results_file, sheet_name='classic')
    cl_hcm_results = df_load_cl_hcm_results.copy()
    df_load_st_hcm_results = pd.read_excel(hcm_results_file, sheet_name='solute_trapping')
    st_hcm_results = df_load_st_hcm_results.copy()

    # Combine classic data
    classic_data = [df_input, processed_data, cl_hcm_results]
    df_classic_combined = pd.concat(classic_data, axis=1)
    print("=== CREATING SOLUTE TRAPPING DATASET ===")
    print(f"Classic combined shape: {df_classic_combined.shape}")
    print(f"Classic columns: {list(df_classic_combined.columns)}")

    hcm_columns_to_remove = ['CSC', 'HCS', 'CSI', 'Solidification_Range']
    df_base = df_classic_combined.drop(columns=hcm_columns_to_remove)

    st_hcm_for_merge = st_hcm_results[['Sheet_Name', 'CSC', 'HCS', 'CSI', 'Solidification_Range']].copy()

    # Merge base with solute trapping HCM results
    df_solute_trap_combined = pd.merge(
        df_base,
        st_hcm_for_merge,
        on='Sheet_Name',
        how='left',  # Keep all rows from base, fill NaN where solute data missing
        suffixes=('', '_solute')
    )

    # Export combined datasets to Excel
    output_filename = latest_results_dir / 'hcm_analysis_data.xlsx'

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_classic_combined.to_excel(writer, sheet_name='classic', index=False)
        df_solute_trap_combined.to_excel(writer, sheet_name='solute_trapping', index=False)

    print(f"Combined datasets exported to {output_filename}")
    print(f"Classic sheet shape: {df_classic_combined.shape}")
    print(f"Solute trapping sheet shape: {df_solute_trap_combined.shape}")
    print("Export complete!")


if __name__ == "__main__":
    prepare_data_for_analysis()
