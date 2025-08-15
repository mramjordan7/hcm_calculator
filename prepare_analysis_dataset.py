# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 22:24:03 2025

@author: amjordan7
"""

import pandas as pd

def prepare_data_for_analysis():
    # Load and clean input file data
    df_load_input = pd.read_csv(r"../Alloy Master Crack Data.csv")
    df_input = df_load_input.copy()
    
    unecessary_columns = [
        'Ref #', 'Source', 'Max Crack Length (µm)', '# of Cracks', 'Area (mm^2)',
        'Process', 'P (W)','Scanning Velo (mm/s)', 'h(µm)', 't (µm)', 'E (J/mm^3)', 'Notes'
        ]
    df_input.drop(columns=unecessary_columns, inplace=True)
    
    # Load and clean processed data file
    df_load_processed_data = pd.read_csv('processed_data.csv')
    df_processed_data = df_load_processed_data.copy()
    
    processed_data = df_processed_data[['database', 'scan_speed_mps']]
    
    # Load alloy hcm results
    df_load_cl_hcm_results = pd.read_excel("alloy_index_results.xlsx", sheet_name='classic')
    cl_hcm_results = df_load_cl_hcm_results.copy()
    df_load_st_hcm_results = pd.read_excel('alloy_index_results.xlsx', sheet_name='solute_trapping')
    st_hcm_results = df_load_st_hcm_results.copy()
    
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
    output_filename = 'hcm_analysis_data.xlsx'
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_classic_combined.to_excel(writer, sheet_name='classic', index=False)
        df_solute_trap_combined.to_excel(writer, sheet_name='solute_trapping', index=False)
    
    print(f"Combined datasets exported to {output_filename}")
    print(f"Classic sheet shape: {df_classic_combined.shape}")
    print(f"Solute trapping sheet shape: {df_solute_trap_combined.shape}")
    print("Export complete!")
    
if __name__ == "__main__":
    prepare_data_for_analysis()
