# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 22:24:03 2025

@author: amjordan7
"""

import pandas as pd

def prepare_data_for_analysis():
    # Load and clean input file data
    df_load_input = pd.read_excel(r"../Alloy Master Crack Data.xlsx")
    df_input = df_load_input.copy()
    
    unecessary_columns = [
        'Ref #', 'Source', 'Max Crack Length (µm)', '# of Cracks', 'Area (mm^2)',
        'Process', 'P (W)','Scanning Velo (mm/s)', 'h(µm)', 't (µm)', 'E (J/mm^3)', 'Notes'
        ]
    df_input.drop(columns=unecessary_columns, inplace=True)
    
    # Load and clean processed data file
    df_load_processed_data = pd.read_excel('processed_data.xlsx')
    df_processed_data = df_load_processed_data.copy()
    
    processed_data = df_processed_data[['database', 'scan_speed_mps']]
    
    # Load alloy hcm results
    df_load_cl_hcm_results = pd.read_excel("alloy_index_results.xlsx", sheet_name='classic')
    df_cl_hcm_results = df_load_cl_hcm_results.copy()
    df_load_st_hcm_results = pd.read_excel('alloy_index_results.xlsx', sheet_name='solute_trapping')
    df_st_hcm_results = df_load_st_hcm_results.copy()
    
    index_columns = ['CSC', 'HCS', 'CSI', 'Solidification_Range']
    cl_hcm_results = df_cl_hcm_results[index_columns]
    st_hcm_results = df_st_hcm_results[index_columns]
    
    classic_data = [df_input, processed_data, cl_hcm_results]
    solute_trap_data = [df_input, processed_data, st_hcm_results]
    
    df_classic_combined = pd.concat(classic_data, axis=1)
    df_solute_trap_combined = pd.concat(solute_trap_data, axis=1)

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
