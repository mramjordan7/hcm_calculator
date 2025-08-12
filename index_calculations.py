# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:28:19 2025

@author: amjordan7
"""

import pandas as pd
import numpy as np
from indexes import csc_model, hcs_model, kous_model

def calculate_indexes():
    classic_path = 'scheil_data_classic.xlsx'
    solute_trap_path = 'scheil_data_solute_trapping.xlsx'
    
    classic_excel_file = pd.ExcelFile(classic_path)
    classic_sheets = classic_excel_file.sheet_names
    
    solute_trap_excel_file = pd.ExcelFile(solute_trap_path)
    solute_trap_sheets = solute_trap_excel_file.sheet_names
    
    calc_modes = {
        'classic': (classic_path, classic_sheets),
        'solute_trapping': (solute_trap_path, solute_trap_sheets)}
    
    all_results = {}
    for mode_name, (file_path, sheets) in calc_modes.items():
        print(f"\n=== Processing {mode_name} mode ===")
        print(f"File path: {file_path}")
        
        mode_results = {
            'Sheet_Name': [],
            'CSC': [],
            'HCS': [],
            'CSI': [],
            'Solidification_Range': []
        }
        for sheet_name in sheets:
            print(f"\n--- Processing sheet: {sheet_name} ---")
    
            df_load = pd.read_excel(file_path, sheet_name=sheet_name)
            df = df_load.copy()
            
            fs = np.asarray(df.iloc[:, 0])
            temp = np.asarray(df.iloc[:, 1])
            latent_heat = np.asarray(df.iloc[:, 2])
            heat_capacity = np.asarray(df.iloc[:, 3])
            
            temp_i = float(df.iloc[4, 9])
            solidification_range = temp_i - temp[-1]
            
            csi = kous_model(fs, temp)
            csc = csc_model(fs, temp, latent_heat, heat_capacity)
            hcs = hcs_model(csc, solidification_range)
            
            # Store the results
            mode_results['Sheet_Name'].append(sheet_name)
            mode_results['CSC'].append(csc)
            mode_results['HCS'].append(hcs)
            mode_results['CSI'].append(csi)
            mode_results['Solidification_Range'].append(solidification_range)
        
        all_results[mode_name] = pd.DataFrame(mode_results)
    
    output_filename = 'alloy_index_results.xlsx'
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for mode_name, results_df in all_results.items():
            results_df.to_excel(writer, sheet_name=mode_name, index=False)
    
    print(f"\nResults exported to {output_filename}")
    print("Sheet structure:")
    for mode_name, results_df in all_results.items():
        print(f"  {mode_name}: {results_df.shape[0]} rows, {results_df.shape[1]} columns")
        print(f"    Columns: {list(results_df.columns)}")

if __name__ == "__main__":
    calculate_indexes()
