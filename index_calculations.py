# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:28:19 2025

@author: amjordan7
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from indexes import csc_model, hcs_model, kous_model

def calculate_indexes():
    """Calculate hot cracking indexes from individual CSV Scheil calculation files."""
    
    # Define paths to the new directory structure
    classic_dir = Path('scheil_results/classic')
    solute_trap_dir = Path('scheil_results/solute_trapping')
    
    calc_modes = {
        'classic': classic_dir,
        'solute_trapping': solute_trap_dir}
    
    all_results = {}
    for mode_name, data_dir in calc_modes.items():
        print(f"\n=== Processing {mode_name} mode ===")
        print(f"Directory: {data_dir}")
        
        mode_results = {
            'Sheet_Name': [],
            'CSC': [],
            'HCS': [],
            'CSI': [],
            'Solidification_Range': []
        }
        # Find all CSV files in the directory (exclude metadata files)
        csv_files = list(data_dir.glob(f'*_{mode_name}.csv'))
        print(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            print(f"\n--- Processing file: {csv_file.name} ---")
    
            df_load = pd.read_csv(csv_file)
            df = df_load.copy()
            df = df[['Fraction_Solid', 'Temperature_C','Specific_Latent_Heat_J_per_g','Specific_Heat_Capacity_J_per_gK']].dropna()
            
            # Extract data arrays
            fs = np.asarray(df['Fraction_Solid'])
            temp = np.asarray(df['Temperature_C'])
            latent_heat = np.asarray(df['Specific_Latent_Heat_J_per_g'])
            heat_capacity = np.asarray(df['Specific_Heat_Capacity_J_per_gK'])
            
            # Load metadata to get corrected initial temperature
            metadata_file = csv_file.with_name(csv_file.stem + '_metadata.json')
            
            temp_i = temp[0]
            with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
            temp_i = metadata.get('initial_temp_corrected', temp[0])
            solidification_range = temp_i - temp[-1]
            
            print(f"    Solidification range: {solidification_range:.1f}°C")
            print(f"    Temperature range: {temp_i:.1f}°C to {temp[-1]:.1f}°C")
            print(f"    Fraction solid range: {fs[0]:.3f} to {fs[-1]:.3f}")
            
            print("    Calculating hot cracking indexes...")
            csi = kous_model(fs, temp)
            csc = csc_model(fs, temp, latent_heat, heat_capacity)
            hcs = hcs_model(csc, solidification_range)
            
            print(f"    CSI = {csi:.4f}")
            print(f"    CSC = {csc:.4f}")
            print(f"    HCS = {hcs:.4f}")
            
            # Create sheet name from filename (remove the mode suffix)
            sheet_name = csv_file.stem.replace(f'_{mode_name}', '')
            
            # Store the results
            mode_results['Sheet_Name'].append(sheet_name)
            mode_results['CSC'].append(csc)
            mode_results['HCS'].append(hcs)
            mode_results['CSI'].append(csi)
            mode_results['Solidification_Range'].append(solidification_range)
            
            print(f"    ✓ Successfully processed {sheet_name}")

        results_df = pd.DataFrame(mode_results)
        def extract_index(sheet_name):
            return int(sheet_name.split('_')[0])
        results_df['_sort_index'] = results_df['Sheet_Name'].apply(extract_index)
        results_df = results_df.sort_values('_sort_index').drop('_sort_index', axis=1).reset_index(drop=True)
    
        all_results[mode_name] = results_df
        
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
