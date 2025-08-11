# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:28:19 2025

@author: amjordan7
"""

import pandas as pd
import numpy as np
from indexes import csc_model, hcs_model, kous_model


classic_path = 'scheil_data_classic.xlsx'
solute_trap_path = 'scheil_data_solute_trapping.xlsx'

classic_excel_file = pd.ExcelFile(classic_path)
classic_sheets = classic_excel_file.sheet_names

solute_trap_excel_file = pd.ExcelFile(solute_trap_path)
solute_trap_sheets = solute_trap_excel_file.sheet_names

calc_modes = {
    'classic': (classic_path, classic_sheets),
    'solute_trapping': (solute_trap_path, solute_trap_sheets)}

for mode_name, (file_path, sheets) in calc_modes.items():
    print(f"\n=== Processing {mode_name} mode ===")
    print(f"File path: {file_path}")
    
    mode_index_results = []
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
        
        alloy_indexes = [csc, hcs, csi, solidification_range]
        mode_index_results.append(alloy_indexes)
    print(mode_index_results)

