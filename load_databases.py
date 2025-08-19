# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 11:26:01 2025

@author: amjordan7
"""
import pandas as pd
import pickle
from pathlib import Path

try:
    from tc_python import TCPython
    
except ImportError:
    raise ImportError(
        "tc_python must be installed separately via ThermoCalc licesne.")


def load_databases_and_elements(input_file='processed_data.pkl'):
    # Load processed data
    df_load = pd.read_pickle(input_file)
    df = df_load.copy()
    
    # Create main cache directory and output folders
    main_cache_dir = Path('tc_cache')
    main_cache_dir.mkdir(exist_ok=True)
    
    # Get database elements once with a temporary session
    print("Loading database information...")
    with TCPython() as temp_session:
        temp_session.set_cache_folder(str(main_cache_dir / 'temp'))
        
        # Get unique databasses
        database_list = df['database'].unique().tolist() + ['SSOL8'] # SSOL8 is a general database
        databases_and_elements = {}
        
        # For each database in the database list, identifies the elements in each database
        for database in database_list:
            print(f"  Loading {database}...")

            builder = temp_session.select_database_and_elements(database, ['Al'])
            system = builder.get_system()
            databases_and_elements[database] = set(system.get_all_elements_in_databases())
            print(f"    Found {len(databases_and_elements[database])} elements")
    
    print(f"Loaded {len([db for db, elements in databases_and_elements.items() if elements])} databases")
    
    # Export to pickle file
    export_file = 'databases_and_elements.pkl'
    with open(export_file, 'wb') as f:
        pickle.dump(databases_and_elements, f)
    print(f"Databases and elements exported to '{export_file}'")

if __name__ == "__main__":
    load_databases_and_elements()