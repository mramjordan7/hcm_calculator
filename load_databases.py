"""
ThermoCalc Database Loader for Hot Cracking Research Tool.

This script loads ThermoCalc databases and extracts available elements for
each database. It creates a mapping that allows the calculation scripts to
automatically select appropriate databases based on alloy compositions and
switch to fallback databases when needed.

"""
import pandas as pd
import pickle
from pathlib import Path

try:
    from tc_python import TCPython

except ImportError:
    raise ImportError(
        "tc_python must be installed separately via ThermoCalc licesne.")


def load_databases_and_elements(input_file='working_files/processed_data.pkl'):
    """Load Thermo-Calc databases and their elements."""
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

        # Get unique databasses (SSOL8 is a general database)
        database_list = df['database'].unique().tolist() + ['SSOL8']
        databases_and_elements = {}

        # For each database in the list dentifies the elements in each database
        for database in database_list:
            print(f"  Loading {database}...")

            builder = temp_session.select_database_and_elements(database,
                                                                ['Al'])
            system = builder.get_system()
            databases_and_elements[database] = set(
                system.get_all_elements_in_databases())
            print(f"   Found {len(databases_and_elements[database])} elements")

    print(f"Loaded {len([db for db, elements in databases_and_elements.items() if elements])} databases")

    # Export to pickle file
    working_dir = Path('working_files')
    working_dir.mkdir(exist_ok=True)

    export_file = working_dir / 'databases_and_elements.pkl'
    with open(export_file, 'wb') as f:
        pickle.dump(databases_and_elements, f)
    print(f"Databases and elements exported to '{export_file}'")


if __name__ == "__main__":
    load_databases_and_elements()
