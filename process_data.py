# map alloy families to databases
# identify dependent elements
# indentify other elements
# Scen speed to m/s

import pandas as pd

def process_data(input_file='Alloy Subset Crack Data.xlsx', output_file='processed_data.xlsx'):
    """
    Process alloy composition data from an Excel file.
    """
    # Load data file and make df copy
    df_load = pd.read_excel(input_file)
    df = df_load.copy()

    # Clean up copied df
    unecessary_columns = [
        'Ref #', 'Reference', 'Source', 'Particle Additions',
        'Amount (wt%)','Crack Y or N (1 or 0)',
        'Max Crack Length (µm)', 'Average Crack Length (µm)',
        '# of Cracks', 'Area (mm^2)', 'Crack Density (mm^-1)',
        'Process','P (W)', 'h(µm)', 't (µm)', 'E (J/mm^3)', 'Notes'
        ]
    df.drop(columns=unecessary_columns, inplace=True)

    # Create a new column for the database mapping
    database_map = {
            'Aluminum':       'TCAL9',
            'Ni-Superalloy':  'TCNI12',
            'Ni alloy':       'TCNI12',
            'HEA':            'TCHEA7',
            'Steel':          'TCFE13',
            'Stainless Steel': 'TCFE13',
            'Titanium':       'TCTI5'
        }
    df['database'] = df['Alloy Family'].map(database_map).fillna('SSOL8')

    # Identify element columns, convert to numeric, fill NaNs with 0
    element_columns = [
            c for c in df.columns
            if c.isalpha() and 1 <= len(c) <= 2]
    for col in element_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[element_columns] = df[element_columns].fillna(0)

    # Identify dependent element
    df['dependent_element'] = df[element_columns].idxmax(axis=1)

    # Identify other elements and the amount in wt%
    df['other_elements'] = df.apply(
        lambda row: [col for col in element_columns
                    if col != row['dependent_element'] and row[col] > 0],
        axis=1
    )

    df['other_elements_wt_pct'] = df.apply(
        lambda row: [float(row[elem]) for elem in row['other_elements'] if elem in df.columns],
        axis=1
    )

    # Identify scan speeds and convert to m/s
    df['scan_speed_mps'] = df['Scanning Velo (mm/s)'] / 1000.0
    df['scan_speed_mps'] = (
            df.groupby('Alloy Family')['scan_speed_mps']
            .transform(lambda x: x.fillna(x.mean()))
            .fillna(1.0)
    )

    # Drop element columns and 'Scanning Velo (mm/s)'
    df.drop(columns=element_columns + ['Scanning Velo (mm/s)'], inplace=True)
    df.to_excel('processed_data.xlsx', index=False)
    print("Processed data saved to 'processed_data.xlsx'")

if __name__ == "__main__":
    process_data(input_file=r"..\Alloy Master Crack Data.xlsx")