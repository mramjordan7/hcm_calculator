"""
ThermoCalc Scheil Solidification Calculator.

Performs individual alloy Scheil solidification calculations using ThermoCalc
Python API. Runs both classic and solute trapping calculation modes,
handles database selection, applies steep drop corrections, and exports
results to CSV files. Typically called as a subprocess by run_all_alloys.py
but can be run independently for single alloy analysis.
"""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from tc_python import (
        TCPython,
        CompositionUnit,
        ScheilCalculationType,
        ScheilQuantity,
        ScheilOptions
    )
except ImportError:
    raise ImportError(
        "tc_python must be installed separately via ThermoCalc licesne.")


class SolidificationTerminationError(ValueError):
    """Raised when solidification ends before reaching fs = 0.9."""

    pass


PROBLEM_ELEMENTS = {'H', 'B', 'N', 'O', 'P', 'S'}
# Ionic elements are those which form ionic liquid in Ni-superalloys
IONIC_ELEMENTS = {'O', 'P', 'S'}


def run_tcpython(cache_dir, input_file, results_dir=None):
    """Run Thermo-Calc Python session to calc Scheil solidification path."""
    # Load processed data
    df_load = pd.read_pickle(input_file)
    df = df_load.copy()

    # Create main cache directory and output folders
    main_cache_dir = Path('tc_cache')
    main_cache_dir.mkdir(exist_ok=True)

    if results_dir:
        results_path = Path(results_dir)
    else:
        # Fallback for direct execution
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_results_dir = Path('results')
        base_results_dir.mkdir(exist_ok=True)
        results_path = base_results_dir / timestamp
        results_path.mkdir(exist_ok=True)

    classic_dir = results_path / 'classic_scheil'
    solute_trapping_dir = results_path / 'solute_trapping_scheil'
    classic_dir.mkdir(exist_ok=True)
    solute_trapping_dir.mkdir(exist_ok=True)

    # Get database elements once with a temporary session
    print("Loading database information...")
    try:
        with open('working_files/databases_and_elements.pkl', 'rb') as f:
            databases_and_elements = pickle.load(f)
            print(f"Loaded database information for {len(databases_and_elements)} databases")
    except FileNotFoundError:
        print("Error: working_files/databases_and_elements.pkl not found. Run load_databases.py first.")
        return

    all_calculation_errors = []

    # Process each alloy in processed data with individual cache isolation
    for idx, row in df.iterrows():
        print("" + "="*70)
        print(f"\nProcessing alloy {idx + 1}/{len(df)}: {row.get('Alloy Name')}")

        # Create individual cache directory for single alloy
        alloy_cache_dir = main_cache_dir / f"alloy_{idx}_{clean_filename(row.get('Alloy Name', 'unknown'))}_{row.get('scan_speed_mps')}mps"
        alloy_cache_dir.mkdir(exist_ok=True)

        # Extract alloy information into a dictionary
        alloy_info_dict = {
            'index': row.name,
            'name': row['Alloy Name'],
            'initial_database': row['database'],
            'dependent_element': row['dependent_element'],
            'other_elements': row['other_elements'],
            'other_elements_wt_pct': row['other_elements_wt_pct'],
            'scan_speed': row['scan_speed_mps'],
            'alloy_family': row['Alloy Family']
        }

        # Process this alloy with its own isolated session and cache
        with TCPython() as isolated_session:
            isolated_session.set_cache_folder(str(alloy_cache_dir))

            alloy_errors = process_alloy(
                session=isolated_session,
                alloy_info=alloy_info_dict,
                databases_and_elements=databases_and_elements,
                classic_dir=classic_dir,
                solute_trapping_dir=solute_trapping_dir
            )

            if alloy_errors:
                all_calculation_errors.extend(alloy_errors)

    if all_calculation_errors:
        calc_errors_file = results_path / 'calculation_errors.json'
        with open(calc_errors_file, 'w') as f:
            json.dump(all_calculation_errors, f, indent=2)
        print(f"\n✓ Calculation errors saved to {calc_errors_file}")

    print("" + "="*70)
    print("\nCalculation complete.")
    print(f"Total calculation errors: {len(all_calculation_errors)}")
    print("" + "="*70)


def clean_filename(filename):
    """Clean filename for use in paths."""
    if not filename:
        return "unknown"

    # Remove bad characters
    bad_chars = ['/', '\\', ':', '*', '?', '[', ']', ' ', '(', ')', '+', '%',
                 '.', ',']
    cleaned = filename
    for char in bad_chars:
        cleaned = cleaned.replace(char, '_')

    # Remove extra underscores
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    cleaned = cleaned.strip('_')

    # Limit length
    if len(cleaned) > 20:
        cleaned = cleaned[:20]

    return cleaned if cleaned else "unknown"


def process_alloy(session, alloy_info, databases_and_elements, classic_dir,
                  solute_trapping_dir):
    """Process a single alloy with isolated cache and immediate CSV export."""
    # Select database
    alloy_elements = [alloy_info['dependent_element']] + alloy_info[
        'other_elements']
    initial_database = alloy_info['initial_database']

    # Check if the database has all the elements for the alloy
    if initial_database in databases_and_elements:
        available_elements = databases_and_elements[initial_database]
        if set(alloy_elements).issubset(available_elements):
            database = initial_database
        else:
            database = 'SSOL8'
            print(f"    Database switched to SSOL8 for {alloy_info['name']}")

    else:
        database = 'SSOL8'
        print(f"    Database switched to SSOL8 for {alloy_info['name']}")

    # Set calculation modes
    scan_speed = alloy_info['scan_speed']
    calculation_modes = {
        'classic': {
            'calc_type': ScheilCalculationType.scheil_classic(),
            'output_dir': classic_dir
        },
        'solute_trapping': {
            'calc_type': ScheilCalculationType.scheil_solute_trapping()
            .set_scanning_speed(scan_speed),
            'output_dir': solute_trapping_dir
        }
    }

    alloy_errors = []

    for mode_name, mode_config in calculation_modes.items():
        print("" + "-"*50)
        print(f"  Running {mode_name} calculation...")

        mode_result = attempt_calculation(
            session=session,
            alloy_info=alloy_info,
            database=database,
            calc_type=mode_config['calc_type'],
            mode_name=mode_name,
            retry_context=None  # No retry context for original attempt
        )

        if mode_result['success']:
            # Export immediately to CSV
            export_single_calculation_csv(
                alloy_info=alloy_info,
                calculation_data=mode_result['data'],
                mode_name=mode_name,
                output_dir=mode_config['output_dir']
            )
            print(f"    {mode_name} calculation successful and exported")
        else:
            print("    CALCULATION FAILED - Analyzing cause...")
            # Collect original errors WITHOUT modification
            original_errors = mode_result['errors']
            alloy_errors.extend(original_errors)

            if original_errors:
                error_details = original_errors[0]
                print(f"      Error Type: {error_details['error_type']}")
                print(f"      Error Message: {error_details['error_msg']}")
                print(f"      Database Used: {error_details.get('database', 'Unknown')}")

            # Check if we should retry with filtered elements
            contains_problem_elements = bool(set(alloy_info['other_elements'])
                                             & PROBLEM_ELEMENTS)
            problem_elements_found = set(alloy_info['other_elements']
                                         ) & PROBLEM_ELEMENTS

            print(f"      Contains Problem Elements: {contains_problem_elements}")
            if problem_elements_found:
                print(f"      Problem Elements: {problem_elements_found}")

            if contains_problem_elements:
                print("    RETRYING with filtered elements...")

                # Filter out problem elements
                filtered_elements = []
                filtered_wt_pcts = []

                for element, wt_pct in zip(alloy_info['other_elements'],
                                           alloy_info['other_elements_wt_pct'
                                                      ]):
                    if element not in PROBLEM_ELEMENTS:
                        filtered_elements.append(element)
                        filtered_wt_pcts.append(wt_pct)

                removed_elements = set(alloy_info['other_elements']) - set(
                    filtered_elements)
                print(f"    Removing problem elements: {removed_elements}")

                # Create filtered alloy info
                filtered_alloy = alloy_info.copy()
                filtered_alloy['other_elements'] = filtered_elements
                filtered_alloy['other_elements_wt_pct'] = filtered_wt_pcts

                retry_context = {
                    'retry_attempted': True,
                    'retry_successful': False,  # Will update this if successful
                    'elements_filtered': True,
                    'removed_elements': list(removed_elements),
                    'original_error_type': original_errors[0]['error_type']
                    if original_errors else 'Unknown',
                    'contains_problem_elements': True
                }

                # Retry calculation with context
                retry_result = attempt_calculation(
                    session=session,
                    alloy_info=filtered_alloy,
                    database=database,
                    calc_type=mode_config['calc_type'],
                    mode_name=mode_name,
                    retry_context=retry_context  # Pass retry context
                )

                if retry_result['success']:
                    retry_result['data']['metadata']['retry_attempted'] = True
                    retry_result['data']['metadata']['retry_successful'] = True
                    retry_result['data']['metadata']['elements_filtered'] = True

                    # Export retry result to CSV
                    export_single_calculation_csv(
                        alloy_info=alloy_info,  # Use original alloy info for naming
                        calculation_data=retry_result['data'],
                        mode_name=mode_name,
                        output_dir=mode_config['output_dir']
                    )

                    print(f"    Retry success: {mode_name} calculation and exported")
                else:
                    # Both attempts failed
                    if retry_result['errors']:
                        alloy_errors.extend(retry_result['errors'])
                    print("    ✗ Retry also failed")
                    # Update retry context for failed retry
                    retry_context['retry_successful'] = False

                    # Collect retry errors (they already have the retry context)
                    retry_errors = retry_result['errors']
                    alloy_errors.extend(retry_errors)
            else:
                # No problem elements, just failed
                mode_result['errors'][0]['retry_attempted'] = False
                mode_result['errors'][0]['contains_problem_elements'] = False
                print(f"    Failed: {mode_result['errors'][-1]['error_msg']}")
    if alloy_errors:
        print(f"\n  ⚠ Total errors for {alloy_info['name']}: {len(alloy_errors)}")

    # Return simple summary
    return alloy_errors


def export_single_calculation_csv(alloy_info, calculation_data, mode_name,
                                  output_dir):
    """Export a single calculation result to CSV file."""
    # Create filename
    alloy_name_clean = clean_filename(alloy_info['name'])
    scan_speed_str = f"{int(alloy_info['scan_speed'] * 1000)}mmps"
    filename = f"{alloy_info['index']}_{alloy_name_clean}_{scan_speed_str}_{mode_name}.csv"

    # Full path
    filepath = output_dir / filename

    # Export Scheil data
    scheil_df = calculation_data['scheil_data']
    scheil_df.to_csv(filepath, index=False)

    # Export metadata to separate file
    metadata_filename = f"{alloy_info['index']}_{alloy_name_clean}_{scan_speed_str}_{mode_name}_metadata.json"
    metadata_filepath = output_dir / metadata_filename

    with open(metadata_filepath, 'w') as f:
        json.dump(calculation_data['metadata'], f, indent=2)

    print(f"    Exported: {filename}")
    print(f"    Metadata: {metadata_filename}")


def attempt_calculation(session, alloy_info, database, calc_type, mode_name, retry_context=None):
    """Attempt a single calculation."""
    # Debug: Print alloy information
    print("    Alloy Info:")
    print(f"      Index: {alloy_info['index']}")
    print(f"      Name: {alloy_info['name']}")
    print(f"      Database: {database}")
    print(f"      Dependent Element: {alloy_info['dependent_element']}")
    print(f"      Other Elements: {alloy_info['other_elements']}")
    print(f"      Element Weights: {alloy_info['other_elements_wt_pct']}")
    print(f"      Scan Speed: {alloy_info['scan_speed']} m/s")

    try:
        alloy_elements = alloy_info['other_elements'] + [
            alloy_info['dependent_element']]
        print(f"    All Elements for TC: {alloy_elements}")

        # Create system and scheil calculation
        print("     Creating ThermoCalc system...")
        system = session.select_database_and_elements(
            database, alloy_elements).get_system()
        print("     System created successfully")

        print("    Setting up Scheil calculation...")
        scheil_calc = (system.with_scheil_calculation()
                       .set_composition_unit(CompositionUnit.MASS_PERCENT))
        print("    Scheil calculation setup complete")

        # Sets inonic liquid for Ni-superalloys with O, P, S
        ionic_elements_present = [el for el in alloy_elements
                                  if el in IONIC_ELEMENTS]
        if database.startswith('TCNI') and ionic_elements_present:
            options = ScheilOptions().set_liquid_phase('IONIC_LIQ')
            scheil_calc = scheil_calc.with_options(options)
            print(f"    Using IONIC_LIQ (ionic elements found: {ionic_elements_present})")

        else:
            print("    Using standard liquid phase")

        print("    Setting element compositions:")
        composition_set = {}
        for element, wt_pct in zip(alloy_info['other_elements'],
                                   alloy_info['other_elements_wt_pct']):
            scheil_calc.set_composition(element, wt_pct)
            composition_set[element] = wt_pct
            print(f"      {element}: {wt_pct} wt%")

        total_other = sum(alloy_info['other_elements_wt_pct'])
        dependent_pct = 100.0 - total_other
        composition_set[alloy_info['dependent_element']] = dependent_pct
        print(f"      {alloy_info['dependent_element']} (balance): {dependent_pct:.3f} wt%")
        print(f"   Total composition: {sum(composition_set.values()):.3f} wt%")

        print("    Running ThermoCalc calculation...")
        calculation = scheil_calc.with_calculation_type(calc_type).calculate()
        print("    ThermoCalc calculation completed successfully")

        def get_curve(quantity):
            return calculation.get_values_grouped_by_stable_phases_of(
                ScheilQuantity.mass_fraction_of_all_solid_phases(),
                quantity,
                sort_and_merge=False)
        # Extract data
        temp_curve = get_curve(ScheilQuantity.temperature())
        latent_heat_curve = get_curve(ScheilQuantity.latent_heat_per_gram())
        heat_capacity_curve = get_curve(ScheilQuantity
                                        .apparent_heat_capacity_per_gram())

        # Iterate through each phase and extract data
        data_blocks = []
        all_phase_names = []
        all_phase_steepness = {}
        all_steep_flags = []
        all_steepness_values = []
        steep_phases = []
        drop_detected = False

        for phase_idx, phase in enumerate(temp_curve):
            fs = np.asarray(temp_curve[phase].x)
            if fs.size == 0:
                continue

            temp_c = np.asarray(temp_curve[phase].y) - 273.15
            latent_heat = np.asarray(latent_heat_curve[phase].y)  # J/g
            heat_capacity = np.asarray(heat_capacity_curve[phase].y)  # J/gK

            drop_detected, phase_steepness = detect_steep_drop(phase, fs,
                                                               temp_c)
            all_phase_steepness[(phase)] = phase_steepness

            if drop_detected:
                steep_phases.append((phase, phase_steepness))
                print(f"    Steep phase detected: {phase}, steepness = {phase_steepness:.1f}")

    # Clean phase curves, skip first point of each phase, and create data block
            if len(fs) > 1:
                min_length = min(len(fs[1:]), len(temp_c[1:]),
                                 len(latent_heat[1:]), len(heat_capacity[1:]))

                fs_truncated = fs[1:1+min_length]
                temp_truncated = temp_c[1:1+min_length]
                latent_truncated = latent_heat[1:1+min_length]
                capacity_truncated = heat_capacity[1:1+min_length]

                block = np.column_stack((fs_truncated, temp_truncated,
                                        latent_truncated, capacity_truncated))

                valid_rows = ~np.isnan(block).any(axis=1)
                block = block[valid_rows]

                if len(block) > 0:
                    data_blocks.append(block)

                    # Add phase info for each valid row
                    all_phase_names.extend([phase] * len(block))

                    # Check if this phase has steep drop
                    is_steep_phase = any(sp == phase for sp, _ in steep_phases)
                    phase_steepness = all_phase_steepness.get(phase, 0)

                    all_steep_flags.extend([is_steep_phase] * len(block))
                    all_steepness_values.extend([phase_steepness] * len(block))
        if not data_blocks:
            return {}

        # Combine all data
        combined_data = np.vstack(data_blocks)

        # Apply same deduplication to all arrays
        _, unique_idx = np.unique(combined_data, axis=0, return_index=True)
        unique_idx_sorted = np.sort(unique_idx)

        combined_data = combined_data[unique_idx_sorted]
        final_phase_names = [all_phase_names[i] for i in unique_idx_sorted]
        final_steep_flags = [all_steep_flags[i] for i in unique_idx_sorted]
        final_steepness_values = [all_steepness_values[i]
                                  for i in unique_idx_sorted]

        # Sort by fraction solid ascending, then temperature descending
        sort_idx = np.lexsort((-combined_data[:, 1], combined_data[:, 0]))
        combined_data = combined_data[sort_idx]
        final_phase_names = [final_phase_names[i] for i in sort_idx]
        final_steep_flags = [final_steep_flags[i] for i in sort_idx]
        final_steepness_values = [final_steepness_values[i] for i in sort_idx]

        if combined_data[-1, 0] < 0.9:
            raise SolidificationTerminationError(
                "Solidification terminated prematurely (fs < 0.9)")

        fraction_solid = combined_data[:, 0]
        temperature = combined_data[:, 1]
        latent_heat = combined_data[:, 2]
        heat_capacity = combined_data[:, 3]

        # Handle steep drop correction
        first_steep_start = None
        first_steep_end = None

        # Create a working copy starting from index 1 (skip first datapoint)
        working_steep_flags = final_steep_flags[1:].copy()

        # Bridge single-point normal regions (length=1 False between True values)
        for i in range(1, len(working_steep_flags) - 1):
            if (working_steep_flags[i-1] == True and
                working_steep_flags[i] == False and
                working_steep_flags[i+1] == True):
                working_steep_flags[i] = True

        # Find the start of the first steep region
        first_steep_start = None
        first_steep_end = None

        for i, is_steep in enumerate(working_steep_flags):
            if is_steep:
                first_steep_start = i + 1  # Add 1 because we skipped the first datapoint
                break

        # If we found a steep region, find where it ends
        if first_steep_start is not None:
            for i in range(first_steep_start - 1, len(working_steep_flags)):  # Adjust for offset
                if not working_steep_flags[i]:  # Found first False after steep region
                    first_steep_end = i + 1  # Add 1 because we skipped the first datapoint
                    break

        # Handle steep drop correction
        initial_temp_original = temperature[0]
        initial_temp_corrected = temperature[0]
        correction_index = 0

        if first_steep_start is not None and first_steep_end is not None:
            initial_temp_corrected = temperature[first_steep_end]
            correction_index = first_steep_end
            print(f"    Steepness correction: {initial_temp_original:.1f}°C ' {initial_temp_corrected:.1f}°C")
            print(f"    First steep region: indices {first_steep_start} to {first_steep_end-1}")
            print(f"    Using temperature at index {first_steep_end}: {initial_temp_corrected:.1f}°C")
            print("" + "-"*50)
        else:
            print(f"    No correction applied - steep_start: {first_steep_start}, steep_end: {first_steep_end}")

        scheil_df = pd.DataFrame({
            'Fraction_Solid': fraction_solid,
            'Temperature_C': temperature,
            'Specific_Latent_Heat_J_per_g': latent_heat,
            'Specific_Heat_Capacity_J_per_gK': heat_capacity,
            'Phase': final_phase_names,
            'Steep_Drop_Phase': final_steep_flags,
            'Steepness': final_steepness_values
        })
        metadata = {
            'database_used': database,
            'steep_drop_detected': len(steep_phases) > 0,
            'steep_phases': steep_phases,
            'initial_temp_original': initial_temp_original,
            'initial_temp_corrected': initial_temp_corrected,
            'correction_index': correction_index,
            'alloy_index': alloy_info['index'],
            'alloy_name': alloy_info['name'],
            'scan_speed_mps': alloy_info['scan_speed'],
            'calculation_mode': mode_name,
            'contains_problem_elements': bool(set(alloy_info['other_elements'])
                                              & PROBLEM_ELEMENTS),
            'retry_attempted': False,
            'retry_successful': False,
            'elements_filtered': False
        }
        return {
            'success': True,
            'data': {'scheil_data': scheil_df, 'metadata': metadata},
            'errors': []
        }
    except Exception as e:
        print(f"    ✗ {mode_name} calculation failed: {type(e).__name__}: {e}")
        error_info = {
            'alloy_index': alloy_info['index'],
            'alloy_name': alloy_info['name'],
            'calc_type': mode_name,
            'database': database,
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'timestamp': datetime.now().isoformat()
        }
        # Add retry context if this is a retry attempt
        if retry_context:
            error_info.update(retry_context)
        else:
            # Mark as original attempt
            error_info['retry_attempted'] = False
            error_info['contains_problem_elements'] = bool(set(alloy_info['other_elements']) & PROBLEM_ELEMENTS)

        return {
            'success': False,
            'data': None,
            'errors': [error_info]
        }


def detect_steep_drop (phase, fs, temp_c, threshold=2000, fs_limit=0.1):
    """Detect steep drop in temperature for a given phase."""
    if fs.size > 1:
        delta_fs = fs[-1] - fs[0]
        delta_temp = temp_c[-1] - temp_c[0]
        steepness = np.abs(delta_temp / delta_fs) if delta_fs != 0 else 0

        is_steep = (max(fs) <= fs_limit) and (steepness >= threshold)
        if is_steep:
            print("" + "-"*50)
            print(f"    Steep drop detected in phase {phase}:"
                  f" steepness = {steepness:.1f}")

        return is_steep, steepness
    return False, 0


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'working_files/processed_data.pkl'
    results_dir = sys.argv[2] if len(sys.argv) > 2 else None
    cache_dir = Path.cwd() / "tc_cache"
    run_tcpython(cache_dir, input_file, results_dir)
