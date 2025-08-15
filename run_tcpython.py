import json
import ast
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

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
IONIC_ELEMENTS = {'O', 'P', 'S'}

def run_tcpython(cache_dir):
    """Run ThermoCalc Python session to calculate Scheil solidification path for all alloys."""
    # Load processed data
    df_load = pd.read_pickle('processed_data.pkl')
    df = df_load.copy()
    
    # Create tc_cache directory
    cache_dir = Path('tc_cache')
    cache_dir.mkdir(exist_ok=True)

    # Get database elements once (this can be done with a temporary session)
    print("Loading database information...")
    with TCPython() as temp_session:
        temp_session.set_cache_folder(str(cache_dir))
        databases = df['database'].unique().tolist() + ['SSOL8']
        databases_and_elements = {}
        for db in databases:
            builder = temp_session.select_database_and_elements(db, ['Al'])
            system = builder.get_system()
            databases_and_elements[db] = set(system.get_all_elements_in_databases())

    all_scheil_data = {}
    all_errors = []
    run_summary = {
        'start_time': datetime.now().isoformat(),
        'total_alloys': len(df),
        'successful_calculations': 0,
        'failed_calculations': 0,
        'databases_used': list(databases_and_elements.keys()),
        'database_switches': 0,
        'ionic_liq_used': 0,
        'steep_drops_detected': 0,
        'retry_successes': 0
    }

    # Process each alloy
    for idx, row in df.iterrows():
        print("" + "="*70)
        print(f"\nProcessing alloy {idx + 1}/{len(df)}: {row.get('Alloy Name')}")
        # Extract alloy information
        
        with TCPython() as fresh_session:
            fresh_session.set_cache_folder(str(cache_dir))
            
            alloy_results, alloy_errors = process_single_alloy(
                session=fresh_session,
                row=row,
                databases_and_elements=databases_and_elements,
                run_summary=run_summary
            )
        

        if alloy_results:
            all_scheil_data.update(alloy_results)
        if alloy_errors:
            all_errors.extend(alloy_errors)

    run_summary['end_time'] = datetime.now().isoformat()
    run_summary['total_errors'] = len(all_errors)

    # Export results
    export_scheil_data(all_scheil_data)
    export_metadata(run_summary, all_errors)

    print("" + "="*70)
    print("\nCalculation complete.")
    print(f"Successful: {run_summary['successful_calculations']}")
    print(f"Failed: {run_summary['failed_calculations']}")
    print(f"Total errors: {run_summary['total_errors']}")
    print("" + "="*70)


def process_single_alloy(session, row, databases_and_elements, run_summary):
    """Process a single alloy with a fresh TC session."""
    # Extract alloy information
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
    # Select database
    alloy_elements = [alloy_info_dict['dependent_element']] + alloy_info_dict['other_elements']
    initial_database = alloy_info_dict['initial_database']
    
    # Check if the database has all the elements for the alloy
    if initial_database in databases_and_elements:
        available_elements = databases_and_elements[initial_database]
        if set(alloy_elements).issubset(available_elements):
            database = initial_database
        else:
            database = 'SSOL8'
            run_summary['database_switches'] += 1
    else:
        database = 'SSOL8'
        run_summary['database_switches'] += 1

    # Set calculation modes
    scan_speed = alloy_info_dict['scan_speed']
    calculation_modes = {
        'classic': ScheilCalculationType.scheil_classic(),
        'solute_trapping': (ScheilCalculationType.scheil_solute_trapping()
                            .set_scanning_speed(scan_speed))}

    # Run Scheil calculation for both modes
    alloy_results = {}
    alloy_errors = []

    for mode_name, calc_type in calculation_modes.items():
        print("" + "-"*50)
        print(f"  Running {mode_name} calculation...")
        mode_result = attempt_calculation(
            session=session,
            alloy_info=alloy_info_dict,
            database=database,
            calc_type=calc_type,
            mode_name=mode_name,
            run_summary=run_summary
        )

        if mode_result['success']:
            alloy_key = create_alloy_key(alloy_info_dict, mode_name)
            alloy_results[alloy_key] = mode_result['data']
            run_summary['successful_calculations'] += 1
        else:
            print("    CALCULATION FAILED - Analyzing cause...")
            if mode_result['errors']:
                error_details = mode_result['errors'][0]  # Get first error
                print(f"      Error Type: {error_details['error_type']}")
                print(f"      Error Message: {error_details['error_msg']}")
                print(f"      Database Used: {error_details.get('database', 'Unknown')}")
            
            # Check if we should retry with filtered elements
            contains_problem_elements = bool(set(alloy_info_dict['other_elements']) & PROBLEM_ELEMENTS)
            problem_elements_found = set(alloy_info_dict['other_elements']) & PROBLEM_ELEMENTS

            print(f"      Contains Problem Elements: {contains_problem_elements}")
            if problem_elements_found:
                print(f"      Problem Elements: {problem_elements_found}")
            
            if contains_problem_elements:
                print("    → RETRYING with filtered elements...")

                # Filter out problem elements
                filtered_elements = []
                filtered_wt_pcts = []
                
                for element, wt_pct in zip(alloy_info_dict['other_elements'], 
                                            alloy_info_dict['other_elements_wt_pct']):
                    if element not in PROBLEM_ELEMENTS:
                        filtered_elements.append(element)
                        filtered_wt_pcts.append(wt_pct)
                
                removed_elements = set(alloy_info_dict['other_elements']) - set(filtered_elements)
                print(f"    Removing problem elements: {removed_elements}")
                
                # Create filtered alloy info
                filtered_alloy = alloy_info_dict.copy()
                filtered_alloy['other_elements'] = filtered_elements
                filtered_alloy['other_elements_wt_pct'] = filtered_wt_pcts
                
                # Retry calculation
                retry_result = attempt_calculation(
                    session=session,
                    alloy_info=filtered_alloy,
                    database=database,
                    calc_type=calc_type,
                    mode_name=mode_name,
                    run_summary=run_summary
                )
                
                if retry_result['success']:
                    alloy_key = create_alloy_key(alloy_info_dict, mode_name)
                    retry_result['data']['metadata']['retry_attempted'] = True
                    retry_result['data']['metadata']['retry_successful'] = True
                    retry_result['data']['metadata']['elements_filtered'] = True
                    alloy_results[alloy_key] = retry_result['data']
                    run_summary['successful_calculations'] += 1
                    run_summary['retry_successes'] += 1
                    print(f"    Retry success: {alloy_key}")
                else:
                    # Both attempts failed
                    mode_result['errors'][0]['retry_attempted'] = True
                    mode_result['errors'][0]['retry_successful'] = False
                    alloy_errors.extend(mode_result['errors'])
                    alloy_errors.extend(retry_result['errors'])
                    run_summary['failed_calculations'] += 1
                    print("    Retry also failed")
            else:
                # No problem elements, just failed
                mode_result['errors'][0]['retry_attempted'] = False
                mode_result['errors'][0]['contains_problem_elements'] = False
                alloy_errors.extend(mode_result['errors'])
                run_summary['failed_calculations'] += 1
                print(f"    Failed: {mode_result['errors'][-1]['error_msg']}")
    return alloy_results, alloy_errors

def attempt_calculation(session, alloy_info, database, calc_type, mode_name, run_summary):
    """"Attempt a single calculation."""    
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
        alloy_elements = alloy_info['other_elements'] + [alloy_info['dependent_element']]
        print(f"    All Elements for TC: {alloy_elements}")

        # Create system and scheil calculation
        print("    → Creating ThermoCalc system...")
        system = session.select_database_and_elements(
            database, alloy_elements).get_system()
        print("    ✓ System created successfully")
        
        print("    → Setting up Scheil calculation...")
        scheil_calc = (system
                        .with_scheil_calculation()
                        .set_composition_unit(CompositionUnit.MASS_PERCENT))
        print("    ✓ Scheil calculation setup complete")

        # Sets inonic liquid for Ni-superalloys with O, P, S
        ionic_elements_present = [el for el in alloy_elements if el in IONIC_ELEMENTS]
        if database.startswith('TCNI') and ionic_elements_present:
            options = ScheilOptions().set_liquid_phase('IONIC_LIQ')
            scheil_calc = scheil_calc.with_options(options)
            run_summary['ionic_liq_used'] += 1
            print(f"    → Using IONIC_LIQ (ionic elements found: {ionic_elements_present})")
        else:
            print("    → Using standard liquid phase")

        print("    → Setting element compositions:")
        composition_set = {}
        for element, wt_pct in zip(alloy_info['other_elements'], alloy_info['other_elements_wt_pct']):
            scheil_calc.set_composition(element, wt_pct)
            composition_set[element] = wt_pct
            print(f"      {element}: {wt_pct} wt%")
        
        total_other = sum(alloy_info['other_elements_wt_pct'])
        dependent_pct = 100.0 - total_other
        composition_set[alloy_info['dependent_element']] = dependent_pct
        print(f"      {alloy_info['dependent_element']} (balance): {dependent_pct:.3f} wt%")
        print(f"    Total composition: {sum(composition_set.values()):.3f} wt%")
        
        print("    → Running ThermoCalc calculation...")
        calculation = scheil_calc.with_calculation_type(calc_type).calculate()
        print("    ✓ ThermoCalc calculation completed successfully")

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
            latent_heat = np.asarray(latent_heat_curve[phase].y) # J/g
            heat_capacity = np.asarray(heat_capacity_curve[phase].y) # J/gK
            
            drop_detected, phase_steepness = detect_steep_drop(phase, fs, temp_c)
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
        final_steepness_values = [all_steepness_values[i] for i in unique_idx_sorted]
        
        # Sort by fraction solid ascending, then temperature descending
        sort_idx = np.lexsort((-combined_data[:, 1], combined_data[:, 0]))
        combined_data = combined_data[sort_idx]
        final_phase_names = [final_phase_names[i] for i in sort_idx]
        final_steep_flags = [final_steep_flags[i] for i in sort_idx]
        final_steepness_values = [final_steepness_values[i] for i in sort_idx]

        if combined_data[-1, 0] < 0.9:
            raise SolidificationTerminationError("Solidification terminated prematurely (fs < 0.9)")

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
            run_summary['steep_drops_detected'] += 1
            print(f"    Steepness correction: {initial_temp_original:.1f}°C → {initial_temp_corrected:.1f}°C")
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
            'contains_problem_elements': bool(set(alloy_info['other_elements']) & PROBLEM_ELEMENTS),
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
        print(f"    Error in {mode_name} calculation: {e}")
        error_info = {
            'alloy_index': alloy_info['index'],
            'alloy_name': alloy_info['name'],
            'calc_type': mode_name,
            'database': database,
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'timestamp': datetime.now().isoformat()
        }
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

def create_alloy_key(alloy_info, mode_name):
    """Create unique key for alloy-mode calculation."""
    index_num = alloy_info['index']    
    
    # Clean alloy name
    name = alloy_info['name']
    # Remove bad characters
    bad_chars = ['/', '\\', ':', '*', '?', '[', ']', ' ', '(', ')', '+', '%']
    for char in bad_chars:
        name = name.replace(char, '_')
    
    # Remove extra underscores
    while '__' in name:
        name = name.replace('__', '_')
    name = name.strip('_')
    
    # Limit to 10 characters
    if len(name) > 10:
        name = name[:10]
    
    processing_str = f"{int(alloy_info['scan_speed'] * 1000)}mmps"
    
    return f"{index_num}_{name}_{processing_str}_{mode_name}"


def export_scheil_data(all_scheil_data):
    """Export Scheil data to separate Excel files by calculation mode."""
    if not all_scheil_data:
        print("No Scheil data to export")
        return
    
    # Separate by mode
    classic_data = {}
    solute_trapping_data = {}
    
    for alloy_key, data in all_scheil_data.items():
        if 'classic' in alloy_key:
            classic_data[alloy_key] = data
        elif 'solute_trapping' in alloy_key:
            solute_trapping_data[alloy_key] = data
    
    # Export classic mode
    print("" + "="*70)
    if classic_data:
        with pd.ExcelWriter('scheil_data_classic.xlsx', engine='openpyxl') as writer:
            for alloy_key, data in classic_data.items():
                sheet_name = alloy_key.replace('_classic', '')
                data['scheil_data'].to_excel(writer, sheet_name=sheet_name, index=False)
                metadata_df = pd.DataFrame([data['metadata']]).T
                metadata_df.columns = ['Value']
                metadata_df.to_excel(writer, sheet_name=sheet_name, startcol=8, header=['Metadata'])
        print("Classic data exported to scheil_data_classic.xlsx")
    
    # Export solute trapping mode
    if solute_trapping_data:
        with pd.ExcelWriter('scheil_data_solute_trapping.xlsx', engine='openpyxl') as writer:
            for alloy_key, data in solute_trapping_data.items():
                sheet_name = alloy_key.replace('_solute_trapping', '')
                data['scheil_data'].to_excel(writer, sheet_name=sheet_name, index=False)
                metadata_df = pd.DataFrame([data['metadata']]).T
                metadata_df.columns = ['Value']
                metadata_df.to_excel(writer, sheet_name=sheet_name, startcol=8, header=['Metadata'])
        print("Solute trapping data exported to scheil_data_solute_trapping.xlsx")
    print("" + "="*70)


def export_metadata(run_summary, all_errors):
    """Export run metadata and errors to JSON file."""
    metadata = {
        'run_summary': run_summary,
        'errors': all_errors,
        'error_summary': {
            'total_errors': len(all_errors),
            'error_types': {},
            'alloys_with_errors': []
        }
    }
    
    # Summarize error types
    for error in all_errors:
        error_type = error['error_type']
        if error_type not in metadata['error_summary']['error_types']:
            metadata['error_summary']['error_types'][error_type] = 0
        metadata['error_summary']['error_types'][error_type] += 1
        
        alloy_id = f"{error['alloy_index']}_{error['alloy_name']}"
        if alloy_id not in metadata['error_summary']['alloys_with_errors']:
            metadata['error_summary']['alloys_with_errors'].append(alloy_id)
    
    # Export to JSON
    output_file = 'calculation_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata exported to {output_file}")


if __name__ == "__main__":
    with TCPython() as session:
        print("Starting Scheil calculations for all alloys...")
        print("" + "="*70)
        
        cache_dir = Path(__file__).parent / "tc_cache"
        run_tcpython(session, cache_dir)
        
        print("All calculations complete.")
