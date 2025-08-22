"""
Master Orchestration Script for Hot Cracking Analysis.

Manages the execution of individual alloy calculations by running each alloy
in a separate subprocess. Creates timestamped result directories, handles
subprocess coordination, and aggregates calculation results and errors.
This is the main script to run for processing entire alloy datasets
through the hot cracking analysis pipeline.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import subprocess
import json
from run_tcpython import clean_filename


def run_all_alloys():
    """Run TC-Python calculations for each alloy in separate subprocesses."""
    # PREREQUISITE CHECKS:
    working_dir = Path('working_files')
    if not (working_dir / 'processed_data.pkl').exists():
        print("Error: working_files/processed_data.pkl not found.")
        return
    if not (working_dir / 'databases_and_elements.pkl').exists():
        print("Error: working_files/databases_and_elements.pkl not found.")
        return
    if not Path('run_tcpython.py').exists():
        print("Error: run_tcpython.py not found.")
        return
    print("✓ All prerequisites found")

    # Create output directories for CSV files
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_results_dir = Path('results')
    base_results_dir.mkdir(exist_ok=True)
    start_time_results_dir = base_results_dir / start_time
    start_time_results_dir.mkdir(exist_ok=True)

    # Create the classic and solute trapping sub-directories
    classic_dir = start_time_results_dir / 'classic_scheil'
    solute_trapping_dir = start_time_results_dir / 'solute_trapping_scheil'
    classic_dir.mkdir(exist_ok=True)
    solute_trapping_dir.mkdir(exist_ok=True)
    print(f"✓ Created timestamped results directory: {start_time_results_dir}")

    df = pd.read_pickle(working_dir / 'processed_data.pkl')
    total_alloys = len(df)
    print(f"Loaded {total_alloys} alloys")

    # Track results
    subprocess_errors = []
    failed_alloys = []
    # alloy_stdout_logs = {}
    start_time = datetime.now()

    print("\nStarting individual alloy processing...")
    print("=" * 60)
    for idx, row in df.iterrows():
        alloy_name = row.get('Alloy Name', f'Unknown_{idx}')
        print(f"\nProcessing alloy {idx + 1}/{total_alloys}: {alloy_name}")

        single_alloy_df = df.iloc[[idx]].copy()
        temp_file = working_dir / f'temp_alloy_{idx}.pkl'
        single_alloy_df.to_pickle(temp_file)
        print(f"  ✓ Created temp file: {temp_file.name}")

        try:
            result = subprocess.run([
                sys.executable, 'run_tcpython.py', str(temp_file),
                str(start_time_results_dir)
                ], capture_output=True, text=True,
                timeout=3600)  # 1.0hr timeout

            print(result.stdout)

            if result.returncode == 0:
                print("  ✓ Subprocess completed successfully")

            else:
                print(
                    f"  ✗ Subprocess failed (exit code: {result.returncode})")
                failed_alloys.append(f"{idx}_{clean_filename(alloy_name)}_subprocess")

                # Collect subprocess failure error
                subprocess_error = {
                    'alloy_index': idx,
                    'alloy_name': alloy_name,
                    'error_type': 'SubprocessFailure',
                    'error_msg': result.stderr[:500]
                    if result.stderr else 'Unknown subprocess error',
                    'exit_code': result.returncode,
                    'stdout_snippet': result.stdout[-200:]
                    if result.stdout else '',
                    'timestamp': datetime.now().isoformat()
                }
                subprocess_errors.append(subprocess_error)
                print(f"    Error output: {result.stderr[:100]}...")

        except subprocess.TimeoutExpired as e:
            print(f"  ✗ Subprocess timed out after {e.timeout} seconds")
            failed_alloys.append(f"{idx}_{clean_filename(alloy_name)}_timeout")

            # Collect timeout error
            timeout_error = {
                'alloy_index': idx,
                'alloy_name': alloy_name,
                'error_type': 'TimeoutError',
                'error_msg': f"Calculation timed out after {e.timeout} seconds",
                'timeout_seconds': e.timeout,
                'timestamp': datetime.now().isoformat()
            }
            subprocess_errors.append(timeout_error)

        except Exception as e:
            print(f"  ✗ Subprocess error: {e}")
            failed_alloys.append(f"{idx}_{clean_filename(alloy_name)}_subprocess_error")

            # Collect general subprocess error
            general_error = {
                'alloy_index': idx,
                'alloy_name': alloy_name,
                'error_type': 'SubprocessError',
                'error_msg': str(e),
                'timestamp': datetime.now().isoformat()
            }
            subprocess_errors.append(general_error)

        temp_file.unlink()
        print("  ✓ Cleaned up temp file")

    # Calculate final timing and statistics
    end_time = datetime.now()
    duration = end_time - start_time
    successful_count = total_alloys - len(failed_alloys)

    # Collect calculation errors from individual alloy processing
    print("\nCollecting calculation errors...")
    calculation_errors = []
    # Look for all calculation_errors_*.json files
    error_files = list(start_time_results_dir.glob('calculation_errors_*.json'))
    print(f"Found {len(error_files)} error files")

    failed_alloy_indices_from_calc_errors = set()  # Track which alloys had calc errors
    failed_alloy_calc_modes = {}

    for error_file in error_files:
        try:
            with open(error_file, 'r') as f:
                file_errors = json.load(f)
                calculation_errors.extend(file_errors)

                # Track alloy indices and their calc modes that had calculation errors
                for error in file_errors:
                    alloy_idx = error.get('alloy_index')
                    calc_mode = error.get('calc_type', 'unknown')

                    failed_alloy_indices_from_calc_errors.add(alloy_idx)

                    # Track calc modes for this alloy
                    if alloy_idx not in failed_alloy_calc_modes:
                        failed_alloy_calc_modes[alloy_idx] = set()
                    failed_alloy_calc_modes[alloy_idx].add(calc_mode)

            print(f"✓ Loaded {len(file_errors)} errors from {error_file.name}")
            # Clean up the individual error file
            error_file.unlink()
        except Exception as e:
            print(f"Warning: Could not load {error_file}: {e}")

    # Add alloys with calculation errors to failed_alloys list

    for calc_error_idx in failed_alloy_indices_from_calc_errors:
        # Get alloy name from df
        if calc_error_idx < len(df):
            alloy_name = df.iloc[calc_error_idx].get('Alloy Name', f'Unknown_{calc_error_idx}')

            # Get calc modes for this alloy and create entries for each mode
            calc_modes = failed_alloy_calc_modes.get(calc_error_idx, {'unknown'})
            for calc_mode in calc_modes:
                failed_alloys.append(f"{calc_error_idx}_{clean_filename(alloy_name)}_{calc_mode}")
                print(f"  Added alloy {calc_error_idx} ({alloy_name}) - {calc_mode} to failed list due to calculation errors")

    print(f"✓ Total calculation errors collected: {len(calculation_errors)}")
    print(f"✓ Total failed alloys (subprocess + calculation): {len(failed_alloys)}")

    # Sort failed_alloys by index (ascending)
    failed_alloys.sort(key=lambda x: int(x.split('_')[0]))

    # Combine all errors (calculation + subprocess)
    all_errors = calculation_errors + subprocess_errors
    all_errors.sort(key=lambda error: error.get('alloy_index', -1))

    # Analyze error types
    error_types = {}
    for error in all_errors:
        error_type = error['error_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1

    # Create simplified error-focused metadata
    error_metadata = {
        'run_summary': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration),
            'total_alloys': total_alloys,
            'successful_alloys': successful_count,
            'failed_alloys': len(failed_alloys),
            'success_rate_percent': round(
                (successful_count / total_alloys) * 100, 1),
            'total_errors': len(all_errors),
            'calculation_errors': len(calculation_errors),
            'subprocess_errors': len(subprocess_errors)
        },
        'error_summary': {
            'error_types': error_types,
            'failed_alloy_names': failed_alloys
        },
        'all_errors': all_errors
    }

    # Export error-focused metadata
    output_file = start_time_results_dir / 'overall_calculation_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(error_metadata, f, indent=2)

    # Final summary output
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total alloys processed: {total_alloys}")
    print(f"Successful: {successful_count} ({round((successful_count/total_alloys)*100, 1)}%)")
    print(f"Failed: {len(failed_alloys)} ({round((len(failed_alloys)/total_alloys)*100, 1)}%)")
    print(f"Total processing time: {duration}")
    print("\nError Summary:")
    print(f"  Total errors logged: {len(all_errors)}")
    print(f"  Calculation errors: {len(calculation_errors)}")
    print(f"  Subprocess errors: {len(subprocess_errors)}")

    if error_types:
        print("\nMost common error types:")
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1],
                               reverse=True)
        for error_type, count in sorted_errors[:5]:  # Top 5 error types
            print(f"  {error_type}: {count}")

    if failed_alloys:
        print(f"\nFailed alloys ({len(failed_alloys)}):")
        for failed_entry in failed_alloys[:10]:  # Show first 10
            print(f"  {failed_entry}")
        if len(failed_alloys) > 10:
            print(f"  ... and {len(failed_alloys) - 10} more")

    print(f"\n✓ Error metadata exported to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_alloys()
