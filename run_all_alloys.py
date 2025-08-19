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
                timeout=10800)  # 3hr timeout

            print(result.stdout)

            if result.returncode == 0:
                print("  ✓ Subprocess completed successfully")

            else:
                print(
                    f"  ✗ Subprocess failed (exit code: {result.returncode})")
                failed_alloys.append((idx, alloy_name))

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
            failed_alloys.append((idx, alloy_name))

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
            failed_alloys.append((idx, alloy_name))

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
    calc_errors_file = start_time_results_dir / 'calculation_errors.json'

    if calc_errors_file.exists():
        try:
            with open(calc_errors_file, 'r') as f:
                calculation_errors = json.load(f)
            print(f"✓ Loaded {len(calculation_errors)} calculation errors")
        except Exception as e:
            print(f"Warning: Could not load calculation errors: {e}")
    else:
        print("No calculation_errors.json file found")

    # Combine all errors (calculation + subprocess)
    all_errors = calculation_errors + subprocess_errors

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
        'all_errors': all_errors,
        'error_summary': {
            'error_types': error_types,
            'failed_alloy_names': [name for idx, name in failed_alloys],
            'most_common_error': max(error_types.items(),
                                     key=lambda x: x[1])[0]
            if error_types else None
        }
    }

    # Export error-focused metadata
    output_file = start_time_results_dir / 'overall_calculation_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(error_metadata, f, indent=2)

    # Clean up temporary calculation errors file
    if calc_errors_file.exists():
        calc_errors_file.unlink()
        print("✓ Cleaned up temporary calculation_errors.json")

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
        for idx, name in failed_alloys[:10]:  # Show first 10
            print(f"  {idx}: {name}")
        if len(failed_alloys) > 10:
            print(f"  ... and {len(failed_alloys) - 10} more")

    print(f"\n✓ Error metadata exported to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_alloys()
