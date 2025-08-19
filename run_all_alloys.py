# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 12:10:02 2025

@author: amjordan7
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import subprocess
import json


def run_all_alloys():
    """Run TC-Python calculations for each alloy in separate subprocesses."""
    # ADD BACK PREREQUISITE CHECKS:
    print("Checking prerequisites...")
    if not Path('processed_data.pkl').exists():
        print("Error: processed_data.pkl not found.")
        return
    if not Path('databases_and_elements.pkl').exists():
        print("Error: databases_and_elements.pkl not found.")
        return
    if not Path('run_tcpython.py').exists():
        print("Error: run_tcpython.py not found.")
        return
    print("✓ All prerequisites found")
    
    # Create output directories for CSV files
    output_dir = Path('scheil_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create the classic and solute trapping sub-directories
    classic_dir = output_dir / 'classic'
    solute_trapping_dir = output_dir / 'solute_trapping'
    classic_dir.mkdir(exist_ok=True)
    solute_trapping_dir.mkdir(exist_ok=True)
    print("✓ Created scheil_results/classic/ and scheil_results/solute_trapping/")
    
    df = pd.read_pickle('processed_data.pkl')
    total_alloys = len(df)
    print(f"Loaded {total_alloys} alloys")
    
    overall_summary = {
        'start_time': datetime.now().isoformat(),
        'total_alloys': total_alloys,
        'successful_calculations': 0,
        'failed_calculations': 0,
        'database_switches': 0,
        'ionic_liq_used': 0,
        'steep_drops_detected': 0,
        'retry_successes': 0
    }
    
    temp_dir = Path('temp_alloys')
    temp_dir.mkdir(exist_ok=True)
    
    # Track results
    successful_alloys = []
    failed_alloys = []
    all_subprocess_errors = []
    start_time = datetime.now()
    
    print("\nStarting individual alloy processing...")
    print("=" * 60)
    for idx,row in df.iterrows():
        alloy_name = row.get('Alloy Name', f'Unknown_{idx}')
        print(f"\nProcessing alloy {idx + 1}/{total_alloys}: {alloy_name}")
        
        single_alloy_df = df.iloc[[idx]].copy()
        
        temp_file = temp_dir / f'temp_alloy_{idx}.pkl'
        single_alloy_df.to_pickle(temp_file)
        print(f"  ✓ Created temp file: {temp_file.name}")
        
        try:
            result = subprocess.run([
                sys.executable, 'run_tcpython.py', str(temp_file)
                ], capture_output=True, text=True, timeout=10800) # 3hr timeout
            print(result.stdout)
            if result.returncode == 0:
                print("  ✓ Subprocess completed successfully")
                successful_alloys.append((idx, alloy_name))
                overall_summary['successful_calculations'] += 2
                
                # PARSE SUBPROCESS OUTPUT FOR SUMMARY DATA (ADD THIS)
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Database switched to SSOL8' in line:
                        overall_summary['database_switches'] += 1
                    elif 'Using IONIC_LIQ' in line:
                        overall_summary['ionic_liq_used'] += 1
                    elif 'Steep drop detected' in line:
                        overall_summary['steep_drops_detected'] += 1
                    elif 'Retry success' in line:
                        overall_summary['retry_successes'] += 1
            
            else:
                print(f"  ✗ Subprocess failed (exit code: {result.returncode})")
                print(f"    Error output: {result.stderr[:200]}...")
                failed_alloys.append((idx, alloy_name, result.stderr))
                overall_summary['failed_calculations'] += 2  # Both modes failed

        except Exception as e:
            print(f"  ✗ Subprocess error: {e}")
            error_detail = {
                'alloy_index': idx,
                'alloy_name': alloy_name,
                'error_type': 'SubprocessError',
                'error_msg': str(e),
                'timestamp': datetime.now().isoformat()
            }
            failed_alloys.append((idx, alloy_name, str(e)))
            all_subprocess_errors.append(error_detail)
            
        temp_file.unlink()
        print("  ✓ Cleaned up temp file")
 
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    overall_summary['end_time'] = datetime.now().isoformat()
    overall_summary['total_errors'] = len(failed_alloys)
    
    # Export final metadata
    overall_summary['end_time'] = end_time.isoformat()
    export_final_metadata(
        overall_summary, successful_alloys, failed_alloys, all_subprocess_errors
    )
      
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total alloys processed: {total_alloys}")
    print(f"Successful: {len(successful_alloys)}")
    print(f"Failed: {len(failed_alloys)}")
    print(f"Success rate: {len(successful_alloys)/total_alloys*100:.1f}%")
    print(f"Total time: {duration}")
    
    if failed_alloys:
        print("\nFailed alloys:")
        for idx, name, error in failed_alloys:
            print(f"  {idx}: {name} - {error[:50]}...")


def export_final_metadata(overall_summary, successful_alloys, failed_alloys, all_subprocess_errors):
    """Export comprehensive metadata for all alloy calculations."""
    
    # Summarize error types
    error_types = {}
    alloys_with_errors = []
    
    for error in all_subprocess_errors:
        error_type = error['error_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        alloy_id = f"{error['alloy_index']}_{error['alloy_name']}"
        if alloy_id not in alloys_with_errors:
            alloys_with_errors.append(alloy_id)
    
    # Create comprehensive metadata
    final_metadata = {
        'run_summary': overall_summary,
        'successful_alloys': [{'index': idx, 'name': name} for idx, name in successful_alloys],
        'failed_alloys': [{'index': idx, 'name': name, 'error': err} for idx, name, err in failed_alloys],
        'detailed_errors': all_subprocess_errors,
        'error_summary': {
            'total_errors': len(all_subprocess_errors),
            'error_types': error_types,
            'alloys_with_errors': alloys_with_errors
        }
    }
    
    # Export to JSON
    output_file = Path('scheil_results/overall_calculation_metadata.json')
    with open(output_file, 'w') as f:
        json.dump(final_metadata, f, indent=2)
    
    print(f"\n✓ Comprehensive metadata exported to {output_file}")
    return output_file

if __name__ == "__main__":
    run_all_alloys()