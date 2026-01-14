#!/usr/bin/env python3
"""
FDA Nozzle Benchmark V&V - Master Script
Runs all V&V comparison analyses and generates summary.
"""

import subprocess
import sys
import shutil
from pathlib import Path


def sync_simulation_data():
    """Sync postProcessing data from simulation folder to simulation_data."""
    root_dir = Path(__file__).parent
    source_dir = root_dir / "simulation" / "postProcessing" / "sampleDict"
    dest_dir = root_dir / "simulation_data"

    if not source_dir.exists():
        print(f"  [WARN] Data source not found: {source_dir}")
        print("         Make sure you ran: postProcess -func sampleDict")
        return False

    print(f"Syncing data from: {source_dir}")
    print(f"               to: {dest_dir}")

    # Create destination if needed
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    # Copy all time directories
    count = 0
    for item in source_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a time directory (number)
            if item.name.replace('.', '').isdigit():
                target = dest_dir / item.name
                if target.exists():
                    shutil.rmtree(target)
                
                # Manual recursive copy to avoid permission/metadata issues
                # (shutil.copytree attempts to preserve permissions which fails on some FS)
                try:
                    shutil.copytree(item, target, copy_function=shutil.copy)
                except shutil.Error as e:
                    # Ignore 'Operation not permitted' errors (metadata/chmod issues)
                    # as long as files are copied
                    pass
                except OSError as e:
                    # Fallback for other errors
                    print(f"  [WARN] Issue copying {item.name}: {e}")
                    
                print(f"  Synced time: {item.name}")
                count += 1
    
    if count == 0:
        print("  [WARN] No time directories found to sync.")
        return False
        
    print(f"  Synced {count} time directories.")
    return True


def run_script(script_name):
    """Run a Python script and return success status."""
    # Scripts are now in 'scripts' subdirectory
    root_dir = Path(__file__).parent
    script_path = root_dir / "scripts" / script_name
    
    if not script_path.exists():
        print(f"  [SKIP] {script_name} not found in scripts/")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        # Run from the 'scripts' directory so writes go to appropriate relative paths
        # and imports work correctly
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=str(root_dir / "scripts"),
            env={**dict(__import__('os').environ), 'MPLBACKEND': 'Agg'}
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    print("="*60)
    print("FDA Nozzle Benchmark - Complete V&V Analysis")
    print("Re = 500, Laminar, Sudden Expansion")
    print("="*60)
    
    # Sync data first
    print("\n[Data Synchronization]")
    sync_simulation_data()
    
    # Check if we have data to analyze
    data_dir = Path(__file__).parent / "simulation_data"
    has_data = False
    if data_dir.exists():
        for item in data_dir.rglob("*.xy"):
            has_data = True
            break
            
    if not has_data:
        print("\n[ERROR] No simulation data found in 'simulation_data/'.")
        print("        Analysis cannot proceed.")
        print("        Please ensure you have run the simulation and post-processing:")
        print("          1. cd simulation")
        print("          2. <run solver> (e.g. pimpleFoam)")
        print("          3. postProcess -func sampleDict")
        sys.exit(1)
    
    scripts = [
        ("vv_axial_velocity.py", "Axial Velocity (Uz) Profiles"),
        ("vv_radial_velocity.py", "Radial Velocity (Uy) Profiles"),
        ("vv_pressure.py", "Pressure Distribution"),
        ("vv_jet_width.py", "Jet Width Analysis"),
    ]
    
    results = {}
    for script, description in scripts:
        print(f"\n[{description}]")
        success = run_script(script)
        results[description] = "PASS" if success else "FAIL"
    
    # Print summary
    print("\n" + "="*60)
    print("V&V ANALYSIS SUMMARY")
    print("="*60)
    
    case_dir = Path(__file__).parent
    # Output plots are now in 'plots' subdirectory
    output_files = [
        "vv_axial_velocity.png",
        "vv_radial_velocity.png", 
        "vv_pressure.png",
        "vv_jet_width.png",
    ]
    
    print("\nGenerated Files (in plots/):")
    for fname in output_files:
        fpath = case_dir / "plots" / fname
        status = "✓" if fpath.exists() else "✗"
        print(f"  {status} {fname}")
    
    print("\nAnalysis Status:")
    for desc, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {desc}: {status}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
