#!/usr/bin/env python3
"""
FDA Nozzle Benchmark V&V - Shared Utilities
Common functions for experimental data parsing and error calculation.
"""

import re
import numpy as np
from pathlib import Path


# Configuration
Z_EXPANSION = 0.122685  # Expansion plane location in simulation coordinates (m)
RHO = 1056.0            # Fluid density (kg/m3) for kinematic pressure conversion


def parse_experimental_file(filepath):
    """Parse FDA experimental PIV data file into sections."""
    data = {}
    current_section = None
    current_data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('plot-'):
                if current_section and current_data:
                    data[current_section] = np.array(current_data)
                current_section = line
                current_data = []
            elif current_section:
                parts = line.replace('\t', '  ').split()
                if len(parts) >= 2:
                    try:
                        r = float(parts[0])
                        u = float(parts[1])
                        current_data.append([r, u])
                    except ValueError:
                        pass
        
        if current_section and current_data:
            data[current_section] = np.array(current_data)
    
    return data


def read_openfoam_sample(sample_dir, set_name, time='latestTime'):
    """Read OpenFOAM sample output in .xy format."""
    sample_dir = Path(sample_dir)
    
    if time == 'latestTime':
        times = [d for d in sample_dir.iterdir() if d.is_dir()]
        if times:
            time_dir = max(times, key=lambda x: float(x.name) if x.name.replace('.', '').isdigit() else 0)
        else:
            return None, None, None
    else:
        time_dir = sample_dir / str(time)
    
    if not time_dir.exists():
        return None, None, None
    
    xy_file = time_dir / f"{set_name}_p_U.xy"
    if not xy_file.exists():
        return None, None, None
    
    data = np.loadtxt(xy_file)
    if data.size == 0:
        return None, None, None
    
    # Return: position, pressure, velocity [Ux, Uy, Uz]
    return data[:, 0], data[:, 1], data[:, 2:5]


def calculate_error_metrics(exp_r, exp_u, sim_r, sim_u):
    """Calculate error metrics between experimental and simulation data."""
    if sim_r is None or len(sim_r) == 0 or exp_r is None:
        return None
    
    # Filter out zero experimental values
    mask = exp_u != 0
    if not np.any(mask):
        return None
    
    exp_r_filt = exp_r[mask]
    exp_u_filt = exp_u[mask]
    
    try:
        sim_interp = np.interp(exp_r_filt, sim_r, sim_u)
    except:
        return None
    
    errors = sim_interp - exp_u_filt
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))
    
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((exp_u_filt - np.mean(exp_u_filt))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    nrmse = rmse / np.max(np.abs(exp_u_filt)) * 100 if np.max(np.abs(exp_u_filt)) > 0 else np.nan
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MaxErr': max_err, 'NRMSE': nrmse}


def get_case_paths():
    """Get standard paths for the case directory."""
    case_dir = Path(__file__).parent.parent
    return {
        'case_dir': case_dir,
        'exp_dir': case_dir / "experimental_data",
        'exp_file': case_dir / "experimental_data" / "PIV_Sudden_Expansion_500_243.txt",
        'postprocess_dir': case_dir / "simulation_data",
        'plots_dir': case_dir / "plots"
    }


def print_metrics_table(all_metrics, title="ERROR METRICS SUMMARY"):
    """Print formatted error metrics table."""
    print(f"\n{'='*70}")
    print(title)
    print("="*70)
    print(f"{'Location':<25} {'RMSE':<12} {'NRMSE (%)':<12} {'R²':<10}")
    print("-"*70)
    
    for name, m in all_metrics.items():
        if m:
            rmse = f"{m['RMSE']:.4f}" if not np.isnan(m.get('RMSE', np.nan)) else "N/A"
            nrmse = f"{m['NRMSE']:.1f}" if not np.isnan(m.get('NRMSE', np.nan)) else "N/A"
            r2 = f"{m['R2']:.4f}" if not np.isnan(m.get('R2', np.nan)) else "N/A"
            print(f"{name:<25} {rmse:<12} {nrmse:<12} {r2:<10}")
        else:
            print(f"{name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("="*70)
