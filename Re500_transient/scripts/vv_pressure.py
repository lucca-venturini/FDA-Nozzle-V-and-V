#!/usr/bin/env python3
"""
FDA Nozzle Benchmark V&V - Pressure Comparison
Compares wall and centerline pressure distributions with experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
from vv_utils import (
    parse_experimental_file, read_openfoam_sample, calculate_error_metrics,
    get_case_paths, print_metrics_table, Z_EXPANSION, RHO
)


def main():
    paths = get_case_paths()
    
    if not paths['postprocess_dir'].exists():
        print(f"Error: Run postProcess first")
        return
    
    if not paths['exp_file'].exists():
        print(f"Error: Experimental file not found: {paths['exp_file']}")
        return
    
    print(f"Loading experimental data from: {paths['exp_file']}")
    exp_data = parse_experimental_file(paths['exp_file'])
    
    # Find pressure data
    wall_p = None
    centerline_p = None
    
    for key in exp_data:
        if 'wall-distribution-pressure' in key:
            wall_p = exp_data[key]
        elif 'z-distribution-pressure' in key:
            centerline_p = exp_data[key]
    
    if wall_p is None and centerline_p is None:
        print("No pressure data found in experimental file")
        return
    
    # Load simulation centerline data
    sim_pos, sim_p_kin, sim_U = read_openfoam_sample(paths['postprocess_dir'], 'centerline')
    
    # Convert kinematic pressure to static pressure (Pa)
    if sim_p_kin is not None:
        sim_p = sim_p_kin * RHO
    else:
        sim_p = None
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FDA Nozzle V&V - Pressure Distribution\nRe = 500 (Laminar), Sudden Expansion', fontsize=14)
    
    all_metrics = {}
    
    # Plot 1: Centerline pressure
    ax = axes[0]
    if centerline_p is not None:
        ax.plot(centerline_p[:, 0] * 1000, centerline_p[:, 1], 
                'ko', markersize=6, label='Experiment (PIV)')
        
        if sim_pos is not None and sim_p is not None:
            # Convert simulation z to experimental coordinates (relative to expansion)
            z_exp_sim = (sim_pos - Z_EXPANSION) * 1000
            
            # Normalize pressure (simulation gives gauge pressure, experimental is relative)
            # Shift simulation pressure to match experimental reference at z=0
            exp_at_zero = np.interp(0, centerline_p[:, 0] * 1000, centerline_p[:, 1])
            sim_at_zero = np.interp(0, z_exp_sim, sim_p)
            p_offset = exp_at_zero - sim_at_zero
            sim_p_shifted = sim_p + p_offset
            
            ax.plot(z_exp_sim, sim_p_shifted, 'b-', linewidth=2, label='OpenFOAM (shifted)')
            
            # Calculate metrics on shifted data
            metrics = calculate_error_metrics(
                centerline_p[:, 0] * 1000, centerline_p[:, 1],
                z_exp_sim, sim_p_shifted
            )
            all_metrics['centerline_p'] = metrics
            
            if metrics:
                ax.set_title(f"Centerline Pressure\nNRMSE: {metrics['NRMSE']:.1f}%, R²: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.1f}")
            else:
                ax.set_title('Centerline Pressure Distribution')
        else:
            ax.set_title('Centerline Pressure Distribution (Sim data missing)')
    else:
        ax.text(0.5, 0.5, 'No experimental data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_title('Centerline Pressure Distribution')
    
    ax.set_xlabel('Axial position from expansion (mm)')
    ax.set_ylabel('Pressure (Pa)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Expansion')
    
    # Plot 2: Wall pressure
    ax = axes[1]
    
    # Load simulation wall data
    sim_wall_pos, sim_wall_p, _ = read_openfoam_sample(paths['postprocess_dir'], 'wall_pressure')
    
    if wall_p is not None:
        ax.plot(wall_p[:, 0] * 1000, wall_p[:, 1], 
                'ko', markersize=6, label='Experiment')
                
        if sim_wall_pos is not None and sim_wall_p is not None:
            # Convert simulation z to experimental coordinates
            z_exp_wall = (sim_wall_pos - Z_EXPANSION) * 1000
            
            # Apply same pressure offset as centerline if available, 
            # otherwise align at start of wall section (z=0)
            if 'p_offset' in locals():
                sim_wall_shifted = sim_wall_p + p_offset
                label_suffix = "(shifted)"
            else:
                 # Calculate independent offset if centerline wasn't processed
                 exp_Start = np.interp(0, wall_p[:, 0] * 1000, wall_p[:, 1])
                 sim_Start = np.interp(0, z_exp_wall, sim_wall_p)
                 wall_offset = exp_Start - sim_Start
                 sim_wall_shifted = sim_wall_p + wall_offset
                 label_suffix = "(indep. shifted)"
            
            ax.plot(z_exp_wall, sim_wall_shifted, 'b-', linewidth=2, label=f'OpenFOAM {label_suffix}')
            
            metrics_wall = calculate_error_metrics(
                wall_p[:, 0] * 1000, wall_p[:, 1],
                z_exp_wall, sim_wall_shifted
            )
            all_metrics['wall_p'] = metrics_wall
            if metrics_wall:
                ax.set_title(f"Wall Pressure\nNRMSE: {metrics_wall['NRMSE']:.1f}%, R²: {metrics_wall['R2']:.3f}, RMSE: {metrics_wall['RMSE']:.1f}")
            else:
                ax.set_title('Wall Pressure Distribution')
        else:
            ax.set_title('Wall Pressure Distribution')
    elif sim_wall_pos is not None:
        # Plot simulation only if exp missing
        z_exp_wall = (sim_wall_pos - Z_EXPANSION) * 1000
        ax.plot(z_exp_wall, sim_wall_p, 'b-', linewidth=2, label='OpenFOAM (gauge)')
        ax.set_title('Wall Pressure Distribution')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_title('Wall Pressure Distribution')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    if not paths['plots_dir'].exists():
        paths['plots_dir'].mkdir(parents=True, exist_ok=True)
        
    output_file = paths['plots_dir'] / "vv_pressure.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    if all_metrics:
        print_metrics_table(all_metrics, "PRESSURE ERROR METRICS")
    
    plt.show()


if __name__ == "__main__":
    main()
