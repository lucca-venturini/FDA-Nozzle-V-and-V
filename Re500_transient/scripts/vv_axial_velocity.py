#!/usr/bin/env python3
"""
FDA Nozzle Benchmark V&V - Axial Velocity Comparison
Compares axial velocity (Uz) profiles at all 12 experimental z-locations.
"""

import numpy as np
import matplotlib.pyplot as plt
from vv_utils import (
    parse_experimental_file, read_openfoam_sample, calculate_error_metrics,
    get_case_paths, print_metrics_table, Z_EXPANSION
)
import re


def plot_centerline_comparison(exp_data, sim_pos, sim_U, ax):
    """Plot centerline velocity comparison."""
    exp_found = None
    for key in exp_data:
        if 'z-distribution-axial-velocity' in key:
            exp_found = exp_data[key]
            break
    
    metrics = None
    if exp_found is not None:
        ax.plot(exp_found[:, 0] * 1000, exp_found[:, 1], 
                'ko', markersize=6, label='Experiment (PIV)')
        
        if sim_pos is not None and sim_U is not None:
            z_exp = (sim_pos - Z_EXPANSION) * 1000
            metrics = calculate_error_metrics(
                exp_found[:, 0] * 1000, exp_found[:, 1],
                z_exp, sim_U[:, 2]
            )
    
    if sim_pos is not None and sim_U is not None:
        z_exp = (sim_pos - Z_EXPANSION) * 1000
        ax.plot(z_exp, sim_U[:, 2], 'b-', linewidth=2, label='OpenFOAM')
    
    ax.set_xlabel('Axial position from expansion (mm)')
    ax.set_ylabel('Centerline axial velocity (m/s)')
    
    title = 'Centerline Axial Velocity'
    if metrics:
        title += f"\n(NRMSE: {metrics['NRMSE']:.1f}%, R²: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.4f})"
    ax.set_title(title, fontsize=10)
    
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    return metrics


def plot_radial_profile_comparison(exp_data, sim_pos, sim_U, z_exp_val, title_base, ax):
    """Plot comparison of radial velocity profile."""
    exp_found = None
    for key in exp_data:
        if 'profile-axial-velocity-at-z' in key:
            match = re.search(r'at-z\s+(-?[\d.]+)', key)
            if match:
                z_in_key = float(match.group(1))
                if abs(z_in_key - z_exp_val) < 0.001:
                    exp_found = exp_data[key]
                    break
    
    metrics = None
    if exp_found is not None:
        mask = exp_found[:, 1] != 0
        ax.plot(exp_found[mask, 0] * 1000, exp_found[mask, 1], 
                'ko', markersize=4, label='Experiment')
        
        if sim_pos is not None and sim_U is not None:
            metrics = calculate_error_metrics(
                exp_found[:, 0] * 1000, exp_found[:, 1],
                sim_pos * 1000, sim_U[:, 2]
            )
    
    if sim_pos is not None and sim_U is not None:
        ax.plot(sim_pos * 1000, sim_U[:, 2], 'b-', linewidth=2, label='OpenFOAM')
    
    ax.set_xlabel('Radial position (mm)')
    ax.set_ylabel('Axial velocity (m/s)')
    
    title = title_base
    if metrics:
        title += f"\n(NRMSE: {metrics['NRMSE']:.1f}%, R²: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.4f})"
    elif exp_found is None:
        title += "\n(No exp. data)"
    ax.set_title(title, fontsize=9)
    
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return metrics


def main():
    paths = get_case_paths()
    
    if not paths['postprocess_dir'].exists():
        print(f"Error: Run postProcess first")
        print(f"Looking for: {paths['postprocess_dir']}")
        return
    
    if not paths['exp_file'].exists():
        print(f"Error: Experimental file not found: {paths['exp_file']}")
        return
    
    print(f"Loading experimental data from: {paths['exp_file']}")
    exp_data = parse_experimental_file(paths['exp_file'])
    print(f"Found {len(exp_data)} data sections")
    
    # All 15 experimental z-locations (aligned exactly to PIV data)
    z_locations = [
        (-0.088, 'radial_z_minus088', 'z = -88mm (inlet)'),
        (-0.064, 'radial_z_minus064', 'z = -64mm'),
        (-0.048, 'radial_z_minus048', 'z = -48mm (collector)'),
        (-0.042, 'radial_z_minus042', 'z = -42mm'),
        (-0.020, 'radial_z_minus020', 'z = -20mm (throat)'),
        (-0.008, 'radial_z_minus008', 'z = -8mm'),
        (0.000, 'radial_z_000', 'z = 0 (expansion)'),
        (0.008, 'radial_z_plus008', 'z = +8mm'),
        (0.016, 'radial_z_plus016', 'z = +16mm'),
        (0.024, 'radial_z_plus024', 'z = +24mm'),
        (0.032, 'radial_z_plus032', 'z = +32mm'),
        (0.040, 'radial_z_plus040', 'z = +40mm'),
        (0.048, 'radial_z_plus048', 'z = +48mm'),
        (0.060, 'radial_z_plus060', 'z = +60mm'),
        (0.080, 'radial_z_plus080', 'z = +80mm'),
    ]
    
    # Create figure: 4 rows x 4 columns (1 centerline + 15 radial = 16 plots)
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    fig.suptitle('FDA Nozzle V&V - Axial Velocity (Uz) Profiles\nRe = 500 (Laminar), Sudden Expansion', fontsize=14)
    
    all_metrics = {}
    
    # Centerline (top-left)
    sim_pos, sim_p, sim_U = read_openfoam_sample(paths['postprocess_dir'], 'centerline')
    metrics = plot_centerline_comparison(exp_data, sim_pos, sim_U, axes[0, 0])
    all_metrics['centerline'] = metrics
    
    # Radial profiles
    for i, (z_exp, set_name, title) in enumerate(z_locations):
        row = (i + 1) // 4
        col = (i + 1) % 4
        sim_pos, sim_p, sim_U = read_openfoam_sample(paths['postprocess_dir'], set_name)
        metrics = plot_radial_profile_comparison(exp_data, sim_pos, sim_U, z_exp, title, axes[row, col])
        all_metrics[set_name] = metrics
    
    # Hide unused subplots
    for i in range(len(z_locations) + 1, 16):
        row = i // 4
        col = i % 4
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if not paths['plots_dir'].exists():
        paths['plots_dir'].mkdir(parents=True, exist_ok=True)
        
    output_file = paths['plots_dir'] / "vv_axial_velocity.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    print_metrics_table(all_metrics, "AXIAL VELOCITY (Uz) ERROR METRICS")
    
    plt.show()


if __name__ == "__main__":
    main()
