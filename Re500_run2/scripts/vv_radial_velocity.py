#!/usr/bin/env python3
"""
FDA Nozzle Benchmark V&V - Radial Velocity Comparison
Compares radial velocity (Uy) profiles at z-locations downstream of expansion.
"""

import numpy as np
import matplotlib.pyplot as plt
from vv_utils import (
    parse_experimental_file, read_openfoam_sample, calculate_error_metrics,
    get_case_paths, print_metrics_table, Z_EXPANSION
)
import re


def plot_radial_velocity_profile(exp_data, sim_pos, sim_U, z_exp_val, title_base, ax):
    """Plot comparison of radial velocity profile."""
    exp_found = None
    for key in exp_data:
        if 'profile-radial-velocity-at-z' in key:
            match = re.search(r'at-z\s+(-?[\d.]+)', key)
            if match:
                z_in_key = float(match.group(1))
                if abs(z_in_key - z_exp_val) < 0.001:
                    exp_found = exp_data[key]
                    break
    
    metrics = None
    if exp_found is not None:
        mask = exp_found[:, 1] != 0
        if np.any(mask):
            ax.plot(exp_found[mask, 0] * 1000, exp_found[mask, 1], 
                    'ko', markersize=4, label='Experiment')
            
            if sim_pos is not None and sim_U is not None:
                metrics = calculate_error_metrics(
                    exp_found[:, 0] * 1000, exp_found[:, 1],
                    sim_pos * 1000, sim_U[:, 1]  # Uy = radial component
                )
        else:
            # All zeros - still plot experimental zeros
            ax.plot(exp_found[:, 0] * 1000, exp_found[:, 1], 
                    'ko', markersize=3, alpha=0.3, label='Experiment')
    
    if sim_pos is not None and sim_U is not None:
        ax.plot(sim_pos * 1000, sim_U[:, 1], 'b-', linewidth=2, label='OpenFOAM')
    
    ax.set_xlabel('Radial position (mm)')
    ax.set_ylabel('Radial velocity (m/s)')
    
    title = title_base
    if metrics:
        title += f"\n(NRMSE: {metrics['NRMSE']:.1f}%, R²: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.4f})"
    elif exp_found is None:
        title += "\n(No exp. data)"
    ax.set_title(title, fontsize=9)
    
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    return metrics


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
    
    # Check which radial velocity profiles are available
    radial_v_sections = [k for k in exp_data.keys() if 'profile-radial-velocity' in k]
    print(f"Found {len(radial_v_sections)} radial velocity profile sections")
    
    # z-locations with radial velocity data (aligned to experiment)
    z_locations = [
        (-0.088, 'radial_z_minus088', 'z = -88mm (inlet)'),
        (-0.064, 'radial_z_minus064', 'z = -64mm'),
        (-0.048, 'radial_z_minus048', 'z = -48mm'),
        (-0.042, 'radial_z_minus042', 'z = -42mm'),
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
    
    # Create figure: 3 rows x 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle('FDA Nozzle V&V - Radial Velocity (Uy) Profiles\nRe = 500 (Laminar), Sudden Expansion', fontsize=14)
    
    all_metrics = {}
    
    for i, (z_exp, set_name, title) in enumerate(z_locations):
        row = i // 3
        col = i % 3
        sim_pos, sim_p, sim_U = read_openfoam_sample(paths['postprocess_dir'], set_name)
        metrics = plot_radial_velocity_profile(exp_data, sim_pos, sim_U, z_exp, title, axes[row, col])
        all_metrics[set_name] = metrics
    
    plt.tight_layout()
    
    # Save figure
    if not paths['plots_dir'].exists():
        paths['plots_dir'].mkdir(parents=True, exist_ok=True)
        
    output_file = paths['plots_dir'] / "vv_radial_velocity.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    print_metrics_table(all_metrics, "RADIAL VELOCITY (Uy) ERROR METRICS")
    
    plt.show()


if __name__ == "__main__":
    main()
