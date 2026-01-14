#!/usr/bin/env python3
"""
FDA Nozzle Benchmark V&V - Jet Width Analysis
Calculates jet half-width from simulation and compares with experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
from vv_utils import (
    parse_experimental_file, read_openfoam_sample, calculate_error_metrics,
    get_case_paths, print_metrics_table, Z_EXPANSION
)


def calculate_jet_width(r, u_axial):
    """
    Calculate jet full width (diameter) as the distance between radial positions 
    where velocity drops to half of centerline.
    Returns the jet width in meters.
    """
    if r is None or u_axial is None or len(r) < 3:
        return None
    
    # Find centerline velocity (at r ≈ 0)
    center_idx = np.argmin(np.abs(r))
    u_center = u_axial[center_idx]
    
    if u_center <= 0:
        return None
    
    u_half = u_center / 2
    
    # Find where u crosses u_half on positive r side
    positive_r_mask = r >= 0
    r_pos = r[positive_r_mask]
    u_pos = u_axial[positive_r_mask]
    
    # Find crossing point by interpolation
    for i in range(len(u_pos) - 1):
        if u_pos[i] >= u_half and u_pos[i+1] < u_half:
            # Linear interpolation
            t = (u_half - u_pos[i]) / (u_pos[i+1] - u_pos[i])
            r_half = r_pos[i] + t * (r_pos[i+1] - r_pos[i])
            return 2 * r_half  # Return Diameter (2 * Radius)
    
    return None

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
    
    # Find jet width experimental data
    jet_width_exp = None
    for key in exp_data:
        if 'jet-width' in key:
            jet_width_exp = exp_data[key]
            break
    
    if jet_width_exp is None:
        print("No jet width data found in experimental file")
        return
    
    print(f"Found jet width data with {len(jet_width_exp)} points")
    
    # z-locations to calculate simulation jet width
    z_locations = [
        (-0.088, 'radial_z_minus088'),
        (-0.064, 'radial_z_minus064'),
        (-0.048, 'radial_z_minus048'),
        (-0.042, 'radial_z_minus042'),
        (-0.020, 'radial_z_minus020'),
        (-0.008, 'radial_z_minus008'),
        (0.000, 'radial_z_000'),
        (0.008, 'radial_z_plus008'),
        (0.016, 'radial_z_plus016'),
        (0.024, 'radial_z_plus024'),
        (0.032, 'radial_z_plus032'),
        (0.040, 'radial_z_plus040'),
        (0.048, 'radial_z_plus048'),
        (0.060, 'radial_z_plus060'),
        (0.080, 'radial_z_plus080'),
    ]
    
    # Calculate simulation jet width at each z-location
    sim_z = []
    sim_jet_width = []
    
    for z_exp_coord, set_name in z_locations:
        sim_pos, sim_p, sim_U = read_openfoam_sample(paths['postprocess_dir'], set_name)
        if sim_pos is not None and sim_U is not None:
            jw = calculate_jet_width(sim_pos, sim_U[:, 2])
            if jw is not None:
                sim_z.append(z_exp_coord)
                sim_jet_width.append(jw)
            else:
                # If width calculation fails (e.g. uniform flow), append NaN or 0? 
                # Better to skip for plotting, but might miss points.
                pass
    
    sim_z = np.array(sim_z)
    sim_jet_width = np.array(sim_jet_width)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FDA Nozzle V&V - Jet Width Analysis\nRe = 500 (Laminar), Sudden Expansion', fontsize=14)
    
    # Plot 1: Jet width vs z (Downstream only)
    ax = axes[0]
    
    # Filter experimental data to downstream region (z >= 0)
    exp_downstream = jet_width_exp[jet_width_exp[:, 0] >= -0.001]
    
    ax.plot(exp_downstream[:, 0] * 1000, exp_downstream[:, 1] * 1000, 
            'ko', markersize=8, label='Experiment (PIV)')
    
    if len(sim_z) > 0:
        # Filter sim data
        mask_down = sim_z >= -0.001
        if np.any(mask_down):
            ax.plot(sim_z[mask_down] * 1000, sim_jet_width[mask_down] * 1000, 
                    'bs-', markersize=6, linewidth=2, label='OpenFOAM')
            
            # Calculate metrics
            metrics = calculate_error_metrics(
                exp_downstream[:, 0] * 1000, exp_downstream[:, 1] * 1000,
                sim_z[mask_down] * 1000, sim_jet_width[mask_down] * 1000
            )
            if metrics:
                ax.set_title(f"Jet Width (Downstream)\nNRMSE: {metrics['NRMSE']:.1f}%, R²: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.3f}")
            else:
                ax.set_title('Jet Width (Downstream)')
        else:
            ax.set_title('Jet Width (Downstream) - No Sim Data')
            
    else:
        ax.set_title('Jet Width (Downstream)\n(No simulation data)')
    
    ax.set_xlabel('Axial position from expansion (mm)')
    ax.set_ylabel('Jet Width (mm)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Full jet width profile including upstream
    ax = axes[1]
    ax.plot(jet_width_exp[:, 0] * 1000, jet_width_exp[:, 1] * 1000, 
            'ko-', markersize=6, label='Experiment (PIV)')
    
    if len(sim_z) > 0:
        ax.plot(sim_z * 1000, sim_jet_width * 1000, 
                'bs-', markersize=6, linewidth=2, label='OpenFOAM')
        
        # Calculate full metrics
        metrics_full = calculate_error_metrics(
            jet_width_exp[:, 0] * 1000, jet_width_exp[:, 1] * 1000,
            sim_z * 1000, sim_jet_width * 1000
        )
        if metrics_full:
            ax.set_title(f"Jet Width (Full Profile)\nNRMSE: {metrics_full['NRMSE']:.1f}%, R²: {metrics_full['R2']:.3f}, RMSE: {metrics_full['RMSE']:.3f}")
        else:
             ax.set_title('Jet Width (Full Profile)')

    else:
         ax.set_title('Jet Width (Full Profile)')

    ax.set_xlabel('Axial position from expansion (mm)')
    ax.set_ylabel('Jet Width (mm)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Expansion')
    
    plt.tight_layout()
    
    # Save figure
    if not paths['plots_dir'].exists():
        paths['plots_dir'].mkdir(parents=True, exist_ok=True)
        
    output_file = paths['plots_dir'] / "vv_jet_width.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("JET WIDTH ANALYSIS SUMMARY")
    print("="*50)
    print(f"{'z (mm)':<10} {'Exp (mm)':<12} {'Sim (mm)':<12} {'Error (%)':<12}")
    print("-"*50)
    
    for z, jw_sim in zip(sim_z, sim_jet_width):
        jw_exp = np.interp(z, jet_width_exp[:, 0], jet_width_exp[:, 1])
        err = abs(jw_sim - jw_exp) / jw_exp * 100 if jw_exp > 0 else 0
        print(f"{z*1000:<10.1f} {jw_exp*1000:<12.3f} {jw_sim*1000:<12.3f} {err:<12.1f}")
    
    print("="*50)
    
    plt.show()


if __name__ == "__main__":
    main()
