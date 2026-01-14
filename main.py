from qiskit_aer import AerSimulator
from qlbm.components import CQLBM, GridMeasurement, MSInitialConditions, EmptyPrimitive
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.tools.utils import create_directory_and_parents
import json
import os

import qlbm_tools as tools

# --- configuration parameters ---
PARAMS = {
    "d": 4,
    "shots": [2**8, 2**10, 2**12],
    "steps": 4,
    "noise_model": "fakefez", # "depolarizing" or "fakefez"
    "zne_scales": [1.0, 3.0, 5.0],
    #"p_err_values": [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01]
    "p_err_values": [2**-12, 2**-11, 2**-10, 2**-9, 2**-8]
}

def run_simulation(p, current_shots, mode="ideal", base_folder="results", p_err=None):
    """
    runs a single qlbm simulation instance.
    mode: 'ideal', 'noisy', 'zne'
    p_err: optional override for error probability (if None, defaults to 0.001)
    """
    p_info = f" | p_err={p_err}" if p_err else ""
    print(f"\n=== running simulation: {mode.upper()} (shots={current_shots}{p_info}) ===")
    
    # setup paths
    out_dir = os.path.join(base_folder, mode)
    create_directory_and_parents(out_dir)

    # setup lattice
    lattice = MSLattice({
        "lattice": {"dim": {"x": p['d'], "y": p['d']}, "velocities": {"x": 4, "y": 4}},
        "geometry": [] 
    })

    # configure backend
    if mode == "ideal":
        backend = AerSimulator(method='automatic')
    else:
        # use provided p_err if active, otherwise use default
        noise_val = p_err if p_err is not None else 0.001
        noise_model = tools.get_noise_model(p['noise_model'], noise_val)
        backend = AerSimulator(noise_model=noise_model, method='automatic')

    # attach spy (monitoring + zne folding)
    is_zne = (mode == "zne")
    scales = p['zne_scales'] if is_zne else None
    backend = tools.attach_spy(backend, p['steps'] + 1, zne_scales=scales)

    # qlbm config
    cfg = SimulationConfig(
        initial_conditions=MSInitialConditions(lattice),
        algorithm=CQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=GridMeasurement(lattice),
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        statevector_sampling=False,
        execution_backend=backend,
        sampling_backend=backend
    )
    cfg.prepare_for_simulation()

    # run simulation
    runner = QiskitRunner(cfg, lattice)
    runner.run(p['steps'], current_shots, out_dir, statevector_snapshots=False)

    # save raw data
    raw_file = os.path.join(out_dir, f"counts_{mode}.json")
    with open(raw_file, 'w') as f:
        json.dump(backend.spy_instance.history, f)
    
    # generate standard visuals
    vti_dir = os.path.join(out_dir, "paraview")
    tools.create_gif(vti_dir, os.path.join(out_dir, "simulation.gif"))
    
    return raw_file, vti_dir

def main():
    p = PARAMS
    shots_list = p["shots"]
    p_err_list = p["p_err_values"]
    

    # sweep over shots
    print(">>> starting shots sweep...")
    all_rmse_noisy_shots = {}
    all_rmse_zne_shots = {}
    
    folder_name = f"{p['d']}x{p['d']}_steps{p['steps']}_{p['noise_model']}"
    
    for shots in shots_list:
        base_folder = os.path.join("sim_results", folder_name, "sweep_shots", f"shots_{shots}")
        
        file_ideal, dir_ideal = run_simulation(p, shots, "ideal", base_folder)
        file_noisy, dir_noisy = run_simulation(p, shots, "noisy", base_folder)
        file_zne_raw, _ = run_simulation(p, shots, "zne", base_folder)
        
        print(f"\n=== mitigating zne data (shots={shots}) ===")
        file_mitigated = os.path.join(base_folder, "zne", "counts_mitigated.json")
        tools.process_zne_data(file_zne_raw, p['d'], file_mitigated, p['zne_scales'], shots)
        dir_mitigated = os.path.join(base_folder, "zne", "paraview-mitigated")
        
        rmse_noisy = tools.calculate_rmse(file_ideal, file_noisy, p['d'])
        rmse_zne = tools.calculate_rmse(file_ideal, file_mitigated, p['d'])
        
        all_rmse_noisy_shots[f"shots-{shots}"] = rmse_noisy
        all_rmse_zne_shots[f"shots-{shots}"] = rmse_zne

        # combined visuals for this run
        vis_dirs = {"ideal": dir_ideal, "noisy": dir_noisy, "zne": dir_mitigated}
        tools.create_comparison_gif(vis_dirs, os.path.join(base_folder, "comparison.gif"))
        tools.create_static_grid(vis_dirs, os.path.join(base_folder, "static_grid.png"), f"noise=default | {shots} shots")

    # plot shots comparison
    summary_folder = os.path.join("sim_results", folder_name)
    tools.plot_errors(
        all_rmse_noisy_shots, 
        all_rmse_zne_shots,
        os.path.join(summary_folder, "rmse_vs_shots.png"),
        f"fakefez noise",
        legend_title="shots"
    )


    # sweep over p_err
    fixed_shots = shots_list[0]
    print(f"\n>>> starting p_err sweep (fixed shots={fixed_shots})...")
    
    all_rmse_noisy_perr = {}
    all_rmse_zne_perr = {}

    base_folder_root = os.path.join("sim_results", folder_name, "sweep_perr")
    
    # run ideal once
    file_ideal, dir_ideal = run_simulation(p, fixed_shots, "ideal", os.path.join(base_folder_root, "reference"))

    for p_err in p_err_list:
        sub_folder = os.path.join(base_folder_root, f"perr_{p_err}")
        
        file_noisy, dir_noisy = run_simulation(p, fixed_shots, "noisy", sub_folder, p_err=p_err)
        file_zne_raw, _ = run_simulation(p, fixed_shots, "zne", sub_folder, p_err=p_err)
        
        print(f"\n=== mitigating zne data (p_err={p_err}) ===")
        file_mitigated = os.path.join(sub_folder, "zne", "counts_mitigated.json")
        tools.process_zne_data(file_zne_raw, p['d'], file_mitigated, p['zne_scales'], fixed_shots)
        
        rmse_noisy = tools.calculate_rmse(file_ideal, file_noisy, p['d'])
        rmse_zne = tools.calculate_rmse(file_ideal, file_mitigated, p['d'])
        
        all_rmse_noisy_perr[f"perr-{p_err}"] = rmse_noisy
        all_rmse_zne_perr[f"perr-{p_err}"] = rmse_zne
        
        print(f"p_err={p_err} | rmse(noisy)={rmse_noisy[-1]:.4f} | rmse(zne)={rmse_zne[-1]:.4f}")

    # plot p_err comparison
    tools.plot_errors(
        all_rmse_noisy_perr, 
        all_rmse_zne_perr,
        os.path.join(summary_folder, "rmse_vs_perr_richardson.png"),
        f"depolarizing noise  |  {fixed_shots} shots  | Richardson interpolation",
        legend_title="p_err"
    )

if __name__ == "__main__":
    main()