from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_aer.noise import NoiseModel
from qlbm.components import CQLBM, GridMeasurement, MSInitialConditions, EmptyPrimitive
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.tools.utils import create_directory_and_parents
import json

# Import our custom toolkit
import qlbm_tools as tools

def run_simulation(d, shots, steps, mode="ideal"):
    """
    Modes: 'ideal', 'noisy', 'zne'
    """
    print(f"\n=== Running Simulation: {mode.upper()} ===")
    
    # 1. Setup Lattice
    lattice = MSLattice({
        "lattice": {"dim": {"x": d[0], "y": d[1]}, "velocities": {"x": 4, "y": 4}},
        "geometry": [] 
    })

    # 2. Configure Backend & Noise
    output_key = f"ms-{d[0]}x{d[1]}"
    if mode == "ideal":
        backend = AerSimulator(method='automatic')
        out_dir = f"qlbm-output/{output_key}-ideal"
    else:
        noise_model = tools.custom_noise_model()
        #noise_model = NoiseModel.from_backend(FakeFez())
        backend = AerSimulator(noise_model=noise_model, method='automatic')
        out_dir = f"qlbm-output/{output_key}-{'noisy' if mode == 'noisy' else 'zne'}"

    create_directory_and_parents(out_dir)

    # 3. Attach Spy (Monitoring or ZNE)
    is_zne = (mode == "zne")
    backend = tools.attach_spy(backend, steps+1, zne=is_zne)

    # 4. QLBM Configuration
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

    # 5. Run
    runner = QiskitRunner(cfg, lattice)
    runner.run(steps, shots, out_dir, statevector_snapshots=False)

    # 6. Save Data & Animation
    raw_file = f"raw_counts/{output_key}_{mode}.json"
    create_directory_and_parents("raw_counts")
    
    # Extract data from spy
    with open(raw_file, 'w') as f:
        json.dump(backend.spy_instance.history, f)
    
    # Animation
    tools.create_gif(f"{out_dir}/paraview", f"{out_dir}/simulation.gif")
    
    return raw_file

if __name__ == "__main__":
    # --- CONFIGURATION ---
    dims = (4, 4)
    shots = 2**11
    steps = 3
    
    # --- EXECUTION ---
    
    # 1. Run Ideal (Reference)
    file_ideal = run_simulation(dims, shots, steps, mode="ideal")
    
    # 2. Run Noisy (Baseline)
    file_noisy = run_simulation(dims, shots, steps, mode="noisy")
    
    # 3. Run ZNE (Mitigation)
    file_zne_raw = run_simulation(dims, shots, steps, mode="zne")
    
    # --- ANALYSIS ---
    
    # 4. Mitigate ZNE Data
    print("\n=== Mitigating ZNE Data ===")
    file_mitigated = f"raw_counts/ms_{dims[0]}x{dims[1]}_mitigated.json"
    tools.process_zne_data(file_zne_raw, dims, file_mitigated, shots=shots)
    
    # 5. Compare & Plot
    print("\n=== Calculating Errors ===")
    rmse_noisy = tools.calculate_rmse(file_ideal, file_noisy, dims)
    rmse_zne = tools.calculate_rmse(file_ideal, file_mitigated, dims)
    
    print(f"Final RMSE (Noisy): {rmse_noisy[-1]:.4f}")
    print(f"Final RMSE (ZNE):   {rmse_zne[-1]:.4f}")
    
    tools.plot_errors({
        "Noisy Baseline": rmse_noisy, 
        "ZNE Mitigated": rmse_zne
    })

    # 6. Generate Side-by-Side Comparison
    print("\n=== Generating Comparison GIF ===")
    
    # define paths to the 'paraview' folders where .vti files live
    dirs = {
        "Ideal": f"qlbm-output/ms-{dims[0]}x{dims[1]}-ideal/paraview",
        "Noisy": f"qlbm-output/ms-{dims[0]}x{dims[1]}-noisy/paraview",
        "ZNE": f"raw_counts/ms-{dims[0]}x{dims[1]}-mitigated_vti"
    }
    
    tools.create_comparison_gif(dirs, f"animations/{dims[0]}x{dims[1]}_comparison.gif")
