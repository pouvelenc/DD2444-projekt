"""
qlbm_tools.py
Shared utilities for QLBM simulation, ZNE mitigation, and visualization.
"""
import sys
from types import ModuleType

# --- 1. CRITICAL PATCHES (MUST BE BEFORE MITIQ IMPORT) ---
# Mitiq issues a legacy import that fails on newer NumPy versions.
# We must inject a fake module into sys.modules to intercept this.
try:
    from numpy import RankWarning
except ImportError:
    class RankWarning(UserWarning): pass

# Create a dummy module "numpy.lib.polynomial"
poly_mod = ModuleType("numpy.lib.polynomial")
poly_mod.RankWarning = RankWarning

# Inject it so mitiq finds it immediately upon import
sys.modules["numpy.lib.polynomial"] = poly_mod
# ---------------------------------------------------------

import json
import time
import numpy as np
import imageio
import pyvista as pv
import matplotlib.pyplot as plt
from os import listdir, path
from PIL import Image, ImageDraw
from pyvista import themes

# Now it is safe to import mitiq
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import RichardsonFactory
from qiskit_aer.noise import NoiseModel, pauli_error

# Set theme globally on import
pv.set_plot_theme(themes.ParaViewTheme())


# --- 2. Backend Spies (ZNE & Progress) ---
class BackendSpy:
    """Wraps a backend to intercept job execution for progress tracking or ZNE."""
    def __init__(self, backend, total_steps, zne_scales=None):
        self.original_run = backend.run
        self.total_steps = total_steps
        self.scales = zne_scales if zne_scales else [1.0]
        self.is_zne = zne_scales is not None
        self.history = []  # Stores raw counts
        self.jobs_started = 0

    def __call__(self, circuits, **kwargs):
        """The function that replaces backend.run"""
        # Handle input (QLBM sends single circuit, Qiskit expects list)
        base_circuit = circuits[0] if isinstance(circuits, list) else circuits
        
        # Prepare circuits (Fold if ZNE)
        run_circuits = []
        for s in self.scales:
            if s > 1.0:
                run_circuits.append(fold_gates_at_random(base_circuit, s))
            else:
                run_circuits.append(base_circuit)
        
        # Run on actual backend
        self.jobs_started += 1
        current_step = self.jobs_started
        
        print(f"Step {current_step}/{self.total_steps} | Submitted...", end="\r")
        job = self.original_run(run_circuits, **kwargs)

        # Wrap result to track when it actually finishes
        return SpyJobWrapper(job, self, current_step)

class SpyJobWrapper:
    """Wraps the Qiskit Job to capture results and measure execution time."""
    def __init__(self, real_job, spy_instance, step_idx):
        self.real_job = real_job
        self.spy = spy_instance
        self.step_idx = step_idx

    def result(self):
        # Measure the specific execution time of this step
        start_time = time.time()
        full_result = self.real_job.result() # This blocks until finished
        duration = time.time() - start_time
        
        # Print completion message
        print(f"Step {self.step_idx}/{self.spy.total_steps} | Finished in {duration:.2f}s   ")
        
        # Save data to history
        all_counts = full_result.get_counts()
        if not isinstance(all_counts, list): all_counts = [all_counts]
        
        step_data = {}
        for i, s in enumerate(self.spy.scales):
            if i < len(all_counts):
                step_data[str(s)] = all_counts[i]
        
        # For ZNE, save the dict of scales. For normal, just save the counts.
        self.spy.history.append(step_data if self.spy.is_zne else step_data.get("1.0"))
        
        # Return wrapper to QLBM
        return SpyResultWrapper(full_result)
    
    def circuits(self): return self.real_job.circuits()

class SpyResultWrapper:
    """Wraps the Qiskit Result to return only the scale 1.0 counts."""
    def __init__(self, result): self._result = result
    def get_counts(self, experiment=None):
        if experiment is not None: return self._result.get_counts(experiment)
        counts = self._result.get_counts()
        return counts[0] if isinstance(counts, list) else counts
    def __getattr__(self, name): return getattr(self._result, name)

def attach_spy(backend, steps, zne=False):
    """Attaches the spy to the backend in-place."""
    scales = [1.0, 3.0, 5.0] if zne else None
    spy = BackendSpy(backend, steps, scales)
    backend.run = spy
    backend.spy_instance = spy 
    return backend

# --- 3. Visualization (VTI & GIF) ---
def save_vti(dense_vector, d, filename):
    """Saves a density vector to a .vti file."""
    grid_data = dense_vector.reshape((d[0], d[1]), order='F') # Fortran order
    image = pv.ImageData(dimensions=(d[0], d[1], 1), spacing=(1, 1, 1), origin=(0, 0, 0))
    image.point_data["Scalars_"] = grid_data.flatten(order='F')
    image.set_active_scalars("Scalars_")
    image.save(filename)

def create_gif(simdir, output_filename):
    """Generates a GIF from .vti files in simdir."""
    print(f"Generating GIF: {output_filename}...")
    vti_files = sorted([path.join(simdir, f) for f in listdir(simdir) if f.endswith(".vti")])
    if not vti_files: return

    # Find global max scalar
    max_s = 0
    for f in vti_files:
        m = pv.read(f)
        if m.active_scalars is not None: max_s = max(max_s, m.active_scalars.max())

    images = []
    sargs = dict(title="Density", title_font_size=20, label_font_size=16, 
                 shadow=True, n_labels=3, fmt="%.1f", font_family="arial", position_x=0.2, position_y=0.05)

    plotter = pv.Plotter(off_screen=True)
    
    # Load background mesh if exists
    stl_files = [path.join(simdir, f) for f in listdir(simdir) if f.endswith(".stl")]
    bg_mesh = pv.read(stl_files) if stl_files else None

    for c, vti in enumerate(vti_files):
        plotter.clear()
        plotter.add_mesh(pv.read(vti), clim=[0, max_s], show_edges=True, scalar_bar_args=sargs)
        if bg_mesh: plotter.add_mesh(bg_mesh, show_scalar_bar=False)
        plotter.view_xy()
        
        img = plotter.screenshot(transparent_background=True)
        
        # Add progress bar
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        bar_w, bar_h = int(w * 0.8), 20
        bx, by = (w - bar_w) // 2, h - 40
        fill = int((c + 1) / len(vti_files) * bar_w)
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], outline="black", width=3)
        draw.rectangle([bx, by, bx + fill, by + bar_h], fill="purple")
        
        images.append(np.array(pil_img))

    plotter.close()
    imageio.mimsave(output_filename, images, fps=1, loop=0)

def create_comparison_gif(dirs_dict, output_filename):
    """
    Creates a side-by-side GIF comparison.
    dirs_dict: {"Ideal": path1, "Noisy": path2, "ZNE": path3}
    """
    print(f"Generating Comparison GIF: {output_filename}...")
    
    # 1. get file lists
    keys = list(dirs_dict.keys())
    file_lists = {}
    lengths = []
    
    global_max = 0
    
    for label, folder in dirs_dict.items():
        files = sorted([path.join(folder, f) for f in listdir(folder) if f.endswith(".vti")])
        file_lists[label] = files
        lengths.append(len(files))
        
        # update global max scalar for consistent colorbar
        for f in files:
            m = pv.read(f)
            # FORCE VALIDATION of active scalars
            if m.active_scalars is None and m.n_arrays > 0:
                m.set_active_scalars(m.array_names[0])
                
            if m.active_scalars is not None: 
                global_max = max(global_max, m.active_scalars.max())

    if not all(x == lengths[0] for x in lengths):
        print("Warning: Simulation lengths differ. Using minimum length.")
    
    steps = min(lengths)
    images = []
    
    # settings
    sargs = dict(
        height=0.1, width=0.5, position_x=0.25, position_y=0.05,
        title_font_size=20, label_font_size=16, fmt="%.2f", color="black"
    )

    # 2. render frame loop
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1200, 500))
    
    for i in range(steps):
        plotter.clear()
        
        for idx, label in enumerate(keys):
            plotter.subplot(0, idx)
            vti_path = file_lists[label][i]
            
            mesh = pv.read(vti_path)
            
            # CRITICAL FIX: Re-assert active scalars immediately after read
            if mesh.active_scalars is None and mesh.n_arrays > 0:
                mesh.set_active_scalars(mesh.array_names[0])
            
            # If still None, something is wrong with the file, skip coloring
            if mesh.active_scalars is not None:
                plotter.add_mesh(mesh, clim=[0, global_max], scalar_bar_args=sargs)
            else:
                # Fallback for empty/broken meshes (shouldn't happen with valid VTI)
                plotter.add_mesh(mesh, color="white")

            plotter.add_text(label, font_size=14, color="black", position="upper_left")
            plotter.view_xy()
            
        img = plotter.screenshot(transparent_background=True)

        # add progress bar to combined image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        bar_w = int(w * 0.9)
        bx = (w - bar_w) // 2
        by = h - 20
        fill = int((i + 1) / steps * bar_w)
        
        draw.rectangle([bx, by, bx + bar_w, by + 10], outline="black", width=2)
        draw.rectangle([bx, by, bx + fill, by + 10], fill="purple")
        
        images.append(np.array(pil_img))
        
    plotter.close()
    imageio.mimsave(output_filename, images, fps=1, loop=0)

# --- 4. ZNE Mitigation Logic ---
def counts_to_prob(counts, n_qubits):
    vec = np.zeros(2**n_qubits)
    total = sum(counts.values())
    if total == 0: return vec
    for b, c in counts.items():
        idx = int(b, 2)
        if idx < len(vec): vec[idx] = c / total
    return vec

def vec_to_counts(vec, n_qubits):
    return {format(i, f'0{n_qubits}b'): val for i, val in enumerate(vec) if val > 1e-6}

def process_zne_data(zne_file, d, output_json, shots=2**10):
    """Extrapolates ZNE data and saves mitigated JSON + VTIs."""
    if not path.exists(zne_file): return
    
    with open(zne_file, 'r') as f: history = json.load(f)
    
    mitigated_data = []
    factory = RichardsonFactory([1.0, 3.0, 5.0])
    
    output_dir = path.dirname(output_json)
    vti_dir = path.join(output_dir, f"ms-{d[0]}x{d[1]}-mitigated_vti")
    if not path.exists(vti_dir): 
        from qlbm.tools.utils import create_directory_and_parents
        create_directory_and_parents(vti_dir)

    print(f"Mitigating {len(history)} steps (scaling to {shots} shots)...")

    for i, step in enumerate(history):
        c1, c3, c5 = step.get("1.0", {}), step.get("3.0", {}), step.get("5.0", {})
        if not c1: continue
        
        n_q = len(next(iter(c1)))
        
        if i == 0:
            mit_vec = counts_to_prob(c1, n_q)
        else:
            y = np.vstack([counts_to_prob(c, n_q) for c in [c1, c3, c5]])
            mit_vec = np.zeros(2**n_q)
            active = np.where(np.sum(y, axis=0) > 0)[0]
            for idx in active:
                val = factory.extrapolate([1.0, 3.0, 5.0], y[:, idx])
                mit_vec[idx] = max(0.0, val)

        mit_vec_counts = mit_vec * shots
        
        save_vti(mit_vec_counts, d, path.join(vti_dir, f"step_{i}.vti"))
        mitigated_data.append(vec_to_counts(mit_vec, n_q))

    with open(output_json, 'w') as f: json.dump(mitigated_data, f)
    create_gif(vti_dir, path.join(output_dir, "mitigated.gif"))

# --- 5. Error Metrics ---
def calculate_rmse(ideal_file, test_file, d):
    with open(ideal_file, 'r') as f: ideal = json.load(f)
    with open(test_file, 'r') as f: test = json.load(f)
    
    rmse = []
    for i in range(min(len(ideal), len(test))):
        n_q = len(next(iter(ideal[i])))
        grid_i = counts_to_prob(ideal[i], n_q).reshape(d, order='F')
        grid_t = counts_to_prob(test[i], n_q).reshape(d, order='F')
        rmse.append(np.sqrt(np.mean((grid_i - grid_t)**2)))
    return rmse

def plot_errors(errors_dict):
    plt.figure(figsize=(10, 6))
    for name, data in errors_dict.items():
        plt.plot(data, label=name, marker='o' if 'ZNE' not in name else 'x')
    plt.xlabel("Step"); plt.ylabel("RMSE"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

# --- 6. Noise Model ---
def custom_noise_model():
    # Example error probabilities
    p_reset = 0.0003
    p_meas = 0.001
    p_gate1 = 0.0005

    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
    
    return noise_bit_flip
