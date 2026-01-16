"""
qlbm_tools.py
shared utilities for qlbm simulation, zne mitigation, and visualization.
"""
import sys
from types import ModuleType

# --- critical patches (must be before mitiq import) ---
try:
    from numpy import RankWarning
except ImportError:
    class RankWarning(UserWarning): pass

poly_mod = ModuleType("numpy.lib.polynomial")
poly_mod.RankWarning = RankWarning
sys.modules["numpy.lib.polynomial"] = poly_mod
# -----------------------------------------------------

import json
import time
import numpy as np
import imageio
import pyvista as pv
import matplotlib.pyplot as plt
from os import listdir, path, makedirs
from PIL import Image, ImageDraw, ImageFont
from pyvista import themes

from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import RichardsonFactory, LinearFactory, ExpFactory
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
# ensure fakefez is available if requested, otherwise catch error
try:
    from qiskit_ibm_runtime.fake_provider import FakeFez
except ImportError:
    FakeFez = None

pv.set_plot_theme(themes.ParaViewTheme())

# --- backend monitoring & zne ---

class BackendSpy:
    """intercepts backend.run for progress tracking and zne scaling."""
    def __init__(self, backend, total_steps, zne_scales=None):
        self.original_run = backend.run
        self.total_steps = total_steps
        self.scales = zne_scales if zne_scales else [1.0]
        self.is_zne = zne_scales is not None
        self.history = []
        self.jobs_started = 0

    def __call__(self, circuits, **kwargs):
        base_circuit = circuits[0] if isinstance(circuits, list) else circuits
        
        # fold circuits if zne is active
        run_circuits = []
        for s in self.scales:
            if s > 1.0:
                run_circuits.append(fold_gates_at_random(base_circuit, s))
            else:
                run_circuits.append(base_circuit)
        
        self.jobs_started += 1
        print(f"step {self.jobs_started}/{self.total_steps} | submitted...", end="\r")
        
        job = self.original_run(run_circuits, **kwargs)
        return SpyJobWrapper(job, self, self.jobs_started)

class SpyJobWrapper:
    """wraps qiskit job to measure time and capture counts."""
    def __init__(self, real_job, spy_instance, step_idx):
        self.real_job = real_job
        self.spy = spy_instance
        self.step_idx = step_idx

    def result(self):
        start_time = time.time()
        full_result = self.real_job.result()
        duration = time.time() - start_time
        
        print(f"step {self.step_idx}/{self.spy.total_steps} | finished in {duration:.2f}s   ")
        
        all_counts = full_result.get_counts()
        if not isinstance(all_counts, list): all_counts = [all_counts]
        
        # map results to scales
        step_data = {}
        for i, s in enumerate(self.spy.scales):
            if i < len(all_counts):
                step_data[str(s)] = all_counts[i]
        
        self.spy.history.append(step_data if self.spy.is_zne else step_data.get("1.0"))
        return SpyResultWrapper(full_result)
    
    def circuits(self): return self.real_job.circuits()

class SpyResultWrapper:
    """wraps result to expose only scale 1.0 to qlbm logic."""
    def __init__(self, result): self._result = result
    def get_counts(self, experiment=None):
        if experiment is not None: return self._result.get_counts(experiment)
        counts = self._result.get_counts()
        return counts[0] if isinstance(counts, list) else counts
    def __getattr__(self, name): return getattr(self._result, name)

def attach_spy(backend, steps, zne_scales=None):
    """attaches spy in-place."""
    spy = BackendSpy(backend, steps, zne_scales)
    backend.run = spy
    backend.spy_instance = spy 
    return backend

# --- visualization ---

def save_vti(dense_vector, d, filename):
    grid_data = dense_vector.reshape((d, d), order='F')
    image = pv.ImageData(dimensions=(d, d, 1), spacing=(1, 1, 1), origin=(0, 0, 0))
    image.point_data["Scalars_"] = grid_data.flatten(order='F')
    image.set_active_scalars("Scalars_")
    if not path.exists(path.dirname(filename)):
        makedirs(path.dirname(filename))
    image.save(filename)

def create_gif(simdir, output_filename, fps=1):
    if not path.exists(simdir): return
    vti_files = sorted([path.join(simdir, f) for f in listdir(simdir) if f.endswith(".vti")])
    if not vti_files: return

    # find max scalar for consistant colorbar
    max_s = 0
    for f in vti_files:
        m = pv.read(f)
        if m.active_scalars is not None: max_s = max(max_s, m.active_scalars.max())

    images = []
    plotter = pv.Plotter(off_screen=True)
    
    for c, vti in enumerate(vti_files):
        plotter.clear()
        plotter.add_mesh(pv.read(vti), clim=[0, max_s], show_edges=True)
        plotter.view_xy()
        
        img = plotter.screenshot(transparent_background=True)
        
        # draw progress bar
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        bar_w = int(w * 0.8)
        bx, by = (w - bar_w) // 2, h - 40
        fill = int((c + 1) / len(vti_files) * bar_w)
        draw.rectangle([bx, by, bx + bar_w, by + 20], outline="black", width=3)
        draw.rectangle([bx, by, bx + fill, by + 20], fill="purple")
        
        images.append(np.array(pil_img))

    plotter.close()
    if not path.exists(path.dirname(output_filename)):
        makedirs(path.dirname(output_filename))
    imageio.mimsave(output_filename, images, fps=fps, loop=0)
    print(f"saved gif: {output_filename}")

def create_comparison_gif(dirs_dict, output_filename):
    """creates side-by-side gif from multiple result directories."""
    keys = list(dirs_dict.keys())
    file_lists = {k: sorted([path.join(v, f) for f in listdir(v) if f.endswith(".vti")]) 
                  for k, v in dirs_dict.items() if path.exists(v)}
    
    if not file_lists: return
    
    lengths = [len(v) for v in file_lists.values()]
    steps = min(lengths)
    
    # find global max for consistent coloring
    global_max = 0
    for flist in file_lists.values():
        for f in flist:
            m = pv.read(f)
            if m.active_scalars is not None: global_max = max(global_max, m.active_scalars.max())

    images = []
    plotter = pv.Plotter(shape=(1, len(keys)), off_screen=True, window_size=(400 * len(keys), 500))
    
    for i in range(steps):
        plotter.clear()
        for idx, label in enumerate(keys):
            plotter.subplot(0, idx)
            mesh = pv.read(file_lists[label][i])
            plotter.add_mesh(mesh, clim=[0, global_max])
            plotter.add_text(label, font_size=12, color="black", position="upper_left")
            plotter.view_xy()
            
        img = plotter.screenshot(transparent_background=True)
        images.append(np.array(Image.fromarray(img)))
        
    plotter.close()
    if not path.exists(path.dirname(output_filename)):
        makedirs(path.dirname(output_filename))
    imageio.mimsave(output_filename, images, fps=1, loop=0)
    print(f"saved comparison gif: {output_filename}")

def create_static_grid(dirs_dict, output_filename, title):
    """creates a static summary image of all steps."""
    keys = list(dirs_dict.keys())
    file_lists = {k: sorted([path.join(v, f) for f in listdir(v) if f.endswith(".vti")]) 
                  for k, v in dirs_dict.items() if path.exists(v)}
    
    if not file_lists: return

    global_max = 0
    for flist in file_lists.values():
        for f in flist:
            m = pv.read(f)
            if m.active_scalars is not None: global_max = max(global_max, m.active_scalars.max())

    cols = min(len(v) for v in file_lists.values())
    rows = len(keys)
    w_plot, h_plot = 240, 260
    
    plotter = pv.Plotter(shape=(rows, cols), off_screen=True, 
                         window_size=(cols * w_plot, rows * h_plot), border=False)

    for r, label in enumerate(keys):
        files = file_lists[label]
        for c in range(cols):
            plotter.subplot(r, c)
            mesh = pv.read(files[c])
            plotter.add_mesh(mesh, clim=[0, global_max], show_scalar_bar=False)
            plotter.view_xy()
            plotter.camera.zoom(1.36)

    plotter.set_background("white")
    img_array = plotter.screenshot()
    plotter.close()
    
    # add labels with pillow
    pil_img = Image.fromarray(img_array)
    l_margin, b_margin = 120, 100
    final_img = Image.new("RGB", (pil_img.width + l_margin, pil_img.height + b_margin), "white")
    final_img.paste(pil_img, (l_margin, 50))
    
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.load_default(34)
    except:
        font = None

    draw.text((final_img.width/2 - 200, 10), title, fill="black", font=font)

    for r, label in enumerate(keys):
        y = (r * h_plot) + (h_plot / 2) + 20
        draw.text((20, y), label, fill="black", font=font)

    for c in range(cols):
        x = l_margin + (c * w_plot) + (w_plot / 2) - 40
        y = pil_img.height + 50
        draw.text((x, y), f"step {c}", fill="black", font=font)

    if not path.exists(path.dirname(output_filename)):
        makedirs(path.dirname(output_filename))
    final_img.save(output_filename)
    print(f"saved static grid: {output_filename}")

# --- zne logic ---

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

def process_zne_data(zne_file, d, output_json, zne_scales, shots=1024):
    """extrapolates zne data."""
    if not path.exists(zne_file): return
    
    with open(zne_file, 'r') as f: history = json.load(f)
    
    mitigated_data = []
    factory = LinearFactory(zne_scales)
    
    out_dir = path.dirname(output_json)
    vti_dir = path.join(out_dir, "paraview-mitigated")
    
    print(f"mitigating {len(history)} steps with scales {zne_scales}...")

    for i, step in enumerate(history):
        # check if all scales exist in step data
        if not all(str(s) in step for s in zne_scales): continue
        
        counts_list = [step[str(s)] for s in zne_scales]
        n_q = len(next(iter(counts_list[0])))
        
        # first step usually has no noise, just use scale 1
        if i == 0:
            mit_vec = counts_to_prob(counts_list[0], n_q)
        else:
            y = np.vstack([counts_to_prob(c, n_q) for c in counts_list])
            mit_vec = np.zeros(2**n_q)
            active = np.where(np.sum(y, axis=0) > 0)[0]
            
            for idx in active:
                val = factory.extrapolate(zne_scales, y[:, idx])
                mit_vec[idx] = max(0.0, val)

        # save visualization
        mit_vec_scaled = mit_vec * shots
        save_vti(mit_vec_scaled, d, path.join(vti_dir, f"step_{i}.vti"))
        mitigated_data.append(vec_to_counts(mit_vec, n_q))

    with open(output_json, 'w') as f: json.dump(mitigated_data, f)
    create_gif(vti_dir, path.join(out_dir, "mitigated.gif"))

# --- metrics & noise ---

def calculate_rmse(ideal_file, test_file, d):
    with open(ideal_file, 'r') as f: ideal = json.load(f)
    with open(test_file, 'r') as f: test = json.load(f)
    
    rmse = []
    for i in range(min(len(ideal), len(test))):
        n_q = len(next(iter(ideal[i])))
        grid_i = counts_to_prob(ideal[i], n_q).reshape((d, d), order='F')
        grid_t = counts_to_prob(test[i], n_q).reshape((d, d), order='F')
        rmse.append(np.sqrt(np.mean((grid_i - grid_t)**2)))
    return rmse

def plot_errors(noisy_dict, zne_dict, output_file, title, legend_title="shots"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    
    cmap = plt.cm.viridis
    
    # helper to plot a dict on an axis
    def _plot_dict(ax, data_dict, title):
        if not data_dict: return
        keys = list(data_dict.keys())
        for i, (label, data) in enumerate(data_dict.items()):
            color = cmap(i / max(1, len(keys) - 1))
            clean_label = label.split('-')[1] if '-' in label else label
            inverse_label = f"1/{int(1/float(clean_label))}"
            ax.plot(data, label=inverse_label, color=color, marker='o', markersize=3)
        ax.set_title(title)
        ax.legend(title=legend_title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")
        ax.set_ylabel("rmse")

    _plot_dict(ax1, noisy_dict, "noisy baseline")
    _plot_dict(ax2, zne_dict, "zne mitigated")

    # sync y-axis limits
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"saved error plot: {output_file}")

def get_noise_model(name="depolarizing", p_err=0.001):
    if name.lower() == "fakefez" and FakeFez:
        return NoiseModel.from_backend(FakeFez())
    
    # uniform depolarizing noise
    noise_model = NoiseModel()
    error_1q = depolarizing_error(p_err * 0.1, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "rz", "sx", "x", "h"])
    error_2q = depolarizing_error(p_err, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "ecr"])
    error_meas = pauli_error([('X', 0.01), ('I', 0.99)])
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    return noise_model 