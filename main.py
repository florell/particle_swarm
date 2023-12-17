import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def objective_function(x):
    return 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2 # 13 вариант


def particle_swarm_optimization(objective_function, num_particles=10, num_dimensions=2, max_iter=10, c1=2.0, c2=2.0,
                                w=0.5):
    particles = np.random.rand(num_particles, num_dimensions)
    velocities = np.random.rand(num_particles, num_dimensions)
    personal_best_positions = particles.copy()
    personal_best_values = np.apply_along_axis(objective_function, 1, particles)
    global_best_position = particles[np.argmin(personal_best_values)]
    global_best_value = np.min(personal_best_values)

    solutions = []
    repeated_values = {}

    for iteration in range(max_iter):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * (
                global_best_position - particles)
        particles = particles + velocities

        current_values = np.apply_along_axis(objective_function, 1, particles)

        update_personal_best = current_values < personal_best_values
        personal_best_values[update_personal_best] = current_values[update_personal_best]
        personal_best_positions[update_personal_best] = particles[update_personal_best]

        min_idx = np.argmin(personal_best_values)
        if personal_best_values[min_idx] < global_best_value:
            global_best_value = personal_best_values[min_idx]
            global_best_position = personal_best_positions[min_idx]

        # Check for repeated values
        rounded_values = np.round(particles, decimals=3)
        unique_counts = np.unique(rounded_values, axis=0, return_counts=True)
        repeated_indices = np.where(unique_counts[1] >= 3)[0]
        for idx in repeated_indices:
            repeated_value = unique_counts[0][idx]
            repeated_values[repr(repeated_value)] = repeated_values.get(repr(repeated_value), 0) + 1

        if any(count >= 3 for count in repeated_values.values()):
            print("Optimization stopped due to repeated values.")
            break

        solutions.append({
            'Iteration': iteration,
            'Particles': particles.copy(),
            'Best Position': personal_best_positions[min_idx],
            'Best Value': personal_best_values[min_idx]
        })

        yield particles, global_best_position, global_best_value, solutions



class ParticleSwarmOptimizationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Particle Swarm Optimization")

        self.num_particles = 10
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.max_iter = 10
        self.current_iteration = 0

        self.create_widgets()

    def create_widgets(self):
        # Entry widgets for user input
        self.num_particles_label = tk.Label(self.master, text="Number of Particles:")
        self.num_particles_entry = tk.Entry(self.master, textvariable=tk.StringVar(value=self.num_particles))
        self.num_particles_label.pack()
        self.num_particles_entry.pack()

        self.c1_label = tk.Label(self.master, text="C1 (Personal Best Coefficient):")
        self.c1_entry = tk.Entry(self.master, textvariable=tk.StringVar(value=self.c1))
        self.c1_label.pack()
        self.c1_entry.pack()

        self.c2_label = tk.Label(self.master, text="C2 (Global Best Coefficient):")
        self.c2_entry = tk.Entry(self.master, textvariable=tk.StringVar(value=self.c2))
        self.c2_label.pack()
        self.c2_entry.pack()

        self.w_label = tk.Label(self.master, text="W (Velocity Coefficient):")
        self.w_entry = tk.Entry(self.master, textvariable=tk.StringVar(value=self.w))
        self.w_label.pack()
        self.w_entry.pack()

        self.max_iter_label = tk.Label(self.master, text="Max Iterations:")
        self.max_iter_entry = tk.Entry(self.master, textvariable=tk.StringVar(value=self.max_iter))
        self.max_iter_label.pack()
        self.max_iter_entry.pack()

        self.create_particles_button = tk.Button(self.master, text="Create Particles", command=self.create_particles)
        self.create_particles_button.pack()

        self.iterate_button = tk.Button(self.master, text="Iterate", command=self.iterate)
        self.iterate_button.pack()

        # Labels for displaying information
        self.best_position_label = tk.Label(self.master, text="Best Position:")
        self.best_position_label.pack()

        self.best_value_label = tk.Label(self.master, text="Best Value:")
        self.best_value_label.pack()

        # ScrolledText for displaying solutions
        self.solutions_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=40, height=10)
        self.solutions_text.pack()

        self.canvas = FigureCanvasTkAgg(self.plot(), master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def plot(self):
        fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = fig.add_subplot(1, 1, 1, projection='3d')

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)
        z = objective_function(np.array([x, y]))

        self.surface = self.ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), cmap='viridis', alpha=0.7)
        self.particles_scatter = self.ax.scatter([], [], [], c='r', marker='o')

        return fig

    def create_particles(self):
        self.num_particles = int(self.num_particles_entry.get())
        self.c1 = float(self.c1_entry.get())
        self.c2 = float(self.c2_entry.get())
        self.w = float(self.w_entry.get())
        self.max_iter = int(self.max_iter_entry.get())
        self.current_iteration = 0

        self.pso_generator = particle_swarm_optimization(
            objective_function, num_particles=self.num_particles, c1=self.c1, c2=self.c2, w=self.w, max_iter=self.max_iter)

    def iterate(self):
        try:
            particles, global_best_position, global_best_value, solutions = next(self.pso_generator)
            self.current_iteration += 1

            # Update plot
            x = particles[:, 0]
            y = particles[:, 1]
            z = np.apply_along_axis(objective_function, 1, particles)

            self.surface.remove()
            self.particles_scatter.remove()

            self.surface = self.ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), cmap='viridis', alpha=0.7)
            self.particles_scatter = self.ax.scatter(x, y, z, c='r', marker='o')

            # Display information
            best_position_str = "Best Position: {}".format(global_best_position)
            best_value_str = "Best Value: {}".format(global_best_value)

            self.best_position_label.config(text=best_position_str)
            self.best_value_label.config(text=best_value_str)

            # Update solutions text
            solutions_str = "\n".join([str(sol) for sol in solutions])
            self.solutions_text.delete(1.0, tk.END)
            self.solutions_text.insert(tk.END, solutions_str)

            self.master.update_idletasks()
        except StopIteration:
            print("Optimization complete.")


root = tk.Tk()
app = ParticleSwarmOptimizationGUI(root)
root.mainloop()
