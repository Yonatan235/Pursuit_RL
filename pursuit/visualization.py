import matplotlib.pyplot as plt

def plot_simulation(t_grid, target, pursuer, obs_times, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(target[:, 0], target[:, 1], 'r-', label='Target')
    ax.plot(pursuer[:, 0], pursuer[:, 1], 'b-', label='Pursuer')
    ax.scatter(*target[0], color='r', marker='o', label='Target start')
    ax.scatter(*pursuer[0], color='b', marker='o', label='Pursuer start')
    ax.scatter(*target[-1], color='k', marker='*', s=100, label='Target end')

    obs_positions = target[np.searchsorted(t_grid, obs_times)]
    ax.scatter(obs_positions[:, 0], obs_positions[:, 1], color='g', marker='x', label='Observations')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
