import numpy as np

def simulate_brownian_pursuit(T=20.0, dt=0.01, sigma=1.0, vp=1.5, obs_times=None):
    """
    Simulates a 2D Brownian target and a pursuer attempting to track it.
    """
    # Total number of steps
    N = int(T / dt)

    # Time grid from 0 to T with N points
    t_grid = np.linspace(0, T, N)

    # Generate 2D Brownian motion for the target: each step ~ N(0, sigma^2 dt)
    increments = np.sqrt(dt) * sigma * np.random.randn(N, 2)
    target = np.cumsum(increments, axis=0)      # Cumulative sum to get position
    target[0] = np.array([0.0, 0.0])             # Start at origin

    # Initialize pursuer trajectory, same shape as target
    pursuer = np.zeros_like(target)
    pursuer[0] = np.array([-10.0, 0.0])          # Pursuer starts to the left of origin

    # Ensure obs_times is a sorted numpy array of observation times
    obs_times = np.array(obs_times)
    obs_times = np.sort(obs_times)

    # Get indices in time grid corresponding to each observation time
    obs_indices = np.searchsorted(t_grid, obs_times)

    # Initialize last observed target position and tracking pointer
    last_obs_pos = target[0]
    next_obs_idx = 1 if len(obs_indices) > 1 else 0

    # Simulate pursuer's motion for each time step
    for k in range(N - 1):
        # Update the pursuer's knowledge of the target if an observation occurs
        if next_obs_idx < len(obs_indices) and k == obs_indices[next_obs_idx]:
            last_obs_pos = target[k]
            next_obs_idx += 1

        # Compute direction from pursuer to last observed target position
        vec = last_obs_pos - pursuer[k]
        dist = np.linalg.norm(vec)
        direction = vec / dist if dist > 1e-8 else np.zeros(2)

        # Move pursuer in that direction with speed vp
        pursuer[k + 1] = pursuer[k] + vp * dt * direction

    return t_grid, target, pursuer
