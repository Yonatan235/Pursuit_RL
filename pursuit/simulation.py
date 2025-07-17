import numpy as np

def simulate_brownian_pursuit(T=20.0, dt=0.01, sigma=1.0, vp=1.5, obs_times=None):
    N = int(T / dt)
    t_grid = np.linspace(0, T, N)
    
    increments = np.sqrt(dt) * sigma * np.random.randn(N, 2)
    target = np.cumsum(increments, axis=0)
    target[0] = np.array([0.0, 0.0])
    
    pursuer = np.zeros_like(target)
    pursuer[0] = np.array([-10.0, 0.0])
    
    obs_times = np.array(obs_times)
    obs_times = np.sort(obs_times)
    obs_indices = np.searchsorted(t_grid, obs_times)

    last_obs_pos = target[0]
    next_obs_idx = 1 if len(obs_indices) > 1 else 0

    for k in range(N-1):
        if next_obs_idx < len(obs_indices) and k == obs_indices[next_obs_idx]:
            last_obs_pos = target[k]
            next_obs_idx += 1

        vec = last_obs_pos - pursuer[k]
        dist = np.linalg.norm(vec)
        direction = vec / dist if dist > 1e-8 else np.zeros(2)
        pursuer[k+1] = pursuer[k] + vp * dt * direction

    return t_grid, target, pursuer
