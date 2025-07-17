import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PursuitEnv(gym.Env):
    """
    Custom environment where a pursuer tracks a target moving in 2D.
    The pursuer observes and decides where to move based on RL policy.
    """
    def __init__(self, T=10.0, vp=4.5, sigma=1.5, max_obs=10, catch_radius=0.5, dt=0.005,
                 levy_walk=False, levy_mu=2.0, levy_scale=0.05):
        super().__init__()

        self.T = T
        self.vp = vp
        self.sigma = sigma
        self.max_obs = max_obs
        self.catch_radius = catch_radius
        self.dt = dt
        self.levy_walk = levy_walk
        self.levy_mu = levy_mu
        self.levy_scale = levy_scale

        self.max_wait = 2.0
        self.action_space = spaces.Box(low=0.0, high=self.max_wait, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*4 + [0, 0]),
            high=np.array([np.inf]*4 + [1, 1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        self.obs_left = self.max_obs
        self.pursuer_pos = np.array([-10.0, 0.0])
        self.target_pos = np.array([0.0, 0.0])
        self.last_obs_pos = self.target_pos.copy()
        self.done = False
        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            *self.pursuer_pos,
            *self.last_obs_pos,
            self.time / self.T,
            self.obs_left / self.max_obs
        ], dtype=np.float32)

    def _levy_step(self, dt):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform()
        step_length = self.levy_scale * (r ** (-1 / (self.levy_mu - 1))) * np.sqrt(dt)
        return step_length * np.array([np.cos(theta), np.sin(theta)])

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, False, {}

        wait_time = float(np.clip(action[0], 0, self.max_wait))
        wait_time = min(wait_time, self.T - self.time)
        steps = int(wait_time / self.dt)

        for _ in range(steps):
            d_target = self._levy_step(self.dt) if self.levy_walk else np.sqrt(self.dt) * self.sigma * np.random.randn(2)
            self.target_pos += d_target

            vec = self.last_obs_pos - self.pursuer_pos
            dist = np.linalg.norm(vec)
            direction = vec / dist if dist > 1e-8 else np.zeros(2)
            self.pursuer_pos += self.vp * self.dt * direction

        self.time += wait_time
        if self.obs_left > 0:
            self.last_obs_pos = self.target_pos.copy()
            self.obs_left -= 1

        terminated = self.time >= self.T or self.obs_left == 0
        reward = 1.0 if np.linalg.norm(self.target_pos - self.pursuer_pos) < self.catch_radius else -np.linalg.norm(self.target_pos - self.pursuer_pos)
        self.done = terminated

        return self._get_state(), reward, terminated, False, {}
