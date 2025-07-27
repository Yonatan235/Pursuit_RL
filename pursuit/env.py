import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PursuitEnv(gym.Env):
    """
    Reinforcement learning environment for a 2D pursuit problem. A pursuer tries to catch a stochastically moving target..
    The RL agent picks the pursuer's observation and decision time with a fixed budget of observations.
    """

    def __init__(self, T=10.0, vp=1.5, sigma=0.5, max_obs=10, catch_radius=0.25, dt=0.005,
                 levy_walk=False, levy_mu=2.0, levy_scale=0.05):
        super().__init__()

        # Total time allowed for pursuit
        self.T = T

        # Pursuer's constant speed
        self.vp = vp

        # Noise level for target's Brownian motion
        self.sigma = sigma

        # Number of observations the pursuer can make of the target's position
        self.max_obs = max_obs

        # Distance threshold to count as a successful catch
        self.catch_radius = catch_radius

        # Time step for simulation
        self.dt = dt

        # If True, use Lévy flight for target motion
        self.levy_walk = levy_walk
        self.levy_mu = levy_mu          # exponent for Lévy distribution
        self.levy_scale = levy_scale    # scale factor for Lévy distribution

        # Maximum time the agent is allowed to wait before the next move
        self.max_wait = 2.0

        # Action space: scalar value for how long to wait before next move
        self.action_space = spaces.Box(low=0.0, high=self.max_wait, shape=(1,), dtype=np.float32)

        # Observation space:
        # [pursuer_x, pursuer_y, last_obs_x, last_obs_y, normalized_time, normalized_obs_left]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*4 + [0, 0]),
            high=np.array([np.inf]*4 + [1, 1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        # Reset the environment to the initial state
        super().reset(seed=seed)
        self.time = 0.0
        self.obs_left = self.max_obs
        self.pursuer_pos = np.array([-10.0, 0.0])   # Start far from target
        self.target_pos = np.array([0.0, 0.0])      # Target starts at origin
        self.last_obs_pos = self.target_pos.copy() # Last known target position
        self.done = False
        return self._get_state(), {}

    def _get_state(self):
        # Construct and return the observation vector
        return np.array([
            *self.pursuer_pos,
            *self.last_obs_pos,
            self.time / self.T,          # normalize time
            self.obs_left / self.max_obs # normalize remaining observations
        ], dtype=np.float32)

    def _levy_step(self, dt):
        # Generate a 2D Lévy-distributed step
        theta = np.random.uniform(0, 2 * np.pi)  # Random direction
        r = np.random.uniform()                 # Uniform draw to get heavy-tailed step
        step_length = self.levy_scale * (r ** (-1 / (self.levy_mu - 1))) * np.sqrt(dt)
        return step_length * np.array([np.cos(theta), np.sin(theta)])

    def step(self, action):
        # Perform a step given the action (how long to wait before next move)
        if self.done:
            return self._get_state(), 0.0, True, False, {}

        # Clip action to allowed wait time
        wait_time = float(np.clip(action[0], 0, self.max_wait))
        wait_time = min(wait_time, self.T - self.time)
        steps = int(wait_time / self.dt)

        for _ in range(steps):
            # Move the target
            if self.levy_walk:
                d_target = self._levy_step(self.dt)
            else:
                d_target = np.sqrt(self.dt) * self.sigma * np.random.randn(2)
            self.target_pos += d_target

            # Move the pursuer toward the last observed position
            vec = self.last_obs_pos - self.pursuer_pos
            dist = np.linalg.norm(vec)
            direction = vec / dist if dist > 1e-8 else np.zeros(2)
            self.pursuer_pos += self.vp * self.dt * direction

        self.time += wait_time

        # If observation tokens are left, update last known target position
        if self.obs_left > 0:
            self.last_obs_pos = self.target_pos.copy()
            self.obs_left -= 1

        # Determine if episode has ended
        terminated = self.time >= self.T or self.obs_left == 0
        self.done = terminated

        # Reward: +1 if target is within catch radius, else negative distance
        distance = np.linalg.norm(self.target_pos - self.pursuer_pos)
        reward = 1.0 if distance < self.catch_radius else -distance

        return self._get_state(), reward, terminated, False, {}
