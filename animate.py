import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from pursuit.env import PursuitEnv
import numpy as np
import os

# Load trained model and create environment
model = PPO.load("models/ppo_pursuit")
env = PursuitEnv()

obs, _ = env.reset()

# Initialize lists to store positions
pursuer_path = [env.pursuer_pos.copy()]  # Include initial position
target_path = [env.target_pos.copy()]

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    # Append new positions
    pursuer_path.append(env.pursuer_pos.copy())
    target_path.append(env.target_pos.copy())

# Convert to numpy arrays
pursuer_path = np.array(pursuer_path)
target_path = np.array(target_path)

# Setup plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_title("RL Pursuer Chasing Target")

p_line, = ax.plot([], [], 'b-', label="Pursuer")
t_line, = ax.plot([], [], 'r-', label="Target")
p_dot, = ax.plot([], [], 'bo')
t_dot, = ax.plot([], [], 'ro')
ax.legend()

def update(frame):
    p_line.set_data(pursuer_path[:frame+1, 0], pursuer_path[:frame+1, 1])
    t_line.set_data(target_path[:frame+1, 0], target_path[:frame+1, 1])
    p_dot.set_data([pursuer_path[frame, 0]], [pursuer_path[frame, 1]])
    t_dot.set_data([target_path[frame, 0]], [target_path[frame, 1]])
    return p_line, t_line, p_dot, t_dot

ani = FuncAnimation(fig, update, frames=len(pursuer_path), blit=True, interval=20)

os.makedirs("animations", exist_ok=True)
ani.save("animations/rl_pursuit.gif", writer=PillowWriter(fps=20))
