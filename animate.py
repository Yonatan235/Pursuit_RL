import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from pursuit.env import PursuitEnv
import numpy as np

model = PPO.load("models/ppo_pursuit")
env = PursuitEnv()
env.record_trajectory = True
obs, _ = env.reset()

pursuer_path, target_path = [], []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    pursuer_path.extend(info.get("dt_pursuer_traj", []))
    target_path.extend(info.get("dt_target_traj", []))

pursuer_path, target_path = map(np.array, (pursuer_path, target_path))

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
    p_dot.set_data(pursuer_path[frame, 0], pursuer_path[frame, 1])
    t_dot.set_data(target_path[frame, 0], target_path[frame, 1])
    return p_line, t_line, p_dot, t_dot

ani = FuncAnimation(fig, update, frames=len(pursuer_path), blit=True, interval=20)
ani.save("animations/rl_pursuit.gif", writer=PillowWriter(fps=50))
