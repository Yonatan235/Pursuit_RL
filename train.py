from stable_baselines3 import PPO
from pursuit.env import PursuitEnv

if __name__ == "__main__":
    env = PursuitEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("models/ppo_pursuit.zip")
