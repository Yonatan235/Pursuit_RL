Simulates and trains a reinforcement learning agent to chase a moving target under different observation/tracking strategies with the aim of catching the target (being within its catch radius).
Comparisons are made to untrained/stationary strategies. 

## Features

- Brownian motion, Lévy walk, and other dynamic targets
- Discrete observations
- Reinforcement Learning environment (OpenAI Gym API)
- PPO training (via Stable-Baselines3)
- GIF visualizations

## Usage

pip install -r requirements.txt

python train.py       # Train the RL agent

python animate.py     # Visualize trained agent behavior
