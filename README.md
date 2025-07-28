Simulates and trains a reinforcement learning (RL) agent to chase a moving target under different observation/tracking strategies with the aim of catching the target (being within its catch radius).
Comparisons can be made to untrained/stationary strategies. 

## Features

- Brownian motion, Lévy walk, and other dynamic targets
- Discrete observations
- Reinforcement Learning environment (OpenAI Gym API)
- PPO training via Stable-Baselines3 (a policy gradient RL algorithm)
- GIF visualizations

## Usage

pip install -r requirements.txt

python train.py       # Train the RL agent

python animate.py     # Visualize trained agent behavior
