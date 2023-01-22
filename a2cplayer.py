import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from quarto.objects import QuartoScape

# Instantiate the env
env = QuartoScape()
# Define and Train the agent
model = A2C("MlpPolicy", env).learn(total_timesteps=1000)
