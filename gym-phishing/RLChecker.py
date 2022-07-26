import gym
from stable_baselines3.common.env_checker import check_env

env = gym.make("gym_phishing:RLPTraining-v0")
check_env(env)

env = gym.make("gym_phishing:RLPTest-v0")
check_env(env)
